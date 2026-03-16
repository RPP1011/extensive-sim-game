//! ML-based room generation via Python subprocess.
//!
//! Calls the trained ELIT-DiT model via `training/roomgen/infer.py`.
//! Falls back to proc-gen if ML generation fails.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::game_core;

use super::lcg::{ObstacleRegion, RampRegion, OBS_FLOOR, OBS_WALL, OBS_RAMP};
use super::nav::NavGrid;
use super::{build_spawn_zone, generate_room, RoomLayout, SpawnZone};
use super::lcg::Lcg;
use super::validation::validate_layout;

/// Configuration for ML room generation.
pub struct MlGenConfig {
    /// Path to trained ELIT-DiT weights.
    pub weights_path: PathBuf,
    /// Path to the Python inference script.
    pub script_path: PathBuf,
    /// Python executable (default: "python3").
    pub python: String,
    /// Number of Euler sampling steps.
    pub n_steps: u32,
    /// CFG scale.
    pub cfg_scale: f32,
    /// Guidance weight for PhyScene constraints.
    pub guidance_weight: f32,
}

impl Default for MlGenConfig {
    fn default() -> Self {
        Self {
            weights_path: PathBuf::from("generated/elit_dit_weights.pt"),
            script_path: PathBuf::from("training/roomgen/infer.py"),
            python: "python3".to_string(),
            n_steps: 40,
            cfg_scale: 3.0,
            guidance_weight: 1.0,
        }
    }
}

/// Generate a room using the ML model, falling back to proc-gen on failure.
pub fn generate_ml_room(
    prompt: &str,
    room_type: game_core::RoomType,
    seed: u64,
    config: &MlGenConfig,
) -> RoomLayout {
    for attempt in 0..3u64 {
        if let Some(layout) = try_ml_generation(prompt, room_type, seed + attempt, config) {
            return layout;
        }
    }
    // Fallback to proc-gen
    generate_room(seed, room_type)
}

fn try_ml_generation(
    prompt: &str,
    room_type: game_core::RoomType,
    seed: u64,
    config: &MlGenConfig,
) -> Option<RoomLayout> {
    let rt_str = match room_type {
        game_core::RoomType::Entry => "Entry",
        game_core::RoomType::Pressure => "Pressure",
        game_core::RoomType::Pivot => "Pivot",
        game_core::RoomType::Setpiece => "Setpiece",
        game_core::RoomType::Recovery => "Recovery",
        game_core::RoomType::Climax => "Climax",
        game_core::RoomType::Open => "Open",
    };

    let request = format!(
        r#"{{"prompt":"{}","room_type":"{}","seed":{}}}"#,
        prompt.replace('"', "\\\""),
        rt_str,
        seed,
    );

    let mut child = Command::new(&config.python)
        .arg(&config.script_path)
        .arg("--weights")
        .arg(&config.weights_path)
        .arg("--n-steps")
        .arg(config.n_steps.to_string())
        .arg("--cfg-scale")
        .arg(config.cfg_scale.to_string())
        .arg("--guidance-weight")
        .arg(config.guidance_weight.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    if let Some(ref mut stdin) = child.stdin {
        stdin.write_all(request.as_bytes()).ok()?;
    }
    drop(child.stdin.take());

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    let response: serde_json::Value =
        serde_json::from_slice(&output.stdout).ok()?;

    if !response["success"].as_bool().unwrap_or(false) {
        return None;
    }

    let width = response["width"].as_u64()? as usize;
    let depth = response["depth"].as_u64()? as usize;

    let obstacle_types = parse_2d_u8(&response["obstacle_type"], width, depth)?;
    let heights = parse_2d_f32(&response["height"], width, depth)?;
    let elevations = parse_2d_f32(&response["elevation"], width, depth)?;

    let (nav, obstacles, ramps) =
        ml_grid_to_navgrid(&obstacle_types, &heights, &elevations, width, depth, 1.0);

    if !validate_layout(&nav) {
        return None;
    }

    // Spawn zones
    let mut rng = Lcg::new(seed);
    let min_sep = (width / 3).max(4);
    let anchor_a = rng.next_usize_range(1, width.saturating_sub(min_sep + 1));
    let anchor_b = anchor_a + min_sep;
    let (p_lo, p_hi, e_lo, e_hi) = if rng.next_u64() % 2 == 0 {
        (
            anchor_a.saturating_sub(2).max(1),
            (anchor_a + 2).min(width - 2),
            anchor_b.saturating_sub(2).max(1),
            (anchor_b + 2).min(width - 2),
        )
    } else {
        (
            anchor_b.saturating_sub(2).max(1),
            (anchor_b + 2).min(width - 2),
            anchor_a.saturating_sub(2).max(1),
            (anchor_a + 2).min(width - 2),
        )
    };
    let player_spawn = build_spawn_zone(&nav, p_lo, p_hi, 1, depth - 2, 6, 8, &mut rng);
    let enemy_spawn = build_spawn_zone(&nav, e_lo, e_hi, 1, depth - 2, 6, 8, &mut rng);

    Some(RoomLayout {
        width: width as f32,
        depth: depth as f32,
        nav,
        player_spawn,
        enemy_spawn,
        room_type,
        seed,
        obstacles,
        ramps,
    })
}

/// Convert ML grid output to NavGrid + ObstacleRegion list.
fn ml_grid_to_navgrid(
    obstacle_types: &[Vec<u8>],
    heights: &[Vec<f32>],
    elevations: &[Vec<f32>],
    width: usize,
    depth: usize,
    cell_size: f32,
) -> (NavGrid, Vec<ObstacleRegion>, Vec<RampRegion>) {
    let mut nav = NavGrid::new(width, depth, cell_size);
    let mut obstacles = Vec::new();
    let mut ramps = Vec::new();

    // Enforce perimeter walls
    nav.set_walkable_rect(0, 0, width - 1, 0, false);
    nav.set_walkable_rect(0, depth - 1, width - 1, depth - 1, false);
    nav.set_walkable_rect(0, 0, 0, depth - 1, false);
    nav.set_walkable_rect(width - 1, 0, width - 1, depth - 1, false);

    // Process interior cells
    for r in 1..depth - 1 {
        for c in 1..width - 1 {
            let obs_type = obstacle_types[r][c];
            let height = heights[r][c];
            let elevation = elevations[r][c];

            if obs_type == OBS_FLOOR {
                // Walkable floor, possibly elevated
                if elevation > 0.1 {
                    nav.set_elevation_rect(c, r, c, r, elevation);
                }
            } else if obs_type == OBS_RAMP {
                // Ramp: walkable with elevation
                nav.set_elevation_rect(c, r, c, r, elevation);
            } else {
                // Blocked obstacle
                let idx = nav.idx(c, r);
                nav.walkable[idx] = false;
                obstacles.push(ObstacleRegion {
                    col0: c,
                    col1: c,
                    row0: r,
                    row1: r,
                    height,
                    obs_type,
                });
            }
        }
    }

    // Extract contiguous ramp regions
    let mut visited = vec![vec![false; width]; depth];
    for r in 1..depth - 1 {
        for c in 1..width - 1 {
            if obstacle_types[r][c] == OBS_RAMP && !visited[r][c] {
                // Flood-fill to find contiguous ramp region
                let (mut min_c, mut max_c, mut min_r, mut max_r) = (c, c, r, r);
                let mut stack = vec![(r, c)];
                while let Some((sr, sc)) = stack.pop() {
                    if visited[sr][sc] {
                        continue;
                    }
                    visited[sr][sc] = true;
                    min_c = min_c.min(sc);
                    max_c = max_c.max(sc);
                    min_r = min_r.min(sr);
                    max_r = max_r.max(sr);

                    for &(dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nr = (sr as i32 + dr) as usize;
                        let nc = (sc as i32 + dc) as usize;
                        if nr > 0
                            && nr < depth - 1
                            && nc > 0
                            && nc < width - 1
                            && !visited[nr][nc]
                            && obstacle_types[nr][nc] == OBS_RAMP
                        {
                            stack.push((nr, nc));
                        }
                    }
                }

                let max_elev = elevations[min_r..=max_r]
                    .iter()
                    .flat_map(|row| row[min_c..=max_c].iter())
                    .cloned()
                    .fold(0.0f32, f32::max);

                ramps.push(RampRegion {
                    col0: min_c,
                    col1: max_c,
                    row0: min_r,
                    row1: max_r,
                    elevation: max_elev,
                });
            }
        }
    }

    // Merge adjacent single-cell obstacles into larger regions for visual spawning
    let obstacles = merge_obstacle_regions(obstacles, width, depth);

    (nav, obstacles, ramps)
}

/// Merge adjacent single-cell obstacles of the same type into rectangular regions.
fn merge_obstacle_regions(
    singles: Vec<ObstacleRegion>,
    width: usize,
    depth: usize,
) -> Vec<ObstacleRegion> {
    if singles.is_empty() {
        return singles;
    }

    // Build type grid
    let mut type_grid = vec![vec![OBS_FLOOR; width]; depth];
    let mut height_grid = vec![vec![0.0f32; width]; depth];
    for obs in &singles {
        for r in obs.row0..=obs.row1 {
            for c in obs.col0..=obs.col1 {
                if r < depth && c < width {
                    type_grid[r][c] = obs.obs_type;
                    height_grid[r][c] = obs.height;
                }
            }
        }
    }

    // Greedy rectangle merging
    let mut visited = vec![vec![false; width]; depth];
    let mut merged = Vec::new();

    for r in 0..depth {
        for c in 0..width {
            if visited[r][c] || type_grid[r][c] == OBS_FLOOR {
                continue;
            }
            let t = type_grid[r][c];
            let h = height_grid[r][c];

            // Expand right
            let mut c1 = c;
            while c1 + 1 < width
                && !visited[r][c1 + 1]
                && type_grid[r][c1 + 1] == t
                && (height_grid[r][c1 + 1] - h).abs() < 0.1
            {
                c1 += 1;
            }

            // Expand down
            let mut r1 = r;
            'outer: while r1 + 1 < depth {
                for cc in c..=c1 {
                    if visited[r1 + 1][cc]
                        || type_grid[r1 + 1][cc] != t
                        || (height_grid[r1 + 1][cc] - h).abs() >= 0.1
                    {
                        break 'outer;
                    }
                }
                r1 += 1;
            }

            // Mark visited
            for rr in r..=r1 {
                for cc in c..=c1 {
                    visited[rr][cc] = true;
                }
            }

            merged.push(ObstacleRegion {
                col0: c,
                col1: c1,
                row0: r,
                row1: r1,
                height: h,
                obs_type: t,
            });
        }
    }

    merged
}

fn parse_2d_u8(val: &serde_json::Value, w: usize, d: usize) -> Option<Vec<Vec<u8>>> {
    let arr = val.as_array()?;
    if arr.len() != d {
        return None;
    }
    let mut result = Vec::with_capacity(d);
    for row_val in arr {
        let row = row_val.as_array()?;
        if row.len() != w {
            return None;
        }
        result.push(row.iter().map(|v| v.as_u64().unwrap_or(0) as u8).collect());
    }
    Some(result)
}

fn parse_2d_f32(val: &serde_json::Value, w: usize, d: usize) -> Option<Vec<Vec<f32>>> {
    let arr = val.as_array()?;
    if arr.len() != d {
        return None;
    }
    let mut result = Vec::with_capacity(d);
    for row_val in arr {
        let row = row_val.as_array()?;
        if row.len() != w {
            return None;
        }
        result.push(
            row.iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        );
    }
    Some(result)
}
