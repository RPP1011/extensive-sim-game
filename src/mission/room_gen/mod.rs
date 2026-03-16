pub mod floorplan;
mod lcg;
pub mod ml_gen;
mod nav;
mod primitives;
mod templates;
mod validation;
mod visuals;

pub use lcg::{ObstacleRegion, ObstacleType, RampRegion};
pub use lcg::{OBS_FLOOR, OBS_WALL, OBS_PILLAR, OBS_BARRICADE, OBS_L_SHAPE, OBS_COVER_CLUSTER, OBS_SANDBAG, OBS_PLATFORM_EDGE, OBS_RAMP};
pub use nav::{NavGrid, SpawnZone};
pub use visuals::{spawn_room, RoomFloor, RoomObstacle, RoomWall};

use crate::game_core;

use lcg::Lcg;
use validation::validate_layout;

// ---------------------------------------------------------------------------
// Room layout
// ---------------------------------------------------------------------------

/// Full description of a procedurally generated room.
#[derive(Debug, Clone)]
pub struct RoomLayout {
    /// Total width along the X axis (world X).
    pub width: f32,
    /// Total depth along the Z axis (world Z).
    pub depth: f32,
    pub nav: NavGrid,
    pub player_spawn: SpawnZone,
    pub enemy_spawn: SpawnZone,
    pub room_type: game_core::RoomType,
    pub seed: u64,
    pub obstacles: Vec<ObstacleRegion>,
    pub ramps: Vec<RampRegion>,
}

// ---------------------------------------------------------------------------
// Room-size table
// ---------------------------------------------------------------------------

fn room_dimensions(rt: game_core::RoomType) -> (f32, f32) {
    match rt {
        game_core::RoomType::Entry => (20.0, 20.0),
        game_core::RoomType::Pressure => (16.0, 16.0),
        game_core::RoomType::Pivot => (18.0, 18.0),
        game_core::RoomType::Setpiece => (32.0, 32.0),
        game_core::RoomType::Recovery => (18.0, 18.0),
        game_core::RoomType::Climax => (30.0, 30.0),
        game_core::RoomType::Open => (100.0, 100.0),
    }
}

// ---------------------------------------------------------------------------
// Composite obstacle generation
// ---------------------------------------------------------------------------

/// Generate obstacles by compositing multiple primitive types for greater variety.
/// Uses the LCG to randomly select 2-3 primitive generators and layer them.
fn generate_composite_obstacles(
    nav: &mut NavGrid,
    rng: &mut Lcg,
) -> Vec<lcg::ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;

    // Each generator is a closure that places obstacles using primitives with
    // randomized positions derived from the grid dimensions.
    type GenFn = fn(&mut NavGrid, &mut Lcg, usize, usize) -> Vec<lcg::ObstacleRegion>;

    let generators: &[GenFn] = &[
        // Pillar grid in centre
        |nav, rng, cols, rows| {
            let spacing = rng.next_usize_range(2, 4);
            primitives::place_pillar_grid(
                nav, rng, cols / 4, rows / 4, 3 * cols / 4, 3 * rows / 4,
                spacing, 1, 1.5,
            )
        },
        // Horizontal wall segment with variable gap
        |nav, rng, cols, rows| {
            let gap = rng.next_usize_range(1, 3);
            primitives::place_wall_segment(
                nav, rng, cols / 4, rows / 2, cols / 2, true, gap, 1.5,
            )
        },
        // L-shape cover
        |nav, rng, cols, rows| {
            let arm = rng.next_usize_range(2, 4);
            let orient = rng.next_usize_range(0, 3);
            primitives::place_l_shape(
                nav, rng, cols / 3, rows / 3, arm, 1, orient, 1.5,
            )
        },
        // Barricade line
        |nav, rng, cols, rows| {
            let seg = rng.next_usize_range(2, 4);
            let gap = rng.next_usize_range(1, 3);
            primitives::place_barricade_line(
                nav, rng, cols / 4, 3 * cols / 4, rows / 2, seg, gap, 1.2,
            )
        },
        // Cover cluster
        |nav, rng, cols, rows| {
            primitives::place_cover_cluster(
                nav, rng, cols / 2, rows / 2, 3, 3, 1.0,
            )
        },
        // Sandbag arc
        |nav, rng, cols, rows| {
            primitives::place_sandbag_arc(
                nav, rng, cols / 2, rows / 2, 3, 5, 0.7,
            )
        },
    ];

    // Pick 2-3 generators without repeats
    let count = 2 + rng.next_usize_range(0, 1);
    let mut used = Vec::new();
    let mut obs = Vec::new();
    for _ in 0..count {
        let idx = rng.next_usize_range(0, generators.len() - 1);
        if !used.contains(&idx) {
            used.push(idx);
            obs.extend(generators[idx](nav, rng, cols, rows));
        }
    }
    obs
}

// ---------------------------------------------------------------------------
// Main generation functions
// ---------------------------------------------------------------------------

/// Generate a procedural `RoomLayout` from a seed and room type.
pub fn generate_room(seed: u64, room_type: game_core::RoomType) -> RoomLayout {
    let (width, depth) = room_dimensions(room_type);
    generate_room_with_dims(seed, room_type, width, depth)
}

/// Generate a room with optionally overridden dimensions.
/// If no override, use default dimensions with ±30% random perturbation per axis.
pub fn generate_room_varied(
    seed: u64,
    room_type: game_core::RoomType,
    dim_override: Option<(f32, f32)>,
) -> RoomLayout {
    let (base_w, base_d) = room_dimensions(room_type);
    let (width, depth) = match dim_override {
        Some((w, d)) => (w.clamp(16.0, 64.0), d.clamp(16.0, 64.0)),
        None => {
            let mut rng = Lcg::new(seed.wrapping_add(0xD1A5_E70F));
            let scale_w = rng.next_f32_range(0.7, 1.3);
            let scale_d = rng.next_f32_range(0.7, 1.3);
            (
                (base_w * scale_w).clamp(16.0, 64.0),
                (base_d * scale_d).clamp(16.0, 64.0),
            )
        }
    };
    generate_room_with_dims(seed, room_type, width, depth)
}

fn generate_room_with_dims(
    seed: u64,
    room_type: game_core::RoomType,
    width: f32,
    depth: f32,
) -> RoomLayout {
    let cell_size: f32 = 1.0;
    let cols = width as usize;
    let rows = depth as usize;
    let mut attempt_seed = seed;

    for attempt in 0..=5u64 {
        let mut nav = NavGrid::new(cols, rows, cell_size);

        // --- Perimeter walls (always unwalkable) ---
        nav.set_walkable_rect(0, 0, cols - 1, 0, false);
        nav.set_walkable_rect(0, rows - 1, cols - 1, rows - 1, false);
        nav.set_walkable_rect(0, 0, 0, rows - 1, false);
        nav.set_walkable_rect(cols - 1, 0, cols - 1, rows - 1, false);

        let mut rng = Lcg::new(attempt_seed);

        let obstacles = if attempt == 5 {
            templates::generate_fallback_obstacles(&mut nav, &mut rng)
        } else if room_type == game_core::RoomType::Open {
            Vec::new()
        } else {
            templates::generate_tactical_obstacles(&mut nav, &mut rng)
        };

        // Clear spawn margins: ensure leftmost and rightmost columns
        // (where spawns will go) are walkable for clean spawn placement.
        let margin = templates::spawn_margin(cols);
        for r in 1..rows - 1 {
            for c in 1..margin.min(cols - 1) {
                let idx = nav.idx(c, r);
                nav.walkable[idx] = true;
            }
            for c in (cols - margin).max(1)..cols - 1 {
                let idx = nav.idx(c, r);
                nav.walkable[idx] = true;
            }
        }
        // Remove any obstacle regions that fall in the cleared spawn margins
        let obstacles: Vec<_> = obstacles.into_iter().filter(|o| {
            o.col0 >= margin && o.col1 < cols - margin
        }).collect();

        // --- Ramps: small elevated platforms within the play zone ---
        // Only on rooms large enough to accommodate them (play_w >= 14, rows >= 16).
        let mut ramps: Vec<RampRegion> = Vec::new();
        let play_w = cols.saturating_sub(2 * margin);
        if play_w >= 14 && rows >= 16 {
            let ramp_col_lo = margin + play_w / 4;
            let ramp_col_hi = margin + 3 * play_w / 4;
            let ramp_row_lo = rows / 4;
            let ramp_row_hi = 3 * rows / 4;
            let num_ramps = rng.next_usize_range(0, 1); // 0 or 1
            for _ in 0..num_ramps {
                let ramp_w = rng.next_usize_range(2, 3);
                let ramp_h = rng.next_usize_range(2, 3);
                let start_col = rng.next_usize_range(ramp_col_lo, ramp_col_hi.saturating_sub(ramp_w));
                let start_row = rng.next_usize_range(ramp_row_lo, ramp_row_hi.saturating_sub(ramp_h));
                let end_col = (start_col + ramp_w).min(cols - 2);
                let end_row = (start_row + ramp_h).min(rows - 2);
                let elevation = rng.next_f32_range(0.5, 1.0);

                nav.set_elevation_rect(start_col, start_row, end_col, end_row, elevation);
                ramps.push(RampRegion {
                    col0: start_col,
                    col1: end_col,
                    row0: start_row,
                    row1: end_row,
                    elevation,
                });
            }
        }

        if attempt < 5 && !validate_layout(&nav) {
            attempt_seed = seed.wrapping_add((attempt + 1).wrapping_mul(0x517c_c1b7_2722_0a95));
            continue;
        }

        // --- Spawn zones ---
        // Place spawns on opposite sides of the room (left vs right),
        // within the cleared spawn margins.
        let row_margin = (rows / 5).max(2);
        let row_lo = row_margin;
        let row_hi = (rows - 1 - row_margin).max(row_lo + 1);

        // Left side vs right side, randomly assigned
        let (p_lo, p_hi, e_lo, e_hi) = if rng.next_u64() % 2 == 0 {
            (1, margin.saturating_sub(1).max(1),
             (cols - margin).max(1), cols - 2)
        } else {
            ((cols - margin).max(1), cols - 2,
             1, margin.saturating_sub(1).max(1))
        };
        let player_spawn = build_spawn_zone(&nav, p_lo, p_hi, row_lo, row_hi, 4, 6, &mut rng);
        let enemy_spawn = build_spawn_zone(&nav, e_lo, e_hi, row_lo, row_hi, 4, 6, &mut rng);

        return RoomLayout {
            width,
            depth,
            nav,
            player_spawn,
            enemy_spawn,
            room_type,
            seed,
            obstacles,
            ramps,
        };
    }
    unreachable!()
}

// ---------------------------------------------------------------------------
// Spawn-zone builder
// ---------------------------------------------------------------------------

pub(crate) fn build_spawn_zone(
    nav: &NavGrid,
    col_lo: usize,
    col_hi: usize,
    row_lo: usize,
    row_hi: usize,
    min_count: usize,
    max_count: usize,
    rng: &mut Lcg,
) -> SpawnZone {
    let count = rng.next_usize_range(min_count, max_count);

    let capped_col_hi = col_hi.min(nav.cols.saturating_sub(1));
    let capped_row_hi = row_hi.min(nav.rows.saturating_sub(1));

    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for r in row_lo..=capped_row_hi {
        for c in col_lo..=capped_col_hi {
            if nav.walkable[r * nav.cols + c] {
                candidates.push((c, r));
            }
        }
    }

    let positions = if candidates.is_empty() {
        Vec::new()
    } else {
        let step = ((candidates.len() as f32) / (count as f32)).max(1.0) as usize;
        candidates
            .iter()
            .step_by(step)
            .take(count)
            .map(|&(c, r)| nav.cell_centre(c, r))
            .collect()
    };

    SpawnZone { positions }
}

// ---------------------------------------------------------------------------
// Multi-channel grid extraction
// ---------------------------------------------------------------------------

/// Per-cell multi-channel grid for ML export.
#[derive(Debug, Clone)]
pub struct RoomGrid {
    pub width: usize,
    pub depth: usize,
    /// Per-cell obstacle type (row-major), 0=floor..8=ramp.
    pub obstacle_type: Vec<u8>,
    /// Per-cell visual height in meters (0.0 for floor).
    pub height: Vec<f32>,
    /// Per-cell walkable surface elevation.
    pub elevation: Vec<f32>,
}

impl RoomLayout {
    /// Extract a multi-channel grid from this layout.
    pub fn to_grid(&self) -> RoomGrid {
        let cols = self.nav.cols;
        let rows = self.nav.rows;
        let n = cols * rows;
        let mut obstacle_type = vec![OBS_FLOOR; n];
        let mut height = vec![0.0f32; n];

        // 1. Mark perimeter as wall
        for c in 0..cols {
            obstacle_type[c] = OBS_WALL; // top row
            height[c] = 2.0;
            obstacle_type[(rows - 1) * cols + c] = OBS_WALL; // bottom row
            height[(rows - 1) * cols + c] = 2.0;
        }
        for r in 0..rows {
            obstacle_type[r * cols] = OBS_WALL; // left col
            height[r * cols] = 2.0;
            obstacle_type[r * cols + cols - 1] = OBS_WALL; // right col
            height[r * cols + cols - 1] = 2.0;
        }

        // 2. Stamp obstacle regions (overwrites perimeter if overlapping)
        for obs in &self.obstacles {
            for r in obs.row0..=obs.row1 {
                for c in obs.col0..=obs.col1 {
                    if c < cols && r < rows {
                        let idx = r * cols + c;
                        obstacle_type[idx] = obs.obs_type;
                        height[idx] = obs.height;
                    }
                }
            }
        }

        // 3. Mark ramp cells
        for ramp in &self.ramps {
            for r in ramp.row0..=ramp.row1 {
                for c in ramp.col0..=ramp.col1 {
                    if c < cols && r < rows {
                        let idx = r * cols + c;
                        if obstacle_type[idx] == OBS_FLOOR {
                            obstacle_type[idx] = OBS_RAMP;
                        }
                    }
                }
            }
        }

        RoomGrid {
            width: cols,
            depth: rows,
            obstacle_type,
            height,
            elevation: self.nav.elevation.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tactical metrics
// ---------------------------------------------------------------------------

/// Tactical quality metrics computed from a room layout.
#[derive(Debug, Clone)]
pub struct RoomMetrics {
    /// Fraction of interior cells that are blocked.
    pub blocked_pct: f32,
    /// Number of cells with chokepoint_score >= 2 (high connectivity importance).
    pub chokepoint_count: usize,
    /// Fraction of walkable cells within 1 cell of a blocked cell.
    pub cover_density: f32,
    /// Number of distinct non-zero elevation values.
    pub elevation_zones: usize,
    /// Number of distinct shortest paths between spawn centroids.
    pub flanking_routes: usize,
    /// Absolute difference in spawn quality (cover access) between teams.
    pub spawn_quality_diff: f32,
    /// Average distance of walkable cells to nearest wall/obstacle.
    pub mean_wall_proximity: f32,
    /// width / depth ratio.
    pub aspect_ratio: f32,
}

impl RoomLayout {
    /// Compute tactical quality metrics.
    pub fn compute_metrics(&self) -> RoomMetrics {
        let cols = self.nav.cols;
        let rows = self.nav.rows;

        // blocked_pct
        let mut total_interior = 0usize;
        let mut blocked_count = 0usize;
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                total_interior += 1;
                if !self.nav.walkable[r * cols + c] {
                    blocked_count += 1;
                }
            }
        }
        let blocked_pct = if total_interior > 0 {
            blocked_count as f32 / total_interior as f32
        } else {
            0.0
        };

        // cover_density: walkable cells within 1 cell of a blocked cell
        let mut near_cover = 0usize;
        let mut walkable_count = 0usize;
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let idx = r * cols + c;
                if !self.nav.walkable[idx] {
                    continue;
                }
                walkable_count += 1;
                let has_adjacent_blocked = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    .iter()
                    .any(|&(dc, dr)| {
                        let nc = c as i32 + dc;
                        let nr = r as i32 + dr;
                        if nc >= 0 && nr >= 0 && (nc as usize) < cols && (nr as usize) < rows {
                            !self.nav.walkable[nr as usize * cols + nc as usize]
                        } else {
                            false
                        }
                    });
                if has_adjacent_blocked {
                    near_cover += 1;
                }
            }
        }
        let cover_density = if walkable_count > 0 {
            near_cover as f32 / walkable_count as f32
        } else {
            0.0
        };

        // elevation_zones
        let mut elevations = std::collections::HashSet::new();
        for &e in &self.nav.elevation {
            if e > 0.01 {
                elevations.insert((e * 100.0).round() as i32);
            }
        }
        let elevation_zones = elevations.len();

        // chokepoint_count: cells where removing them would disconnect regions
        // Approximate: count cells with exactly 2 walkable orthogonal neighbors
        // that form a line (not diagonal)
        let mut chokepoint_count = 0usize;
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let idx = r * cols + c;
                if !self.nav.walkable[idx] {
                    continue;
                }
                let walkable_neighbors: Vec<(i32, i32)> = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                    .iter()
                    .filter(|&&(dc, dr)| {
                        let nc = c as i32 + dc;
                        let nr = r as i32 + dr;
                        nc >= 0 && nr >= 0
                            && (nc as usize) < cols
                            && (nr as usize) < rows
                            && self.nav.walkable[nr as usize * cols + nc as usize]
                    })
                    .copied()
                    .collect();
                // A chokepoint has exactly 2 walkable neighbors on opposite sides
                if walkable_neighbors.len() == 2 {
                    let (d1, d2) = (walkable_neighbors[0], walkable_neighbors[1]);
                    if d1.0 + d2.0 == 0 && d1.1 + d2.1 == 0 {
                        chokepoint_count += 1;
                    }
                }
            }
        }

        // flanking_routes: BFS-based, count distinct paths with different midpoints
        let flanking_routes = self.count_flanking_routes();

        // spawn_quality_diff: cover density near each spawn
        let p_quality = self.spawn_cover_quality(&self.player_spawn);
        let e_quality = self.spawn_cover_quality(&self.enemy_spawn);
        let spawn_quality_diff = (p_quality - e_quality).abs();

        // mean_wall_proximity: average min distance to nearest blocked cell
        let mut total_dist = 0.0f32;
        let mut count = 0usize;
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                if !self.nav.walkable[r * cols + c] {
                    continue;
                }
                let mut min_dist = f32::MAX;
                // Search in a small radius
                for dr in -3i32..=3 {
                    for dc in -3i32..=3 {
                        let nc = c as i32 + dc;
                        let nr = r as i32 + dr;
                        if nc < 0 || nr < 0 || nc as usize >= cols || nr as usize >= rows {
                            continue;
                        }
                        if !self.nav.walkable[nr as usize * cols + nc as usize] {
                            let dist = ((dc * dc + dr * dr) as f32).sqrt();
                            min_dist = min_dist.min(dist);
                        }
                    }
                }
                if min_dist < f32::MAX {
                    total_dist += min_dist;
                    count += 1;
                }
            }
        }
        let mean_wall_proximity = if count > 0 {
            total_dist / count as f32
        } else {
            0.0
        };

        let aspect_ratio = self.width / self.depth;

        RoomMetrics {
            blocked_pct,
            chokepoint_count,
            cover_density,
            elevation_zones,
            flanking_routes,
            spawn_quality_diff,
            mean_wall_proximity,
            aspect_ratio,
        }
    }

    fn spawn_cover_quality(&self, spawn: &SpawnZone) -> f32 {
        if spawn.positions.is_empty() {
            return 0.0;
        }
        let cols = self.nav.cols;
        let mut near_cover_count = 0usize;
        for pos in &spawn.positions {
            let (c, r) = self.nav.cell_of(*pos);
            let has_cover = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                .iter()
                .any(|&(dc, dr)| {
                    let nc = c as i32 + dc;
                    let nr = r as i32 + dr;
                    if nc >= 0 && nr >= 0 && (nc as usize) < self.nav.cols && (nr as usize) < self.nav.rows {
                        !self.nav.walkable[nr as usize * cols + nc as usize]
                    } else {
                        false
                    }
                });
            if has_cover {
                near_cover_count += 1;
            }
        }
        near_cover_count as f32 / spawn.positions.len() as f32
    }

    fn count_flanking_routes(&self) -> usize {
        // Simplified: BFS from player centroid to enemy centroid,
        // then check how many "band" rows the path could take
        if self.player_spawn.positions.is_empty() || self.enemy_spawn.positions.is_empty() {
            return 0;
        }

        let _cols = self.nav.cols;
        let rows = self.nav.rows;

        let p_cx = self.player_spawn.positions.iter().map(|p| p.x).sum::<f32>()
            / self.player_spawn.positions.len() as f32;
        let p_cy = self.player_spawn.positions.iter().map(|p| p.y).sum::<f32>()
            / self.player_spawn.positions.len() as f32;
        let e_cx = self.enemy_spawn.positions.iter().map(|p| p.x).sum::<f32>()
            / self.enemy_spawn.positions.len() as f32;
        let e_cy = self.enemy_spawn.positions.iter().map(|p| p.y).sum::<f32>()
            / self.enemy_spawn.positions.len() as f32;

        let (sc, _sr) = self.nav.cell_of(crate::ai::core::SimVec2 { x: p_cx, y: p_cy });
        let (gc, _gr) = self.nav.cell_of(crate::ai::core::SimVec2 { x: e_cx, y: e_cy });

        // Count distinct row bands that have a clear walkable path
        // between the spawn columns. Divide room into 3 horizontal bands.
        let band_size = rows / 3;
        let mut route_count = 0usize;
        for band in 0..3 {
            let band_lo = (band * band_size).max(1);
            let band_hi = ((band + 1) * band_size).min(rows - 2);
            let band_mid = (band_lo + band_hi) / 2;

            // Check if there's a walkable path near this band
            let start = validation::find_nearest_walkable(&self.nav, sc, band_mid);
            let goal = validation::find_nearest_walkable(&self.nav, gc, band_mid);
            if let (Some(s), Some(g)) = (start, goal) {
                if validation::cells_connected(&self.nav, s.0, s.1, g.0, g.1) {
                    route_count += 1;
                }
            }
        }
        route_count
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_room_types() -> Vec<game_core::RoomType> {
        vec![
            game_core::RoomType::Entry,
            game_core::RoomType::Pressure,
            game_core::RoomType::Pivot,
            game_core::RoomType::Setpiece,
            game_core::RoomType::Recovery,
            game_core::RoomType::Climax,
        ]
    }

    #[test]
    fn generates_all_room_types_without_panic() {
        for rt in all_room_types() {
            let layout = generate_room(42, rt);
            assert!(layout.width > 0.0);
            assert!(layout.depth > 0.0);
        }
    }

    #[test]
    fn perimeter_is_always_unwalkable() {
        let layout = generate_room(1234, game_core::RoomType::Entry);
        let nav = &layout.nav;
        let cols = nav.cols;
        let rows = nav.rows;

        for c in 0..cols {
            assert!(!nav.walkable[c], "top row col {c} should be blocked");
            assert!(
                !nav.walkable[(rows - 1) * cols + c],
                "bottom row col {c} should be blocked"
            );
        }
        for r in 0..rows {
            assert!(!nav.walkable[r * cols], "left col row {r} should be blocked");
            assert!(
                !nav.walkable[r * cols + (cols - 1)],
                "right col row {r} should be blocked"
            );
        }
    }

    #[test]
    fn spawn_zones_are_non_empty() {
        for rt in all_room_types() {
            let layout = generate_room(99, rt);
            assert!(!layout.player_spawn.positions.is_empty(), "{rt:?} player spawn is empty");
            assert!(!layout.enemy_spawn.positions.is_empty(), "{rt:?} enemy spawn is empty");
        }
    }

    #[test]
    fn spawn_zones_have_separation() {
        for rt in all_room_types() {
            for seed in [0u64, 7, 42, 100, 999] {
                let layout = generate_room(seed, rt);
                if layout.player_spawn.positions.is_empty() || layout.enemy_spawn.positions.is_empty() {
                    continue;
                }
                // Compute centroids
                let p_cx: f32 = layout.player_spawn.positions.iter().map(|p| p.x).sum::<f32>()
                    / layout.player_spawn.positions.len() as f32;
                let e_cx: f32 = layout.enemy_spawn.positions.iter().map(|p| p.x).sum::<f32>()
                    / layout.enemy_spawn.positions.len() as f32;
                let sep = (p_cx - e_cx).abs();
                let min_expected = layout.width / 3.0 - 3.0;
                assert!(
                    sep >= min_expected.max(2.0),
                    "{rt:?} seed={seed}: spawn separation {sep:.1} < {min_expected:.1}"
                );
            }
        }
    }

    #[test]
    fn ramps_only_for_eligible_types() {
        for rt in all_room_types() {
            let layout = generate_room(55, rt);
            let has_ramps = !layout.ramps.is_empty();
            let eligible = matches!(
                rt,
                game_core::RoomType::Setpiece
                    | game_core::RoomType::Climax
                    | game_core::RoomType::Pressure
                    | game_core::RoomType::Pivot
            );
            if has_ramps {
                assert!(eligible, "{rt:?} should not have ramps but got some");
            }
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate_room(0xDEAD_BEEF, game_core::RoomType::Climax);
        let b = generate_room(0xDEAD_BEEF, game_core::RoomType::Climax);
        assert_eq!(a.nav.walkable, b.nav.walkable);
        assert_eq!(a.player_spawn.positions.len(), b.player_spawn.positions.len());
        assert_eq!(a.enemy_spawn.positions.len(), b.enemy_spawn.positions.len());
    }

    #[test]
    fn nav_grid_dimensions_match_room_size() {
        for rt in all_room_types() {
            let layout = generate_room(0, rt);
            assert_eq!(layout.nav.cols, layout.width as usize);
            assert_eq!(layout.nav.rows, layout.depth as usize);
            assert_eq!(layout.nav.walkable.len(), layout.nav.cols * layout.nav.rows);
        }
    }

    #[test]
    fn spawn_zones_are_connected() {
        for rt in all_room_types() {
            for seed in [0u64, 1, 42, 999, 0xCAFE] {
                let layout = generate_room(seed, rt);
                assert!(validate_layout(&layout.nav), "{rt:?} seed={seed} failed connectivity/blocked validation");
            }
        }
    }

    #[test]
    fn blocked_percentage_in_range() {
        for rt in all_room_types() {
            for seed in [0u64, 7, 42, 100, 5555] {
                let layout = generate_room(seed, rt);
                let nav = &layout.nav;
                let cols = nav.cols;
                let rows = nav.rows;
                let mut total = 0usize;
                let mut blocked = 0usize;
                for r in 1..rows - 1 {
                    for c in 1..cols - 1 {
                        total += 1;
                        if !nav.walkable[r * cols + c] {
                            blocked += 1;
                        }
                    }
                }
                let pct = blocked as f32 / total as f32;
                assert!(pct >= 0.02 && pct <= 0.35, "{rt:?} seed={seed} blocked={:.1}% out of 2-35% range", pct * 100.0);
            }
        }
    }

    #[test]
    fn templates_vary_by_seed() {
        for rt in all_room_types() {
            let a = generate_room(0, rt);
            let b = generate_room(1, rt);
            let c = generate_room(12345, rt);
            let grids = [&a.nav.walkable, &b.nav.walkable, &c.nav.walkable];
            let diffs = (grids[0] != grids[1]) as usize
                + (grids[1] != grids[2]) as usize
                + (grids[0] != grids[2]) as usize;
            assert!(diffs >= 1, "{rt:?} all 3 seeds produced identical grids");
        }
    }

    #[test]
    fn retry_converges() {
        for rt in all_room_types() {
            for seed in 0..100u64 {
                let layout = generate_room(seed, rt);
                assert!(!layout.player_spawn.positions.is_empty(), "{rt:?} seed={seed} player spawn empty");
                assert!(!layout.enemy_spawn.positions.is_empty(), "{rt:?} seed={seed} enemy spawn empty");
            }
        }
    }

    #[test]
    fn generate_room_varied_works() {
        for rt in all_room_types() {
            let layout = generate_room_varied(42, rt, None);
            assert!(layout.width >= 8.0 && layout.width <= 64.0);
            assert!(layout.depth >= 8.0 && layout.depth <= 64.0);
        }
    }

    #[test]
    fn generate_room_varied_with_override() {
        let layout = generate_room_varied(42, game_core::RoomType::Entry, Some((20.0, 25.0)));
        assert_eq!(layout.width, 20.0);
        assert_eq!(layout.depth, 25.0);
    }

    #[test]
    fn to_grid_produces_correct_dimensions() {
        let layout = generate_room(42, game_core::RoomType::Entry);
        let grid = layout.to_grid();
        assert_eq!(grid.width, layout.nav.cols);
        assert_eq!(grid.depth, layout.nav.rows);
        assert_eq!(grid.obstacle_type.len(), grid.width * grid.depth);
    }

    #[test]
    fn to_grid_perimeter_is_wall() {
        let layout = generate_room(42, game_core::RoomType::Entry);
        let grid = layout.to_grid();
        for c in 0..grid.width {
            assert_eq!(grid.obstacle_type[c], OBS_WALL, "top row col {c}");
            assert_eq!(grid.obstacle_type[(grid.depth - 1) * grid.width + c], OBS_WALL, "bottom row col {c}");
        }
        for r in 0..grid.depth {
            assert_eq!(grid.obstacle_type[r * grid.width], OBS_WALL, "left col row {r}");
            assert_eq!(grid.obstacle_type[r * grid.width + grid.width - 1], OBS_WALL, "right col row {r}");
        }
    }

    #[test]
    fn compute_metrics_works() {
        for rt in all_room_types() {
            let layout = generate_room(42, rt);
            let metrics = layout.compute_metrics();
            assert!(metrics.blocked_pct >= 0.0 && metrics.blocked_pct <= 1.0);
            assert!(metrics.cover_density >= 0.0 && metrics.cover_density <= 1.0);
            assert!(metrics.aspect_ratio > 0.0);
        }
    }

    #[test]
    fn varied_dimensions_differ_across_seeds() {
        let a = generate_room_varied(0, game_core::RoomType::Entry, None);
        let b = generate_room_varied(1, game_core::RoomType::Entry, None);
        let c = generate_room_varied(12345, game_core::RoomType::Entry, None);
        // At least one pair should have different dimensions
        let same = (a.width == b.width && a.depth == b.depth)
            && (b.width == c.width && b.depth == c.depth);
        assert!(!same, "All 3 seeds produced identical dimensions");
    }
}
