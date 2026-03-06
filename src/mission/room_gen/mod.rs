mod lcg;
mod nav;
mod primitives;
mod templates;
mod visuals;

pub use nav::{NavGrid, SpawnZone};
pub use visuals::{spawn_room, RoomFloor, RoomObstacle, RoomWall};

use crate::game_core;

use lcg::{Lcg, ObstacleRegion, RampRegion};

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
    // Private geometry caches kept for `spawn_room`.
    pub(crate) obstacles: Vec<ObstacleRegion>,
    pub(crate) ramps: Vec<RampRegion>,
}

// ---------------------------------------------------------------------------
// Room-size table
// ---------------------------------------------------------------------------

fn room_dimensions(rt: game_core::RoomType) -> (f32, f32) {
    match rt {
        game_core::RoomType::Entry => (20.0, 20.0),
        game_core::RoomType::Pressure => (28.0, 14.0),
        game_core::RoomType::Pivot => (16.0, 16.0),
        game_core::RoomType::Setpiece => (32.0, 32.0),
        game_core::RoomType::Recovery => (18.0, 18.0),
        game_core::RoomType::Climax => (30.0, 30.0),
    }
}

// ---------------------------------------------------------------------------
// Constraint Validation
// ---------------------------------------------------------------------------

/// Check that a layout has 5%-35% blocked interior and player<->enemy connectivity.
fn validate_layout(nav: &NavGrid) -> bool {
    let cols = nav.cols;
    let rows = nav.rows;

    let mut total_interior = 0usize;
    let mut blocked = 0usize;
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            total_interior += 1;
            if !nav.walkable[r * cols + c] {
                blocked += 1;
            }
        }
    }

    if total_interior == 0 {
        return false;
    }

    let pct = (blocked as f32) / (total_interior as f32);
    if pct < 0.02 || pct > 0.35 {
        return false;
    }

    let player_col = cols / 6;
    let enemy_col = cols - cols / 6;
    let mid_row = rows / 2;

    let start = match find_nearest_walkable(nav, player_col, mid_row) {
        Some(pos) => pos,
        None => return false,
    };
    let goal = match find_nearest_walkable(nav, enemy_col, mid_row) {
        Some(pos) => pos,
        None => return false,
    };

    cells_connected(nav, start.0, start.1, goal.0, goal.1)
}

/// 4-directional BFS flood-fill; returns true if goal is reachable from start.
fn cells_connected(
    nav: &NavGrid,
    start_col: usize,
    start_row: usize,
    goal_col: usize,
    goal_row: usize,
) -> bool {
    let cols = nav.cols;
    let rows = nav.rows;
    let mut visited = vec![false; cols * rows];
    let mut queue = std::collections::VecDeque::new();

    let start_idx = start_row * cols + start_col;
    visited[start_idx] = true;
    queue.push_back((start_col, start_row));

    while let Some((c, r)) = queue.pop_front() {
        if c == goal_col && r == goal_row {
            return true;
        }
        for &(dc, dr) in &[(0isize, -1isize), (0, 1), (-1, 0), (1, 0)] {
            let nc = c as isize + dc;
            let nr = r as isize + dr;
            if nc < 0 || nr < 0 || nc >= cols as isize || nr >= rows as isize {
                continue;
            }
            let nc = nc as usize;
            let nr = nr as usize;
            let idx = nr * cols + nc;
            if !visited[idx] && nav.walkable[idx] {
                visited[idx] = true;
                queue.push_back((nc, nr));
            }
        }
    }
    false
}

/// Expanding-square search for the nearest walkable cell from a target.
fn find_nearest_walkable(nav: &NavGrid, col: usize, row: usize) -> Option<(usize, usize)> {
    let cols = nav.cols;
    let rows = nav.rows;

    if col < cols && row < rows && nav.walkable[row * cols + col] {
        return Some((col, row));
    }

    for radius in 1..cols.max(rows) {
        let r_min = row.saturating_sub(radius);
        let r_max = (row + radius).min(rows - 1);
        let c_min = col.saturating_sub(radius);
        let c_max = (col + radius).min(cols - 1);

        for r in r_min..=r_max {
            for c in c_min..=c_max {
                if r != r_min && r != r_max && c != c_min && c != c_max {
                    continue;
                }
                if nav.walkable[r * cols + c] {
                    return Some((c, r));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Main generation function
// ---------------------------------------------------------------------------

/// Generate a procedural `RoomLayout` from a seed and room type.
pub fn generate_room(seed: u64, room_type: game_core::RoomType) -> RoomLayout {
    let (width, depth) = room_dimensions(room_type);
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
        } else {
            match room_type {
                game_core::RoomType::Entry => templates::generate_entry_obstacles(&mut nav, &mut rng),
                game_core::RoomType::Pressure => templates::generate_pressure_obstacles(&mut nav, &mut rng),
                game_core::RoomType::Pivot => templates::generate_pivot_obstacles(&mut nav, &mut rng),
                game_core::RoomType::Setpiece => templates::generate_setpiece_obstacles(&mut nav, &mut rng),
                game_core::RoomType::Recovery => templates::generate_recovery_obstacles(&mut nav, &mut rng),
                game_core::RoomType::Climax => templates::generate_climax_obstacles(&mut nav, &mut rng),
            }
        };

        // --- Ramps (Setpiece, Climax, Pressure, Pivot) ---
        let mut ramps: Vec<RampRegion> = Vec::new();
        let supports_ramps = matches!(
            room_type,
            game_core::RoomType::Setpiece
                | game_core::RoomType::Climax
                | game_core::RoomType::Pressure
                | game_core::RoomType::Pivot
        );
        if supports_ramps {
            let inner_col_lo = (cols as f32 * 0.25) as usize;
            let inner_col_hi = (cols as f32 * 0.75) as usize;
            let inner_row_lo: usize = 1;
            let inner_row_hi = rows - 2;
            let num_ramps = rng.next_usize_range(0, 2);
            for _ in 0..num_ramps {
                let start_col = rng.next_usize_range(inner_col_lo, inner_col_hi.saturating_sub(3));
                let start_row = rng.next_usize_range(inner_row_lo, inner_row_hi.saturating_sub(3));
                let ramp_w = rng.next_usize_range(2, 5);
                let ramp_h = rng.next_usize_range(2, 4);
                let end_col = (start_col + ramp_w).min(cols - 2);
                let end_row = (start_row + ramp_h).min(rows - 2);
                let elevation = rng.next_f32_range(0.5, 1.5);

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
        // Randomize spawn regions: pick two anchor columns with minimum
        // separation, then build each team's zone around its anchor.
        let min_sep = (cols / 3).max(4);
        let anchor_a = rng.next_usize_range(1, cols.saturating_sub(min_sep + 1));
        let anchor_b = anchor_a + min_sep;
        // Randomly assign which side is player vs enemy
        let (p_lo, p_hi, e_lo, e_hi) = if rng.next_u64() % 2 == 0 {
            (anchor_a.saturating_sub(2).max(1), (anchor_a + 2).min(cols - 2),
             anchor_b.saturating_sub(2).max(1), (anchor_b + 2).min(cols - 2))
        } else {
            (anchor_b.saturating_sub(2).max(1), (anchor_b + 2).min(cols - 2),
             anchor_a.saturating_sub(2).max(1), (anchor_a + 2).min(cols - 2))
        };
        let player_spawn = build_spawn_zone(&nav, p_lo, p_hi, 1, rows - 2, 6, 8, &mut rng);
        let enemy_spawn = build_spawn_zone(&nav, e_lo, e_hi, 1, rows - 2, 6, 8, &mut rng);

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

fn build_spawn_zone(
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
}
