use crate::sim::{distance, SimState, UnitState};
use super::game_state::MAX_POSITION_TOKENS;

// ---------------------------------------------------------------------------
// Position token extraction: areas of interest for pointer-based actions
// ---------------------------------------------------------------------------

/// A candidate position with intrinsic spatial properties.
struct PositionCandidate {
    pos: crate::sim::SimVec2,
    /// Tactical value for sorting (higher = more interesting).
    score: f32,
}

/// Extract up to MAX_POSITION_TOKENS areas of interest as 8-dim feature vectors.
///
/// Position features (8):
///   0: dx from self / 20
///   1: dy from self / 20
///   2: path distance from self / 30 (A* distance; divergence from euclidean reveals walls)
///   3: elevation / 5
///   4: chokepoint_score / 3 (blocked cardinal neighbors)
///   5: wall_proximity / 5 (min raycast distance to nearest wall)
///   6: n_hostile_zones / 3
///   7: n_friendly_zones / 3
pub(super) fn extract_position_tokens(state: &SimState, unit: &UnitState) -> Vec<Vec<f32>> {
    use crate::sim::sim_vec2;

    let nav = match &state.grid_nav {
        Some(nav) => nav,
        None => return Vec::new(),
    };

    let self_pos = unit.position;
    let move_range = unit.move_speed_per_sec * 1.5; // ~1.5 seconds of movement

    let is_fully_open = nav.fully_open;

    let mut candidates: Vec<PositionCandidate> = Vec::new();

    // Sample positions in 8 directions at 2 distances
    let directions: [(f32, f32); 8] = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
    ];

    for &(dx, dy) in &directions {
        for dist_mult in &[0.5, 1.0, 1.5] {
            let d = move_range * dist_mult;
            let pos = sim_vec2(self_pos.x + dx * d, self_pos.y + dy * d);

            // Skip out-of-bounds or blocked positions
            if !nav.is_walkable_pos(pos) {
                continue;
            }

            if is_fully_open {
                // All open-room candidates score identically (no elevation,
                // no chokepoints), so just push with score 0.
                candidates.push(PositionCandidate { pos, score: 0.0 });
            } else {
                // Use precomputed features where available
                let elevation = nav.elevation_at_pos(pos);
                let blocked_neighbors = nav.chokepoint_at_pos(pos) as usize;

                // Score: elevation + chokepoint value (1-2 blocked = good, 3+ = cornered)
                let choke_val = match blocked_neighbors {
                    1 => 1.0,
                    2 => 1.5,
                    _ => 0.0,
                };

                // Retreat paths: count reachable cells within 2 steps
                let cell = nav.cell_of(pos);
                let mut reachable = 0u32;
                for &(odx, ody) in &[(1i32,0),(-1,0),(0,1),(0,-1)] {
                    let n = (cell.0 + odx, cell.1 + ody);
                    if !nav.blocked.contains(&n) && nav.in_bounds(n.0, n.1) {
                        reachable += 1;
                        // Check second hop
                        for &(odx2, ody2) in &[(1i32,0),(-1,0),(0,1),(0,-1)] {
                            let n2 = (n.0 + odx2, n.1 + ody2);
                            if !nav.blocked.contains(&n2) && nav.in_bounds(n2.0, n2.1) {
                                reachable += 1;
                            }
                        }
                    }
                }
                let retreat_score = (reachable as f32 / 20.0).min(1.0);

                // Team-relative: distance to nearest ally relative to move range
                let nearest_ally_dist = state.units.iter()
                    .filter(|u| u.team == unit.team && u.id != unit.id && u.hp > 0)
                    .map(|u| distance(pos, u.position))
                    .fold(f32::MAX, f32::min);
                let ally_proximity_score = if nearest_ally_dist < f32::MAX {
                    // Prefer positions within ~1.5x move range of an ally
                    (1.0 - (nearest_ally_dist / (move_range * 1.5)).min(1.0)) * 0.3
                } else {
                    0.0
                };

                let score = elevation * 2.0 + choke_val + retreat_score * 0.5 + ally_proximity_score;

                candidates.push(PositionCandidate { pos, score });
            }
        }
    }

    // Deduplicate: remove candidates within 2.0 of a higher-scored candidate
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut filtered: Vec<PositionCandidate> = Vec::new();
    for c in candidates {
        let too_close = filtered.iter().any(|f| distance(c.pos, f.pos) < 2.0);
        if !too_close {
            filtered.push(c);
        }
        if filtered.len() >= MAX_POSITION_TOKENS {
            break;
        }
    }

    // Encode each position as 8-dim feature vector
    filtered.iter().map(|c| {
        let dx = c.pos.x - self_pos.x;
        let dy = c.pos.y - self_pos.y;
        let euclidean = (dx * dx + dy * dy).sqrt();

        if is_fully_open {
            // Fast path: no obstacles means path_dist == euclidean, no
            // elevation, no chokepoints, max wall proximity, and zone
            // counts are the only variable features.
            let hostile_zones = state.zones.iter()
                .filter(|z| z.source_team != unit.team && distance(c.pos, z.position) < 3.0)
                .count();
            let friendly_zones = state.zones.iter()
                .filter(|z| z.source_team == unit.team && distance(c.pos, z.position) < 3.0)
                .count();

            return vec![
                dx / 20.0,
                dy / 20.0,
                euclidean / 30.0, // path_dist == euclidean (no detours)
                0.0,              // elevation = 0
                0.0,              // blocked_neighbors = 0
                1.0,              // wall_prox = 5.0 / 5.0 = 1.0 (max)
                hostile_zones as f32 / 3.0,
                friendly_zones as f32 / 3.0,
            ];
        }

        // Path distance: use line-of-sight check (cheap) instead of A* (expensive).
        // If line is clear, path = euclidean. If blocked, estimate ~1.4× euclidean.
        let path_dist = {
            use crate::pathing::has_line_of_sight;
            if has_line_of_sight(nav, self_pos, c.pos) {
                euclidean
            } else {
                euclidean * 1.4 // approximate detour factor
            }
        };

        let elevation = nav.elevation_at_pos(c.pos);
        let blocked_neighbors = nav.chokepoint_at_pos(c.pos) as usize;
        let wall_prox = nav.wall_proximity_at_pos(c.pos);

        // Zone counts at this position
        let hostile_zones = state.zones.iter()
            .filter(|z| z.source_team != unit.team && distance(c.pos, z.position) < 3.0)
            .count();
        let friendly_zones = state.zones.iter()
            .filter(|z| z.source_team == unit.team && distance(c.pos, z.position) < 3.0)
            .count();

        vec![
            dx / 20.0,
            dy / 20.0,
            path_dist / 30.0,
            elevation / 5.0,
            blocked_neighbors as f32 / 3.0,
            wall_prox / 5.0,
            hostile_zones as f32 / 3.0,
            friendly_zones as f32 / 3.0,
        ]
    }).collect()
}
