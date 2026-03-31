//! Path-following system — NPCs walk tile-by-tile along cached A* paths.
//!
//! When an NPC's active goal has a target_pos and they're in a settlement with
//! a city grid, this system computes an A* path and caches it. Each tick, the
//! NPC advances one cell along the path.
//!
//! Cadence: every tick (movement is per-tick for smooth tile-by-tile walking).

use crate::world_sim::state::*;

/// Maximum path length to cache (prevent memory bloat from huge paths).
const MAX_PATH_LEN: usize = 128;

/// How close (in world units) the NPC must be to a waypoint to advance to the next one.
const WAYPOINT_ARRIVE_DIST: f32 = 1.5;

/// Advance NPCs along their cached grid paths.
/// Called post-apply from runtime.rs.
/// Path computation is staggered: only recompute for a subset of NPCs each tick.
pub fn advance_pathfinding(state: &mut WorldState) {
    // CityGrid pathfinding disabled — NPCs use world-space move_target directly.
    // The tile system handles movement cost modifiers in movement.rs.
    if state.city_grids.is_empty() { return; }

    // Only compute new paths every 10 ticks to reduce A* overhead.
    let compute_paths = state.tick % 10 == 0;
    // Pre-collect settlement grid info: (settlement_id, city_grid_idx, settlement_pos).
    let grid_info: Vec<(u32, usize, (f32, f32))> = state.settlements.iter()
        .filter_map(|s| s.city_grid_idx.map(|gi| (s.id, gi, s.pos)))
        .collect();

    let entity_count = state.entities.len();
    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Get the NPC's settlement grid.
        let sid = match npc.home_settlement_id { Some(id) => id, None => continue };
        let (_, grid_idx, settlement_pos) = match grid_info.iter().find(|(id, _, _)| *id == sid) {
            Some(info) => *info,
            None => continue,
        };
        if grid_idx >= state.city_grids.len() { continue; }

        // Get the target position from the active goal.
        let target_pos = match npc.goal_stack.current() {
            Some(goal) => match goal.target_pos {
                Some(pos) => pos,
                None => continue, // no spatial target
            },
            None => continue, // no active goal
        };

        let entity_pos = entity.pos;

        // Check if we need to compute a new path.
        let needs_path = npc.cached_path.is_empty()
            || npc.path_index as usize >= npc.cached_path.len();

        // Stagger: only 1/10 of NPCs recompute per eligible tick.
        let my_slot = entity.id as u64 % 10;
        let compute_my_path = compute_paths && (state.tick / 10) % 10 == my_slot;

        if needs_path && compute_my_path {
            // Skip A* for short distances — set move_target directly.
            let dx = target_pos.0 - entity_pos.0;
            let dy = target_pos.1 - entity_pos.1;
            if dx * dx + dy * dy < 100.0 { // within ~10 world units (~5 cells)
                state.entities[i].move_target = Some(target_pos);
                continue;
            }

            // Try flow field first (O(1) per step) if target is near grid center.
            let grid = &state.city_grids[grid_idx];
            let start = grid.world_to_grid(entity_pos, settlement_pos);
            let goal = grid.world_to_grid(target_pos, settlement_pos);

            // Flow field: if goal is within 15 cells of center, use precomputed field.
            let goal_near_center = {
                let dx = goal.0 as i32 - grid.center.0 as i32;
                let dy = goal.1 as i32 - grid.center.1 as i32;
                dx * dx + dy * dy < 225 // 15 cells
            };

            if goal_near_center && !grid.center_flow.is_empty() {
                // Follow flow field: set move_target to next step toward center.
                if let Some(next) = grid.flow_field_next(&grid.center_flow, start) {
                    let wp_world = grid.grid_to_world(next.0, next.1, settlement_pos);
                    state.entities[i].move_target = Some(wp_world);
                    continue;
                }
            }

            // Fallback: A* pathfinding for targets far from center.
            if let Some(path) = grid.find_path(start, goal) {
                let npc = state.entities[i].npc.as_mut().unwrap();
                npc.cached_path = path.into_iter()
                    .take(MAX_PATH_LEN)
                    .map(|(c, r)| (c as u16, r as u16))
                    .collect();
                npc.path_index = 0;
            } else {
                // No path found — clear cache and fall back to move_target.
                let npc = state.entities[i].npc.as_mut().unwrap();
                npc.cached_path.clear();
                npc.path_index = 0;

                state.entities[i].move_target = Some(target_pos);
                continue;
            }
        }

        // Follow the cached path: move toward the current waypoint.
        let npc = state.entities[i].npc.as_ref().unwrap();
        let path_idx = npc.path_index as usize;
        if path_idx >= npc.cached_path.len() { continue; }

        let (wp_col, wp_row) = npc.cached_path[path_idx];
        let grid = &state.city_grids[grid_idx];
        let wp_world = grid.grid_to_world(wp_col as usize, wp_row as usize, settlement_pos);

        let dx = wp_world.0 - entity_pos.0;
        let dy = wp_world.1 - entity_pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < WAYPOINT_ARRIVE_DIST {
            // Arrived at waypoint — advance to next.
            let npc = state.entities[i].npc.as_mut().unwrap();
            npc.path_index += 1;

            // If we've reached the end of the path, clear it.
            if npc.path_index as usize >= npc.cached_path.len() {
                npc.cached_path.clear();
                npc.path_index = 0;
            }
        } else {
            // Set move_target to the current waypoint — advance_movement() handles stepping.
            state.entities[i].move_target = Some(wp_world);
        }
    }
}
