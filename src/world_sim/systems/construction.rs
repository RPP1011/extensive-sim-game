//! Room growth automaton — grows enclosed rooms from seed tiles.
//!
//! When an NPC places a BuildSeed, this system grows it into a complete room:
//! 1. Flood-fill from seed to find current floor tiles
//! 2. If room is large enough AND enclosed → place door, mark complete
//! 3. If too small → expand by placing floor tiles on boundary
//! 4. If at minimum size → close by placing wall tiles on boundary
//!
//! Terrain adaptation: existing walls/cliffs count as free enclosure.
//! Multi-room: seeds adjacent to existing rooms create internal doorways.
//!
//! Cadence: every 10 ticks per active seed.

use crate::world_sim::state::*;

const GROWTH_INTERVAL: u64 = 10;
const MAX_ROOM_SIZE: u32 = 64; // cap to prevent unbounded growth

/// Advance all active build seeds. Called post-apply from runtime.
pub fn advance_construction(state: &mut WorldState) {
    if state.tick % GROWTH_INTERVAL != 0 { return; }

    let tick = state.tick;
    let seed_count = state.build_seeds.len();

    // Take pooled flood-fill buffers from SimScratch. Restored at end.
    let mut visited = std::mem::take(&mut state.sim_scratch.flood_visited);
    let mut queue = std::mem::take(&mut state.sim_scratch.flood_queue);
    let mut interior = std::mem::take(&mut state.sim_scratch.flood_interior);
    let mut boundary = std::mem::take(&mut state.sim_scratch.flood_boundary);

    for i in 0..seed_count {
        if state.build_seeds[i].complete { continue; }

        let seed = state.build_seeds[i].clone();
        // grow_room fills `interior` and `boundary` in place via pooled buffers.
        grow_room(
            &seed, &mut state.tiles, tick,
            &mut visited, &mut queue, &mut interior, &mut boundary,
        );
        if interior.len() as u32 >= seed.minimum_interior && is_enclosed(&boundary, &state.tiles) {
            if !has_door(&boundary, &state.tiles) {
                if let Some(door_pos) = find_door_position(&boundary, &state.tiles) {
                    state.tiles.insert(door_pos, Tile {
                        tile_type: TileType::Door,
                        placed_by: Some(seed.placed_by),
                        placed_tick: tick,
                    });
                }
            }
            state.build_seeds[i].complete = true;
        }
    }

    // Restore pooled buffers.
    state.sim_scratch.flood_visited = visited;
    state.sim_scratch.flood_queue = queue;
    state.sim_scratch.flood_interior = interior;
    state.sim_scratch.flood_boundary = boundary;

    // Prune completed seeds older than 500 ticks.
    state.build_seeds.retain(|s| !s.complete || tick.saturating_sub(s.tick) < 500);
}

/// Grow a room by one step: expand if too small, close if at minimum.
/// Fills `interior` + `boundary` via pooled buffers — caller owns them.
fn grow_room(
    seed: &BuildSeed,
    tiles: &mut std::collections::HashMap<TilePos, Tile, ahash::RandomState>,
    tick: u64,
    visited: &mut Vec<bool>,
    queue: &mut std::collections::VecDeque<TilePos>,
    interior: &mut Vec<TilePos>,
    boundary: &mut Vec<TilePos>,
) {
    // Ensure seed position has a floor tile.
    tiles.entry(seed.pos).or_insert(Tile {
        tile_type: TileType::Floor(TileMaterial::Wood),
        placed_by: Some(seed.placed_by),
        placed_tick: tick,
    });

    // Fused flood-fill + boundary computation: single BFS pass.
    flood_fill_with_boundary(seed.pos, tiles, visited, queue, interior, boundary);

    if (interior.len() as u32) < seed.minimum_interior {
        // Below minimum: expand by placing floor tiles on empty boundary positions.
        // We iterate boundary directly — it's the pooled buffer, but we only
        // READ from it while MUTATING tiles; no aliasing.
        let boundary_len = boundary.len();
        let mut expanded = 0u32;
        for i in 0..boundary_len {
            if expanded >= 3 { break; }
            let bpos = boundary[i];
            if !tiles.contains_key(&bpos) {
                tiles.insert(bpos, Tile {
                    tile_type: TileType::Floor(TileMaterial::Wood),
                    placed_by: Some(seed.placed_by),
                    placed_tick: tick,
                });
                expanded += 1;
            }
        }
        // Re-flood to pick up fresh interior + boundary reflecting new floors.
        if expanded > 0 {
            flood_fill_with_boundary(seed.pos, tiles, visited, queue, interior, boundary);
        }
    } else if (interior.len() as u32) < MAX_ROOM_SIZE {
        // At or above minimum: close open boundary with walls. Walls are
        // placed on boundary positions — they stay on the boundary set
        // (just now solid), so we don't need to re-flood.
        for bpos in boundary.iter() {
            if !tiles.contains_key(bpos) {
                tiles.insert(*bpos, Tile {
                    tile_type: TileType::Wall(TileMaterial::Wood),
                    placed_by: Some(seed.placed_by),
                    placed_tick: tick,
                });
            }
        }
    }
}

/// Side of the local visited grid. Covers ±64 tiles from seed (4× MAX_ROOM_SIZE).
const FLOOD_GRID_SIDE: i32 = 128;
const FLOOD_GRID_HALF: i32 = FLOOD_GRID_SIDE / 2;
const FLOOD_GRID_SIZE: usize = (FLOOD_GRID_SIDE * FLOOD_GRID_SIDE) as usize;

#[inline]
fn visited_idx(pos: TilePos, origin: TilePos) -> Option<usize> {
    let lx = pos.x - origin.x + FLOOD_GRID_HALF;
    let ly = pos.y - origin.y + FLOOD_GRID_HALF;
    if lx < 0 || lx >= FLOOD_GRID_SIDE || ly < 0 || ly >= FLOOD_GRID_SIDE {
        None
    } else {
        Some((ly * FLOOD_GRID_SIDE + lx) as usize)
    }
}

/// Flood-fill from a position, collecting connected floor tiles AND the
/// boundary (non-floor tiles adjacent to the interior) in a single pass.
/// Uses a flat bool grid for visited — zero hashing in the hot loop.
/// Any tile outside the ±64 window from seed is treated as already-visited
/// (effectively bounds the BFS; rooms never approach this size).
fn flood_fill_with_boundary(
    start: TilePos,
    tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>,
    visited: &mut Vec<bool>,
    queue: &mut std::collections::VecDeque<TilePos>,
    interior: &mut Vec<TilePos>,
    boundary: &mut Vec<TilePos>,
) {
    // Ensure visited buffer is sized + zeroed.
    if visited.len() != FLOOD_GRID_SIZE {
        visited.clear();
        visited.resize(FLOOD_GRID_SIZE, false);
    } else {
        visited.fill(false);
    }
    queue.clear();
    interior.clear();
    boundary.clear();

    if !tiles.get(&start).map(|t| t.tile_type.is_floor() || t.tile_type.is_furniture()).unwrap_or(false) {
        interior.push(start); // seed is floor by convention
        return;
    }

    queue.push_back(start);
    if let Some(i) = visited_idx(start, start) { visited[i] = true; }

    while let Some(pos) = queue.pop_front() {
        interior.push(pos);
        if interior.len() > MAX_ROOM_SIZE as usize { break; }

        for neighbor in pos.neighbors4() {
            let i = match visited_idx(neighbor, start) {
                Some(i) => i,
                None => continue, // out of window — treat as visited/ignored
            };
            if visited[i] { continue; }
            visited[i] = true;
            match tiles.get(&neighbor) {
                Some(tile)
                    if tile.tile_type.is_floor() || tile.tile_type.is_furniture()
                        || matches!(tile.tile_type, TileType::Door) =>
                {
                    queue.push_back(neighbor);
                }
                _ => {
                    boundary.push(neighbor);
                }
            }
        }
    }
}

/// Back-compat wrapper for call sites that only need interior (tests).
#[cfg(test)]
fn flood_fill_floor(start: TilePos, tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> Vec<TilePos> {
    let mut visited = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    let mut interior = Vec::new();
    let mut boundary = Vec::new();
    flood_fill_with_boundary(start, tiles, &mut visited, &mut queue, &mut interior, &mut boundary);
    interior
}

/// Compute boundary positions: tiles adjacent to interior that are not interior.
fn compute_boundary(interior: &[TilePos], _tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> Vec<TilePos> {
    let mut interior_set: std::collections::HashSet<TilePos, ahash::RandomState> =
        std::collections::HashSet::default();
    interior_set.extend(interior.iter().copied());
    let mut boundary = Vec::new();
    let mut seen: std::collections::HashSet<TilePos, ahash::RandomState> =
        std::collections::HashSet::default();

    for &pos in interior {
        for neighbor in pos.neighbors4() {
            if !interior_set.contains(&neighbor) && seen.insert(neighbor) {
                boundary.push(neighbor);
            }
        }
    }
    boundary
}

/// Check if all boundary positions are solid (walls, existing structures, or placed walls).
/// Takes the pre-computed boundary from flood_fill_with_boundary — no need
/// to reconstruct an interior_set since the boundary already excludes
/// interior tiles by construction.
fn is_enclosed(boundary: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> bool {
    for &pos in boundary {
        match tiles.get(&pos) {
            Some(tile) if tile.tile_type.is_wall() || matches!(tile.tile_type, TileType::Door) => {}
            _ => return false, // open boundary = not enclosed
        }
    }
    true
}

/// Check if any boundary tile is a door.
fn has_door(boundary: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> bool {
    for &pos in boundary {
        if let Some(tile) = tiles.get(&pos) {
            if matches!(tile.tile_type, TileType::Door) { return true; }
        }
    }
    false
}

/// Find the best position for a door: boundary wall tile with 2 wall neighbors
/// (in a line, not a corner). Uses the pre-computed boundary directly.
fn find_door_position(boundary: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> Option<TilePos> {
    let mut best: Option<(TilePos, usize)> = None;
    for &pos in boundary {
        match tiles.get(&pos) {
            Some(tile) if tile.tile_type.is_wall() => {}
            _ => continue,
        }
        let wall_count = pos.neighbors4().iter()
            .filter(|n| tiles.get(n).map(|t| t.tile_type.is_wall()).unwrap_or(false))
            .count();
        if wall_count == 2 {
            match &best {
                None => { best = Some((pos, wall_count)); }
                Some((_, prev_count)) if wall_count < *prev_count => {
                    best = Some((pos, wall_count));
                }
                _ => {}
            }
        } else if best.is_none() {
            best = Some((pos, wall_count));
        }
    }
    best.map(|(p, _)| p)
}

/// Detect what function a room provides based on its furniture contents.
pub fn detect_room_function(interior: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> RoomFunction {
    let mut rf = RoomFunction::default();
    for &pos in interior {
        if let Some(tile) = tiles.get(&pos) {
            match tile.tile_type {
                TileType::Bed => rf.provides_shelter = true,
                TileType::Workspace(_) => rf.provides_work = true,
                TileType::Altar => rf.provides_worship = true,
                TileType::Bookshelf => rf.provides_knowledge = true,
                TileType::WeaponRack | TileType::TrainingDummy => rf.provides_defense = true,
                TileType::MarketStall => rf.provides_trade = true,
                TileType::StorageContainer => rf.provides_storage = true,
                TileType::Hearth => { rf.provides_shelter = true; }
                _ => {}
            }
        }
    }
    rf.capacity = (interior.len() / 4).max(1);
    rf
}

/// What a room provides — emergent from contents.
#[derive(Debug, Clone, Default)]
pub struct RoomFunction {
    pub provides_shelter: bool,
    pub provides_work: bool,
    pub provides_worship: bool,
    pub provides_knowledge: bool,
    pub provides_defense: bool,
    pub provides_trade: bool,
    pub provides_storage: bool,
    pub capacity: usize,
}
