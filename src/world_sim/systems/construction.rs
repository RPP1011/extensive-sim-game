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

    for i in 0..seed_count {
        if state.build_seeds[i].complete { continue; }

        let seed = state.build_seeds[i].clone();
        grow_room(&seed, &mut state.tiles, tick);

        // Check completion.
        let interior = flood_fill_floor(seed.pos, &state.tiles);
        if interior.len() as u32 >= seed.minimum_interior && is_enclosed(&interior, &state.tiles) {
            // Room complete — place door if needed.
            if !has_door(&interior, &state.tiles) {
                if let Some(door_pos) = find_door_position(&interior, &state.tiles) {
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

    // Prune completed seeds older than 500 ticks.
    state.build_seeds.retain(|s| !s.complete || tick.saturating_sub(s.tick) < 500);
}

/// Grow a room by one step: expand if too small, close if at minimum.
fn grow_room(seed: &BuildSeed, tiles: &mut std::collections::HashMap<TilePos, Tile>, tick: u64) {
    // Ensure seed position has a floor tile.
    tiles.entry(seed.pos).or_insert(Tile {
        tile_type: TileType::Floor(TileMaterial::Wood),
        placed_by: Some(seed.placed_by),
        placed_tick: tick,
    });

    let interior = flood_fill_floor(seed.pos, tiles);
    let boundary = compute_boundary(&interior, tiles);

    if (interior.len() as u32) < seed.minimum_interior {
        // Below minimum: expand by placing floor tiles on empty boundary positions.
        let mut expanded = 0u32;
        for bpos in &boundary {
            if expanded >= 3 { break; } // grow at most 3 tiles per step
            if !tiles.contains_key(bpos) {
                tiles.insert(*bpos, Tile {
                    tile_type: TileType::Floor(TileMaterial::Wood),
                    placed_by: Some(seed.placed_by),
                    placed_tick: tick,
                });
                expanded += 1;
            }
        }
    } else if (interior.len() as u32) < MAX_ROOM_SIZE {
        // At or above minimum: close open boundary with walls.
        for bpos in &boundary {
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

/// Flood-fill from a position, collecting connected floor tiles.
fn flood_fill_floor(start: TilePos, tiles: &std::collections::HashMap<TilePos, Tile>) -> Vec<TilePos> {
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut result = Vec::new();

    if !tiles.get(&start).map(|t| t.tile_type.is_floor() || t.tile_type.is_furniture()).unwrap_or(false) {
        return vec![start]; // seed is floor by convention even if not yet placed
    }

    queue.push_back(start);
    visited.insert(start);

    while let Some(pos) = queue.pop_front() {
        result.push(pos);
        if result.len() > MAX_ROOM_SIZE as usize { break; }

        for neighbor in pos.neighbors4() {
            if visited.contains(&neighbor) { continue; }
            visited.insert(neighbor);
            if let Some(tile) = tiles.get(&neighbor) {
                if tile.tile_type.is_floor() || tile.tile_type.is_furniture()
                    || matches!(tile.tile_type, TileType::Door)
                {
                    queue.push_back(neighbor);
                }
            }
        }
    }
    result
}

/// Compute boundary positions: tiles adjacent to interior that are not interior.
fn compute_boundary(interior: &[TilePos], _tiles: &std::collections::HashMap<TilePos, Tile>) -> Vec<TilePos> {
    let interior_set: std::collections::HashSet<TilePos> = interior.iter().copied().collect();
    let mut boundary = Vec::new();
    let mut seen = std::collections::HashSet::new();

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
fn is_enclosed(interior: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile>) -> bool {
    let interior_set: std::collections::HashSet<TilePos> = interior.iter().copied().collect();

    for &pos in interior {
        for neighbor in pos.neighbors4() {
            if interior_set.contains(&neighbor) { continue; }
            // This boundary tile must be solid (wall, door, or similar).
            match tiles.get(&neighbor) {
                Some(tile) if tile.tile_type.is_wall() || matches!(tile.tile_type, TileType::Door) => {}
                _ => return false, // open boundary = not enclosed
            }
        }
    }
    true
}

/// Check if any boundary tile is a door.
fn has_door(interior: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile>) -> bool {
    let interior_set: std::collections::HashSet<TilePos> = interior.iter().copied().collect();
    for &pos in interior {
        for neighbor in pos.neighbors4() {
            if interior_set.contains(&neighbor) { continue; }
            if let Some(tile) = tiles.get(&neighbor) {
                if matches!(tile.tile_type, TileType::Door) { return true; }
            }
        }
    }
    false
}

/// Find the best position for a door: boundary wall tile with fewest wall neighbors
/// (corner of the room, facing outward).
fn find_door_position(interior: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile>) -> Option<TilePos> {
    let interior_set: std::collections::HashSet<TilePos> = interior.iter().copied().collect();
    let mut best: Option<(TilePos, usize)> = None;

    for &pos in interior {
        for neighbor in pos.neighbors4() {
            if interior_set.contains(&neighbor) { continue; }
            if let Some(tile) = tiles.get(&neighbor) {
                if !tile.tile_type.is_wall() { continue; }
            } else {
                continue;
            }

            // Count wall neighbors of this candidate door position.
            let wall_count = neighbor.neighbors4().iter()
                .filter(|n| {
                    tiles.get(n).map(|t| t.tile_type.is_wall()).unwrap_or(false)
                })
                .count();

            // Prefer positions with exactly 2 wall neighbors (in a line, not corner).
            if wall_count == 2 {
                match &best {
                    None => { best = Some((neighbor, wall_count)); }
                    Some((_, prev_count)) if wall_count < *prev_count => {
                        best = Some((neighbor, wall_count));
                    }
                    _ => {}
                }
            } else if best.is_none() {
                best = Some((neighbor, wall_count));
            }
        }
    }
    best.map(|(pos, _)| pos)
}

/// Detect what function a room provides based on its furniture contents.
pub fn detect_room_function(interior: &[TilePos], tiles: &std::collections::HashMap<TilePos, Tile>) -> RoomFunction {
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
