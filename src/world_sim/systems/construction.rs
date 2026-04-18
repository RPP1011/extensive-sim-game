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
        // grow_room returns the post-growth interior + boundary. Reusing
        // them for completion checks eliminates three interior_set
        // HashSet constructions (is_enclosed/has_door/find_door_position).
        let (interior, boundary) = grow_room(&seed, &mut state.tiles, tick);
        if interior.len() as u32 >= seed.minimum_interior && is_enclosed(&boundary, &state.tiles) {
            // Room complete — place door if needed.
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

    // Prune completed seeds older than 500 ticks.
    state.build_seeds.retain(|s| !s.complete || tick.saturating_sub(s.tick) < 500);
}

/// Grow a room by one step: expand if too small, close if at minimum.
/// Returns the post-growth (interior, boundary) — caller reuses them for
/// completion checks instead of re-flooding and rebuilding sets.
fn grow_room(
    seed: &BuildSeed,
    tiles: &mut std::collections::HashMap<TilePos, Tile, ahash::RandomState>,
    tick: u64,
) -> (Vec<TilePos>, Vec<TilePos>) {
    // Ensure seed position has a floor tile.
    tiles.entry(seed.pos).or_insert(Tile {
        tile_type: TileType::Floor(TileMaterial::Wood),
        placed_by: Some(seed.placed_by),
        placed_tick: tick,
    });

    // Fused flood-fill + boundary computation: single BFS pass.
    let (mut interior, mut boundary) = flood_fill_with_boundary(seed.pos, tiles);

    if (interior.len() as u32) < seed.minimum_interior {
        // Below minimum: expand by placing floor tiles on empty boundary positions.
        let mut expanded = 0u32;
        // Snapshot boundary since we'll mutate tiles during iteration.
        let boundary_snapshot: Vec<TilePos> = boundary.clone();
        for bpos in &boundary_snapshot {
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
        // Re-flood to pick up fresh interior + boundary reflecting new floors.
        // Skip only when nothing expanded.
        if expanded > 0 {
            let refreshed = flood_fill_with_boundary(seed.pos, tiles);
            interior = refreshed.0;
            boundary = refreshed.1;
        }
    } else if (interior.len() as u32) < MAX_ROOM_SIZE {
        // At or above minimum: close open boundary with walls. Walls are
        // placed on boundary positions — they stay on the boundary set
        // (just now solid), so we don't need to re-flood.
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

    (interior, boundary)
}

/// Flood-fill from a position, collecting connected floor tiles AND the
/// boundary (non-floor tiles adjacent to the interior) in a single pass.
/// This replaces `flood_fill_floor` + `compute_boundary` — previously each
/// visited the same graph separately.
fn flood_fill_with_boundary(
    start: TilePos,
    tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>,
) -> (Vec<TilePos>, Vec<TilePos>) {
    let cap = (MAX_ROOM_SIZE as usize) * 4 + 16;
    let mut visited: std::collections::HashSet<TilePos, ahash::RandomState> =
        std::collections::HashSet::with_capacity_and_hasher(
            cap, ahash::RandomState::default());
    let mut queue = std::collections::VecDeque::with_capacity(cap);
    let mut interior = Vec::with_capacity(MAX_ROOM_SIZE as usize);
    let mut boundary = Vec::with_capacity(cap);

    if !tiles.get(&start).map(|t| t.tile_type.is_floor() || t.tile_type.is_furniture()).unwrap_or(false) {
        return (vec![start], Vec::new()); // seed is floor by convention
    }

    queue.push_back(start);
    visited.insert(start);

    while let Some(pos) = queue.pop_front() {
        interior.push(pos);
        if interior.len() > MAX_ROOM_SIZE as usize { break; }

        for neighbor in pos.neighbors4() {
            if !visited.insert(neighbor) { continue; }
            match tiles.get(&neighbor) {
                Some(tile)
                    if tile.tile_type.is_floor() || tile.tile_type.is_furniture()
                        || matches!(tile.tile_type, TileType::Door) =>
                {
                    queue.push_back(neighbor);
                }
                _ => {
                    // Not-a-floor (or no tile at all) adjacent to an interior
                    // tile — that's the boundary by definition.
                    boundary.push(neighbor);
                }
            }
        }
    }
    (interior, boundary)
}

/// Back-compat wrapper for call sites that only need interior.
fn flood_fill_floor(start: TilePos, tiles: &std::collections::HashMap<TilePos, Tile, ahash::RandomState>) -> Vec<TilePos> {
    flood_fill_with_boundary(start, tiles).0
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
