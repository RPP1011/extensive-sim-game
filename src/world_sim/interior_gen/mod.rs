//! Procedural building interior generation using Wave Function Collapse.
//!
//! Generates floor plans with walls, doors, rooms, and furniture from a
//! deterministic seed. Each building's `room_seed` produces a unique but
//! reproducible interior layout.

pub mod tiles;
pub mod solver;

use tiles::{Tile, Socket, build_rules, apply_weight_overrides};
use super::state::BuildingType;

/// Generated interior layout for a building.
pub struct InteriorLayout {
    pub width: usize,
    pub height: usize,
    /// One tile grid per floor. `floors[0]` is ground floor.
    pub floors: Vec<Vec<Tile>>,
}

/// Number of floors based on building tier.
fn floor_count(tier: u8) -> usize {
    match tier {
        0 | 1 => 1,
        2 => 2,
        3 => 3,
        _ => 1,
    }
}

/// Footprint size in grid cells (width, height) based on building type and tier.
pub fn footprint_size(building_type: BuildingType, tier: u8) -> (usize, usize) {
    let base = match building_type {
        BuildingType::House => (1, 1),
        BuildingType::Longhouse => (2, 1),
        BuildingType::Inn => (2, 1),
        BuildingType::Farm => (2, 1),
        BuildingType::Forge => (1, 1),
        BuildingType::Mine => (1, 1),
        BuildingType::Sawmill => (2, 1),
        BuildingType::Apothecary => (1, 1),
        BuildingType::Temple => (2, 2),
        BuildingType::Library => (2, 1),
        BuildingType::Barracks => (2, 2),
        BuildingType::Market => (2, 2),
        BuildingType::GuildHall => (2, 2),
        BuildingType::Treasury => (2, 2),
        BuildingType::Warehouse => (2, 1),
        _ => (1, 1),
    };
    let growth = match tier {
        2 => 1,
        3 => 2,
        _ => 0,
    };
    (base.0 + growth, base.1 + growth)
}

/// Interior tiles per city grid cell (each cell = 2 world units, each tile = 0.5 wu).
const TILES_PER_CELL: usize = 4;

/// Generate a complete interior layout for a building.
///
/// Deterministic: same `room_seed` always produces the same layout.
pub fn generate_interior(
    building_type: BuildingType,
    tier: u8,
    room_seed: u64,
    footprint_w: usize,
    footprint_h: usize,
) -> Option<InteriorLayout> {
    let iw = footprint_w * TILES_PER_CELL;
    let ih = footprint_h * TILES_PER_CELL;
    let num_floors = floor_count(tier);

    let mut rules = build_rules();
    apply_weight_overrides(&mut rules, building_type);

    // Find the all-wall rule index for boundary pinning
    let wall_idx = rules.iter().position(|r| {
        r.tile == Tile::Wall && r.sockets == [Socket::WallFace; 4]
    })?;

    // Find a door rule (floor on N+S, wall on E+W)
    let door_idx = rules.iter().position(|r| r.tile == Tile::Door)?;

    let mut floors = Vec::with_capacity(num_floors);

    for floor_idx in 0..num_floors {
        let floor_seed = room_seed ^ (floor_idx as u64 * 0x1234_5678_9ABC_DEF0);

        // For small interiors (< 6 tiles in either dimension), skip WFC and
        // generate a simple box: wall border + floor interior + door.
        if iw < 6 || ih < 6 {
            let mut tiles = vec![Tile::Floor; iw * ih];
            // Walls on boundary
            for x in 0..iw {
                tiles[x] = Tile::Wall;
                tiles[(ih - 1) * iw + x] = Tile::Wall;
            }
            for y in 0..ih {
                tiles[y * iw] = Tile::Wall;
                tiles[y * iw + iw - 1] = Tile::Wall;
            }
            // Door on south edge center (ground floor only)
            if floor_idx == 0 {
                tiles[(ih - 1) * iw + iw / 2] = Tile::Door;
            }
            // Stairs for multi-story
            if num_floors > 1 && iw > 2 && ih > 2 {
                tiles[1 * iw + 1] = Tile::Stairs;
            }
            floors.push(tiles);
            continue;
        }

        let mut pins = Vec::new();
        let door_x = iw / 2;
        let has_door = floor_idx == 0;

        // Pin boundary to wall (skip door position)
        for x in 0..iw {
            if !(has_door && x == door_x) {
                pins.push((x, ih - 1, wall_idx));
            }
            pins.push((x, 0, wall_idx));
        }
        for y in 1..ih-1 {
            pins.push((0, y, wall_idx));
            pins.push((iw - 1, y, wall_idx));
        }

        // Place entrance door on ground floor (south edge, center)
        if has_door {
            pins.push((door_x, ih - 1, door_idx));
        }

        // Place stairs for multi-story buildings — use floor rule (stairs have same sockets as floor)
        // The tile gets overwritten to Stairs after WFC solves.
        let stairs_pos = if num_floors > 1 {
            // Place stairs safely in the interior (at least 2 cells from boundary)
            let sx = 2 + ((floor_seed >> 8) as usize % (iw.saturating_sub(5)).max(1));
            let sy = 2 + ((floor_seed >> 16) as usize % (ih.saturating_sub(5)).max(1));
            // Pin to floor (stairs have floor sockets, WFC treats them the same)
            let floor_idx_rule = rules.iter().position(|r| r.tile == Tile::Floor).unwrap_or(0);
            pins.push((sx, sy, floor_idx_rule));
            Some((sx, sy))
        } else {
            None
        };

        let result = solver::solve(iw, ih, &rules, floor_seed, &pins)?;
        let mut tiles = result.tiles;
        // Overwrite stairs position
        if let Some((sx, sy)) = stairs_pos {
            tiles[sy * iw + sx] = Tile::Stairs;
        }
        floors.push(tiles);
    }

    Some(InteriorLayout { width: iw, height: ih, floors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_house_interior() {
        let layout = generate_interior(BuildingType::House, 0, 42, 1, 1);
        assert!(layout.is_some(), "Failed to generate house interior");
        let layout = layout.unwrap();
        assert_eq!(layout.width, 4);
        assert_eq!(layout.height, 4);
        assert_eq!(layout.floors.len(), 1);
        assert_eq!(layout.floors[0].len(), 16);
    }

    #[test]
    fn generate_guildhall_interior() {
        let layout = generate_interior(BuildingType::GuildHall, 1, 99, 2, 2);
        assert!(layout.is_some(), "Failed to generate guildhall interior");
        let layout = layout.unwrap();
        assert_eq!(layout.width, 8);
        assert_eq!(layout.height, 8);
        assert_eq!(layout.floors.len(), 1);
    }

    #[test]
    fn generate_multistory() {
        let layout = generate_interior(BuildingType::Temple, 2, 777, 3, 3);
        assert!(layout.is_some(), "Failed to generate multi-story temple");
        let layout = layout.unwrap();
        assert_eq!(layout.floors.len(), 2, "Tier 2 should have 2 floors");
    }

    #[test]
    fn deterministic_generation() {
        let a = generate_interior(BuildingType::Inn, 1, 12345, 2, 1).unwrap();
        let b = generate_interior(BuildingType::Inn, 1, 12345, 2, 1).unwrap();
        assert_eq!(a.floors[0], b.floors[0], "Same seed must produce same interior");
    }

    #[test]
    fn different_seeds_differ() {
        // Use a building large enough for WFC (>= 6x6 interior = 2x2 footprint)
        let a = generate_interior(BuildingType::GuildHall, 1, 111, 2, 2).unwrap();
        let b = generate_interior(BuildingType::GuildHall, 1, 222, 2, 2).unwrap();
        assert_ne!(a.floors[0], b.floors[0], "Different seeds should differ");
    }
}
