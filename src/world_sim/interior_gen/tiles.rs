//! Tile definitions and edge socket constraints for WFC interior generation.

use serde::{Deserialize, Serialize};

/// Interior tile types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Tile {
    Empty     = 0,
    Floor     = 1,
    Wall      = 2,
    Door      = 3,
    Window    = 4,
    Pillar    = 5,
    Stairs    = 6,
    // Furniture (placed on floor as overlay)
    Table     = 7,
    Bed       = 8,
    Anvil     = 9,
    Bookshelf = 10,
    Altar     = 11,
    CounterT  = 12,
    Chest     = 13,
    HearthT   = 14,
    Barrel    = 15,
}

impl Tile {
    /// Is this tile walkable (floor-like)?
    pub fn is_floor(self) -> bool {
        matches!(self, Tile::Floor | Tile::Door | Tile::Stairs
            | Tile::Table | Tile::Bed | Tile::Anvil | Tile::Bookshelf
            | Tile::Altar | Tile::CounterT | Tile::Chest | Tile::HearthT | Tile::Barrel)
    }

    /// Is this tile a solid wall/pillar?
    pub fn is_wall(self) -> bool {
        matches!(self, Tile::Wall | Tile::Pillar | Tile::Window)
    }
}

/// Edge socket types for adjacency constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Socket {
    /// Outside the building (connects to Empty)
    Void,
    /// Solid wall face
    WallFace,
    /// Wall face with opening (door/window position)
    WallOpen,
    /// Floor-to-floor connection
    FloorEdge,
}

/// Check if two sockets can be adjacent (a is one tile's edge, b is the neighbor's opposing edge).
pub fn compatible(a: Socket, b: Socket) -> bool {
    matches!((a, b),
        (Socket::FloorEdge, Socket::FloorEdge) |
        (Socket::WallFace, Socket::WallFace) |
        (Socket::WallOpen, Socket::FloorEdge) |
        (Socket::FloorEdge, Socket::WallOpen) |
        (Socket::WallFace, Socket::Void) |
        (Socket::Void, Socket::WallFace) |
        (Socket::Void, Socket::Void)
    )
}

/// A tile variant with edge sockets and generation weight.
#[derive(Debug, Clone, Copy)]
pub struct TileRule {
    pub tile: Tile,
    /// Edge sockets: [North, East, South, West]
    pub sockets: [Socket; 4],
    /// Frequency weight for WFC selection (higher = more common).
    pub weight: f32,
}

/// Direction indices for socket arrays.
pub const NORTH: usize = 0;
pub const EAST: usize = 1;
pub const SOUTH: usize = 2;
pub const WEST: usize = 3;

/// Returns the opposing direction index.
pub fn opposite(dir: usize) -> usize {
    (dir + 2) % 4
}

/// Build the full tile rule set for building interiors.
/// Includes rotated variants where applicable.
pub fn build_rules() -> Vec<TileRule> {
    let mut rules = Vec::with_capacity(40);

    // Empty (outside building)
    rules.push(TileRule {
        tile: Tile::Empty,
        sockets: [Socket::Void, Socket::Void, Socket::Void, Socket::Void],
        weight: 0.01, // very low — only at boundary
    });

    // Floor (all edges connect to floor)
    rules.push(TileRule {
        tile: Tile::Floor,
        sockets: [Socket::FloorEdge, Socket::FloorEdge, Socket::FloorEdge, Socket::FloorEdge],
        weight: 10.0, // most common interior tile
    });

    // Wall (all edges are wall faces) — interior partition
    rules.push(TileRule {
        tile: Tile::Wall,
        sockets: [Socket::WallFace, Socket::WallFace, Socket::WallFace, Socket::WallFace],
        weight: 3.0,
    });

    // Pillar (same as wall but different visual)
    rules.push(TileRule {
        tile: Tile::Pillar,
        sockets: [Socket::WallFace, Socket::WallFace, Socket::WallFace, Socket::WallFace],
        weight: 0.3,
    });

    // Door — 4 rotations. A door has wall on 2 sides, floor on 2 sides.
    // Door oriented N-S (opening connects north and south rooms)
    for rot in 0..4 {
        let base = [Socket::FloorEdge, Socket::WallFace, Socket::FloorEdge, Socket::WallFace];
        let sockets = rotate_sockets(base, rot);
        rules.push(TileRule {
            tile: Tile::Door,
            sockets,
            weight: 1.5,
        });
    }

    // Window — like wall but with WallOpen on one side (2 rotations meaningful)
    for rot in 0..4 {
        let base = [Socket::WallOpen, Socket::WallFace, Socket::WallFace, Socket::WallFace];
        let sockets = rotate_sockets(base, rot);
        rules.push(TileRule {
            tile: Tile::Window,
            sockets,
            weight: 0.5,
        });
    }

    // Wall-floor transitions: wall on one side, floor on others (L-shapes, T-shapes)
    // Wall with floor on 3 sides (end of wall segment)
    for rot in 0..4 {
        let base = [Socket::WallFace, Socket::FloorEdge, Socket::FloorEdge, Socket::FloorEdge];
        let sockets = rotate_sockets(base, rot);
        rules.push(TileRule {
            tile: Tile::Wall,
            sockets,
            weight: 2.0,
        });
    }

    // Wall with floor on 2 adjacent sides (corner)
    for rot in 0..4 {
        let base = [Socket::WallFace, Socket::WallFace, Socket::FloorEdge, Socket::FloorEdge];
        let sockets = rotate_sockets(base, rot);
        rules.push(TileRule {
            tile: Tile::Wall,
            sockets,
            weight: 2.5,
        });
    }

    // Wall with floor on 2 opposite sides (partition wall)
    for rot in 0..2 {
        let base = [Socket::FloorEdge, Socket::WallFace, Socket::FloorEdge, Socket::WallFace];
        let sockets = rotate_sockets(base, rot);
        rules.push(TileRule {
            tile: Tile::Wall,
            sockets,
            weight: 1.5,
        });
    }

    // Stairs (floor on all sides — behaves like floor for adjacency)
    rules.push(TileRule {
        tile: Tile::Stairs,
        sockets: [Socket::FloorEdge, Socket::FloorEdge, Socket::FloorEdge, Socket::FloorEdge],
        weight: 0.0, // never placed by WFC — only pinned explicitly
    });

    rules
}

/// Rotate socket array clockwise by `n` 90-degree steps.
fn rotate_sockets(sockets: [Socket; 4], n: usize) -> [Socket; 4] {
    let n = n % 4;
    let mut out = sockets;
    for _ in 0..n {
        out = [out[3], out[0], out[1], out[2]];
    }
    out
}

/// Apply building-type-specific weight overrides.
pub fn apply_weight_overrides(rules: &mut [TileRule], building_type: super::super::state::BuildingType) {
    use super::super::state::BuildingType;
    let overrides: &[(Tile, f32)] = match building_type {
        BuildingType::Forge => &[(Tile::Anvil, 2.0), (Tile::HearthT, 1.5), (Tile::Barrel, 1.0)],
        BuildingType::Library => &[(Tile::Bookshelf, 3.0), (Tile::Table, 1.5)],
        BuildingType::Temple => &[(Tile::Altar, 2.0), (Tile::Pillar, 1.5)],
        BuildingType::Barracks => &[(Tile::Bed, 2.0), (Tile::Chest, 1.0)],
        BuildingType::Market => &[(Tile::CounterT, 2.5), (Tile::Barrel, 1.5)],
        BuildingType::Inn => &[(Tile::Bed, 1.5), (Tile::Table, 2.0), (Tile::Barrel, 1.0)],
        BuildingType::House | BuildingType::Longhouse => &[(Tile::Bed, 1.5), (Tile::Table, 1.0), (Tile::HearthT, 1.0)],
        _ => &[],
    };
    for &(tile, extra_weight) in overrides {
        for rule in rules.iter_mut() {
            if rule.tile == tile {
                rule.weight += extra_weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_is_symmetric() {
        let sockets = [Socket::Void, Socket::WallFace, Socket::WallOpen, Socket::FloorEdge];
        for &a in &sockets {
            for &b in &sockets {
                assert_eq!(compatible(a, b), compatible(b, a),
                    "Asymmetric: {:?} vs {:?}", a, b);
            }
        }
    }

    #[test]
    fn rules_are_nonempty() {
        let rules = build_rules();
        assert!(rules.len() >= 20, "Expected at least 20 tile rules, got {}", rules.len());
    }

    #[test]
    fn rotate_identity() {
        let s = [Socket::FloorEdge, Socket::WallFace, Socket::Void, Socket::WallOpen];
        assert_eq!(rotate_sockets(s, 0), s);
        assert_eq!(rotate_sockets(s, 4), s);
    }
}
