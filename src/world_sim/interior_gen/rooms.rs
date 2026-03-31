//! Room assignment via flood fill for WFC-generated building interiors.
//!
//! Identifies connected floor regions in a tile grid, then maps each region
//! to a functional `RoomKind` based on the building type.

use super::tiles::Tile;
use super::InteriorLayout;
use crate::world_sim::state::{BuildingType, Room, RoomKind};

/// A connected region of floor tiles discovered by flood fill.
pub struct FloorRegion {
    /// Grid coordinates of every tile in this region.
    pub cells: Vec<(usize, usize)>,
    /// Centre-of-mass of the region in tile coordinates.
    pub centroid: (f32, f32),
    /// Number of tiles (== cells.len()).
    pub area: usize,
}

/// Flood-fill all connected floor tiles into distinct regions separated by
/// walls/pillars/windows/empty tiles.
pub fn find_floor_regions(tiles: &[Tile], w: usize, h: usize) -> Vec<FloorRegion> {
    assert_eq!(tiles.len(), w * h, "tile count must match w*h");
    let mut visited = vec![false; w * h];
    let mut regions = Vec::new();

    for start_y in 0..h {
        for start_x in 0..w {
            let idx = start_y * w + start_x;
            if visited[idx] || !tiles[idx].is_floor() {
                continue;
            }
            // BFS flood fill from this unvisited floor tile.
            let mut cells = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((start_x, start_y));
            visited[idx] = true;

            while let Some((x, y)) = queue.pop_front() {
                cells.push((x, y));
                // 4-connected neighbours.
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let ni = ny * w + nx;
                    if !visited[ni] && tiles[ni].is_floor() {
                        visited[ni] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }

            let area = cells.len();
            let (sx, sy) = cells.iter().fold((0.0f32, 0.0f32), |(ax, ay), &(cx, cy)| {
                (ax + cx as f32, ay + cy as f32)
            });
            let centroid = (sx / area as f32, sy / area as f32);

            regions.push(FloorRegion { cells, centroid, area });
        }
    }

    regions
}

/// Map discovered floor regions to `RoomKind`s appropriate for the building
/// type. Returns `Room` structs with offsets derived from region centroids
/// relative to the grid centre.
///
/// Strategy:
/// - The largest region is assigned `Entrance` (if the default_rooms list
///   contains one) or `Hearth`.
/// - Remaining regions are matched to the remaining kinds from the building
///   type's `default_rooms()` list, cycling if there are more regions than
///   default rooms.
/// - `seed` is used to shuffle assignment when there are more default kinds
///   than regions, so different seeds pick different subsets.
pub fn assign_room_kinds(
    building_type: BuildingType,
    regions: &[FloorRegion],
    seed: u64,
) -> Vec<Room> {
    if regions.is_empty() {
        return Vec::new();
    }

    let defaults = building_type.default_rooms();
    let default_kinds: Vec<RoomKind> = defaults.iter().map(|r| r.kind).collect();

    // Find index of the largest region.
    let largest_idx = regions
        .iter()
        .enumerate()
        .max_by_key(|(_, r)| r.area)
        .map(|(i, _)| i)
        .unwrap(); // regions is non-empty

    // Determine the primary kind for the largest region.
    let primary_kind = if default_kinds.contains(&RoomKind::Entrance) {
        RoomKind::Entrance
    } else if default_kinds.contains(&RoomKind::Hearth) {
        RoomKind::Hearth
    } else {
        default_kinds[0]
    };

    // Remaining kinds (everything except the primary).
    let mut remaining_kinds: Vec<RoomKind> = default_kinds
        .iter()
        .copied()
        .filter(|k| *k != primary_kind)
        .collect();
    if remaining_kinds.is_empty() {
        // Building only has one room kind — use it for everything.
        remaining_kinds.push(primary_kind);
    }

    // Simple deterministic shuffle of remaining_kinds based on seed.
    {
        let n = remaining_kinds.len();
        if n > 1 {
            let mut s = seed;
            for i in (1..n).rev() {
                // xorshift-style mix.
                s ^= s.wrapping_shl(13);
                s ^= s.wrapping_shr(7);
                s ^= s.wrapping_shl(17);
                let j = (s as usize) % (i + 1);
                remaining_kinds.swap(i, j);
            }
        }
    }

    // Grid centre for offset computation.
    // We don't know the grid dimensions from regions alone, so derive from
    // the bounding box of all cells across all regions.
    let (mut max_x, mut max_y) = (0usize, 0usize);
    for r in regions {
        for &(x, y) in &r.cells {
            if x > max_x { max_x = x; }
            if y > max_y { max_y = y; }
        }
    }
    let grid_cx = max_x as f32 / 2.0;
    let grid_cy = max_y as f32 / 2.0;

    let mut rooms = Vec::with_capacity(regions.len());
    let mut remaining_idx = 0usize;

    for (i, region) in regions.iter().enumerate() {
        let kind = if i == largest_idx {
            primary_kind
        } else {
            let k = remaining_kinds[remaining_idx % remaining_kinds.len()];
            remaining_idx += 1;
            k
        };

        // Normalise centroid into building-local offset (roughly -1.0 .. 1.0).
        let scale = (max_x.max(max_y) as f32).max(1.0);
        let offset = (
            (region.centroid.0 - grid_cx) / scale,
            (region.centroid.1 - grid_cy) / scale,
        );

        rooms.push(Room {
            kind,
            offset,
            occupant_id: None,
        });
    }

    rooms
}

/// Convenience: discover floor regions in an `InteriorLayout` and assign room
/// kinds. Operates on the ground floor (`floors[0]`).
pub fn generate_rooms(
    layout: &InteriorLayout,
    building_type: BuildingType,
    seed: u64,
) -> Vec<Room> {
    if layout.floors.is_empty() {
        return Vec::new();
    }
    let tiles = &layout.floors[0];
    let regions = find_floor_regions(tiles, layout.width, layout.height);
    assign_room_kinds(building_type, &regions, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::interior_gen::generate_interior;

    #[test]
    fn generated_interior_has_rooms() {
        // Use a footprint large enough for WFC (2x2 = 8x8 tiles).
        let layout = generate_interior(BuildingType::GuildHall, 1, 42, 2, 2)
            .expect("interior generation should succeed");
        let rooms = generate_rooms(&layout, BuildingType::GuildHall, 42);
        assert!(
            !rooms.is_empty(),
            "At least one room should be found in a generated interior"
        );
    }

    #[test]
    fn room_offsets_within_bounds() {
        let layout = generate_interior(BuildingType::Inn, 1, 99, 2, 1)
            .expect("interior generation should succeed");
        let rooms = generate_rooms(&layout, BuildingType::Inn, 99);
        assert!(!rooms.is_empty());
        for room in &rooms {
            assert!(
                room.offset.0.abs() <= 1.0,
                "x offset {} out of [-1, 1]",
                room.offset.0
            );
            assert!(
                room.offset.1.abs() <= 1.0,
                "y offset {} out of [-1, 1]",
                room.offset.1
            );
        }
    }

    #[test]
    fn simple_grid_flood_fill() {
        // Hand-crafted 4x4: wall border, floor interior, door on south.
        #[rustfmt::skip]
        let tiles = [
            Tile::Wall,  Tile::Wall,  Tile::Wall,  Tile::Wall,
            Tile::Wall,  Tile::Floor, Tile::Floor, Tile::Wall,
            Tile::Wall,  Tile::Floor, Tile::Floor, Tile::Wall,
            Tile::Wall,  Tile::Door,  Tile::Wall,  Tile::Wall,
        ];
        let regions = find_floor_regions(&tiles, 4, 4);
        // The 4 floor tiles + 1 door tile are connected → exactly 1 region.
        assert_eq!(regions.len(), 1, "Should be exactly 1 connected floor region");
        assert_eq!(regions[0].area, 5);
    }

    #[test]
    fn two_rooms_separated_by_wall() {
        // 5x3 grid: two floor pockets separated by a wall column.
        #[rustfmt::skip]
        let tiles = [
            Tile::Wall,  Tile::Wall,  Tile::Wall,  Tile::Wall,  Tile::Wall,
            Tile::Wall,  Tile::Floor, Tile::Wall,  Tile::Floor, Tile::Wall,
            Tile::Wall,  Tile::Wall,  Tile::Wall,  Tile::Wall,  Tile::Wall,
        ];
        let regions = find_floor_regions(&tiles, 5, 3);
        assert_eq!(regions.len(), 2, "Two floor pockets should produce 2 regions");
        assert_eq!(regions[0].area, 1);
        assert_eq!(regions[1].area, 1);
    }

    #[test]
    fn assign_room_kinds_largest_is_entrance() {
        // Build two regions of different sizes.
        let small = FloorRegion {
            cells: vec![(1, 1)],
            centroid: (1.0, 1.0),
            area: 1,
        };
        let large = FloorRegion {
            cells: vec![(3, 1), (4, 1), (3, 2), (4, 2)],
            centroid: (3.5, 1.5),
            area: 4,
        };
        let rooms = assign_room_kinds(BuildingType::GuildHall, &[small, large], 0);
        assert_eq!(rooms.len(), 2);
        // The larger region (index 1) should get Entrance.
        assert_eq!(rooms[1].kind, RoomKind::Entrance);
    }

    #[test]
    fn no_floor_tiles_yields_no_rooms() {
        let tiles = vec![Tile::Wall; 9];
        let regions = find_floor_regions(&tiles, 3, 3);
        assert!(regions.is_empty());
        let rooms = assign_room_kinds(BuildingType::House, &regions, 0);
        assert!(rooms.is_empty());
    }
}
