//! Procedural dungeon layout generation and voxel carving.
//!
//! `DungeonLayout::generate()` produces a set of rectangular rooms connected
//! by L-shaped corridors. `carve_into_chunk()` stamps the layout into an
//! existing chunk by clearing solid voxels to Air (Granite is never touched).

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use super::noise;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Axis-aligned rectangular room in world-voxel space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Room {
    pub min: (i32, i32, i32),
    pub max: (i32, i32, i32),
}

/// A corridor connecting two points (L-shaped: horizontal first, then vertical).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Corridor {
    pub from: (i32, i32, i32),
    pub to:   (i32, i32, i32),
    pub width: i32,
}

/// A fully-specified dungeon layout for a single cell/sector.
#[derive(Debug, Clone)]
pub struct DungeonLayout {
    pub rooms: Vec<Room>,
    pub corridors: Vec<Corridor>,
}

// ---------------------------------------------------------------------------
// Deterministic mini-LCG
// ---------------------------------------------------------------------------

fn lcg(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) ^ (*state >> 17)) as u32
}

fn lcg_range(state: &mut u64, lo: i32, hi: i32) -> i32 {
    assert!(hi > lo);
    lo + (lcg(state) as i32).abs() % (hi - lo)
}

// ---------------------------------------------------------------------------
// DungeonLayout impl
// ---------------------------------------------------------------------------

impl DungeonLayout {
    /// Generate a dungeon layout for a dungeon at cell (`cell_col`, `cell_row`)
    /// at world-Z starting at `cell_z`, with the given `depth` (affects room count).
    ///
    /// The generated coordinates are in world-voxel space, anchored at
    /// `(cell_col * CHUNK_SIZE, cell_row * CHUNK_SIZE, cell_z)`.
    pub fn generate(cell_col: i32, cell_row: i32, cell_z: i32, depth: u8, seed: u64) -> Self {
        let mut rng = seed
            .wrapping_add(cell_col as u64 * 0xDEAD_BEEF)
            .wrapping_add(cell_row as u64 * 0xCAFE_BABE)
            .wrapping_add(depth as u64 * 0x1234_5678);

        let base_x = cell_col * CHUNK_SIZE as i32;
        let base_y = cell_row * CHUNK_SIZE as i32;

        let hash_extra = noise::hash_3d(cell_col, cell_row, depth as i32, seed) % 3;
        let num_rooms = (3 + depth as i32 * 2 + hash_extra as i32).min(20);

        let mut rooms: Vec<Room> = Vec::with_capacity(num_rooms as usize);

        // Room bounding box: stays within [base_x, base_x + CHUNK_SIZE)
        let margin = 1i32;
        let max_coord = CHUNK_SIZE as i32 - margin;

        for _ in 0..num_rooms {
            // Width/height 4-10, depth (Z) 3-5
            let w = lcg_range(&mut rng, 4, 11);
            let h = lcg_range(&mut rng, 4, 11);
            let d = lcg_range(&mut rng, 3, 6);

            let x0 = base_x + margin + lcg_range(&mut rng, 0, (max_coord - w).max(1));
            let y0 = base_y + margin + lcg_range(&mut rng, 0, (max_coord - h).max(1));
            let z0 = cell_z + lcg_range(&mut rng, 0, (CHUNK_SIZE as i32 - d).max(1));

            rooms.push(Room {
                min: (x0, y0, z0),
                max: (x0 + w, y0 + h, z0 + d),
            });
        }

        // Connect adjacent rooms with L-shaped corridors (room i → room i+1)
        let mut corridors: Vec<Corridor> = Vec::new();
        for i in 0..(rooms.len().saturating_sub(1)) {
            let (ax, ay, az) = room_centre(&rooms[i]);
            let (bx, by, bz) = room_centre(&rooms[i + 1]);
            // Use Z midpoint for corridor
            let cz = (az + bz) / 2;
            let width = 1 + lcg_range(&mut rng, 0, 2); // 1 or 2
            // L-shaped: horizontal segment then vertical segment
            // Bend point: (bx, ay, cz)
            corridors.push(Corridor { from: (ax, ay, cz), to: (bx, ay, cz), width });
            corridors.push(Corridor { from: (bx, ay, cz), to: (bx, by, cz), width });
        }

        DungeonLayout { rooms, corridors }
    }

    /// Carve the dungeon layout into `chunk` by replacing solid voxels with Air.
    /// Granite is never carved.
    pub fn carve_into_chunk(&self, chunk: &mut Chunk, cp: ChunkPos) {
        let base_x = cp.x * CHUNK_SIZE as i32;
        let base_y = cp.y * CHUNK_SIZE as i32;
        let base_z = cp.z * CHUNK_SIZE as i32;
        let end_x = base_x + CHUNK_SIZE as i32;
        let end_y = base_y + CHUNK_SIZE as i32;
        let end_z = base_z + CHUNK_SIZE as i32;

        // Rooms
        for room in &self.rooms {
            let (rx0, ry0, rz0) = room.min;
            let (rx1, ry1, rz1) = room.max;
            let x0 = rx0.max(base_x);
            let y0 = ry0.max(base_y);
            let z0 = rz0.max(base_z);
            let x1 = rx1.min(end_x);
            let y1 = ry1.min(end_y);
            let z1 = rz1.min(end_z);
            for vz in z0..z1 {
                for vy in y0..y1 {
                    for vx in x0..x1 {
                        carve_voxel(chunk, vx, vy, vz, base_x, base_y, base_z);
                    }
                }
            }
        }

        // Corridors
        for corridor in &self.corridors {
            let (fx, fy, fz) = corridor.from;
            let (tx, ty, tz) = corridor.to;
            let w = corridor.width;
            let h = 3i32; // corridor height

            // Axis-aligned segment
            let x0 = fx.min(tx) - w / 2;
            let x1 = fx.max(tx) + w / 2 + 1;
            let y0 = fy.min(ty) - w / 2;
            let y1 = fy.max(ty) + w / 2 + 1;
            let z_mid = fz.min(tz);

            let cx0 = x0.max(base_x);
            let cy0 = y0.max(base_y);
            let cz0 = z_mid.max(base_z);
            let cx1 = x1.min(end_x);
            let cy1 = y1.min(end_y);
            let cz1 = (z_mid + h).min(end_z);

            for vz in cz0..cz1 {
                for vy in cy0..cy1 {
                    for vx in cx0..cx1 {
                        carve_voxel(chunk, vx, vy, vz, base_x, base_y, base_z);
                    }
                }
            }
        }

        chunk.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn room_centre(room: &Room) -> (i32, i32, i32) {
    (
        (room.min.0 + room.max.0) / 2,
        (room.min.1 + room.max.1) / 2,
        (room.min.2 + room.max.2) / 2,
    )
}

#[inline]
fn carve_voxel(chunk: &mut Chunk, vx: i32, vy: i32, vz: i32, base_x: i32, base_y: i32, base_z: i32) {
    let lx = (vx - base_x) as usize;
    let ly = (vy - base_y) as usize;
    let lz = (vz - base_z) as usize;
    if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE || lz >= CHUNK_SIZE {
        return;
    }
    let idx = local_index(lx, ly, lz);
    let mat = chunk.voxels[idx].material;
    if mat == VoxelMaterial::Granite || !mat.is_solid() {
        return;
    }
    chunk.voxels[idx] = Voxel::new(VoxelMaterial::Air);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::ChunkPos;

    fn fill_chunk(cp: ChunkPos, mat: VoxelMaterial) -> Chunk {
        let mut chunk = Chunk::new_air(cp);
        for v in chunk.voxels.iter_mut() {
            *v = Voxel::new(mat);
        }
        chunk
    }

    #[test]
    fn dungeon_carves_rooms() {
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Stone);
        let layout = DungeonLayout::generate(0, 0, 0, 2, 42);
        layout.carve_into_chunk(&mut chunk, cp);
        let air_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        assert!(air_count > 50, "dungeon carved too little: {air_count} Air voxels");
    }

    #[test]
    fn dungeon_has_rooms_and_corridors() {
        let layout = DungeonLayout::generate(1, 2, 0, 3, 99);
        assert!(layout.rooms.len() >= 3, "too few rooms: {}", layout.rooms.len());
        // L-shaped corridors: each room pair → 2 corridor segments
        assert!(!layout.corridors.is_empty(), "no corridors generated");
    }

    #[test]
    fn dungeon_is_deterministic() {
        let cp = ChunkPos::new(2, 3, -1);
        let layout_a = DungeonLayout::generate(2, 3, -16, 4, 7777);
        let layout_b = DungeonLayout::generate(2, 3, -16, 4, 7777);
        assert_eq!(layout_a.rooms.len(), layout_b.rooms.len());
        for (r1, r2) in layout_a.rooms.iter().zip(layout_b.rooms.iter()) {
            assert_eq!(r1, r2);
        }

        let mut a = fill_chunk(cp, VoxelMaterial::Stone);
        let mut b = fill_chunk(cp, VoxelMaterial::Stone);
        layout_a.carve_into_chunk(&mut a, cp);
        layout_b.carve_into_chunk(&mut b, cp);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }

    #[test]
    fn granite_not_carved_by_dungeon() {
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Granite);
        let layout = DungeonLayout::generate(0, 0, 0, 5, 42);
        layout.carve_into_chunk(&mut chunk, cp);
        let carved = chunk.voxels.iter().filter(|v| v.material != VoxelMaterial::Granite).count();
        assert_eq!(carved, 0, "{carved} granite voxels were carved");
    }
}
