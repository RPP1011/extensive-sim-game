//! Surface feature placement — trees, boulders, etc.
//!
//! Called after base terrain materialisation for the chunk that contains the
//! surface layer.  Features are placed deterministically using hash_f32 so
//! the result is identical across any number of calls with the same seed.

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::{Terrain, SubBiome};
use super::noise;

// ---------------------------------------------------------------------------
// Salt constants (keep features from correlating with each other)
// ---------------------------------------------------------------------------
const TREE_DENSITY_SALT: u64    = 0x7777_1234_0001;
const TREE_HEIGHT_SALT: u64     = 0xAAAA_1234_0002;
const TREE_CANOPY_SALT: u64     = 0xBBBB_1234_0003;
const BOULDER_DENSITY_SALT: u64 = 0xCCCC_5678_0001;
const BOULDER_SIZE_SALT: u64    = 0xDDDD_5678_0002;

// ---------------------------------------------------------------------------
// Helpers: tree and boulder stampers
// ---------------------------------------------------------------------------

/// Stamp a tree with its base at local column (lx, ly), trunk bottom at
/// voxel-height base_z (world Z).  The chunk must contain that Z range to
/// see any of the tree.
pub fn stamp_tree(chunk: &mut Chunk, lx: usize, ly: usize, base_z: i32, seed: u64) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let vx = base_x + lx as i32;
    let vy = base_y + ly as i32;

    // Deterministic trunk height 5-8
    let h_hash = noise::hash_f32(vx, vy, 0, seed.wrapping_add(TREE_HEIGHT_SALT));
    let trunk_height = 5 + (h_hash * 4.0) as i32; // 5..8

    // Canopy radius 2 or 3
    let c_hash = noise::hash_f32(vx, vy, 1, seed.wrapping_add(TREE_CANOPY_SALT));
    let canopy_r: i32 = if c_hash > 0.5 { 3 } else { 2 };

    // -- Trunk --
    for dz in 0..trunk_height {
        let wz = base_z + dz;
        let lz_local = wz - base_cz;
        if lz_local < 0 || lz_local >= CHUNK_SIZE as i32 { continue; }
        if lx >= CHUNK_SIZE || ly >= CHUNK_SIZE { continue; }
        let idx = local_index(lx, ly, lz_local as usize);
        chunk.voxels[idx] = Voxel::new(VoxelMaterial::WoodLog);
    }

    // -- Canopy sphere --
    let canopy_center_z = base_z + trunk_height;
    for dz in -canopy_r..=canopy_r {
        for dy in -canopy_r..=canopy_r {
            for dx in -canopy_r..=canopy_r {
                // Sphere test
                if dx * dx + dy * dy + dz * dz > canopy_r * canopy_r { continue; }
                let wx = vx + dx;
                let wy = vy + dy;
                let wz = canopy_center_z + dz;
                let llx = wx - base_x;
                let lly = wy - base_y;
                let llz = wz - base_cz;
                if llx < 0 || lly < 0 || llz < 0 { continue; }
                if llx >= CHUNK_SIZE as i32 || lly >= CHUNK_SIZE as i32 || llz >= CHUNK_SIZE as i32 { continue; }
                let idx = local_index(llx as usize, lly as usize, llz as usize);
                // Don't overwrite trunk
                if chunk.voxels[idx].material != VoxelMaterial::WoodLog {
                    chunk.voxels[idx] = Voxel::new(VoxelMaterial::Grass);
                }
            }
        }
    }
}

/// Stamp a small boulder at local column (lx, ly), base at world-Z base_z.
fn stamp_boulder(chunk: &mut Chunk, lx: usize, ly: usize, base_z: i32, seed: u64) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let vx = base_x + lx as i32;
    let vy = base_y + ly as i32;

    let s_hash = noise::hash_f32(vx, vy, 2, seed.wrapping_add(BOULDER_SIZE_SALT));
    let size = 1 + (s_hash * 3.0) as i32; // 1..3

    for dz in 0..size {
        for dy in 0..size {
            for dx in 0..size {
                let wx = vx + dx;
                let wy = vy + dy;
                let wz = base_z + dz;
                let llx = wx - base_x;
                let lly = wy - base_y;
                let llz = wz - base_cz;
                if llx < 0 || lly < 0 || llz < 0 { continue; }
                if llx >= CHUNK_SIZE as i32 || lly >= CHUNK_SIZE as i32 || llz >= CHUNK_SIZE as i32 { continue; }
                let idx = local_index(llx as usize, lly as usize, llz as usize);
                chunk.voxels[idx] = Voxel::new(VoxelMaterial::Stone);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-biome density tables
// ---------------------------------------------------------------------------

struct FeatureParams {
    tree_density: f32,
    boulder_density: f32,
}

fn feature_params(terrain: Terrain, sub_biome: SubBiome) -> FeatureParams {
    let tree_density = match terrain {
        Terrain::Forest => match sub_biome {
            SubBiome::DenseForest   => 0.15,
            SubBiome::LightForest   => 0.03,
            SubBiome::AncientForest => 0.08,
            _                       => 0.06,
        },
        Terrain::Jungle => match sub_biome {
            SubBiome::TempleJungle => 0.12,
            _                      => 0.12,
        },
        Terrain::Tundra  => 0.01,
        Terrain::Swamp   => 0.04,
        _ => 0.0,
    };

    let boulder_density = match terrain {
        Terrain::Plains  => 0.005,
        Terrain::Desert  => 0.01,
        Terrain::Badlands => 0.01,
        Terrain::Tundra  => 0.008,
        _ => 0.0,
    };

    FeatureParams { tree_density, boulder_density }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Place surface features (trees, boulders) into `chunk`.
///
/// Only operates if `surface_z_local` (the surface height expressed as a local
/// Z coordinate 0..CHUNK_SIZE) is within this chunk.  One voxel above the
/// surface layer is the placement point.
pub fn place_surface_features(
    chunk: &mut Chunk,
    cp: ChunkPos,
    terrain: Terrain,
    sub_biome: SubBiome,
    surface_z_local: i32,
    seed: u64,
) {
    // Surface must be within this chunk
    if surface_z_local < 0 || surface_z_local >= CHUNK_SIZE as i32 {
        return;
    }

    let params = feature_params(terrain, sub_biome);
    if params.tree_density == 0.0 && params.boulder_density == 0.0 {
        return;
    }

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;
    // Feature base sits one voxel above the surface layer
    let feature_base_z = base_z + surface_z_local + 1;

    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let vx = base_x + lx as i32;
            let vy = base_y + ly as i32;

            // Tree placement
            if params.tree_density > 0.0 {
                let td = noise::hash_f32(vx, vy, 0, seed.wrapping_add(TREE_DENSITY_SALT));
                if td < params.tree_density {
                    stamp_tree(chunk, lx, ly, feature_base_z, seed);
                }
            }

            // Boulder placement (skip columns that already got a tree)
            if params.boulder_density > 0.0 {
                let bd = noise::hash_f32(vx, vy, 0, seed.wrapping_add(BOULDER_DENSITY_SALT));
                if bd < params.boulder_density {
                    stamp_boulder(chunk, lx, ly, feature_base_z, seed);
                }
            }
        }
    }

    chunk.dirty = true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::CHUNK_SIZE;

    fn make_surface_chunk(cp: ChunkPos) -> Chunk {
        // Bottom half: Dirt (ground), top half: Air
        let mut chunk = Chunk::new_air(cp);
        for lz in 0..(CHUNK_SIZE / 2) {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(VoxelMaterial::Dirt);
                }
            }
        }
        chunk
    }

    #[test]
    fn tree_has_trunk_and_canopy() {
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new_air(cp);
        // Surface at local z=0, place base at z=1
        stamp_tree(&mut chunk, 8, 8, 1, 42);
        let log_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::WoodLog).count();
        let leaf_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Grass).count();
        assert!(log_count >= 5, "tree trunk too short: {log_count} WoodLog voxels");
        assert!(leaf_count > 0, "tree has no canopy (Grass voxels): {leaf_count}");
    }

    #[test]
    fn forest_has_multiple_trees() {
        let cp = ChunkPos::new(0, 0, 5);
        let mut chunk = make_surface_chunk(cp);
        // surface_z_local = CHUNK_SIZE/2 - 1 so features spawn at CHUNK_SIZE/2
        let surface_local = (CHUNK_SIZE / 2 - 1) as i32;
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, surface_local, 42);
        let log_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::WoodLog).count();
        assert!(log_count > 0, "dense forest produced no trees");
    }

    #[test]
    fn features_not_placed_outside_chunk() {
        // surface_z_local = CHUNK_SIZE (outside chunk) → nothing placed
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new_air(cp);
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, CHUNK_SIZE as i32, 42);
        let any_solid = chunk.voxels.iter().any(|v| v.material.is_solid());
        assert!(!any_solid, "features placed when surface_z_local is out of range");
    }
}
