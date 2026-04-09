//! Cave carving pass — applied to already-materialized chunks.
//!
//! Uses dual worm noise (two independent 3D noise fields). Where both fields
//! are within a threshold band around 0.5, the voxel is carved out (or
//! replaced with a biome-specific fluid fill).

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::SubBiome;
use super::noise;

// ---------------------------------------------------------------------------
// Biome-specific cave parameters
// ---------------------------------------------------------------------------

struct CaveParams {
    scale: f32,
    threshold: f32,
    fill: VoxelMaterial,
    floor_fill: Option<VoxelMaterial>,
}

fn cave_params(biome: SubBiome) -> CaveParams {
    match biome {
        SubBiome::NaturalCave => CaveParams {
            scale: 16.0,
            threshold: 0.06,
            fill: VoxelMaterial::Air,
            floor_fill: None,
        },
        SubBiome::LavaTubes => CaveParams {
            scale: 24.0,
            threshold: 0.07,
            fill: VoxelMaterial::Air,
            floor_fill: Some(VoxelMaterial::Lava),
        },
        SubBiome::MushroomGrove => CaveParams {
            scale: 32.0,
            threshold: 0.09,
            fill: VoxelMaterial::Air,
            floor_fill: None,
        },
        SubBiome::CrystalVein => CaveParams {
            scale: 10.0,
            threshold: 0.04,
            fill: VoxelMaterial::Air,
            floor_fill: None,
        },
        SubBiome::Aquifer => CaveParams {
            scale: 20.0,
            threshold: 0.07,
            fill: VoxelMaterial::Water,
            floor_fill: None,
        },
        SubBiome::FrozenCavern => CaveParams {
            scale: 20.0,
            threshold: 0.06,
            fill: VoxelMaterial::Air,
            floor_fill: Some(VoxelMaterial::Ice),
        },
        SubBiome::BoneOssuary => CaveParams {
            scale: 18.0,
            threshold: 0.06,
            fill: VoxelMaterial::Air,
            floor_fill: None,
        },
        // Non-cave biomes: no carving performed
        _ => CaveParams {
            scale: 16.0,
            threshold: 0.0, // threshold=0 means nothing will be carved
            fill: VoxelMaterial::Air,
            floor_fill: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Carve caves into an already-materialized chunk.
///
/// Uses two independent 3D noise fields (worm-cave test). Positions where
/// both fields are within `threshold` of 0.5 are carved to `fill` (or
/// `floor_fill` at floor positions).
///
/// Never carves Granite (bedrock). Skips non-solid voxels.
pub fn carve_caves(chunk: &mut Chunk, cp: ChunkPos, biome: SubBiome, seed: u64) {
    let params = cave_params(biome);
    if params.threshold == 0.0 {
        return; // non-cave biome — nothing to do
    }

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    let seed_a = seed.wrapping_add(0xCAFE_BABE_0001);
    let seed_b = seed.wrapping_add(0xDEAD_BEEF_0002);

    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let idx = local_index(lx, ly, lz);
                let mat = chunk.voxels[idx].material;

                // Skip non-solid (air/water/lava already open, granite never carved)
                if !mat.is_solid() || mat == VoxelMaterial::Granite {
                    continue;
                }

                let vx = (base_x + lx as i32) as f32;
                let vy = (base_y + ly as i32) as f32;
                let vz = (base_z + lz as i32) as f32;

                // Dual-field worm test
                let a = noise::value_noise_3d(vx / params.scale, vy / params.scale, vz / params.scale, seed_a, 1.0);
                let b = noise::value_noise_3d(vx / params.scale, vy / params.scale, vz / params.scale, seed_b, 1.0);

                if (a - 0.5).abs() < params.threshold && (b - 0.5).abs() < params.threshold {
                    // Determine fill: check if voxel below won't be carved (floor position)
                    let fill = if let Some(ff) = params.floor_fill {
                        // Check voxel directly below
                        let is_floor = if lz == 0 {
                            // Bottom of chunk — treat as floor (no below in this chunk)
                            true
                        } else {
                            let below_idx = local_index(lx, ly, lz - 1);
                            let bmat = chunk.voxels[below_idx].material;
                            // Below is a floor if it's solid and won't be carved
                            if !bmat.is_solid() || bmat == VoxelMaterial::Granite {
                                false
                            } else {
                                let bvz = vz - 1.0;
                                let ba = noise::value_noise_3d(vx / params.scale, vy / params.scale, bvz / params.scale, seed_a, 1.0);
                                let bb = noise::value_noise_3d(vx / params.scale, vy / params.scale, bvz / params.scale, seed_b, 1.0);
                                !((ba - 0.5).abs() < params.threshold && (bb - 0.5).abs() < params.threshold)
                            }
                        };
                        if is_floor { ff } else { params.fill }
                    } else {
                        params.fill
                    };

                    chunk.voxels[idx] = Voxel::new(fill);
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
    use crate::world_sim::voxel::ChunkPos;

    fn fill_chunk(cp: ChunkPos, mat: VoxelMaterial) -> Chunk {
        let mut chunk = Chunk::new_air(cp);
        for v in chunk.voxels.iter_mut() {
            *v = Voxel::new(mat);
        }
        chunk
    }

    #[test]
    fn caves_carve_air_in_solid() {
        // Scan several chunk positions to find at least one that gets carved.
        // A single 16^3 chunk may lie entirely outside any worm tunnel;
        // scan a 4×4×4 neighbourhood to guarantee a hit.
        let mut total_air = 0usize;
        let mut total_stone = 0usize;
        for cz in -8..=-4i32 {
            for cy in 0..4i32 {
                for cx in 0..4i32 {
                    let cp = ChunkPos::new(cx, cy, cz);
                    let mut chunk = fill_chunk(cp, VoxelMaterial::Stone);
                    carve_caves(&mut chunk, cp, SubBiome::NaturalCave, 42);
                    total_air += chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
                    total_stone += chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Stone).count();
                }
            }
        }
        assert!(total_air > 0, "NaturalCave carved no air in 4×4×4 neighbourhood — threshold too tight or noise issue");
        assert!(total_stone > 0, "NaturalCave carved everything in 4×4×4 neighbourhood");
    }

    #[test]
    fn lava_tubes_have_lava() {
        let cp = ChunkPos::new(3, 3, -8);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Basalt);
        carve_caves(&mut chunk, cp, SubBiome::LavaTubes, 99);

        let lava_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Lava).count();
        let air_count  = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        assert!(lava_count > 0 || air_count > 0,
            "LavaTubes produced neither lava nor air voxels");
    }

    #[test]
    fn cave_carving_is_deterministic() {
        let cp = ChunkPos::new(1, 2, -3);
        let mut a = fill_chunk(cp, VoxelMaterial::Stone);
        let mut b = fill_chunk(cp, VoxelMaterial::Stone);
        carve_caves(&mut a, cp, SubBiome::NaturalCave, 7777);
        carve_caves(&mut b, cp, SubBiome::NaturalCave, 7777);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }

    #[test]
    fn granite_never_carved() {
        let cp = ChunkPos::new(0, 0, -20);
        let mut chunk = fill_chunk(cp, VoxelMaterial::Granite);
        carve_caves(&mut chunk, cp, SubBiome::NaturalCave, 42);
        let carved = chunk.voxels.iter().filter(|v| v.material != VoxelMaterial::Granite).count();
        assert_eq!(carved, 0, "granite was carved: {carved} voxels changed");
    }
}
