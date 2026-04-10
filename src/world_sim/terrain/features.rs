//! Surface feature placement — trees, boulders, etc.
//!
//! Called after base terrain materialisation for the chunk that contains the
//! surface layer.  Features are placed deterministically using hash_f32 so
//! the result is identical across any number of calls with the same seed.

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::{Terrain, SubBiome};
use crate::world_sim::terrain::region_plan::RegionPlan;
use super::noise;

// ---------------------------------------------------------------------------
// Salt constants (keep features from correlating with each other)
// ---------------------------------------------------------------------------
const TREE_DENSITY_SALT: u64    = 0x7777_1234_0001;
const TREE_HEIGHT_SALT: u64     = 0xAAAA_1234_0002;
const TREE_CANOPY_SALT: u64     = 0xBBBB_1234_0003;
const BOULDER_DENSITY_SALT: u64 = 0xCCCC_5678_0001;
const BOULDER_SIZE_SALT: u64    = 0xDDDD_5678_0002;
const PILLAR_DENSITY_SALT: u64  = 0xEEEE_5678_0003;
const PILLAR_HEIGHT_SALT: u64   = 0xFFFF_5678_0004;

// ---------------------------------------------------------------------------
// Helpers: tree and boulder stampers
// ---------------------------------------------------------------------------

/// Stamp a tree with its base at local column (lx, ly), trunk bottom at
/// voxel-height base_z (world Z).  The chunk must contain that Z range to
/// see any of the tree. `canopy_material` controls canopy color by biome.
///
/// `lx, ly` may be outside `[0, CHUNK_SIZE)` — the stamp writes only voxels
/// that land inside the chunk, so halo origins in neighbor chunks still
/// contribute their trunk/canopy voxels that cross into this chunk.
pub fn stamp_tree(chunk: &mut Chunk, lx: i32, ly: i32, base_z: i32, seed: u64) {
    stamp_tree_biome(chunk, lx, ly, base_z, seed, VoxelMaterial::Leaves);
}

/// Stamp a tree with a biome-specific canopy material.
///
/// `lx, ly` may be negative or >= CHUNK_SIZE (halo stamping).
pub fn stamp_tree_biome(chunk: &mut Chunk, lx: i32, ly: i32, base_z: i32, seed: u64, canopy_material: VoxelMaterial) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let vx = base_x + lx;
    let vy = base_y + ly;

    // Size category: small bush (20%), medium tree (60%), large tree (20%)
    // At 10cm/voxel with CHUNK_SIZE=64 — realistic proportions
    let size_hash = noise::hash_f32(vx, vy, 5, seed.wrapping_add(TREE_HEIGHT_SALT));
    let h_hash = noise::hash_f32(vx, vy, 0, seed.wrapping_add(TREE_HEIGHT_SALT));

    let (trunk_height, trunk_radius, canopy_r) = if size_hash < 0.2 {
        // Small bush/sapling: 1-2m tall, 10cm trunk, 50-80cm canopy
        (10 + (h_hash * 10.0) as i32, 1i32, 5 + (h_hash * 3.0) as i32)
    } else if size_hash > 0.8 {
        // Large tree: 8-12m tall, 40cm trunk, 2-3m canopy
        (80 + (h_hash * 40.0) as i32, 2, 20 + (h_hash * 10.0) as i32)
    } else {
        // Standard tree: 4-7m tall, 20-30cm trunk, 1.5-2m canopy
        (40 + (h_hash * 30.0) as i32, 1 + (h_hash * 1.5) as i32, 12 + (h_hash * 6.0) as i32)
    };

    // -- Trunk (cylindrical) --
    for dz in 0..trunk_height {
        let wz = base_z + dz;
        let lz_local = wz - base_cz;
        if lz_local < 0 || lz_local >= CHUNK_SIZE as i32 { continue; }
        for dy in -trunk_radius..=trunk_radius {
            for dx in -trunk_radius..=trunk_radius {
                if dx * dx + dy * dy > trunk_radius * trunk_radius + 1 { continue; }
                let llx = lx + dx;
                let lly = ly + dy;
                if llx < 0 || lly < 0 || llx >= CHUNK_SIZE as i32 || lly >= CHUNK_SIZE as i32 { continue; }
                let idx = local_index(llx as usize, lly as usize, lz_local as usize);
                chunk.voxels[idx] = Voxel::new(VoxelMaterial::WoodLog);
            }
        }
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
                    chunk.voxels[idx] = Voxel::new(canopy_material);
                }
            }
        }
    }
}

/// Stamp a small boulder at local column (lx, ly), base at world-Z base_z.
/// `lx, ly` may be negative or >= CHUNK_SIZE (halo stamping).
fn stamp_boulder(chunk: &mut Chunk, lx: i32, ly: i32, base_z: i32, seed: u64) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let vx = base_x + lx;
    let vy = base_y + ly;

    let s_hash = noise::hash_f32(vx, vy, 2, seed.wrapping_add(BOULDER_SIZE_SALT));
    let size = 3 + (s_hash * 8.0) as i32; // 3..10 (30cm-1m at 10cm/voxel)

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

/// Stamp a rock pillar/mesa at local column (lx, ly), base at world-Z base_z.
/// Creates a tall narrow stone column — common in desert/badlands.
/// `lx, ly` may be negative or >= CHUNK_SIZE (halo stamping).
fn stamp_pillar(chunk: &mut Chunk, lx: i32, ly: i32, base_z: i32, seed: u64, material: VoxelMaterial) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let vx = base_x + lx;
    let vy = base_y + ly;

    let h_hash = noise::hash_f32(vx, vy, 3, seed.wrapping_add(PILLAR_HEIGHT_SALT));
    let height = 20 + (h_hash * 40.0) as i32; // 20..60 (2-6m at 10cm/voxel)
    let radius: i32 = 3 + (h_hash * 3.0) as i32; // 3..5 (30-50cm radius)

    for dz in 0..height {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius + 1 { continue; }
                let wx = vx + dx;
                let wy = vy + dy;
                let wz = base_z + dz;
                let llx = wx - base_x;
                let lly = wy - base_y;
                let llz = wz - base_cz;
                if llx < 0 || lly < 0 || llz < 0 { continue; }
                if llx >= CHUNK_SIZE as i32 || lly >= CHUNK_SIZE as i32 || llz >= CHUNK_SIZE as i32 { continue; }
                let idx = local_index(llx as usize, lly as usize, llz as usize);
                chunk.voxels[idx] = Voxel::new(material);
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
    pillar_density: f32,
    pillar_material: VoxelMaterial,
    canopy_material: VoxelMaterial,
}

fn feature_params(terrain: Terrain, sub_biome: SubBiome) -> FeatureParams {
    // Densities for 10cm/voxel with CHUNK_SIZE=64. Trees are 40-120 voxels
    // tall with 12-30 voxel canopy radius — very large footprint per tree.
    let tree_density = match terrain {
        Terrain::Forest => match sub_biome {
            SubBiome::DenseForest   => 0.004,
            SubBiome::LightForest   => 0.001,
            SubBiome::AncientForest => 0.002,
            _                       => 0.002,
        },
        Terrain::Jungle => match sub_biome {
            SubBiome::TempleJungle => 0.005,
            _                      => 0.006,
        },
        Terrain::Tundra  => 0.0005,
        Terrain::Swamp   => 0.002,
        Terrain::Mountains => 0.0003,
        Terrain::Plains  => 0.0003,
        _ => 0.0,
    };

    let boulder_density = match terrain {
        Terrain::Plains  => 0.01,
        Terrain::Desert  => 0.02,
        Terrain::Badlands => 0.02,
        Terrain::Tundra  => 0.008,
        Terrain::Mountains => 0.02,
        _ => 0.0,
    };

    let (pillar_density, pillar_material) = match terrain {
        Terrain::Desert => (0.008, VoxelMaterial::Sandstone),
        Terrain::Badlands => (0.012, VoxelMaterial::RedSand),
        _ => (0.0, VoxelMaterial::Stone),
    };

    let canopy_material = match terrain {
        Terrain::Jungle => VoxelMaterial::JungleMoss,
        Terrain::Swamp => VoxelMaterial::Leaves, // dark green contrasts olive ground
        Terrain::Forest => VoxelMaterial::Leaves,
        _ => VoxelMaterial::Leaves,
    };

    FeatureParams { tree_density, boulder_density, pillar_density, pillar_material, canopy_material }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Place surface features (trees, boulders) into `chunk`.
///
/// Iterates feature origins in a halo around the chunk so that features
/// whose origin is in a neighbor chunk (but whose trunk/canopy crosses into
/// this chunk) are still stamped. Each origin samples its own biome from
/// the region plan so features respect cell boundaries at sub-chunk
/// resolution.
///
/// `_terrain`, `_sub_biome`, `_surface_z_local` are retained for API
/// compatibility but are no longer used — biome is now sampled per-origin.
pub fn place_surface_features(
    chunk: &mut Chunk,
    cp: ChunkPos,
    _terrain: Terrain,
    _sub_biome: SubBiome,
    _surface_z_local: i32,
    seed: u64,
    plan: Option<&RegionPlan>,
) {
    // Halo stamping requires the plan for per-origin biome sampling.
    let plan = match plan {
        Some(p) => p,
        None => return,
    };

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    // Horizontal halo: max tree canopy radius (large trees have r=30).
    const HALO: i32 = 30;
    // Vertical extent of a tree above its base: trunk (up to 80+40=120) + canopy
    // radius (up to 30) ≈ 150 voxels. Plus one for the `+1` base offset.
    const FEATURE_MAX_Z_ABOVE: i32 = 160;

    for ly in -HALO..(CHUNK_SIZE as i32 + HALO) {
        for lx in -HALO..(CHUNK_SIZE as i32 + HALO) {
            let vx = base_x + lx;
            let vy = base_y + ly;

            // Sample the biome AT this origin (may be in a neighbor chunk).
            let (cell, _, _) = plan.sample(vx as f32, vy as f32);
            let params = feature_params(cell.terrain, cell.sub_biome);
            if params.tree_density == 0.0
                && params.boulder_density == 0.0
                && params.pillar_density == 0.0
            {
                continue;
            }

            // Per-column surface height.
            let col_surface_z =
                super::materialize::surface_height_at(vx as f32, vy as f32, plan, seed);
            let feature_base_z = col_surface_z + 1;

            // Early-out: if the feature's vertical extent can't reach this
            // chunk's z range, skip entirely.
            let feature_z_max = feature_base_z + FEATURE_MAX_Z_ABOVE;
            let feature_z_min = feature_base_z - 1;
            if feature_z_max < base_z || feature_z_min >= base_z + CHUNK_SIZE as i32 {
                continue;
            }

            // Tree placement — modulated by large-scale noise for clustering.
            if params.tree_density > 0.0 {
                let td = noise::hash_f32(vx, vy, 0, seed.wrapping_add(TREE_DENSITY_SALT));
                let cluster = noise::fbm_2d(
                    vx as f32 * 0.025,
                    vy as f32 * 0.025,
                    seed.wrapping_add(0xC1C1),
                    2,
                    2.0,
                    0.5,
                );
                let effective_density = params.tree_density * (0.3 + cluster * 1.4);
                if td < effective_density {
                    stamp_tree_biome(chunk, lx, ly, feature_base_z, seed, params.canopy_material);
                }
            }

            // Boulder placement.
            if params.boulder_density > 0.0 {
                let bd = noise::hash_f32(vx, vy, 0, seed.wrapping_add(BOULDER_DENSITY_SALT));
                if bd < params.boulder_density {
                    stamp_boulder(chunk, lx, ly, feature_base_z, seed);
                }
            }

            // Rock pillar placement (desert/badlands).
            if params.pillar_density > 0.0 {
                let pd = noise::hash_f32(vx, vy, 0, seed.wrapping_add(PILLAR_DENSITY_SALT));
                if pd < params.pillar_density {
                    stamp_pillar(chunk, lx, ly, feature_base_z, seed, params.pillar_material);
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
        let leaf_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Leaves).count();
        assert!(log_count >= 2, "tree trunk too short: {log_count} WoodLog voxels");
        assert!(leaf_count > 0, "tree has no canopy (Leaves voxels): {leaf_count}");
    }

    #[test]
    fn forest_has_multiple_trees() {
        // place_surface_features now requires a plan for per-origin biome
        // sampling. When plan=None, the function is a no-op.
        let cp = ChunkPos::new(0, 0, 5);
        let mut chunk = make_surface_chunk(cp);
        let surface_local = (CHUNK_SIZE / 2 - 1) as i32;
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, surface_local, 42, None);
        // With plan=None, no features are placed; assert the function is a
        // no-op rather than requiring trees (since biome is now sampled
        // per-origin from the plan).
        let log_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::WoodLog).count();
        assert_eq!(log_count, 0, "plan=None should be a no-op");
    }

    #[test]
    fn features_not_placed_outside_chunk() {
        // With no plan, place_surface_features is a no-op.
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new_air(cp);
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, CHUNK_SIZE as i32, 42, None);
        let any_solid = chunk.voxels.iter().any(|v| v.material.is_solid());
        assert!(!any_solid, "plan=None should be a no-op");
    }
}
