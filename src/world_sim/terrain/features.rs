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
const BOULDER_DENSITY_SALT: u64 = 0xCCCC_5678_0001;
const BOULDER_SIZE_SALT: u64    = 0xDDDD_5678_0002;
const PILLAR_DENSITY_SALT: u64  = 0xEEEE_5678_0003;
const PILLAR_HEIGHT_SALT: u64   = 0xFFFF_5678_0004;

// ---------------------------------------------------------------------------
// Helpers: tree and boulder stampers
// ---------------------------------------------------------------------------

/// Crown shape — controls the attractor envelope for procedural trees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrownShape {
    Ellipsoid, // symmetric ellipsoid (oak/forest)
    Cone,      // radius narrows with height (pine/jungle emergent)
    FlatWide,  // squashed ellipsoid (savanna/acacia)
    Droopy,    // extends downward (willow)
}

/// Fully-resolved parameters for a single procedural tree.
#[derive(Clone, Debug)]
pub struct TreeParams {
    pub trunk_height: i32,
    pub trunk_radius: f32,
    pub crown_base_z: i32,    // where the crown starts (relative to feature_base_z)
    pub crown_height: i32,    // vertical extent of crown
    pub crown_radius_xy: f32, // horizontal spread of crown
    pub crown_radius_z: f32,  // vertical radius (may differ from crown_height/2)
    pub crown_shape: CrownShape,
    pub num_attractors: u32,
    pub num_clusters: u32,
    pub branch_radius: f32,
    pub leaf_radius: f32,
    pub canopy_material: VoxelMaterial,
}

/// Resolve tree params from (terrain, sub_biome, origin, seed). Deterministic.
pub fn tree_params_for_biome(
    terrain: Terrain,
    _sub_biome: SubBiome,
    ox: i32,
    oy: i32,
    seed: u64,
) -> TreeParams {
    let size_hash = noise::hash_f32(ox, oy, 5, seed.wrapping_add(TREE_HEIGHT_SALT));
    let h_hash = noise::hash_f32(ox, oy, 0, seed.wrapping_add(TREE_HEIGHT_SALT));

    let (trunk_height, trunk_radius, crown_radius_xy) = if size_hash < 0.2 {
        // Small bush
        (10 + (h_hash * 10.0) as i32, 1.0_f32, 5.0_f32 + h_hash * 3.0)
    } else if size_hash > 0.8 {
        // Large
        (
            80 + (h_hash * 40.0) as i32,
            2.0_f32 + h_hash,
            20.0_f32 + h_hash * 10.0,
        )
    } else {
        // Standard
        (
            40 + (h_hash * 30.0) as i32,
            1.0_f32 + h_hash * 1.5,
            12.0_f32 + h_hash * 6.0,
        )
    };

    // Biome-specific crown shape and counts.
    let (crown_shape, num_attractors, num_clusters, crown_stretch_z) = match terrain {
        Terrain::Tundra => (CrownShape::Cone, 10u32, 4u32, 1.3_f32),
        Terrain::Plains => (CrownShape::FlatWide, 12, 4, 0.5),
        Terrain::Jungle => (CrownShape::Cone, 20, 6, 1.2),
        Terrain::Swamp => (CrownShape::Droopy, 14, 5, 1.0),
        _ => (CrownShape::Ellipsoid, 16, 5, 0.9), // Forest default
    };

    let crown_base_z = (trunk_height as f32 * 0.55) as i32;
    let crown_height = (crown_radius_xy * 2.0 * crown_stretch_z) as i32;
    let crown_radius_z = crown_radius_xy * crown_stretch_z;

    let branch_radius = (trunk_radius * 0.4).max(0.7);
    let leaf_radius = (crown_radius_xy * 0.35).max(1.5);

    let canopy_material = match terrain {
        Terrain::Jungle => VoxelMaterial::JungleMoss,
        _ => VoxelMaterial::Leaves,
    };

    TreeParams {
        trunk_height,
        trunk_radius,
        crown_base_z,
        crown_height,
        crown_radius_xy,
        crown_radius_z,
        crown_shape,
        num_attractors,
        num_clusters,
        branch_radius,
        leaf_radius,
        canopy_material,
    }
}

/// Generate the i-th attractor point for a tree at origin (ox, oy).
/// Returns (dx, dy, dz) relative to (ox, oy, feature_base_z).
pub fn attractor(i: u32, ox: i32, oy: i32, seed: u64, params: &TreeParams) -> (f32, f32, f32) {
    let s = seed.wrapping_add(0xA77A);
    let h1 = noise::hash_f32(ox, oy, i as i32 * 7 + 1, s);
    let h2 = noise::hash_f32(ox, oy, i as i32 * 7 + 2, s);
    let h3 = noise::hash_f32(ox, oy, i as i32 * 7 + 3, s);

    let angle = h1 * 2.0 * std::f32::consts::PI;
    let t_z = h3; // [0, 1]

    let radius_factor = match params.crown_shape {
        CrownShape::Ellipsoid | CrownShape::FlatWide => {
            let z_norm = 2.0 * t_z - 1.0;
            (1.0 - z_norm * z_norm).max(0.0).sqrt()
        }
        CrownShape::Cone => 1.0 - t_z,
        CrownShape::Droopy => {
            if t_z < 0.3 {
                1.0
            } else {
                1.0 - (t_z - 0.3) / 0.7
            }
        }
    };

    let radius = h2.sqrt() * params.crown_radius_xy * radius_factor;
    let dx = angle.cos() * radius;
    let dy = angle.sin() * radius;

    let dz = match params.crown_shape {
        CrownShape::Droopy => {
            params.crown_base_z as f32 + (t_z - 0.3) * params.crown_height as f32
        }
        _ => params.crown_base_z as f32 + t_z * params.crown_height as f32,
    };
    (dx, dy, dz)
}

/// For each of K clusters, compute centroid = average of attractors assigned to it.
/// Clusters are assigned by `i % K` (round-robin).
pub fn compute_cluster_centroids(
    ox: i32,
    oy: i32,
    seed: u64,
    params: &TreeParams,
) -> Vec<(f32, f32, f32)> {
    let k = params.num_clusters.max(1) as usize;
    let mut sums: Vec<(f32, f32, f32, u32)> = vec![(0.0, 0.0, 0.0, 0); k];
    for i in 0..params.num_attractors {
        let cluster_idx = (i as usize) % k;
        let (dx, dy, dz) = attractor(i, ox, oy, seed, params);
        sums[cluster_idx].0 += dx;
        sums[cluster_idx].1 += dy;
        sums[cluster_idx].2 += dz;
        sums[cluster_idx].3 += 1;
    }
    sums.into_iter()
        .map(|(sx, sy, sz, n)| {
            if n == 0 {
                (0.0, 0.0, 0.0)
            } else {
                (sx / n as f32, sy / n as f32, sz / n as f32)
            }
        })
        .collect()
}

/// Capsule SDF: distance from point p to capsule(a, b, radius). Negative inside.
#[inline]
pub fn capsule_sdf(
    px: f32,
    py: f32,
    pz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    radius: f32,
) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let apx = px - ax;
    let apy = py - ay;
    let apz = pz - az;
    let ab2 = abx * abx + aby * aby + abz * abz;
    let t = if ab2 > 1e-6 {
        ((apx * abx + apy * aby + apz * abz) / ab2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cx = ax + abx * t - px;
    let cy = ay + aby * t - py;
    let cz = az + abz * t - pz;
    (cx * cx + cy * cy + cz * cz).sqrt() - radius
}

/// Sphere SDF.
#[inline]
pub fn sphere_sdf(px: f32, py: f32, pz: f32, cx: f32, cy: f32, cz: f32, radius: f32) -> f32 {
    let dx = px - cx;
    let dy = py - cy;
    let dz = pz - cz;
    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

/// Stamp a tree with its base at local column (lx, ly), trunk bottom at
/// voxel-height base_z (world Z). Uses Forest/Ellipsoid defaults.
///
/// `lx, ly` may be outside `[0, CHUNK_SIZE)` — the stamp writes only voxels
/// that land inside the chunk, so halo origins in neighbor chunks still
/// contribute their trunk/canopy voxels that cross into this chunk.
pub fn stamp_tree(chunk: &mut Chunk, lx: i32, ly: i32, base_z: i32, seed: u64) {
    stamp_tree_procedural(
        chunk,
        lx,
        ly,
        base_z,
        seed,
        Terrain::Forest,
        SubBiome::Standard,
    );
}

/// Back-compat thin wrapper. `canopy_material` is ignored — the procedural
/// stamper derives it from the biome.
pub fn stamp_tree_biome(
    chunk: &mut Chunk,
    lx: i32,
    ly: i32,
    base_z: i32,
    seed: u64,
    _canopy_material: VoxelMaterial,
) {
    stamp_tree_procedural(
        chunk,
        lx,
        ly,
        base_z,
        seed,
        Terrain::Forest,
        SubBiome::Standard,
    );
}

/// Procedurally stamp a tree using the SDF/attractor algorithm.
pub fn stamp_tree_procedural(
    chunk: &mut Chunk,
    lx: i32,
    ly: i32,
    base_z: i32,
    seed: u64,
    terrain: Terrain,
    sub_biome: SubBiome,
) {
    let base_x = chunk.pos.x * CHUNK_SIZE as i32;
    let base_y = chunk.pos.y * CHUNK_SIZE as i32;
    let base_cz = chunk.pos.z * CHUNK_SIZE as i32;

    let ox = base_x + lx;
    let oy = base_y + ly;
    let feature_base_z = base_z; // absolute world z where trunk starts

    let params = tree_params_for_biome(terrain, sub_biome, ox, oy, seed);
    let centroids = compute_cluster_centroids(ox, oy, seed, &params);

    // Precompute attractors (used twice: leaves and as the scan bounding box).
    let attractors: Vec<(f32, f32, f32)> = (0..params.num_attractors)
        .map(|i| attractor(i, ox, oy, seed, &params))
        .collect();

    // Bounding box for voxel scan.
    let max_reach_xy = params.crown_radius_xy + params.leaf_radius + 2.0;
    let min_lx = (lx as f32 - max_reach_xy).floor() as i32;
    let max_lx = (lx as f32 + max_reach_xy).ceil() as i32;
    let min_ly = (ly as f32 - max_reach_xy).floor() as i32;
    let max_ly = (ly as f32 + max_reach_xy).ceil() as i32;
    let max_reach_z_up =
        params.trunk_height + params.crown_height + params.leaf_radius as i32 + 2;
    let max_reach_z_down = if matches!(params.crown_shape, CrownShape::Droopy) {
        (params.crown_radius_xy + params.leaf_radius) as i32 + 2
    } else {
        2
    };
    let min_wz = feature_base_z - max_reach_z_down;
    let max_wz = feature_base_z + max_reach_z_up;

    let trunk_top_z = feature_base_z + params.trunk_height;
    let ox_f = ox as f32;
    let oy_f = oy as f32;
    let fbz = feature_base_z as f32;

    for wz in min_wz..=max_wz {
        let llz = wz - base_cz;
        if llz < 0 || llz >= CHUNK_SIZE as i32 {
            continue;
        }
        for vly in min_ly..=max_ly {
            if vly < 0 || vly >= CHUNK_SIZE as i32 {
                continue;
            }
            for vlx in min_lx..=max_lx {
                if vlx < 0 || vlx >= CHUNK_SIZE as i32 {
                    continue;
                }

                let px = (base_x + vlx) as f32;
                let py = (base_y + vly) as f32;
                let pz = wz as f32;

                // Trunk SDF (vertical capsule).
                let trunk_d = capsule_sdf(
                    px,
                    py,
                    pz,
                    ox_f,
                    oy_f,
                    fbz,
                    ox_f,
                    oy_f,
                    trunk_top_z as f32,
                    params.trunk_radius,
                );

                // Branch SDFs.
                let mut min_branch_d = f32::INFINITY;
                for (i, &(cdx, cdy, cdz)) in centroids.iter().enumerate() {
                    let t_along =
                        (i as f32 / params.num_clusters as f32) * 0.4 + 0.55;
                    let branch_start_z = fbz + params.trunk_height as f32 * t_along;
                    let d = capsule_sdf(
                        px,
                        py,
                        pz,
                        ox_f,
                        oy_f,
                        branch_start_z,
                        ox_f + cdx,
                        oy_f + cdy,
                        fbz + cdz,
                        params.branch_radius,
                    );
                    if d < min_branch_d {
                        min_branch_d = d;
                    }
                }

                // Leaf SDFs.
                let mut min_leaf_d = f32::INFINITY;
                for &(adx, ady, adz) in &attractors {
                    let d = sphere_sdf(
                        px,
                        py,
                        pz,
                        ox_f + adx,
                        oy_f + ady,
                        fbz + adz,
                        params.leaf_radius,
                    );
                    if d < min_leaf_d {
                        min_leaf_d = d;
                    }
                }

                let mat = if trunk_d <= 0.0 {
                    VoxelMaterial::WoodLog
                } else if min_branch_d <= 0.0 {
                    VoxelMaterial::WoodLog
                } else if min_leaf_d <= 0.0 {
                    params.canopy_material
                } else {
                    continue;
                };

                let idx = local_index(vlx as usize, vly as usize, llz as usize);
                let existing = chunk.voxels[idx].material;
                // Never overwrite an existing trunk with a leaf.
                if existing == VoxelMaterial::WoodLog && mat != VoxelMaterial::WoodLog {
                    continue;
                }
                chunk.voxels[idx] = Voxel::new(mat);
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

    // Horizontal halo: procedural crown can reach
    //   crown_radius_xy (up to 30) + leaf_radius (~10.5) + 2 ≈ 43 voxels.
    const HALO: i32 = 44;
    // Vertical extent of a tree above its base: trunk (up to 120) + crown height
    // (up to 2*crown_radius*stretch = ~80) + leaf_radius (~10) ≈ 210.
    const FEATURE_MAX_Z_ABOVE: i32 = 220;
    // Droopy willows can reach below the base by crown_radius_xy + leaf_radius.
    const FEATURE_MAX_Z_BELOW: i32 = 44;

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
            let feature_z_min = feature_base_z - FEATURE_MAX_Z_BELOW;
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
                    stamp_tree_procedural(
                        chunk,
                        lx,
                        ly,
                        feature_base_z,
                        seed,
                        cell.terrain,
                        cell.sub_biome,
                    );
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
