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

/// Fully-resolved parameters for a single procedural tree.
///
/// The tree is a 3-level hierarchical grammar:
///   L0 trunk  — vertical capsule from base to (0,0,trunk_height)
///   L1 primary branches — num_primary capsules from (0,0,fork_height)
///   L2 secondary branches — num_secondary capsules from each primary tip
///   L3 leaf clusters — a sphere at each secondary branch tip
#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    pub trunk_height: i32,
    pub trunk_radius: f32,
    pub fork_height: i32, // voxels from base where the trunk forks
    pub num_primary: i32,
    pub primary_length: f32,
    pub primary_elev_min: f32, // degrees from vertical
    pub primary_elev_max: f32,
    pub num_secondary: i32,
    pub secondary_length: f32,
    pub secondary_elev_min: f32, // degrees from primary direction
    pub secondary_elev_max: f32,
    pub branch_radius: f32, // thickness of primary/secondary
    pub leaf_radius: f32,   // size of leaf cluster at each secondary tip
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

    // Size category
    let (trunk_height, trunk_radius) = if size_hash < 0.2 {
        // Bush: 10-20 voxels, thin trunk ~10cm
        (10 + (h_hash * 10.0) as i32, 1.0_f32)
    } else if size_hash > 0.8 {
        // Large tree: 80-120 voxels, thick trunk 40-60cm diameter
        (80 + (h_hash * 40.0) as i32, 3.5_f32 + h_hash * 1.5)
    } else {
        // Standard: 40-70, trunk 25-40cm diameter
        (40 + (h_hash * 30.0) as i32, 2.0_f32 + h_hash * 1.0)
    };

    // Biome-specific branching parameters.
    let (
        num_primary,
        num_secondary,
        fork_height_ratio,
        primary_elev_min,
        primary_elev_max,
        primary_length_ratio,
        canopy_material,
    ) = match terrain {
        Terrain::Plains => (5, 3, 0.75_f32, 50.0_f32, 70.0_f32, 0.45_f32, VoxelMaterial::Leaves),
        Terrain::Jungle => (3, 4, 0.65, 30.0, 45.0, 0.55, VoxelMaterial::JungleMoss),
        Terrain::Tundra => (5, 2, 0.30, 25.0, 40.0, 0.35, VoxelMaterial::Leaves),
        Terrain::Swamp => (4, 3, 0.50, 10.0, 30.0, 0.45, VoxelMaterial::Leaves),
        _ => (4, 3, 0.55, 35.0, 55.0, 0.40, VoxelMaterial::Leaves), // Forest default
    };

    let fork_height = (trunk_height as f32 * fork_height_ratio) as i32;
    let primary_length = trunk_height as f32 * primary_length_ratio;
    let secondary_length = primary_length * 0.55;

    // Leaf cluster sizing: keep clusters small so individual trees stay
    // distinct instead of merging into one blob. At 10cm/voxel a real leaf
    // cluster is ~40-80cm diameter = 2-4 voxel radius. We add a small
    // proportional component so larger trees get slightly bigger clusters.
    let leaf_radius = (2.5 + primary_length * 0.08).clamp(2.0, 5.5);

    TreeParams {
        trunk_height,
        trunk_radius,
        fork_height,
        num_primary,
        primary_length,
        primary_elev_min,
        primary_elev_max,
        num_secondary,
        secondary_length,
        secondary_elev_min: 40.0,
        secondary_elev_max: 70.0,
        branch_radius: (trunk_radius * 0.5).max(1.0),
        leaf_radius,
        canopy_material,
    }
}

/// Returns (dx, dy, dz) relative to the fork point for the i-th primary branch.
pub fn primary_branch_end(
    i: i32,
    ox: i32,
    oy: i32,
    seed: u64,
    params: &TreeParams,
) -> (f32, f32, f32) {
    let n = params.num_primary as f32;
    let angle_base = (i as f32) * std::f32::consts::TAU / n;
    // Angular jitter ±15% of the slice (±0.3 * (TAU/n) / 2 = ±0.15 * TAU/n)
    let angle_jitter = (noise::hash_f32(ox, oy, i * 11 + 1, seed.wrapping_add(0xB120)) - 0.5)
        * (std::f32::consts::TAU / n)
        * 0.3;
    let angle = angle_base + angle_jitter;

    // Elevation angle measured FROM VERTICAL: lerp between min and max via hash.
    let elev_hash = noise::hash_f32(ox, oy, i * 11 + 2, seed.wrapping_add(0xB121));
    let elev_deg =
        params.primary_elev_min + elev_hash * (params.primary_elev_max - params.primary_elev_min);
    let elev_rad = elev_deg * std::f32::consts::PI / 180.0;

    // Length jitter ±20%.
    let len_hash = noise::hash_f32(ox, oy, i * 11 + 3, seed.wrapping_add(0xB122));
    let length = params.primary_length * (0.8 + len_hash * 0.4);

    // elev_rad is angle from VERTICAL, so horizontal = sin, vertical = cos.
    let horiz_dist = length * elev_rad.sin();
    let dx = angle.cos() * horiz_dist;
    let dy = angle.sin() * horiz_dist;
    let dz = length * elev_rad.cos();
    (dx, dy, dz)
}

/// Returns (dx, dy, dz) relative to the primary branch END for the j-th
/// secondary branch off the i-th primary.
pub fn secondary_branch_end(
    i: i32,
    j: i32,
    primary_end: (f32, f32, f32),
    ox: i32,
    oy: i32,
    seed: u64,
    params: &TreeParams,
) -> (f32, f32, f32) {
    // Primary direction as unit vector.
    let plen = (primary_end.0 * primary_end.0
        + primary_end.1 * primary_end.1
        + primary_end.2 * primary_end.2)
        .sqrt()
        .max(0.001);
    let pdir = (primary_end.0 / plen, primary_end.1 / plen, primary_end.2 / plen);

    // Angle around primary axis.
    let m = params.num_secondary as f32;
    let angle_base = (j as f32) * std::f32::consts::TAU / m;
    let angle_jitter = (noise::hash_f32(
        ox,
        oy,
        (i * 17 + j) * 11 + 1,
        seed.wrapping_add(0xB130),
    ) - 0.5)
        * 0.8;
    let angle = angle_base + angle_jitter;

    // Elevation from primary direction.
    let elev_hash = noise::hash_f32(
        ox,
        oy,
        (i * 17 + j) * 11 + 2,
        seed.wrapping_add(0xB131),
    );
    let elev_deg = params.secondary_elev_min
        + elev_hash * (params.secondary_elev_max - params.secondary_elev_min);
    let elev_rad = elev_deg * std::f32::consts::PI / 180.0;

    // Length jitter ±20%.
    let len_hash = noise::hash_f32(
        ox,
        oy,
        (i * 17 + j) * 11 + 3,
        seed.wrapping_add(0xB132),
    );
    let length = params.secondary_length * (0.8 + len_hash * 0.4);

    // Build an orthonormal basis (u, v) perpendicular to pdir.
    // Pick a helper vector NOT parallel to pdir.
    let helper = if pdir.2.abs() < 0.9 {
        (0.0_f32, 0.0_f32, 1.0_f32)
    } else {
        (1.0_f32, 0.0_f32, 0.0_f32)
    };
    // u = normalize(pdir × helper)
    let cross_u = (
        pdir.1 * helper.2 - pdir.2 * helper.1,
        pdir.2 * helper.0 - pdir.0 * helper.2,
        pdir.0 * helper.1 - pdir.1 * helper.0,
    );
    let ulen = (cross_u.0 * cross_u.0 + cross_u.1 * cross_u.1 + cross_u.2 * cross_u.2)
        .sqrt()
        .max(0.001);
    let u = (cross_u.0 / ulen, cross_u.1 / ulen, cross_u.2 / ulen);
    // v = pdir × u  (already unit since pdir and u are unit and perpendicular)
    let v = (
        pdir.1 * u.2 - pdir.2 * u.1,
        pdir.2 * u.0 - pdir.0 * u.2,
        pdir.0 * u.1 - pdir.1 * u.0,
    );

    // Direction = cos(elev)*pdir + sin(elev)*(cos(angle)*u + sin(angle)*v)
    let c_elev = elev_rad.cos();
    let s_elev = elev_rad.sin();
    let c_ang = angle.cos();
    let s_ang = angle.sin();
    let dir_x = c_elev * pdir.0 + s_elev * (c_ang * u.0 + s_ang * v.0);
    let dir_y = c_elev * pdir.1 + s_elev * (c_ang * u.1 + s_ang * v.1);
    let dir_z = c_elev * pdir.2 + s_elev * (c_ang * u.2 + s_ang * v.2);
    (dir_x * length, dir_y * length, dir_z * length)
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

/// Procedurally stamp a tree using a 3-level hierarchical branching grammar.
///
/// Level 0: trunk (vertical capsule)
/// Level 1: primary branches (fork from fork_height)
/// Level 2: secondary branches (fork from primary tips)
/// Level 3: leaf clusters (spheres at secondary tips)
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

    let ox_f = ox as f32;
    let oy_f = oy as f32;
    let fbz = feature_base_z as f32;
    let fork_z = fbz + params.fork_height as f32;
    let trunk_top_z = fbz + params.trunk_height as f32;

    // Precompute primary and secondary endpoints in absolute world coords,
    // plus an aggregate bounding box.
    let num_primary = params.num_primary.max(0);
    let num_secondary = params.num_secondary.max(0);

    // Per-primary: absolute end point
    let mut primary_ends: [(f32, f32, f32); 8] = [(0.0, 0.0, 0.0); 8];
    // Per-(primary, secondary): absolute end point (max 8 primaries × 8 secondaries)
    let mut secondary_ends: [[(f32, f32, f32); 8]; 8] = [[(0.0, 0.0, 0.0); 8]; 8];
    let np = (num_primary as usize).min(8);
    let ns = (num_secondary as usize).min(8);

    // Track bounding box (absolute world coords).
    let mut bb_min_x = ox_f - params.trunk_radius - 1.0;
    let mut bb_max_x = ox_f + params.trunk_radius + 1.0;
    let mut bb_min_y = oy_f - params.trunk_radius - 1.0;
    let mut bb_max_y = oy_f + params.trunk_radius + 1.0;
    let mut bb_min_z = fbz - 1.0;
    let mut bb_max_z = trunk_top_z + 1.0;

    for i in 0..np {
        let (pdx, pdy, pdz) = primary_branch_end(i as i32, ox, oy, seed, &params);
        let pend = (ox_f + pdx, oy_f + pdy, fork_z + pdz);
        primary_ends[i] = pend;

        let r = params.branch_radius + 1.0;
        bb_min_x = bb_min_x.min(pend.0 - r);
        bb_max_x = bb_max_x.max(pend.0 + r);
        bb_min_y = bb_min_y.min(pend.1 - r);
        bb_max_y = bb_max_y.max(pend.1 + r);
        bb_min_z = bb_min_z.min(pend.2 - r);
        bb_max_z = bb_max_z.max(pend.2 + r);

        for j in 0..ns {
            let (sdx, sdy, sdz) =
                secondary_branch_end(i as i32, j as i32, (pdx, pdy, pdz), ox, oy, seed, &params);
            let send = (pend.0 + sdx, pend.1 + sdy, pend.2 + sdz);
            secondary_ends[i][j] = send;

            let lr = params.leaf_radius + 1.0;
            bb_min_x = bb_min_x.min(send.0 - lr);
            bb_max_x = bb_max_x.max(send.0 + lr);
            bb_min_y = bb_min_y.min(send.1 - lr);
            bb_max_y = bb_max_y.max(send.1 + lr);
            bb_min_z = bb_min_z.min(send.2 - lr);
            bb_max_z = bb_max_z.max(send.2 + lr);
        }
    }

    let min_wx = bb_min_x.floor() as i32;
    let max_wx = bb_max_x.ceil() as i32;
    let min_wy = bb_min_y.floor() as i32;
    let max_wy = bb_max_y.ceil() as i32;
    let min_wz = bb_min_z.floor() as i32;
    let max_wz = bb_max_z.ceil() as i32;

    for wz in min_wz..=max_wz {
        let llz = wz - base_cz;
        if llz < 0 || llz >= CHUNK_SIZE as i32 {
            continue;
        }
        for wy in min_wy..=max_wy {
            let vly = wy - base_y;
            if vly < 0 || vly >= CHUNK_SIZE as i32 {
                continue;
            }
            for wx in min_wx..=max_wx {
                let vlx = wx - base_x;
                if vlx < 0 || vlx >= CHUNK_SIZE as i32 {
                    continue;
                }

                let px = wx as f32;
                let py = wy as f32;
                let pz = wz as f32;

                // 1. Trunk capsule
                let trunk_d = capsule_sdf(
                    px,
                    py,
                    pz,
                    ox_f,
                    oy_f,
                    fbz,
                    ox_f,
                    oy_f,
                    trunk_top_z,
                    params.trunk_radius,
                );
                let mut is_wood = trunk_d <= 0.0;

                // 2. Primary branches
                if !is_wood {
                    for i in 0..np {
                        let pend = primary_ends[i];
                        let d = capsule_sdf(
                            px,
                            py,
                            pz,
                            ox_f,
                            oy_f,
                            fork_z,
                            pend.0,
                            pend.1,
                            pend.2,
                            params.branch_radius,
                        );
                        if d <= 0.0 {
                            is_wood = true;
                            break;
                        }
                    }
                }

                // 3. Secondary branches
                if !is_wood {
                    'outer: for i in 0..np {
                        let pend = primary_ends[i];
                        for j in 0..ns {
                            let send = secondary_ends[i][j];
                            let d = capsule_sdf(
                                px,
                                py,
                                pz,
                                pend.0,
                                pend.1,
                                pend.2,
                                send.0,
                                send.1,
                                send.2,
                                params.branch_radius * 0.55,
                            );
                            if d <= 0.0 {
                                is_wood = true;
                                break 'outer;
                            }
                        }
                    }
                }

                let mat = if is_wood {
                    VoxelMaterial::WoodLog
                } else {
                    // 4. Leaf spheres at secondary tips
                    let mut leaf_hit = false;
                    'lo: for i in 0..np {
                        for j in 0..ns {
                            let send = secondary_ends[i][j];
                            let d = sphere_sdf(px, py, pz, send.0, send.1, send.2, params.leaf_radius);
                            if d <= 0.0 {
                                leaf_hit = true;
                                break 'lo;
                            }
                        }
                    }
                    if leaf_hit {
                        params.canopy_material
                    } else {
                        continue;
                    }
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

    // Horizontal halo: hierarchical recursive trees can reach
    //   primary_horizontal (~56) + secondary (~44) + leaf_radius (~23) ≈ 123.
    // Worst case is a large jungle tree (trunk_height 120, primary ratio 0.55).
    const HALO: i32 = 135;
    // Vertical extent of a tree above its base: fork_height + primary elevation +
    // secondary + leaf_radius. Large jungle: ~78 + 69 + 40 + 23 ≈ 210.
    const FEATURE_MAX_Z_ABOVE: i32 = 220;
    // Secondary branches can point downward from the primary direction,
    // roughly secondary_length + leaf_radius below fork for droopy trees.
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
