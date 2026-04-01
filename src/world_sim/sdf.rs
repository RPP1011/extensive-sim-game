//! Signed Distance Field generation and queries for voxel chunks.
//!
//! Each chunk can generate an SDF where every voxel stores its signed distance
//! to the nearest surface (solid↔air boundary). Positive = in air, negative = inside solid.
//!
//! Generated via Jump Flooding Algorithm (JFA) — O(n log n) per chunk.
//! Used for:
//! - Smooth rendering (raymarching against SDF, no staircase artifacts)
//! - Spatial queries (distance to nearest wall, surface normal, buildability)
//! - Construction AI (vulnerability, chokepoint detection, gap finding)

use super::voxel::*;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// ChunkSDF
// ---------------------------------------------------------------------------

/// SDF data for a single chunk. Lazily computed, cached until chunk is dirtied.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChunkSDF {
    /// Signed distance per voxel. Positive = air side, negative = solid side.
    /// Magnitude = distance to nearest surface in voxel units.
    pub distances: Vec<f32>,
    /// Packed surface normal per voxel (only meaningful near surfaces).
    /// Encoded as (nx*127+127, ny*127+127, nz*127+127, 0) in a u32.
    pub normals: Vec<u32>,
}

impl std::fmt::Debug for ChunkSDF {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let surface_count = self.distances.iter().filter(|d| d.abs() < 1.5).count();
        write!(f, "ChunkSDF(surface_voxels={})", surface_count)
    }
}

impl ChunkSDF {
    /// Distance at a local voxel position.
    pub fn distance_at(&self, lx: usize, ly: usize, lz: usize) -> f32 {
        self.distances[local_index(lx, ly, lz)]
    }

    /// Decoded normal at a local voxel position.
    pub fn normal_at(&self, lx: usize, ly: usize, lz: usize) -> (f32, f32, f32) {
        decode_normal(self.normals[local_index(lx, ly, lz)])
    }

    /// Is this voxel position on or very near the surface? (|distance| < threshold)
    pub fn is_surface(&self, lx: usize, ly: usize, lz: usize, threshold: f32) -> bool {
        self.distances[local_index(lx, ly, lz)].abs() < threshold
    }

    /// Count of voxels near the surface (|d| < 1.5).
    pub fn surface_count(&self) -> usize {
        self.distances.iter().filter(|d| d.abs() < 1.5).count()
    }

    /// Minimum distance (most deeply inside solid).
    pub fn min_distance(&self) -> f32 {
        self.distances.iter().cloned().fold(f32::MAX, f32::min)
    }

    /// Maximum distance (deepest into air).
    pub fn max_distance(&self) -> f32 {
        self.distances.iter().cloned().fold(f32::MIN, f32::max)
    }
}

// ---------------------------------------------------------------------------
// Normal encoding
// ---------------------------------------------------------------------------

fn encode_normal(nx: f32, ny: f32, nz: f32) -> u32 {
    let x = ((nx * 127.0 + 127.0) as u8) as u32;
    let y = ((ny * 127.0 + 127.0) as u8) as u32;
    let z = ((nz * 127.0 + 127.0) as u8) as u32;
    (x << 16) | (y << 8) | z
}

fn decode_normal(packed: u32) -> (f32, f32, f32) {
    let x = ((packed >> 16) & 0xFF) as f32 / 127.0 - 1.0;
    let y = ((packed >> 8) & 0xFF) as f32 / 127.0 - 1.0;
    let z = (packed & 0xFF) as f32 / 127.0 - 1.0;
    (x, y, z)
}

// ---------------------------------------------------------------------------
// JFA-based SDF generation
// ---------------------------------------------------------------------------

/// Seed entry for Jump Flooding: tracks the nearest known surface voxel.
#[derive(Clone, Copy)]
struct JFASeed {
    /// Source voxel that this seed propagated from. (-1,-1,-1) = no seed.
    sx: i16,
    sy: i16,
    sz: i16,
}

impl Default for JFASeed {
    fn default() -> Self { Self { sx: -1, sy: -1, sz: -1 } }
}

impl JFASeed {
    fn is_valid(self) -> bool { self.sx >= 0 }

    fn distance_to(self, x: usize, y: usize, z: usize) -> f32 {
        if !self.is_valid() { return f32::MAX; }
        let dx = x as f32 - self.sx as f32;
        let dy = y as f32 - self.sy as f32;
        let dz = z as f32 - self.sz as f32;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Generate SDF for a chunk using the Jump Flooding Algorithm.
///
/// The JFA computes, for every voxel, the nearest surface voxel (a solid voxel
/// adjacent to air, or vice versa). The signed distance is then the euclidean
/// distance to that nearest surface, negative if inside solid.
///
/// `neighbor_fn` provides voxels from adjacent chunks for boundary accuracy.
/// Pass `None` for standalone chunk SDF (less accurate at boundaries).
pub fn generate_chunk_sdf(chunk: &Chunk, neighbor_fn: Option<&dyn Fn(i32, i32, i32) -> Voxel>) -> ChunkSDF {
    let n = CHUNK_SIZE;
    let vol = CHUNK_VOLUME;

    // Phase 1: Identify surface voxels (solid adjacent to air, or air adjacent to solid).
    let mut seeds = vec![JFASeed::default(); vol];

    for lz in 0..n {
        for ly in 0..n {
            for lx in 0..n {
                let idx = local_index(lx, ly, lz);
                let is_solid = chunk.voxels[idx].material.is_solid();

                // Check 6 face neighbors for surface detection.
                let is_surface = [(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                    .iter()
                    .any(|&(dx, dy, dz)| {
                        let nx = lx as i32 + dx;
                        let ny = ly as i32 + dy;
                        let nz = lz as i32 + dz;

                        let neighbor_solid = if nx >= 0 && nx < n as i32
                            && ny >= 0 && ny < n as i32
                            && nz >= 0 && nz < n as i32
                        {
                            chunk.voxels[local_index(nx as usize, ny as usize, nz as usize)]
                                .material.is_solid()
                        } else if let Some(f) = &neighbor_fn {
                            // Cross-chunk neighbor query.
                            let gx = chunk.pos.x * n as i32 + nx;
                            let gy = chunk.pos.y * n as i32 + ny;
                            let gz = chunk.pos.z * n as i32 + nz;
                            f(gx, gy, gz).material.is_solid()
                        } else {
                            is_solid // assume same as self at boundary (conservative)
                        };

                        is_solid != neighbor_solid // surface = solid/air boundary
                    });

                if is_surface {
                    seeds[idx] = JFASeed { sx: lx as i16, sy: ly as i16, sz: lz as i16 };
                }
            }
        }
    }

    // Phase 2: Jump Flooding passes.
    // Step sizes: n/2, n/4, ..., 1 (for n=16: 8, 4, 2, 1)
    let mut step = (n / 2) as i32;
    while step >= 1 {
        for lz in 0..n {
            for ly in 0..n {
                for lx in 0..n {
                    let idx = local_index(lx, ly, lz);
                    let mut best = seeds[idx];
                    let mut best_dist = best.distance_to(lx, ly, lz);

                    // Check 26 neighbors at ±step distance.
                    for dz in [-step, 0, step] {
                        for dy in [-step, 0, step] {
                            for dx in [-step, 0, step] {
                                if dx == 0 && dy == 0 && dz == 0 { continue; }
                                let nx = lx as i32 + dx;
                                let ny = ly as i32 + dy;
                                let nz = lz as i32 + dz;

                                if nx < 0 || nx >= n as i32
                                    || ny < 0 || ny >= n as i32
                                    || nz < 0 || nz >= n as i32 { continue; }

                                let nidx = local_index(nx as usize, ny as usize, nz as usize);
                                let candidate = seeds[nidx];
                                if !candidate.is_valid() { continue; }

                                let candidate_dist = candidate.distance_to(lx, ly, lz);
                                if candidate_dist < best_dist {
                                    best = candidate;
                                    best_dist = candidate_dist;
                                }
                            }
                        }
                    }
                    seeds[idx] = best;
                }
            }
        }
        step /= 2;
    }

    // Phase 3: Compute signed distances and normals from seed map.
    let mut distances = vec![f32::MAX; vol];
    let mut normals = vec![0u32; vol];

    for lz in 0..n {
        for ly in 0..n {
            for lx in 0..n {
                let idx = local_index(lx, ly, lz);
                let seed = seeds[idx];
                let is_solid = chunk.voxels[idx].material.is_solid();

                let dist = if seed.is_valid() {
                    let d = seed.distance_to(lx, ly, lz);
                    // Surface voxels (dist=0) get a half-voxel offset so they're
                    // not exactly zero, preserving sign information.
                    if d < 0.001 { 0.5 } else { d }
                } else {
                    // No seed found — deep interior or deep exterior.
                    // Use a large default scaled by chunk size.
                    (n as f32) * 0.75
                };

                // Sign: negative inside solid, positive in air.
                distances[idx] = if is_solid { -dist } else { dist };

                // Normal: computed from SDF gradient (central differences) after distances are set.
                // Deferred to post-pass below.
            }
        }
    }

    // Phase 4: Compute normals from SDF gradient (central differences).
    for lz in 0..n {
        for ly in 0..n {
            for lx in 0..n {
                let idx = local_index(lx, ly, lz);
                let d = |x: usize, y: usize, z: usize| -> f32 {
                    if x < n && y < n && z < n {
                        distances[local_index(x, y, z)]
                    } else {
                        distances[idx] // clamp at boundary
                    }
                };
                let gx = d(lx.wrapping_add(1).min(n - 1), ly, lz) - d(lx.saturating_sub(1), ly, lz);
                let gy = d(lx, ly.wrapping_add(1).min(n - 1), lz) - d(lx, ly.saturating_sub(1), lz);
                let gz = d(lx, ly, lz.wrapping_add(1).min(n - 1)) - d(lx, ly, lz.saturating_sub(1));
                let mag = (gx * gx + gy * gy + gz * gz).sqrt().max(0.001);
                normals[idx] = encode_normal(gx / mag, gy / mag, gz / mag);
            }
        }
    }

    ChunkSDF { distances, normals }
}

// ---------------------------------------------------------------------------
// VoxelWorld SDF integration
// ---------------------------------------------------------------------------

/// Generate or retrieve cached SDF for a chunk.
/// Returns None if the chunk isn't loaded.
pub fn get_or_generate_sdf(world: &mut VoxelWorld, cp: ChunkPos) -> Option<ChunkSDF> {
    let chunk = world.chunks.get(&cp)?;
    if !chunk.dirty {
        // TODO: cache SDF on chunk. For now, always regenerate.
    }

    let chunk_clone = chunk.clone();
    let neighbor = |gx: i32, gy: i32, gz: i32| -> Voxel {
        world.get_voxel(gx, gy, gz)
    };

    let sdf = generate_chunk_sdf(&chunk_clone, Some(&neighbor));
    Some(sdf)
}

/// Query the SDF distance at a world voxel position.
/// Generates SDF for the containing chunk if needed.
/// Returns positive (air), negative (solid), or 0.0 (unloaded).
pub fn sdf_distance_at(world: &mut VoxelWorld, vx: i32, vy: i32, vz: i32) -> f32 {
    let (cp, _) = voxel_to_chunk_local(vx, vy, vz);
    let sdf = match get_or_generate_sdf(world, cp) {
        Some(s) => s,
        None => return 0.0,
    };
    let lx = vx.rem_euclid(CHUNK_SIZE as i32) as usize;
    let ly = vy.rem_euclid(CHUNK_SIZE as i32) as usize;
    let lz = vz.rem_euclid(CHUNK_SIZE as i32) as usize;
    sdf.distance_at(lx, ly, lz)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_half_solid_chunk() -> Chunk {
        // Bottom half solid (z < 8), top half air (z >= 8).
        let mut chunk = Chunk::new_air(ChunkPos::new(0, 0, 0));
        for lz in 0..8 {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    chunk.set(lx, ly, lz, Voxel::new(VoxelMaterial::Stone));
                }
            }
        }
        chunk
    }

    #[test]
    fn sdf_surface_detected() {
        let chunk = make_half_solid_chunk();
        let sdf = generate_chunk_sdf(&chunk, None);

        // At z=7 (top of solid), should be near-surface with small negative distance.
        let d_solid_surface = sdf.distance_at(8, 8, 7);
        assert!(d_solid_surface < 0.0, "solid side of surface should be negative, got {}", d_solid_surface);
        assert!(d_solid_surface > -2.0, "should be close to surface, got {}", d_solid_surface);

        // At z=8 (bottom of air), should be near-surface with small positive distance.
        let d_air_surface = sdf.distance_at(8, 8, 8);
        assert!(d_air_surface > 0.0, "air side of surface should be positive, got {}", d_air_surface);
        assert!(d_air_surface < 2.0, "should be close to surface, got {}", d_air_surface);
    }

    #[test]
    fn sdf_distance_increases_from_surface() {
        let chunk = make_half_solid_chunk();
        let sdf = generate_chunk_sdf(&chunk, None);

        // Distance should increase as we move away from the surface (z=7/8 boundary).
        let d_at_6 = sdf.distance_at(8, 8, 6).abs();
        let d_at_4 = sdf.distance_at(8, 8, 4).abs();
        let d_at_2 = sdf.distance_at(8, 8, 2).abs();
        assert!(d_at_4 > d_at_6, "distance should increase deeper into solid: {} vs {}", d_at_4, d_at_6);
        assert!(d_at_2 > d_at_4, "distance should increase deeper: {} vs {}", d_at_2, d_at_4);

        // Air side: distance should increase away from surface (within the chunk interior,
        // avoiding boundary artifacts).
        let d_at_9 = sdf.distance_at(8, 8, 9).abs();
        let d_at_11 = sdf.distance_at(8, 8, 11).abs();
        assert!(d_at_11 > d_at_9, "distance should increase into air: z=11({}) vs z=9({})", d_at_11, d_at_9);
    }

    #[test]
    fn sdf_solid_negative_air_positive() {
        let chunk = make_half_solid_chunk();
        let sdf = generate_chunk_sdf(&chunk, None);

        // Deep solid = large negative.
        assert!(sdf.distance_at(8, 8, 0) < 0.0, "deep solid should be negative");
        // Deep air = large positive.
        assert!(sdf.distance_at(8, 8, 15) > 0.0, "deep air should be positive");
    }

    #[test]
    fn sdf_normals_point_correct_direction() {
        let chunk = make_half_solid_chunk();
        let sdf = generate_chunk_sdf(&chunk, None);

        // Surface normal at z=7 (solid side) should point upward (+z, toward air).
        // SDF gradient normals always point toward increasing distance (outward from surface).
        // At z=7 (solid, just below surface), gradient points +z (toward air = positive distance).
        let (_nx, _ny, nz) = sdf.normal_at(8, 8, 7);
        assert!(nz > 0.3, "solid surface normal should point toward air (+z), got nz={}", nz);

        // At z=8 (air, just above surface), gradient also points +z (toward more air).
        let (_nx2, _ny2, nz2) = sdf.normal_at(8, 8, 8);
        assert!(nz2 > 0.3, "air surface gradient should also point +z (toward increasing distance), got nz={}", nz2);
    }

    #[test]
    fn sdf_all_air_chunk() {
        let chunk = Chunk::new_air(ChunkPos::new(0, 0, 0));
        let sdf = generate_chunk_sdf(&chunk, None);

        // No surfaces — all distances should be large positive.
        for d in &sdf.distances {
            assert!(*d > 0.0, "all-air chunk should have positive distances");
        }
        assert_eq!(sdf.surface_count(), 0);
    }

    #[test]
    fn sdf_all_solid_chunk() {
        let chunk = Chunk::new_filled(ChunkPos::new(0, 0, 0), VoxelMaterial::Stone);
        let sdf = generate_chunk_sdf(&chunk, None);

        // No surfaces (boundary detection without neighbors defaults to same-as-self).
        // Interior should be large negative.
        let d = sdf.distance_at(8, 8, 8);
        assert!(d < 0.0, "all-solid chunk center should be negative, got {}", d);
    }

    #[test]
    fn sdf_single_solid_voxel() {
        let mut chunk = Chunk::new_air(ChunkPos::new(0, 0, 0));
        chunk.set(8, 8, 8, Voxel::new(VoxelMaterial::Stone));

        let sdf = generate_chunk_sdf(&chunk, None);

        // The solid voxel should be negative and near-surface.
        let d = sdf.distance_at(8, 8, 8);
        assert!(d < 0.0, "solid voxel should be negative");
        assert!(d > -2.0, "single solid voxel should be near surface");

        // Adjacent air voxels should be positive and near-surface.
        let d_adj = sdf.distance_at(9, 8, 8);
        assert!(d_adj > 0.0, "adjacent air should be positive");
        assert!(d_adj < 2.0, "adjacent air should be near surface");

        // Far voxel should have larger distance.
        let d_far = sdf.distance_at(15, 15, 15);
        assert!(d_far > d_adj, "far voxel should have larger distance");
    }

    #[test]
    fn sdf_surface_count() {
        let chunk = make_half_solid_chunk();
        let sdf = generate_chunk_sdf(&chunk, None);

        // Surface is a 16x16 plane at z=7/8 boundary.
        // Surface voxels: z=7 (solid side) + z=8 (air side) = 2 * 16 * 16 = 512.
        let count = sdf.surface_count();
        assert!(count >= 256, "should have at least 256 surface voxels, got {}", count);
        assert!(count <= 1024, "should have at most 1024 surface voxels, got {}", count);
    }

    #[test]
    fn sdf_with_neighbor_fn() {
        // Create a chunk that's all solid, with a neighbor function that returns air above.
        let chunk = Chunk::new_filled(ChunkPos::new(0, 0, 0), VoxelMaterial::Stone);

        let neighbor = |_gx: i32, _gy: i32, gz: i32| -> Voxel {
            // Everything above chunk (z >= 16) is air.
            if gz >= 16 { Voxel::default() } else { Voxel::new(VoxelMaterial::Stone) }
        };

        let sdf = generate_chunk_sdf(&chunk, Some(&neighbor));

        // Top layer (z=15) should be near-surface (neighbor above is air).
        let d_top = sdf.distance_at(8, 8, 15);
        assert!(d_top < 0.0, "top of solid chunk should be negative");
        assert!(d_top > -2.0, "top of solid chunk should be near surface with air above");

        // Deep interior should be more negative.
        let d_deep = sdf.distance_at(8, 8, 0);
        assert!(d_deep < d_top, "deep interior {} should be more negative than surface {}", d_deep, d_top);
    }

    #[test]
    fn normal_encode_decode_roundtrip() {
        let cases = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.577, 0.577, 0.577),
        ];
        for (nx, ny, nz) in cases {
            let packed = encode_normal(nx, ny, nz);
            let (dx, dy, dz) = decode_normal(packed);
            assert!((dx - nx).abs() < 0.02, "x: {} vs {}", dx, nx);
            assert!((dy - ny).abs() < 0.02, "y: {} vs {}", dy, ny);
            assert!((dz - nz).abs() < 0.02, "z: {} vs {}", dz, nz);
        }
    }
}
