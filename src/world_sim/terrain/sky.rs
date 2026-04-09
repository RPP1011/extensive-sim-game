//! Flying island chunk generator.
//!
//! Islands are placed on a deterministic 64-voxel grid.  ~30% of grid cells
//! have an island (hash < 0.3).  Each island is an inverted-cone SDF with
//! noise perturbation for organic shaping.
//!
//! Optional waterfalls (~33% of islands) emit a Water column below the island.

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use super::noise;

pub use crate::world_sim::constants::SKY_BASE_Z;

// ---------------------------------------------------------------------------
// Island grid
// ---------------------------------------------------------------------------

const ISLAND_SPACING: i32 = 64;
const ISLAND_PRESENCE_SALT: u64 = 0xF1A1_B1A0_0001;
const ISLAND_RADIUS_SALT: u64   = 0xF1A1_B1A0_0002;
const ISLAND_THICK_SALT: u64    = 0xF1A1_B1A0_0003;
const ISLAND_WFALL_SALT: u64    = 0xF1A1_B1A0_0004;
const ISLAND_NOISE_SALT_A: u64  = 0xF1A1_B1A0_0005;
const ISLAND_NOISE_SALT_B: u64  = 0xF1A1_B1A0_0006;

struct IslandDef {
    /// World-space center X/Y of the island.
    cx: f32,
    cy: f32,
    /// Vertical centre in world Z.
    cz: f32,
    radius: f32,
    thickness: f32,
    waterfall: bool,
}

fn islands_for_chunk(cp: ChunkPos, seed: u64) -> Vec<IslandDef> {
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    // Scan a generous neighbourhood of grid cells that could overlap this chunk
    let search_radius = 2; // grid cells
    let gx0 = (base_x as f32 / ISLAND_SPACING as f32).floor() as i32 - search_radius;
    let gx1 = ((base_x + CHUNK_SIZE as i32) as f32 / ISLAND_SPACING as f32).ceil() as i32 + search_radius;
    let gy0 = (base_y as f32 / ISLAND_SPACING as f32).floor() as i32 - search_radius;
    let gy1 = ((base_y + CHUNK_SIZE as i32) as f32 / ISLAND_SPACING as f32).ceil() as i32 + search_radius;

    let mut islands = Vec::new();

    for gy in gy0..=gy1 {
        for gx in gx0..=gx1 {
            // Presence check
            let presence = noise::hash_f32(gx, gy, 0, seed.wrapping_add(ISLAND_PRESENCE_SALT));
            if presence >= 0.3 {
                continue;
            }

            // Island centre (slight random offset within the cell)
            let off_x = noise::hash_f32(gx, gy, 1, seed.wrapping_add(ISLAND_RADIUS_SALT));
            let off_y = noise::hash_f32(gx, gy, 2, seed.wrapping_add(ISLAND_RADIUS_SALT));
            let cx = gx as f32 * ISLAND_SPACING as f32 + off_x * ISLAND_SPACING as f32;
            let cy = gy as f32 * ISLAND_SPACING as f32 + off_y * ISLAND_SPACING as f32;

            // Vertical position variation around SKY_BASE_Z
            let voff = noise::hash_f32(gx, gy, 3, seed.wrapping_add(ISLAND_THICK_SALT));
            let cz = SKY_BASE_Z as f32 + voff * 40.0 - 20.0;

            let r_hash = noise::hash_f32(gx, gy, 4, seed.wrapping_add(ISLAND_RADIUS_SALT));
            let radius = 15.0 + r_hash * 25.0; // 15..40

            let t_hash = noise::hash_f32(gx, gy, 5, seed.wrapping_add(ISLAND_THICK_SALT));
            let thickness = 8.0 + t_hash * 12.0; // 8..20

            let wf_hash = noise::hash_f32(gx, gy, 6, seed.wrapping_add(ISLAND_WFALL_SALT));
            let waterfall = wf_hash < 0.33;

            // Rough AABB cull against this chunk
            let max_r = radius + 3.0; // +3 for noise perturbation
            if (cx - base_x as f32).abs() > max_r + CHUNK_SIZE as f32 { continue; }
            if (cy - base_y as f32).abs() > max_r + CHUNK_SIZE as f32 { continue; }
            if (cz - base_z as f32).abs() > thickness * 0.5 + CHUNK_SIZE as f32 { continue; }

            islands.push(IslandDef { cx, cy, cz, radius, thickness, waterfall });
        }
    }

    islands
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a chunk for the sky / flying-island layer.
///
/// Returns an air chunk with island voxels stamped in.  If the chunk is
/// entirely outside the island layer a pure-air chunk is returned cheaply
/// (the islands_for_chunk scan will yield nothing).
pub fn generate_flying_island_chunk(cp: ChunkPos, seed: u64) -> Chunk {
    let mut chunk = Chunk::new_air(cp);

    let islands = islands_for_chunk(cp, seed);
    if islands.is_empty() {
        return chunk;
    }

    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    let noise_seed_a = seed.wrapping_add(ISLAND_NOISE_SALT_A);
    let noise_seed_b = seed.wrapping_add(ISLAND_NOISE_SALT_B);

    for island in &islands {
        for lz in 0..CHUNK_SIZE {
            let vz = base_z + lz as i32;
            let dz = vz as f32 - island.cz;

            // z_frac: 0 at bottom of island, 1 at top
            let z_frac = (dz + island.thickness * 0.5) / island.thickness;
            if z_frac < 0.0 || z_frac > 1.0 {
                // Waterfall column: emit below the island
                if island.waterfall {
                    let below_bottom = island.cz - island.thickness * 0.5;
                    if (vz as f32) < below_bottom && (vz as f32) >= below_bottom - 8.0 {
                        // Stamp a 1-voxel wide water column at island centre
                        let wlx = (island.cx - base_x as f32).round() as i32;
                        let wly = (island.cy - base_y as f32).round() as i32;
                        if wlx >= 0 && wly >= 0 && wlx < CHUNK_SIZE as i32 && wly < CHUNK_SIZE as i32 {
                            let idx = local_index(wlx as usize, wly as usize, lz);
                            chunk.voxels[idx] = Voxel::new(VoxelMaterial::Water);
                        }
                    }
                }
                continue;
            }

            // Radius shrinks toward bottom (inverted-cone SDF)
            let local_radius = island.radius * z_frac;

            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let vx = base_x + lx as i32;
                    let vy = base_y + ly as i32;

                    let dx = vx as f32 - island.cx;
                    let dy = vy as f32 - island.cy;
                    let dist_xy = (dx * dx + dy * dy).sqrt();

                    // Organic noise perturbation ±3 voxels
                    let perturb = (noise::value_noise_3d(
                        vx as f32 * 0.1,
                        vy as f32 * 0.1,
                        vz as f32 * 0.1,
                        noise_seed_a,
                        1.0,
                    ) * 2.0 - 1.0) * 3.0;
                    let perturb2 = (noise::value_noise_3d(
                        vx as f32 * 0.1,
                        vy as f32 * 0.1,
                        vz as f32 * 0.1,
                        noise_seed_b,
                        1.0,
                    ) * 2.0 - 1.0) * 2.0;

                    if dist_xy > local_radius + perturb + perturb2 {
                        continue;
                    }

                    // Material by z_frac
                    let mat = if z_frac > 0.85 {
                        VoxelMaterial::Grass
                    } else if z_frac > 0.4 {
                        VoxelMaterial::Dirt
                    } else {
                        VoxelMaterial::Stone
                    };

                    let idx = local_index(lx, ly, lz);
                    chunk.voxels[idx] = Voxel::new(mat);
                }
            }
        }
    }

    chunk.dirty = true;
    chunk
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};

    /// Find a chunk near SKY_BASE_Z that reliably contains island mass.
    /// We scan until we find one with solid voxels.
    fn find_island_chunk(seed: u64) -> Option<Chunk> {
        let sky_cz = SKY_BASE_Z / CHUNK_SIZE as i32;
        // Scan a grid of positions at the right Z height
        for cy in 0..10 {
            for cx in 0..10 {
                let cp = ChunkPos::new(cx, cy, sky_cz);
                let chunk = generate_flying_island_chunk(cp, seed);
                let solid = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
                if solid > 0 {
                    return Some(chunk);
                }
            }
        }
        None
    }

    #[test]
    fn island_has_solid_mass() {
        // Scan 20×20 chunks at sky level. Sum all solid voxels across the area.
        // An island with radius 15-40 spans multiple chunks; the total should be large.
        let sky_cz = SKY_BASE_Z / CHUNK_SIZE as i32;
        let total_solid: usize = (0..20)
            .flat_map(|cy: i32| (0..20i32).map(move |cx| (cx, cy)))
            .map(|(cx, cy)| {
                let cp = ChunkPos::new(cx, cy, sky_cz);
                generate_flying_island_chunk(cp, 42)
                    .voxels.iter().filter(|v| v.material.is_solid()).count()
            })
            .sum();
        assert!(total_solid > 50, "no significant island mass found in 20×20 scan at sky level (seed=42): only {total_solid} solid voxels");
    }

    #[test]
    fn island_has_grass_on_top() {
        // Scan higher z slices (z+1) where z_frac would be >0.85
        let sky_cz_base = SKY_BASE_Z / CHUNK_SIZE as i32;
        let mut found_grass = false;
        'outer: for dz in 0..3i32 {
            for cy in 0..20 {
                for cx in 0..20 {
                    let cp = ChunkPos::new(cx, cy, sky_cz_base + dz);
                    let chunk = generate_flying_island_chunk(cp, 42);
                    if chunk.voxels.iter().any(|v| v.material == VoxelMaterial::Grass) {
                        found_grass = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(found_grass, "no Grass voxels found on island tops (seed=42)");
    }

    #[test]
    fn island_is_deterministic() {
        let cp = ChunkPos::new(3, 5, SKY_BASE_Z / CHUNK_SIZE as i32);
        let a = generate_flying_island_chunk(cp, 42);
        let b = generate_flying_island_chunk(cp, 42);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }
}
