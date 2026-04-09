//! Biome-driven chunk materialization.
//!
//! Replaces the flat terrain generator in `VoxelWorld` with a region-plan-aware
//! column builder. Each voxel column samples the `RegionPlan` to determine
//! biome, computes a surface height from bilinear interpolation + detail noise,
//! then assigns material by depth layer.

use crate::world_sim::voxel::{Chunk, ChunkPos, Voxel, VoxelMaterial, CHUNK_SIZE, local_index};
use crate::world_sim::state::Terrain;
use crate::world_sim::terrain::region_plan::{RegionPlan, SEA_LEVEL, MAX_SURFACE_Z};
use crate::world_sim::terrain::biome::{surface_materials, resolve_biome};
use crate::world_sim::terrain::noise;
use crate::world_sim::terrain::{caves, rivers, features, sky, dungeons};

// ---------------------------------------------------------------------------
// Ore vein helper
// ---------------------------------------------------------------------------

/// Return an ore material if a vein exists at this position.
/// `ore_boost` adds extra probability (positive float, e.g. 0.03 for +3%).
fn ore_at(vx: i32, vy: i32, vz: i32, seed: u64, ore_boost: f32) -> Option<VoxelMaterial> {
    let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0x0EE1));
    let threshold_base = 0.92f32 - ore_boost;
    let copper_base   = 0.88f32 - ore_boost;
    let crystal_base  = 0.95f32 - ore_boost;

    if n > threshold_base {
        if vz < 8 { Some(VoxelMaterial::GoldOre) }
        else if vz < 12 { Some(VoxelMaterial::IronOre) }
        else { Some(VoxelMaterial::Coal) }
    } else if n > copper_base && vz < 10 {
        Some(VoxelMaterial::CopperOre)
    } else if n > crystal_base && vz < 6 {
        Some(VoxelMaterial::Crystal)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Main materializer
// ---------------------------------------------------------------------------

/// Build a fully-populated chunk driven by the region plan.
///
/// - Samples the plan bilinearly at each column (vx, vy) for surface height.
/// - Adds detail noise (±10 voxels, fbm_2d scale 0.02, 3 octaves).
/// - Assigns materials by depth layer and biome.
pub fn materialize_chunk(cp: ChunkPos, plan: &RegionPlan, seed: u64) -> Chunk {
    let mut chunk = Chunk::new_air(cp);
    let base_x = cp.x * CHUNK_SIZE as i32;
    let base_y = cp.y * CHUNK_SIZE as i32;
    let base_z = cp.z * CHUNK_SIZE as i32;

    for lz in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let vx = base_x + lx as i32;
                let vy = base_y + ly as i32;
                let vz = base_z + lz as i32;

                // --- Sample region plan ---
                let (cell, _, _) = plan.sample(vx as f32, vy as f32);
                let terrain = cell.terrain;

                // --- Surface height ---
                let base_height = plan.interpolate_height(vx as f32, vy as f32); // [0, 1]
                let detail = noise::fbm_2d(
                    vx as f32 * 0.02,
                    vy as f32 * 0.02,
                    seed.wrapping_add(0xface_cafe),
                    3,
                    2.0,
                    0.5,
                );
                // detail is [0,1]; map to [-10, +10]
                let detail_offset = (detail * 2.0 - 1.0) * 10.0;
                let surface_z = (base_height * MAX_SURFACE_Z as f32 + detail_offset).round() as i32;

                // Depth: positive → underground (below surface), negative → above surface
                let depth = surface_z - vz;

                // --- Biome materials ---
                let mats = surface_materials(terrain);

                // --- Ore boost for mountains / caverns ---
                let ore_boost: f32 = match terrain {
                    Terrain::Mountains | Terrain::Caverns => 0.03,
                    _ => 0.0,
                };

                // --- Layer assignment ---
                let material = if vz < -120 {
                    // Deep bedrock — always granite
                    VoxelMaterial::Granite
                } else if depth > 80 {
                    // Deep stone zone — ore veins possible
                    ore_at(vx, vy, vz, seed, ore_boost)
                        .unwrap_or(mats.deep_stone)
                } else if depth > 20 {
                    // Subsoil zone — mix stone + subsoil by noise
                    let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0x1234));
                    if n > 0.4 { mats.deep_stone } else { mats.subsoil }
                } else if depth > 0 {
                    // Near-surface soil
                    mats.subsoil
                } else if depth >= -1 {
                    // Surface layer (depth 0 or -1 means vz == surface_z or surface_z+1)
                    // Mountains high up → snow cap
                    if matches!(terrain, Terrain::Mountains) && vz > 250 {
                        VoxelMaterial::Snow
                    } else {
                        mats.surface
                    }
                } else if vz <= SEA_LEVEL && matches!(terrain, Terrain::DeepOcean | Terrain::Coast | Terrain::CoralReef | Terrain::Swamp) {
                    // Above terrain surface but below sea level → water (only for water biomes)
                    VoxelMaterial::Water
                } else {
                    VoxelMaterial::Air
                };

                if material != VoxelMaterial::Air {
                    chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(material);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Post-pass feature integration
    // -----------------------------------------------------------------------

    // 1. Cave carving — for chunks that are predominantly underground.
    //    We sample the biome at the chunk centre to decide which cave type.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let cz = base_z + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        let base_h = plan.interpolate_height(cx as f32, cy as f32);
        let surface_z_centre = (base_h * MAX_SURFACE_Z as f32).round() as i32;
        let depth_below = surface_z_centre - cz;
        if depth_below > 20 {
            let bv = resolve_biome(cell.terrain, cell.sub_biome, depth_below, seed);
            caves::carve_caves(&mut chunk, cp, bv.underground, seed);
        }
    }

    // 2. River carving — for all rivers in the plan.
    for river in &plan.rivers {
        // Build a closure that reproduces the surface-height formula used above.
        let plan_ref: &RegionPlan = plan;
        let seed_inner = seed;
        let surface_z_fn = |vx: f32, vy: f32| -> i32 {
            let bh = plan_ref.interpolate_height(vx, vy);
            let detail = noise::fbm_2d(
                vx * 0.02,
                vy * 0.02,
                seed_inner.wrapping_add(0xface_cafe),
                3, 2.0, 0.5,
            );
            let detail_offset = (detail * 2.0 - 1.0) * 10.0;
            (bh * MAX_SURFACE_Z as f32 + detail_offset).round() as i32
        };
        rivers::carve_river_in_chunk(&mut chunk, cp, river, &surface_z_fn);
    }

    // 3. Surface features — check each column's surface_z and see if it falls
    //    in this chunk.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        let terrain = cell.terrain;
        let sub_biome = cell.sub_biome;
        let base_h = plan.interpolate_height(cx as f32, cy as f32);
        let detail = noise::fbm_2d(
            cx as f32 * 0.02,
            cy as f32 * 0.02,
            seed.wrapping_add(0xface_cafe),
            3, 2.0, 0.5,
        );
        let detail_offset = (detail * 2.0 - 1.0) * 10.0;
        let surface_z = (base_h * MAX_SURFACE_Z as f32 + detail_offset).round() as i32;
        let surface_z_local = surface_z - base_z;
        features::place_surface_features(&mut chunk, cp, terrain, sub_biome, surface_z_local, seed);
    }

    // 4. Flying islands — only for sky-level chunks in FlyingIslands biome.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        if matches!(cell.terrain, Terrain::FlyingIslands) {
            // Blend in island voxels — stamp any non-air voxel from the island
            // generator into this chunk (island generator is standalone).
            let island_chunk = sky::generate_flying_island_chunk(cp, seed);
            for i in 0..island_chunk.voxels.len() {
                if island_chunk.voxels[i].material != VoxelMaterial::Air {
                    chunk.voxels[i] = island_chunk.voxels[i];
                }
            }
        }
    }

    // 5. Dungeon carving — for chunks that overlap dungeon sites in the plan.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        // Determine which plan cell we're in
        let col = (cx as f32 / crate::world_sim::terrain::region_plan::CELL_SIZE as f32)
            .floor().clamp(0.0, (plan.cols - 1) as f32) as i32;
        let row = (cy as f32 / crate::world_sim::terrain::region_plan::CELL_SIZE as f32)
            .floor().clamp(0.0, (plan.rows - 1) as f32) as i32;
        for dungeon_plan in &cell.dungeons {
            let layout = dungeons::DungeonLayout::generate(col, row, base_z, dungeon_plan.depth, seed);
            layout.carve_into_chunk(&mut chunk, cp);
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
    use crate::world_sim::terrain::region_plan::{generate_continent, SEA_LEVEL};
    use crate::world_sim::voxel::{ChunkPos, VoxelMaterial, CHUNK_SIZE};
    use crate::world_sim::state::Terrain;

    fn test_plan() -> RegionPlan {
        generate_continent(10, 10, 42)
    }

    #[test]
    fn chunk_at_surface_has_ground() {
        let plan = test_plan();
        // Find a land cell (not ocean) to sample surface height from.
        let land_idx = plan.cells.iter().position(|c| {
            c.height > 0.3
                && !matches!(c.terrain, Terrain::DeepOcean | Terrain::Coast | Terrain::CoralReef)
        });
        if land_idx.is_none() {
            return; // degenerate plan — skip
        }
        let li = land_idx.unwrap();
        let col = li % plan.cols;
        let row = li / plan.cols;
        // Chunk coords: one cell = CELL_SIZE voxels = CELL_SIZE/CHUNK_SIZE chunks.
        // Pick the chunk in the middle of the cell, at the expected surface z.
        let cell = plan.get(col, row);
        use crate::world_sim::terrain::region_plan::MAX_SURFACE_Z;
        let vx_mid = (col as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE) + 8;
        let vy_mid = (row as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE) + 8;
        let approx_surface_z = (cell.height * MAX_SURFACE_Z as f32).round() as i32;
        let cz = approx_surface_z.div_euclid(CHUNK_SIZE as i32);
        let cx = vx_mid.div_euclid(CHUNK_SIZE as i32);
        let cy = vy_mid.div_euclid(CHUNK_SIZE as i32);
        // Check a range of z chunks around the expected surface
        let mut found_solid = false;
        for dz in -2..=2i32 {
            let chunk = materialize_chunk(ChunkPos::new(cx, cy, cz + dz), &plan, 42);
            if chunk.voxels.iter().any(|v| v.material.is_solid()) {
                found_solid = true;
                break;
            }
        }
        assert!(found_solid, "surface area around land cell ({col},{row}) has no solid voxels");
    }

    #[test]
    fn chunk_deep_underground_is_solid() {
        let plan = test_plan();
        let chunk = materialize_chunk(ChunkPos::new(5, 5, -5), &plan, 42);
        let solid_count = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert!(solid_count > total / 2, "deep chunk not mostly solid: {solid_count}/{total}");
    }

    #[test]
    fn chunk_high_sky_is_air() {
        let plan = test_plan();
        // z=30 in chunk-space → voxel z = 30 * 16 = 480, well above MAX_SURFACE_Z (400)
        let chunk = materialize_chunk(ChunkPos::new(5, 5, 30), &plan, 42);
        let air_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert_eq!(air_count, total, "sky chunk is not all air");
    }

    #[test]
    fn different_biomes_produce_different_surfaces() {
        let plan = test_plan();
        let plains_idx = plan.cells.iter().position(|c| c.terrain == Terrain::Plains);
        let other_idx = plan.cells.iter().position(|c| {
            c.terrain != Terrain::Plains
                && c.terrain != Terrain::DeepOcean
                && c.terrain != Terrain::Coast
        });
        if plains_idx.is_none() || other_idx.is_none() {
            return; // not enough biome variety in this seed — skip
        }
        let pi = plains_idx.unwrap();
        let oi = other_idx.unwrap();
        let pc = (pi % plan.cols, pi / plan.cols);
        let oc = (oi % plan.cols, oi / plan.cols);
        let surface_z = (SEA_LEVEL / CHUNK_SIZE as i32) + 2;
        let chunk_a = materialize_chunk(ChunkPos::new(pc.0 as i32, pc.1 as i32, surface_z), &plan, 42);
        let chunk_b = materialize_chunk(ChunkPos::new(oc.0 as i32, oc.1 as i32, surface_z), &plan, 42);
        let diffs = chunk_a.voxels.iter()
            .zip(chunk_b.voxels.iter())
            .filter(|(a, b)| a.material != b.material)
            .count();
        assert!(diffs > 0, "different biomes produced identical chunks");
    }

    #[test]
    fn materialization_is_deterministic() {
        let plan = test_plan();
        let cp = ChunkPos::new(3, 3, 2);
        let a = materialize_chunk(cp, &plan, 42);
        let b = materialize_chunk(cp, &plan, 42);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }
}
