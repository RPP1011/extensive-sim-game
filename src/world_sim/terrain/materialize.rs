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
// Surface height computation (shared by main pass + post-passes)
// ---------------------------------------------------------------------------

/// Compute the surface Z at a given voxel (x, y) position.
/// Uses plan height + multi-scale detail noise with biome-dependent amplitude.
pub fn surface_height_at(vx: f32, vy: f32, plan: &RegionPlan, seed: u64) -> i32 {
    let base_height = plan.interpolate_height(vx, vy);
    let (cell, _, _) = plan.sample(vx, vy);
    let terrain = cell.terrain;

    let large = noise::fbm_2d(vx * 0.004, vy * 0.004, seed.wrapping_add(0xface_cafe), 4, 2.0, 0.5);
    let medium = noise::fbm_2d(vx * 0.015, vy * 0.015, seed.wrapping_add(0xdead_beef), 3, 2.0, 0.5);
    let small = noise::fbm_2d(vx * 0.06, vy * 0.06, seed.wrapping_add(0xcafe_babe), 2, 2.0, 0.5);

    let (large_amp, medium_amp, small_amp) = match terrain {
        Terrain::Mountains | Terrain::Glacier => (80.0, 30.0, 5.0),
        Terrain::Volcano => (60.0, 25.0, 4.0),
        Terrain::Badlands => (50.0, 35.0, 10.0),
        Terrain::Forest => (30.0, 12.0, 3.0),
        Terrain::Jungle => (45.0, 20.0, 6.0),
        Terrain::Tundra => (25.0, 10.0, 2.0),
        Terrain::Desert => (35.0, 20.0, 4.0),
        Terrain::Swamp | Terrain::Coast => (5.0, 3.0, 1.0),
        Terrain::DeepOcean | Terrain::CoralReef => (10.0, 5.0, 1.0),
        _ => (25.0, 10.0, 3.0),
    };

    let detail_offset = (large * 2.0 - 1.0) * large_amp
        + (medium * 2.0 - 1.0) * medium_amp
        + (small * 2.0 - 1.0) * small_amp;
    (base_height * MAX_SURFACE_Z as f32 + detail_offset).round() as i32
}

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
pub fn materialize_chunk(cp: ChunkPos, plan: &RegionPlan, seed: u64, clearing_center: Option<(f32, f32)>) -> Chunk {
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
                let surface_z = surface_height_at(vx as f32, vy as f32, plan, seed);

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
                let material = if vz < -500 {
                    // Deep bedrock — always granite (50m below datum)
                    VoxelMaterial::Granite
                } else if depth > 80 {
                    // Deep stone zone — ore veins possible
                    ore_at(vx, vy, vz, seed, ore_boost)
                        .unwrap_or(mats.deep_stone)
                } else if depth > 20 {
                    // Subsoil zone — mix stone + subsoil by noise
                    if matches!(terrain, Terrain::Mountains | Terrain::Glacier) {
                        // Mountains: geological banding on cliff faces
                        let band = ((vz as f32 * 0.15).sin() * 0.5 + 0.5) as f32;
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0x1234));
                        if band > 0.75 {
                            VoxelMaterial::Granite
                        } else if band > 0.5 && n > 0.3 {
                            VoxelMaterial::Gravel
                        } else {
                            VoxelMaterial::Stone
                        }
                    } else {
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0x1234));
                        if n > 0.4 { mats.deep_stone } else { mats.subsoil }
                    }
                } else if depth > 0 {
                    // Near-surface soil — badlands get layered sediment
                    if matches!(terrain, Terrain::Badlands) {
                        let band = ((vz as f32 * 0.3).sin() * 0.5 + 0.5) as f32;
                        if band > 0.7 { VoxelMaterial::Clay }
                        else if band > 0.4 { VoxelMaterial::Sandstone }
                        else { VoxelMaterial::RedSand }
                    } else {
                        mats.subsoil
                    }
                } else if depth >= -1 {
                    // Surface layer (depth 0 or -1 means vz == surface_z or surface_z+1)
                    if matches!(terrain, Terrain::Mountains | Terrain::Glacier) {
                        // Mountain surface varies by altitude + noise.
                        // Mountain peaks reach ~800-900 in current terrain,
                        // so snow line at ~700 with transition 600-700.
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xA1B2));
                        if vz > 750 {
                            VoxelMaterial::Snow
                        } else if vz > 600 {
                            // Transition: snow probability based on altitude
                            let snow_prob = ((vz - 600) as f32) / 150.0;
                            if n < snow_prob { VoxelMaterial::Snow }
                            else if n > 0.8 { VoxelMaterial::Gravel }
                            else { VoxelMaterial::Stone }
                        } else if n > 0.7 {
                            VoxelMaterial::Gravel
                        } else {
                            VoxelMaterial::Stone
                        }
                    } else if matches!(terrain, Terrain::Jungle) {
                        // Jungle floor: dense undergrowth with mud, roots, and moss
                        let patch = noise::fbm_2d(vx as f32 * 0.06, vy as f32 * 0.06, seed.wrapping_add(0xBE_AF), 2, 2.0, 0.5);
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xD_00));
                        if patch > 0.65 {
                            VoxelMaterial::Clay // muddy patches
                        } else if n > 0.9 {
                            VoxelMaterial::WoodLog // fallen branches/roots
                        } else if n > 0.8 {
                            VoxelMaterial::Dirt // exposed soil
                        } else {
                            VoxelMaterial::JungleMoss
                        }
                    } else if matches!(terrain, Terrain::Desert) {
                        // Desert: dune ridge patterns — directional wave using rotated coords
                        let angle = 0.7f32; // dune orientation
                        let rx = vx as f32 * angle.cos() + vy as f32 * angle.sin();
                        let dune = ((rx * 0.08).sin() * 0.5 + 0.5) as f32;
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xD0E5));
                        if dune > 0.7 {
                            VoxelMaterial::Sandstone // exposed harder layer at dune crests
                        } else if n > 0.9 {
                            VoxelMaterial::Gravel // scattered pebbles
                        } else {
                            VoxelMaterial::Sand
                        }
                    } else if matches!(terrain, Terrain::Forest) {
                        // Forest floor: mostly grass with occasional leaf litter (peat).
                        // Dirt patches are rare so the forest reads as predominantly green.
                        let patch = noise::fbm_2d(vx as f32 * 0.04, vy as f32 * 0.04, seed.wrapping_add(0xF0E5), 2, 2.0, 0.5);
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xF100));
                        if patch > 0.78 {
                            VoxelMaterial::Peat // dark leaf litter
                        } else if n > 0.95 {
                            VoxelMaterial::WoodLog // rare fallen branch
                        } else {
                            VoxelMaterial::Grass
                        }
                    } else if matches!(terrain, Terrain::Plains) {
                        // Plains: patches of regular grass mixed into tall grass
                        let patch = noise::fbm_2d(vx as f32 * 0.04, vy as f32 * 0.04, seed.wrapping_add(0xF1E1D), 2, 2.0, 0.5);
                        if patch > 0.6 {
                            VoxelMaterial::Grass // darker green patches
                        } else {
                            VoxelMaterial::TallGrass
                        }
                    } else if matches!(terrain, Terrain::Badlands) {
                        // Badlands: layered sediment bands visible on surface
                        // Use vz to create horizontal stripes of different materials
                        let band = ((vz as f32 * 0.3).sin() * 0.5 + 0.5) as f32;
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xBAD1));
                        if band > 0.7 {
                            VoxelMaterial::Sandstone
                        } else if band > 0.4 && n > 0.5 {
                            VoxelMaterial::Clay
                        } else {
                            VoxelMaterial::RedSand
                        }
                    } else if matches!(terrain, Terrain::Tundra) {
                        // Tundra: patchy snow over frozen peat (peat-dominant)
                        let n = noise::hash_f32(vx, vy, vz, seed.wrapping_add(0xC01D));
                        // Larger scale + higher threshold = smaller, sparser snow patches
                        let patch = noise::fbm_2d(vx as f32 * 0.015, vy as f32 * 0.015, seed.wrapping_add(0xF_0_2), 3, 2.0, 0.5);
                        if patch > 0.68 {
                            VoxelMaterial::Snow
                        } else if patch > 0.6 && n > 0.6 {
                            VoxelMaterial::Ice
                        } else if n > 0.85 {
                            VoxelMaterial::Gravel
                        } else {
                            VoxelMaterial::Peat
                        }
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

    // 0. Swamp water pooling — clustered ponds in low spots.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        if matches!(cell.terrain, Terrain::Swamp) {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let vx = base_x + lx as i32;
                    let vy = base_y + ly as i32;
                    let col_surface = surface_height_at(vx as f32, vy as f32, plan, seed);
                    // Use larger-scale noise to cluster ponds (15% coverage in clusters)
                    let pond_field = noise::fbm_2d(
                        vx as f32 * 0.012, vy as f32 * 0.012,
                        seed.wrapping_add(0x5A_A_0), 2, 2.0, 0.5,
                    );
                    if pond_field > 0.65 {
                        // Inside a pond cluster — fill 2-5 voxels deep
                        for dz in 1..=4 {
                            let wz = col_surface + dz;
                            let lz = wz - base_z;
                            if lz >= 0 && lz < CHUNK_SIZE as i32 {
                                let idx = local_index(lx, ly, lz as usize);
                                if chunk.voxels[idx].material == VoxelMaterial::Air {
                                    chunk.voxels[idx] = Voxel::new(VoxelMaterial::Water);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 1. Cave carving — for chunks that are predominantly underground.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let cz = base_z + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        let surface_z_centre = surface_height_at(cx as f32, cy as f32, plan, seed);
        let depth_below = surface_z_centre - cz;
        if depth_below > 20 {
            let bv = resolve_biome(cell.terrain, cell.sub_biome, depth_below, seed);
            caves::carve_caves(&mut chunk, cp, bv.underground, seed);
        }
    }

    // 2. River carving — for all rivers in the plan.
    for river in &plan.rivers {
        let plan_ref: &RegionPlan = plan;
        let seed_inner = seed;
        let surface_z_fn = |vx: f32, vy: f32| -> i32 {
            surface_height_at(vx, vy, plan_ref, seed_inner)
        };
        rivers::carve_river_in_chunk(&mut chunk, cp, river, &surface_z_fn);
    }

    // 3. Surface features — check each column's surface_z and see if it falls
    //    in this chunk.
    {
        let cx = base_x + CHUNK_SIZE as i32 / 2;
        let cy = base_y + CHUNK_SIZE as i32 / 2;
        let (cell, _, _) = plan.sample(cx as f32, cy as f32);
        let surface_z = surface_height_at(cx as f32, cy as f32, plan, seed);
        let surface_z_local = surface_z - base_z;
        features::place_surface_features(&mut chunk, cp, cell.terrain, cell.sub_biome, surface_z_local, seed, Some(plan), clearing_center);
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
    use crate::world_sim::terrain::region_plan::generate_continent;
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
            let chunk = materialize_chunk(ChunkPos::new(cx, cy, cz + dz), &plan, 42, None);
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
        let chunk = materialize_chunk(ChunkPos::new(5, 5, -5), &plan, 42, None);
        let solid_count = chunk.voxels.iter().filter(|v| v.material.is_solid()).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert!(solid_count > total / 2, "deep chunk not mostly solid: {solid_count}/{total}");
    }

    #[test]
    fn chunk_high_sky_is_air() {
        let plan = test_plan();
        // z=30 in chunk-space → voxel z = 30 * 16 = 480, well above MAX_SURFACE_Z (400)
        let chunk = materialize_chunk(ChunkPos::new(5, 5, 30), &plan, 42, None);
        let air_count = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Air).count();
        let total = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        assert_eq!(air_count, total, "sky chunk is not all air");
    }

    #[test]
    fn different_biomes_produce_different_surfaces() {
        let plan = test_plan();
        let plains_idx = plan.cells.iter().position(|c| c.terrain == Terrain::Plains && c.height > 0.2);
        let other_idx = plan.cells.iter().position(|c| {
            c.height > 0.2
                && c.terrain != Terrain::Plains
                && c.terrain != Terrain::DeepOcean
                && c.terrain != Terrain::Coast
        });
        if plains_idx.is_none() || other_idx.is_none() {
            return;
        }
        let pi = plains_idx.unwrap();
        let oi = other_idx.unwrap();
        let pc = (pi % plan.cols, pi / plan.cols);
        let oc = (oi % plan.cols, oi / plan.cols);
        use crate::world_sim::terrain::region_plan::CELL_SIZE;
        // Use actual surface height for each biome cell
        let vx_a = pc.0 as f32 * CELL_SIZE as f32 + 8.0;
        let vy_a = pc.1 as f32 * CELL_SIZE as f32 + 8.0;
        let vx_b = oc.0 as f32 * CELL_SIZE as f32 + 8.0;
        let vy_b = oc.1 as f32 * CELL_SIZE as f32 + 8.0;
        let sz_a = surface_height_at(vx_a, vy_a, &plan, 42);
        let sz_b = surface_height_at(vx_b, vy_b, &plan, 42);
        let cz_a = sz_a / CHUNK_SIZE as i32;
        let cz_b = sz_b / CHUNK_SIZE as i32;
        let cx_a = vx_a as i32 / CHUNK_SIZE as i32;
        let cy_a = vy_a as i32 / CHUNK_SIZE as i32;
        let cx_b = vx_b as i32 / CHUNK_SIZE as i32;
        let cy_b = vy_b as i32 / CHUNK_SIZE as i32;
        let chunk_a = materialize_chunk(ChunkPos::new(cx_a, cy_a, cz_a), &plan, 42, None);
        let chunk_b = materialize_chunk(ChunkPos::new(cx_b, cy_b, cz_b), &plan, 42, None);
        let diffs = chunk_a.voxels.iter()
            .zip(chunk_b.voxels.iter())
            .filter(|(a, b)| a.material != b.material)
            .count();
        assert!(diffs > 0, "different biomes produced identical chunks");
    }

    /// Sanity check: land biome chunks should NOT be majority water.
    #[test]
    fn land_biome_not_flooded() {
        let plan = generate_continent(20, 20, 42);
        let land_biomes = [Terrain::Plains, Terrain::Forest, Terrain::Desert,
                           Terrain::Mountains, Terrain::Tundra, Terrain::Jungle,
                           Terrain::Badlands, Terrain::Caverns];

        for cell in &plan.cells {
            if !land_biomes.contains(&cell.terrain) { continue; }
            if cell.height < 0.15 { continue; } // skip very low cells

            // Generate a chunk at this cell's surface
            let col = plan.cells.iter().position(|c| std::ptr::eq(c, cell)).unwrap();
            let c = col % plan.cols;
            let r = col / plan.cols;
            let vx = c as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE + 8;
            let vy = r as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE + 8;
            let surface_z = (cell.height * MAX_SURFACE_Z as f32) as i32;
            let cz = surface_z / CHUNK_SIZE as i32;
            let cx = vx / CHUNK_SIZE as i32;
            let cy = vy / CHUNK_SIZE as i32;

            let chunk = materialize_chunk(ChunkPos::new(cx, cy, cz), &plan, 42, None);
            let water = chunk.voxels.iter().filter(|v| v.material == VoxelMaterial::Water).count();
            let total = chunk.voxels.len();
            let water_pct = water as f32 / total as f32;

            assert!(water_pct < 0.5,
                "{:?} at ({},{}) height={:.2} has {:.0}% water",
                cell.terrain, c, r, cell.height, water_pct * 100.0);

            break; // one check per biome type is enough
        }
    }

    /// Sanity check: material variety across biomes.
    #[test]
    fn biomes_produce_distinct_materials() {
        let plan = generate_continent(30, 30, 42);

        // Collect surface material for several biome types
        let mut biome_surface: std::collections::HashMap<String, u8> = std::collections::HashMap::new();

        for (i, cell) in plan.cells.iter().enumerate() {
            if cell.height < 0.15 { continue; }
            let key = format!("{:?}", cell.terrain);
            if biome_surface.contains_key(&key) { continue; }

            let c = i % plan.cols;
            let r = i / plan.cols;
            let vx = c as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE + 8;
            let vy = r as i32 * crate::world_sim::terrain::region_plan::CELL_SIZE + 8;
            let surface_z = (cell.height * MAX_SURFACE_Z as f32) as i32;
            let cx = vx / CHUNK_SIZE as i32;
            let cy = vy / CHUNK_SIZE as i32;
            let cz = surface_z / CHUNK_SIZE as i32;

            let chunk = materialize_chunk(ChunkPos::new(cx, cy, cz), &plan, 42, None);
            // Find the most common non-air material (only check valid material indices)
            let mut counts = [0u32; 46]; // VoxelMaterial has ~42 variants
            for v in &chunk.voxels {
                let idx = v.material as u8 as usize;
                if idx > 0 && idx < counts.len() {
                    counts[idx] += 1;
                }
            }
            if let Some((mat_idx, &count)) = counts.iter().enumerate().max_by_key(|(_, &c)| c) {
                if count > 0 {
                    biome_surface.insert(key, mat_idx as u8);
                }
            }
        }

        // We should have at least 3 distinct surface materials across biomes
        let unique_mats: std::collections::HashSet<u8> = biome_surface.values().copied().collect();
        assert!(unique_mats.len() >= 3,
            "only {} distinct surface materials across {} biomes: {:?}",
            unique_mats.len(), biome_surface.len(), biome_surface);
    }

    #[test]
    fn materialization_is_deterministic() {
        let plan = test_plan();
        let cp = ChunkPos::new(3, 3, 2);
        let a = materialize_chunk(cp, &plan, 42, None);
        let b = materialize_chunk(cp, &plan, 42, None);
        for i in 0..a.voxels.len() {
            assert_eq!(a.voxels[i].material, b.voxels[i].material, "voxel {i} differs");
        }
    }
}
