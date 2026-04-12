//! Parity test: GPU compute produces the same materials as CPU materialize_chunk.
//! Task 8 uploads the region plan to GPU, so surface heights should match
//! bilinearly. Remaining mismatches come from unimplemented features
//! (caves, rivers, features, dungeons, post-passes) — Tasks 9-12 close those.

#![cfg(feature = "app")]

use game::world_sim::state::Terrain;
use game::world_sim::terrain::{
    generate_continent, materialize_chunk, surface_height_at, CELL_SIZE,
};
use game::world_sim::voxel::{ChunkPos, CHUNK_SIZE};

use voxel_engine::terrain_compute::TerrainComputePipeline;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

#[test]
fn gpu_cpu_parity_baseline() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let mut pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    let plan = generate_continent(60, 40, 42);
    let gpu_cells = plan.to_gpu_cells();
    pipeline
        .upload_region_plan(
            &ctx,
            &mut alloc,
            plan.cols as u32,
            plan.rows as u32,
            CELL_SIZE as u32,
            &gpu_cells,
        )
        .expect("upload region plan");

    let (river_points, river_headers) = plan.to_gpu_rivers();
    pipeline
        .upload_rivers(&ctx, &mut alloc, &river_points, &river_headers)
        .expect("upload rivers");

    // Find a land cell (not ocean/coast) that straddles the surface.
    // We want a chunk where the CPU surface intersects it so we're validating
    // more than just "all-air above ground" parity.
    // Only use Plains/Forest — these biomes have simple material layers that
    // mostly match the GPU shader's simplified output (dirt/stone/grass).
    // More complex biomes (badlands banding, jungle moss, tundra snow, etc.)
    // will need Task 9+ shader work to match.
    let land_biomes = [Terrain::Plains, Terrain::Forest];
    let (col, row) = (0..plan.rows).flat_map(|r| (0..plan.cols).map(move |c| (c, r)))
        .find(|(c, r)| {
            let cell = plan.get(*c, *r);
            cell.height > 0.2 && cell.height < 0.5 && land_biomes.contains(&cell.terrain)
        })
        .expect("no suitable land cell found");
    eprintln!("Using land cell ({col},{row}) terrain={:?} height={:.3}",
        plan.get(col, row).terrain, plan.get(col, row).height);

    let vx_mid = col as i32 * CELL_SIZE + CELL_SIZE / 2;
    let vy_mid = row as i32 * CELL_SIZE + CELL_SIZE / 2;
    let surface_z = surface_height_at(vx_mid as f32, vy_mid as f32, &plan, 42);
    let cx = vx_mid.div_euclid(CHUNK_SIZE as i32);
    let cy = vy_mid.div_euclid(CHUNK_SIZE as i32);
    let cz = surface_z.div_euclid(CHUNK_SIZE as i32);
    let cp = ChunkPos::new(cx, cy, cz);
    eprintln!("Chunk {cp:?}  surface_z={surface_z}");

    let cpu_chunk = materialize_chunk(cp, &plan, 42, None);
    let gpu_mats = pipeline.generate_chunk(&ctx, [cp.x, cp.y, cp.z], 42u32).expect("dispatch");

    let cs = CHUNK_SIZE;
    assert_eq!(gpu_mats.len(), cs * cs * cs);

    // Phase 3: the GPU shader now writes in engine-oriented (x, y-up, z)
    // layout so the renderer can sample chunks directly from the pool. Sim
    // coords are (x, y, z-up), so to compare with `cpu_chunk.get(lx, ly, lz)`
    // we index the GPU bytes at (lx, sim_z=lz, sim_y=ly) → linear index
    // `ly * cs * cs + lz * cs + lx`.
    let mut mismatches = 0usize;
    let mut cpu_air = 0;
    let mut gpu_air = 0;
    for lz in 0..cs {
        for ly in 0..cs {
            for lx in 0..cs {
                let cpu_mat = cpu_chunk.get(lx, ly, lz).material as u8;
                let gpu_mat = gpu_mats[ly * cs * cs + lz * cs + lx];
                if cpu_mat != gpu_mat {
                    mismatches += 1;
                }
                if cpu_mat == 0 { cpu_air += 1; }
                if gpu_mat == 0 { gpu_air += 1; }
            }
        }
    }
    let total = cs * cs * cs;
    let pct = mismatches as f32 / total as f32 * 100.0;
    eprintln!("TASK8 PARITY: {}/{} mismatches ({:.1}%)", mismatches, total, pct);
    eprintln!("CPU air voxels: {}, GPU air voxels: {}", cpu_air, gpu_air);

    // Task 9: full biome-aware material assignment ported to GPU shader.
    // Remaining mismatches come from surface-height float precision edges
    // and post-passes (caves, rivers, dungeons, features) that Tasks 10-12
    // will address.
    assert!(
        pct < 2.0,
        "parity regressed: {:.1}% mismatches (expected <2% after Task 9)",
        pct
    );
    pipeline.destroy(&ctx, &mut alloc);
}
