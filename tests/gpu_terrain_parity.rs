//! Parity test: GPU compute produces the same materials as CPU materialize_chunk.
//! In Task 7 (this baseline) we expect significant mismatches because the GPU
//! shader doesn't yet sample the region plan — it hardcodes base_height=0.3.
//! Task 8 will reduce mismatches to <5%.

#![cfg(feature = "app")]

use game::world_sim::terrain::{generate_continent, materialize::materialize_chunk};
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
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    let plan = generate_continent(60, 40, 42);
    // Pick a chunk near the surface for any biome — log the result for review.
    // Use a chunk where the CPU surface is around vz=600 (matches GPU hardcoded base).
    let cp = ChunkPos::new(0, 0, 9);
    let cpu_chunk = materialize_chunk(cp, &plan, 42);
    let gpu_mats = pipeline.generate_chunk(&ctx, [cp.x, cp.y, cp.z], 42u32).expect("dispatch");

    let cs = CHUNK_SIZE;
    assert_eq!(gpu_mats.len(), cs * cs * cs);

    let mut mismatches = 0usize;
    let mut cpu_air = 0;
    let mut gpu_air = 0;
    for lz in 0..cs {
        for ly in 0..cs {
            for lx in 0..cs {
                let cpu_mat = cpu_chunk.get(lx, ly, lz).material as u8;
                let gpu_mat = gpu_mats[lz * cs * cs + ly * cs + lx];
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
    eprintln!("BASELINE PARITY: {}/{} mismatches ({:.1}%)", mismatches, total, pct);
    eprintln!("CPU air voxels: {}, GPU air voxels: {}", cpu_air, gpu_air);

    // Phase 7 baseline: GPU has no region plan, so significant mismatches expected.
    // After Task 8 this should drop to <5%, after Task 9 to <1%.
    assert!(mismatches < total, "all voxels mismatched — pipeline broken");
    pipeline.destroy(&ctx, &mut alloc);
}
