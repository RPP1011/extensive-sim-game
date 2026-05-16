//! Plan I.4b runtime e2e for max/min/replace merge ops.
//!
//! The bit_or op is covered by `room_known_pattern_probe_pin` and
//! `belief_merge_propagation_probe_pin`. This pin completes the
//! 4-op coverage by exercising the remaining three on GPU.
//!
//! Each op gets its own belief decl + AllyDied event in
//! `belief_merge_ops_probe.sim`. Pre-seed agent 0's cell with a
//! known value, fire AllyDied, verify each receiver picks up the
//! op-correct merge result.

#![allow(non_snake_case)]

use sims::belief_merge_ops_probe::GeneratedRuntime;

const SEED: u64 = 0xB0_FF5_DEAD_BEEF;
const N: u32 = 4;

/// Take agent_count + gpu by ref (no &mut state) so the buffer can
/// be passed alongside without borrow conflicts.
fn read_buf(
    gpu: &engine::GpuContext,
    agent_count: u32,
    buf: &wgpu::Buffer,
) -> Vec<u32> {
    let n = agent_count as usize;
    let bytes = (n as u64 * 4u64).max(16);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("belief_merge_ops_probe::staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("belief_merge_ops_probe::readback"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("map_async failed")
    });
    gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let storage: Vec<u32> = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn max_min_replace_merge_ops_propagate_correctly_on_gpu() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[belief_merge_ops_probe] skipping: no wgpu adapter");
            return;
        }
    };
    let n = state.agent_count as usize;

    // Seed all agents alive.
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    // Pre-seed each belief's storage. For each op, agent 0 starts
    // with a known seed; agents 1..N start at variable values that
    // the op will mutate differently.
    //
    // max:     source=100, receivers=[0, 50, 200, 30]
    //          → expected: [100, 100, 200, 100] (max-merge: 0+source=100, 50+100=100, 200+100=200, 30+100=100)
    // min:     source=50,  receivers=[0xFFFFFFFF, 100, 30, 60]
    //          → expected: [50, 50, 30, 50] (min: trickier — source pre-seeded as 50, others get min(self, 50))
    //          Wait — initial=0 and we pre-seed receivers[0]=50. Receiver
    //          1's seed = 100 → min(100, 50) = 50. Receiver 2's seed = 30 →
    //          min(30, 50) = 30. Receiver 3's seed = 60 → min(60, 50) = 50.
    //          → expected: [50, 50, 30, 50]
    // replace: source=42, receivers=[any]
    //          → expected: [42, 42, 42, 42] (overwrite)
    let max_seed = vec![100u32, 50, 200, 30];
    let min_seed = vec![50u32, 100, 30, 60];
    let repl_seed = vec![42u32, 99, 11, 77];

    state.gpu.queue.write_buffer(
        &state.view_storage_maxv_primary_buf,
        0,
        bytemuck::cast_slice(&max_seed),
    );
    state.gpu.queue.write_buffer(
        &state.view_storage_minv_primary_buf,
        0,
        bytemuck::cast_slice(&min_seed),
    );
    state.gpu.queue.write_buffer(
        &state.view_storage_replv_primary_buf,
        0,
        bytemuck::cast_slice(&repl_seed),
    );

    // Step once. Each StampX emits an AllyDiedX event from every
    // alive agent (4 events per op = 12 total). Each merge kernel
    // filters by its own kind and applies its op.
    state.step();

    let max_post = read_buf(&state.gpu, state.agent_count, &state.view_storage_maxv_primary_buf);
    let min_post = read_buf(&state.gpu, state.agent_count, &state.view_storage_minv_primary_buf);
    let repl_post = read_buf(&state.gpu, state.agent_count, &state.view_storage_replv_primary_buf);

    println!("[belief_merge_ops_probe]");
    println!("  max post:     {max_post:?}  (seed [100,50,200,30] + max(_, 100) → expect [100,100,200,100])");
    println!("  min post:     {min_post:?}  (seed [50,100,30,60] + min(_, 50)  → expect [50,50,30,50])");
    println!("  replace post: {repl_post:?}  (seed [42,99,11,77] + replace(42) → expect [42,42,42,42])");

    // max — every receiver picks up max(self, 100) = max(self, 100).
    // Agent 0: max(100, 100) = 100. Agent 1: max(50, 100) = 100.
    // Agent 2: max(200, 100) = 200. Agent 3: max(30, 100) = 100.
    assert_eq!(max_post, vec![100, 100, 200, 100], "max merge mismatch");

    // min — every receiver picks up min(self, 50).
    assert_eq!(min_post, vec![50, 50, 30, 50], "min merge mismatch");

    // replace — every receiver overwritten with source value 42.
    assert_eq!(repl_post, vec![42, 42, 42, 42], "replace merge mismatch");

    println!("[belief_merge_ops_probe] PASS: all 3 non-bit_or merge ops propagate correctly");
}
