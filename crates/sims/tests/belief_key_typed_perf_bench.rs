//! Plan I slice I.3b perf bench — measures merge-kernel wall-clock
//! for the key-typed `(Agent, u32)` shape with `@key_pop(K = 8)` so
//! the scaling regime can be compared directly against
//! `belief_merge_perf_bench` (Agent×Agent at the same N).
//!
//! Storage is `N × 8` cells; dispatch is currently still
//! `(agent_cap, agent_cap)` per the shared PerAgentEventScan
//! shape (so wasted threads at s ≥ 8 bounds-check out early —
//! optimisation deferred to a follow-up that adds a K-aware
//! dispatch variant). The Storage-side win is real either way:
//! K = 8 means atomic ops touch one of 8 cells per receiver row
//! instead of one of N cells, so cache locality on the receiver's
//! row is dramatically better at large N.

#![allow(non_snake_case)]

use sims::belief_key_typed_probe::GeneratedRuntime;

const SEED: u64 = 0xBE_5C_EE_F0_0D_BA_BE_88;
const WARMUP_TICKS: usize = 2;
const MEASURE_TICKS: usize = 16;
const K: u32 = 8; // matches @key_pop(K = 8) in the .sim

fn run_at_n(n: u32) -> Option<(f64, f64)> {
    let mut state = GeneratedRuntime::try_new(SEED, n)?;
    let alive: Vec<u32> = vec![1u32; n as usize];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    for _ in 0..WARMUP_TICKS {
        state.step();
    }

    let mut samples_ms: Vec<f64> = Vec::with_capacity(MEASURE_TICKS);
    let total_start = std::time::Instant::now();
    for _ in 0..MEASURE_TICKS {
        let t = std::time::Instant::now();
        state.step();
        let _ = read_view_storage(&mut state, 1);
        samples_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mean = total_ms / MEASURE_TICKS as f64;
    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = sorted[((sorted.len() * 95) / 100).min(sorted.len() - 1)];
    Some((mean, p95))
}

fn read_view_storage(state: &mut GeneratedRuntime, n: usize) -> Vec<u32> {
    let bytes = (n as u64 * K as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf_bench::staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("perf_bench::readback"),
        });
    encoder.copy_buffer_to_buffer(
        &state.view_storage_seen_in_room_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let storage: Vec<u32> = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..(n * K as usize).min(words.len())].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn key_typed_merge_kernel_perf_at_varied_n() {
    let sizes = [
        4u32, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    ];
    println!("==== belief key-typed merge kernel perf bench ====");
    println!("  (Shape: (observer: Agent, room: u32), @key_pop(K = 8))");
    println!("  Storage: N × 8 × 4 bytes — flat in N for the second dim.");
    println!("  Warmup={WARMUP_TICKS} ticks, measured={MEASURE_TICKS} ticks per N");
    println!("  N      mean ms/tick    p95 ms/tick    storage_kb");
    for &n in &sizes {
        match run_at_n(n) {
            Some((mean, p95)) => {
                let storage_kb = ((n as u64) * K as u64 * 4) / 1024;
                println!(
                    "  {n:>4}   {mean:>9.3}      {p95:>9.3}      {storage_kb}"
                );
            }
            None => {
                eprintln!(
                    "[key_typed_perf_bench] N={n}: runtime init failed; skipping rest"
                );
                break;
            }
        }
    }
    println!("===================================================");
    println!(
        "  Note: kernel is functionally correct (see belief_key_typed_probe_pin).\n\
         \x20   Compare against belief_merge_perf_bench (Agent×Agent at same N):\n\
         \x20   • storage win: N × K vs N² → at N=8192, K=8: 256 KB vs 256 MB.\n\
         \x20   • dispatch shape is still (N, N) per the shared\n\
         \x20     PerAgentEventScan; threads at s ≥ 8 bounds-check out early.\n\
         \x20     A K-aware dispatch variant (deferred) would drop launched\n\
         \x20     threads to (N, K) for a true wall-clock win at large N."
    );
}
