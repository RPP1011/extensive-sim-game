//! maze_explorer_belief_smart pin — visited-aware Adventurer
//! where the visited bitmap is backed by an I.3b key-typed belief
//! `(observer: Agent, room: u32) @key_pop(K=16)` and READ from a
//! physics rule body via the pair-keyed view-read prelude shipped
//! alongside.
//!
//! This is the first fixture to exercise the pair-keyed
//! `view_<id>_get(observer, key)` helper end-to-end. Comparison
//! anchor: maze_explorer_smart (SoA-field + room_bit table) hit
//! median 128 / mean 263 over 16 seeds. This fixture should land
//! in a comparable range — the policy is identical, only the
//! visited substrate differs.

#![allow(non_snake_case)]

use sims::maze_explorer_belief_smart::GeneratedRuntime;

const SEED: u64 = 0xB1_BB_BB_BB_BB_BB_BB_77u64;
const AGENT_COUNT: u32 = 1;
const TICK_BUDGET: u64 = 5000;

fn read_u32_at(state: &GeneratedRuntime, buf: &wgpu::Buffer, slot: u32) -> u32 {
    let bytes = 4u64;
    let offset = (slot as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("belief_smart::read_u32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("belief_smart::readback"),
        });
    encoder.copy_buffer_to_buffer(buf, offset, &staging, 0, bytes);
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
    let value = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[0]
    };
    staging.unmap();
    value
}

fn run_one(seed: u64) -> Option<u32> {
    let mut state = GeneratedRuntime::try_new(seed, AGENT_COUNT)?;
    let alive: [u32; 1] = [1];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    let mut ticks = 0u64;
    while ticks < TICK_BUDGET {
        state.step();
        ticks += 1;
        let fk = read_u32_at(&state, &state.agent_found_key_buf, 0);
        if fk == 1 {
            return Some(read_u32_at(&state, &state.agent_tick_found_buf, 0));
        }
    }
    None
}

#[test]
fn belief_smart_walker_finds_key_via_pair_keyed_view_read() {
    let tick_found = match run_one(SEED) {
        Some(t) => t,
        None => {
            if GeneratedRuntime::try_new(SEED, AGENT_COUNT).is_none() {
                eprintln!("[belief_smart] skipping: no wgpu adapter");
                return;
            }
            panic!(
                "belief-smart walker did not reach key within {TICK_BUDGET} ticks — \
                 pair-keyed view-read may be broken"
            );
        }
    };
    println!("[belief_smart] tick_found={tick_found}");
    assert!(tick_found < TICK_BUDGET as u32);
}

#[test]
fn belief_smart_distribution_matches_soa_smart() {
    // 16-seed sweep — compare against maze_explorer_smart's
    // SoA-backed visited-aware policy: median 128, mean 263. The
    // belief-backed variant runs the same policy via a different
    // substrate (pair-keyed view-read instead of SoA-field read).
    // Distributions should be in the same ballpark.
    let seeds: Vec<u64> = (0..16u64).map(|i| SEED.wrapping_add(i * 0xDEADBEEFu64)).collect();
    let mut samples: Vec<u32> = Vec::new();
    let mut timeouts = 0usize;
    let mut skipped = false;
    for s in &seeds {
        match run_one(*s) {
            Some(t) => samples.push(t),
            None => {
                if GeneratedRuntime::try_new(*s, AGENT_COUNT).is_none() {
                    eprintln!("[belief_smart] skipping: no wgpu adapter");
                    skipped = true;
                    break;
                }
                timeouts += 1;
            }
        }
    }
    if skipped {
        return;
    }
    samples.sort();
    let min = samples.first().copied().unwrap_or(0);
    let max = samples.last().copied().unwrap_or(0);
    let median = samples.get(samples.len() / 2).copied().unwrap_or(0);
    let mean = samples.iter().map(|t| *t as f64).sum::<f64>() / samples.len() as f64;
    println!(
        "[belief_smart] 16 seeds: finished={}, timeouts={timeouts}, \
         min={min}, median={median}, max={max}, mean={mean:.1}",
        samples.len(),
    );
    assert!(timeouts == 0, "{timeouts}/16 seeds timed out");
    assert!(
        median < 300,
        "belief-smart median {median} is too high — the pair-keyed view-read may be returning stale or wrong values"
    );
}
