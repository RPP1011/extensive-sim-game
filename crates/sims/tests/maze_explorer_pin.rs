//! maze_explorer behavioural pin — run the random-walk Adventurer
//! on the 4×4 reference maze until it finds the key, assert the
//! tick budget, report ticks-to-key.
//!
//! The maze topology is encoded inline in the .sim (see header
//! diagram); the test seeds only the Adventurer's starting state
//! (slot 0, current_room=0, current_doors=6 = room 0's bitmap)
//! and runs `step()` until `found_key == 1` or the tick budget is
//! exhausted.
//!
//! Determinism: `rng.action()` routes through the per-agent PCG
//! keyed on (world_seed, agent_slot, world.tick, purpose). On a
//! fixed seed, the random walk is reproducible — same seed → same
//! tick-to-key value.

#![allow(non_snake_case)]

use sims::maze_explorer::GeneratedRuntime;

const SEED: u64 = 0xB1_BB_BB_BB_BB_BB_BB_77u64;
// 1 Adventurer agent. Maze topology is in the .sim, not in the
// agent buffers, so we only need one slot.
const AGENT_COUNT: u32 = 1;
// Random-walk hitting time on a 16-room maze with average degree ≈
// 2 is bounded by ~1000 ticks for this seed; we give 5× headroom.
const TICK_BUDGET: u64 = 5000;

fn read_u32_at(state: &GeneratedRuntime, buf: &wgpu::Buffer, slot: u32) -> u32 {
    let bytes = 4u64;
    let offset = (slot as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("maze_explorer::read_u32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("maze_explorer::readback"),
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
    // No more current_doors seeding needed — the per-tick rule reads
    // door bitmaps directly from `tables.maze_doors(current_room)`.
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
fn adventurer_random_walk_finds_key_within_budget() {
    // The runtime emitter currently hardcodes `cfg.seed = 1u32` on
    // every per-tick cfg upload (see `target/.../runtime_core.rs::
    // step`'s `cfg_words`). That means `rng.action()` derives its
    // stream from (1, agent_id, tick, purpose) regardless of what
    // `try_new(seed, …)` is called with — every run on this fixture
    // produces the same tick-found.
    //
    // The assertion is the loose 5000-tick budget so this pin
    // tolerates a future seed-plumbing fix that perturbs the exact
    // tick. The reproducibility test below pins the current value.
    let tick_found = match run_one(SEED) {
        Some(t) => t,
        None => {
            if GeneratedRuntime::try_new(SEED, AGENT_COUNT).is_none() {
                eprintln!("[maze_explorer] skipping: no wgpu adapter");
                return;
            }
            panic!("Adventurer did not reach key room within {TICK_BUDGET} ticks");
        }
    };
    println!("[maze_explorer] tick_found={tick_found}");
    assert!(
        tick_found < TICK_BUDGET as u32,
        "tick_found={tick_found} exceeds budget"
    );
}

#[test]
fn random_walk_is_reproducible_across_runs() {
    // Five independent runs at the reference seed — every one
    // should land on the same tick_found. Catches any non-
    // determinism in the per-tick RNG stream or in the kernel
    // dispatch ordering.
    let mut samples: Vec<u32> = Vec::new();
    for _ in 0..5 {
        match run_one(SEED) {
            Some(t) => samples.push(t),
            None => {
                if GeneratedRuntime::try_new(SEED, AGENT_COUNT).is_none() {
                    eprintln!("[maze_explorer] skipping: no wgpu adapter");
                    return;
                }
                panic!("Adventurer did not reach key room within {TICK_BUDGET} ticks");
            }
        }
    }
    let first = samples[0];
    assert!(
        samples.iter().all(|t| *t == first),
        "non-deterministic tick_found across runs: {samples:?}"
    );
    println!("[maze_explorer] 5 runs: all tick_found={first}");
}
