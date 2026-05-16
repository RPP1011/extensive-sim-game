//! maze_explorer_visited behavioural pin — verifies the
//! belief-emitting variant runs end-to-end + records baseline
//! tick_found for comparison against `maze_explorer_pin` (the
//! random-walk-only fixture). Also reports the visited belief's
//! storage at termination so we can confirm the propagation
//! handler actually latched bits.
//!
//! Same agent count (1), same maze topology, same RNG keying →
//! the bare-move policy will produce the SAME tick_found as
//! `maze_explorer_pin` (198) because we haven't gated the move on
//! `visited` yet. The difference will surface in the perf bench:
//! this fixture pays for an extra event emit + the fold_visited
//! kernel dispatch every tick.

#![allow(non_snake_case)]

use sims::maze_explorer_visited::GeneratedRuntime;

const SEED: u64 = 0xB1_BB_BB_BB_BB_BB_BB_77u64;
const AGENT_COUNT: u32 = 1;
const TICK_BUDGET: u64 = 5000;
const K: u32 = 16;

fn read_u32_at(state: &GeneratedRuntime, buf: &wgpu::Buffer, slot: u32) -> u32 {
    let bytes = 4u64;
    let offset = (slot as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("maze_explorer_visited::read_u32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("maze_explorer_visited::readback"),
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

fn read_visited_row(state: &GeneratedRuntime) -> Vec<u32> {
    let bytes = (K as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("maze_explorer_visited::visited_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("maze_explorer_visited::visited_readback"),
        });
    encoder.copy_buffer_to_buffer(
        &state.view_storage_visited_primary_buf,
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
        words[..K as usize].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn visited_belief_propagates_room_bits_during_exploration() {
    let mut state = match GeneratedRuntime::try_new(SEED, AGENT_COUNT) {
        Some(s) => s,
        None => {
            eprintln!("[maze_explorer_visited] skipping: no wgpu adapter");
            return;
        }
    };

    let alive: [u32; 1] = [1];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    let mut ticks = 0u64;
    let mut found = false;
    while ticks < TICK_BUDGET {
        state.step();
        ticks += 1;
        let fk = read_u32_at(&state, &state.agent_found_key_buf, 0);
        if fk == 1 {
            found = true;
            break;
        }
    }

    let final_room = read_u32_at(&state, &state.agent_current_room_buf, 0);
    let tick_found = read_u32_at(&state, &state.agent_tick_found_buf, 0);
    let visited = read_visited_row(&state);
    let visited_count: usize = visited.iter().filter(|v| **v != 0).count();
    let visited_bits: Vec<usize> = visited
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(i, _)| i)
        .collect();

    println!(
        "[maze_explorer_visited] ticks={ticks}, found={found}, \
         current_room={final_room}, tick_found={tick_found}"
    );
    println!(
        "[maze_explorer_visited] visited cells with bit set: {visited_count}/16, \
         rooms: {visited_bits:?}"
    );

    assert!(found, "did not reach key room within {TICK_BUDGET} ticks");
    assert_eq!(final_room, 15);
    // The propagation handler should have latched the bit for at
    // least the key room (15) and the spawn room (0). With ticks
    // around the random-walk hitting time, we expect most reachable
    // rooms (15 of 16 — room 10 is reachable via 14→10) to be hit
    // before termination. A loose floor of 4 catches a totally
    // dead propagation handler (where no bits get set at all).
    assert!(
        visited_count >= 4,
        "visited belief saw only {visited_count} rooms — propagation handler is dead?"
    );
}
