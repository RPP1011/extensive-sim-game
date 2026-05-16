//! Plan I slice I.3b end-to-end pin — observes the social-merge
//! kernel propagating a key-typed belief row (`@key_pop(K = 8)`)
//! across alive agents when an `AllyDied { dead: 0 }` event fires.
//!
//! Setup mirrors `belief_merge_propagation_probe_pin` but with the
//! second key as a `u32` room_id, not an `Agent`. Storage is
//! `N × 8` cells instead of `N × N`. Verifies:
//!   * The lowering accepts `(Agent, u32)` with `@key_pop(K = 8)`
//!     (previously rejected with `UnsupportedBeliefShape`).
//!   * The per-view allocator sizes the buffer as `N × 8 × 4` bytes.
//!   * The merge kernel uses the literal `8u` for the second-dim
//!     bound + indexing (`r * 8 + s` instead of `r * N + s`).
//!   * The OR-merge semantics propagate agent 0's per-room bitmaps
//!     into every receiver's row.

#![allow(non_snake_case)]

use sims::belief_key_typed_probe::GeneratedRuntime;

const SEED: u64 = 0xBE1E_F00D_5A1_00B1;
const N: u32 = 4;
const K: u32 = 8; // matches @key_pop(K = 8) in the .sim

fn read_view_storage(state: &mut GeneratedRuntime) -> Vec<u32> {
    let n = state.agent_count as usize;
    let bytes = (n as u64 * K as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("belief_key_typed_probe::staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("belief_key_typed_probe::readback"),
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
        words[..(n * K as usize)].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn key_typed_belief_propagates_per_room_bitmap_to_every_receiver() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[belief_key_typed_probe] skipping: no wgpu adapter");
            return;
        }
    };

    let n = state.agent_count as usize;

    // Seed all agents alive so the StampAllyDied physics rule fires.
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    // Pre-seed agent 0's per-room row with distinct bitmaps. After
    // the merge fires, every receiver should pick up these bits in
    // the matching room column (one row of K = 8 cells per agent).
    let row0: [u32; 8] = [0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x17, 0x28];
    let mut storage_init: Vec<u32> = vec![0u32; n * K as usize];
    for s in 0..K as usize {
        storage_init[0 * K as usize + s] = row0[s];
    }
    state.gpu.queue.write_buffer(
        &state.view_storage_seen_in_room_primary_buf,
        0,
        bytemuck::cast_slice(&storage_init),
    );

    // Step once. StampAllyDied emits AllyDied{dead:0} per alive
    // agent. The merge kernel processes each one and OR-merges
    // agent 0's row into every receiver's row.
    state.step();

    let storage = read_view_storage(&mut state);

    println!("[belief_key_typed_probe] post-step storage (K = {K}):");
    for r in 0..n {
        let row: Vec<String> = (0..K as usize)
            .map(|s| format!("0x{:02X}", storage[r * K as usize + s]))
            .collect();
        println!("  agent {r} rooms: [{}]", row.join(", "));
    }

    // Every agent's row should now equal agent 0's seeded row,
    // because the merge OR'd agent 0's row into every receiver's
    // row (idempotent under bit_or; agent 0's own row OR's with
    // itself and stays the same).
    for r in 0..n {
        for s in 0..K as usize {
            let got = storage[r * K as usize + s];
            assert_eq!(
                got, row0[s],
                "agent {r} room {s}: expected 0x{:X}, got 0x{:X}",
                row0[s], got
            );
        }
    }
}
