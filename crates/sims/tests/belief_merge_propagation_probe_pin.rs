//! Plan I.4b end-to-end pin — observes the social-merge kernel
//! propagating agent 0's belief row to every other agent's row when
//! an `AllyDied { dead: 0 }` event fires.
//!
//! Setup:
//!   * 4 alive agents.
//!   * Pre-seed agent 0's belief row host-side with a known bitmap.
//!   * Tick 0: physics_StampAllyDied fires per alive agent → emits
//!     AllyDied { dead: 0 } via the regular chronicle path.
//!   * Tick 0: merge_seen_ally_died_bit_or runs over the appended
//!     events; for each AllyDied it OR-merges agent 0's row into
//!     every receiver's row.
//!   * Read back: agent 1's row should contain agent 0's bits.
//!
//! This is the long-deferred end-to-end social-merge pin. The
//! `belief_smoke_probe` test ran the merge KERNEL and verified it
//! dispatched without GPU validation errors; this pin verifies the
//! kernel actually MUTATES storage in the documented way.

#![allow(non_snake_case)]

use sims::belief_merge_propagation_probe::GeneratedRuntime;

const SEED: u64 = 0xBE1E_F0_5A1_0019;
const N: u32 = 4;

fn read_view_storage(state: &mut GeneratedRuntime) -> Vec<u32> {
    let n = state.agent_count as usize;
    let bytes = (n as u64 * n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("belief_merge_propagation_probe::staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        state
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("belief_merge_propagation_probe::readback"),
            });
    encoder.copy_buffer_to_buffer(
        &state.view_storage_seen_primary_buf,
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
        words[..n * n].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn ally_died_propagates_belief_row_to_every_receiver() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[belief_merge_propagation_probe] skipping: no wgpu adapter");
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

    // Pre-seed agent 0's belief row with a known bitmap pattern.
    // After the AllyDied merge fires, every receiver should pick up
    // these bits in the matching column.
    let mut storage_init: Vec<u32> = vec![0u32; n * n];
    storage_init[0 * n + 0] = 0xC0DE; // agent 0, subject 0
    storage_init[0 * n + 1] = 0xBEEF; // agent 0, subject 1
    storage_init[0 * n + 2] = 0xF00D; // agent 0, subject 2
    storage_init[0 * n + 3] = 0xFACE; // agent 0, subject 3
    state.gpu.queue.write_buffer(
        &state.view_storage_seen_primary_buf,
        0,
        bytemuck::cast_slice(&storage_init),
    );

    // Step once. The StampAllyDied rule emits AllyDied{dead:0} per
    // alive agent (4 events). The merge kernel processes each one
    // and OR-merges agent 0's row into every receiver's row.
    state.step();

    let storage = read_view_storage(&mut state);

    println!("[belief_merge_propagation_probe] post-step storage:");
    for r in 0..n {
        let row: Vec<String> = (0..n)
            .map(|s| format!("0x{:X}", storage[r * n + s]))
            .collect();
        println!("  agent {r}: [{}]", row.join(", "));
    }

    // Agent 0's row should still contain the seeded bits (the merge
    // OR'd 0's row into 0's own row → idempotent under bit_or).
    assert_eq!(
        storage[0 * n + 0],
        0xC0DE,
        "agent 0's row should retain its seeded bits in column 0"
    );

    // Agent 1, 2, 3's rows should now contain the bits from agent 0
    // (the merge OR'd them in).
    for receiver in 1..n {
        for subject in 0..n {
            let receiver_cell = storage[receiver * n + subject];
            let source_cell = storage_init[0 * n + subject];
            assert_eq!(
                receiver_cell, source_cell,
                "agent {receiver}'s row should have inherited agent 0's bits in column {subject}; \
                 expected 0x{source_cell:X}, got 0x{receiver_cell:X}"
            );
        }
    }
}
