//! Plan I.6 pattern pin — single-key bitmap with party-wide
//! AllyDied gossip via the social-merge primitive.
//!
//! Stand-in for dungeon_horde's host-side `hero_known_rooms: [u64;
//! 5]`: when a hero dies, every surviving party member inherits
//! the dying hero's room-knowledge bitmap. Today the pattern lives
//! in `viewer_runtime/src/lib.rs` as a host-side u64 OR-loop; this
//! pin shows the same gossip semantics expressed in 4 lines of DSL
//! and verified end-to-end on GPU.

#![allow(non_snake_case)]

use sims::room_known_pattern_probe::GeneratedRuntime;

const SEED: u64 = 0xC0DE_FACE_BABE_F00D;
const N: u32 = 5; // mirrors dungeon_horde's N_HEROES

fn read_view_storage(state: &mut GeneratedRuntime) -> Vec<u32> {
    let n = state.agent_count as usize;
    // Single-key shape — N cells (not N²).
    let bytes = (n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("room_known_pattern_probe::staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        state
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("room_known_pattern_probe::readback"),
            });
    encoder.copy_buffer_to_buffer(
        &state.view_storage_room_known_primary_buf,
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
        words[..n].to_vec()
    };
    staging.unmap();
    storage
}

#[test]
fn ally_died_propagates_room_knowledge_to_party() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[room_known_pattern_probe] skipping: no wgpu adapter");
            return;
        }
    };

    let n = state.agent_count as usize;

    // Seed all heroes alive.
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    // Pre-seed hero 0's known rooms = bits {0, 1, 2}.
    // Heroes 1-4 start with empty bitmaps.
    let initial_bits: u32 = 0b111; // rooms 0, 1, 2
    let mut storage_init: Vec<u32> = vec![0u32; n];
    storage_init[0] = initial_bits;
    state.gpu.queue.write_buffer(
        &state.view_storage_room_known_primary_buf,
        0,
        bytemuck::cast_slice(&storage_init),
    );

    // Read pre-seed back to verify the write landed.
    let pre = read_view_storage(&mut state);
    println!("[room_known_pattern_probe] pre-step storage: {pre:?}");

    // Step once. StampAllyDied emits AllyDied{dead:0}; the merge
    // kernel propagates hero 0's bitmap to every receiver.
    state.step();

    let storage = read_view_storage(&mut state);

    println!("[room_known_pattern_probe] post-AllyDied storage:");
    for r in 0..n {
        println!("  hero {r}: 0x{:X} (rooms = {:b})", storage[r], storage[r]);
    }

    // Pin: every alive hero now has bits {0, 1, 2} set —
    // hero 0 retains them (idempotent merge into self), heroes
    // 1-4 inherit them via the bit_or social merge.
    for receiver in 0..n {
        assert_eq!(
            storage[receiver],
            initial_bits,
            "hero {receiver} should know rooms {initial_bits:b}; got 0x{:X}",
            storage[receiver],
        );
    }

    println!(
        "[room_known_pattern_probe] PASS: all {n} heroes know rooms {{0, 1, 2}} \
         via social-merge gossip"
    );
}
