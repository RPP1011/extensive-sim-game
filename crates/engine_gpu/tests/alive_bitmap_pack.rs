// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! Correctness test for the per-tick alive bitmap pack kernel.
//!
//! Seeds a `GpuAgentSlot` array with a known alive pattern, dispatches
//! [`AlivePackKernel::encode_pack`], reads back the bitmap, and
//! verifies that bit `i` == `agents[i].alive != 0` for every slot.
//!
//! Also exercises naga parse: the WGSL source for the pack kernel and
//! the `alive_bit` helper are validated via wgpu's shader module
//! compilation at `AlivePackKernel::new` time — this test indirectly
//! asserts they compile by calling `new()`.

#![cfg(feature = "gpu")]

use engine_gpu::{
    alive_bitmap::{alive_bitmap_bytes, alive_bitmap_words, AlivePackKernel},
    physics::GpuAgentSlot,
    test_device,
};

/// Construct a fresh device/queue pair for the test.
fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
    test_device().expect("test_device()")
}

/// Slot builder — only the `alive` field matters for these tests;
/// everything else gets a zero-like placeholder.
fn slot(alive: bool) -> GpuAgentSlot {
    GpuAgentSlot {
        hp: 0.0,
        max_hp: 0.0,
        shield_hp: 0.0,
        attack_damage: 0.0,
        alive: if alive { 1 } else { 0 },
        creature_type: 0,
        engaged_with: GpuAgentSlot::ENGAGED_NONE,
        stun_expires_at: 0,
        slow_expires_at: 0,
        slow_factor_q8: 0,
        cooldown_next_ready: 0,
        pos_x: 0.0,
        pos_y: 0.0,
        pos_z: 0.0,
        _pad0: 0,
        _pad1: 0,
    }
}

/// Upload `slots` to a fresh agents buffer + run the pack kernel +
/// read back the bitmap. Returns the flat bitmap words.
fn run_pack(slots: &[GpuAgentSlot]) -> Vec<u32> {
    let (device, queue) = gpu_device_queue();
    let agent_cap = slots.len() as u32;

    let agents_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::agents"),
        size: (slots.len() as u64) * (std::mem::size_of::<GpuAgentSlot>() as u64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&agents_buf, 0, bytemuck::cast_slice(slots));

    let bitmap_buf = engine_gpu::alive_bitmap::create_alive_bitmap_buffer(&device, agent_cap);
    let bitmap_words = alive_bitmap_words(agent_cap) as usize;

    let mut kernel = AlivePackKernel::new(&device).expect("AlivePackKernel::new");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test::encoder"),
    });
    kernel.encode_pack(&device, &queue, &mut encoder, &agents_buf, &bitmap_buf, agent_cap);

    // Readback via an intermediate staging buffer.
    let readback_bytes = alive_bitmap_bytes(agent_cap);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::bitmap_readback"),
        size: readback_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&bitmap_buf, 0, &readback, 0, readback_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap().expect("map_async");
    let data = slice.get_mapped_range();
    let words: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data)[..bitmap_words].to_vec();
    drop(data);
    readback.unmap();
    words
}

fn cpu_expected_bitmap(slots: &[GpuAgentSlot]) -> Vec<u32> {
    let agent_cap = slots.len() as u32;
    let words = alive_bitmap_words(agent_cap) as usize;
    let mut packed = vec![0u32; words.max(1)];
    for (slot_idx, s) in slots.iter().enumerate() {
        if s.alive != 0 {
            packed[slot_idx >> 5] |= 1u32 << (slot_idx & 31);
        }
    }
    packed
}

#[test]
fn pack_all_alive() {
    let slots: Vec<GpuAgentSlot> = (0..100).map(|_| slot(true)).collect();
    let got = run_pack(&slots);
    let want = cpu_expected_bitmap(&slots);
    assert_eq!(got, want, "all-alive bitmap mismatch");
    // Sanity: 100 bits means 3 full words + 4 bits in the last — all set.
    assert_eq!(got[0], 0xFFFF_FFFF);
    assert_eq!(got[1], 0xFFFF_FFFF);
    assert_eq!(got[2], 0xFFFF_FFFF);
    assert_eq!(got[3] & 0xF, 0xF, "low 4 bits of word 3 set");
}

#[test]
fn pack_all_dead() {
    let slots: Vec<GpuAgentSlot> = (0..100).map(|_| slot(false)).collect();
    let got = run_pack(&slots);
    let want = cpu_expected_bitmap(&slots);
    assert_eq!(got, want, "all-dead bitmap mismatch");
    for &w in &got {
        assert_eq!(w, 0, "dead slots must leave zero bits");
    }
}

#[test]
fn pack_alternating() {
    // Every even slot alive, every odd slot dead.
    let slots: Vec<GpuAgentSlot> = (0..128).map(|i| slot(i % 2 == 0)).collect();
    let got = run_pack(&slots);
    let want = cpu_expected_bitmap(&slots);
    assert_eq!(got, want, "alternating pattern mismatch");
    // 0101...01 = 0x5555_5555 for 32 bits.
    assert_eq!(got[0], 0x5555_5555);
    assert_eq!(got[1], 0x5555_5555);
    assert_eq!(got[2], 0x5555_5555);
    assert_eq!(got[3], 0x5555_5555);
}

#[test]
fn pack_word_boundary() {
    // Exactly one slot set on each side of a word boundary.
    let mut slots: Vec<GpuAgentSlot> = (0..64).map(|_| slot(false)).collect();
    slots[31] = slot(true);
    slots[32] = slot(true);
    let got = run_pack(&slots);
    let want = cpu_expected_bitmap(&slots);
    assert_eq!(got, want, "word-boundary bitmap mismatch");
    assert_eq!(got[0], 1u32 << 31, "bit 31 set in word 0");
    assert_eq!(got[1], 1u32 << 0, "bit 0 set in word 1");
}

#[test]
fn pack_non_multiple_of_32() {
    // `agent_cap = 17` — not a multiple of 32. Last word has 17 bits
    // of valid data; the remaining 15 bits must stay clear.
    let slots: Vec<GpuAgentSlot> = (0..17).map(|i| slot(i < 10)).collect();
    let got = run_pack(&slots);
    let want = cpu_expected_bitmap(&slots);
    assert_eq!(got, want, "non-multiple-of-32 bitmap mismatch");
    // Low 10 bits set, bits 10..16 clear, bits 17..31 clear.
    assert_eq!(got[0], 0x0000_03FF);
}

#[test]
fn pack_is_idempotent() {
    // Running the pack kernel twice with the same input should
    // produce the same output (non-atomic write means no history).
    let slots: Vec<GpuAgentSlot> = (0..200).map(|i| slot(i % 3 == 0)).collect();
    let first = run_pack(&slots);
    let second = run_pack(&slots);
    assert_eq!(first, second, "pack kernel must be deterministic");
}

#[test]
fn naga_parse_smoke() {
    // `AlivePackKernel::new` runs wgpu's shader-module compilation
    // (which drives naga's WGSL parser). If the pack kernel's WGSL
    // source has a syntax / type error, this fails at init.
    let (device, _queue) = gpu_device_queue();
    let _ = AlivePackKernel::new(&device).expect("pack kernel compiles");
}
