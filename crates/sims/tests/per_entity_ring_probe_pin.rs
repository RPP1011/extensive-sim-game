//! `per_entity_ring_probe` pin — the load-bearing GPU proof that
//! `@per_entity_ring(K = N)` (G3a, scalar payload) actually works at
//! runtime, not just compiles. No test existed for this fixture before
//! this file; the fixture's own header comment even says "runtime not
//! built yet" — stale as of this session (the WGSL emit — a single
//! atomic cursor allocation plus a modulo-indexed write, see
//! `fold_recent_damages.wgsl` — is real and, per this test, correct),
//! but nothing had confirmed it end to end on real GPU hardware before
//! this.
//!
//! Events emitted on tick T are folded on tick T+1, so the first
//! `step()` appends nothing and, with `inject_base=10, inject_step=10`,
//! the N-th append carries the tick-(N-1) amount `10*N`: 10, 20, 30, 40,
//! 50, 60, 70, ... Two checkpoints: after exactly 4 appends the
//! ring is exactly full with no wrap yet; after exactly 7, three slots
//! have wrapped once. Agent 1 receives nothing (`InjectDamage` damages
//! `self`, gated on `self.alive` — agent 1 stays alive=0 on purpose,
//! not via the `target_slot` config value, which is documented but not
//! actually wired into the emit) and must stay all-zero throughout.
//!
//! **Re-pinned with the fold-ring snapshot fix (fix/spatial-grid).**
//! The values used to be one tick "ahead" (20, 30, 40, 50, … — read as
//! "`world.tick` is 1 on the first step"). That was the fold-row bug,
//! not tick numbering: folds consume the PRIOR tick's events, but read
//! them from the live ring after the tail reset, and `InjectDamage`
//! (scheduled before the fold) had already overwritten slot 0 with the
//! CURRENT tick's event — so each fold appended this tick's amount
//! instead of last tick's. Folds now read a tick-start snapshot of the
//! prior-tick rows: the tick-0 emit (amount 10) is folded on the second
//! step, and the N-th append carries amount `10*N`.

use sims::per_entity_ring_probe::GeneratedRuntime;

#[test]
fn ring_append_fills_then_wraps_at_k_fifo() {
    let Some(mut state) = GeneratedRuntime::try_new(0xA51A_0001, 2) else {
        eprintln!("[per_entity_ring_probe] skipping: no wgpu adapter on host.");
        return;
    };

    // No init/spawn block in this fixture — storage buffers start
    // zero-initialized, and only agent 0 should ever act.
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32]));

    // Checkpoint 1: exactly 4 appends, no wrap yet. Found empirically
    // (a per-tick trace, not assumed): the FIRST `step()` produces no
    // append at all — `InjectDamage` emits on tick 1, not tick 0, so it
    // takes 5 total `step()` calls to observe 4 completed appends, not
    // 4. Read back after every step rather than in one batch afterward,
    // just to keep the readback pattern identical to the trace that
    // established these numbers.
    let (mut primary, mut cursor) = (Vec::new(), Vec::new());
    for _ in 0..5 {
        state.step();
        primary = { let b = state.view_storage_recent_damages_primary_buf.clone(); read_f32(&mut state, &b, 2 * 4) };
        cursor = { let b = state.view_storage_recent_damages_anchor_buf.clone(); read_u32(&mut state, &b, 2) };
    }
    assert_eq!(
        &primary[0..4],
        &[10.0, 20.0, 30.0, 40.0],
        "after exactly 4 appends (K=4) the ring should hold all 4 values in arrival order, no wrap yet: {:?}",
        &primary[0..4]
    );
    assert_eq!(&primary[4..8], &[0.0, 0.0, 0.0, 0.0], "agent 1 (untouched control) should still be all-zero: {:?}", &primary[4..8]);
    assert_eq!(cursor, vec![4, 0], "cursor should read 4 raw appends for agent 0, 0 for agent 1: {cursor:?}");

    // Checkpoint 2: 3 more appends (amounts 50/60/70) — slots 0, 1, 2
    // each wrap exactly once (cursor%4 for append counts 5,6,7 is
    // 0,1,2); slot 3 (from append 4, value 40) is untouched.
    for _ in 0..3 {
        state.step();
        primary = { let b = state.view_storage_recent_damages_primary_buf.clone(); read_f32(&mut state, &b, 2 * 4) };
        cursor = { let b = state.view_storage_recent_damages_anchor_buf.clone(); read_u32(&mut state, &b, 2) };
    }
    assert_eq!(
        &primary[0..4],
        &[50.0, 60.0, 70.0, 40.0],
        "after 7 appends (K=4) slots 0-2 should each have wrapped once (50,60,70), slot 3 (40) untouched: {:?}",
        &primary[0..4]
    );
    assert_eq!(&primary[4..8], &[0.0, 0.0, 0.0, 0.0], "agent 1 should still be all-zero after agent 0's wraps: {:?}", &primary[4..8]);
    assert_eq!(cursor, vec![7, 0], "cursor is a raw ever-incrementing counter, not itself wrapped: {cursor:?}");
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_entity_ring_probe::readback_f32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("per_entity_ring_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback_f32 map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_entity_ring_probe::readback_u32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("per_entity_ring_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback_u32 map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
