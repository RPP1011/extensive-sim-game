//! `per_entity_ring_struct_probe` pin — GPU proof that the struct-payload
//! ring-append primitive (`self.append(field: expr, ...)`, Plan G G3b/G3c)
//! works, not just `per_entity_ring_probe`'s single-f32-payload path.
//! This is the actual shape a working-memory design needs (a cell with
//! several named fields — entity, fact, value, tick — not one scalar).
//!
//! Same fixture/timing shape as `per_entity_ring_probe`: `InjectDamage`
//! fires on `self`, gated on `self.alive`, one `Damaged` event per tick;
//! events emitted on tick T are folded on tick T+1, so N appends take
//! N+1 total steps to observe and the N-th append carries amount `10*N`. What's genuinely new to verify here: does each
//! cell actually get BOTH fields written at the correct struct stride
//! (`ring_idx * 2 + field_offset`, per the emitted WGSL), and does the
//! ring's own `let now = world.tick` local (the G3c multi-statement body
//! admission) resolve to a real, correct value rather than garbage.
//!
//! **Re-pinned with the fold-ring snapshot fix (fix/spatial-grid).** The
//! amount field used to equal the fold's own tick (`(5, 60.0)`): the fold
//! was reading the CURRENT tick's event, which `InjectDamage` had just
//! written over the prior tick's row in the live ring. Folds now read a
//! tick-start snapshot of the prior-tick rows, so each cell pairs the
//! fold's processing tick `now` with the amount emitted one tick earlier
//! (`(5, 50.0)`: emitted on tick 4, folded on tick 5).

use sims::per_entity_ring_struct_probe::GeneratedRuntime;

#[test]
fn struct_cell_ring_append_writes_both_fields_at_correct_stride() {
    let Some(mut state) = GeneratedRuntime::try_new(0xA51A_0002, 2) else {
        eprintln!("[per_entity_ring_struct_probe] skipping: no wgpu adapter on host.");
        return;
    };

    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32]));

    // Same interleaved step+read pattern as the scalar sibling pin —
    // trace every tick rather than assume the timestamp field's exact
    // value, since it's `cfg.tick` read INSIDE the fold kernel (processing
    // time), not necessarily the same tick number the event named at
    // emission time.
    for t in 1..=8 {
        state.step();
        let raw = { let b = state.view_storage_recent_damage_records_primary_buf.clone(); read_u32(&mut state, &b, 2 * 4 * 2) };
        let cells: Vec<(u32, f32)> = (0..4).map(|i| (raw[i * 2], f32::from_bits(raw[i * 2 + 1]))).collect();
        println!("[per_entity_ring_struct_probe trace] after step {t}: agent0 cells={cells:?}");
    }

    let raw = { let b = state.view_storage_recent_damage_records_primary_buf.clone(); read_u32(&mut state, &b, 2 * 4 * 2) };
    let agent0: Vec<(u32, f32)> = (0..4).map(|i| (raw[i * 2], f32::from_bits(raw[i * 2 + 1]))).collect();
    let agent1: Vec<(u32, f32)> = (4..8).map(|i| (raw[i * 2], f32::from_bits(raw[i * 2 + 1]))).collect();

    // Exact expected cells after 7 appends (K=4): cells 0-2 each
    // wrapped once (the 5th/6th/7th appends, folded on ticks 5/6/7,
    // amounts 50/60/70 emitted on ticks 4/5/6); cell 3 keeps the 4th
    // append (folded tick 4, amount 40) untouched. `now` (the G3c `let`
    // binding) tracks the FOLD KERNEL's own processing tick, one tick
    // after the amount's emission tick.
    assert_eq!(
        agent0,
        vec![(5, 50.0), (6, 60.0), (7, 70.0), (4, 40.0)],
        "struct cells should hold exact (timestamp, amount) pairs after fill+wrap: {agent0:?}"
    );
    assert_eq!(agent1, vec![(0, 0.0); 4], "agent 1 (untouched control) should stay all-zero: {agent1:?}");
}

fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_entity_ring_struct_probe::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("per_entity_ring_struct_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
