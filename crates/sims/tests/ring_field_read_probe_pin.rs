//! `ring_field_read_probe` pin — THE load-bearing GPU proof for the new
//! ring-field-read primitive (`<ring_view>.<field>(key, index)`): a
//! genuinely new engine capability built this session, not a
//! verification of something that already existed. Before this, the
//! `@per_entity_ring` write side (`self.append(...)`, proven in the two
//! sibling pins) had no way to be read back from outside its own fold
//! body at all.
//!
//! Uses `per_entity_ring_struct_probe`'s exact, already-proven write
//! shape and wraparound values as the known-good baseline: after 7
//! appends, cell 0 = (timestamp=5, amount=50.0), cell 3 = (timestamp=4,
//! amount=40.0) (see that pin's own trace/assertions for the full
//! fill-then-wrap sequence, and its note on the re-pin: the amounts
//! moved down one tick when folds stopped reading this tick's rows). `ReadBackCells` reads exactly those two
//! cells' two fields each, every tick, via the NEW primitive — if the
//! read side's key/index/field-offset arithmetic disagrees with the
//! write side's in ANY way, this catches it as a wrong value, not a
//! compile error.

use sims::ring_field_read_probe::GeneratedRuntime;

#[test]
fn ring_field_read_matches_the_known_good_write_side_values() {
    let Some(mut state) = GeneratedRuntime::try_new(0xA51A_0003, 1) else {
        eprintln!("[ring_field_read_probe] skipping: no wgpu adapter on host.");
        return;
    };

    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32]));

    // Same timing shape as the write-side sibling pins: 8 total steps
    // observes 7 completed appends (the first step produces none).
    for _ in 0..8 {
        state.step();
    }

    let buf = state.agent_read_cell0_timestamp_buf.clone();
    let cell0_ts = read_u32(&mut state, &buf, 1)[0];
    let buf = state.agent_read_cell0_amount_buf.clone();
    let cell0_amt = read_f32(&mut state, &buf, 1)[0];
    let buf = state.agent_read_cell3_timestamp_buf.clone();
    let cell3_ts = read_u32(&mut state, &buf, 1)[0];
    let buf = state.agent_read_cell3_amount_buf.clone();
    let cell3_amt = read_f32(&mut state, &buf, 1)[0];

    assert_eq!((cell0_ts, cell0_amt), (5, 50.0), "cell 0 (timestamp, amount) read back wrong via the new ring-field-read primitive");
    assert_eq!((cell3_ts, cell3_amt), (4, 40.0), "cell 3 (timestamp, amount) read back wrong via the new ring-field-read primitive");
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ring_field_read_probe::readback_f32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ring_field_read_probe::readback") });
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
        label: Some("ring_field_read_probe::readback_u32"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ring_field_read_probe::readback") });
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
