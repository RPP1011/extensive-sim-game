//! Plan 1 — Task 1 characterization + end-to-end lock for the `@runtime`
//! config input channel.
//!
//! The `input_probe.sim` fixture moves every alive agent by the
//! `@runtime` config field `config.probe.drive` each tick. This test
//! drives the generated host setter and asserts the GPU kernel actually
//! reads the per-tick host write.
//!
//! CHARACTERIZATION OUTCOME (Task 1, Step 4): **(c) — already works
//! end-to-end.** The "Plan G tunable cfg" lowering (runtime ConfigConst
//! → `cfg.<field>` uniform read, not baked const) + the "Plan E-A6"
//! generated setter (`set_config_probe_drive`, writes the host mirror
//! AND the per-kernel cfg buffer at offset 16) are both present in the
//! emitter today. The generated setter name is
//! `set_config_<block>_<field>` → `set_config_probe_drive` here (NOT
//! `set_probe_drive` as the plan draft assumed). No emitter change was
//! needed; Tasks 2 and 3 are skipped.
//!
//! Generated symbols (from `OUT_DIR/input_probe/runtime_core.rs`):
//!   - host mirror:  `pub config_probe_drive: f32`
//!   - host setter:  `fn set_config_probe_drive(&mut self, value: f32)`
//!     → writes `cfg_physics_DriveX_buf` at byte offset 16.
//!   - WGSL read:    `agent_pos[id] += vec3(cfg.config_probe_drive,0,0)`

use sims::input_probe::GeneratedRuntime;

/// Staging readback of `agent_pos_buf[slot].x`. Pos buffer is vec3<f32>
/// padded to vec4 (16-byte stride), mirroring the vampire_survivors /
/// pirate_fleet readback idiom.
fn read_pos_x(rt: &mut GeneratedRuntime, slot: usize) -> f32 {
    let bytes = (slot as u64 + 1) * 16;
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input_probe::pos_rb"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("input_probe::pos_readback"),
        });
    enc.copy_buffer_to_buffer(&rt.agent_pos_buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let sl = staging.slice(..bytes);
    sl.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let x = {
        let view = sl.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        f32::from_bits(words[slot * 4])
    };
    staging.unmap();
    x
}

/// Seed agent `slot` alive at the origin. No fixture seed fn exists for
/// `input_probe`, so write the alive bit + zeroed position directly via
/// the queue (the plan's Step 3 fallback).
fn seed_agent_alive_at_origin(rt: &mut GeneratedRuntime, slot: usize) {
    // alive bit (u32, stride 4)
    rt.gpu.queue.write_buffer(
        &rt.agent_alive_buf,
        (slot as u64) * 4,
        bytemuck::bytes_of(&1u32),
    );
    // position (vec4<f32> stride 16) at origin
    rt.gpu.queue.write_buffer(
        &rt.agent_pos_buf,
        (slot as u64) * 16,
        bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 0.0]),
    );
}

#[test]
fn input_probe_moves_agent() {
    let Some(mut rt) = GeneratedRuntime::try_new(0xABCD, 8) else {
        eprintln!("[input_probe] no GPU adapter; skipping");
        return;
    };

    seed_agent_alive_at_origin(&mut rt, 0);

    // The host setter under test — exact generated name is
    // `set_config_probe_drive` (block=probe, field=drive).
    rt.set_config_probe_drive(2.0);

    let before = read_pos_x(&mut rt, 0);
    rt.step();
    let after = read_pos_x(&mut rt, 0);

    eprintln!("[input_probe] pos.x before={before} after={after} (expected +2.0)");
    assert!(
        (after - before - 2.0).abs() < 1e-3,
        "expected +2.0 from drive, got {before} -> {after}"
    );
}

/// Second pin: the setter is the *channel* — changing it between ticks
/// changes the per-tick displacement. Proves the value is read live each
/// step (not baked once), which is the property the playable game needs.
#[test]
fn input_probe_setter_drives_per_tick() {
    let Some(mut rt) = GeneratedRuntime::try_new(0xABCD, 8) else {
        eprintln!("[input_probe] no GPU adapter; skipping");
        return;
    };

    seed_agent_alive_at_origin(&mut rt, 0);

    let p0 = read_pos_x(&mut rt, 0);

    rt.set_config_probe_drive(1.5);
    rt.step();
    let p1 = read_pos_x(&mut rt, 0);
    assert!(
        (p1 - p0 - 1.5).abs() < 1e-3,
        "tick 1: expected +1.5, got {p0} -> {p1}"
    );

    // Change the input mid-run; the next step must use the new value.
    rt.set_config_probe_drive(-0.5);
    rt.step();
    let p2 = read_pos_x(&mut rt, 0);
    assert!(
        (p2 - p1 - (-0.5)).abs() < 1e-3,
        "tick 2: expected -0.5 after re-set, got {p1} -> {p2}"
    );

    eprintln!("[input_probe] per-tick drive: {p0} -> {p1} (+1.5) -> {p2} (-0.5) PASS");
}

/// Third pin: a `u32` `@runtime` field must generate a `u32`-typed setter
/// that writes u32 bits (not the f32 bit pattern) into the cfg slot the
/// WGSL reads as `u32`. `gate % 2` toggles the X drive: odd = on, even =
/// off. This is the property the VS contract's `bolt_rate_level: u32`
/// (an integer modulo divisor) depends on.
#[test]
fn input_probe_u32_gate_toggles_drive() {
    let Some(mut rt) = GeneratedRuntime::try_new(0xABCD, 8) else {
        eprintln!("[input_probe] no GPU adapter; skipping");
        return;
    };

    seed_agent_alive_at_origin(&mut rt, 0);
    rt.set_config_probe_drive(2.0);

    // gate = 1 (odd) -> drive ON: agent moves +2.0.
    rt.set_config_probe_gate(1);
    let p0 = read_pos_x(&mut rt, 0);
    rt.step();
    let p1 = read_pos_x(&mut rt, 0);
    assert!(
        (p1 - p0 - 2.0).abs() < 1e-3,
        "gate=1 (odd) should enable drive: expected +2.0, got {p0} -> {p1}"
    );

    // gate = 2 (even) -> drive OFF: agent stays put.
    rt.set_config_probe_gate(2);
    rt.step();
    let p2 = read_pos_x(&mut rt, 0);
    assert!(
        (p2 - p1).abs() < 1e-3,
        "gate=2 (even) should disable drive: expected no move, got {p1} -> {p2}"
    );

    eprintln!("[input_probe] u32 gate toggle: gate=1 +2.0 ({p0}->{p1}), gate=2 hold ({p1}->{p2}) PASS");
}
