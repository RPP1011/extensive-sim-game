//! `spatial_aoe_probe` pin — ability AoE Path B covers an area larger
//! than one spatial-grid cell, and hits each target exactly once.
//!
//! Two Casters each cast `Blast` (`damage 1.0 in circle(20.0)`) once, on
//! tick 1, at the Anchor beside them — one in the world interior, one at
//! the world corner (+62, +62), with some of its targets beyond the ±64
//! world extent (clamped into the edge cells). Every agent's hp loss must
//! equal the number of Anchors within 20 units of it (0, 1 or 2 here):
//!
//! * agents 7-19 units from an Anchor were beyond the old fixed 27-cell
//!   walk (6-12 units of reach) and took no damage;
//! * at the corner the old clamped neighbour walk visited the edge cells
//!   up to 4 times, so their agents took the damage up to 4 times.
//!
//! Positions are integers; no agent sits exactly 20 units from an Anchor.

use sims::spatial_aoe_probe::GeneratedRuntime;

const SEED: u64 = 0xA0E_5EED;
const RADIUS: f32 = 20.0;
const CT_ANCHOR: u32 = 0;
const CT_CASTER: u32 = 1;
const CT_DUMMY: u32 = 2;
const HP0: f32 = 100.0;

fn layout() -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut pos = Vec::new();
    let mut ct = Vec::new();
    let mut add = |p: [f32; 2], t: u32| {
        pos.push([p[0], p[1], 0.0]);
        ct.push(t);
    };
    // Interior cast: caster at the origin, anchor (the AoE centre) 2u away.
    add([0.0, 0.0], CT_CASTER);
    add([2.0, 0.0], CT_ANCHOR);
    for p in [
        [5.0, 0.0], [2.0, 7.0], [-9.0, 0.0], [2.0, -15.0], [17.0, 10.0], // 3..19 u — inside
        [-17.0, 1.0], [2.0, 23.0], [30.0, 0.0], [-20.0, -9.0],          // 19.03, 23, 28, 24.2 — first inside
    ] {
        add(p, CT_DUMMY);
    }
    // Corner cast: anchor at (62, 62), targets on both sides of the ±64 edge.
    add([60.0, 62.0], CT_CASTER);
    add([62.0, 62.0], CT_ANCHOR);
    for p in [
        [70.0, 70.0], [55.0, 50.0], [63.0, 75.0], [75.0, 60.0], // 11.3, 13.9, 13.0, 13.2 — inside
        [40.0, 62.0], [80.0, 80.0], [62.0, 40.0], [90.0, 90.0], // 22, 25.5, 22, 39.6 — outside
    ] {
        add(p, CT_DUMMY);
    }
    (pos, ct)
}

fn expected_hits(pos: &[[f32; 3]], ct: &[u32]) -> Vec<f32> {
    let anchors: Vec<[f32; 3]> =
        pos.iter().zip(ct).filter(|(_, &t)| t == CT_ANCHOR).map(|(p, _)| *p).collect();
    pos.iter()
        .map(|p| {
            anchors
                .iter()
                .filter(|a| {
                    let d = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= RADIUS * RADIUS
                })
                .count() as f32
        })
        .collect()
}

fn run(pos: &[[f32; 3]], ct: &[u32]) -> Option<Vec<f32>> {
    let n = pos.len() as u32;
    let mut state = GeneratedRuntime::try_new(SEED, n)?;
    let pos4: Vec<[f32; 4]> = pos.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&pos4));
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(ct));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n as usize]));
    state.gpu.queue.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&vec![HP0; n as usize]));
    for _ in 0..6 {
        state.step();
    }
    let buf = state.agent_hp_buf.clone();
    let hp = read_f32(&mut state, &buf, n as usize);
    Some(hp.iter().map(|h| HP0 - h).collect())
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial_aoe_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spatial_aoe_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

#[test]
fn aoe_larger_than_a_cell_hits_every_target_once() {
    let (pos, ct) = layout();
    let Some(got) = run(&pos, &ct) else {
        eprintln!("[spatial_aoe_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let want = expected_hits(&pos, &ct);
    // The scenario really exercises beyond-one-cell reach and misses.
    assert!(want.iter().filter(|&&w| w == 1.0).count() >= 12);
    assert!(want.iter().filter(|&&w| w == 0.0).count() >= 7);
    let bad: Vec<String> = pos
        .iter()
        .enumerate()
        .filter(|(i, _)| got[*i] != want[*i])
        .map(|(i, p)| format!("agent {i} at {p:?}: damage taken {} (expected {})", got[i], want[i]))
        .collect();
    assert!(bad.is_empty(), "AoE hits disagree with the circle:\n{}", bad.join("\n"));
    let again = run(&pos, &ct).expect("adapter present");
    assert_eq!(got, again, "same-seed reruns diverged");
    println!("spatial_aoe_probe: {} agents, damage matches the r=20 circle exactly: {got:?}", pos.len());
}
