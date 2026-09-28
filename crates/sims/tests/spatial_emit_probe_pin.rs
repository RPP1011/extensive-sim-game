//! `spatial_emit_probe` pin — a per_agent rule that walks a radius-2
//! `spatial_query` in a `for` body and emits an event per neighbour
//! (the webband colony's `Jostle` shape), end to end: the walk finds the
//! neighbour, the emit lands in the event ring, and the materialized view
//! folds each event exactly once.
//!
//! Layout: A (slot 0) and B (slot 1) 1.0u apart far from everyone,
//! C (slot 2) 3.0u from B (outside radius 2), plus `N_CROWD` agents in a
//! dense knot elsewhere that bump each other on every pulse — so the
//! pair's rows share the event ring with a busy tick, as in the colony.
//! Pulses fire on ticks with `tick % 20 == 10`.

use sims::spatial_emit_probe::GeneratedRuntime;

const SEED: u64 = 0x0E_417_5EED;
const N_CROWD: usize = 60;
const TICKS: u32 = 60; // pulses at ticks 10, 30, 50

fn layout() -> Vec<[f32; 3]> {
    let mut p = vec![[40.0, 40.0, 0.0], [41.0, 40.0, 0.0], [44.0, 40.0, 0.0]];
    for k in 0..N_CROWD {
        // 0.25u lattice: everyone in the knot is within radius 2 of many.
        p.push([-20.0 + (k % 8) as f32 * 0.25, -20.0 + (k / 8) as f32 * 0.25, 0.0]);
    }
    p
}

fn expected(pos: &[[f32; 3]]) -> Vec<f32> {
    // Bumps taken per pulse = number of other agents within radius 2.
    pos.iter()
        .enumerate()
        .map(|(i, a)| {
            pos.iter()
                .enumerate()
                .filter(|(j, b)| {
                    *j != i && {
                        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= 4.0
                    }
                })
                .count() as f32
        })
        .collect()
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial_emit_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spatial_emit_probe::readback") });
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

fn run(pos: &[[f32; 3]], chatty: bool) -> Option<(Vec<f32>, Vec<f32>)> {
    let n = pos.len() as u32;
    let mut state = GeneratedRuntime::try_new(SEED, n)?;
    if chatty {
        state.gpu.queue.write_buffer(&state.agent_chatty_buf, 0, bytemuck::cast_slice(&vec![1u32; n as usize]));
    }
    let pos4: Vec<[f32; 4]> = pos.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&pos4));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n as usize]));
    for _ in 0..TICKS {
        state.step();
    }
    let seen_buf = state.agent_seen_near_buf.clone();
    let seen = read_f32(&mut state, &seen_buf, n as usize);
    let view_buf = state.view_storage_bumps_taken_primary_buf.clone();
    let taken = read_f32(&mut state, &view_buf, n as usize);
    Some((seen, taken))
}

#[test]
fn radius_walk_emits_land_once_per_neighbour() {
    let pos = layout();
    let Some((seen, taken)) = run(&pos, false) else {
        eprintln!("[spatial_emit_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let per_pulse = expected(&pos);
    let pulses = 3.0;
    println!("pair seen: {:?}  pair taken: {:?}", &seen[..3], &taken[..3]);
    // The walk: the emitting side (a self-write, no event ring involved).
    for i in 0..pos.len() {
        assert_eq!(seen[i], per_pulse[i] * pulses, "agent {i} walk saw {} neighbours (expected {})", seen[i], per_pulse[i] * pulses);
    }
    assert_eq!((seen[0], seen[1], seen[2]), (3.0, 3.0, 0.0), "the 1u pair must see each other once per pulse");
    // The fold side: every emitted Bumped lands in the view exactly once.
    // (Events from the last pulse are folded on the following tick,
    // which runs well within TICKS.)
    for i in 0..pos.len() {
        assert_eq!(taken[i], per_pulse[i] * pulses, "agent {i}: {} Bumped landed in the view (expected {})", taken[i], per_pulse[i] * pulses);
    }
    let again = run(&pos, false).expect("adapter present");
    assert_eq!((seen, taken), again, "same-seed reruns diverged");
}

/// Fold rows survive a busy following tick.
///
/// View folds consume the PRIOR tick's events. Here `Chat` emits on the
/// tick after each pulse and is scheduled before `FoldBumpsTaken`; the
/// prior-tick rows it would overwrite in the live ring (the sorted head,
/// i.e. the lowest target ids — the pair in slots 0/1) must still reach
/// the view exactly once. Failed on 370557ec..dc05a9ca (0 of 3 landed —
/// the webband colony's "parked pair's brawls never landed"); folds now
/// read a tick-start snapshot of the prior-tick rows
/// (`fold_event_ring_buf`, dsl_compiler build_helper.rs).
#[test]
fn fold_rows_survive_a_busy_following_tick() {
    let pos = layout();
    let Some((seen, taken)) = run(&pos, true) else {
        eprintln!("[spatial_emit_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let per_pulse = expected(&pos);
    for i in 0..pos.len() {
        assert_eq!(seen[i], per_pulse[i] * 3.0, "agent {i}: walk");
        assert_eq!(taken[i], per_pulse[i] * 3.0, "agent {i}: {} Bumped landed (expected {})", taken[i], per_pulse[i] * 3.0);
    }
}
