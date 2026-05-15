//! Plan G G3f follow-up — runtime behavioral pin for the pos-keyed
//! `view_<id>_get` ring walk.
//!
//! Drives `sims::threats_struct_probe::GeneratedRuntime` with 4 agents
//! seeded at varying distances from the cell center (0, 0). Cells are
//! written each tick by the @per_entity_ring fold (per
//! `assets/sim/threats_struct_probe.sim`'s `self.append(...)` body) with
//! constant content: `zone_kind = 1`, `center = (0, 0)`, `radius = 4.0`,
//! `expires_at_tick = world.tick + 100`.
//!
//! The Probe verb's `score (threats.intensity_at(self))` triggers the
//! WGSL helper `view_0_get(observer)` — which now (post commit
//! 70a6634d) walks the K=4 cells, computes
//! `distance(agent_pos[observer], cell.center)` per cell, and accumulates
//! `max(0, radius - distance)`. Utility = sum over the 4 cells (all
//! same content) = `4 * max(0, 4.0 - dist)`.
//!
//! **The pin.** Per-agent best_utility (read from scoring_output) must
//! decrease as distance from the cell center increases. The agent past
//! the radius gets exactly 0. A scalar-count helper would give every
//! agent the same value (= 4) regardless of position — this is the
//! anti-regression guard.

use sims::threats_struct_probe::GeneratedRuntime;

fn try_runtime(seed: u64, n: u32) -> Option<GeneratedRuntime> {
    GeneratedRuntime::try_new(seed, n)
}

#[test]
fn pos_keyed_intensity_decreases_with_distance_to_cell_center() {
    const N: u32 = 4;
    const RADIUS: f32 = 4.0;
    // Distances pinned to drive a strict monotone decrease + a
    // beyond-radius zero. Each value covers a distinct regime: at
    // center, just inside, near edge, beyond radius.
    let positions: [[f32; 4]; N as usize] = [
        [0.0, 0.0, 0.0, 0.0],   // observer 0: dist=0
        [1.0, 0.0, 0.0, 0.0],   // observer 1: dist=1
        [3.5, 0.0, 0.0, 0.0],   // observer 2: dist=3.5
        [10.0, 0.0, 0.0, 0.0],  // observer 3: dist=10 > radius
    ];
    let expected_per_cell = [
        RADIUS - 0.0,
        RADIUS - 1.0,
        RADIUS - 3.5,
        0.0, // out of radius
    ];

    let mut state = match try_runtime(0xCAFEBABE, N) {
        Some(s) => s,
        None => {
            eprintln!("[pos_keyed_intensity] skipping: no wgpu adapter on host.");
            return;
        }
    };

    // Seed agent_alive=1 + per-agent positions before stepping. The
    // generated runtime zero-initialises both buffers; without these
    // writes the MarkCasterBusy gate (`self.alive`) wouldn't fire and
    // every position would read as (0, 0).
    let alive: Vec<u32> = vec![1u32; N as usize];
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // Schedule order for this fixture (per the fused emit):
    //   * fold_threats runs FIRST — at tick 0 it sees no busy
    //     candidates (busy stamp lives in the fused MarkCasterBusy
    //     kernel that runs after), so no cells get written.
    //   * fused_MarkCasterBusy stamps busy + runs scoring against the
    //     still-empty ring (utility = 0 for everyone).
    //   * Tick 1: fold sees busy → writes K=4 cells per observer.
    //     fused_MarkCasterBusy's scoring then sees real cells.
    // Two ticks gets us a populated ring; we step once more for safety
    // so any pipeline race has time to settle.
    for _ in 0..3 {
        state.step();
    }

    let utilities = read_scoring_utilities(&mut state, N);
    let cells = read_threats_cells(&mut state, N, 4);
    let positions_back = read_agent_pos(&mut state, N);
    eprintln!("[pos_keyed_intensity] utilities: {:?}", utilities);
    eprintln!("[pos_keyed_intensity] agent_pos after step: {:?}", positions_back);
    for obs in 0..(N as usize) {
        let base = obs * 4;
        eprintln!(
            "[pos_keyed_intensity] observer {} cells: {:?}",
            obs,
            &cells[base..base + 4],
        );
    }

    // Every reading must be finite + non-negative — the helper's
    // `max(0, radius - distance)` invariant.
    for (i, u) in utilities.iter().enumerate() {
        assert!(
            u.is_finite() && *u >= 0.0,
            "observer {i} utility = {u} (must be finite + non-negative)",
        );
    }

    // Strict monotone decrease with distance. The K=4 cells per
    // observer all carry the same constant content, so utility = 4 *
    // max(0, radius - distance). Pre-pin the ordering rather than the
    // exact magnitudes — the Probe vs Idle argmax winner depends on
    // whether utility > 0; what matters for the regression is that the
    // helper actually responded to position.
    assert!(
        utilities[0] > utilities[1],
        "observer 0 (dist=0) must outscore observer 1 (dist=1): {:?}",
        utilities
    );
    assert!(
        utilities[1] > utilities[2],
        "observer 1 (dist=1) must outscore observer 2 (dist=3.5): {:?}",
        utilities
    );
    assert!(
        utilities[2] > utilities[3],
        "observer 2 (dist=3.5) must outscore observer 3 (dist=10): {:?}",
        utilities
    );

    // Beyond-radius observer must score exactly 0 — every cell falls
    // through the `if (dist < radius)` accumulator gate, leaving the
    // running sum untouched. (The Idle verb's score is also 0; the
    // argmax tiebreak picks the lower action id, which here is Probe;
    // but the utility magnitude is what we assert.)
    assert_eq!(
        utilities[3], 0.0,
        "observer 3 (dist=10 > radius=4) must score 0; got {}",
        utilities[3],
    );

    // Tighter cross-check: per-cell expected ≈ utility / K. Allow
    // ULP-scale slack from the f32 distance + accumulator path.
    for i in 0..(N as usize) {
        if expected_per_cell[i] == 0.0 {
            continue; // already pinned above
        }
        let expected_total = expected_per_cell[i] * 4.0; // K=4 cells per observer
        let observed = utilities[i];
        let delta = (observed - expected_total).abs();
        assert!(
            delta < 0.05,
            "observer {i}: expected {expected_total}, got {observed} (delta {delta:.4})",
        );
    }
}

/// One ThreatZoneCell: 8 u32 words per cell, K cells per observer.
#[derive(Debug)]
#[allow(dead_code)]
struct CellSnapshot {
    zone_kind: u32,
    center_x_q8: u32,
    center_y_q8: u32,
    radius_q8: u32,
    _dir_x_q8: u32,
    _dir_y_q8: u32,
    expires_at_tick: u32,
    _source: u32,
}

fn read_threats_cells(
    state: &mut GeneratedRuntime,
    n: u32,
    k: u32,
) -> Vec<CellSnapshot> {
    let bytes = (n as u64) * (k as u64) * 8 * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threats_struct_probe_pos_keyed::cells_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threats_struct_probe_pos_keyed::cells_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(
        &state.view_storage_threats_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("primary map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        (0..((n * k) as usize))
            .map(|i| {
                let b = i * 8;
                CellSnapshot {
                    zone_kind: words[b],
                    center_x_q8: words[b + 1],
                    center_y_q8: words[b + 2],
                    radius_q8: words[b + 3],
                    _dir_x_q8: words[b + 4],
                    _dir_y_q8: words[b + 5],
                    expires_at_tick: words[b + 6],
                    _source: words[b + 7],
                }
            })
            .collect()
    };
    staging.unmap();
    out
}

fn read_agent_pos(state: &mut GeneratedRuntime, n: u32) -> Vec<[f32; 4]> {
    let bytes = (n as u64) * 4 * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threats_struct_probe_pos_keyed::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threats_struct_probe_pos_keyed::pos_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(&state.agent_pos_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("pos map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        (0..(n as usize))
            .map(|i| [words[i * 4], words[i * 4 + 1], words[i * 4 + 2], words[i * 4 + 3]])
            .collect()
    };
    staging.unmap();
    out
}

/// Read the per-agent best_utility from `scoring_output` (4 u32 per
/// agent: best_action, best_target, bitcast<u32>(best_utility), _).
fn read_scoring_utilities(state: &mut GeneratedRuntime, n: u32) -> Vec<f32> {
    let bytes = (n as u64) * 4 * 4; // 4 u32 per agent × 4 bytes
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threats_struct_probe_pos_keyed::scoring_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threats_struct_probe_pos_keyed::scoring_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(&state.scoring_output_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("scoring_output map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        (0..(n as usize))
            .map(|i| f32::from_bits(words[i * 4 + 2]))
            .collect()
    };
    staging.unmap();
    out
}
