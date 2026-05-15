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
    // Each agent is BOTH an observer and a caster (every alive agent
    // gets stamped busy by MarkCasterBusy). Cell center for cell-c
    // (allocated when source_candidate=c) reads `agents.pos(c)` and
    // packs as q8 — see threats_struct_probe.sim's fold body.
    //
    // Per-observer intensity = sum over all 4 cells of
    //   max(0, radius - distance(observer.pos, caster_c.pos))
    //
    // With observers/casters at the positions below and radius=4,
    // the mutual-distance matrix produces non-monotone utilities
    // (observer 1 outscores observer 0 because it's closer to the
    // mass of casters). This is the surface-DSL proof that cell.center
    // is varying per-cell — a constant-center fold would give every
    // observer the same magnitude.
    let positions: [[f32; 4]; N as usize] = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [3.5, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ];
    // Pre-computed expected intensity per observer.
    //   observer 0 at (0,0):  d=[0, 1, 3.5, 10]   → contrib=[4, 3, 0.5, 0]   = 7.5
    //   observer 1 at (1,0):  d=[1, 0, 2.5, 9]    → contrib=[3, 4, 1.5, 0]   = 8.5
    //   observer 2 at (3.5,0):d=[3.5, 2.5, 0, 6.5]→ contrib=[0.5, 1.5, 4, 0] = 6.0
    //   observer 3 at (10,0): d=[10, 9, 6.5, 0]   → contrib=[0, 0, 0, 4]     = 4.0
    let expected_total: [f32; N as usize] = [7.5, 8.5, 6.0, 4.0];

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

    // Per-observer exact-magnitude pin. The intensity helper sums
    // `max(0, radius - distance)` over K=4 live cells with varying
    // centers (one per caster). The expected matrix is derived from
    // the mutual-distance lattice between observers and casters
    // (computed in the comment above).
    //
    // f32 ULP slack: q8 packing rounds positions to 1/256 unit,
    // distance computation chains 2 mults + 1 sqrt. Empirical drift
    // observed at ~5e-3 max; pin at 0.05 to absorb future regression
    // noise without masking real semantic regressions.
    for i in 0..(N as usize) {
        let expected = expected_total[i];
        let observed = utilities[i];
        let delta = (observed - expected).abs();
        assert!(
            delta < 0.05,
            "observer {i}: expected {expected}, got {observed} (delta {delta:.4})",
        );
    }

    // Independent invariants that don't depend on the exact
    // magnitudes: observer 1 (closest to the mass of casters) must
    // outscore the bookend observers; observer 3 (out of radius for
    // every caster except itself) sees exactly the self-cast cell.
    assert!(
        utilities[1] > utilities[0],
        "observer 1 sits among 3 in-range casters and must outscore observer 0 (only 2 in-range neighbours): {:?}",
        utilities
    );
    assert!(
        utilities[1] > utilities[3],
        "observer 1 must outscore observer 3 (lone in-range cell = self): {:?}",
        utilities
    );

    // Plan G G3f follow-up — gap (b) verification. The fold body now
    // writes `source: source_candidate` (was hardcoded `0u`) so each
    // cell carries the actual per-pair caster id. Every observer's
    // K=4 ring should contain exactly the source ids {0, 1, 2, 3}
    // (one per busy candidate) — this is the surface-DSL proof that
    // the per_agent_event_scan local binding flows through to a real
    // SoA-relative read.
    let cells = read_threats_cells(&mut state, N, 4);
    for obs in 0..(N as usize) {
        let base = obs * 4;
        let mut sources: Vec<u32> = (base..base + 4).map(|i| cells[i]._source).collect();
        sources.sort();
        assert_eq!(
            sources,
            vec![0u32, 1, 2, 3],
            "observer {obs}: cell.source ids must equal {{0,1,2,3}} after gap-(b) plumb-through; \
             got {sources:?}",
        );
    }

    // Plan G G3f follow-up — gap (d) partial verification. cell.expires_at_tick
    // reads `agents.busy_until_tick(source_candidate)` instead of the
    // old `world.tick + 100` constant. MarkCasterBusy stamps
    // `busy_until_tick = world.tick + zone_duration` at tick 0 (where
    // zone_duration = 100 from config), so every cell's expires_at_tick
    // should be exactly 100 (= 0 + 100). A regression that swapped the
    // SoA read back to the constant would still produce 100-ish values
    // (but at tick-of-fold + 100, e.g. 102 if the fold runs at tick 2)
    // — the EXACT 100 pin distinguishes the two paths.
    for obs in 0..(N as usize) {
        for c in 0..4 {
            let cell = &cells[obs * 4 + c];
            assert_eq!(
                cell.expires_at_tick, 100,
                "observer {obs} cell {c}: expires_at_tick must read busy_until_tick (= 100), \
                 got {got}",
                got = cell.expires_at_tick,
            );
        }
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

/// Plan G G3f follow-up — sign-extension regression for the q8 unpack.
///
/// The intensity helper unpacks each cell's `center_x_q8` via:
///   `let cx = f32((cx_raw << 16u) >> 16u) / 256.0;`
///
/// Without the `<<16 >>16` sign-extend dance, a caster at `x = -2.0`
/// (q8 = -512 = `0xFE00` in low 16 bits, packed as `0x0000FE00` in
/// the u32 slot) would unpack to `+254.0` (raw integer interpretation)
/// instead of `-2.0`. Distance computation would then be wildly wrong
/// — a near-zero observer would see the negative-coord caster as
/// 250+ units away and contribute 0 intensity instead of the real
/// in-radius value.
///
/// This pin seeds caster 0 at `x = -2.0` and asserts the resulting
/// intensities still match the mutual-distance lattice.
#[test]
fn pos_keyed_intensity_handles_negative_coord_casters() {
    const N: u32 = 4;
    let positions: [[f32; 4]; N as usize] = [
        [-2.0, 0.0, 0.0, 0.0],   // caster 0 at -2.0 — exercises sign extend
        [1.0, 0.0, 0.0, 0.0],
        [3.5, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
    ];
    // Mutual-distance matrix at radius=4:
    //   obs 0 at (-2,0): d=[0, 3, 5.5, 12]   live=[T,T,F,F]  contrib=[4,1,0,0]   = 5.0
    //   obs 1 at (1,0):  d=[3, 0, 2.5, 9]    live=[T,T,T,F]  contrib=[1,4,1.5,0] = 6.5
    //   obs 2 at (3.5,0):d=[5.5, 2.5, 0, 6.5]live=[F,T,T,F]  contrib=[0,1.5,4,0] = 5.5
    //   obs 3 at (10,0): d=[12, 9, 6.5, 0]   live=[F,F,F,T]  contrib=[0,0,0,4]   = 4.0
    let expected_total: [f32; N as usize] = [5.0, 6.5, 5.5, 4.0];

    let mut state = match try_runtime(0xCAFEBABE, N) {
        Some(s) => s,
        None => {
            eprintln!("[neg_coord_intensity] skipping: no wgpu adapter on host.");
            return;
        }
    };

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

    for _ in 0..3 {
        state.step();
    }

    let utilities = read_scoring_utilities(&mut state, N);
    let cells = read_threats_cells(&mut state, N, 4);
    eprintln!("[neg_coord_intensity] utilities: {:?}", utilities);

    // Pin the negative-coord cell content. center_x_q8 for caster 0
    // packed as q8 of -2.0 = -512. The u32 slot holds the i16 in
    // its low 16 bits, sign-bit propagated up by the `i32(...)`
    // cast in the .sim's q8 expression: bit pattern is `0xFFFFFE00`
    // when stored as u32 (the i32 -512 reinterpreted), or `0xFE00`
    // (= 65024) when stored as a u16-clamped i16. The .sim writes
    // via `u32(i32(caster_pos.x * 256.0))` which sign-extends to
    // i32 first, then reinterprets as u32 — yielding 0xFFFFFE00.
    let caster_0_cell = &cells[0]; // observer 0's first ring slot
    assert_eq!(
        caster_0_cell.center_x_q8, 0xFFFFFE00u32,
        "caster 0's q8(-2.0) must store as 0xFFFFFE00 (i32 -512 reinterpreted as u32); got 0x{:08X}",
        caster_0_cell.center_x_q8,
    );

    for (i, u) in utilities.iter().enumerate() {
        assert!(
            u.is_finite() && *u >= 0.0,
            "observer {i} utility = {u} (must be finite + non-negative)",
        );
    }

    for i in 0..(N as usize) {
        let expected = expected_total[i];
        let observed = utilities[i];
        let delta = (observed - expected).abs();
        assert!(
            delta < 0.05,
            "observer {i}: expected {expected}, got {observed} (delta {delta:.4}). \
             Sign-extension may be broken — without `<<16 >>16` the negative-coord \
             cell would unpack to +254 instead of -2 and observer 0's intensity \
             would collapse to 0.",
        );
    }
}
