//! `threat_scoring_stresstest` — effectiveness of threat horizon on
//! scoring decisions. Mirrors `assets/sim/threat_scoring_stresstest.sim`.
//!
//! ## What this pin measures
//!
//! For each of three modes — threat-aware (weight=1.0), threat-blind
//! (weight=0.0), and partial-aware (weight=0.05) — run 32 ticks and
//! read the per-tick scoring output for each of the 4 dodger slots:
//!
//!   * **per-tick Flee count** — how many of the 4 dodgers had Flee
//!     win the argmax.
//!   * **Flee-tick fraction** — total ticks any dodger picked Flee,
//!     divided by `4 * TICKS`.
//!   * **utility trajectory** — the best_utility value per tick for
//!     dodger slot 4. This reveals the threat fold + decay shape.
//!
//! ## Expected behaviour
//!
//!   * weight=1.0 → with long-fuse casters contributing +1/tick each,
//!     steady-state threats ≈ 2/(1-0.85) = 13.33. Flee score ≈ 13.33
//!     > Hold's 0.5 → Flee wins every tick → Flee-tick fraction = 1.0.
//!   * weight=0.0 → Flee scores 0.0 < Hold's 0.5 → Hold wins every
//!     tick → Flee-tick fraction = 0.0.
//!   * weight=0.05 → Flee scores ≈ 0.05 * threats(tick); during
//!     early ticks threats < 10 → score < 0.5 → Hold wins. Once
//!     threats * 0.05 ≥ 0.5 → Flee wins. The crossover tick is the
//!     observable "shape of threat horizon affecting scoring" metric.
//!
//! ## Load-bearing pins
//!
//!   1. **Decision shift**: `flee_ticks(1.0) - flee_ticks(0.0)` ≥
//!      `3.5 * TICKS` (≥ 87.5% of the maximum 4 × TICKS spread).
//!      Confirms the threat-aware vs threat-blind delta is real and
//!      load-bearing.
//!   2. **Horizon shape**: weight=0.05 mode shows a crossover (Hold
//!      → Flee tick boundary) somewhere in the middle of the run,
//!      not at the start (always-Hold) or the end (always-Flee).
//!      Proves the @decay-shaped fold trajectory shapes scoring.

#![allow(non_snake_case)]

use sims::threat_scoring_stresstest::GeneratedRuntime;

const SEED: u64 = 0xF1EE_5C0E_5DEC_1DE7;
const N_TOTAL: u32 = 8;
const N_LONG_FUSE: usize = 2;   // slots 0..2
const N_SHORT_FUSE: usize = 2;  // slots 2..4
const N_DODGERS: usize = 4;     // slots 4..8
const DODGER_BASE: usize = 4;
const TICKS: u32 = 32;

const CT_CASTER: u32 = 0;
const CT_DODGER: u32 = 1;

/// Run one mode of the simulation, returning per-tick observed
/// scoring data for the 4 dodger slots. Each entry is the action id
/// (0 = Flee, 1 = Hold, by .sim decl order) and the best_utility for
/// dodger slot 4.
fn run_one(weight: f32) -> Option<RunOutput> {
    let mut state = GeneratedRuntime::try_new(SEED, N_TOTAL)?;
    state.set_config_scoring_flee_weight(weight);

    seed_topology(&mut state);

    let mut per_tick_actions: Vec<[u32; N_DODGERS]> = Vec::with_capacity(TICKS as usize);
    let mut per_tick_utility_slot4: Vec<f32> = Vec::with_capacity(TICKS as usize);

    for tick in 0..TICKS {
        state.step();
        // After tick 0, clear short-fuse busy SoA cells so only the
        // long-fuse casters keep contributing. The MarkInitialBusy
        // physics rule fires only at tick 0; from tick 1+ the
        // host owns the SoA.
        if tick == 0 {
            clear_short_fuse_busy(&mut state);
        }
        let scoring = read_scoring_output(&mut state);
        let mut row: [u32; N_DODGERS] = [0; N_DODGERS];
        for i in 0..N_DODGERS {
            row[i] = scoring[(DODGER_BASE + i) * 4]; // best_action
        }
        per_tick_actions.push(row);
        let util_slot4 = f32::from_bits(scoring[DODGER_BASE * 4 + 2]);
        per_tick_utility_slot4.push(util_slot4);
    }

    Some(RunOutput {
        per_tick_actions,
        per_tick_utility_slot4,
    })
}

struct RunOutput {
    /// Per-tick rows of [dodger0_action, dodger1_action, ...] —
    /// `u32` action ids by .sim verb decl order (Flee=0, Hold=1).
    per_tick_actions: Vec<[u32; N_DODGERS]>,
    /// Per-tick scalar best_utility for dodger slot 4 (the canonical
    /// observer used in single-trace reporting).
    per_tick_utility_slot4: Vec<f32>,
}

impl RunOutput {
    /// Count how many (dodger × tick) cells picked the given action
    /// id. Max value = N_DODGERS * TICKS = 128.
    fn count_action(&self, action_id: u32) -> u32 {
        self.per_tick_actions
            .iter()
            .map(|row| row.iter().filter(|&&a| a == action_id).count() as u32)
            .sum()
    }
}

#[test]
fn threat_scoring_decision_impact_report() {
    let aware = match run_one(1.0) {
        Some(r) => r,
        None => {
            eprintln!("[threat_scoring_stresstest] skipping: no wgpu adapter");
            return;
        }
    };
    let blind = run_one(0.0).expect("blind run already worked aware");
    let partial = run_one(0.05).expect("partial run already worked aware");

    // The argmax kernel writes the per-agent action id with verbs
    // indexed by their .sim decl order; threat_scoring_stresstest.sim
    // declares `verb Flee` before `verb Hold`. So Flee=0, Hold=1.
    const FLEE: u32 = 0;
    const HOLD: u32 = 1;

    let max_cells = (N_DODGERS as u32) * TICKS;

    let aware_flee = aware.count_action(FLEE);
    let aware_hold = aware.count_action(HOLD);
    let blind_flee = blind.count_action(FLEE);
    let blind_hold = blind.count_action(HOLD);
    let partial_flee = partial.count_action(FLEE);
    let partial_hold = partial.count_action(HOLD);

    let aware_pct = (aware_flee as f32) / (max_cells as f32) * 100.0;
    let blind_pct = (blind_flee as f32) / (max_cells as f32) * 100.0;
    let partial_pct = (partial_flee as f32) / (max_cells as f32) * 100.0;

    // Crossover tick for the partial-aware mode: first tick where
    // any dodger picked Flee (Hold-to-Flee transition). Computed
    // from slot 4's action trajectory.
    let crossover = partial
        .per_tick_actions
        .iter()
        .enumerate()
        .find(|(_, row)| row[0] == FLEE)
        .map(|(i, _)| i);

    println!("==== threat_scoring_stresstest report ====");
    println!(
        "  composition: {} long-fuse + {} short-fuse casters + {} dodgers   ticks={}",
        N_LONG_FUSE, N_SHORT_FUSE, N_DODGERS, TICKS,
    );
    println!(
        "  mode AWARE   (weight=1.00) → Flee {aware_flee}/{max_cells} ({aware_pct:.1}%)  Hold {aware_hold}",
    );
    println!(
        "  mode BLIND   (weight=0.00) → Flee {blind_flee}/{max_cells} ({blind_pct:.1}%)  Hold {blind_hold}",
    );
    println!(
        "  mode PARTIAL (weight=0.05) → Flee {partial_flee}/{max_cells} ({partial_pct:.1}%)  Hold {partial_hold}  crossover_tick={crossover:?}",
    );
    println!("  slot 4 utility trajectory (mode=AWARE):");
    for (tick, u) in aware.per_tick_utility_slot4.iter().enumerate() {
        if tick % 4 == 0 || tick == TICKS as usize - 1 {
            println!("    tick {tick:3}: util={u:.3}");
        }
    }
    println!("  slot 4 utility trajectory (mode=PARTIAL):");
    for (tick, u) in partial.per_tick_utility_slot4.iter().enumerate() {
        if tick % 4 == 0 || tick == TICKS as usize - 1 {
            println!("    tick {tick:3}: util={u:.3}");
        }
    }
    println!("==========================================");

    // (1) Aware vs blind decision shift — the headline "threat impact
    // on scoring" metric.
    let delta = aware_flee.saturating_sub(blind_flee);
    let min_delta = (max_cells * 7) / 8; // ≥ 87.5% of the span
    assert!(
        delta >= min_delta,
        "decision shift between aware and blind should span ≥87.5% of the {max_cells} max; got delta={delta} (aware={aware_flee}, blind={blind_flee})"
    );

    // (2) Aware mode should pin to ~Flee-always (threats * 1.0
    // dominates Hold's 0.5 every tick after the first fold lands).
    assert!(
        aware_flee >= max_cells - 4,
        "aware mode should pick Flee in ≥{}/{} cells; got {aware_flee}",
        max_cells - 4, max_cells
    );

    // (3) Blind mode should pin to Hold-always.
    assert!(
        blind_flee == 0,
        "blind mode should pick Flee in 0 cells; got {blind_flee}"
    );

    // (4) Partial mode should show a crossover somewhere in the
    // middle of the run (not at tick 0, not never). Proves the
    // belief fold + decay shape modulates scoring as the trajectory
    // grows.
    match crossover {
        Some(t) => assert!(
            t > 0 && (t as u32) < TICKS,
            "partial-mode crossover should be in-run (0 < t < {TICKS}); got {t}"
        ),
        None => {
            // If no crossover at all, the partial-mode score never
            // exceeded Hold's 0.5 → either decay is too aggressive
            // or weight=0.05 is too low. Either way the
            // horizon-shape claim fails.
            panic!("partial-mode never crossed over; per_tick_actions={:?}", partial.per_tick_actions);
        }
    }
}

fn seed_topology(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    // creature_type — slots 0..4 are Caster (long-fuse + short-fuse),
    // slots 4..8 are Dodger. Matches .sim entity decl alphabetical:
    // Caster=0, Dodger=1.
    let mut creature_type: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..(N_LONG_FUSE + N_SHORT_FUSE) {
        creature_type.push(CT_CASTER);
    }
    for _ in 0..N_DODGERS {
        creature_type.push(CT_DODGER);
    }
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
}

/// Host-side patch: after tick 0, zero the busy SoA cells for the
/// short-fuse caster slots so only long-fuse contributions keep
/// folding from tick 1+. Same pattern threat_horizon_stresstest
/// uses.
fn clear_short_fuse_busy(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    let cast_ability_id = 1u32;
    let mut vals: Vec<u32> = vec![0u32; n];
    // Keep long-fuse slots stamped with their original ability id.
    for i in 0..N_LONG_FUSE {
        vals[i] = cast_ability_id;
    }
    // Short-fuse slots [N_LONG_FUSE .. N_LONG_FUSE+N_SHORT_FUSE)
    // stay at 0. Dodger slots also at 0 (no need to fold their own
    // busy state into the threat belief).
    state.gpu.queue.write_buffer(
        &state.agent_busy_with_ability_id_buf,
        0,
        bytemuck::cast_slice(&vals),
    );
}

/// Read scoring_output: 4 u32 per agent: [best_action, best_target,
/// bitcast<u32>(best_utility), _]. Returns a flat slice of
/// `4 * N_TOTAL` u32s for the caller to index.
fn read_scoring_output(state: &mut GeneratedRuntime) -> Vec<u32> {
    let count = (N_TOTAL as usize) * 4;
    let bytes = (count as u64) * 4;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threat_scoring_stresstest::scoring_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threat_scoring_stresstest::scoring_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(&state.scoring_output_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
