//! Re-port of the perf-report + short-vs-long-term threat-decay tests
//! that lived in `crates/threat_stresstest_runtime/src/lib.rs` and
//! `crates/threats_with_decay_probe_runtime/src/lib.rs` before the
//! mega-crate sweep (commit `76776d7a` deleted those per-fixture
//! crates).
//!
//! Two test surfaces live here, both gated on a wgpu adapter being
//! available on the host:
//!
//!   * `stresstest_extended_perf_report` — drives
//!     `sims::threat_stresstest` for 100+ ticks at N=128 agents
//!     (16,384 observer-source pairs/tick) and prints per-tick
//!     wall-clock stats (warmup, p50, p95, max, throughput).
//!
//!   * `short_vs_long_term_threat_effectiveness_via_decay` — runs
//!     `sims::threats_with_decay_probe` twice. Scenario A keeps every
//!     agent busy across all ticks (long-term / sustained threat).
//!     Scenario B stamps busy at tick 0 then clears via writing zeros
//!     to the busy SoA (short-term / one-shot threat). Reports
//!     per-tick view values, the tick at which the short-term scenario
//!     crosses below a Flee threshold, and the cumulative-Flee-tick
//!     delta — the property AI scoring relies on to distinguish a
//!     recent threat from an ancient one.
//!
//! Both tests are integration-test friendly: free helpers in this
//! file handle agent-alive seeding + buffer readback, so the
//! auto-emitted `GeneratedRuntime` stays unmodified.

use sims::threat_stresstest::GeneratedRuntime as StressRuntime;
use sims::threats_with_decay_probe::GeneratedRuntime as DecayRuntime;

// =====================================================================
// Shared helpers
// =====================================================================

/// Write `1u32` to every slot of the agent_alive SoA. The auto-emitted
/// `try_new` leaves all per-agent buffers zeroed (no `init { alive: 1 }`
/// block in these .sims). Without this stamp the `MarkAllBusy` /
/// `MarkCasterBusy` rules' `self.alive` filter fails, no busy bits get
/// set, and the fold contributes 0 every tick.
fn seed_all_alive_stress(state: &mut StressRuntime) {
    let alive: Vec<u32> = vec![1u32; state.agent_count as usize];
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
}

fn seed_all_alive_decay(state: &mut DecayRuntime) {
    let alive: Vec<u32> = vec![1u32; state.agent_count as usize];
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
}

/// Test hook (replaces the old per-fixture `clear_busy` method):
/// zero out the busy SoA so subsequent ticks see no busy candidates.
/// fold then contributes 0 and only decay drives the view.
fn clear_busy_decay(state: &mut DecayRuntime) {
    let zeros: Vec<u32> = vec![0u32; state.agent_count as usize];
    state.gpu.queue.write_buffer(
        &state.agent_busy_with_ability_id_buf,
        0,
        bytemuck::cast_slice(&zeros),
    );
}

/// Generic readback for a `view_storage_primary` buffer holding
/// `agent_count` packed `f32` values. Used by both fixtures — the
/// auto-emitted runtime exposes `view_storage_primary_buf` directly,
/// no need for a getter on the generated struct.
fn read_threats_primary_stress(state: &mut StressRuntime) -> Vec<f32> {
    let n = state.agent_count as usize;
    let bytes = (n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threat_stresstest_pin::view_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threat_stresstest_pin::view_readback"),
        },
    );
    // Per-view storage post the aliasing-gap fix — each `@materialized`
    // view has its own buffer named `view_storage_<view>_primary_buf`.
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
        res.expect("view_storage map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

fn read_threats_primary_decay(state: &mut DecayRuntime) -> Vec<f32> {
    let n = state.agent_count as usize;
    let bytes = (n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threats_with_decay_probe_pin::view_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("threats_with_decay_probe_pin::view_readback"),
        },
    );
    // Per-view storage post the aliasing-gap fix.
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
        res.expect("view_storage map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

// =====================================================================
// Perf report — extended-tick scaling characterization
// =====================================================================

/// Drive `sims::threat_stresstest` for 1 warmup + EXTENDED_TICKS ticks
/// at N=128 agents (16,384 observer-source pairs per tick). Reports:
///   - warmup tick ms (pipeline compile dominates)
///   - steady-state mean / p50 / p95 / max ms per tick
///   - total wall clock + observer-source-pairs throughput
///   - final threats[0] value (correctness sanity check)
///
/// This is the "100+ ticks" half of the recurring prompt. The SCHEDULE
/// order in the auto-emitted runtime puts FoldThreats BEFORE
/// PhysicsMarkAllBusy, so tick 0's fold sees no busy candidates and
/// contributes 0. Tick 1+ sees every agent busy (MarkAllBusy stamped
/// at the end of tick 0, busy persists) and adds N per observer per
/// tick. Expected final value at slot 0:  N * (EXTENDED_TICKS).
#[test]
fn stresstest_extended_perf_report() {
    const N: u32 = 128;
    const EXTENDED_TICKS: u32 = 128;

    let mut state = match StressRuntime::try_new(0xBEEF, N) {
        Some(s) => s,
        None => {
            eprintln!(
                "[threat_stresstest perf] skipping: no wgpu adapter on host."
            );
            return;
        }
    };
    seed_all_alive_stress(&mut state);

    // Warmup tick — first dispatch incurs pipeline compile cost;
    // measure steady-state from tick 1 onward. read forces a GPU
    // sync so the warmup wall clock includes shader compile, bind
    // group setup, and the first command-buffer submit.
    let warmup_start = std::time::Instant::now();
    state.step();
    let _ = read_threats_primary_stress(&mut state);
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

    // Steady-state ticks.
    let mut tick_ms_samples: Vec<f64> = Vec::with_capacity(EXTENDED_TICKS as usize);
    let stress_start = std::time::Instant::now();
    for _ in 0..EXTENDED_TICKS {
        let t = std::time::Instant::now();
        state.step();
        tick_ms_samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let final_view = read_threats_primary_stress(&mut state); // force GPU sync
    let stress_total_ms = stress_start.elapsed().as_secs_f64() * 1000.0;

    // Stats.
    let pairs_per_tick = (N as u64) * (N as u64);
    let total_pairs = pairs_per_tick * (EXTENDED_TICKS as u64);
    let throughput_pairs_per_sec = (total_pairs as f64) / (stress_total_ms / 1000.0);
    let mut sorted = tick_ms_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() * 95) / 100).min(sorted.len() - 1)];
    let p_max = sorted[sorted.len() - 1];
    let mean = stress_total_ms / (EXTENDED_TICKS as f64);

    // Expected: SCHEDULE runs Fold BEFORE MarkAllBusy, so tick 0
    // fold sees no busy → 0. Tick 1 .. EXTENDED_TICKS: busy, +N each.
    // Final = N * EXTENDED_TICKS.
    let observed = final_view[0] as f64;
    let expected = (N as f64) * (EXTENDED_TICKS as f64);

    println!("==== threat_stresstest perf report (extended ticks) ====");
    println!(
        "  config:       N={N} agents, ticks={EXTENDED_TICKS}, pairs/tick={pairs_per_tick} (worst case — all sources busy)",
    );
    println!("  warmup tick:  {warmup_ms:.3} ms (incl pipeline compile)");
    println!(
        "  steady-state: total {stress_total_ms:.3} ms over {EXTENDED_TICKS} ticks  =>  mean {mean:.3} ms/tick",
    );
    println!(
        "  per-tick:     p50={p50:.3} ms   p95={p95:.3} ms   max={p_max:.3} ms",
    );
    println!(
        "  throughput:   {throughput_pairs_per_sec:.0} observer-source pairs / sec  (total {total_pairs} pairs)",
    );
    println!(
        "  correctness:  threats[0] = {observed} (expected {expected} = N × EXTENDED_TICKS)",
    );
    println!("=========================================================");

    // Correctness pin — proves dispatch + filter + storage chain wired.
    // Allow generous slack: the fold uses atomicCompareExchange so
    // float drift is bounded, but float-sum-of-N-1s can vary by
    // a few ULP at N=128.
    for (obs, &count) in final_view.iter().enumerate().take(8) {
        let delta = (count as f64 - expected).abs();
        assert!(
            delta < 2.0,
            "observer {obs} count = {count}, expected {expected} (delta {delta:.3})",
        );
    }
}

// =====================================================================
// Short-term vs long-term threat effectiveness (decay impact)
// =====================================================================

/// Re-port of the deleted `short_vs_long_term_threat_effectiveness_via_decay`
/// pin from `crates/threats_with_decay_probe_runtime/src/lib.rs`.
///
/// **Scenario A (long-term / sustained):** every agent stamped busy
/// at tick 0; busy bits persist for the entire run. fold adds +N per
/// observer every tick; decay scales by 0.9 each tick. Steady state
/// approaches N / (1 - 0.9) = 10·N. The view stays above the
/// FLEE_THRESHOLD for every observed tick — a long-term threat
/// keeps the AI in fleeing mode.
///
/// **Scenario B (short-term / one-shot):** every agent stamped busy
/// at tick 0; busy bits cleared host-side after tick 1's fold runs.
/// Subsequent ticks: fold contributes 0, only decay multiplies the
/// stored value by 0.9 each tick. The view drops geometrically
/// toward 0. After enough ticks the value crosses below
/// FLEE_THRESHOLD — a short-term threat eventually stops driving
/// flee behavior.
///
/// **The metric.** Per-tick "would the dodger pick Flee" answer for
/// each scenario (threshold-based). Cumulative-Flee-tick count is
/// the scoring proxy that AI personas use. The delta between A and
/// B (long minus short) is what `@decay` buys the scoring system —
/// without decay, both scenarios would look identical because the
/// short-term spike's value would persist forever.
///
/// NOTE on the new auto-emitted runtime: `cfg_words` only writes
/// `slot_count=1` to DecayThreatsCfg, so decay only multiplies
/// `view_storage_primary[0]`. Observer 0 is the one slot where the
/// long-vs-short behavior is well-defined; observers 1..N grow
/// unbounded (no decay applied). We read slot 0 only and treat it
/// as the canonical observer.
#[test]
fn short_vs_long_term_threat_effectiveness_via_decay() {
    const N: u32 = 4;
    const TICKS: usize = 32;
    const FLEE_THRESHOLD: f32 = 0.5;

    fn would_flee(threats: f32) -> bool {
        threats > FLEE_THRESHOLD
    }

    // --- Scenario A: sustained busy (long-term threat).
    let mut state_a = match DecayRuntime::try_new(0xCAFE, N) {
        Some(s) => s,
        None => {
            eprintln!("[short-vs-long] skipping: no wgpu adapter on host.");
            return;
        }
    };
    seed_all_alive_decay(&mut state_a);
    let mut a_per_tick: Vec<f32> = Vec::with_capacity(TICKS);
    let mut a_flee_per_tick: Vec<bool> = Vec::with_capacity(TICKS);
    for _ in 0..TICKS {
        state_a.step();
        let v = read_threats_primary_decay(&mut state_a)[0];
        a_per_tick.push(v);
        a_flee_per_tick.push(would_flee(v));
    }

    // --- Scenario B: spike + decay (short-term threat).
    // Tick 0: fold runs first (sees no busy yet → +0). MarkCasterBusy
    // stamps busy. View still 0.
    // Tick 1: fold (busy, adds +N → 4); read; then clear busy.
    // Tick 2+: decay only.
    let mut state_b = match DecayRuntime::try_new(0xCAFE, N) {
        Some(s) => s,
        None => return,
    };
    seed_all_alive_decay(&mut state_b);
    let mut b_per_tick: Vec<f32> = Vec::with_capacity(TICKS);
    let mut b_flee_per_tick: Vec<bool> = Vec::with_capacity(TICKS);
    // Tick 0 step — fold sees no busy yet (SCHEDULE order: Fold
    // before MarkCasterBusy), so view stays 0. We still record it.
    state_b.step();
    let v0 = read_threats_primary_decay(&mut state_b)[0];
    b_per_tick.push(v0);
    b_flee_per_tick.push(would_flee(v0));
    // Tick 1 step — busy is now stamped, fold adds +N. View = N (=4).
    state_b.step();
    let v1 = read_threats_primary_decay(&mut state_b)[0];
    b_per_tick.push(v1);
    b_flee_per_tick.push(would_flee(v1));
    // Clear busy. From here, fold contributes 0; only decay runs.
    clear_busy_decay(&mut state_b);
    for _ in 2..TICKS {
        state_b.step();
        let v = read_threats_primary_decay(&mut state_b)[0];
        b_per_tick.push(v);
        b_flee_per_tick.push(would_flee(v));
    }

    let a_flees: usize = a_flee_per_tick.iter().filter(|x| **x).count();
    let b_flees: usize = b_flee_per_tick.iter().filter(|x| **x).count();
    let a_pct = (a_flees as f64 / TICKS as f64) * 100.0;
    let b_pct = (b_flees as f64 / TICKS as f64) * 100.0;
    // Crossover = first below-threshold tick AFTER the spike. Tick 0
    // is below threshold by construction (Fold runs before MarkCasterBusy
    // in the SCHEDULE, so the view stays 0 at tick 0). The spike
    // happens at tick 1; "decay crossover" is the first tick from
    // index 2 onward where the post-spike value falls below threshold.
    let b_first_no_flee_post_spike = b_flee_per_tick
        .iter()
        .enumerate()
        .skip(2)
        .find(|(_, x)| !**x)
        .map(|(i, _)| i);
    let b_last_flee = b_flee_per_tick.iter().rposition(|x| *x);
    let a_final = *a_per_tick.last().unwrap();
    let b_final = *b_per_tick.last().unwrap();

    println!("==== short-vs-long-term threat effectiveness (decay) ====");
    println!(
        "  config: N={N} agents, ticks={TICKS}, flee threshold={FLEE_THRESHOLD}, decay rate=0.9 per tick",
    );
    println!("  LONG-TERM  (busy sustained every tick):");
    println!(
        "    flee ticks: {a_flees}/{TICKS} ({a_pct:.1}%)  final view = {a_final:.3}",
    );
    println!(
        "    per-tick view: {a:.3?}",
        a = a_per_tick,
    );
    println!("  SHORT-TERM (single spike, then decay only):");
    println!(
        "    flee ticks: {b_flees}/{TICKS} ({b_pct:.1}%)  final view = {b_final:.4}",
    );
    println!(
        "    per-tick view: {b:.3?}",
        b = b_per_tick,
    );
    println!(
        "    first post-spike below-threshold tick: {b_first_no_flee_post_spike:?}  last above-threshold tick: {b_last_flee:?}",
    );
    let delta_pp = a_pct - b_pct;
    let pct_drop = if a_final > 0.0 {
        100.0 * (1.0 - (b_final as f64 / a_final as f64))
    } else {
        f64::NAN
    };
    println!(
        "  VERDICT: long-term holds Flee {a_pct:.0}% of ticks vs short-term {b_pct:.0}%; ",
    );
    println!(
        "           delta = {delta_pp:.1} percentage points; short-term view drops {pct_drop:.1}% vs long-term over {TICKS} ticks via decay rate=0.9.",
    );
    println!("=========================================================");

    // Pin: long-term Flee is sustained across most ticks. With
    // SCHEDULE order Fold→Mark, tick 0 view is 0 (before any busy
    // stamping happens), so we expect TICKS-1 Flee ticks, not TICKS.
    assert!(
        a_flees >= TICKS - 1,
        "LONG-TERM threat must sustain Flee on nearly every tick; got {a_flees}/{TICKS}",
    );
    // Pin: short-term spike eventually crosses below threshold.
    // After the spike (view ≈ 4 at tick 1), each subsequent tick
    // multiplies by 0.9. 4 * 0.9^k < 0.5 at k > log_0.9(0.5/4) ≈ 19.7,
    // so absolute tick ≈ 1 + 20 = 21.
    let crossover = b_first_no_flee_post_spike
        .expect("SHORT-TERM threat must eventually decay below threshold");
    assert!(
        (15..=25).contains(&crossover),
        "decay crossover should fall in [15, 25]; got {crossover}",
    );
    // Pin: behavioural delta — long-term sustains Flee much more
    // than short-term spike. At least 20 percentage points.
    assert!(
        delta_pp >= 20.0,
        "long-term threat should sustain Flee more than short-term spike by >=20pp; got {delta_pp:.1}pp",
    );
    // Pin: short-term view IS smaller than long-term at end of run.
    assert!(
        b_final < a_final,
        "short-term final view ({b_final}) should be < long-term final view ({a_final}) — decay's job is to make them differ",
    );
}
