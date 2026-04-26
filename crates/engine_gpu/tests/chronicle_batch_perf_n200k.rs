//! Batch-path perf test at N=200,000 agents.
//!
//! The sync path at N=200k (measured in `perf_n100.rs`) runs ~6.25
//! sec/tick on RTX 4090, with the GPU cascade phase dominating. This
//! test exercises the batch path under the same workload to establish
//! a batch reference at the intended production scale.
//!
//! Scope:
//!   - Spawn the same interleaved combat fixture as `perf_n100.rs`
//!     (humans 40% / wolves 40% / deer 20%, ~0.1 agents/unit² density
//!     so every agent has neighbours within attack range from tick 0).
//!   - One sync warmup tick (pipeline compile + resident init).
//!   - One prime snapshot so the double-buffer is populated.
//!   - `step_batch(TICKS)` with wall-clock timing.
//!   - Triple-snapshot dance to read the post-batch state.
//!   - Assert: chronicle emits non-empty (guards the Task #68 bug
//!     recurring at scale); dump per-tick µs, event counts, chronicle
//!     counts so regressions show up as obvious number deltas in
//!     `cargo test -- --nocapture`.
//!
//! GPU-only — no CPU parity comparison (CPU at N=200k takes hours per
//! tick, per `perf_n100.rs`). This test is the batch-path analogue of
//! that file.
//!
//! Marked `#[ignore]` by default because the test takes ~30-60 seconds
//! of wall clock (mostly the sync warmup tick at N=200k plus the
//! batch itself). Run explicitly with:
//!
//! ```
//! cargo test --release --features gpu -p engine_gpu \
//!     --test chronicle_batch_perf_n200k -- --ignored --nocapture
//! ```

#![cfg(feature = "gpu")]

use std::time::Instant;

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xC0FFEE_B001_BABE_42;
const N_AGENTS: u32 = 200_000;
const AGENT_CAP: u32 = N_AGENTS + 8;
const TICKS: u32 = 50;

fn spawn_crowd() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    // Same geometry as perf_n100 / chronicle_batch_stress_n20k:
    // xorshift-jittered positions in a square sized so density is
    // ~0.1 agents/unit² (≤32 per 16m spatial cell), species cycled
    // 40% humans / 40% wolves / 20% deer.
    let area_side = (N_AGENTS as f32 * 10.0).sqrt().ceil();
    let mut rng_state = SEED;
    let mut xs_next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };
    for i in 0..N_AGENTS {
        let rx = xs_next();
        let ry = xs_next();
        let x = (rx as f32 / u64::MAX as f32) * area_side - area_side * 0.5;
        let y = (ry as f32 / u64::MAX as f32) * area_side - area_side * 0.5;
        let species_pick = i % 5;
        let (ct, hp) = match species_pick {
            0 | 1 => (CreatureType::Human, 100.0),
            2 | 3 => (CreatureType::Wolf, 80.0),
            _ => (CreatureType::Deer, 60.0),
        };
        state
            .spawn_agent(AgentSpawn {
                creature_type: ct,
                pos: Vec3::new(x, y, 0.0),
                hp,
                ..Default::default()
            })
            .expect("spawn");
    }
    state
}

#[test]
#[ignore = "perf test — takes ~30-60s, run with --ignored --nocapture"]
fn chronicle_batch_perf_n200k() {
    let mut gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("chronicle_batch_perf_n200k: GPU init failed — skipping ({e})");
            return;
        }
    };
    let t_spawn = Instant::now();
    let mut state = spawn_crowd();
    let spawn_ms = t_spawn.elapsed().as_millis();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1 << 22); // 4M event ring cap for 100k × 50 ticks
    let cascade = CascadeRegistry::with_engine_builtins();

    eprintln!("chronicle_batch_perf_n200k: N={N_AGENTS} spawn={spawn_ms}ms backend={}", gpu.backend_label());

    // Warmup sync step: pipeline compile + resident init + chronicle
    // ring alloc. This is expensive at N=200k (~5-6 sec on 4090) but
    // unavoidable — the first batch tick would otherwise absorb the
    // same cost. Separating it makes the batch timing clean.
    let t_warmup = Instant::now();
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let warmup_ms = t_warmup.elapsed().as_millis();
    eprintln!("  warmup sync step: {warmup_ms} ms");

    // Prime the snapshot double-buffer so the first post-batch
    // snapshot() call returns the batch's state (not warmup state).
    let _prime = gpu.snapshot(&mut state).expect("prime snapshot");

    // Perf Stage A.2 — snapshot BG-cache counters BEFORE the batch so
    // the delta isolates the batch's cache behaviour from the warmup
    // sync step.
    let (hits_before, misses_before) = gpu
        .physics_resident_bg_cache_stats()
        .unwrap_or((0, 0));

    // Timed batch — split first tick from the rest to surface any
    // first-tick JIT / pipeline-compile / buffer-state-transition
    // cost that would otherwise be averaged into a misleading mean.
    let t_first = Instant::now();
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        1,
    );
    let first_tick_ms = t_first.elapsed().as_millis();
    eprintln!("  step_batch(1) first tick: {first_tick_ms} ms");

    let t_batch = Instant::now();
    let remaining = TICKS - 1;
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        remaining,
    );
    let batch_elapsed = t_batch.elapsed();
    let batch_ms = batch_elapsed.as_millis();
    let batch_us_per_tick = batch_elapsed.as_micros() / remaining as u128;
    eprintln!(
        "  step_batch({remaining}) steady-state: total={batch_ms} ms, avg={batch_us_per_tick} µs/tick"
    );

    // Perf Stage A.1 — per-phase GPU µs breakdown. Empty on adapters
    // without TIMESTAMP_QUERY; non-empty output on a real discrete GPU.
    //
    // Per-dispatch attribution (2026-04-24): labels prefixed "gap:"
    // mark between-pass boundaries; the delta between consecutive
    // marks attributes GPU-µs to the bracketed work (dispatch +
    // barrier + pipeline swap). Legacy per-phase labels (no prefix)
    // ride alongside for regression continuity with Stage A.
    if gpu.gpu_profiler_enabled() {
        let phases = gpu.last_batch_phase_us();
        if phases.is_empty() {
            eprintln!("  gpu timestamps: profiler enabled but no samples produced");
        } else {
            eprintln!("  gpu timestamps (µs/tick, mean over {TICKS} ticks):");
            // Print gap: marks first (per-dispatch attribution table),
            // then legacy per-phase marks, so the research doc picks up
            // the gap table cleanly at the top of the output.
            eprintln!("    --- per-dispatch gap table ---");
            for (label, total_us) in phases {
                if label.starts_with("gap:") {
                    let per_tick = *total_us / TICKS as u64;
                    eprintln!("    {label:40}: {per_tick}");
                }
            }
            eprintln!("    --- legacy per-phase marks ---");
            for (label, total_us) in phases {
                if !label.starts_with("gap:") {
                    let per_tick = *total_us / TICKS as u64;
                    eprintln!("    {label:40}: {per_tick}");
                }
            }
        }
    } else {
        eprintln!("  gpu timestamps: TIMESTAMP_QUERY unavailable on this adapter (CPU wall-clock only)");
    }

    // Submit-granularity sweep mode: echo the active granularity so
    // the research doc can tie the output block to a specific K.
    match std::env::var("ENGINE_GPU_SUBMIT_GRANULARITY").ok() {
        Some(k) => eprintln!("  submit_granularity: K={k} (sub-batches of K ticks each)"),
        None => eprintln!("  submit_granularity: default (1 submit per step_batch)"),
    }

    // Perf Stage A.2 — bind-group cache stats.
    if let Some((hits_after, misses_after)) =
        gpu.physics_resident_bg_cache_stats()
    {
        let hits = hits_after.saturating_sub(hits_before);
        let misses = misses_after.saturating_sub(misses_before);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        };
        eprintln!(
            "  resident_bg_cache: {misses} misses, {hits} hits ({hit_rate:.1}% hit rate, {total} lookups)"
        );
    }

    // Triple-snapshot dance.
    let t_snap = Instant::now();
    let _swap = gpu.snapshot(&mut state).expect("swap snapshot");
    let snap = gpu.snapshot(&mut state).expect("read snapshot");
    let snap_ms = t_snap.elapsed().as_millis();

    let mut attack_events = 0usize;
    let mut died_events = 0usize;
    for e in snap.events_since_last.iter() {
        match e.kind {
            1 => attack_events += 1,
            2 => died_events += 1,
            _ => {}
        }
    }

    let chronicle_attacks = snap
        .chronicle_since_last
        .iter()
        .filter(|r| r.template_id == 2)
        .count();

    eprintln!("  snapshot dance: {snap_ms} ms");
    eprintln!(
        "  post-batch: tick={} events_since_last={} chronicle_since_last={}",
        snap.tick,
        snap.events_since_last.len(),
        snap.chronicle_since_last.len()
    );
    eprintln!("    AgentAttacked: {attack_events}");
    eprintln!("    AgentDied:     {died_events}");
    eprintln!("    chronicle_attack records: {chronicle_attacks}");
    eprintln!(
        "    alive agents:  {}",
        state.agents_alive().count()
    );

    // Summary line for easy greppability in bench comparisons.
    eprintln!(
        "=== PERF N100k batch: {batch_us_per_tick} µs/tick ({TICKS} ticks, {attack_events} attacks) ==="
    );

    // Correctness guard (not a perf assertion): chronicle ring must
    // emit records at N=200k if it does at N=20k. If this fails, the
    // scale-up surfaces a per-iteration regression we don't see in
    // smaller tests.
    assert!(
        !snap.chronicle_since_last.is_empty(),
        "expected chronicle_since_last non-empty after step_batch({TICKS}) \
         on N={N_AGENTS}; got 0. AgentAttacked events observed: {attack_events}"
    );
    assert!(
        chronicle_attacks > 0,
        "expected >=1 template_id=2 (chronicle_attack) record at N={N_AGENTS}; got 0"
    );
}
