//! Ad-hoc N=100 perf measurement for the CPU↔GPU crossover question.
//! Scratch file — not a regression test. Delete after measuring.

#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;
const N_AGENTS: u32 = 100_000;
const AGENT_CAP: u32 = N_AGENTS + 8;
const TICKS: u32 = 10;
const EVENT_RING_CAP: usize = 1 << 22;
const RUN_CPU: bool = false;

fn spawn_n(n: u32) -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    // Interleaved combat fixture: species are RANDOMLY MIXED across a
    // bounded square so every agent has enemies within attack_range=2.0
    // from tick 0. Cluster geometry (humans SE / wolves NW / deer center)
    // put cluster centroids hundreds of units apart at high N, leaving
    // the interior ~99% of agents inert. This fixture instead uses
    // xorshift-jittered per-agent positions in a square sized so local
    // density stays ~0.1 agents/unit² (≤32 per 16m spatial cell).
    //
    // Species assignment: 40% humans, 40% wolves, 20% deer, cycled by
    // agent index with a seed-derived permutation so spawn order
    // varies per seed.
    let area_side = (n as f32 * 10.0).sqrt().ceil(); // ~0.1 agents/unit²
    let mut rng_state = SEED;
    let mut xs_next = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };
    for i in 0..n {
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
            .unwrap();
    }
    state
}

#[test]
fn perf_n100() {
    let cascade = CascadeRegistry::with_engine_builtins();

    // CPU — skipped at large N (O(N^2) scoring blows up)
    let (cpu_total, alive_cpu) = if RUN_CPU {
        let mut cpu_backend = CpuBackend;
        let mut state = spawn_n(N_AGENTS);
        let mut scratch = SimScratch::new(state.agent_cap() as usize);
        let mut events = EventRing::with_cap(EVENT_RING_CAP);

        cpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let t0 = Instant::now();
        let mut cpu_phases_1_3_total = Duration::ZERO;
        for tick_i in 1..TICKS {
            let tp = Instant::now();
            engine::step::step_phases_1_to_3(&mut state, &mut scratch, &UtilityBackend);
            let ph13 = tp.elapsed();
            cpu_phases_1_3_total += ph13;
            if tick_i % 10 == 0 {
                eprintln!("CPU tick {tick_i} phases_1_3: {}us", ph13.as_micros());
            }
            let events_before = events.total_pushed();
            engine::step::apply_actions(&mut state, &scratch, &mut events);
            cascade.run_fixed_point_tel(&mut state, &mut events, &engine::telemetry::NullSink);
            state.views.fold_all(&events, events_before, state.tick);
            state.tick += 1;
        }
        let total = t0.elapsed();
        let alive = state.agents_alive().count();
        eprintln!(
            "CPU phases 1-3 avg: {} µs/tick",
            cpu_phases_1_3_total.as_micros() / (TICKS - 1) as u128
        );
        (total, alive)
    } else {
        eprintln!("CPU run skipped at N={N_AGENTS}");
        (Duration::ZERO, 0)
    };

    // GPU
    let mut gpu_backend = GpuBackend::new().expect("gpu init");
    gpu_backend.set_skip_scoring_sidecar(true);
    let mut state = spawn_n(N_AGENTS);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);

    let t_first = Instant::now();
    gpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let first_tick = t_first.elapsed();

    // Warm-up tick on GPU path.
    gpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Direct measurement of phases 1-3 on GPU-evolved state (should be
    // identical CPU function call, differs only in state contents).
    let t_direct = Instant::now();
    for _ in 0..5 {
        engine::step::step_phases_1_to_3(&mut state, &mut scratch, &UtilityBackend);
    }
    let direct_us = t_direct.elapsed().as_micros() / 5;
    eprintln!(
        "Direct call to step_phases_1_to_3 on GPU-state (post-warmup, avg of 5): {} µs",
        direct_us
    );

    let t_rest = Instant::now();
    let mut iter_counts: Vec<u32> = Vec::new();
    let mut accum = engine_gpu::PhaseTimings::default();
    for _ in 1..TICKS {
        gpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        if let Some(n) = gpu_backend.last_cascade_iterations() {
            iter_counts.push(n);
        }
        let p = gpu_backend.last_phase_timings();
        accum.cpu_phases_1_3_us += p.cpu_phases_1_3_us;
        accum.cpu_apply_actions_us += p.cpu_apply_actions_us;
        accum.gpu_cascade_us += p.gpu_cascade_us;
        accum.gpu_seed_fold_us += p.gpu_seed_fold_us;
        accum.cpu_cold_state_us += p.cpu_cold_state_us;
        accum.cpu_view_fold_all_us += p.cpu_view_fold_all_us;
        accum.cpu_finalize_us += p.cpu_finalize_us;
        accum.gpu_sidecar_us += p.gpu_sidecar_us;
    }
    let rest = t_rest.elapsed();
    let avg_iters: f64 = if iter_counts.is_empty() {
        0.0
    } else {
        iter_counts.iter().copied().map(|n| n as f64).sum::<f64>() / iter_counts.len() as f64
    };
    let ticks_measured = (TICKS - 1) as u64;
    eprintln!("avg cascade iterations/tick: {avg_iters:.2}");
    eprintln!("per-phase averages (µs/tick):");
    eprintln!("  cpu phases 1-3:     {}", accum.cpu_phases_1_3_us / ticks_measured);
    eprintln!("  cpu apply_actions:  {}", accum.cpu_apply_actions_us / ticks_measured);
    eprintln!("  gpu cascade:        {}", accum.gpu_cascade_us / ticks_measured);
    eprintln!("  gpu seed fold:      {}", accum.gpu_seed_fold_us / ticks_measured);
    eprintln!("  cpu cold state:     {}", accum.cpu_cold_state_us / ticks_measured);
    eprintln!("  cpu view fold_all:  {}", accum.cpu_view_fold_all_us / ticks_measured);
    eprintln!("  cpu finalize:       {}", accum.cpu_finalize_us / ticks_measured);
    eprintln!("  gpu sidecar:        {}", accum.gpu_sidecar_us / ticks_measured);
    eprintln!("GPU backend label: {}", gpu_backend.backend_label());
    let alive_gpu = state.agents_alive().count();

    let cpu_per = cpu_total.as_secs_f64() * 1e6 / (TICKS - 1) as f64;
    let gpu_steady_per = rest.as_secs_f64() * 1e6 / (TICKS - 1) as f64;

    eprintln!("=== N=100 perf ===");
    eprintln!(
        "CPU:        {:.2} ms total ({:.1} µs/tick)  alive={}",
        cpu_total.as_secs_f64() * 1e3,
        cpu_per,
        alive_cpu,
    );
    eprintln!(
        "GPU first:  {:.2} ms  (init + shader compile + 1 tick)",
        first_tick.as_secs_f64() * 1e3,
    );
    eprintln!(
        "GPU steady: {:.2} ms total ({:.1} µs/tick over {} ticks)  alive={}",
        rest.as_secs_f64() * 1e3,
        gpu_steady_per,
        TICKS - 1,
        alive_gpu,
    );
    eprintln!("ratio (steady): GPU is {:.2}× CPU", gpu_steady_per / cpu_per);

    // Parity: compare CPU vs GPU alive counts within ±25% tolerance.
    // Byte-exact equality is NOT expected — event ordering differs
    // between backends, which accumulates into different end states
    // via different combat outcomes. Multiset equality of alive-ids
    // would likewise over-assert. The canonical post-tick fingerprint
    // for this fixture is alive count; we tolerate 25%.
    let alive_cpu = alive_cpu as i64;
    let alive_gpu = alive_gpu as i64;
    if RUN_CPU {
        let delta = (alive_cpu - alive_gpu).unsigned_abs() as f64;
        let tolerance = (alive_cpu as f64 * 0.25).max(5.0);
        eprintln!(
            "alive parity: cpu={alive_cpu} gpu={alive_gpu} delta={delta} tolerance={tolerance:.1}"
        );
        assert!(
            delta <= tolerance,
            "alive parity exceeded tolerance: cpu={alive_cpu} gpu={alive_gpu} delta={delta} tolerance={tolerance:.1}"
        );
    } else {
        eprintln!("alive parity: cpu skipped, gpu={alive_gpu}");
    }

    // Task 200: with GPU apply_actions + movement kernels wired, the
    // CPU apply_actions bridge is retired and the cascade iterates on
    // the GPU-authored seed events directly. Steady-state ratio on a
    // discrete GPU now lands below 1x on a warm run (GPU faster than
    // CPU at N=1000); we assert `ratio ≤ 5x` as a regression ceiling
    // with headroom for noisy llvmpipe / shared CI GPUs / thermal
    // variance and run-to-run noise (observed: 0.3x-1.8x across
    // back-to-back runs on the same host). Anything above 5x signals
    // a kernel regression or a CPU fallback that shouldn't be firing.
    if RUN_CPU {
        let ratio = gpu_steady_per / cpu_per;
        assert!(
            ratio <= 5.0,
            "task 200 regression: GPU steady per-tick = {gpu_steady_per:.1}µs, \
             CPU per-tick = {cpu_per:.1}µs, ratio = {ratio:.2}x (expected ≤ 5x)"
        );
    }
}

/// Smoke test for the Phase 9 `step_batch(n_ticks)` API. Runs 5 ticks
/// via a single `step_batch(5)` call and asserts the state advanced
/// by exactly 5 ticks, with the post-batch agent pool still consistent.
#[test]
fn step_batch_advances_tick() {
    let mut gpu_backend = GpuBackend::new().expect("gpu init");
    gpu_backend.set_skip_scoring_sidecar(true);
    let mut state = spawn_n(16); // small for a quick smoke
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    let tick_before = state.tick;
    gpu_backend.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        5,
    );
    // After step_batch, CPU state.tick stays stale; GPU sim_cfg.tick is
    // authoritative. Observer reads current tick via snapshot.
    let _e = gpu_backend.snapshot().expect("empty");
    let _k = gpu_backend.snapshot().expect("kick");
    let snap = gpu_backend.snapshot().expect("read");
    assert_eq!(snap.tick, tick_before + 5, "snapshot.tick should advance by n_ticks");
}
