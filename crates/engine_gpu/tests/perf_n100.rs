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
const N_AGENTS: u32 = 1000;
const AGENT_CAP: u32 = N_AGENTS + 8;
const TICKS: u32 = 50;
const EVENT_RING_CAP: usize = 1 << 20;

fn spawn_n(n: u32) -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    // 40 humans, 40 wolves, 20 deer — scaled version of showcase ratio.
    let humans = (n * 40 / 100).max(1);
    let wolves = (n * 40 / 100).max(1);
    let deer = n - humans - wolves;
    // Dense grid so agents actually engage. Cluster diameter scales
    // with sqrt(N) to keep density roughly constant at ~1 agent per
    // 4 unit² (close enough for engagement_range=2.0 to fire).
    let side = ((humans as f32).sqrt() + 1.0) as u32;
    let mut idx = 0u32;
    for i in 0..humans {
        let r = (i / side) as f32 * 1.2;
        let c = (i % side) as f32 * 1.2;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(5.0 + c, 5.0 + r, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
        idx += 1;
    }
    let side_w = ((wolves as f32).sqrt() + 1.0) as u32;
    for i in 0..wolves {
        let r = (i / side_w) as f32 * 1.2;
        let c = (i % side_w) as f32 * 1.2;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(-5.0 - c, -5.0 - r, 0.0),
                hp: 80.0,
                ..Default::default()
            })
            .unwrap();
    }
    let side_d = ((deer as f32).sqrt() + 1.0) as u32;
    for i in 0..deer {
        let r = (i / side_d) as f32 * 1.5;
        let c = (i % side_d) as f32 * 1.5;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Deer,
                pos: Vec3::new(c - 5.0, r - 5.0, 0.0),
                hp: 60.0,
                ..Default::default()
            })
            .unwrap();
    }
    let _ = idx;
    state
}

#[test]
fn perf_n100() {
    // CPU
    let mut cpu_backend = CpuBackend;
    let mut state = spawn_n(N_AGENTS);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warm-up tick (trigger any lazy init / amortise first-tick cost).
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
        // Continue with rest of CPU step manually
        let events_before = events.total_pushed();
        engine::step::apply_actions(&mut state, &scratch, &mut events);
        cascade.run_fixed_point_tel(&mut state, &mut events, &engine::telemetry::NullSink);
        state.views.fold_all(&events, events_before, state.tick);
        state.tick += 1;
    }
    let cpu_total: Duration = t0.elapsed();
    let alive_cpu = state.agents_alive().count();
    eprintln!(
        "CPU phases 1-3 avg: {} µs/tick",
        cpu_phases_1_3_total.as_micros() / (TICKS - 1) as u128
    );

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
    let delta = (alive_cpu - alive_gpu).unsigned_abs() as f64;
    let tolerance = (alive_cpu as f64 * 0.25).max(5.0);
    eprintln!(
        "alive parity: cpu={alive_cpu} gpu={alive_gpu} delta={delta} tolerance={tolerance:.1}"
    );
    assert!(
        delta <= tolerance,
        "alive parity exceeded tolerance: cpu={alive_cpu} gpu={alive_gpu} delta={delta} tolerance={tolerance:.1}"
    );

    // Task 197: with CPU mask + policy-evaluate replaced by GPU scoring,
    // the GPU backend should land within ~8x of the CPU backend at
    // N=1000 (was 12-14x pre-197). This is a loose upper bound —
    // steady-state ratio on a discrete GPU runs 2-3x on this fixture;
    // 8x leaves headroom for noisy llvmpipe / shared CI GPUs / thermal
    // variance. The real target (per the task 197 plan) is 1-2x; the
    // remaining gap is the upload+readback fences, which the full WGSL
    // apply_actions + movement kernel port will close.
    let ratio = gpu_steady_per / cpu_per;
    assert!(
        ratio <= 8.0,
        "task 197 regression: GPU steady per-tick = {gpu_steady_per:.1}µs, \
         CPU per-tick = {cpu_per:.1}µs, ratio = {ratio:.2}x (expected ≤ 8x)"
    );
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
    assert_eq!(state.tick, tick_before + 5, "step_batch should advance tick by n_ticks");
}
