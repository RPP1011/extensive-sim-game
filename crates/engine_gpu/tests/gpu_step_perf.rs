// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! Task 193 (Phase 6g) perf smoke — CPU vs GPU `step()` wall-clock on the
//! canonical 3h+2w fixture, N=8 agents. Not a benchmark; just a
//! diagnostic the task's report can cite. Runs under the normal test
//! harness so CI surfaces it if the gap ever inverts.
//!
//! Expectations (documented here because the test doesn't assert on
//! them): at N=8 the GPU step is slower than the CPU step — wgpu
//! dispatch overhead + buffer upload / readback + DSL-load-once-on-
//! first-tick swamps the ~microsecond CPU step. Break-even is in the
//! hundreds-of-agents range, which the later scale-up pass targets.

#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use engine::backend::{CpuBackend, ComputeBackend};
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xD00D_FACE_0042_0042;
const TICKS: u32 = 50;
const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

fn spawn_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(0.0, 0.0, 0.0), hp: 100.0, ..Default::default() }).unwrap();
    state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(2.0, 0.0, 0.0), hp: 100.0, ..Default::default() }).unwrap();
    state.spawn_agent(AgentSpawn { creature_type: CreatureType::Human, pos: Vec3::new(-2.0, 0.0, 0.0), hp: 100.0, ..Default::default() }).unwrap();
    state.spawn_agent(AgentSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(3.0, 0.0, 0.0), hp: 80.0, ..Default::default() }).unwrap();
    state.spawn_agent(AgentSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(-3.0, 0.0, 0.0), hp: 80.0, ..Default::default() }).unwrap();
    state
}

fn time_cpu_step_loop(n_ticks: u32) -> Duration {
    let mut backend = CpuBackend;
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    let t0 = Instant::now();
    for _ in 0..n_ticks {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    t0.elapsed()
}

fn time_gpu_step_loop(n_ticks: u32) -> (Duration, Duration) {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warm up — first step pays the cascade-ctx init + shader compile.
    // Include that cost in the reported `with_init_ms` number; the
    // steady-state `steady_ms` strips it.
    let t0 = Instant::now();
    backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let first_tick = t0.elapsed();

    let t1 = Instant::now();
    for _ in 1..n_ticks {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    let remaining = t1.elapsed();
    let total = first_tick + remaining;
    (total, remaining)
}

#[test]
fn gpu_vs_cpu_step_perf_smoke_n8() {
    // Prime the filesystem / page cache so the asset read isn't
    // the dominant cost; repeat a handful of runs and take the
    // median to dampen jitter.
    const RUNS: usize = 3;
    let mut cpu_times: Vec<Duration> = Vec::with_capacity(RUNS);
    let mut gpu_totals: Vec<Duration> = Vec::with_capacity(RUNS);
    let mut gpu_steadys: Vec<Duration> = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        cpu_times.push(time_cpu_step_loop(TICKS));
        let (total, steady) = time_gpu_step_loop(TICKS);
        gpu_totals.push(total);
        gpu_steadys.push(steady);
    }
    cpu_times.sort();
    gpu_totals.sort();
    gpu_steadys.sort();
    let cpu_med = cpu_times[RUNS / 2];
    let gpu_total_med = gpu_totals[RUNS / 2];
    let gpu_steady_med = gpu_steadys[RUNS / 2];

    eprintln!(
        "perf_smoke_n8: CPU {} ticks = {:.2} ms ({:.1} µs/tick)",
        TICKS,
        cpu_med.as_secs_f64() * 1e3,
        cpu_med.as_secs_f64() * 1e6 / TICKS as f64,
    );
    eprintln!(
        "perf_smoke_n8: GPU {} ticks with-init = {:.2} ms; steady ({} ticks) = {:.2} ms ({:.1} µs/tick)",
        TICKS,
        gpu_total_med.as_secs_f64() * 1e3,
        TICKS - 1,
        gpu_steady_med.as_secs_f64() * 1e3,
        gpu_steady_med.as_secs_f64() * 1e6 / (TICKS - 1) as f64,
    );
    let ratio =
        gpu_steady_med.as_secs_f64() / cpu_med.as_secs_f64() * TICKS as f64 / (TICKS - 1) as f64;
    eprintln!("perf_smoke_n8: GPU/CPU per-tick ratio (steady) = {ratio:.1}x");

    // Not an assertion target — the smoke test just needs to run,
    // not pass a threshold. Logging is the deliverable.
}
