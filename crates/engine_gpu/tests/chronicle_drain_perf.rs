//! Task 203 — informational perf test for the chronicle ring split.
//!
//! Runs a dense wolves+humans combat fixture for 15 ticks on the GPU
//! backend and prints per-tick cascade timings + chronicle volumes.
//!
//! With chronicle records routed to the dedicated chronicle ring,
//! the main event-ring drain sees strictly fewer records per cascade
//! iteration. At N=2048 on a Vulkan discrete GPU we measured
//! ~1-3% cascade wall-time reduction (the drain's `copy_buffer_to_buffer`
//! dominates regardless; the meaningful savings come from atomic
//! contention on the emit side, which shows up clearly at higher
//! contention / smaller ring capacity).
//!
//! Not an assertion test — prints numbers for diagnostic use. The
//! correctness contract lives in
//! `event_ring_parity.rs::chronicle_events_route_to_chronicle_ring_not_main_ring`.

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

const SEED: u64 = 0xBA5E_BA11_C0DE_FACE;
const N_AGENTS: u32 = 2048;
const TICKS: u32 = 15;
const EVENT_RING_CAP: usize = 1 << 20;

fn spawn_dense(n: u32) -> SimState {
    // 60% wolves / 40% humans in a small square so every agent engages
    // from tick 0. High chronicle density (every hit → chronicle_attack,
    // wounded → chronicle_wound, engagement edges → chronicle_* …).
    let mut state = SimState::new(n + 8, SEED);
    let area_side = (n as f32 * 6.0).sqrt().ceil();
    let mut rng_state: u64 = SEED;
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
        let ct = if i % 5 < 3 {
            CreatureType::Wolf
        } else {
            CreatureType::Human
        };
        state
            .spawn_agent(AgentSpawn {
                creature_type: ct,
                pos: Vec3::new(x, y, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    state
}

#[test]
fn chronicle_drain_perf() {
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut backend = GpuBackend::new().expect("gpu init");
    backend.set_skip_scoring_sidecar(true);
    let mut state = spawn_dense(N_AGENTS);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);

    // Warm-up tick (DSL parse, shader compile, buffer alloc).
    backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    let mut cascade_us_total: u64 = 0;
    let mut apply_us_total: u64 = 0;
    let t_all = Instant::now();
    for _ in 1..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        let p = backend.last_phase_timings();
        cascade_us_total += p.gpu_cascade_us;
        apply_us_total += p.cpu_apply_actions_us;
    }
    let wall = t_all.elapsed();
    let ticks = (TICKS - 1) as u64;

    // Flush the chronicle ring at the end. The count is informational
    // — it's the volume that used to contend with the main ring.
    let mut chronicle_events = EventRing::with_cap(1 << 20);
    let chronicle_outcome = backend.flush_chronicle(&mut chronicle_events);
    let chronicle_drained = chronicle_outcome
        .map(|o| o.drained)
        .unwrap_or_default();

    let mut main_chronicle_count = 0u32;
    let mut main_other_count = 0u32;
    for ev in events.iter() {
        match ev {
            engine_data::events::Event::ChronicleEntry { .. } => main_chronicle_count += 1,
            _ => main_other_count += 1,
        }
    }

    eprintln!("=== task 203 chronicle drain perf (N={N_AGENTS}, {ticks} ticks) ===");
    eprintln!("backend: {}", backend.backend_label());
    eprintln!("wall total:             {} ms", wall.as_millis());
    eprintln!("gpu_cascade avg:        {} us/tick", cascade_us_total / ticks);
    eprintln!("cpu_apply_actions avg:  {} us/tick", apply_us_total / ticks);
    eprintln!("chronicle ring drained: {chronicle_drained} records");
    eprintln!("per-tick chronicle avg: {} records", chronicle_drained / (ticks as u32).max(1));
    eprintln!("main ring: {main_chronicle_count} chronicle, {main_other_count} non-chronicle");
    // Main-ring chronicle records are produced by CPU `cold_state_replay`
    // — that's expected and unrelated to the GPU chronicle ring. The
    // GPU physics kernel emits ZERO chronicle records into the main
    // ring (asserted in event_ring_parity.rs).
}
