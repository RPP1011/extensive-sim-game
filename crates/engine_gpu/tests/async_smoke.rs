//! End-to-end smoke: step_batch(100) at N=2048, one snapshot, assert
//! structural invariants. Does NOT compare to sync path — non-
//! deterministic by design.

#![cfg(feature = "gpu")]

use engine::backend::ComputeBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xAAAA_BBBB_CCCC_DDDD;
const N_AGENTS: u32 = 2048;
const BATCH_TICKS: u32 = 100;

fn spawn_fixture() -> SimState {
    let mut state = SimState::new(N_AGENTS + 16, SEED);
    let area = (N_AGENTS as f32 * 10.0).sqrt().ceil();
    let mut s: u64 = SEED;
    for i in 0..N_AGENTS {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = (s as f32 / u64::MAX as f32) * area - area * 0.5;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let y = (s as f32 / u64::MAX as f32) * area - area * 0.5;
        let (ct, hp) = match i % 5 {
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
fn step_batch_then_snapshot() {
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup: one sync step to pay shader-compile cost.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let tick_after_warmup = state.tick;

    // First snapshot — returns empty (no step_batch has run).
    let empty = gpu.snapshot(&mut state).expect("first snapshot");
    assert!(empty.agents.is_empty(), "first snapshot (pre-step_batch) should be empty");

    // Run the batch.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, BATCH_TICKS);

    // Second snapshot — front buffer is still empty at this point
    // because step_batch doesn't call snapshot internally.
    let _ = gpu.snapshot(&mut state);

    // Third snapshot — now returns the batch state.
    let snap = gpu.snapshot(&mut state).expect("third snapshot");
    // After step_batch, CPU state.tick stays stale; GPU sim_cfg.tick is
    // authoritative. Observer reads current tick via snapshot.
    assert_eq!(
        snap.tick,
        tick_after_warmup + BATCH_TICKS,
        "snapshot.tick should advance by BATCH_TICKS after step_batch",
    );
    assert_eq!(
        snap.agents.len(),
        (N_AGENTS + 16) as usize,
        "snapshot should contain all agent slots"
    );
    assert!(
        !snap.events_since_last.is_empty(),
        "events must accumulate during batch"
    );

    // Alive-count sanity. The batch path does NOT sync CPU
    // `SimState` back from the GPU mid-batch, so `state.agents_alive()`
    // reflects tick-0 (pre-batch) alive count — every spawned agent.
    // The snapshot, by contrast, reflects actual GPU-side combat
    // across 100 batch ticks and SHOULD show meaningful casualties.
    //
    // Bound: snap alive count must be strictly < spawn count (combat
    // happened) but > 0 (world isn't empty). Refined in Phase E when
    // the mask/scoring unpack kernels replaced the stale-state
    // host-side pack — previously the batch targeted tick-0 neighbours
    // and combat was underwhelming; now GPU targeting tracks current
    // state so more agents actually die.
    let alive_in_snap = snap.agents.iter().filter(|a| a.alive != 0).count();
    assert!(
        alive_in_snap > 0,
        "snapshot alive count should be > 0 — world cleared itself ({alive_in_snap} alive)"
    );
    assert!(
        (alive_in_snap as u32) < N_AGENTS,
        "snapshot alive count ({alive_in_snap}) == spawn ({N_AGENTS}) — combat appears to be a no-op"
    );
}
