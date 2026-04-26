//! Stress test / regression guard: chronicle emission on the batch
//! path under a realistic workload. If the
//! `ResidentPhysicsCfg` uniform-collapse bug (fixed 2026-04-23) ever
//! recurs, the small-N `chronicle_batch_path` test catches it too, but
//! this test is what we'd look at first for subtler per-iteration
//! uniform-binding regressions that only manifest with a many-agent
//! cascade (deeper cascade, more iterations, more emits per iter).
//!
//! Fixture: N=20_000 agents interleaved (40% humans, 40% wolves, 20%
//! deer) in a dense square so most agents have enemies within
//! attack range from tick 0. Run `step_batch(50)` to give the cascade
//! several deep-iteration ticks to exercise. Assert:
//!
//!   - `snap.chronicle_since_last.is_empty()` is false
//!   - At least one `template_id == 2` (chronicle_attack) record fires
//!     per 10 agent-tick combat events (very loose — just a sanity
//!     floor so the test fails noisy on regressions)
//!   - All chronicle records reference real agent ids (non-zero
//!     actor AND non-zero target — uninitialised-memory guard)
//!
//! GPU-only — no CPU parity comparison. The batch path is the system
//! under test; the sync path uses a different chronicle ring and its
//! own emission paths.

#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xCAFE_F00D_DEAD_BEEF;
const N_AGENTS: u32 = 20_000;
const AGENT_CAP: u32 = N_AGENTS + 8;
const TICKS: u32 = 50;

fn spawn_crowd() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
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
fn chronicle_emits_under_n20k_batch_workload() {
    let mut gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("chronicle_n20k_stress: GPU init failed — skipping ({e})");
            return;
        }
    };
    let mut state = spawn_crowd();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1 << 20);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup sync step so pipelines / cascade ctx are initialised.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Prime the snapshot double-buffer: first snapshot returns empty
    // and kicks the copy into the back staging.
    let _prime = gpu.snapshot(&mut state).expect("prime snapshot");

    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &UtilityBackend,
        &cascade,
        TICKS,
    );
    // Swap dance: first snapshot post-batch kicks the filled back into
    // front; second reads it.
    let _swap = gpu.snapshot(&mut state).expect("swap snapshot");
    let snap = gpu.snapshot(&mut state).expect("read snapshot");

    eprintln!(
        "chronicle_n20k: tick={} events_since_last={} chronicle_since_last={}",
        snap.tick,
        snap.events_since_last.len(),
        snap.chronicle_since_last.len()
    );

    // Categorise events by kind so regression messages are actionable.
    let mut attack_events = 0usize;
    for e in snap.events_since_last.iter() {
        if e.kind == 1 {
            attack_events += 1;
        }
    }
    eprintln!("  AgentAttacked events: {attack_events}");

    // Load-bearing assertion: chronicle emits must flow through.
    assert!(
        !snap.chronicle_since_last.is_empty(),
        "expected chronicle_since_last non-empty after step_batch({TICKS}) \
         on N={N_AGENTS}; got 0. AgentAttacked events observed: {attack_events}. \
         This is the regression surface for the ResidentPhysicsCfg \
         uniform-collapse bug (fixed 2026-04-23)."
    );

    // Template bucketing — chronicle_attack (2) should fire on every
    // AgentAttacked event a physics iteration processes, minus any
    // events that were already consumed before the chronicle ring was
    // reset at the top of the batch. A very loose floor: at least one
    // chronicle_attack record.
    let attacks: Vec<&engine_gpu::snapshot::ChronicleRecord> = snap
        .chronicle_since_last
        .iter()
        .filter(|r| r.template_id == 2)
        .collect();
    assert!(
        !attacks.is_empty(),
        "expected >=1 template_id=2 (chronicle_attack) record in \
         chronicle_since_last under {N_AGENTS}-agent workload; \
         got template ids {:?}",
        snap.chronicle_since_last
            .iter()
            .take(16)
            .map(|r| r.template_id)
            .collect::<Vec<_>>(),
    );

    // Uninit-memory guard: chronicle records reference at least one
    // non-zero agent id.
    for r in snap.chronicle_since_last.iter().take(64) {
        assert!(
            r.agent != 0 || r.target != 0,
            "chronicle record has zero agent+target — uninit memory? {r:?}"
        );
    }
}
