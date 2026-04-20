//! Phase 0 parity harness (task #181 / plan: `docs/plans/gpu_megakernel_plan.md`).
//!
//! Gated on `--features gpu` so the CPU-only build doesn't pay the
//! compile-time cost of the fixture / backend wiring. Run with
//! `cargo test -p engine_gpu --features gpu`. The root workspace's `gpu`
//! feature re-exports this one (see root `Cargo.toml`), so
//! `cargo test --features gpu` from the workspace root picks it up too.
//!
//! Runs the same minimal fixture through `CpuBackend` and `GpuBackend` for N
//! ticks and asserts every observable artefact matches byte-for-byte:
//!   1. the event log — `Vec<Event>` collected from each `EventRing`
//!   2. the post-tick state — a deterministic text fingerprint built from
//!      `(tick, agent_cap, per-agent {id, alive, pos, hp})`
//!
//! `SimState` doesn't derive `Clone`/`PartialEq` (it owns several SoA `Vec`s
//! that cross-reference the `AgentSlotPool`), so we can't compare the
//! structs directly. The fingerprint covers the fields the sim actually
//! mutates each tick — if any of them drift between backends the test
//! fails. `Event` *is* `Copy + PartialEq` (see `engine_rules::events`), so
//! the event-log comparison is a straight `Vec<Event>` equality.
//!
//! Phase 0 invariant: `GpuBackend` forwards to `engine::step::step`, so
//! every assertion below passes trivially. Once Phase 1+ replaces the
//! forward with real GPU dispatch, this harness is the tripwire — any
//! divergence (a misplaced atomic, non-deterministic event order, a rng
//! stream that doesn't match CPU byte-for-byte) surfaces here.

#![cfg(feature = "gpu")]

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xD00D_FACE_0042_0042;
const TICKS: u32 = 50;
const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

/// Spawn the canonical 3-humans-and-2-wolves fixture that the engine's
/// `wolves_and_humans_parity.rs` anchor uses. Kept in lockstep (same seed,
/// same positions, same HP) so a failure here also shows up in the engine's
/// own parity baseline, making root-causing easier.
fn spawn_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos:           Vec3::new(0.0, 0.0, 0.0),
        hp:            100.0,
        ..Default::default()
    }).expect("human 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos:           Vec3::new(2.0, 0.0, 0.0),
        hp:            100.0,
        ..Default::default()
    }).expect("human 2 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos:           Vec3::new(-2.0, 0.0, 0.0),
        hp:            100.0,
        ..Default::default()
    }).expect("human 3 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos:           Vec3::new(3.0, 0.0, 0.0),
        hp:            80.0,
        ..Default::default()
    }).expect("wolf 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos:           Vec3::new(-3.0, 0.0, 0.0),
        hp:            80.0,
        ..Default::default()
    }).expect("wolf 2 spawn");
    state
}

/// Deterministic per-state fingerprint. Covers the observable mutations
/// each tick makes: `tick`, alive bitmap, and per-agent `(pos, hp)` for
/// every slot in spawn order. Format is stable text — `{:.6}` for all
/// f32 values so a glam::Vec3 `Debug` format change doesn't silently
/// invalidate the fingerprint. Keeps the assertion message readable when
/// it does fire (Phase 1+).
fn fingerprint(state: &SimState) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(256);
    writeln!(s, "tick={} agent_cap={}", state.tick, state.agent_cap()).unwrap();
    // `agents_alive()` is slot-order deterministic (SoA indexed iteration,
    // not a `HashMap`), so we can rely on its yield order for the
    // fingerprint without an explicit sort.
    for id in state.agents_alive() {
        let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
        let hp = state.agent_hp(id).unwrap_or(f32::NAN);
        writeln!(
            s,
            "id={} pos=({:.6},{:.6},{:.6}) hp={:.6}",
            id.raw(),
            pos.x, pos.y, pos.z,
            hp,
        ).unwrap();
    }
    s
}

/// Collect every event the ring holds, in push order, into a `Vec<Event>`
/// so we can `assert_eq!` across backends. The `EventRing` itself isn't
/// `PartialEq`, but its element type is.
fn collect_events(ring: &EventRing) -> Vec<Event> {
    ring.iter().copied().collect()
}

/// Run `TICKS` ticks of the canonical fixture through `backend` and return
/// `(state, events)` for parity comparison. Mirrors the setup `xtask
/// chronicle --gpu` uses — `UtilityBackend` as the policy, the
/// engine-builtin cascade registry, no views / invariants / telemetry
/// (that matches `engine::step::step`'s internal call to `step_full`).
fn run<B: SimBackend>(mut backend: B) -> (SimState, EventRing) {
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for _ in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    (state, events)
}

/// Phase 0 parity check: GpuBackend ≡ CpuBackend byte-for-byte on events
/// and state fingerprint. Trivially true while GpuBackend forwards to CPU;
/// becomes load-bearing in Phase 1+.
#[test]
fn gpu_backend_matches_cpu_on_canonical_fixture() {
    let (cpu_state, cpu_events) = run(CpuBackend);
    let (gpu_state, gpu_events) = run(GpuBackend::new());

    // Event log parity — the most sensitive surface. Any divergence in
    // mask build, action scoring, action shuffle, or cascade fixed-point
    // produces a different event sequence.
    let cpu_evs = collect_events(&cpu_events);
    let gpu_evs = collect_events(&gpu_events);
    assert_eq!(
        cpu_evs.len(),
        gpu_evs.len(),
        "event count diverged: cpu={} gpu={}",
        cpu_evs.len(),
        gpu_evs.len(),
    );
    assert_eq!(
        cpu_evs, gpu_evs,
        "event log differs between CpuBackend and GpuBackend",
    );
    assert_eq!(
        cpu_events.total_pushed(),
        gpu_events.total_pushed(),
        "EventRing::total_pushed diverged",
    );

    // State fingerprint parity — covers tick, alive bitmap, positions,
    // HP. Captures anything the event log missed (e.g. an agent that
    // moved without emitting `AgentMoved`, or a rng desync that only
    // surfaces in a tick that happens to be a no-op).
    let cpu_fp = fingerprint(&cpu_state);
    let gpu_fp = fingerprint(&gpu_state);
    assert_eq!(
        cpu_fp, gpu_fp,
        "state fingerprint differs after {} ticks\ncpu:\n{}\ngpu:\n{}",
        TICKS, cpu_fp, gpu_fp,
    );
}

/// Sanity check: running the same fixture through the same backend twice
/// produces identical output. Without this, a failing `gpu_backend_matches_
/// cpu_on_canonical_fixture` would be ambiguous between "GPU diverged" and
/// "the fixture / setup is non-deterministic". This isolates the latter.
#[test]
fn cpu_backend_is_deterministic_on_canonical_fixture() {
    let (state_a, events_a) = run(CpuBackend);
    let (state_b, events_b) = run(CpuBackend);
    assert_eq!(collect_events(&events_a), collect_events(&events_b));
    assert_eq!(fingerprint(&state_a), fingerprint(&state_b));
}
