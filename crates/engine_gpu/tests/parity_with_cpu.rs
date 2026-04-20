//! Phase 1 parity harness (task #182 / plan: `docs/plans/gpu_megakernel_plan.md`).
//!
//! Gated on `--features gpu` so the CPU-only build doesn't pay the
//! compile-time cost of the fixture / backend wiring. Run with
//! `cargo test -p engine_gpu --features gpu`. The root workspace's `gpu`
//! feature re-exports this one (see root `Cargo.toml`), so
//! `cargo test --features gpu` from the workspace root picks it up too.
//!
//! ## What Phase 1 adds on top of Phase 0
//!
//! Phase 0 ran the fixture through `CpuBackend` and `GpuBackend`
//! separately and asserted their observable outputs (event log + state
//! fingerprint) matched. That guard stays — it catches any regression
//! that breaks the CPU-forwarding path.
//!
//! Phase 1 adds a per-tick **Attack mask bitmap** comparison. After each
//! tick of the GPU backend run, we:
//!   1. Read `GpuBackend::last_attack_mask_bitmap()` — the output of
//!      the GPU WGSL kernel that just ran.
//!   2. Compute the same bitmap on CPU using
//!      `engine_gpu::attack_mask::cpu_attack_mask_bitmap(&state)`.
//!   3. Assert they're byte-equal. Any divergence dumps the disagreeing
//!      slot indices so root-causing is fast.
//!
//! That's the Phase 1 proof point: WGSL emit → naga parse → dispatch →
//! readback all work, and the Attack mask the GPU computes matches
//! CPU.

#![cfg(feature = "gpu")]

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::{attack_mask::cpu_attack_mask_bitmap, GpuBackend};
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
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 2 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 3 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(-3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 2 spawn");
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
    for id in state.agents_alive() {
        let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
        let hp = state.agent_hp(id).unwrap_or(f32::NAN);
        writeln!(
            s,
            "id={} pos=({:.6},{:.6},{:.6}) hp={:.6}",
            id.raw(),
            pos.x,
            pos.y,
            pos.z,
            hp,
        )
        .unwrap();
    }
    s
}

/// Collect every event the ring holds, in push order, into a `Vec<Event>`
/// so we can `assert_eq!` across backends. The `EventRing` itself isn't
/// `PartialEq`, but its element type is.
fn collect_events(ring: &EventRing) -> Vec<Event> {
    ring.iter().copied().collect()
}

/// Run `TICKS` ticks through `CpuBackend` and return `(state, events)`.
fn run_cpu() -> (SimState, EventRing) {
    let mut backend = CpuBackend;
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for _ in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    (state, events)
}

/// Run `TICKS` ticks through `GpuBackend`, checking the GPU-computed
/// Attack mask bitmap against a CPU reference at every tick. Returns
/// `(state, events)` for the top-level CPU-vs-GPU state fingerprint
/// parity check.
fn run_gpu() -> (SimState, EventRing) {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    eprintln!("parity_with_cpu: wgpu backend = {}", backend.backend_label());
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let gpu_bitmap = backend.last_attack_mask_bitmap().to_vec();
        assert!(
            !gpu_bitmap.is_empty(),
            "tick {tick_i}: GpuBackend::last_attack_mask_bitmap empty — kernel dispatch failed?"
        );
        let cpu_bitmap = cpu_attack_mask_bitmap(&state);
        if gpu_bitmap != cpu_bitmap {
            let diff = bitmap_diff(&cpu_bitmap, &gpu_bitmap);
            panic!(
                "tick {tick_i}: GPU Attack mask bitmap differs from CPU\n\
                 cpu = {cpu_bitmap:#010x?}\n\
                 gpu = {gpu_bitmap:#010x?}\n\
                 disagreeing slots: {diff:?}",
            );
        }
    }
    (state, events)
}

/// List every slot index where the two bitmaps disagree — used for
/// diagnostic output when the per-tick comparison fails.
fn bitmap_diff(cpu: &[u32], gpu: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let n_words = cpu.len().max(gpu.len());
    for w in 0..n_words {
        let c = cpu.get(w).copied().unwrap_or(0);
        let g = gpu.get(w).copied().unwrap_or(0);
        let mut xor = c ^ g;
        while xor != 0 {
            let bit = xor.trailing_zeros();
            out.push((w as u32) * 32 + bit);
            xor &= xor - 1;
        }
    }
    out
}

/// Phase 1 headline test: `GpuBackend::step` computes an Attack mask on
/// the real GPU every tick and it matches the CPU-computed bitmap byte
/// for byte across 50 ticks of the canonical fixture. Also retains the
/// Phase 0 CPU-forwarding parity (event log + state fingerprint) —
/// those assertions guard against regressions in the side-by-side CPU
/// path that `GpuBackend::step` still runs.
#[test]
fn gpu_backend_matches_cpu_on_canonical_fixture() {
    let (cpu_state, cpu_events) = run_cpu();
    let (gpu_state, gpu_events) = run_gpu();

    // Event log parity — CPU forwards inside GpuBackend::step, so the
    // event log should still match byte-for-byte. Any regression here
    // means GpuBackend is mutating state in a way that leaks into CPU
    // side-effects (it shouldn't — Phase 1 GPU work is observation-
    // only).
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

    // State fingerprint parity.
    let cpu_fp = fingerprint(&cpu_state);
    let gpu_fp = fingerprint(&gpu_state);
    assert_eq!(
        cpu_fp, gpu_fp,
        "state fingerprint differs after {} ticks\ncpu:\n{}\ngpu:\n{}",
        TICKS, cpu_fp, gpu_fp,
    );
}

/// Phase 0 holdover: running the same fixture through `CpuBackend`
/// twice produces identical output. Without this, a failing
/// `gpu_backend_matches_cpu_on_canonical_fixture` would be ambiguous
/// between "GPU diverged" and "the fixture / setup is non-deterministic".
/// This isolates the latter.
#[test]
fn cpu_backend_is_deterministic_on_canonical_fixture() {
    let (state_a, events_a) = run_cpu();
    let (state_b, events_b) = run_cpu();
    assert_eq!(collect_events(&events_a), collect_events(&events_b));
    assert_eq!(fingerprint(&state_a), fingerprint(&state_b));
}

/// Direct GPU-vs-CPU Attack-mask bitmap check before any step runs.
/// Runs the GPU kernel against a freshly-spawned fixture (tick 0, no
/// movement yet) and compares against the CPU reference. Isolates
/// "kernel correctness on a known state" from "kernel correctness
/// across the tick-by-tick state evolution" — if this test passes but
/// the multi-tick version fails, the bug is in how state changes
/// between ticks affect the kernel, not in the kernel itself.
#[test]
fn gpu_attack_mask_matches_cpu_on_spawn_state() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let state = spawn_fixture();
    let gpu = backend
        .verify_attack_mask_on_gpu(&state)
        .expect("GPU Attack mask dispatch");
    let cpu = cpu_attack_mask_bitmap(&state);
    assert_eq!(
        gpu, cpu,
        "GPU Attack mask differs from CPU on spawn state\ngpu={gpu:#010x?}\ncpu={cpu:#010x?}",
    );
    // The spawn fixture: humans at (0,0,0), (2,0,0), (-2,0,0);
    // wolves at (3,0,0), (-3,0,0). Attack range is 2.0. Human-Wolf
    // hostility is mutual. Expected attackers:
    //   - human1 @ (0,0,0): closest wolf at (3,0,0), dist 3 > 2 → NO
    //                       closest wolf at (-3,0,0), dist 3 > 2 → NO
    //     actually human2 (at 2) is within range but same-creature
    //     → not hostile. So human1 has no attackable target.
    //   - human2 @ (2,0,0): wolf1 @ (3,0,0), dist 1 < 2 → YES
    //   - human3 @ (-2,0,0): wolf2 @ (-3,0,0), dist 1 < 2 → YES
    //   - wolf1 @ (3,0,0): human2 @ (2,0,0), dist 1 < 2 → YES
    //   - wolf2 @ (-3,0,0): human3 @ (-2,0,0), dist 1 < 2 → YES
    // Slots are 0..=4 for AgentId(1..=5). Expected bits: 2,3,4 (human2,
    // human3, wolf1) and 5-1=4 (wolf2) — ids 2,3,4,5 → slots 1,2,3,4.
    let expected_bits: u32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
    assert_eq!(
        cpu[0], expected_bits,
        "CPU reference bitmap didn't match hand-computed spawn expectation"
    );
}
