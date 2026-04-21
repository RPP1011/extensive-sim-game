//! Phase 2 parity harness (task #183 / plan: `docs/plans/gpu_megakernel_plan.md`).
//!
//! Gated on `--features gpu` so the CPU-only build doesn't pay the
//! compile-time cost of the fixture / backend wiring. Run with
//! `cargo test -p engine_gpu --features gpu`. The root workspace's `gpu`
//! feature re-exports this one (see root `Cargo.toml`), so
//! `cargo test --features gpu` from the workspace root picks it up too.
//!
//! ## What Phase 2 adds on top of Phase 1
//!
//! Phase 1 ran one kernel (Attack only) alongside the CPU step and
//! asserted its output matched a CPU reference bitmap each tick.
//! Phase 2 generalises: the backend now runs ONE fused kernel every
//! tick that emits N bitmaps — one per supported mask — and this test
//! asserts each one against its CPU reference.
//!
//! ## Masks in the fused kernel
//!
//! Seven masks are byte-parity checked:
//!
//!   * Attack      — target-bound, radius-filtered, hostility predicate
//!   * MoveToward  — target-bound, radius-filtered, alive + self-exclusion
//!   * Hold / Flee / Eat / Drink / Rest — self-only, alive gate
//!
//! Cast is **skipped**: its `mask Cast(ability: AbilityId)` head takes
//! a non-Agent parameter and its predicate reads views + cooldowns that
//! the Phase 2 emitter has no GPU backing for. The test surfaces the
//! skip via `eprintln!` so a regression that accidentally pulls Cast
//! back into the fused kernel surfaces early.
//!
//! If the fixture dies out before tick 50, later ticks have no alive
//! agents and every bitmap is zero on both sides — the byte-equal
//! check still passes trivially.

#![cfg(feature = "gpu")]

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::{
    mask::cpu_mask_bitmap,
    scoring::{cpu_score_outputs, ScoreOutput, NO_TARGET},
    GpuBackend,
};
use glam::Vec3;

const SEED: u64 = 0xD00D_FACE_0042_0042;
const TICKS: u32 = 50;
const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

/// Masks the fused kernel emits in Phase 2. Any name not in this list
/// is skipped in the parity comparison with a logged reason — currently
/// only "Cast" falls into that bucket, since its parametric head /
/// view dependencies land in Phase 4+.
const PARITY_MASK_NAMES: &[&str] = &[
    "Attack",
    "MoveToward",
    "Hold",
    "Flee",
    "Eat",
    "Drink",
    "Rest",
];

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
/// it does fire.
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

/// Run `TICKS` ticks through `GpuBackend`, checking every fused-mask
/// GPU bitmap against a CPU reference at every tick. Returns
/// `(state, events)` for the top-level CPU-vs-GPU state fingerprint
/// parity check.
fn run_gpu() -> (SimState, EventRing) {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    eprintln!("parity_with_cpu: wgpu backend = {}", backend.backend_label());

    // Log which masks the GPU actually runs — if the fused-kernel emitter
    // silently drops one, this is where it surfaces. Also warn about
    // Cast being skipped so the reason is in the test output, not just
    // the docstring.
    let gpu_masks: Vec<&str> = backend
        .mask_bindings()
        .iter()
        .map(|b| b.mask_name.as_str())
        .collect();
    eprintln!("parity_with_cpu: fused-kernel masks = {:?}", gpu_masks);
    eprintln!(
        "parity_with_cpu: skipped (Phase 4+ blocker) = [\"Cast\"] — parametric head + view/cooldown deps"
    );
    assert_eq!(
        gpu_masks.len(),
        PARITY_MASK_NAMES.len(),
        "unexpected fused-kernel mask count — regression or new mask added without updating PARITY_MASK_NAMES"
    );

    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();
    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let gpu_bitmaps = backend.last_mask_bitmaps();
        assert!(
            !gpu_bitmaps.is_empty(),
            "tick {tick_i}: GpuBackend::last_mask_bitmaps empty — kernel dispatch failed?"
        );
        assert_eq!(
            gpu_bitmaps.len(),
            PARITY_MASK_NAMES.len(),
            "tick {tick_i}: expected {} per-mask bitmaps, got {}",
            PARITY_MASK_NAMES.len(),
            gpu_bitmaps.len(),
        );

        // Per-mask byte-equality check.
        for name in PARITY_MASK_NAMES {
            let gpu_bitmap = backend
                .last_bitmap_for(name)
                .unwrap_or_else(|| panic!("tick {tick_i}: no GPU bitmap for mask `{name}`"));
            let cpu_bitmap = cpu_mask_bitmap(&state, name).unwrap_or_else(|| {
                panic!("tick {tick_i}: no CPU reference for mask `{name}`")
            });
            if gpu_bitmap != cpu_bitmap.as_slice() {
                let diff = bitmap_diff(&cpu_bitmap, gpu_bitmap);
                panic!(
                    "tick {tick_i}: GPU mask `{name}` bitmap differs from CPU\n\
                     cpu = {cpu_bitmap:#010x?}\n\
                     gpu = {gpu_bitmap:#010x?}\n\
                     disagreeing slots: {diff:?}",
                );
            }
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

/// Phase 2 headline test: `GpuBackend::step` dispatches the fused mask
/// kernel every tick and each of the seven supported mask bitmaps
/// matches the CPU-computed reference byte-for-byte across 50 ticks of
/// the canonical fixture. Also retains the Phase 0 CPU-forwarding
/// parity (event log + state fingerprint) — those assertions guard
/// against regressions in the side-by-side CPU path that
/// `GpuBackend::step` still runs.
#[test]
fn gpu_backend_matches_cpu_on_canonical_fixture() {
    let (cpu_state, cpu_events) = run_cpu();
    let (gpu_state, gpu_events) = run_gpu();

    // Event log parity — CPU forwards inside GpuBackend::step, so the
    // event log should still match byte-for-byte. Any regression here
    // means GpuBackend is mutating state in a way that leaks into CPU
    // side-effects (it shouldn't — Phase 2 GPU work is observation-
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

/// Direct GPU-vs-CPU fused-mask check before any step runs. Runs the
/// fused kernel against a freshly-spawned fixture (tick 0, no movement
/// yet) and compares every per-mask bitmap against the CPU reference.
/// Isolates "kernel correctness on a known state" from "kernel
/// correctness across the tick-by-tick state evolution" — if this test
/// passes but the multi-tick version fails, the bug is in how state
/// changes between ticks affect the kernel, not in the kernel itself.
#[test]
fn gpu_fused_masks_match_cpu_on_spawn_state() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let state = spawn_fixture();
    let gpu_bitmaps = backend
        .verify_masks_on_gpu(&state)
        .expect("GPU fused-mask dispatch");
    assert_eq!(gpu_bitmaps.len(), PARITY_MASK_NAMES.len());
    let bindings: Vec<String> = backend
        .mask_bindings()
        .iter()
        .map(|b| b.mask_name.clone())
        .collect();

    for (i, name) in bindings.iter().enumerate() {
        let gpu = &gpu_bitmaps[i];
        let cpu = cpu_mask_bitmap(&state, name).expect("CPU reference for mask");
        assert_eq!(
            gpu, &cpu,
            "GPU mask `{name}` differs from CPU on spawn state\ngpu={gpu:#010x?}\ncpu={cpu:#010x?}",
        );
    }

    // Hand-computed anchor for Attack (unchanged from Phase 1) — keeps
    // the CPU reference honest. Humans at (0,0,0), (2,0,0), (-2,0,0);
    // wolves at (3,0,0), (-3,0,0). Attack range is 2.0. Human↔Wolf is
    // mutually hostile. Expected attackers: ids 2, 3, 4, 5 → slots
    // 1, 2, 3, 4.
    let attack_cpu = cpu_mask_bitmap(&state, "Attack").expect("attack cpu");
    let expected_attack: u32 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
    assert_eq!(
        attack_cpu[0], expected_attack,
        "CPU Attack reference didn't match hand-computed spawn expectation"
    );

    // Self-only masks — every alive slot is set. 5 agents → bits 0..=4.
    let hold_cpu = cpu_mask_bitmap(&state, "Hold").expect("hold cpu");
    let expected_alive: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
    assert_eq!(
        hold_cpu[0], expected_alive,
        "CPU Hold reference should set every alive slot"
    );

    // MoveToward — every alive agent has at least one other alive
    // agent within `max_move_radius` (20.0 by default — all 5 fixture
    // agents sit on a 6m span), so every slot is set.
    let move_toward_cpu = cpu_mask_bitmap(&state, "MoveToward").expect("mt cpu");
    assert_eq!(
        move_toward_cpu[0], expected_alive,
        "CPU MoveToward reference should set every alive slot (all within max_move_radius)"
    );
}

// ---------------------------------------------------------------------------
// Phase 3 — scoring parity tests
// ---------------------------------------------------------------------------
//
// Two scoring tests live below. The first one is the byte-exact one
// (spawn state, view buffers all empty so the GPU's view stub matches
// CPU's "view returns 0"). The second is a multi-tick best-effort
// fixture (humans only — no hostile pairs, so no AgentAttacked events
// land, so view storage stays empty for the whole run).
//
// The full canonical 3v2 fixture is intentionally NOT used for scoring
// parity: as soon as the wolves attack, the CPU's view buffers
// (threat_level, my_enemies) populate but the GPU stub returns 0,
// flipping the argmax on Attack/Flee rows that depend on view
// modifiers. Phase 4 (task 185) wires real view storage; at that
// point this comment can come down and the canonical fixture joins
// the scoring parity sweep.

/// Fixture with 4 humans only, spaced inside max_move_radius.
/// Humans aren't hostile to each other; no `AgentAttacked` events ever
/// land; views (threat_level / my_enemies / kin_fear / pack_focus /
/// rally_boost) stay empty for the entire run. That makes the GPU's
/// view stub (returning 0) byte-equivalent to the CPU's actual view
/// reads (also 0 because the views are empty), so scoring parity
/// holds tick-by-tick.
fn spawn_no_combat_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(8.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 8.0),
        Vec3::new(8.0, 0.0, 8.0),
    ];
    for pos in positions {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos,
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
    }
    state
}

/// Phase 3 byte-exact test: the GPU scoring kernel produces the same
/// `(chosen_action, chosen_target)` per agent as the CPU reference on
/// the canonical 3v2 spawn state. Tick 0 — no combat has occurred,
/// every view buffer is empty, so the view stub on the GPU side
/// (returning 0) matches what the CPU's view reads return on an empty
/// view (also 0). This isolates "kernel correctness on a known state"
/// from "kernel correctness across the tick-by-tick state evolution".
#[test]
fn gpu_scoring_matches_cpu_on_spawn_state() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let state = spawn_fixture();
    let gpu_outs = backend
        .verify_scoring_on_gpu(&state)
        .expect("GPU scoring dispatch");
    let cpu_outs = cpu_score_outputs(&state);

    assert_eq!(
        gpu_outs.len(),
        cpu_outs.len(),
        "scoring output length differs"
    );

    // Diagnostic: print per-slot summaries before asserting so a fail
    // shows what diverged at a glance.
    eprintln!("gpu_scoring_matches_cpu_on_spawn_state: per-slot outputs");
    for slot in 0..gpu_outs.len() {
        let gpu = &gpu_outs[slot];
        let cpu = &cpu_outs[slot];
        eprintln!(
            "  slot {slot}: GPU=(action={}, target={}, score={:.4}) CPU=(action={}, target={}, score={:.4}){}",
            gpu.chosen_action,
            gpu.chosen_target,
            f32::from_bits(gpu.best_score_bits),
            cpu.chosen_action,
            cpu.chosen_target,
            f32::from_bits(cpu.best_score_bits),
            if gpu.chosen_action != cpu.chosen_action || gpu.chosen_target != cpu.chosen_target {
                "  <-- DIVERGENCE"
            } else {
                ""
            },
        );
    }

    // Per-slot equality. We compare on (chosen_action, chosen_target)
    // and the score's bit-pattern for full-byte parity. Pad bytes are
    // zero on both sides (GPU initialises to 0 before dispatch; CPU
    // ScoreOutput::default has _pad: 0).
    for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
        assert_scoring_eq(slot, gpu, cpu);
    }
}

/// Same shape as the byte-exact spawn-state test, but runs through
/// `TICKS` ticks of a no-combat fixture (4 humans, no wolves). Every
/// tick the GPU's `(chosen_action, chosen_target)` per agent must
/// match the CPU reference — view buffers stay empty for the whole
/// run because no hostile events fire, so the view stub doesn't
/// poison the argmax.
#[test]
fn gpu_scoring_matches_cpu_no_combat_fixture() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_no_combat_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

        let gpu_outs = backend.last_scoring_outputs();
        assert!(
            !gpu_outs.is_empty(),
            "tick {tick_i}: GpuBackend::last_scoring_outputs empty — kernel dispatch failed?"
        );
        assert_eq!(
            gpu_outs.len(),
            state.agent_cap() as usize,
            "tick {tick_i}: scoring output length mismatch"
        );

        let cpu_outs = cpu_score_outputs(&state);
        for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
            assert_scoring_eq_with_tick(tick_i, slot, gpu, cpu);
        }
    }
}

/// Best-effort scoring parity on the canonical 3v2 fixture. The
/// expectation is that for tick 0 (before any combat) the outputs
/// match byte-exact, but for later ticks the GPU's stubbed view-call
/// evaluator may diverge from the CPU's real view reads. We assert
/// only the first tick rigorously and log mismatches for later ticks
/// to make the divergence boundary visible without failing CI.
///
/// Phase 4 (task 185) wires real view storage; at that point this
/// test can drop the "best effort after tick 0" treatment and assert
/// byte-exact parity for the full run.
#[test]
fn gpu_scoring_canonical_fixture_best_effort() {
    let mut backend = GpuBackend::new().expect("GpuBackend init");
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut divergences = 0usize;
    let mut first_diverging_tick: Option<u32> = None;

    for tick_i in 0..TICKS {
        backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        let gpu_outs = backend.last_scoring_outputs();
        let cpu_outs = cpu_score_outputs(&state);
        assert_eq!(gpu_outs.len(), cpu_outs.len(), "tick {tick_i}: len");

        let mut tick_diff = 0;
        for (slot, (gpu, cpu)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
            if !scoring_eq(gpu, cpu) {
                tick_diff += 1;
                if tick_i == 0 {
                    // Tick 0 mismatch is a real bug — assert hard.
                    assert_scoring_eq_with_tick(tick_i, slot, gpu, cpu);
                }
            }
        }
        if tick_diff > 0 {
            divergences += tick_diff;
            if first_diverging_tick.is_none() {
                first_diverging_tick = Some(tick_i);
            }
        }
    }
    eprintln!(
        "gpu_scoring_canonical_fixture_best_effort: {divergences} per-(slot, tick) divergences \
         across {TICKS} ticks (first at tick {:?}). View storage lands in Phase 4 (task 185); \
         this number should drop to 0.",
        first_diverging_tick
    );
}

fn scoring_eq(a: &ScoreOutput, b: &ScoreOutput) -> bool {
    a.chosen_action == b.chosen_action
        && a.chosen_target == b.chosen_target
        && a.best_score_bits == b.best_score_bits
}

fn assert_scoring_eq(slot: usize, gpu: &ScoreOutput, cpu: &ScoreOutput) {
    assert_scoring_eq_with_tick(u32::MAX, slot, gpu, cpu);
}

fn assert_scoring_eq_with_tick(tick: u32, slot: usize, gpu: &ScoreOutput, cpu: &ScoreOutput) {
    let tick_str = if tick == u32::MAX {
        "spawn".to_string()
    } else {
        format!("tick {tick}")
    };
    assert_eq!(
        gpu.chosen_action, cpu.chosen_action,
        "{tick_str} slot {slot}: chosen_action GPU={} CPU={}",
        gpu.chosen_action, cpu.chosen_action
    );
    assert_eq!(
        gpu.chosen_target, cpu.chosen_target,
        "{tick_str} slot {slot}: chosen_target GPU={} CPU={} (NO_TARGET={NO_TARGET})",
        gpu.chosen_target, cpu.chosen_target,
    );
    if gpu.best_score_bits != cpu.best_score_bits {
        let gpu_score = f32::from_bits(gpu.best_score_bits);
        let cpu_score = f32::from_bits(cpu.best_score_bits);
        panic!(
            "{tick_str} slot {slot}: best_score_bits GPU=0x{:08x} ({gpu_score}) CPU=0x{:08x} ({cpu_score})",
            gpu.best_score_bits, cpu.best_score_bits,
        );
    }
}
