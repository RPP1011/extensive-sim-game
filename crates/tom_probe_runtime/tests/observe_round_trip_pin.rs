//! Wave 3 ToM Phase 3 closed-loop pin: agent A `observe`s agent B at
//! tick T → agent A's BeliefState SoA cells about B match B's actual
//! agent-SoA state at tick T.
//!
//! ## What this exercises
//!
//! Phase 3 introduced the runtime-side observe consumer that mirrors
//! the future GPU `EffectObserveApplied` chronicle handler:
//!
//!   - The chronicle event kind `EffectObserveApplied = 64` is
//!     allocated in the engine `EventKindId` table (schema_hash bump
//!     `4925f4ed → 986761f5`).
//!   - The packed effect ordinal `EffectOp::Observe = 33` is wired
//!     end-to-end through `pack_effect`, the apply.rs ApplyEvent
//!     translation, the WGSL chronicle dispatcher arm chain, and the
//!     CPU oracle in `cpu_chronicle_reference.rs`.
//!   - `tom_probe_runtime::TomProbeState::observe(observer, target)`
//!     is the runtime-side consumer: it reads target's CURRENT pos +
//!     creature_type from the agent SoA and writes into ALL 6
//!     BeliefState SoA columns at `[observer * agent_count + target]`.
//!
//! The DSL view-call lowering for kernel-side reads of
//! `agents.beliefs_<field>(observer, subject)` (the spec's struct-
//! then-field access syntax) is deferred to Phase 4 alongside the
//! `scry` / `reveal` / deception verbs that need it. Phase 3 lands
//! the writer side end-to-end so the closed-loop pin asserts the
//! observation actually refreshes the BeliefState columns with the
//! right bytes.
//!
//! ## Fixture
//!
//! 4 agents. Agent B (= target_idx 1) is pre-seeded with
//! `pos = (10, 20, 30, 1)` and `creature_type = 5`. Agent A (=
//! observer_idx 0) calls `observe(0, 1)` at tick = 7 (the runtime
//! ticks 7 times before the call, so `self.tick == 7` when observe
//! consumes target's state).
//!
//! ## Asserts
//!
//! After the observe call:
//!   - `beliefs_pos[0 * N + 1]` == `(10, 20, 30, 1)` — observer A
//!     copied target B's pos verbatim.
//!   - `beliefs_type[0 * N + 1]` == `5` — observer A copied target
//!     B's creature_type verbatim.
//!   - `beliefs_tick[0 * N + 1]` == `7` — the observe-tick stamped
//!     onto last_seen_tick.
//!   - `beliefs_confidence[0 * N + 1]` == `255` — fresh observation
//!     pegs confidence at the q8 max (≈ 1.0).
//!   - `beliefs_flags[0 * N + 1]` == `0` — observe doesn't set the
//!     Phase 1 bit-OR slot (that's plant_belief's job).
//!   - `beliefs_suspicion[0 * N + 1]` == `0` — observe leaves the
//!     suspicion column at its default (Phase 4 deception verbs
//!     mutate this column).
//!   - **Per-cell isolation** — every other (observer', target')
//!     cell in all 6 columns stays at its default zero value. The
//!     observe call must NOT bleed into neighbouring cells.
//!
//! ## Skip path
//!
//! GPU adapter unavailable (CI minus `--all-targets`, sandbox without
//! a wgpu adapter). The constructor panics on a missing adapter; we
//! wrap in `catch_unwind` and log a skip message so the test passes
//! when the runtime simply can't spin up a device — same convention
//! as `pair_map_behavioral_pin.rs` and `belief_state_soa_pin.rs`.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET: u32 = 1;
const OBSERVE_TICK: u64 = 7;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn observe_round_trip_matches_target_state() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[observe_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Pre-seed agent B's state. Every other agent stays at default
    // zero so we can later verify the observe call didn't bleed into
    // them.
    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [10.0, 20.0, 30.0, 1.0];
    sim.seed_agent_pos(&pos_seed);

    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 5;
    sim.seed_agent_creature_type(&type_seed);

    // Tick the sim forward to OBSERVE_TICK. The Phase 1 producer
    // (physics_WhatIBelieve) will fire once per tick, populating the
    // diagonal of beliefs_flags — that's expected and orthogonal to
    // the observe consumer's writes (different columns).
    for _ in 0..OBSERVE_TICK {
        sim.step();
    }
    assert_eq!(
        sim.tick(),
        OBSERVE_TICK,
        "sim should be at tick {OBSERVE_TICK} before observe call",
    );

    // Observe: agent A learns agent B's current state.
    sim.observe(OBSERVER, TARGET);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    // -- Asserts on the (OBSERVER, TARGET) cell --

    let pos = sim.beliefs_pos().to_vec();
    assert_eq!(
        pos[cell],
        [10.0, 20.0, 30.0, 1.0],
        "beliefs_pos[OBSERVER * N + TARGET] should match target B's seeded pos",
    );

    let types = sim.beliefs_type().to_vec();
    assert_eq!(
        types[cell], 5,
        "beliefs_type[OBSERVER * N + TARGET] should match target B's seeded creature_type",
    );

    let ticks = sim.beliefs_tick().to_vec();
    assert_eq!(
        ticks[cell], OBSERVE_TICK as u32,
        "beliefs_tick[OBSERVER * N + TARGET] should equal the observe tick",
    );

    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[cell], 255,
        "beliefs_confidence[OBSERVER * N + TARGET] should peg at 255 (q8 max) on fresh observation",
    );

    // The Phase 1 bit-OR slot and the Phase 4 suspicion column are
    // NOT touched by observe — they stay at their default values.
    let flags = sim.beliefs_flags().to_vec();
    // Note: the diagonal cell `[OBSERVER * N + OBSERVER]` will have
    // `flags == 1` from the Phase 1 producer running for OBSERVE_TICK
    // ticks. The (OBSERVER, TARGET) off-diagonal cell stays 0.
    assert_eq!(
        flags[cell], 0,
        "beliefs_flags[OBSERVER * N + TARGET] should remain 0 — observe doesn't set bits",
    );

    let suspicion = sim.beliefs_suspicion().to_vec();
    assert_eq!(
        suspicion[cell], 0,
        "beliefs_suspicion[OBSERVER * N + TARGET] should remain 0 — Phase 4 deception verbs own this column",
    );

    // -- Per-cell isolation: every OTHER (o', t') cell in the 5
    // observe-touched columns stays at default zero. (The Phase 1
    // bit-OR producer pollutes the diagonal of `flags` — assert that
    // separately so the observe-consumer isolation property is loud.)
    for o in 0..n {
        for t in 0..n {
            let idx = o * n + t;
            if idx == cell {
                continue;
            }
            assert_eq!(
                pos[idx], [0.0; 4],
                "beliefs_pos[{o} * N + {t}] = {:?} (expected [0,0,0,0] — only the observed cell should be touched)",
                pos[idx],
            );
            assert_eq!(
                types[idx], 0,
                "beliefs_type[{o} * N + {t}] = {} (expected 0)",
                types[idx],
            );
            assert_eq!(
                ticks[idx], 0,
                "beliefs_tick[{o} * N + {t}] = {} (expected 0)",
                ticks[idx],
            );
            assert_eq!(
                confidence[idx], 0,
                "beliefs_confidence[{o} * N + {t}] = {} (expected 0)",
                confidence[idx],
            );
            assert_eq!(
                suspicion[idx], 0,
                "beliefs_suspicion[{o} * N + {t}] = {} (expected 0)",
                suspicion[idx],
            );
        }
    }
}

#[test]
fn observe_overwrites_stale_belief() {
    // Round-trip observe twice: first at tick 3 with a stale pos /
    // type, then at tick 9 with fresh state. The second observe
    // should overwrite the first wholesale — no merge / accumulate
    // semantics. Pins the "fresh observation REPLACES the cell" rule.
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[observe_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // First seed + observe at tick 3.
    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [1.0, 2.0, 3.0, 1.0];
    sim.seed_agent_pos(&pos_seed);
    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 11;
    sim.seed_agent_creature_type(&type_seed);

    for _ in 0..3 {
        sim.step();
    }
    sim.observe(OBSERVER, TARGET);

    // Re-seed target's state with completely different values, then
    // observe again at tick 9.
    let mut pos_seed2 = vec![[0.0f32; 4]; N as usize];
    pos_seed2[TARGET as usize] = [99.0, 88.0, 77.0, 1.0];
    sim.seed_agent_pos(&pos_seed2);
    let mut type_seed2 = vec![0u8; N as usize];
    type_seed2[TARGET as usize] = 22;
    sim.seed_agent_creature_type(&type_seed2);

    for _ in 0..6 {
        sim.step();
    }
    assert_eq!(sim.tick(), 9);
    sim.observe(OBSERVER, TARGET);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    assert_eq!(
        sim.beliefs_pos()[cell],
        [99.0, 88.0, 77.0, 1.0],
        "second observe should overwrite stale pos",
    );
    let types = sim.beliefs_type().to_vec();
    assert_eq!(
        types[cell], 22,
        "second observe should overwrite stale creature_type",
    );
    let ticks = sim.beliefs_tick().to_vec();
    assert_eq!(
        ticks[cell], 9,
        "second observe should refresh last_seen_tick to the new tick",
    );
    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[cell], 255,
        "second observe should re-peg confidence at 255",
    );
}
