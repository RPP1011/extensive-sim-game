//! Wave 3 ToM Phase 4 closed-loop pin: caster A `decoy`s observer T's
//! belief row about subject S — after the call, T's row about S
//! contains the attacker-controlled values from `fake_pos` (no agent
//! SoA read).
//!
//! ## What this exercises
//!
//! Phase 4 introduces 3 deception verbs; this pin proves the decoy
//! verb's chronicle event → consumer pipeline works end-to-end. The
//! chronicle event payload carries:
//!
//!   * actor=slot 2 (caster A — informational only),
//!   * target=slot 3 (the OBSERVER T whose row gets clobbered),
//!   * subject_idx=slot 4 (the agent slot the belief is ABOUT — distinct
//!     from `target`),
//!   * fake_pos=slot 5 (packed (x_q8, y_q8, z_q8, fake_type) quartet —
//!     unpacked by the .sim consumer via `% 256` / `/ 256` arithmetic).
//!
//! The .sim consumer (`physics_ApplyDecoyBeliefUpdate`) writes 4 cells
//! of T's belief row about S: `last_known_pos` (decoded), `last_known_creature_type`
//! (= fake_type byte), `last_seen_tick` (= world.tick), `confidence` (= 255).
//! Suspicion + flags slots are NOT touched.
//!
//! ## Fixture
//!
//! 4 agents. Caster A (= 0) decoys observer T (= 2) about subject S (= 3)
//! with `fake_pos = (FAKE_TYPE << 24) | (FAKE_Z << 16) | (FAKE_Y << 8) | FAKE_X`
//! at tick = 5.
//!
//! ## Asserts
//!
//! After the decoy call:
//!   - `beliefs_pos[T * N + S]` == `(FAKE_X, FAKE_Y, FAKE_Z, 1)` —
//!     the consumer decoded the packed quartet's low 3 bytes.
//!   - `beliefs_type[T * N + S]` == `FAKE_TYPE` — the high byte.
//!   - `beliefs_tick[T * N + S]` == `5` — the world.tick at consume time.
//!   - `beliefs_confidence[T * N + S]` == `255` — q8 max.
//!   - `beliefs_suspicion[T * N + S]` == `0` — decoy doesn't touch suspicion.
//!   - `beliefs_flags[T * N + S]` == `0` — decoy doesn't touch flags.
//!   - **Per-cell isolation** — every other cell stays at its default
//!     zero (modulo the Phase 1 producer's diagonal flag-bit writes).

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_F00D_DEAD_BEEF;
const N: u32 = 4;
const CASTER: u32 = 0;
const TARGET: u32 = 2;
const SUBJECT: u32 = 3;
const DECOY_TICK: u64 = 5;
const FAKE_X: u32 = 9;
const FAKE_Y: u32 = 5;
const FAKE_Z: u32 = 3;
const FAKE_TYPE: u32 = 7;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn decoy_writes_attacker_controlled_belief_row() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[decoy_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Tick the sim forward to DECOY_TICK. The Phase 1 producer
    // (physics_WhatIBelieve) writes the diagonal of beliefs_flags;
    // off-diagonal cells (including TARGET * N + SUBJECT) stay 0.
    for _ in 0..DECOY_TICK {
        sim.step();
    }
    assert_eq!(sim.tick(), DECOY_TICK);

    // Pack: high byte = fake_type, then z, y, x (low byte).
    let fake_pos = (FAKE_TYPE << 24) | (FAKE_Z << 16) | (FAKE_Y << 8) | FAKE_X;
    sim.decoy(CASTER, TARGET, SUBJECT, fake_pos);

    let n = N as usize;
    let cell = (TARGET as usize) * n + (SUBJECT as usize);

    let pos = sim.beliefs_pos().to_vec();
    assert_eq!(
        pos[cell],
        [FAKE_X as f32, FAKE_Y as f32, FAKE_Z as f32, 1.0],
        "beliefs_pos[T * N + S] should equal the unpacked fake_pos quartet",
    );

    let types = sim.beliefs_type().to_vec();
    assert_eq!(
        types[cell], FAKE_TYPE as u8,
        "beliefs_type[T * N + S] should equal the high byte (fake_type)",
    );

    let ticks = sim.beliefs_tick().to_vec();
    assert_eq!(
        ticks[cell], DECOY_TICK as u32,
        "beliefs_tick[T * N + S] should equal world.tick at consume time",
    );

    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[cell], 255,
        "beliefs_confidence[T * N + S] should peg at 255 (q8 max — fresh decoy)",
    );

    let suspicion = sim.beliefs_suspicion().to_vec();
    assert_eq!(
        suspicion[cell], 0,
        "beliefs_suspicion[T * N + S] should remain 0 — decoy doesn't touch suspicion",
    );

    let flags = sim.beliefs_flags().to_vec();
    assert_eq!(
        flags[cell], 0,
        "beliefs_flags[T * N + S] should remain 0 — decoy doesn't touch flags",
    );

    // Per-cell isolation: every OTHER (o, t) cell in the 4 decoy-touched
    // columns stays at its default zero. (The Phase 1 producer's diagonal
    // writes pollute beliefs_flags only — orthogonal to the columns
    // decoy touches.)
    for o in 0..n {
        for t in 0..n {
            let idx = o * n + t;
            if idx == cell {
                continue;
            }
            assert_eq!(
                pos[idx], [0.0; 4],
                "beliefs_pos[{o} * N + {t}] = {:?} (expected [0,0,0,0]; only the decoy cell should be touched)",
                pos[idx],
            );
            assert_eq!(types[idx], 0, "beliefs_type[{o} * N + {t}]");
            assert_eq!(ticks[idx], 0, "beliefs_tick[{o} * N + {t}]");
            assert_eq!(confidence[idx], 0, "beliefs_confidence[{o} * N + {t}]");
        }
    }
}
