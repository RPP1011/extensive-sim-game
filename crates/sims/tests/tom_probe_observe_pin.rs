//! Mega-crate-hosted port of the Wave 3 ToM Phase 3 observe round-trip
//! pin (formerly `crates/tom_probe_runtime/tests/observe_round_trip_pin.rs`).
//!
//! ## What this exercises
//!
//! After agent A `observe`s agent B at tick T, A's BeliefState SoA cells
//! about B match B's actual agent-SoA state at tick T:
//!   - `beliefs_pos[A * N + B]` == B's seeded pos
//!   - `beliefs_type[A * N + B]` == B's seeded creature_type
//!   - `beliefs_tick[A * N + B]` == T (the observe tick)
//!   - `beliefs_confidence[A * N + B]` == 255 (q8 max)
//!   - `beliefs_flags[A * N + B]` == 0 (observe doesn't touch flags)
//!   - `beliefs_suspicion[A * N + B]` == 0 (Phase 4 verbs own suspicion)
//!   - per-cell isolation: every other cell in the 5 observe-touched
//!     columns stays at default zero
//!
//! Sources the host-side scaffolding from the in-test
//! `tom_probe_helpers` module (synchronous verb wrappers, seed/readback
//! against the auto-emitted `GeneratedRuntime` buffer fields).

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET: u32 = 1;
const OBSERVE_TICK: u64 = 7;

#[test]
fn observe_round_trip_matches_target_state() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_observe_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [10.0, 20.0, 30.0, 1.0];
    h::seed_agent_pos(&mut sim, &pos_seed);

    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 5;
    h::seed_agent_creature_type(&mut sim, &type_seed);

    for _ in 0..OBSERVE_TICK {
        sim.step();
    }
    assert_eq!(
        sim.tick, OBSERVE_TICK,
        "sim should be at tick {OBSERVE_TICK} before observe call",
    );

    h::dispatch_observe(&mut sim, OBSERVER, TARGET);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    let pos = h::read_beliefs_pos(&sim);
    assert_eq!(
        pos[cell],
        [10.0, 20.0, 30.0, 1.0],
        "beliefs_pos[OBSERVER * N + TARGET] should match target B's seeded pos",
    );

    let types = h::read_beliefs_type(&sim);
    assert_eq!(
        types[cell], 5,
        "beliefs_type[OBSERVER * N + TARGET] should match target B's seeded creature_type",
    );

    let ticks = h::read_beliefs_tick(&sim);
    assert_eq!(
        ticks[cell], OBSERVE_TICK as u32,
        "beliefs_tick[OBSERVER * N + TARGET] should equal the observe tick",
    );

    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(
        confidence[cell], 255,
        "beliefs_confidence[OBSERVER * N + TARGET] should peg at 255 on fresh observation",
    );

    let flags = h::read_beliefs_flags(&sim);
    assert_eq!(
        flags[cell], 0,
        "beliefs_flags[OBSERVER * N + TARGET] should remain 0 — observe doesn't set bits",
    );

    let suspicion = h::read_beliefs_suspicion(&sim);
    assert_eq!(
        suspicion[cell], 0,
        "beliefs_suspicion[OBSERVER * N + TARGET] should remain 0",
    );

    for o in 0..n {
        for t in 0..n {
            let idx = o * n + t;
            if idx == cell {
                continue;
            }
            assert_eq!(
                pos[idx], [0.0; 4],
                "beliefs_pos[{o} * N + {t}] = {:?} (expected [0,0,0,0])",
                pos[idx],
            );
            assert_eq!(types[idx], 0, "beliefs_type[{o} * N + {t}] = {}", types[idx]);
            assert_eq!(ticks[idx], 0, "beliefs_tick[{o} * N + {t}] = {}", ticks[idx]);
            assert_eq!(
                confidence[idx], 0,
                "beliefs_confidence[{o} * N + {t}] = {}",
                confidence[idx],
            );
            assert_eq!(
                suspicion[idx], 0,
                "beliefs_suspicion[{o} * N + {t}] = {}",
                suspicion[idx],
            );
        }
    }
}

#[test]
fn observe_overwrites_stale_belief() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_observe_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [1.0, 2.0, 3.0, 1.0];
    h::seed_agent_pos(&mut sim, &pos_seed);
    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 11;
    h::seed_agent_creature_type(&mut sim, &type_seed);

    for _ in 0..3 {
        sim.step();
    }
    h::dispatch_observe(&mut sim, OBSERVER, TARGET);

    let mut pos_seed2 = vec![[0.0f32; 4]; N as usize];
    pos_seed2[TARGET as usize] = [99.0, 88.0, 77.0, 1.0];
    h::seed_agent_pos(&mut sim, &pos_seed2);
    let mut type_seed2 = vec![0u8; N as usize];
    type_seed2[TARGET as usize] = 22;
    h::seed_agent_creature_type(&mut sim, &type_seed2);

    for _ in 0..6 {
        sim.step();
    }
    assert_eq!(sim.tick, 9);
    h::dispatch_observe(&mut sim, OBSERVER, TARGET);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    assert_eq!(
        h::read_beliefs_pos(&sim)[cell],
        [99.0, 88.0, 77.0, 1.0],
        "second observe should overwrite stale pos",
    );
    let types = h::read_beliefs_type(&sim);
    assert_eq!(types[cell], 22, "second observe should overwrite stale creature_type");
    let ticks = h::read_beliefs_tick(&sim);
    assert_eq!(ticks[cell], 9, "second observe should refresh last_seen_tick");
    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(confidence[cell], 255, "second observe should re-peg confidence at 255");
}
