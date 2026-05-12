//! Mega-crate-hosted port of the Wave 3 ToM Phase 4 decoy round-trip
//! pin (formerly `crates/tom_probe_runtime/tests/decoy_round_trip_pin.rs`).
//!
//! Caster overwrites observer's beliefs about subject with attacker-
//! controlled `(x_q8, y_q8, z_q8, fake_type)` quartet packed into a u32.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

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

#[test]
fn decoy_writes_attacker_controlled_belief_row() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_decoy_pin] skipping: no wgpu adapter");
            return;
        }
    };

    for _ in 0..DECOY_TICK {
        sim.step();
    }
    assert_eq!(sim.tick, DECOY_TICK);

    let fake_pos = (FAKE_TYPE << 24) | (FAKE_Z << 16) | (FAKE_Y << 8) | FAKE_X;
    h::dispatch_decoy(&mut sim, CASTER, TARGET, SUBJECT, fake_pos);

    let n = N as usize;
    let cell = (TARGET as usize) * n + (SUBJECT as usize);

    let pos = h::read_beliefs_pos(&sim);
    assert_eq!(
        pos[cell],
        [FAKE_X as f32, FAKE_Y as f32, FAKE_Z as f32, 1.0],
        "beliefs_pos[T*N+S] should equal the unpacked fake_pos quartet",
    );

    let types = h::read_beliefs_type(&sim);
    assert_eq!(
        types[cell], FAKE_TYPE as u8,
        "beliefs_type[T*N+S] should equal the high byte (fake_type)",
    );

    let ticks = h::read_beliefs_tick(&sim);
    assert_eq!(
        ticks[cell], DECOY_TICK as u32,
        "beliefs_tick[T*N+S] should equal world.tick at consume time",
    );

    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(
        confidence[cell], 255,
        "beliefs_confidence[T*N+S] should peg at 255 (q8 max — fresh decoy)",
    );

    let suspicion = h::read_beliefs_suspicion(&sim);
    assert_eq!(
        suspicion[cell], 0,
        "beliefs_suspicion[T*N+S] should remain 0 — decoy doesn't touch suspicion",
    );

    let flags = h::read_beliefs_flags(&sim);
    assert_eq!(
        flags[cell], 0,
        "beliefs_flags[T*N+S] should remain 0 — decoy doesn't touch flags",
    );

    for o in 0..n {
        for t in 0..n {
            let idx = o * n + t;
            if idx == cell {
                continue;
            }
            assert_eq!(
                pos[idx], [0.0; 4],
                "beliefs_pos[{o},{t}] = {:?} (expected [0,0,0,0])",
                pos[idx],
            );
            assert_eq!(types[idx], 0);
            assert_eq!(ticks[idx], 0);
            assert_eq!(confidence[idx], 0);
        }
    }
}
