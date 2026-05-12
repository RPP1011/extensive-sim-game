//! Mega-crate-hosted port of the Wave 3 ToM Phase 3.5 scry round-trip
//! pin (formerly `crates/tom_probe_runtime/tests/scry_round_trip_pin.rs`).
//!
//! Cross-observer access: caster reads another agent's belief row and
//! folds it into the caster's own belief row about the same subject —
//! the .sim consumer's 6-column verbatim copy.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET_OBSERVER: u32 = 1;
const SUBJECT: u32 = 2;

#[test]
fn scry_copies_target_observer_belief_row_verbatim() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_scry_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let src_cell = (TARGET_OBSERVER as usize) * n + (SUBJECT as usize);
    let dst_cell = (OBSERVER as usize) * n + (SUBJECT as usize);

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [10.0, 20.0, 30.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[src_cell] = 5;
    h::seed_beliefs_type(&mut sim, &type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[src_cell] = 100;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[src_cell] = 200;
    h::seed_beliefs_confidence(&mut sim, &conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[src_cell] = 50;
    h::seed_beliefs_suspicion(&mut sim, &susp_seed);

    let mut flags_seed = vec![0u32; cell_count];
    flags_seed[src_cell] = 0xCAFE;
    h::seed_beliefs_flags(&mut sim, &flags_seed);

    h::dispatch_scry(&mut sim, OBSERVER, TARGET_OBSERVER, SUBJECT);

    let pos = h::read_beliefs_pos(&sim);
    assert_eq!(
        pos[dst_cell], [10.0, 20.0, 30.0, 1.0],
        "scry should copy target_observer's beliefs_pos verbatim",
    );

    let types = h::read_beliefs_type(&sim);
    assert_eq!(
        types[dst_cell], 5,
        "scry should copy target_observer's beliefs_type verbatim",
    );

    let ticks = h::read_beliefs_tick(&sim);
    assert_eq!(
        ticks[dst_cell], 100,
        "scry should copy target_observer's beliefs_tick verbatim",
    );

    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(
        confidence[dst_cell], 200,
        "scry should copy target_observer's beliefs_confidence verbatim",
    );

    let suspicion = h::read_beliefs_suspicion(&sim);
    assert_eq!(
        suspicion[dst_cell], 50,
        "scry should copy target_observer's beliefs_suspicion verbatim",
    );

    let flags = h::read_beliefs_flags(&sim);
    assert_eq!(
        flags[dst_cell], 0xCAFE,
        "scry should copy target_observer's beliefs_flags verbatim",
    );

    // Source cell unchanged.
    assert_eq!(pos[src_cell], [10.0, 20.0, 30.0, 1.0]);
    assert_eq!(types[src_cell], 5);
    assert_eq!(ticks[src_cell], 100);
    assert_eq!(confidence[src_cell], 200);
    assert_eq!(suspicion[src_cell], 50);
    assert_eq!(flags[src_cell], 0xCAFE);

    // Per-cell isolation.
    for o in 0..n {
        for s in 0..n {
            let idx = o * n + s;
            if idx == src_cell || idx == dst_cell {
                continue;
            }
            assert_eq!(pos[idx], [0.0; 4], "pos[{o},{s}]");
            assert_eq!(types[idx], 0, "types[{o},{s}]");
            assert_eq!(ticks[idx], 0, "ticks[{o},{s}]");
            assert_eq!(confidence[idx], 0, "conf[{o},{s}]");
            assert_eq!(suspicion[idx], 0, "susp[{o},{s}]");
            assert_eq!(flags[idx], 0, "flags[{o},{s}]");
        }
    }
}

#[test]
fn self_scry_is_idempotent() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_scry_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let cell = (OBSERVER as usize) * n + (SUBJECT as usize);

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [7.0, 8.0, 9.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 128;
    h::seed_beliefs_confidence(&mut sim, &conf_seed);

    h::dispatch_scry(&mut sim, OBSERVER, OBSERVER, SUBJECT);

    let pos = h::read_beliefs_pos(&sim);
    assert_eq!(
        pos[cell], [7.0, 8.0, 9.0, 1.0],
        "self-scry must preserve the original belief tuple",
    );
    assert_eq!(h::read_beliefs_confidence(&sim)[cell], 128);
}
