//! Mega-crate-hosted port of the Wave 3 ToM Phase 3.5 reveal round-trip
//! pin (formerly `crates/tom_probe_runtime/tests/reveal_round_trip_pin.rs`).
//!
//! One-to-many propagation: caster broadcasts its beliefs about a
//! subject to all observers — the .sim consumer's per-observer copy.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const CASTER: u32 = 0;
const SUBJECT: u32 = 2;

#[test]
fn reveal_broadcasts_caster_belief_row_to_every_observer() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_reveal_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let src_cell = (CASTER as usize) * n + (SUBJECT as usize);

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [1.0, 2.0, 3.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[src_cell] = 9;
    h::seed_beliefs_type(&mut sim, &type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[src_cell] = 50;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[src_cell] = 180;
    h::seed_beliefs_confidence(&mut sim, &conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[src_cell] = 30;
    h::seed_beliefs_suspicion(&mut sim, &susp_seed);

    let mut flags_seed = vec![0u32; cell_count];
    flags_seed[src_cell] = 0xBEEF;
    h::seed_beliefs_flags(&mut sim, &flags_seed);

    h::dispatch_reveal(&mut sim, CASTER, SUBJECT);

    let pos = h::read_beliefs_pos(&sim);
    let types = h::read_beliefs_type(&sim);
    let ticks = h::read_beliefs_tick(&sim);
    let confidence = h::read_beliefs_confidence(&sim);
    let suspicion = h::read_beliefs_suspicion(&sim);
    let flags = h::read_beliefs_flags(&sim);

    for observer in 0..n {
        let cell = observer * n + (SUBJECT as usize);
        assert_eq!(pos[cell], [1.0, 2.0, 3.0, 1.0], "pos[{observer}]");
        assert_eq!(types[cell], 9, "types[{observer}]");
        assert_eq!(ticks[cell], 50, "ticks[{observer}]");
        assert_eq!(confidence[cell], 180, "conf[{observer}]");
        assert_eq!(suspicion[cell], 30, "susp[{observer}]");
        assert_eq!(flags[cell], 0xBEEF, "flags[{observer}]");
    }

    // Cells about subjects OTHER than SUBJECT stay at default zero.
    for o in 0..n {
        for s in 0..n {
            if s == SUBJECT as usize {
                continue;
            }
            let cell = o * n + s;
            assert_eq!(pos[cell], [0.0; 4], "pos[{o},{s}] should stay 0");
            assert_eq!(types[cell], 0);
            assert_eq!(ticks[cell], 0);
            assert_eq!(confidence[cell], 0);
            assert_eq!(suspicion[cell], 0);
            assert_eq!(flags[cell], 0);
        }
    }
}

#[test]
fn reveal_with_non_zero_caster_addresses_correct_source_cell() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_reveal_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    const NON_ZERO_CASTER: u32 = 3;
    let src_cell = (NON_ZERO_CASTER as usize) * n + (SUBJECT as usize);

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [99.0, 88.0, 77.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    h::dispatch_reveal(&mut sim, NON_ZERO_CASTER, SUBJECT);

    let pos = h::read_beliefs_pos(&sim);
    for observer in 0..n {
        let cell = observer * n + (SUBJECT as usize);
        assert_eq!(
            pos[cell], [99.0, 88.0, 77.0, 1.0],
            "broadcast must source from caster=3, not caster=0",
        );
    }
}
