//! Mega-crate-hosted port of the Wave 3 ToM Phase 4 erase_belief
//! round-trip pin (formerly
//! `crates/tom_probe_runtime/tests/erase_belief_round_trip_pin.rs`).
//!
//! Caster sends an EraseBelief chronicle event; the consumer (fused
//! with disguise into `physics_ApplyEraseBeliefUpdate_and_ApplyDisguise`)
//! clears the cells matching the `fields` bitset, leaves others intact.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xC0DE_F00D_FACE_BABE;
const N: u32 = 4;
const CASTER: u32 = 0;
const TARGET: u32 = 1;
const SUBJECT: u32 = 2;
const ERASE_TICK: u64 = 5;

fn cell_index(observer: u32, subject: u32) -> usize {
    (observer as usize) * (N as usize) + (subject as usize)
}

#[test]
fn erase_belief_with_all_fields_clears_all_six_columns() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_erase_belief_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let cell = cell_index(TARGET, SUBJECT);
    let neighbour = cell_index(TARGET, 0);
    let cell_count = (N * N) as usize;

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [1.0, 2.0, 3.0, 1.0];
    pos_seed[neighbour] = [9.0, 8.0, 7.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[cell] = 11;
    type_seed[neighbour] = 22;
    h::seed_beliefs_type(&mut sim, &type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 100;
    tick_seed[neighbour] = 200;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 200;
    conf_seed[neighbour] = 100;
    h::seed_beliefs_confidence(&mut sim, &conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[cell] = 50;
    susp_seed[neighbour] = 25;
    h::seed_beliefs_suspicion(&mut sim, &susp_seed);

    let mut flag_seed = vec![0u32; cell_count];
    flag_seed[cell] = 0xCAFE;
    flag_seed[neighbour] = 0xBEEF;
    h::seed_beliefs_flags(&mut sim, &flag_seed);

    for _ in 0..ERASE_TICK {
        sim.step();
    }
    assert_eq!(sim.tick, ERASE_TICK);

    h::dispatch_erase_belief(&mut sim, CASTER, TARGET, SUBJECT, 0b00111111);

    // Erased cell.
    assert_eq!(h::read_beliefs_pos(&sim)[cell], [0.0, 0.0, 0.0, 1.0], "pos xyz cleared");
    assert_eq!(h::read_beliefs_type(&sim)[cell], 0, "type cleared");
    assert_eq!(h::read_beliefs_tick(&sim)[cell], 0, "tick cleared");
    assert_eq!(h::read_beliefs_confidence(&sim)[cell], 0, "confidence cleared");
    assert_eq!(h::read_beliefs_suspicion(&sim)[cell], 0, "suspicion cleared");
    assert_eq!(h::read_beliefs_flags(&sim)[cell], 0, "flags cleared");

    // Neighbour cell intact.
    assert_eq!(
        h::read_beliefs_pos(&sim)[neighbour],
        [9.0, 8.0, 7.0, 1.0],
        "neighbour pos intact",
    );
    assert_eq!(h::read_beliefs_type(&sim)[neighbour], 22, "neighbour type intact");
    assert_eq!(h::read_beliefs_tick(&sim)[neighbour], 200, "neighbour tick intact");
    assert_eq!(h::read_beliefs_confidence(&sim)[neighbour], 100, "neighbour conf intact");
    assert_eq!(h::read_beliefs_suspicion(&sim)[neighbour], 25, "neighbour susp intact");
    assert_eq!(h::read_beliefs_flags(&sim)[neighbour], 0xBEEF, "neighbour flags intact");
}

#[test]
fn erase_belief_with_partial_fields_clears_only_matching_bits() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_erase_belief_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let cell = cell_index(TARGET, SUBJECT);
    let cell_count = (N * N) as usize;

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [1.0, 2.0, 3.0, 1.0];
    h::seed_beliefs_pos(&mut sim, &pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[cell] = 11;
    h::seed_beliefs_type(&mut sim, &type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 100;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 200;
    h::seed_beliefs_confidence(&mut sim, &conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[cell] = 50;
    h::seed_beliefs_suspicion(&mut sim, &susp_seed);

    let mut flag_seed = vec![0u32; cell_count];
    flag_seed[cell] = 0xCAFE;
    h::seed_beliefs_flags(&mut sim, &flag_seed);

    for _ in 0..ERASE_TICK {
        sim.step();
    }

    h::dispatch_erase_belief(&mut sim, CASTER, TARGET, SUBJECT, 0b00000101);

    assert_eq!(h::read_beliefs_pos(&sim)[cell], [0.0, 0.0, 0.0, 1.0], "pos cleared");
    assert_eq!(h::read_beliefs_tick(&sim)[cell], 0, "tick cleared");
    assert_eq!(h::read_beliefs_type(&sim)[cell], 11, "type intact (bit 1 not set)");
    assert_eq!(h::read_beliefs_confidence(&sim)[cell], 200, "conf intact");
    assert_eq!(h::read_beliefs_suspicion(&sim)[cell], 50, "susp intact");
    assert_eq!(h::read_beliefs_flags(&sim)[cell], 0xCAFE, "flags intact");
}
