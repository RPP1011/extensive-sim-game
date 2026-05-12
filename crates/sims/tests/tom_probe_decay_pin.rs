//! Mega-crate-hosted port of the Wave 3 ToM Phase 3 decay-slope pin
//! (formerly `crates/tom_probe_runtime/tests/decay_pin.rs`).
//!
//! Confidence decays linearly at `decay_per_tick = 1` per tick of
//! staleness, saturating at 0. Fresh observation re-pegs at 255 and
//! resets the decay anchor.
//!
//! ## Decay kernel sourcing
//!
//! Today the decay kernel is dispatched out-of-band via the in-test
//! `tom_probe_helpers::decay_step` shim, which compiles
//! `dsl_compiler::belief_decay_wgsl::decay_kernel_wgsl()` and dispatches
//! it manually (the original tom_probe_runtime did the same). A parallel
//! workstream is folding decay into the compiler-emitted `@decay`
//! pipeline; once that lands, the shim can retire.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xDECA_F00D_BEEF_BAD;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET: u32 = 1;

#[test]
fn decay_slope_is_one_per_tick() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);
    let cell_count = (N * N) as usize;

    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    h::seed_beliefs_confidence(&mut sim, &confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    for _ in 0..100 {
        sim.step();
        h::decay_step(&mut sim);
    }
    assert_eq!(sim.tick, 100);

    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(
        confidence[cell], 155,
        "after 100 ticks at decay_per_tick=1, confidence should be 255 - 100 = 155",
    );

    for c in 0..cell_count {
        if c == cell {
            continue;
        }
        assert_eq!(
            confidence[c], 0,
            "decay sweep should not corrupt unseeded cell {c}",
        );
    }
}

#[test]
fn decay_saturates_at_zero() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);
    let cell_count = (N * N) as usize;

    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    h::seed_beliefs_confidence(&mut sim, &confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    for _ in 0..300 {
        sim.step();
        h::decay_step(&mut sim);
    }

    let confidence = h::read_beliefs_confidence(&sim);
    assert_eq!(
        confidence[cell], 0,
        "after 300 ticks, decay should saturate confidence at 0",
    );
}

#[test]
fn fresh_observation_resets_decay() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [42.0, 0.0, 0.0, 1.0];
    h::seed_agent_pos(&mut sim, &pos_seed);
    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 7;
    h::seed_agent_creature_type(&mut sim, &type_seed);

    let cell_count = (N * N) as usize;
    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    h::seed_beliefs_confidence(&mut sim, &confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    h::seed_beliefs_tick(&mut sim, &tick_seed);

    for _ in 0..155 {
        sim.step();
        h::decay_step(&mut sim);
    }
    assert_eq!(sim.tick, 155);
    let conf_after_decay = h::read_beliefs_confidence(&sim);
    assert_eq!(
        conf_after_decay[cell], 100,
        "after 155 ticks decay should land confidence at 255 - 155 = 100",
    );

    h::dispatch_observe(&mut sim, OBSERVER, TARGET);
    let conf_after_observe = h::read_beliefs_confidence(&sim);
    assert_eq!(
        conf_after_observe[cell], 255,
        "fresh observation should re-peg confidence at 255",
    );
    let ticks_after_observe = h::read_beliefs_tick(&sim);
    assert_eq!(
        ticks_after_observe[cell], 155,
        "fresh observation should set last_seen_tick to current tick (155)",
    );

    for _ in 0..50 {
        sim.step();
        h::decay_step(&mut sim);
    }
    assert_eq!(sim.tick, 205);
    let conf_after_more_decay = h::read_beliefs_confidence(&sim);
    assert_eq!(
        conf_after_more_decay[cell], 205,
        "after 50 more ticks, decay should land confidence at 255 - 50 = 205",
    );
}
