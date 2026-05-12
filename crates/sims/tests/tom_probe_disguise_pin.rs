//! Mega-crate-hosted port of the Wave 3 ToM Phase 5 disguise round-trip
//! pin (formerly `crates/tom_probe_runtime/tests/disguise_round_trip_pin.rs`).
//!
//! Caster's disguise verb stamps the per-agent
//! `disguise_expires_at_tick` + `disguise_fake_type` SoA columns; while
//! the disguise window is active, observers see the fake type when
//! observing the caster.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xD15_C0_DE_FACE;
const N: u32 = 2;
const CASTER: u32 = 0;
const OBSERVER: u32 = 1;
const TRUE_TYPE: u8 = 5;
const FAKE_TYPE: u8 = 7;
const DURATION: u32 = 10;

#[test]
fn disguise_active_substitutes_fake_type_at_observe_time() {
    let mut sim = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_disguise_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let mut type_seed = vec![0u8; N as usize];
    type_seed[CASTER as usize] = TRUE_TYPE;
    h::seed_agent_creature_type(&mut sim, &type_seed);

    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[CASTER as usize] = [11.0, 22.0, 33.0, 1.0];
    h::seed_agent_pos(&mut sim, &pos_seed);

    h::dispatch_disguise(&mut sim, CASTER, FAKE_TYPE, DURATION);

    let exp = h::read_agent_disguise_expires_at_tick(&sim);
    assert_eq!(
        exp[CASTER as usize], DURATION,
        "disguise_expires_at_tick[CASTER] should equal world.tick + DURATION = 0 + 10",
    );
    let fake = h::read_agent_disguise_fake_type(&sim);
    assert_eq!(
        fake[CASTER as usize], FAKE_TYPE as u32,
        "disguise_fake_type[CASTER] should equal FAKE_TYPE",
    );

    for _ in 0..5 {
        sim.step();
    }
    assert_eq!(sim.tick, 5);

    h::dispatch_observe(&mut sim, OBSERVER, CASTER);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (CASTER as usize);
    let types_active = h::read_beliefs_type(&sim);
    assert_eq!(
        types_active[cell], FAKE_TYPE,
        "While disguise active, observer should see FAKE_TYPE",
    );

    for _ in 0..6 {
        sim.step();
    }
    assert_eq!(sim.tick, 11);

    h::dispatch_observe(&mut sim, OBSERVER, CASTER);
    let types_expired = h::read_beliefs_type(&sim);
    assert_eq!(
        types_expired[cell], TRUE_TYPE,
        "After disguise expires, observer should see TRUE_TYPE",
    );
}

#[test]
fn disguise_consumer_writes_only_caster_slot() {
    const N4: u32 = 4;
    let mut sim = match h::try_new(SEED, N4) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_disguise_pin] skipping: no wgpu adapter");
            return;
        }
    };

    h::dispatch_disguise(&mut sim, 1, 9, 50);

    let exp = h::read_agent_disguise_expires_at_tick(&sim);
    assert_eq!(exp, vec![0u32, 50, 0, 0]);
    let fake = h::read_agent_disguise_fake_type(&sim);
    assert_eq!(fake, vec![0u32, 9, 0, 0]);
}
