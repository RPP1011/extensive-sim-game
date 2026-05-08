//! Wave 3 ToM Phase 5 closed-loop pin: caster A's disguise verb stamps
//! the per-agent `disguise_expires_at_tick` + `disguise_fake_type` SoA
//! columns, and observe substitutes the fake type for observers — until
//! the disguise window expires, after which observe sees the true type
//! again.
//!
//! ## What this exercises
//!
//! Phase 5 introduces 2 new per-agent SoA columns
//! (`disguise_expires_at_tick`, `disguise_fake_type`) and a
//! .sim-authored consumer rule (`physics_ApplyDisguise` in
//! `tom_probe.sim`) that decodes the chronicle event payload and writes
//! both columns. The observe consumer (also in `tom_probe.sim`) now
//! consults `disguise_expires_at_tick` against `world.tick` and
//! substitutes `disguise_fake_type` for the true `creature_type` while
//! the disguise window is active.
//!
//! ## Fixture
//!
//! 2 agents.
//! - Agent 0 (= caster A) is pre-seeded with `creature_type = 5` (the
//!   "true" type — say, "spy").
//! - Agent 1 (= observer B) is the watcher.
//!
//! At tick 0, A casts `disguise(fake_type = 7, duration_ticks = 10)`.
//! At tick 5, B observes A → `beliefs_creature_type[1*N+0]` should be
//! `7` (the fake type).
//! After tick 10, B re-observes A → `beliefs_creature_type[1*N+0]`
//! should be `5` (the true type — the disguise window expired).

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xD15_C0_DE_FACE;
const N: u32 = 2;
const CASTER: u32 = 0;
const OBSERVER: u32 = 1;
const TRUE_TYPE: u8 = 5;
const FAKE_TYPE: u8 = 7;
const DURATION: u32 = 10;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn disguise_active_substitutes_fake_type_at_observe_time() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[disguise_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Seed caster A's true creature_type and a position so observe has
    // something to copy.
    let mut type_seed = vec![0u8; N as usize];
    type_seed[CASTER as usize] = TRUE_TYPE;
    sim.seed_agent_creature_type(&type_seed);

    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[CASTER as usize] = [11.0, 22.0, 33.0, 1.0];
    sim.seed_agent_pos(&pos_seed);

    // Tick 0 — A casts disguise (fake_type = 7, duration = 10).
    // The consumer writes:
    //   disguise_expires_at_tick[CASTER] = world.tick + duration = 10
    //   disguise_fake_type[CASTER]       = 7
    sim.disguise(CASTER, FAKE_TYPE, DURATION);

    // Verify the SoA columns landed.
    let exp = sim.agent_disguise_expires_at_tick().to_vec();
    assert_eq!(
        exp[CASTER as usize], DURATION,
        "disguise_expires_at_tick[CASTER] should equal world.tick + DURATION = 0 + 10",
    );
    let fake = sim.agent_disguise_fake_type().to_vec();
    assert_eq!(
        fake[CASTER as usize], FAKE_TYPE as u32,
        "disguise_fake_type[CASTER] should equal FAKE_TYPE",
    );

    // Tick the sim forward to tick 5 — disguise still active.
    for _ in 0..5 {
        sim.step();
    }
    assert_eq!(sim.tick(), 5);

    // Observer B observes caster A. Because A's disguise is active
    // (`disguise_expires_at_tick = 10 > world.tick = 5`), the observe
    // consumer substitutes A's fake_type for the true creature_type.
    sim.observe(OBSERVER, CASTER);

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (CASTER as usize);
    let types_active = sim.beliefs_type().to_vec();
    assert_eq!(
        types_active[cell], FAKE_TYPE,
        "While disguise active (tick 5 < expires 10), observer should see FAKE_TYPE in beliefs_type",
    );

    // Tick the sim past the disguise expiry. After tick 10, the
    // disguise window has lapsed.
    for _ in 0..6 {
        sim.step();
    }
    assert_eq!(sim.tick(), 11);
    // 11 > 10 → disguise inactive.

    // Re-observe — should see the TRUE type now.
    sim.observe(OBSERVER, CASTER);
    let types_expired = sim.beliefs_type().to_vec();
    assert_eq!(
        types_expired[cell], TRUE_TYPE,
        "After disguise expires (tick 11 >= expires 10), observer should see TRUE_TYPE",
    );
}

#[test]
fn disguise_consumer_writes_only_caster_slot() {
    // Disguise must NOT touch any non-caster agent's SoA cells. With
    // 4 agents and a self-cast on slot 1, slots 0/2/3 should remain at
    // their default zero.
    const N4: u32 = 4;
    let mut sim = match std::panic::catch_unwind(|| TomProbeState::new(SEED, N4)).ok() {
        Some(s) => s,
        None => {
            eprintln!("[disguise_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    sim.disguise(1, 9, 50);

    let exp = sim.agent_disguise_expires_at_tick().to_vec();
    assert_eq!(exp, vec![0u32, 50, 0, 0]);
    let fake = sim.agent_disguise_fake_type().to_vec();
    assert_eq!(fake, vec![0u32, 9, 0, 0]);
}
