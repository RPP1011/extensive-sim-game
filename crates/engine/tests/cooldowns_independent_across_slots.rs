//! Cooldowns Task 5 — local cooldowns are independent per (agent, slot).
//!
//! Scenario:
//!   1. Cast A (local cd 20) at tick 0. Global cursor → GCD (=5), A's
//!      local cursor → 20.
//!   2. Cast B (local cd 3) at tick GCD (=5). Global cursor → GCD*2
//!      (=10), B's local cursor → GCD + 3 = 8. A's cursor (20) untouched.
//!   3. At tick GCD+1 (=6): A is locally gated (20>6); B is globally
//!      gated because B's cast just bumped the GCD cursor to 10.
//!   4. At tick GCD*2 (=10): global clears (10<=10), B's local clears
//!      (8<=10) → B ready. A's local (20) still blocks A.
//!   5. At tick 20: A's local clears. B remains ready.
//!
//! This end-to-end run exercises the dual-cursor write from Task 4: the
//! two casts mutate only the per-slot cursor for the ability that just
//! fired, while the shared global cursor is rewritten on every cast.

use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: ct,
            pos,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn local_cooldowns_are_independent_across_slots() {
    let mut b = AbilityRegistryBuilder::new();
    let ability_a = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 20, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let ability_b = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 3, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let registry = b.build();

    let mut state = SimState::new(8, 42);
    state.ability_registry = registry;
    state.config.combat.global_cooldown_ticks = 5;
    let gcd = state.config.combat.global_cooldown_ticks;

    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(2.0, 0.0, 0.0));

    let slot_a = ability_a.slot() as u8;
    let slot_b = ability_b.slot() as u8;

    let cascade = CascadeRegistry::with_engine_builtins();
    let mut events = EventRing::with_cap(512);

    // Step 1: cast A at tick 0.
    events.push(Event::AgentCast {
        actor: caster,
        ability: ability_a,
        target,
        depth: 0,
        tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(gcd));

    // Step 2: cast B at tick GCD. Set state.tick so any emitter that
    // reads the world clock sees the advanced value; pass `tick: gcd` on
    // the AgentCast so `record_cast_cooldowns` computes cursors off the
    // right `now`. (The physics cast rule reads the event's tick, not
    // state.tick, for the cooldown write — but keeping them in sync
    // matches how the step loop drives things.)
    state.tick = gcd;
    events.push(Event::AgentCast {
        actor: caster,
        ability: ability_b,
        target,
        depth: 0,
        tick: gcd,
    });
    cascade.run_fixed_point(&mut state, &mut events);
    // B's cast bumped the global cursor to gcd + gcd = 2*GCD.
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(gcd * 2));

    // Step 3: at tick GCD+1, A still locally gated (local=20 > GCD+1),
    // B still globally gated (global=10 > GCD+1=6).
    let now = gcd + 1;
    assert!(
        !state.can_cast_ability(caster, slot_a, now),
        "A locally gated at tick GCD+1={now} (A's local=20)"
    );
    assert!(
        !state.can_cast_ability(caster, slot_b, now),
        "B globally gated at tick GCD+1={now} (global={})",
        gcd * 2
    );

    // Step 4: at tick GCD*2, global clears; B's local (gcd+3=8) has
    // already cleared; A's local (20) still blocks.
    let now = gcd * 2;
    assert!(
        state.can_cast_ability(caster, slot_b, now),
        "B ready at tick GCD*2={now} (global cleared, B's local {} <= {now})",
        gcd + 3
    );
    assert!(
        !state.can_cast_ability(caster, slot_a, now),
        "A still locally gated at tick GCD*2={now} (A's local=20)"
    );

    // Step 5: at tick 20, A's local clears.
    assert!(
        state.can_cast_ability(caster, slot_a, 20),
        "A clears locally at tick 20"
    );
    assert!(
        state.can_cast_ability(caster, slot_b, 20),
        "B still ready at tick 20"
    );
}
