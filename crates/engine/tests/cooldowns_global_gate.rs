//! Cooldowns Task 5 — global gate end-to-end through the real cast path.
//!
//! Registers two abilities A and B on different local slots, then pushes
//! an `Event::AgentCast` for A through `CascadeRegistry::<Event>::with_engine_builtins()`
//! so the DSL-emitted `physics cast` rule fires. That rule calls
//! `record_cast_cooldowns`, which writes both the global (GCD) cursor and
//! A's per-slot local cursor.
//!
//! Invariant<Event> pinned: the global cursor, written with
//! `combat.global_cooldown_ticks` (default 5), blocks B even though B's own
//! local cursor is still zero. Once the GCD elapses, B clears.

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
fn global_gate_blocks_other_slot_until_gcd_elapses() {
    // Two abilities on distinct local slots. A has a long local cooldown
    // (30) so we can isolate the global-gate semantics from A's own local
    // cursor. B's local cursor stays at 0 so the only thing that could
    // block B is the shared GCD written by A's cast.
    let mut b = AbilityRegistryBuilder::new();
    let ability_a = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
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
    // Pin GCD to the default (5) explicitly; the test reads it back below
    // so the assertions are phrased in terms of the config value rather
    // than a magic number.
    state.config.combat.global_cooldown_ticks = 5;
    let gcd = state.config.combat.global_cooldown_ticks;

    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(2.0, 0.0, 0.0));

    // Slot indices derived from the ability id (raw - 1). Ability A is
    // registered first → slot 0; B second → slot 1.
    let slot_a = ability_a.slot() as u8;
    let slot_b = ability_b.slot() as u8;
    assert_eq!(slot_a, 0);
    assert_eq!(slot_b, 1);

    // Fresh agent: both gates open at tick 0.
    assert!(state.can_cast_ability(caster, slot_a, 0));
    assert!(state.can_cast_ability(caster, slot_b, 0));

    // Cast A at tick 0 by pushing AgentCast directly — the cascade runs
    // the physics cast rule, which writes both cursors via
    // `record_cast_cooldowns(caster, ability_a, 0)`.
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();
    let mut events = EventRing::<Event>::with_cap(512);
    events.push(Event::AgentCast {
        actor: caster,
        ability: ability_a,
        target,
        depth: 0,
        tick: 0,
    });
    cascade.run_fixed_point(&mut state, &mut events);

    // Post-cast: global cursor at GCD, A's local cursor at 30.
    assert_eq!(state.agent_cooldown_next_ready(caster), Some(gcd));

    // At tick 1 (< GCD), the global gate blocks B even though B's local
    // cursor is still 0.
    assert!(
        !state.can_cast_ability(caster, slot_b, 1),
        "global gate must block B while state.tick=1 < GCD={gcd}"
    );
    // And A is blocked too (by both gates).
    assert!(
        !state.can_cast_ability(caster, slot_a, 1),
        "A is also globally gated at tick 1"
    );

    // At tick == GCD, global clears. B's local is 0 → B ready.
    assert!(
        state.can_cast_ability(caster, slot_b, gcd),
        "B must be ready at tick == GCD ({gcd})"
    );
    // A's local (30) still blocks A even though the global has cleared.
    assert!(
        !state.can_cast_ability(caster, slot_a, gcd),
        "A's local cursor still blocks at tick == GCD"
    );
}
