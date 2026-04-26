//! Combat Foundation — 2v2 cast regression fixture.
//!
//! Named scenario: two casters per side with a single-target damage ability.
//! Mechanics exercised:
//!   - `AgentCast` event triggers `EffectDamageApplied` via the builtin cascade.
//!   - Cooldown is written after a cast fires.
//!   - HP drops by the cast amount on the target.
//!   - Lethal cast emits `AgentDied`.
//!
//! Uses `dispatch_*` directly (same pattern as `cast_handler_damage.rs`).
//! A `with_engine_builtins()` cascade handles the `AgentCast → EffectDamageApplied`
//! chain through `run_fixed_point`.

use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;
use glam::Vec3;

fn spawn_human(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos,
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .expect("spawn human")
}

fn spawn_wolf(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos,
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .expect("spawn wolf")
}

/// Build a single-target damage ability with `amount=25` and `cooldown=10`.
fn register_bolt_ability(state: &mut SimState) -> engine::ability::AbilityId {
    let mut b = AbilityRegistryBuilder::new();
    let id = b.register(AbilityProgram::new_single_target(
        6.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: true,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 25.0 }],
    ));
    state.ability_registry = b.build();
    id
}

/// Two humans vs two wolves — cast cross-fires produce correct HP deltas.
#[test]
fn two_casters_per_side_cast_deals_expected_damage() {
    let mut state = SimState::new(8, 42);
    let cascade = engine_rules::with_engine_builtins();
    let ability = register_bolt_ability(&mut state);

    // Humans on one side; wolves on the other, within cast range (6 m).
    let h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let w1 = spawn_wolf(&mut state, Vec3::new(4.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0));

    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    // H1 casts at W1; H2 casts at W2.
    events.push(Event::AgentCast {
        actor: h1,
        ability,
        target: w1,
        depth: 0,
        tick: state.tick,
    });
    events.push(Event::AgentCast {
        actor: h2,
        ability,
        target: w2,
        depth: 0,
        tick: state.tick,
    });

    cascade.run_fixed_point(&mut state, &mut views, &mut events);

    // Each wolf took 25 damage from 80 hp → 55.
    assert_eq!(
        state.agent_hp(w1),
        Some(55.0),
        "W1 hp after bolt from H1: expected 55"
    );
    assert_eq!(
        state.agent_hp(w2),
        Some(55.0),
        "W2 hp after bolt from H2: expected 55"
    );
    // Humans untouched.
    assert_eq!(state.agent_hp(h1), Some(100.0));
    assert_eq!(state.agent_hp(h2), Some(100.0));

    // Exactly two EffectDamageApplied events emitted.
    let damage_events = events
        .iter()
        .filter(|e| matches!(e, Event::EffectDamageApplied { .. }))
        .count();
    assert_eq!(damage_events, 2, "expected 2 damage effect events");
}

/// Cooldown is set after the first cast; a second cast at the same tick
/// does NOT re-fire (the cascade handles `AgentCast` once per push, but the
/// gate state is updated so the next cast after cooldown_ticks would pass).
#[test]
fn cast_sets_cooldown_on_caster() {
    let mut state = SimState::new(8, 42);
    let cascade = engine_rules::with_engine_builtins();
    let ability = register_bolt_ability(&mut state);

    let caster = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let target = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));

    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    assert_eq!(
        state.agent_cooldown_next_ready(caster),
        Some(0),
        "cooldown should be 0 before any cast"
    );

    events.push(Event::AgentCast {
        actor: caster,
        ability,
        target,
        depth: 0,
        tick: state.tick,
    });
    cascade.run_fixed_point(&mut state, &mut views, &mut events);

    // GCD must be set (> 0). The exact value matches config.combat.global_cooldown_ticks.
    let gcd_after = state.agent_cooldown_next_ready(caster).unwrap_or(0);
    assert!(
        gcd_after > 0,
        "GCD should be set after a cast; got {gcd_after}"
    );

    // Local ability slot cooldown must also be set to cooldown_ticks=10.
    let agent_idx = (caster.raw() - 1) as usize;
    let local_cd = state.ability_cooldowns[agent_idx][0];
    assert_eq!(local_cd, 10, "local ability cooldown must be 10 after cast");
}

/// Lethal cast (damage > remaining HP) emits AgentDied and kills the target.
#[test]
fn lethal_cast_kills_target() {
    let mut state = SimState::new(8, 42);
    let cascade = engine_rules::with_engine_builtins();

    // Ability with damage=200, enough to one-shot any target.
    let mut b = AbilityRegistryBuilder::new();
    let kill_bolt = b.register(AbilityProgram::new_single_target(
        6.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: true,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 200.0 }],
    ));
    state.ability_registry = b.build();

    let caster = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let target = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));

    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();

    events.push(Event::AgentCast {
        actor: caster,
        ability: kill_bolt,
        target,
        depth: 0,
        tick: state.tick,
    });
    cascade.run_fixed_point(&mut state, &mut views, &mut events);

    assert_eq!(state.agent_hp(target), Some(0.0), "HP must reach 0 on lethal cast");
    assert!(!state.agent_alive(target), "target must be dead");

    let died = events
        .iter()
        .any(|e| matches!(e, Event::AgentDied { agent_id, .. } if *agent_id == target));
    assert!(died, "AgentDied must be emitted for lethal cast");
}
