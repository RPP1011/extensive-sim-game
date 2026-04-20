//! Combat Foundation Task 9 — `evaluate_cast_gate` predicate.
//!
//! Each test flips exactly one condition from the "can cast" baseline to
//! `false`, proving that every branch of the six-part conjunction matters.

use engine::ability::{
    evaluate_cast_gate, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0 }).unwrap()
}

struct Fx {
    state:    SimState,
    registry: engine::ability::AbilityRegistry,
    ability:  engine::ability::AbilityId,
    caster:   AgentId,
    target:   AgentId,
}

/// Baseline: Human caster, Wolf target in range, hostile-only gate,
/// ability has 10-tick cooldown, caster un-stunned, not engaged elsewhere.
fn fixture() -> Fx {
    let mut state = SimState::new(8, 42);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(3.0, 0.0, 0.0));

    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,  // range 5m; target is 3m away → in range
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 10.0 }],
    ));
    let registry = b.build();
    Fx { state, registry, ability, caster, target }
}

#[test]
fn baseline_passes() {
    let fx = fixture();
    assert!(evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn stunned_caster_cannot_cast() {
    let mut fx = fixture();
    fx.state.set_agent_stun_remaining(fx.caster, 1);
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn cooldown_pending_blocks_cast() {
    let mut fx = fixture();
    // state.tick starts at 0; set next_ready=5 → gate.tick(0) < 5 → false.
    fx.state.set_agent_cooldown_next_ready(fx.caster, 5);
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn unknown_ability_id_rejects() {
    let fx = fixture();
    let bogus = engine::ability::AbilityId::new(999).unwrap();
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, bogus, fx.target));
}

#[test]
fn dead_target_rejects() {
    let mut fx = fixture();
    fx.state.kill_agent(fx.target);
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn target_out_of_range_rejects() {
    let mut fx = fixture();
    // Move target 10m away; baseline range=5m → out of range.
    fx.state.set_agent_pos(fx.target, Vec3::new(10.0, 0.0, 0.0));
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn non_hostile_target_rejects_when_gate_is_hostile_only() {
    // Re-build with a second Human target (non-hostile).
    let mut state = SimState::new(8, 42);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Human, Vec3::new(2.0, 0.0, 0.0));

    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 10.0 }],
    ));
    let registry = b.build();

    assert!(!evaluate_cast_gate(&state, &registry, caster, ability, target));
}

#[test]
fn non_hostile_target_passes_when_gate_hostile_only_is_false() {
    // Same-species pair, but the ability's gate doesn't require hostility.
    let mut state = SimState::new(8, 42);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Human, Vec3::new(2.0, 0.0, 0.0));

    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 10.0 }],
    ));
    let registry = b.build();

    assert!(evaluate_cast_gate(&state, &registry, caster, ability, target));
}

#[test]
fn engagement_with_non_target_rejects() {
    // Caster is engaged with SOME OTHER agent (not target) → cast at `target`
    // should be rejected. The engager doesn't need to exist as an agent for
    // the predicate's purposes — it's a slot read.
    let mut fx = fixture();
    let third = spawn(&mut fx.state, CreatureType::Wolf, Vec3::new(-3.0, 0.0, 0.0));
    fx.state.set_agent_engaged_with(fx.caster, Some(third));
    assert_ne!(third, fx.target);
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn engagement_with_the_target_itself_passes() {
    // Caster engaged with its intended target → cast is allowed (the common
    // "hit the thing you're in melee with" case).
    let mut fx = fixture();
    fx.state.set_agent_engaged_with(fx.caster, Some(fx.target));
    assert!(evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}

#[test]
fn dead_caster_cannot_cast() {
    let mut fx = fixture();
    fx.state.kill_agent(fx.caster);
    assert!(!evaluate_cast_gate(&fx.state, &fx.registry, fx.caster, fx.ability, fx.target));
}
