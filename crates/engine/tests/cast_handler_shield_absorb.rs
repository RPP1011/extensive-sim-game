//! Combat Foundation Task 12 — shield + damage absorption invariant.
//!
//! Pins the "shield stacks additively + damage bleeds through only on
//! overflow" contract end-to-end: dispatches a sequence of `EffectShieldApplied`
//! and `EffectDamageApplied` events directly and verifies both state fields
//! after each hop.
//!
//! The legacy `DamageHandler` / `ShieldHandler` unit-struct shims were
//! removed in the 2026-04-19 event-taxonomy rename (task 136). Tests now
//! call the compiler-emitted per-event-kind dispatcher directly.

use engine::generated::physics::{dispatch_effect_damage_applied, dispatch_effect_shield_applied};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn_hp(state: &mut SimState, ct: CreatureType, hp: f32) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos: Vec3::ZERO, hp, ..Default::default() }).unwrap()
}

#[test]
fn shields_stack_additively() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 30.0, tick: 0 },
        &mut state,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // 30 + 20 = 50 total absorb pool.
    assert_eq!(state.agent_shield_hp(target), Some(50.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));
}

#[test]
fn damage_below_shield_consumes_shield_only() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 50.0, tick: 0 },
        &mut state,
        &mut events,
    );
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 40.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // 50 - 40 = 10 shield left; hp untouched.
    assert_eq!(state.agent_shield_hp(target), Some(10.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));
}

#[test]
fn damage_through_overflow_hits_hp_on_second_strike() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    // Shield 30 + Shield 20 = 50 absorb pool.
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 30.0, tick: 0 },
        &mut state,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(50.0));

    // Damage 40 → shield 10, hp 100.
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 40.0, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(10.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));

    // Damage 20 → shield soaks 10, hp takes 10 overflow.
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(0.0));
    assert_eq!(state.agent_hp(target),        Some(90.0));
}

#[test]
fn non_positive_shield_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 0.0, tick: 0 },
        &mut state,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: -5.0, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(0.0));
}
