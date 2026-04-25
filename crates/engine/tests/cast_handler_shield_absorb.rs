//! Combat Foundation Task 12 — shield + damage absorption invariant.

use engine_rules::physics::{dispatch_effect_damage_applied, dispatch_effect_shield_applied};
use engine_rules::views::ViewRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn_hp(state: &mut SimState, ct: CreatureType, hp: f32) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos: Vec3::ZERO, hp, ..Default::default() }).unwrap()
}

#[test]
fn shields_stack_additively() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 30.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(state.agent_shield_hp(target), Some(50.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));
}

#[test]
fn damage_below_shield_consumes_shield_only() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 50.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 40.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );

    assert_eq!(state.agent_shield_hp(target), Some(10.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));
}

#[test]
fn damage_through_overflow_hits_hp_on_second_strike() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 30.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(50.0));

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 40.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(10.0));
    assert_eq!(state.agent_hp(target),        Some(100.0));

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(0.0));
    assert_eq!(state.agent_hp(target),        Some(90.0));
}

#[test]
fn non_positive_shield_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);

    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: 0.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    dispatch_effect_shield_applied(
        &Event::EffectShieldApplied { actor: caster, target, amount: -5.0, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_shield_hp(target), Some(0.0));
}
