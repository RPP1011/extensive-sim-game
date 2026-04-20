//! Combat Foundation Task 10 — `DamageHandler` applies damage with shield-first
//! absorption and emits `AgentDied` + `kill_agent` on lethal overflow.
//!
//! Every test dispatches an `Event::EffectDamageApplied` directly and inspects
//! the resulting state mutation + event tail. That's the layer the cascade
//! exposes; wiring a full `AgentCast` through the pipeline is covered by
//! `action_cast_emits_agentcast.rs` (Task 9) and the acceptance tests.

use engine::ability::DamageHandler;
use engine::cascade::CascadeHandler;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn_hp(state: &mut SimState, ct: CreatureType, hp: f32) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos: Vec3::ZERO, hp }).unwrap()
}

#[test]
fn shield_absorbs_before_hp_drops() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  100.0);
    state.set_agent_shield_hp(target, 10.0);

    DamageHandler.handle(
        &Event::EffectDamageApplied { caster, target, amount: 30.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // Shield soaked 10, hp took 20.
    assert_eq!(state.agent_shield_hp(target), Some(0.0));
    assert_eq!(state.agent_hp(target),        Some(80.0));
    // Survived → no AgentDied emitted.
    let died = events.iter().any(|e| matches!(e, Event::AgentDied { .. }));
    assert!(!died, "non-lethal damage must not emit AgentDied");
}

#[test]
fn damage_bleeds_through_zero_shield_entirely_to_hp() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  100.0);
    assert_eq!(state.agent_shield_hp(target), Some(0.0));

    DamageHandler.handle(
        &Event::EffectDamageApplied { caster, target, amount: 25.0, tick: 0 },
        &mut state,
        &mut events,
    );

    assert_eq!(state.agent_shield_hp(target), Some(0.0));
    assert_eq!(state.agent_hp(target),        Some(75.0));
}

#[test]
fn lethal_damage_emits_agent_died_and_kills() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  50.0);

    DamageHandler.handle(
        &Event::EffectDamageApplied { caster, target, amount: 100.0, tick: 7 },
        &mut state,
        &mut events,
    );

    assert_eq!(state.agent_hp(target), Some(0.0));
    assert!(!state.agent_alive(target), "kill_agent must flip the alive bit");

    let died = events.iter().find_map(|e| match e {
        Event::AgentDied { agent_id, tick } => Some((*agent_id, *tick)),
        _ => None,
    });
    assert_eq!(died, Some((target, 7)));
}

#[test]
fn damage_on_dead_target_is_a_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  50.0);
    state.kill_agent(target);

    DamageHandler.handle(
        &Event::EffectDamageApplied { caster, target, amount: 999.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // No state mutation, no second AgentDied.
    let n_died = events.iter().filter(|e| matches!(e, Event::AgentDied { .. })).count();
    assert_eq!(n_died, 0);
}
