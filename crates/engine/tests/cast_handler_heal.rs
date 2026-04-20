//! Combat Foundation Task 11 — `HealHandler` clamps at `max_hp`.

use engine::ability::HealHandler;
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
fn heal_under_cap_applies_full_amount() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);
    // Drop target to 40/100.
    state.set_agent_hp(target, 40.0);

    HealHandler.handle(
        &Event::EffectHealApplied { caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut events,
    );

    assert_eq!(state.agent_hp(target), Some(60.0));
}

#[test]
fn heal_clamps_to_max_hp_when_saturated() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);
    state.set_agent_hp(target, 95.0);  // 5 hp of headroom

    HealHandler.handle(
        &Event::EffectHealApplied { caster, target, amount: 20.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // Real applied delta is 5 → hp lands exactly at max.
    assert_eq!(state.agent_hp(target), Some(100.0));
    assert_eq!(state.agent_hp(target), state.agent_max_hp(target));
}

#[test]
fn heal_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);
    state.set_agent_hp(target, 0.0);
    state.kill_agent(target);

    HealHandler.handle(
        &Event::EffectHealApplied { caster, target, amount: 50.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // hp not mutated; alive bit stays false.
    assert_eq!(state.agent_hp(target), Some(0.0));
    assert!(!state.agent_alive(target));
}

#[test]
fn heal_non_positive_amount_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 100.0);
    let target = spawn_hp(&mut state, CreatureType::Human, 100.0);
    state.set_agent_hp(target, 40.0);

    HealHandler.handle(
        &Event::EffectHealApplied { caster, target, amount: 0.0, tick: 0 },
        &mut state,
        &mut events,
    );
    HealHandler.handle(
        &Event::EffectHealApplied { caster, target, amount: -10.0, tick: 0 },
        &mut state,
        &mut events,
    );

    assert_eq!(state.agent_hp(target), Some(40.0));
}
