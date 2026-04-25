//! Combat Foundation Task 13 + Task 143 — stun writes
//! `hot_stun_expires_at_tick` with a longer-stun-wins rule.

use engine::ability::{AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate};
use engine_rules::mask::mask_cast;
use engine_rules::physics::dispatch_effect_stun_applied;
use engine_rules::views::ViewRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
}

#[test]
fn stun_writes_expiry_when_later() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(10));
}

#[test]
fn longer_stun_overrides_shorter_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_expires_at(target, 3);
    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(10));
}

#[test]
fn shorter_stun_does_not_override_longer_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_expires_at(target, 15);
    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 5, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(15));
}

#[test]
fn stun_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    state.kill_agent(target);

    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(0));
}

#[test]
fn stun_gates_caster_until_tick_reaches_expiry() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(128);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    state.ability_registry = b.build();
    let _ = target;

    assert!(mask_cast(&state, caster, ability));

    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: target, target: caster, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(caster), Some(10));
    assert!(state.agent_stunned(caster));

    for tick in 0..10u32 {
        state.tick = tick;
        assert!(
            !mask_cast(&state, caster, ability),
            "mask must reject at tick {tick} (expires_at={:?})",
            state.agent_stun_expires_at(caster)
        );
        assert!(state.agent_stunned(caster), "agent must read stunned at tick {tick}");
    }

    state.tick = 10;
    assert!(!state.agent_stunned(caster));
    assert!(mask_cast(&state, caster, ability));
    assert_eq!(state.agent_stun_expires_at(caster), Some(10));

    for e in events.iter() {
        match e {
            Event::EffectStunApplied { .. } => {}
            _ => panic!("unexpected event emitted: {e:?}"),
        }
    }
}
