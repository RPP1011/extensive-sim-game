//! Combat Foundation Task 13 + Task 143 — stun writes
//! `hot_stun_expires_at_tick` with a longer-stun-wins rule. Expiry is a
//! synthetic boundary: `state.tick < expires_at_tick` means stunned.
//!
//! Cross-check: while `state.tick < stun_expires_at_tick`,
//! `evaluate_cast_gate` returns false (branch 1 of the gate conjunction).
//! Once `state.tick` reaches the expiry, the gate allows casting again —
//! no per-tick decrement pass, no `StunExpired` event.
//!
//! The legacy `StunHandler` unit-struct shim was removed in the 2026-04-19
//! event-taxonomy rename (task 136). Tests now call the compiler-emitted
//! per-event-kind dispatcher directly.

use std::sync::Arc;

use engine::ability::{
    evaluate_cast_gate, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::generated::physics::dispatch_effect_stun_applied;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
}

#[test]
fn stun_writes_expiry_when_later() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(10));
}

#[test]
fn longer_stun_overrides_shorter_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_expires_at(target, 3);
    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(10));
}

#[test]
fn shorter_stun_does_not_override_longer_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_expires_at(target, 15);
    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 5, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(15));
}

#[test]
fn stun_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    state.kill_agent(target);

    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: caster, target, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(target), Some(0));
}

#[test]
fn stun_gates_caster_until_tick_reaches_expiry() {
    // A stunned AGENT cannot cast — branch 1 of evaluate_cast_gate.
    // Task 143: set `stun_expires_at_tick = 10` at tick 0; gate must
    // reject for ticks 0..10 and accept at tick 10. No per-tick decrement;
    // the gate just reads the absolute tick.
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(128);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    let mut b = AbilityRegistryBuilder::new();
    let ability = b.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 1.0 }],
    ));
    let registry: Arc<_> = Arc::new(b.build());

    // Baseline: gate passes.
    assert!(evaluate_cast_gate(&state, &registry, caster, ability, target));

    // Apply a stun that expires at tick 10. The actor is `target` (it's
    // the one stunning the caster); the stunned agent is `caster`.
    dispatch_effect_stun_applied(
        &Event::EffectStunApplied { actor: target, target: caster, expires_at_tick: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_expires_at(caster), Some(10));
    assert!(state.agent_stunned(caster));

    // For ticks 0..10 the gate rejects (state.tick < 10 = stunned).
    for tick in 0..10u32 {
        state.tick = tick;
        assert!(
            !evaluate_cast_gate(&state, &registry, caster, ability, target),
            "gate must reject at tick {tick} (expires_at={:?})",
            state.agent_stun_expires_at(caster)
        );
        assert!(state.agent_stunned(caster), "agent must read stunned at tick {tick}");
    }

    // At tick 10, state.tick >= expires_at_tick → no longer stunned.
    state.tick = 10;
    assert!(!state.agent_stunned(caster));
    assert!(evaluate_cast_gate(&state, &registry, caster, ability, target));
    // The stored expiry is unchanged — the "expiry" is purely a read-side
    // synthesis, no `StunExpired` event was emitted (in fact the event
    // variant no longer exists in the enum).
    assert_eq!(state.agent_stun_expires_at(caster), Some(10));

    // And no StunExpired-like event appears anywhere in the ring.
    // (The Event enum no longer even declares that variant.)
    for e in events.iter() {
        // Just ensure we only see the EffectStunApplied we pushed.
        match e {
            Event::EffectStunApplied { .. } => {}
            _ => panic!("unexpected event emitted: {e:?}"),
        }
    }
}
