//! Combat Foundation Task 13 — `StunHandler` writes `hot_stun_remaining_ticks`
//! with a longer-stun-wins rule; the unified tick-start decrement then counts
//! it down to zero and emits `StunExpired` exactly once.
//!
//! Cross-check: while `hot_stun_remaining_ticks > 0`, `evaluate_cast_gate`
//! returns false (branch 1 of the gate conjunction). After `StunExpired`
//! fires, the gate allows casting again.

use std::sync::Arc;

use engine::ability::{
    evaluate_cast_gate, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate, StunHandler,
};
use engine::ability::expire::tick_start;
use engine::cascade::CascadeHandler;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0 }).unwrap()
}

#[test]
fn stun_writes_duration_when_longer() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    StunHandler.handle(
        &Event::EffectStunApplied { caster, target, duration_ticks: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_remaining(target), Some(10));
}

#[test]
fn longer_stun_overrides_shorter_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_remaining(target, 3);
    StunHandler.handle(
        &Event::EffectStunApplied { caster, target, duration_ticks: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_remaining(target), Some(10));
}

#[test]
fn shorter_stun_does_not_override_longer_existing() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_stun_remaining(target, 15);
    StunHandler.handle(
        &Event::EffectStunApplied { caster, target, duration_ticks: 5, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_remaining(target), Some(15));
}

#[test]
fn stun_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    state.kill_agent(target);

    StunHandler.handle(
        &Event::EffectStunApplied { caster, target, duration_ticks: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_remaining(target), Some(0));
}

#[test]
fn stun_gates_caster_for_exact_duration_then_expires() {
    // A stunned AGENT cannot cast — branch 1 of evaluate_cast_gate. Set a
    // 10-tick stun on the caster, verify gate=false for 10 ticks, after 10
    // tick_start decrements `StunExpired` fires once, and gate=true again.
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

    // Apply 10-tick stun. StunHandler is dispatched directly on the caster.
    StunHandler.handle(
        &Event::EffectStunApplied { caster: target, target: caster, duration_ticks: 10, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_stun_remaining(caster), Some(10));

    // For 10 ticks the gate rejects. After each `tick_start`, the remaining
    // count decrements by 1; only the 10th call takes it to 0 and emits
    // `StunExpired`.
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    for tick in 0..10u32 {
        assert!(
            !evaluate_cast_gate(&state, &registry, caster, ability, target),
            "gate must reject at tick {tick} (stun_remaining={:?})",
            state.agent_stun_remaining(caster)
        );
        tick_start(&mut state, &mut scratch, &mut events);
        state.tick += 1;
    }

    // Exactly one StunExpired emitted over the 10 ticks.
    let n_expired = events.iter().filter(|e| matches!(e, Event::StunExpired { .. })).count();
    assert_eq!(n_expired, 1);
    assert_eq!(state.agent_stun_remaining(caster), Some(0));
    assert!(evaluate_cast_gate(&state, &registry, caster, ability, target));
}
