//! Combat Foundation Task 17 — `ModifyStandingHandler` routes
//! `Event::EffectStandingDelta` into `SparseStandings::adjust`.
//!
//! Invariants pinned:
//! - Initial standing is 0 (default for any pair not yet written).
//! - Symmetric key: `(A, B)` and `(B, A)` alias the same slot.
//! - Clamp `[-1000, 1000]` is applied on the final value (silent saturation).
//! - Zero delta short-circuits — no entry inserted, pair stays untracked.

use engine::generated::physics::dispatch_effect_standing_delta;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos: Vec3::ZERO, hp: 100.0 }).unwrap()
}

#[test]
fn positive_delta_adds_to_zero_baseline() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: 50, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 50);
}

#[test]
fn saturates_at_upper_clamp() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    // i16::MAX for delta would push past 1000 → clamped at 1000.
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: i16::MAX, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 1000);
}

#[test]
fn saturates_at_lower_clamp() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: i16::MIN, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), -1000);
}

#[test]
fn standing_is_symmetric_across_directions() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    // +100 on (A, B) followed by -100 on (B, A) must land on ZERO —
    // both events reference the same symmetric slot.
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: 100, tick: 0 },
        &mut state,
        &mut events,
    );
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a: b, b: a, delta: -100, tick: 1 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 0);
    assert_eq!(state.standing(b, a), 0);
}

#[test]
fn accumulated_adjustments_are_stable_after_clamp() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    // Take it up to the ceiling, then walk it back down — every step
    // observes the post-clamp value.
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: 1_500, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 1000);
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: -50, tick: 1 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 950);
    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: -2000, tick: 2 },
        &mut state,
        &mut events,
    );
    // 950 + (-2000) saturates at -1000.
    assert_eq!(state.standing(a, b), -1000);
}

#[test]
fn zero_delta_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);
    assert_eq!(state.standing(a, b), 0);

    dispatch_effect_standing_delta(
        &Event::EffectStandingDelta { a, b, delta: 0, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.standing(a, b), 0);
    assert!(state.cold_standing().is_empty(), "no entry should be inserted for a zero delta");
}

#[test]
fn registry_dispatches_standing_delta_via_builtins() {
    use engine::cascade::CascadeRegistry;
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let cascade = CascadeRegistry::with_engine_builtins();
    let a = spawn(&mut state, CreatureType::Human);
    let b = spawn(&mut state, CreatureType::Human);

    events.push(Event::EffectStandingDelta { a, b, delta: 200, tick: 0 });
    cascade.run_fixed_point(&mut state, &mut events);
    assert_eq!(state.standing(a, b), 200);

    events.push(Event::EffectStandingDelta { a: b, b: a, delta: -50, tick: 1 });
    cascade.run_fixed_point(&mut state, &mut events);
    assert_eq!(state.standing(a, b), 150);
}
