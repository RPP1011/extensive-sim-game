//! Combat Foundation Task 14 — `SlowHandler` + MoveToward speed composition.
//!
//! Three axes verified:
//! 1. The handler writes `hot_slow_remaining_ticks` + `hot_slow_factor_q8`
//!    under a "longer OR stronger wins" rule.
//! 2. With a slow active, `MoveToward` displacement shrinks by
//!    `factor_q8 / 256` of the base `MOVE_SPEED_MPS`. After the unified
//!    tick-start pass decrements the counter to zero, `SlowExpired` fires
//!    and the factor is zeroed.
//! 3. Engagement-slow (Task 4 — `ENGAGEMENT_SLOW_FACTOR`) composes
//!    multiplicatively with effect-slow: both apply, neither replaces the
//!    other.

use engine::ability::SlowHandler;
use engine::ability::expire::{tick_start, ENGAGEMENT_SLOW_FACTOR};
use engine::cascade::{CascadeHandler, CascadeRegistry};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch, MOVE_SPEED_MPS};
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0 }).unwrap()
}

struct EmitOnce { agent: AgentId, kind: ActionKind }
impl PolicyBackend for EmitOnce {
    fn evaluate(&self, _state: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        out.push(Action { agent: self.agent, kind: self.kind });
    }
}

#[test]
fn slow_writes_duration_and_factor_from_zero() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    SlowHandler.handle(
        &Event::EffectSlowApplied { caster, target, duration_ticks: 5, factor_q8: 51, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_slow_remaining(target),  Some(5));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

#[test]
fn longer_slow_overrides_when_duration_is_greater() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_remaining(target, 3);
    state.set_agent_slow_factor_q8(target, 128);

    // New: longer duration (10 > 3), weaker factor (200 > 128). Longer wins →
    // BOTH replaced.
    SlowHandler.handle(
        &Event::EffectSlowApplied { caster, target, duration_ticks: 10, factor_q8: 200, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_slow_remaining(target),  Some(10));
    assert_eq!(state.agent_slow_factor_q8(target), Some(200));
}

#[test]
fn stronger_slow_overrides_when_factor_is_smaller() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_remaining(target, 10);
    state.set_agent_slow_factor_q8(target, 200);

    // New: shorter duration (3 < 10), stronger factor (51 < 200). Stronger wins → replace.
    SlowHandler.handle(
        &Event::EffectSlowApplied { caster, target, duration_ticks: 3, factor_q8: 51, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_slow_remaining(target),  Some(3));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

#[test]
fn weaker_and_shorter_slow_does_not_override() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_remaining(target, 10);
    state.set_agent_slow_factor_q8(target, 51);

    SlowHandler.handle(
        &Event::EffectSlowApplied { caster, target, duration_ticks: 3, factor_q8: 200, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_slow_remaining(target),  Some(10));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

#[test]
fn move_toward_is_slowed_by_effect_slow_factor() {
    // factor_q8 = 51 → 51/256 ≈ 0.199 multiplier. Over one MoveToward tick,
    // displacement should be (51/256) * MOVE_SPEED_MPS.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(64);
    let cascade = CascadeRegistry::new();  // no builtins — test isolates movement math

    let mover = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    // Second agent so the mask allows MoveToward.
    spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 10.0));

    state.set_agent_slow_remaining(mover, 5);
    state.set_agent_slow_factor_q8(mover, 51);

    let backend = EmitOnce {
        agent: mover,
        kind:  ActionKind::Micro {
            kind:   MicroKind::MoveToward,
            target: MicroTarget::Position(Vec3::new(10.0, 0.0, 0.0)),
        },
    };

    let pos_before = state.agent_pos(mover).unwrap();
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    let pos_after = state.agent_pos(mover).unwrap();
    let displacement = (pos_after - pos_before).length();

    let expected = MOVE_SPEED_MPS * (51.0 / 256.0);
    assert!(
        (displacement - expected).abs() < 1e-4,
        "expected displacement ≈ {expected}, got {displacement}"
    );
}

#[test]
fn slow_decrements_and_expires_with_factor_zeroed() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_slow_remaining(a, 5);
    state.set_agent_slow_factor_q8(a, 51);

    // 5 tick_start decrements → the 5th emits SlowExpired and zeroes factor.
    for i in 0..5u32 {
        tick_start(&mut state, &mut events);
        state.tick += 1;
        let n_expired = events.iter().filter(|e| matches!(e, Event::SlowExpired { .. })).count();
        if i < 4 {
            assert_eq!(n_expired, 0, "SlowExpired must not fire before decrement hits 0 (iter {i})");
            assert_eq!(state.agent_slow_factor_q8(a), Some(51));
        } else {
            assert_eq!(n_expired, 1);
            assert_eq!(state.agent_slow_remaining(a),  Some(0));
            assert_eq!(state.agent_slow_factor_q8(a), Some(0));
        }
    }
}

#[test]
fn engagement_slow_and_effect_slow_compose_multiplicatively() {
    // Cross-task regression: engagement-slow (Task 4, 0.3×) AND effect-slow
    // (Task 14, factor_q8) should both apply when an engaged agent moves
    // away from its engager. The two factors must multiply.
    //
    // Setup: engage Human(0) with Wolf(3.5,0,0) → within 2m no (distance 3.5);
    // instead set up by manually flipping the engagement bits so the test
    // doesn't depend on tick_start's auto-engagement heuristic.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(64);
    let cascade = CascadeRegistry::new();  // isolate movement math

    let mover = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let engager = spawn(&mut state, CreatureType::Wolf, Vec3::new(-1.0, 0.0, 0.0));

    state.set_agent_engaged_with(mover, Some(engager));
    state.set_agent_engaged_with(engager, Some(mover));

    state.set_agent_slow_remaining(mover, 5);
    state.set_agent_slow_factor_q8(mover, 51);

    // MoveToward +X → AWAY from engager → engagement-slow engages.
    let backend = EmitOnce {
        agent: mover,
        kind:  ActionKind::Micro {
            kind:   MicroKind::MoveToward,
            target: MicroTarget::Position(Vec3::new(10.0, 0.0, 0.0)),
        },
    };

    let pos_before = state.agent_pos(mover).unwrap();
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    let pos_after = state.agent_pos(mover).unwrap();
    let displacement = (pos_after - pos_before).length();

    // BOTH slows apply: engagement 0.3 × effect_slow (51/256) × MOVE_SPEED_MPS.
    let expected = MOVE_SPEED_MPS * ENGAGEMENT_SLOW_FACTOR * (51.0 / 256.0);
    assert!(
        (displacement - expected).abs() < 1e-4,
        "engagement × effect slow must compose multiplicatively; \
         expected ≈ {expected}, got {displacement}"
    );
}

#[test]
fn slow_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    state.kill_agent(target);

    SlowHandler.handle(
        &Event::EffectSlowApplied { caster, target, duration_ticks: 5, factor_q8: 51, tick: 0 },
        &mut state,
        &mut events,
    );
    assert_eq!(state.agent_slow_remaining(target),  Some(0));
    assert_eq!(state.agent_slow_factor_q8(target), Some(0));
}
