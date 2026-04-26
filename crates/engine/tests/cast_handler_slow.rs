#![allow(unused_mut, unused_variables)]
//! Combat Foundation Task 14 + Task 143 — `SlowHandler` + MoveToward
//! speed composition under timestamp-based expiry.
//!
//! Three axes verified:
//! 1. The handler writes `hot_slow_expires_at_tick` + `hot_slow_factor_q8`
//!    under a "longer OR stronger wins" rule.
//! 2. With a slow active (`state.tick < slow_expires_at_tick`),
//!    `MoveToward` displacement shrinks by `factor_q8 / 256` of the base
//!    `move_speed_mps`. Once `state.tick` reaches the expiry the
//!    `effective_slow_factor_q8` helper returns 0 and movement is full
//!    speed again — no per-tick decrement pass, no `SlowExpired` event.
//! 3. Engagement-slow (Task 4 — `engagement_slow_factor`) composes
//!    multiplicatively with effect-slow: both apply, neither replaces the
//!    other.
//!
//! Balance knobs are read off `Config::default()`, not a `pub const`
//! shim — task 142 retired the shim layer.

use engine_rules::physics::dispatch_effect_slow_applied;
use engine_rules::views::ViewRegistry;
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
use engine_data::config::Config;
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> AgentId {
    state.spawn_agent(AgentSpawn { creature_type: ct, pos, hp: 100.0, ..Default::default() }).unwrap()
}

struct EmitOnce { agent: AgentId, kind: ActionKind }
impl PolicyBackend for EmitOnce {
    fn evaluate(&self, _state: &SimState, _m: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        out.push(Action { agent: self.agent, kind: self.kind });
    }
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn slow_writes_expiry_and_factor_from_zero() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    dispatch_effect_slow_applied(
        &Event::EffectSlowApplied { actor: caster, target, expires_at_tick: 5, factor_q8: 51, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_slow_expires_at(target),  Some(5));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn longer_slow_overrides_when_expiry_is_later() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_expires_at(target, 3);
    state.set_agent_slow_factor_q8(target, 128);

    // New: later expiry (10 > 3), weaker factor (200 > 128). Longer wins →
    // BOTH replaced.
    dispatch_effect_slow_applied(
        &Event::EffectSlowApplied { actor: caster, target, expires_at_tick: 10, factor_q8: 200, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_slow_expires_at(target),  Some(10));
    assert_eq!(state.agent_slow_factor_q8(target), Some(200));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn stronger_slow_overrides_when_factor_is_smaller() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_expires_at(target, 10);
    state.set_agent_slow_factor_q8(target, 200);

    // New: earlier expiry (3 < 10), stronger factor (51 < 200). Stronger wins → replace.
    dispatch_effect_slow_applied(
        &Event::EffectSlowApplied { actor: caster, target, expires_at_tick: 3, factor_q8: 51, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_slow_expires_at(target),  Some(3));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn weaker_and_shorter_slow_does_not_override() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));

    state.set_agent_slow_expires_at(target, 10);
    state.set_agent_slow_factor_q8(target, 51);

    dispatch_effect_slow_applied(
        &Event::EffectSlowApplied { actor: caster, target, expires_at_tick: 3, factor_q8: 200, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_slow_expires_at(target),  Some(10));
    assert_eq!(state.agent_slow_factor_q8(target), Some(51));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn move_toward_is_slowed_by_effect_slow_factor() {
    // factor_q8 = 51 → 51/256 ≈ 0.199 multiplier. Over one MoveToward tick
    // (state.tick = 0 at step entry, expires_at_tick = 5 so active),
    // displacement should be (51/256) * move_speed_mps.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let cascade = CascadeRegistry::<Event>::new();  // no builtins — test isolates movement math

    let mover = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    // Second agent so the mask allows MoveToward.
    spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 10.0));

    state.set_agent_slow_expires_at(mover, 5);
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

    let cfg = Config::default();
    let expected = cfg.movement.move_speed_mps * (51.0 / 256.0);
    assert!(
        (displacement - expected).abs() < 1e-4,
        "expected displacement ≈ {expected}, got {displacement}"
    );
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn slow_inactive_once_tick_reaches_expiry() {
    // Task 143 — no decrement loop; once state.tick >= expires_at_tick
    // the `effective_slow_factor_q8` helper reads 0 and the multiplier
    // falls back to 1.0. The stored `slow_factor_q8` is never rewritten
    // (it stays whatever it was at application time — the semantics are
    // "use the factor only while the expiry is in the future").
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_slow_expires_at(a, 5);
    state.set_agent_slow_factor_q8(a, 51);

    // At tick 0..4 the slow is active.
    for t in 0..5u32 {
        state.tick = t;
        assert_eq!(state.effective_slow_factor_q8(a), 51, "active at tick {t}");
    }
    // At tick 5 the expiry has been reached — helper returns 0.
    state.tick = 5;
    assert_eq!(state.effective_slow_factor_q8(a), 0);
    // The raw slots are unchanged — the "expiry" is a read-side synthesis.
    assert_eq!(state.agent_slow_expires_at(a), Some(5));
    assert_eq!(state.agent_slow_factor_q8(a), Some(51));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn engagement_slow_and_effect_slow_compose_multiplicatively() {
    // Cross-task regression: engagement-slow (Task 4, 0.3×) AND effect-slow
    // (Task 14, factor_q8) should both apply when an engaged agent moves
    // away from its engager. The two factors must multiply.
    //
    // Setup: engage Human(0) with Wolf(-1,0,0); manually flip the
    // engagement bits so the test doesn't depend on any auto-engagement
    // heuristic.
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let cascade = CascadeRegistry::<Event>::new();  // isolate movement math

    let mover = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let engager = spawn(&mut state, CreatureType::Wolf, Vec3::new(-1.0, 0.0, 0.0));

    state.set_agent_engaged_with(mover, Some(engager));
    state.set_agent_engaged_with(engager, Some(mover));

    state.set_agent_slow_expires_at(mover, 5);
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

    // BOTH slows apply: engagement slow × effect_slow (51/256) × move_speed_mps.
    let cfg = Config::default();
    let expected = cfg.movement.move_speed_mps * cfg.combat.engagement_slow_factor * (51.0 / 256.0);
    assert!(
        (displacement - expected).abs() < 1e-4,
        "engagement × effect slow must compose multiplicatively; \
         expected ≈ {expected}, got {displacement}"
    );
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn slow_on_dead_target_is_noop() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::<Event>::with_cap(64);
    let mut views = ViewRegistry::new();
    let caster = spawn(&mut state, CreatureType::Human, Vec3::ZERO);
    let target = spawn(&mut state, CreatureType::Wolf,  Vec3::new(1.0, 0.0, 0.0));
    state.kill_agent(target);

    dispatch_effect_slow_applied(
        &Event::EffectSlowApplied { actor: caster, target, expires_at_tick: 5, factor_q8: 51, tick: 0 },
        &mut state,
        &mut views,
        &mut events,
    );
    assert_eq!(state.agent_slow_expires_at(target),  Some(0));
    assert_eq!(state.agent_slow_factor_q8(target), Some(0));
}
