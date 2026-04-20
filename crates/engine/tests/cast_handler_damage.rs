//! Combat Foundation Task 10 — damage applied with shield-first absorption
//! and emits `AgentDied` + `kill_agent` on lethal overflow.
//!
//! Every test dispatches an `Event::EffectDamageApplied` directly and inspects
//! the resulting state mutation + event tail. That's the layer the cascade
//! exposes; wiring a full `AgentCast` through the pipeline is covered by
//! `action_cast_emits_agentcast.rs` (Task 9) and the acceptance tests.
//!
//! The legacy `DamageHandler` unit-struct shim was removed in the 2026-04-19
//! event-taxonomy rename (task 136). Tests now call the compiler-emitted
//! per-event-kind dispatcher directly.

use engine::generated::physics::dispatch_effect_damage_applied;
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

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 30.0, tick: 0 },
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

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 25.0, tick: 0 },
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

    // Implicit tick: emit sites read `state.tick` rather than the
    // incoming event's tick. Advance the world clock so the emitted
    // `AgentDied` carries the expected stamp.
    state.tick = 7;
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 100.0, tick: 7 },
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

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 999.0, tick: 0 },
        &mut state,
        &mut events,
    );

    // No state mutation, no second AgentDied, no AgentAttacked.
    let n_died = events.iter().filter(|e| matches!(e, Event::AgentDied { .. })).count();
    assert_eq!(n_died, 0);
    let n_attacked = events.iter().filter(|e| matches!(e, Event::AgentAttacked { .. })).count();
    assert_eq!(n_attacked, 0);
}

/// Audit fix CRITICAL #3: cast-delivered damage must emit `AgentAttacked`
/// alongside the state mutation, mirroring the melee `MicroKind::Attack`
/// event sequence so cross-backend replay hashes and hostility views stay
/// identical across delivery paths.
#[test]
fn cast_damage_emits_agent_attacked_like_melee() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  100.0);

    state.tick = 3;
    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 25.0, tick: 3 },
        &mut state,
        &mut events,
    );

    let attacked: Vec<(AgentId, AgentId, f32, u32)> = events
        .iter()
        .filter_map(|e| match e {
            Event::AgentAttacked { actor, target, damage, tick } =>
                Some((*actor, *target, *damage, *tick)),
            _ => None,
        })
        .collect();
    assert_eq!(attacked.len(), 1, "cast damage must emit exactly one AgentAttacked");
    assert_eq!(attacked[0], (caster, target, 25.0, 3));
}

/// Cast-delivered kill emits the same [AgentAttacked, AgentDied] sequence
/// the melee `MicroKind::Attack` branch emits (step.rs). Pins parity with
/// `action_attack_kill.rs`.
#[test]
fn cast_lethal_damage_emits_attacked_then_died() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(64);
    let caster = spawn_hp(&mut state, CreatureType::Human, 50.0);
    let target = spawn_hp(&mut state, CreatureType::Wolf,  50.0);

    dispatch_effect_damage_applied(
        &Event::EffectDamageApplied { actor: caster, target, amount: 100.0, tick: 9 },
        &mut state,
        &mut events,
    );

    let seq: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            Event::AgentAttacked { .. } => Some("attacked"),
            Event::AgentDied     { .. } => Some("died"),
            _ => None,
        })
        .collect();
    assert_eq!(seq, vec!["attacked", "died"],
        "cast lethal damage must emit AgentAttacked then AgentDied");
    assert!(!state.agent_alive(target));
}
