//! Engagement SoA field + `state.views.standing` + `CreatureType::is_hostile_to`.
//!
//! Combat Foundation Task 1. Storage-only; bidirectional invariant
//! enforcement lives in Task 3 (`ability::expire::tick_start`).

use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn spawn_defaults_engaged_with_none() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(state.agent_engaged_with(a), None);
}

#[test]
fn set_and_read_engaged_with() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_engaged_with(a, Some(b));
    assert_eq!(state.agent_engaged_with(a), Some(b));
    state.set_agent_engaged_with(a, None);
    assert_eq!(state.agent_engaged_with(a), None);
}

#[test]
fn engagement_slot_persists_through_kill_without_update_engagements() {
    // Zombie invariant: `kill_agent` does not touch engagement slots.
    // `update_engagements` (Task 3) is what clears them. This test pins the
    // separation so a future refactor doesn't silently couple the two paths.
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_engaged_with(a, Some(b));
    state.kill_agent(b);
    // Slot still reports the stale engagement — no auto-clear on kill.
    assert_eq!(state.agent_engaged_with(a), Some(b));
}

#[test]
fn bulk_slice_has_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_engaged_with().len(), 8);
    assert!(state.hot_engaged_with().iter().all(|e| e.is_none()));
}

#[test]
fn hostility_matrix_is_symmetric_and_pinned() {
    use CreatureType::*;
    // Hostile pairs.
    assert!(Wolf.is_hostile_to(Human));
    assert!(Human.is_hostile_to(Wolf));
    assert!(Wolf.is_hostile_to(Deer));
    assert!(Deer.is_hostile_to(Wolf));
    // Dragon is universally hostile.
    assert!(Dragon.is_hostile_to(Human));
    assert!(Human.is_hostile_to(Dragon));
    assert!(Dragon.is_hostile_to(Wolf));
    assert!(Wolf.is_hostile_to(Dragon));
    assert!(Dragon.is_hostile_to(Deer));
    assert!(Deer.is_hostile_to(Dragon));
    // Non-hostile pairs.
    assert!(!Human.is_hostile_to(Human));
    assert!(!Human.is_hostile_to(Deer));
    assert!(!Deer.is_hostile_to(Human));
    assert!(!Wolf.is_hostile_to(Wolf));
    assert!(!Deer.is_hostile_to(Deer));
}

#[test]
fn standing_defaults_zero() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(state.views.standing.get(a, b), 0);
    assert_eq!(state.views.standing.get(b, a), 0);
}

#[test]
fn standing_adjust_is_symmetric_and_clamped() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let b = state.spawn_agent(AgentSpawn::default()).unwrap();

    let tick = state.tick;
    assert_eq!(state.views.standing.adjust(a, b, 50, tick), 50);
    // Reading with swapped order returns same value — pairing is symmetric.
    assert_eq!(state.views.standing.get(b, a), 50);

    // Saturation at upper bound.
    let tick = state.tick;
    assert_eq!(state.views.standing.adjust(a, b, 2000, tick), 1000);
    assert_eq!(state.views.standing.get(a, b), 1000);

    // Saturation at lower bound.
    let tick = state.tick;
    assert_eq!(state.views.standing.adjust(a, b, -5000, tick), -1000);
    assert_eq!(state.views.standing.get(b, a), -1000);
}
