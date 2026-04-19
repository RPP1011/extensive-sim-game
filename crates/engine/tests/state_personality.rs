//! 5-dim personality (state.md §Personality). Task E.
//!
//! Engine uses `altruism` where state.md uses `compassion` per the engine
//! plan instructions — both name the same helping/empathy trait and the
//! port decision is documented in the plan file. Range is [0, 1].

use engine::state::{AgentSpawn, SimState};

#[test]
fn spawn_defaults_personality_to_midpoint() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    // Midpoint 0.5 — "no strong bias" until event-driven drift kicks in.
    assert_eq!(state.agent_risk_tolerance(a), Some(0.5));
    assert_eq!(state.agent_social_drive(a), Some(0.5));
    assert_eq!(state.agent_ambition(a), Some(0.5));
    assert_eq!(state.agent_altruism(a), Some(0.5));
    assert_eq!(state.agent_curiosity(a), Some(0.5));
}

#[test]
fn set_and_read_personality() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_risk_tolerance(a, 0.9);
    state.set_agent_social_drive(a, 0.1);
    state.set_agent_ambition(a, 0.8);
    state.set_agent_altruism(a, 0.2);
    state.set_agent_curiosity(a, 0.7);
    assert_eq!(state.agent_risk_tolerance(a), Some(0.9));
    assert_eq!(state.agent_social_drive(a), Some(0.1));
    assert_eq!(state.agent_ambition(a), Some(0.8));
    assert_eq!(state.agent_altruism(a), Some(0.2));
    assert_eq!(state.agent_curiosity(a), Some(0.7));
}

#[test]
fn personality_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_risk_tolerance().len(), 8);
    assert_eq!(state.hot_social_drive().len(), 8);
    assert_eq!(state.hot_ambition().len(), 8);
    assert_eq!(state.hot_altruism().len(), 8);
    assert_eq!(state.hot_curiosity().len(), 8);
}
