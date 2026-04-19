//! Psychological needs (state.md §Needs, Maslow minus hunger): `safety`,
//! `shelter`, `social`, `purpose`, `esteem`. Task D.
//!
//! The engine keeps physiological needs (hunger/thirst/rest) from the MVP
//! and adds the Maslow-5 on top — see state/mod.rs doc comment.

use engine::state::{AgentSpawn, SimState};

#[test]
fn spawn_defaults_psych_needs_to_full() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    // "Fully satisfied" on spawn so Plan 1 need-drift systems have headroom.
    assert_eq!(state.agent_safety(a), Some(1.0));
    assert_eq!(state.agent_shelter(a), Some(1.0));
    assert_eq!(state.agent_social(a), Some(1.0));
    assert_eq!(state.agent_purpose(a), Some(1.0));
    assert_eq!(state.agent_esteem(a), Some(1.0));
}

#[test]
fn set_and_read_psych_needs() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.set_agent_safety(a, 0.2);
    state.set_agent_shelter(a, 0.4);
    state.set_agent_social(a, 0.5);
    state.set_agent_purpose(a, 0.6);
    state.set_agent_esteem(a, 0.7);
    assert_eq!(state.agent_safety(a), Some(0.2));
    assert_eq!(state.agent_shelter(a), Some(0.4));
    assert_eq!(state.agent_social(a), Some(0.5));
    assert_eq!(state.agent_purpose(a), Some(0.6));
    assert_eq!(state.agent_esteem(a), Some(0.7));
}

#[test]
fn psych_need_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_safety().len(), 8);
    assert_eq!(state.hot_shelter().len(), 8);
    assert_eq!(state.hot_social().len(), 8);
    assert_eq!(state.hot_purpose().len(), 8);
    assert_eq!(state.hot_esteem().len(), 8);
}
