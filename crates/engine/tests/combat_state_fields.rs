//! Combat-timing SoA fields added by Combat Foundation Task 2:
//! `hot_stun_remaining_ticks`, `hot_slow_remaining_ticks`,
//! `hot_slow_factor_q8`, `hot_cooldown_next_ready_tick`.
//!
//! Storage only; decrement + expiry live in Task 3 (`ability::expire::tick_start`).

use engine::state::{AgentSpawn, SimState};

#[test]
fn spawn_defaults_combat_timing_fields_to_zero() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(state.agent_stun_remaining(a), Some(0));
    assert_eq!(state.agent_slow_remaining(a), Some(0));
    assert_eq!(state.agent_slow_factor_q8(a), Some(0));
    assert_eq!(state.agent_cooldown_next_ready(a), Some(0));
}

#[test]
fn set_and_read_combat_timing_fields() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();

    state.set_agent_stun_remaining(a, 10);
    state.set_agent_slow_remaining(a, 5);
    state.set_agent_slow_factor_q8(a, 51); // ~0.2x speed
    state.set_agent_cooldown_next_ready(a, 100);

    assert_eq!(state.agent_stun_remaining(a), Some(10));
    assert_eq!(state.agent_slow_remaining(a), Some(5));
    assert_eq!(state.agent_slow_factor_q8(a), Some(51));
    assert_eq!(state.agent_cooldown_next_ready(a), Some(100));
}

#[test]
fn bulk_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_stun_remaining_ticks().len(), 8);
    assert_eq!(state.hot_slow_remaining_ticks().len(), 8);
    assert_eq!(state.hot_slow_factor_q8().len(), 8);
    assert_eq!(state.hot_cooldown_next_ready_tick().len(), 8);
    // Defaults: all zero.
    assert!(state.hot_stun_remaining_ticks().iter().all(|v| *v == 0));
    assert!(state.hot_slow_remaining_ticks().iter().all(|v| *v == 0));
    assert!(state.hot_slow_factor_q8().iter().all(|v| *v == 0));
    assert!(state.hot_cooldown_next_ready_tick().iter().all(|v| *v == 0));
}
