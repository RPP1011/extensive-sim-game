use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn spawn_initializes_needs_to_full() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO,
        hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert_eq!(state.agent_hunger(a), Some(1.0));
    assert_eq!(state.agent_thirst(a), Some(1.0));
    assert_eq!(state.agent_rest_timer(a), Some(1.0));
}

#[test]
fn set_and_read_needs() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO,
        hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.set_agent_hunger(a, 0.3);
    state.set_agent_thirst(a, 0.5);
    state.set_agent_rest_timer(a, 0.1);
    assert_eq!(state.agent_hunger(a), Some(0.3));
    assert_eq!(state.agent_thirst(a), Some(0.5));
    assert_eq!(state.agent_rest_timer(a), Some(0.1));
}

#[test]
fn hot_needs_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_hunger().len(), 8);
    assert_eq!(state.hot_thirst().len(), 8);
    assert_eq!(state.hot_rest_timer().len(), 8);
}
