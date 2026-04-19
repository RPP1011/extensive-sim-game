use engine::state::{AgentSpawn, MovementMode, SimState};
use engine::creature::CreatureType;
use glam::Vec3;

#[test]
fn spawn_and_read_agent() {
    let mut state = SimState::new(100, 42);
    let id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 10.0),
            hp: 100.0,
        })
        .expect("spawn");
    assert_eq!(state.agent_pos(id), Some(Vec3::new(0.0, 0.0, 10.0)));
    assert_eq!(state.agent_hp(id), Some(100.0));
    assert!(state.agent_alive(id));
    assert_eq!(state.agent_creature_type(id), Some(CreatureType::Human));
}

#[test]
fn pool_exhaustion_returns_none() {
    let mut state = SimState::new(2, 42);
    let _a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let _b = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert!(state.spawn_agent(AgentSpawn::default()).is_none());
}

#[test]
fn kill_frees_slot() {
    let mut state = SimState::new(2, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();
    let _b = state.spawn_agent(AgentSpawn::default()).unwrap();
    state.kill_agent(a);
    assert!(!state.agent_alive(a));
    // Slot not reclaimed (kill just flips alive); spawn still fails.
    assert!(state.spawn_agent(AgentSpawn::default()).is_none());
}

#[test]
fn hot_slices_are_independent_vecs() {
    let mut state = SimState::new(5, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    // The point of SoA — can iterate only pos without touching hp.
    let pos_slice: &[Vec3] = state.hot_pos();
    assert_eq!(pos_slice.len(), 5); // full capacity, including dead slots
    assert_eq!(pos_slice[0], Vec3::new(0.0, 0.0, 10.0));
    assert_eq!(pos_slice[1], Vec3::new(1.0, 0.0, 10.0));
    assert_eq!(pos_slice[2], Vec3::new(2.0, 0.0, 10.0));
    // Empty slots stay Vec3::ZERO.
    assert_eq!(pos_slice[3], Vec3::ZERO);
    // Hot slices are separate allocations — mutating hp must not affect pos.
    let hp_addr = state.hot_hp().as_ptr() as usize;
    let pos_addr = state.hot_pos().as_ptr() as usize;
    assert_ne!(hp_addr, pos_addr);
}

// Verify MovementMode is accessible from the public API.
#[test]
fn movement_mode_default_is_walk() {
    let mut state = SimState::new(4, 0);
    let id = state.spawn_agent(AgentSpawn::default()).unwrap();
    assert_eq!(state.agent_movement_mode(id), Some(MovementMode::Walk));
    state.set_agent_movement_mode(id, MovementMode::Fly);
    assert_eq!(state.agent_movement_mode(id), Some(MovementMode::Fly));
}
