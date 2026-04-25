use engine::creature::CreatureType;
use engine::obs::{FeatureSource, PositionSource};
use engine::state::{AgentSpawn, MovementMode, SimState};
use glam::Vec3;

#[test]
fn position_dim_is_seven() {
    assert_eq!(PositionSource.dim(), 7);
}

#[test]
fn position_pack_writes_xyz_and_one_hot_walk() {
    let mut state = SimState::new(2, 0);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.5, -2.5, 3.75),
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    // Default movement mode is Walk.
    let mut out = [0.0f32; 7];
    PositionSource.pack(&state, a, &mut out);
    assert_eq!(out[0], 1.5);
    assert_eq!(out[1], -2.5);
    assert_eq!(out[2], 3.75);
    assert_eq!(&out[3..], &[1.0, 0.0, 0.0, 0.0]); // walk one-hot
}

#[test]
fn position_pack_reflects_movement_mode_changes() {
    let mut state = SimState::new(2, 0);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();

    state.set_agent_movement_mode(a, MovementMode::Fly);
    let mut out = [0.0f32; 7];
    PositionSource.pack(&state, a, &mut out);
    assert_eq!(&out[3..], &[0.0, 1.0, 0.0, 0.0]); // fly one-hot

    state.set_agent_movement_mode(a, MovementMode::Swim);
    PositionSource.pack(&state, a, &mut out);
    assert_eq!(&out[3..], &[0.0, 0.0, 1.0, 0.0]);

    state.set_agent_movement_mode(a, MovementMode::Climb);
    PositionSource.pack(&state, a, &mut out);
    assert_eq!(&out[3..], &[0.0, 0.0, 0.0, 1.0]);

    // Fall zeros out the one-hot (transient state, not represented).
    state.set_agent_movement_mode(a, MovementMode::Fall);
    PositionSource.pack(&state, a, &mut out);
    assert_eq!(&out[3..], &[0.0, 0.0, 0.0, 0.0]);
}
