//! Spatial-extras fields (state.md §Physical State): `level`, `grid_id`,
//! `local_pos`, `move_target`, `move_speed`, `move_speed_mult`.
//!
//! Task A of the 2026-04-19 engine-plan-state-port — see
//! `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md`. Fields are
//! storage-only; no movement logic reads them yet.

use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn spawn_defaults_spatial_extras() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    // Power tier starts at 1 so downstream scaling is non-zero by default.
    assert_eq!(state.agent_level(a), Some(1));
    // Optional location fields default to None — no settlement / room context.
    assert_eq!(state.agent_grid_id(a), None);
    assert_eq!(state.agent_local_pos(a), None);
    assert_eq!(state.agent_move_target(a), None);
    // Movement defaults: 1.0 unit/tick base, 1.0 multiplier (no slow/haste).
    assert_eq!(state.agent_move_speed(a), Some(1.0));
    assert_eq!(state.agent_move_speed_mult(a), Some(1.0));
}

#[test]
fn set_and_read_spatial_extras() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn::default()).unwrap();

    state.set_agent_level(a, 7);
    state.set_agent_grid_id(a, Some(42));
    state.set_agent_local_pos(a, Some(Vec3::new(1.0, 2.0, 3.0)));
    state.set_agent_move_target(a, Some(Vec3::new(-5.0, 0.0, 0.0)));
    state.set_agent_move_speed(a, 2.5);
    state.set_agent_move_speed_mult(a, 0.3);

    assert_eq!(state.agent_level(a), Some(7));
    assert_eq!(state.agent_grid_id(a), Some(42));
    assert_eq!(state.agent_local_pos(a), Some(Vec3::new(1.0, 2.0, 3.0)));
    assert_eq!(state.agent_move_target(a), Some(Vec3::new(-5.0, 0.0, 0.0)));
    assert_eq!(state.agent_move_speed(a), Some(2.5));
    assert_eq!(state.agent_move_speed_mult(a), Some(0.3));
}

#[test]
fn bulk_slices_have_cap_length() {
    let state = SimState::new(8, 42);
    assert_eq!(state.hot_level().len(), 8);
    assert_eq!(state.hot_move_speed().len(), 8);
    assert_eq!(state.hot_move_speed_mult().len(), 8);
    assert_eq!(state.cold_grid_id().len(), 8);
    assert_eq!(state.cold_local_pos().len(), 8);
    assert_eq!(state.cold_move_target().len(), 8);
}
