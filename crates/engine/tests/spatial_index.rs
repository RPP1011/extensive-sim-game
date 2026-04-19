use engine::spatial::SpatialIndex;
use engine::state::{AgentSpawn, MovementMode, SimState};
use engine::creature::CreatureType;
use engine::ids::AgentId;
use glam::Vec3;

fn setup(positions: &[(Vec3, MovementMode)]) -> (SimState, SpatialIndex) {
    let mut state = SimState::new(positions.len() as u32 + 1, 42);
    for (p, mode) in positions {
        let id = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: *p,
            hp: 100.0,
        }).unwrap();
        if *mode != MovementMode::Walk {
            state.set_agent_movement_mode(id, *mode);
        }
    }
    let index = SpatialIndex::build(&state);
    (state, index)
}

#[test]
fn column_query_returns_walkers_and_sidecar_flyers() {
    let (state, idx) = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(5.0, 5.0, 10.0), MovementMode::Walk),
        (Vec3::new(100.0, 0.0, 10.0), MovementMode::Walk),  // far walker
        (Vec3::new(1.0, 1.0, 15.0), MovementMode::Fly),     // sidecar, within 3D radius
    ]);
    let query_point = Vec3::new(0.0, 0.0, 10.0);
    let mut hits: Vec<AgentId> = idx.query_within_radius(&state, query_point, 10.0).collect();
    hits.sort();
    // Two walkers within 10m (self + (5,5,10) is sqrt(50) ≈ 7.07 < 10) + one flyer (sidecar scan).
    // Flyer at (1,1,15): 3D dist = sqrt(1+1+25) ≈ 5.2m < 10.
    assert_eq!(hits.len(), 3, "two walkers + one flyer, all within 3D radius 10");
}

#[test]
fn column_query_excludes_distant_flyers() {
    let (state, idx) = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(1.0, 1.0, 50.0), MovementMode::Fly),     // 40m z above → outside 3D radius 10
    ]);
    let query_point = Vec3::new(0.0, 0.0, 10.0);
    let hits: Vec<AgentId> = idx.query_within_radius(&state, query_point, 10.0).collect();
    assert_eq!(hits.len(), 1, "flyer outside 3D radius must NOT be returned");
}

#[test]
fn planar_query_ignores_z_but_respects_xy_distance() {
    let (state, idx) = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(0.0, 0.0, 50.0), MovementMode::Walk), // same xy, far z
    ]);
    let hits: Vec<AgentId> =
        idx.query_within_planar(&state, Vec3::new(0.5, 0.0, 10.0), 5.0).collect();
    assert_eq!(hits.len(), 2, "both walkers match planar even with z spread");
}
