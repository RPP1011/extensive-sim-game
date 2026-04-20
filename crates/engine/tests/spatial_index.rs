use engine::spatial::SpatialIndex;
use engine::state::{AgentSpawn, MovementMode, SimState};
use engine::creature::CreatureType;
use engine::ids::AgentId;
use glam::Vec3;

/// Audit fix CRITICAL #1: `SimState::spatial()` must stay fresh across
/// spawn / kill / set_agent_pos / set_agent_movement_mode. A caller
/// should never observe a stale index on `state.spatial()`.
#[test]
fn state_spatial_is_fresh_after_spawn_move_and_kill() {
    let mut state = SimState::new(16, 42);
    // Spawn 5 agents clustered near origin.
    let mut ids: Vec<AgentId> = Vec::new();
    for i in 0..5 {
        let id = state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32, 0.0, 0.0),
                hp: 100.0,
            })
            .unwrap();
        ids.push(id);
    }
    // All 5 should land in the ±5m query window.
    let hits: Vec<AgentId> = state
        .spatial()
        .query_within_radius(&state, Vec3::ZERO, 10.0)
        .collect();
    assert_eq!(hits.len(), 5, "expected 5 spawned agents, got {}", hits.len());

    // Move one agent 100m away. Query from origin with radius 10 must
    // now return 4, NOT 5 — proving the index tracks moves.
    state.set_agent_pos(ids[0], Vec3::new(100.0, 0.0, 0.0));
    let hits_after_move: Vec<AgentId> = state
        .spatial()
        .query_within_radius(&state, Vec3::ZERO, 10.0)
        .collect();
    assert_eq!(hits_after_move.len(), 4, "moved agent must leave the local set");

    // Query near the new position — the moved agent must appear.
    let distant: Vec<AgentId> = state
        .spatial()
        .query_within_radius(&state, Vec3::new(100.0, 0.0, 0.0), 5.0)
        .collect();
    assert_eq!(distant, vec![ids[0]]);

    // Kill one of the near cluster; index should drop them.
    state.kill_agent(ids[1]);
    let hits_after_kill: Vec<AgentId> = state
        .spatial()
        .query_within_radius(&state, Vec3::ZERO, 10.0)
        .collect();
    assert_eq!(hits_after_kill.len(), 3, "killed agent must leave the index");
    assert!(!hits_after_kill.contains(&ids[1]));

    // Change movement mode of another — should still be found by a
    // planar query (sidecar path).
    state.set_agent_movement_mode(ids[2], MovementMode::Fly);
    let planar: Vec<AgentId> = state
        .spatial()
        .query_within_planar(&state, Vec3::new(2.0, 0.0, 0.0), 1.0)
        .collect();
    assert!(planar.contains(&ids[2]), "flyer must still be reachable via sidecar");
}

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
