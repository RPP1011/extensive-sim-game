use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, MovementMode, SimState};
use glam::Vec3;

/// `SimState::spatial()` must stay fresh across spawn / kill / set_agent_pos
/// / set_agent_movement_mode WITHOUT any explicit rebuild. The incremental
/// SpatialHash mutators wire into each SoA mutator directly.
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
                ..Default::default()
            })
            .unwrap();
        ids.push(id);
    }
    // All 5 should land in the ±5m query window — without any rebuild call.
    let hits = state.spatial().within_radius(&state, Vec3::ZERO, 10.0);
    assert_eq!(hits.len(), 5, "expected 5 spawned agents, got {}", hits.len());

    // Move one agent 100m away. Query from origin with radius 10 must
    // now return 4, NOT 5 — proving the incremental update fires on
    // `set_agent_pos`.
    state.set_agent_pos(ids[0], Vec3::new(100.0, 0.0, 0.0));
    let hits_after_move = state.spatial().within_radius(&state, Vec3::ZERO, 10.0);
    assert_eq!(hits_after_move.len(), 4, "moved agent must leave the local set");

    // Query near the new position — the moved agent must appear.
    let distant = state.spatial().within_radius(&state, Vec3::new(100.0, 0.0, 0.0), 5.0);
    assert_eq!(distant, vec![ids[0]]);

    // Kill one of the near cluster; index should drop them.
    state.kill_agent(ids[1]);
    let hits_after_kill = state.spatial().within_radius(&state, Vec3::ZERO, 10.0);
    assert_eq!(hits_after_kill.len(), 3, "killed agent must leave the index");
    assert!(!hits_after_kill.contains(&ids[1]));

    // Change movement mode of another — should still be found by a
    // planar query (sidecar path).
    state.set_agent_movement_mode(ids[2], MovementMode::Fly);
    let planar = state.spatial().within_planar(&state, Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(planar.contains(&ids[2]), "flyer must still be reachable via sidecar");
}

fn setup(positions: &[(Vec3, MovementMode)]) -> SimState {
    let mut state = SimState::new(positions.len() as u32 + 1, 42);
    for (p, mode) in positions {
        let id = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: *p,
            hp: 100.0,
            ..Default::default()
        }).unwrap();
        if *mode != MovementMode::Walk {
            state.set_agent_movement_mode(id, *mode);
        }
    }
    state
}

#[test]
fn column_query_returns_walkers_and_sidecar_flyers() {
    let state = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(5.0, 5.0, 10.0), MovementMode::Walk),
        (Vec3::new(100.0, 0.0, 10.0), MovementMode::Walk),  // far walker
        (Vec3::new(1.0, 1.0, 15.0), MovementMode::Fly),     // sidecar, within 3D radius
    ]);
    let query_point = Vec3::new(0.0, 0.0, 10.0);
    let hits = state.spatial().within_radius(&state, query_point, 10.0);
    // Two walkers within 10m (self + (5,5,10) is sqrt(50) ≈ 7.07 < 10) + one flyer (sidecar scan).
    // Flyer at (1,1,15): 3D dist = sqrt(1+1+25) ≈ 5.2m < 10.
    assert_eq!(hits.len(), 3, "two walkers + one flyer, all within 3D radius 10");
}

#[test]
fn column_query_excludes_distant_flyers() {
    let state = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(1.0, 1.0, 50.0), MovementMode::Fly),     // 40m z above → outside 3D radius 10
    ]);
    let query_point = Vec3::new(0.0, 0.0, 10.0);
    let hits = state.spatial().within_radius(&state, query_point, 10.0);
    assert_eq!(hits.len(), 1, "flyer outside 3D radius must NOT be returned");
}

#[test]
fn planar_query_ignores_z_but_respects_xy_distance() {
    let state = setup(&[
        (Vec3::new(0.0, 0.0, 10.0), MovementMode::Walk),
        (Vec3::new(0.0, 0.0, 50.0), MovementMode::Walk), // same xy, far z
    ]);
    let hits = state.spatial().within_planar(&state, Vec3::new(0.5, 0.0, 10.0), 5.0);
    assert_eq!(hits.len(), 2, "both walkers match planar even with z spread");
}

// --- Incremental-mutator coverage (per the rewrite plan) ----------------

#[test]
fn incremental_insert_two_agents_visible_without_rebuild() {
    // Spawn 2 walkers; query immediately. No `rebuild_spatial()` call exists
    // anymore, so this would silently fail under any "rebuild on query" model.
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    let hits = state.spatial().within_radius(&state, Vec3::ZERO, 5.0);
    assert_eq!(hits, vec![a, b], "both spawned agents must be in the index");
}

#[test]
fn incremental_update_crosses_cell_boundary() {
    // Sub-16m moves stay in the same cell — index records the same key.
    // Crossing the cell boundary must move the agent to the new cell.
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
    // Move within cell — same key.
    state.set_agent_pos(a, Vec3::new(15.0, 15.0, 0.0));
    assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
    // Cross to the next cell.
    state.set_agent_pos(a, Vec3::new(17.0, 0.0, 0.0));
    assert_eq!(state.spatial().cell_of_agent(a), Some((1, 0)));
    // And back.
    state.set_agent_pos(a, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
}

#[test]
fn in_cell_move_is_a_noop_for_index_membership() {
    // Sub-cell movement must not rotate bucket membership at all.
    let mut state = SimState::new(8, 42);
    for i in 0..5 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        }).unwrap();
    }
    let pop_before = state.spatial().populated_cell_count();
    let a = AgentId::new(1).unwrap();
    // Slide a around within cell (0,0) — pos.x stays in [0, 16).
    for x in [1.0_f32, 5.0, 10.0, 15.999] {
        state.set_agent_pos(a, Vec3::new(x, 0.0, 0.0));
        assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
    }
    assert_eq!(
        state.spatial().populated_cell_count(),
        pop_before,
        "sub-cell movement must not allocate or evacuate any cell bucket",
    );
}

#[test]
fn incremental_remove_drops_agent_from_queries() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    assert_eq!(state.spatial().within_radius(&state, Vec3::ZERO, 5.0), vec![a, b]);
    state.kill_agent(a);
    assert_eq!(state.spatial().within_radius(&state, Vec3::ZERO, 5.0), vec![b]);
    assert_eq!(state.spatial().cell_of_agent(a), None, "killed slot has no cell");
}

#[test]
fn mode_transition_walk_to_fly_moves_agent_to_sidecar() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    // Walk: in a column.
    assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
    assert!(state.spatial().sidecar_ids().is_empty());
    // Switch to Fly: pulled out of columns into sidecar.
    state.set_agent_movement_mode(a, MovementMode::Fly);
    assert_eq!(state.spatial().cell_of_agent(a), None);
    assert_eq!(state.spatial().sidecar_ids(), &[a]);
    // Planar query still finds the flyer via the sidecar linear scan.
    let hits = state.spatial().within_planar(&state, Vec3::ZERO, 1.0);
    assert!(hits.contains(&a), "sidecar agent reachable via planar query");
    // Switch back to Walk: pulled back into columns.
    state.set_agent_movement_mode(a, MovementMode::Walk);
    assert_eq!(state.spatial().cell_of_agent(a), Some((0, 0)));
    assert!(state.spatial().sidecar_ids().is_empty());
}
