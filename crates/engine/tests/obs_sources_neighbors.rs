use engine_data::entities::CreatureType;
use engine::obs::{FeatureSource, NeighborSource};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn neighbor_dim_is_six_k() {
    assert_eq!(NeighborSource::<3>.dim(), 18);
    assert_eq!(NeighborSource::<5>.dim(), 30);
}

#[test]
fn neighbor_pack_picks_k_nearest_and_zero_fills_rest() {
    // Self at origin. Others at distances 1 (closest), 2, 5, 10.
    let mut state = SimState::new(8, 0);
    let self_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    for (i, dist) in [1.0, 2.0, 5.0, 10.0].iter().enumerate() {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(*dist, 0.0, 0.0),
                hp: 50.0 + (i as f32) * 10.0, // varied hp for identity check
                max_hp: 100.0,
            })
            .unwrap();
    }

    // K=2: only the two nearest (dist 1 and dist 2) should land.
    let mut out = [0.0f32; 12]; // K=2, 6 slots each
    NeighborSource::<2>.pack(&state, self_id, &mut out);

    // First slot: dist=1, rel_x=1, hp_frac = 50/100 = 0.5.
    assert!((out[0] - 1.0).abs() < 1e-6, "rel_x[0] = {}", out[0]);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 0.0);
    assert!((out[3] - 1.0).abs() < 1e-6, "dist[0] = {}", out[3]);
    assert!((out[4] - 0.5).abs() < 1e-6, "hp_frac[0] = {}", out[4]);
    assert_eq!(out[5], 1.0, "present flag[0]");

    // Second slot: dist=2, rel_x=2, hp_frac = 60/100.
    assert!((out[6] - 2.0).abs() < 1e-6);
    assert!((out[9] - 2.0).abs() < 1e-6);
    assert!((out[10] - 0.6).abs() < 1e-6);
    assert_eq!(out[11], 1.0);
}

#[test]
fn neighbor_pack_zero_fills_when_fewer_than_k_agents_exist() {
    let mut state = SimState::new(4, 0);
    let self_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            max_hp: 100.0,
        })
        .unwrap();
    // Only one other exists; K=3 → slots 1 and 2 must be zero-filled with
    // present_flag = 0.

    let mut out = [0.0f32; 18]; // K=3
    NeighborSource::<3>.pack(&state, self_id, &mut out);

    // Slot 0 filled (dist 3), present_flag = 1.
    assert_eq!(out[5], 1.0);
    // Slots 1 and 2 all zero, present_flag = 0.
    for i in 6..18 {
        assert_eq!(out[i], 0.0, "slot idx {} should be zero", i);
    }
}

#[test]
fn neighbor_pack_omits_self() {
    let mut state = SimState::new(2, 0);
    let self_id = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    // Only self exists — NeighborSource should produce all-zero output.
    let mut out = [0.0f32; 12];
    NeighborSource::<2>.pack(&state, self_id, &mut out);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 0.0, "slot {} nonzero when only self exists", i);
    }
}
