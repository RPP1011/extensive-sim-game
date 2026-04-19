//! Property: `SpatialIndex::query_within_radius` matches a brute-force
//! filter over `state.agents_alive()`. Complements `tests/spatial_index.rs`
//! (3 hand-picked scenarios).
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::spatial::SpatialIndex;
use engine::state::{AgentSpawn, MovementMode, SimState};
use glam::Vec3;
use proptest::prelude::*;
use std::collections::HashSet;

/// An operation against the state-index pair. `Move` and `ChangeMode` force
/// rebuilds and prod the sidecar path; `Kill` exercises dead-slot behavior.
#[derive(Copy, Clone, Debug)]
enum SpatialOp {
    Spawn { pos: Vec3, mode: MovementMode },
    Kill(u32),
    Move { id: u32, pos: Vec3 },
    ChangeMode { id: u32, mode: MovementMode },
}

fn arb_vec3() -> impl Strategy<Value = Vec3> {
    (-20.0f32..20.0, -20.0f32..20.0, -20.0f32..20.0)
        .prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn arb_mode() -> impl Strategy<Value = MovementMode> {
    prop_oneof![
        Just(MovementMode::Walk),
        Just(MovementMode::Fly),
        Just(MovementMode::Swim),
        Just(MovementMode::Climb),
    ]
}

fn arb_op() -> impl Strategy<Value = SpatialOp> {
    prop_oneof![
        (arb_vec3(), arb_mode()).prop_map(|(pos, mode)| SpatialOp::Spawn { pos, mode }),
        (1u32..=16).prop_map(SpatialOp::Kill),
        (1u32..=16, arb_vec3()).prop_map(|(id, pos)| SpatialOp::Move { id, pos }),
        (1u32..=16, arb_mode()).prop_map(|(id, mode)| SpatialOp::ChangeMode { id, mode }),
    ]
}

fn brute_force_within_radius(
    state: &SimState, center: Vec3, radius: f32,
) -> HashSet<u32> {
    state
        .agents_alive()
        .filter_map(|id| {
            state.agent_pos(id).map(|p| (id, p))
        })
        .filter(|(_, p)| p.distance(center) <= radius)
        .map(|(id, _)| id.raw())
        .collect()
}

fn index_within_radius(
    state: &SimState, center: Vec3, radius: f32,
) -> HashSet<u32> {
    let idx = SpatialIndex::build(state);
    idx.query_within_radius(state, center, radius).map(|id| id.raw()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        max_shrink_iters: 3000,
        .. ProptestConfig::default()
    })]

    /// For any random op sequence + query, the spatial index and a brute-force
    /// filter agree on the set of agents within `radius` of `center`.
    #[test]
    fn spatial_index_matches_brute_force(
        ops in proptest::collection::vec(arb_op(), 1..40),
        center in arb_vec3(),
        radius in 0.01f32..30.0,
    ) {
        let mut state = SimState::new(16, 42);
        for op in ops {
            match op {
                SpatialOp::Spawn { pos, mode } => {
                    if let Some(id) = state.spawn_agent(AgentSpawn {
                        creature_type: CreatureType::Human, pos, hp: 100.0,
                    }) {
                        state.set_agent_movement_mode(id, mode);
                    }
                }
                SpatialOp::Kill(raw) => {
                    if let Some(id) = AgentId::new(raw) {
                        if state.agent_alive(id) {
                            state.kill_agent(id);
                        }
                    }
                }
                SpatialOp::Move { id, pos } => {
                    if let Some(aid) = AgentId::new(id) {
                        if state.agent_alive(aid) {
                            state.set_agent_pos(aid, pos);
                        }
                    }
                }
                SpatialOp::ChangeMode { id, mode } => {
                    if let Some(aid) = AgentId::new(id) {
                        if state.agent_alive(aid) {
                            state.set_agent_movement_mode(aid, mode);
                        }
                    }
                }
            }
            let brute = brute_force_within_radius(&state, center, radius);
            let indexed = index_within_radius(&state, center, radius);
            prop_assert_eq!(
                brute.clone(), indexed.clone(),
                "index/brute disagreement: brute={:?} indexed={:?} center={:?} r={}",
                brute, indexed, center, radius
            );
        }
    }
}
