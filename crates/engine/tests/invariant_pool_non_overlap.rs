use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{Invariant, PoolNonOverlapInvariant};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn pool_non_overlap_holds_for_healthy_spawns() {
    let mut state = SimState::new(4, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let inv = PoolNonOverlapInvariant;
    let events = EventRing::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}

#[test]
fn pool_non_overlap_holds_after_kill_and_respawn() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    state.kill_agent(a);
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::X, hp: 100.0,
    }).unwrap();

    let inv = PoolNonOverlapInvariant;
    let events = EventRing::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}
