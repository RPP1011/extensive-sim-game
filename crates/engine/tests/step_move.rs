use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{SimState, AgentSpawn};
use engine::creature::CreatureType;
use engine::step::step;
use glam::Vec3;

#[test]
fn agent_moves_toward_nearest_other() {
    let mut state = SimState::new(4, 42);
    let mut events = EventRing::with_cap(100);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(50.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    step(&mut state, &mut events, &UtilityBackend);
    let pos_a = state.agent_pos(a).unwrap();
    assert!(pos_a.x > 0.0, "a moved toward b (expected +x, got {:?})", pos_a);
    assert!(events.iter().any(|e| matches!(e, Event::AgentMoved { agent_id, .. } if *agent_id == a)));
}

#[test]
fn no_move_when_alone() {
    let mut state = SimState::new(2, 42);
    let mut events = EventRing::with_cap(100);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    step(&mut state, &mut events, &UtilityBackend);
    let pos_a = state.agent_pos(a).unwrap();
    assert_eq!(pos_a, Vec3::new(0.0, 0.0, 10.0), "alone agent doesn't move");
    assert!(events.iter().all(|e| !matches!(e, Event::AgentMoved { .. })));
}

#[test]
fn colocated_agents_do_not_emit_agentmoved() {
    let mut state = SimState::new(3, 42);
    let mut events = EventRing::with_cap(100);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(5.0, 5.0, 10.0), hp: 100.0,
    }).unwrap();
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(5.0, 5.0, 10.0), hp: 100.0,  // identical pos
    }).unwrap();
    step(&mut state, &mut events, &UtilityBackend);
    assert_eq!(state.agent_pos(a).unwrap(), Vec3::new(5.0, 5.0, 10.0), "position unchanged");
    let move_events: Vec<_> = events.iter().filter(|e| matches!(e, Event::AgentMoved { .. })).collect();
    assert!(move_events.is_empty(), "no AgentMoved when zero-delta direction");
}
