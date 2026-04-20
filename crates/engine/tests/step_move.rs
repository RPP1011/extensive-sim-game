use engine::cascade::CascadeRegistry;
use engine::event::{Event, EventRing};
use engine::policy::UtilityBackend;
use engine::state::{SimState, AgentSpawn};
use engine::creature::CreatureType;
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn agent_moves_toward_nearest_other() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100);
    let cascade = CascadeRegistry::new();
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(50.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let pos_a = state.agent_pos(a).unwrap();
    // a moves from (0,0,10) toward b at (50,0,10) by exactly MOVE_SPEED_MPS=1.0.
    // Direction is pure +x (y and z are equal), so pos_a must be (1.0, 0.0, 10.0).
    // Pins MOVE_SPEED_MPS — an impl with speed 0.01 or 100 would fail here.
    assert!((pos_a.x - 1.0).abs() < 1e-5,
        "x should be exactly 1.0m after one tick at MOVE_SPEED_MPS=1.0, got {}", pos_a.x);
    assert!((pos_a.y - 0.0).abs() < 1e-6, "y drift {}", pos_a.y);
    assert!((pos_a.z - 10.0).abs() < 1e-6, "z drift {}", pos_a.z);
    assert!(events.iter().any(|e| matches!(e, Event::AgentMoved { actor, .. } if *actor == a)));
}

#[test]
fn no_move_when_alone() {
    let mut state = SimState::new(2, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100);
    let cascade = CascadeRegistry::new();
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();
    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let pos_a = state.agent_pos(a).unwrap();
    assert_eq!(pos_a, Vec3::new(0.0, 0.0, 10.0), "alone agent doesn't move");
    assert!(events.iter().all(|e| !matches!(e, Event::AgentMoved { .. })));
}

#[test]
fn colocated_agents_do_not_emit_agentmoved() {
    let mut state = SimState::new(3, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100);
    let cascade = CascadeRegistry::new();
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(5.0, 5.0, 10.0), hp: 100.0,
    }).unwrap();
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(5.0, 5.0, 10.0), hp: 100.0,  // identical pos
    }).unwrap();
    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    assert_eq!(state.agent_pos(a).unwrap(), Vec3::new(5.0, 5.0, 10.0), "position unchanged");
    let move_events: Vec<_> = events.iter().filter(|e| matches!(e, Event::AgentMoved { .. })).collect();
    assert!(move_events.is_empty(), "no AgentMoved when zero-delta direction");
}
