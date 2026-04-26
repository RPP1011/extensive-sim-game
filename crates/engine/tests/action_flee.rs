use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
use engine::ids::AgentId;
use glam::Vec3;

struct FleeFromFirst;
impl PolicyBackend for FleeFromFirst {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        let threat = AgentId::new(1).unwrap();
        for id in state.agents_alive() {
            if id == threat { out.push(Action::hold(id)); continue; }
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind: MicroKind::Flee,
                    target: MicroTarget::Agent(threat),
                },
            });
        }
    }
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn flee_moves_in_opposite_direction_from_threat() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    // Same z-plane so the away-vector is purely +x.
    let _threat = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 10.0), hp: 100.0,
        ..Default::default()
    }).unwrap();
    let prey = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(1.0, 0.0, 10.0), hp: 100.0,
        ..Default::default()
    }).unwrap();

    let pos_before = state.agent_pos(prey).unwrap();
    step(&mut state, &mut scratch, &mut events, &FleeFromFirst, &cascade);
    let pos_after = state.agent_pos(prey).unwrap();

    // prey was east of threat on same z; flee → prey moves further east by
    // exactly one MOVE_SPEED_MPS (1.0m) along +x. Pins speed magnitude —
    // an impl using 0.001 or 100 would fail here.
    let dx = pos_after.x - pos_before.x;
    assert!((dx - 1.0).abs() < 1e-5,
        "flee should advance x by exactly MOVE_SPEED_MPS=1.0, got dx={}", dx);
    assert!((pos_after.y - pos_before.y).abs() < 1e-6, "y drift {}", pos_after.y);
    assert!((pos_after.z - pos_before.z).abs() < 1e-6, "z drift {}", pos_after.z);

    let found = events.iter().any(|e| match e {
        Event::AgentFled { agent_id, .. } => *agent_id == prey,
        _ => false,
    });
    assert!(found, "AgentFled event emitted for prey");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn flee_with_threat_at_same_position_no_move_no_event() {
    // Degenerate case: co-located; direction is zero; agent should stay put
    // and NOT emit AgentFled (matches MoveToward zero-delta behavior).
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::new();

    let _threat = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();
    let prey = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
        ..Default::default()
    }).unwrap();

    let pos_before = state.agent_pos(prey).unwrap();
    step(&mut state, &mut scratch, &mut events, &FleeFromFirst, &cascade);
    let pos_after = state.agent_pos(prey).unwrap();

    assert_eq!(pos_before, pos_after);
    let any_fled = events.iter().any(|e| matches!(e, Event::AgentFled { .. }));
    assert!(!any_fled, "zero-delta flee should not emit event");
}
