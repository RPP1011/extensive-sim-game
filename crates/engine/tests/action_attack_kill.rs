use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::ids::AgentId;
use glam::Vec3;

struct AttackFixed(AgentId);
impl PolicyBackend for AttackFixed {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.0 { out.push(Action::hold(id)); continue; }
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind: MicroKind::Attack,
                    target: MicroTarget::Agent(self.0),
                },
            });
        }
    }
}

#[test]
fn attack_reduces_hp_within_range() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 1m away — within 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(90.0));

    assert!(events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn attack_beyond_range_is_a_no_op() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        // 10m away — beyond 2m ATTACK_RANGE
        creature_type: CreatureType::Human, pos: Vec3::new(10.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(100.0));
    assert!(!events.iter().any(|e| matches!(e, Event::AgentAttacked { .. })));
}

#[test]
fn hp_hits_zero_kills_agent_and_emits_died_event() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 10.0,  // one-shot
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(0.0));
    assert!(!state.agent_alive(victim), "hp=0 agent is dead");
    assert!(events.iter().any(|e| matches!(e, Event::AgentDied { agent_id, .. } if *agent_id == victim)));
}

#[test]
fn attack_dead_target_is_no_op() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 5.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    // First tick kills the victim.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    let events_after_kill = events.len();
    // Second tick: attacker tries again on dead victim — should produce nothing new.
    step(&mut state, &mut scratch, &mut events, &AttackFixed(victim), &cascade);
    assert_eq!(events.len(), events_after_kill, "no new events against dead target");
}
