use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use engine::ids::AgentId;
use glam::Vec3;

fn all_emit(kind: MicroKind) -> impl PolicyBackend {
    struct All(MicroKind);
    impl PolicyBackend for All {
        fn evaluate(&self, state: &SimState, _: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
            for id in state.agents_alive() {
                out.push(Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind: self.0,
                        target: MicroTarget::None,
                    },
                });
            }
        }
    }
    All(kind)
}

fn make() -> (SimState, SimScratch, EventRing, CascadeRegistry, AgentId) {
    let mut state = SimState::new(4, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    (state, scratch, events, cascade, a)
}

#[test]
fn eating_restores_hunger_and_emits_event_with_clamped_delta() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_hunger(a, 0.2);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Eat), &cascade);

    // 0.2 + 0.25 = 0.45, no clamp
    let h = state.agent_hunger(a).unwrap();
    assert!((h - 0.45).abs() < 1e-6, "hunger now {}", h);

    let delta = events.iter().find_map(|e| match e {
        Event::AgentAte { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentAte emitted");
    assert!((delta - 0.25).abs() < 1e-6);
}

#[test]
fn eating_at_saturated_hunger_clamps_delta_to_zero_restoration() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_hunger(a, 0.9);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Eat), &cascade);

    // 0.9 + 0.25 = 1.15 → clamped to 1.0, actual delta = 0.1
    assert_eq!(state.agent_hunger(a), Some(1.0));
    let delta = events.iter().find_map(|e| match e {
        Event::AgentAte { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentAte emitted with clamped delta");
    assert!((delta - 0.1).abs() < 1e-6, "delta was {}", delta);
}

#[test]
fn drinking_restores_thirst_and_emits_event_with_clamped_delta() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_thirst(a, 0.1);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Drink), &cascade);

    // 0.1 + 0.30 = 0.40, no clamp
    let t = state.agent_thirst(a).unwrap();
    assert!((t - 0.4).abs() < 1e-6, "thirst now {}", t);
    let delta = events.iter().find_map(|e| match e {
        Event::AgentDrank { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentDrank emitted");
    assert!((delta - 0.30).abs() < 1e-6, "drink delta should be 0.30, got {}", delta);
}

#[test]
fn drinking_at_saturated_thirst_clamps_delta() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_thirst(a, 0.85);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Drink), &cascade);

    // 0.85 + 0.30 = 1.15 → clamped to 1.0, actual delta = 0.15
    assert_eq!(state.agent_thirst(a), Some(1.0));
    let delta = events.iter().find_map(|e| match e {
        Event::AgentDrank { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentDrank emitted with clamped delta");
    assert!((delta - 0.15).abs() < 1e-6, "clamped drink delta should be 0.15, got {}", delta);
}

#[test]
fn resting_restores_rest_timer_and_emits_event_with_clamped_delta() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_rest_timer(a, 0.0);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Rest), &cascade);

    // 0.0 + 0.15 = 0.15, no clamp
    let r = state.agent_rest_timer(a).unwrap();
    assert!((r - 0.15).abs() < 1e-6, "rest now {}", r);
    let delta = events.iter().find_map(|e| match e {
        Event::AgentRested { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentRested emitted");
    assert!((delta - 0.15).abs() < 1e-6, "rest delta should be 0.15, got {}", delta);
}

#[test]
fn resting_at_saturated_rest_clamps_delta() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    state.set_agent_rest_timer(a, 0.9);

    step(&mut state, &mut scratch, &mut events, &all_emit(MicroKind::Rest), &cascade);

    // 0.9 + 0.15 = 1.05 → clamped to 1.0, actual delta = 0.10
    assert_eq!(state.agent_rest_timer(a), Some(1.0));
    let delta = events.iter().find_map(|e| match e {
        Event::AgentRested { agent_id, delta, .. } if *agent_id == a => Some(*delta),
        _ => None,
    }).expect("AgentRested emitted with clamped delta");
    assert!((delta - 0.10).abs() < 1e-6, "clamped rest delta should be 0.10, got {}", delta);
}
