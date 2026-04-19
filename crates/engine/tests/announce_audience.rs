use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, AnnounceAudience, MacroAction, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct OneAnnounce(AgentId, AnnounceAudience);
impl PolicyBackend for OneAnnounce {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.0,
            kind: ActionKind::Macro(MacroAction::Announce {
                speaker: self.0,
                audience: self.1,
                fact_payload: 0xDEADBEEF_CAFEF00D,
            }),
        });
        for id in state.agents_alive() {
            if id != self.0 {
                out.push(Action::hold(id));
            }
        }
    }
}

#[test]
fn announce_area_emits_recordmemory_for_each_agent_within_radius() {
    let mut state = SimState::new(32, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(2048);
    let cascade = CascadeRegistry::new();

    let center = Vec3::new(0.0, 0.0, 10.0);
    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center,
            hp: 100.0,
        })
        .unwrap();
    // 3 within radius 10:
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(5.0 + i as f32, 0.0, 0.0),
            hp: 100.0,
        });
    }
    // 2 outside radius 10:
    for i in 0..2 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(50.0 + i as f32, 0.0, 0.0),
            hp: 100.0,
        });
    }

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(center, 10.0)),
        &cascade,
    );

    let recipients: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recipients, 3, "three agents within 10m");

    assert!(events.iter().any(|e| matches!(e,
        Event::AnnounceEmitted { speaker: s, audience_tag, .. }
            if *s == speaker && *audience_tag == 1 /* Area */)));
}

#[test]
fn speaker_excluded_from_recipients() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 100.0)),
        &cascade,
    );

    let speaker_recipient = events.iter().any(|e| match e {
        Event::RecordMemory { observer, .. } => *observer == speaker,
        _ => false,
    });
    assert!(!speaker_recipient, "speaker should not receive their own announce");
}

#[test]
fn announce_bounded_by_max_recipients() {
    let mut state = SimState::new(128, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(8192);
    let cascade = CascadeRegistry::new();

    let center = Vec3::new(0.0, 0.0, 10.0);
    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center,
            hp: 100.0,
        })
        .unwrap();
    // 64 agents, all within radius — should cap at MAX_ANNOUNCE_RECIPIENTS=32.
    for i in 0..64 {
        let angle = (i as f32 / 64.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.0),
            hp: 100.0,
        });
    }
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(center, 50.0)),
        &cascade,
    );

    let recipients: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recipients, 32, "bounded by MAX_ANNOUNCE_RECIPIENTS");
}

#[test]
fn announce_anyone_uses_max_announce_radius_around_speaker() {
    let mut state = SimState::new(16, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 10.0),
            hp: 100.0,
        })
        .unwrap();
    // Close agent — within MAX_ANNOUNCE_RADIUS=80:
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(50.0, 0.0, 10.0),
        hp: 100.0,
    });
    // Far agent — beyond 80:
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(200.0, 0.0, 10.0),
        hp: 100.0,
    });

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Anyone),
        &cascade,
    );

    let recipients: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recipients, 1, "only the agent within MAX_ANNOUNCE_RADIUS");
}
