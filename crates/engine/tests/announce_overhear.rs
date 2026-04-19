use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, AnnounceAudience, MacroAction, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct AnnouncerSmall(AgentId);
impl PolicyBackend for AnnouncerSmall {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        // Announce to a tiny Area of radius 2 (essentially no primary audience,
        // forcing the overhear scan to do the work).
        out.push(Action {
            agent: self.0,
            kind: ActionKind::Macro(MacroAction::Announce {
                speaker: self.0,
                audience: AnnounceAudience::Area(state.agent_pos(self.0).unwrap(), 2.0),
                fact_payload: 0x1234,
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
fn bystander_within_overhear_range_gets_recordmemory_at_0_6_confidence() {
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
    let bystander = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(15.0, 0.0, 0.0), // within OVERHEAR_RANGE=30, outside primary=2
            hp: 100.0,
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &AnnouncerSmall(speaker),
        &cascade,
    );

    // Exactly one RecordMemory: the bystander, at 0.6 confidence.
    let recs: Vec<(AgentId, f32)> = events
        .iter()
        .filter_map(|e| match e {
            Event::RecordMemory {
                observer,
                confidence,
                ..
            } => Some((*observer, *confidence)),
            _ => None,
        })
        .collect();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].0, bystander);
    assert!((recs[0].1 - 0.6).abs() < 1e-6, "got {}", recs[0].1);
}

#[test]
fn agent_beyond_overhear_range_gets_nothing() {
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
    let _distant = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(100.0, 0.0, 0.0), // way beyond 30
            hp: 100.0,
        })
        .unwrap();
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &AnnouncerSmall(speaker),
        &cascade,
    );
    let count = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(count, 0);
}

#[test]
fn primary_recipient_not_also_added_as_overhear_bystander() {
    // Single agent both within primary Area (large radius) AND within OVERHEAR_RANGE.
    // They should get ONE RecordMemory at 0.8 (primary), not a second at 0.6.
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
    let primary = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    struct BigArea(AgentId);
    impl PolicyBackend for BigArea {
        fn evaluate(&self, _: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
            out.push(Action {
                agent: self.0,
                kind: ActionKind::Macro(MacroAction::Announce {
                    speaker: self.0,
                    audience: AnnounceAudience::Area(Vec3::ZERO, 10.0),
                    fact_payload: 0,
                }),
            });
        }
    }

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &BigArea(speaker),
        &cascade,
    );

    let recs_for_primary: Vec<f32> = events
        .iter()
        .filter_map(|e| match e {
            Event::RecordMemory {
                observer,
                confidence,
                ..
            } if *observer == primary => Some(*confidence),
            _ => None,
        })
        .collect();

    assert_eq!(recs_for_primary.len(), 1);
    assert!(
        (recs_for_primary[0] - 0.8).abs() < 1e-6,
        "primary should get exactly one at 0.8, got {:?}",
        recs_for_primary
    );
}
