//! Audit fix MEDIUM #9 — Announce audience must share a communication channel
//! with the speaker. Human Speech (30m) + Wolf PackSignal (20m) do not
//! overlap, so a Wolf observer never receives `RecordMemory` from a Human
//! speaker — even when the spatial radius would have included them.

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
                fact_payload: 0xABC,
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
fn human_speaker_wolf_listener_do_not_share_channel() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    let _wolf = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 10.0)),
        &cascade,
    );

    let recs: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recs, 0,
        "Human Speech ∩ Wolf PackSignal = ∅ → no RecordMemory for the wolf");
}

#[test]
fn human_speaker_human_listener_share_channel() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    let _human = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 10.0)),
        &cascade,
    );

    let recs: usize = events
        .iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recs, 1,
        "Human speakers share Speech+Testimony with Humans — must deliver one RecordMemory");
}

#[test]
fn human_anyone_radius_is_speech_range_clamped() {
    // Human Anyone-radius falls back to channel-derived range. Speech at
    // vocal 1.0 = 30m, capped at MAX_ANNOUNCE_RADIUS=80m. An observer at
    // 25m IS heard; at 35m is NOT.
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(256);
    let cascade = CascadeRegistry::new();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    // Close (25m): in range of Speech (30m)
    let _close = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(25.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    // Far (35m): outside Speech (30m)
    let _far = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(35.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    // Very far (50m): outside
    let _very = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Anyone),
        &cascade,
    );

    // Anyone is PRIMARY audience (confidence 0.8). Overhear (0.6) scans a
    // separate 30m radius around the speaker; the close observer is also
    // within OVERHEAR_RANGE so it only gets one memory at 0.8 (dedup).
    let primary: usize = events.iter().filter(|e| matches!(e,
        Event::RecordMemory { confidence, .. } if (*confidence - 0.8).abs() < 1e-6)).count();
    assert_eq!(primary, 1,
        "primary audience covers only the 25m observer at channel-derived Speech range");
}
