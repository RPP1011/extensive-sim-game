//! Audit fix HIGH #4 — `RecordMemory` events must also land in the observer's
//! `cold_memory` slot, not just in the event ring.
//!
//! `announce_audience.rs` + `announce_overhear.rs` already pin the event-ring
//! side; these tests pin the state-side writer so downstream consumers who
//! read `state.agent_memory(observer)` actually see the broadcast.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, AnnounceAudience, MacroAction, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct OneAnnounce(AgentId, AnnounceAudience);
impl PolicyBackend for OneAnnounce {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.0,
            kind: ActionKind::Macro(MacroAction::Announce {
                speaker: self.0,
                audience: self.1,
                fact_payload: 0xCAFE_D00D_F00D_F00D,
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
fn primary_recipient_cold_memory_contains_the_broadcast() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let listener = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(5.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 10.0)),
        &cascade,
    );

    let mem = state.agent_memory(listener).expect("listener memory slice");
    assert_eq!(mem.len(), 1,
        "expected exactly one memory entry for the primary recipient, got {}", mem.len());
    let ev = mem[0];
    assert_eq!(ev.source, speaker);
    assert_eq!(ev.payload, 0xCAFE_D00D_F00D_F00D);
    // 0.8 primary confidence → round(0.8 * 255) = 204.
    assert_eq!(ev.confidence_q8, 204,
        "confidence_q8 should encode primary confidence 0.8 → 204");
    assert_eq!(ev.tick, 0);
}

#[test]
fn overhear_bystander_cold_memory_has_lower_confidence() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    // Outside primary (Area radius = 2) but inside OVERHEAR_RANGE=30.
    let bystander = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(10.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();

    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 2.0)),
        &cascade,
    );

    let mem = state.agent_memory(bystander).expect("bystander memory slice");
    assert_eq!(mem.len(), 1);
    // 0.6 overhear confidence → round(0.6 * 255) = 153.
    assert_eq!(mem[0].confidence_q8, 153,
        "overhear confidence 0.6 → 153 in q8");
}

#[test]
fn speaker_receives_no_memory_of_their_own_announce() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    let speaker = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    step(
        &mut state,
        &mut scratch,
        &mut events,
        &OneAnnounce(speaker, AnnounceAudience::Area(Vec3::ZERO, 100.0)),
        &cascade,
    );
    let mem = state.agent_memory(speaker).expect("slot");
    assert_eq!(mem.len(), 0, "speaker must not record their own broadcast");
}
