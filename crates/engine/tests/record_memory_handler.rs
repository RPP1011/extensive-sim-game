//! Audit fix HIGH #4 — `RecordMemory` events must also land in the
//! observer's memory slot, not just in the event ring.
//!
//! `announce_audience.rs` + `announce_overhear.rs` already pin the
//! event-ring side; these tests pin the state-side writer so
//! downstream consumers who read `state.views.memory.entries(observer)`
//! actually see the broadcast.
//!
//! Subsystem 2 Phase 4 migrated the storage from
//! `state.cold_memory[slot]` to the generated `state.views.memory`
//! view (`@per_entity_ring(K=64)`). The view's minimal v1 projection
//! stores `{source, value = 1.0, anchor_tick = tick}` — `payload` /
//! `confidence_q8` are dropped until the entry shape widens. These
//! tests assert the reachable fields (source + cursor + tick).

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
fn primary_recipient_view_memory_contains_the_broadcast() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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

    let cursor = state.views.memory.cursor(listener);
    assert_eq!(
        cursor, 1,
        "expected exactly one push into listener's ring, cursor={cursor}"
    );
    let row = state.views.memory.entries(listener).expect("listener row");
    assert_eq!(row[0].source, speaker.raw());
    assert_eq!(row[0].anchor_tick, 0);
}

#[test]
fn overhear_bystander_view_memory_records_source() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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

    let cursor = state.views.memory.cursor(bystander);
    assert_eq!(cursor, 1, "bystander got one overhear push");
    let row = state.views.memory.entries(bystander).expect("bystander row");
    assert_eq!(row[0].source, speaker.raw());
}

#[test]
fn speaker_receives_no_memory_of_their_own_announce() {
    let mut state = SimState::new(8, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1024);
    let cascade = CascadeRegistry::<Event>::with_engine_builtins();

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
    assert_eq!(
        state.views.memory.cursor(speaker),
        0,
        "speaker must not record their own broadcast"
    );
}
