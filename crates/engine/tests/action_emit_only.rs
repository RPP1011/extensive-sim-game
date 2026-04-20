use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::PolicyBackend;
use engine::policy::{Action, ActionKind, MicroTarget};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn make() -> (SimState, SimScratch, EventRing, CascadeRegistry, AgentId) {
    let mut state = SimState::new(8, 42);
    let scratch = SimScratch::new(state.agent_cap() as usize);
    let events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
        })
        .unwrap();
    (state, scratch, events, cascade, a)
}

struct EmitOnce(AgentId, ActionKind);
impl PolicyBackend for EmitOnce {
    fn evaluate(&self, _state: &SimState, _m: &MaskBuffer, out: &mut Vec<Action>) {
        out.push(Action {
            agent: self.0,
            kind: self.1,
        });
    }
}

#[test]
fn cast_emits_agentcast_event_without_state_change() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let hp_before = state.agent_hp(a).unwrap();
    // Combat Foundation Task 9: Cast now targets a specific agent via an
    // `AbilityId`. Without a CastHandler registered (empty cascade here),
    // the AgentCast event is emitted but no effects fire.
    let ability = engine::ability::AbilityId::new(3).unwrap();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Cast,
            target: MicroTarget::Ability { id: ability, target: a },
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);

    assert_eq!(state.agent_hp(a), Some(hp_before));
    let got = events.iter().any(|e| matches!(e,
        Event::AgentCast { caster, ability: ab, target, .. }
            if *caster == a && ab.raw() == 3 && *target == a));
    assert!(got);
}

#[test]
fn useitem_emits_agentuseditem_event() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::UseItem,
            target: MicroTarget::ItemSlot(2),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentUsedItem { agent_id, item_slot, .. }
            if *agent_id == a && *item_slot == 2)));
}

#[test]
fn harvest_emits_agentharvested_event() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Harvest,
            target: MicroTarget::Opaque(0xABCD),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentHarvested { resource, .. } if *resource == 0xABCD)));
}

#[test]
fn placetile_emits_event_with_position() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::PlaceTile,
            target: MicroTarget::Position(Vec3::new(3.0, 4.0, 5.0)),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| match e {
        Event::AgentPlacedTile { where_pos, .. } =>
            where_pos.x == 3.0 && where_pos.y == 4.0 && where_pos.z == 5.0,
        _ => false,
    }));
}

#[test]
fn placevoxel_emits_event_with_position() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::PlaceVoxel,
            target: MicroTarget::Position(Vec3::new(1.0, 2.0, 3.0)),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::AgentPlacedVoxel { .. })));
}

#[test]
fn harvestvoxel_emits_event_with_position() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::HarvestVoxel,
            target: MicroTarget::Position(Vec3::new(6.0, 7.0, 8.0)),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::AgentHarvestedVoxel { .. })));
}

#[test]
fn converse_emits_partner() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Converse,
            target: MicroTarget::Agent(b),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentConversed { partner, .. } if *partner == b)));
}

#[test]
fn sharestory_emits_topic() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::ShareStory,
            target: MicroTarget::Opaque(77),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentSharedStory { topic, .. } if *topic == 77)));
}

#[test]
fn communicate_emits_speaker_recipient_factref() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    // Communicate needs the fact_ref in the Opaque slot (or we could extend MicroTarget,
    // but Opaque is the extension hatch for now).
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Communicate,
            target: MicroTarget::Agent(b),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentCommunicated { speaker, recipient, .. }
            if *speaker == a && *recipient == b)));
}

#[test]
fn ask_agent_emits_informationrequested() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let b = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
        })
        .unwrap();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Ask,
            target: MicroTarget::Agent(b),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::InformationRequested { asker, target, .. }
            if *asker == a && *target == b)));
}

#[test]
fn remember_emits_event_with_subject() {
    let (mut state, mut scratch, mut events, cascade, a) = make();
    let backend = EmitOnce(
        a,
        ActionKind::Micro {
            kind: MicroKind::Remember,
            target: MicroTarget::Opaque(42),
        },
    );
    step(&mut state, &mut scratch, &mut events, &backend, &cascade);
    assert!(events.iter().any(|e| matches!(e,
        Event::AgentRemembered { subject, .. } if *subject == 42)));
}
