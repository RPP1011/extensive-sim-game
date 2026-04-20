use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn push_assigns_sequential_event_ids_within_a_tick() {
    let mut ring = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 0,
    });
    let id1 = ring.push(Event::AgentMoved {
        actor: a, from: Vec3::X, location: Vec3::Y, tick: 0,
    });
    assert_eq!(id0.tick, 0);
    assert_eq!(id0.seq, 0);
    assert_eq!(id1.tick, 0);
    assert_eq!(id1.seq, 1);
}

#[test]
fn seq_resets_when_tick_advances() {
    let mut ring = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let _ = ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 0,
    });
    let id1 = ring.push(Event::AgentMoved {
        actor: a, from: Vec3::X, location: Vec3::Y, tick: 1,
    });
    assert_eq!(id1.tick, 1);
    assert_eq!(id1.seq, 0, "seq resets per tick");
}

#[test]
fn push_caused_by_stores_parent_in_sidecar() {
    let mut ring = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = ring.push(Event::AgentAttacked {
        actor: a, target: a, damage: 10.0, tick: 0,
    });
    let id1 = ring.push_caused_by(
        Event::AgentDied { agent_id: a, tick: 0 },
        id0,
    );
    assert_eq!(ring.cause_of(id1), Some(id0));
    assert_eq!(ring.cause_of(id0), None);
}

#[test]
fn cause_field_does_not_affect_replayable_hash() {
    let mut r1 = EventRing::with_cap(16);
    let mut r2 = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = r1.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 0,
    });
    r1.push_caused_by(Event::AgentDied { agent_id: a, tick: 0 }, id0);
    r2.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 0,
    });
    r2.push(Event::AgentDied { agent_id: a, tick: 0 });
    assert_eq!(r1.replayable_sha256(), r2.replayable_sha256(),
        "cause field must not alter the hash");
}
