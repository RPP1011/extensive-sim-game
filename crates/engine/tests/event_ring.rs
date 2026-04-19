use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn ring_preserves_order_and_wraps() {
    let mut ring = EventRing::with_cap(4);
    let a = AgentId::new(1).unwrap();
    for i in 0..6 {
        ring.push(Event::AgentMoved {
            agent_id: a, from: Vec3::ZERO, to: Vec3::new(i as f32, 0.0, 0.0), tick: i,
        });
    }
    // Cap=4, pushed 6, oldest 2 dropped. Remaining ticks: [2, 3, 4, 5].
    let ticks: Vec<u32> = ring.iter().map(|e| e.tick()).collect();
    assert_eq!(ticks, vec![2, 3, 4, 5]);
}

#[test]
fn replayable_subset_hashes_stably() {
    let mut ring = EventRing::with_cap(64);
    let a = AgentId::new(1).unwrap();
    ring.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 10,
    });
    ring.push(Event::ChronicleEntry { tick: 11, template_id: 7 }); // non-replayable
    ring.push(Event::AgentDied { agent_id: a, tick: 12 });

    let h1 = ring.replayable_sha256();
    let h2 = ring.replayable_sha256();
    assert_eq!(h1, h2, "same content → same hash");

    // Re-insert the same replayable events separately; hash matches.
    let mut ring2 = EventRing::with_cap(64);
    ring2.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 10,
    });
    ring2.push(Event::AgentDied { agent_id: a, tick: 12 });
    assert_eq!(ring.replayable_sha256(), ring2.replayable_sha256());
}

#[test]
fn agent_attacked_hashes_stably_with_float_damage() {
    let mut ring = EventRing::with_cap(8);
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(2).unwrap();
    ring.push(Event::AgentAttacked { attacker: a, target: b, damage: 12.5, tick: 3 });
    ring.push(Event::AgentAttacked { attacker: a, target: b, damage: 0.0, tick: 4 });
    ring.push(Event::AgentAttacked { attacker: a, target: b, damage: -0.0, tick: 5 });

    let h1 = ring.replayable_sha256();
    let h2 = ring.replayable_sha256();
    assert_eq!(h1, h2);

    // +0.0 and -0.0 must produce distinct hashes when byte-packed via to_bits().
    let mut ring_pos = EventRing::with_cap(4);
    let mut ring_neg = EventRing::with_cap(4);
    ring_pos.push(Event::AgentAttacked { attacker: a, target: b, damage: 0.0, tick: 1 });
    ring_neg.push(Event::AgentAttacked { attacker: a, target: b, damage: -0.0, tick: 1 });
    assert_ne!(ring_pos.replayable_sha256(), ring_neg.replayable_sha256(),
               "+0.0 and -0.0 must hash differently (byte-packed via to_bits)");
}

#[test]
fn chronicle_content_does_not_affect_hash() {
    let a = AgentId::new(1).unwrap();
    let make = |chron_tid: u32| {
        let mut ring = EventRing::with_cap(8);
        ring.push(Event::AgentMoved {
            agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 1,
        });
        ring.push(Event::ChronicleEntry { tick: 2, template_id: chron_tid });
        ring.push(Event::AgentDied { agent_id: a, tick: 3 });
        ring
    };
    assert_eq!(
        make(7).replayable_sha256(),
        make(42).replayable_sha256(),
        "chronicle template_id must not influence replayable-subset hash"
    );
}

#[test]
fn golden_hash_anchors_format() {
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(2).unwrap();
    let mut ring = EventRing::with_cap(8);
    ring.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::new(1.0, 2.0, 3.0), tick: 10,
    });
    ring.push(Event::AgentAttacked { attacker: a, target: b, damage: 12.5, tick: 11 });
    ring.push(Event::AgentDied { agent_id: b, tick: 12 });
    // Pin the digest — changes here require a schema-hash bump.
    let actual = ring.replayable_sha256();
    let hex: String = actual.iter().map(|b| format!("{:02x}", b)).collect();
    assert_eq!(hex, "a6e663e9d88b14d850e0aa121906cc19c5db4de7e8046789962a9ea2a9e20ffd",
               "hash drift detected — see comment");
}
