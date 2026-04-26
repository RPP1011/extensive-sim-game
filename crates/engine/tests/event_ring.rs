use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn ring_preserves_order_and_wraps() {
    let mut ring = EventRing::<Event>::with_cap(4);
    let a = AgentId::new(1).unwrap();
    for i in 0..6 {
        ring.push(Event::AgentMoved {
            actor: a, from: Vec3::ZERO, location: Vec3::new(i as f32, 0.0, 0.0), tick: i,
        });
    }
    // Cap=4, pushed 6, oldest 2 dropped. Remaining ticks: [2, 3, 4, 5].
    let ticks: Vec<u32> = ring.iter().map(|e| e.tick()).collect();
    assert_eq!(ticks, vec![2, 3, 4, 5]);
}

#[test]
fn replayable_subset_hashes_stably() {
    let mut ring = EventRing::<Event>::with_cap(64);
    let a = AgentId::new(1).unwrap();
    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 10,
    });
    ring.push(Event::ChronicleEntry {
        tick: 11, template_id: 7, agent: a, target: a,
    }); // non-replayable
    ring.push(Event::AgentDied { agent_id: a, tick: 12 });

    let h1 = ring.replayable_sha256();
    let h2 = ring.replayable_sha256();
    assert_eq!(h1, h2, "same content → same hash");

    // Re-insert the same replayable events separately; hash matches.
    let mut ring2 = EventRing::<Event>::with_cap(64);
    ring2.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 10,
    });
    ring2.push(Event::AgentDied { agent_id: a, tick: 12 });
    assert_eq!(ring.replayable_sha256(), ring2.replayable_sha256());
}

#[test]
fn agent_attacked_hashes_stably_with_float_damage() {
    let mut ring = EventRing::<Event>::with_cap(8);
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(2).unwrap();
    ring.push(Event::AgentAttacked { actor: a, target: b, damage: 12.5, tick: 3 });
    ring.push(Event::AgentAttacked { actor: a, target: b, damage: 0.0, tick: 4 });
    ring.push(Event::AgentAttacked { actor: a, target: b, damage: -0.0, tick: 5 });

    let h1 = ring.replayable_sha256();
    let h2 = ring.replayable_sha256();
    assert_eq!(h1, h2);

    // +0.0 and -0.0 must produce distinct hashes when byte-packed via to_bits().
    let mut ring_pos = EventRing::<Event>::with_cap(4);
    let mut ring_neg = EventRing::<Event>::with_cap(4);
    ring_pos.push(Event::AgentAttacked { actor: a, target: b, damage: 0.0, tick: 1 });
    ring_neg.push(Event::AgentAttacked { actor: a, target: b, damage: -0.0, tick: 1 });
    assert_ne!(ring_pos.replayable_sha256(), ring_neg.replayable_sha256(),
               "+0.0 and -0.0 must hash differently (byte-packed via to_bits)");
}

#[test]
fn chronicle_content_does_not_affect_hash() {
    let a = AgentId::new(1).unwrap();
    let make = |chron_tid: u32| {
        let mut ring = EventRing::<Event>::with_cap(8);
        ring.push(Event::AgentMoved {
            actor: a, from: Vec3::ZERO, location: Vec3::X, tick: 1,
        });
        ring.push(Event::ChronicleEntry {
            tick: 2, template_id: chron_tid, agent: a, target: a,
        });
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
fn push_count_is_monotonic_across_eviction() {
    let mut ring = EventRing::<Event>::with_cap(2);
    let a = AgentId::new(1).unwrap();
    assert_eq!(ring.push_count(), 0);
    for i in 0..5 {
        ring.push(Event::AgentMoved {
            actor: a, from: Vec3::ZERO, location: Vec3::new(i as f32, 0.0, 0.0), tick: i,
        });
    }
    // Cap=2, pushed 5. push_count counts total pushes including evicted.
    assert_eq!(ring.push_count(), 5);
    assert_eq!(ring.len(), 2);
    assert_eq!(ring.push_count(), ring.total_pushed(), "push_count is an alias");
}

#[test]
fn iter_since_yields_only_events_after_snapshot() {
    let mut ring = EventRing::<Event>::with_cap(8);
    let a = AgentId::new(1).unwrap();
    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::new(0.0, 0.0, 0.0), tick: 0,
    });
    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::new(1.0, 0.0, 0.0), tick: 1,
    });

    // Simulate "top of a tick" snapshot.
    let events_before = ring.push_count();
    assert_eq!(events_before, 2);

    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::new(2.0, 0.0, 0.0), tick: 2,
    });
    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::new(3.0, 0.0, 0.0), tick: 2,
    });

    let ticks: Vec<u32> = ring.iter_since(events_before).map(|e| e.tick()).collect();
    assert_eq!(ticks, vec![2, 2], "iter_since yields only this-tick events");

    // Snapshot at the current push_count → empty iterator.
    let empty: Vec<_> = ring.iter_since(ring.push_count()).collect();
    assert!(empty.is_empty());

    // Snapshot of 0 → full ring.
    let all: Vec<u32> = ring.iter_since(0).map(|e| e.tick()).collect();
    assert_eq!(all, vec![0, 1, 2, 2]);
}

#[test]
fn iter_since_tolerates_eviction_before_snapshot() {
    // When events before the snapshot have been evicted, iter_since must
    // still return events with index >= start_idx; it cannot resurrect
    // evicted ones but it also must not panic or skip the wrong slice.
    let mut ring = EventRing::<Event>::with_cap(2);
    let a = AgentId::new(1).unwrap();
    ring.push(Event::AgentMoved { actor: a, from: Vec3::ZERO, location: Vec3::ZERO, tick: 0 });
    ring.push(Event::AgentMoved { actor: a, from: Vec3::ZERO, location: Vec3::ZERO, tick: 1 });
    let snap = ring.push_count(); // 2
    ring.push(Event::AgentMoved { actor: a, from: Vec3::ZERO, location: Vec3::ZERO, tick: 2 });
    ring.push(Event::AgentMoved { actor: a, from: Vec3::ZERO, location: Vec3::ZERO, tick: 3 });
    // After cap-2 eviction, entries hold tick=2 and tick=3. Snapshot was at
    // index 2, first retained index is 2, so we see both.
    let ticks: Vec<u32> = ring.iter_since(snap).map(|e| e.tick()).collect();
    assert_eq!(ticks, vec![2, 3]);

    // A stale snapshot from before an eviction ceiling should clamp to the
    // earliest retained event, not underflow.
    let stale_snap = 0usize;
    let ticks2: Vec<u32> = ring.iter_since(stale_snap).map(|e| e.tick()).collect();
    assert_eq!(ticks2, vec![2, 3], "stale snapshot clamps to retained window");
}

#[test]
fn golden_hash_anchors_format() {
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(2).unwrap();
    let mut ring = EventRing::<Event>::with_cap(8);
    ring.push(Event::AgentMoved {
        actor: a, from: Vec3::ZERO, location: Vec3::new(1.0, 2.0, 3.0), tick: 10,
    });
    ring.push(Event::AgentAttacked { actor: a, target: b, damage: 12.5, tick: 11 });
    ring.push(Event::AgentDied { agent_id: b, tick: 12 });
    // Pin the digest — changes here require a schema-hash bump. The event
    // layout is byte-identical to the pre-rename packing (field names are
    // cosmetic, the ring hasher writes raw bytes in source order), so this
    // digest stays the same after task-136's `attacker/caster → actor`
    // rename.
    let actual = ring.replayable_sha256();
    let hex: String = actual.iter().map(|b| format!("{:02x}", b)).collect();
    assert_eq!(hex, "a6e663e9d88b14d850e0aa121906cc19c5db4de7e8046789962a9ea2a9e20ffd",
               "hash drift detected — see comment");
}
