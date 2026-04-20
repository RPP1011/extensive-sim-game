//! Smoke test for the compiler-emitted `Event` enum. Verifies that the
//! milestone-2 integration step (cutover of `engine::event::Event` to the
//! DSL-emitted `engine_rules::events::Event`) preserves the hand-written
//! struct-variant shape every engine call site relies on.

use engine_rules::events::Event;
use engine_rules::ids::AgentId;

#[test]
fn can_construct_agent_died_variant() {
    let agent_id = AgentId::new(1).expect("nonzero id");
    let event = Event::AgentDied { agent_id, tick: 42 };

    // Variant exists with the expected struct-style field shape.
    match event {
        Event::AgentDied { agent_id: got, tick } => {
            assert_eq!(got, agent_id);
            assert_eq!(tick, 42);
        }
        _ => panic!("expected AgentDied variant"),
    }
}

#[test]
fn event_tick_method_returns_u32() {
    let event = Event::AgentDied {
        agent_id: AgentId::new(7).unwrap(),
        tick: 99,
    };
    // The helper collapses every variant's `tick` field into a single
    // u32 so downstream engine code (EventRing) can bucket events by tick.
    let t: u32 = event.tick();
    assert_eq!(t, 99);
}

#[test]
fn replayable_annotation_carries_through() {
    // AgentDied is annotated @replayable.
    let replayable = Event::AgentDied {
        agent_id: AgentId::new(1).unwrap(),
        tick: 0,
    };
    assert!(replayable.is_replayable());

    // ChronicleEntry is the one variant without @replayable — the side-
    // channel prose stream.
    let chron = Event::ChronicleEntry { tick: 0, template_id: 0 };
    assert!(!chron.is_replayable());
}

#[test]
fn event_hash_is_stamped() {
    // The schema-hash constant is wired up; non-zero means the emitter
    // ran and produced bytes for the current taxonomy.
    let hash = engine_rules::EVENT_HASH;
    assert_ne!(hash, [0u8; 32]);
}
