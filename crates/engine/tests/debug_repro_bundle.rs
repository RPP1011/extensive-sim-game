//! Round-trip test for `ReproBundle` — capture → write → read → assert equality.

use engine::debug::repro_bundle::ReproBundle;
use engine::event::EventRing;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use glam::Vec3;

fn tmp_path(name: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("repro_bundle_test_{}_{}_{}.bundle", name, pid, nonce))
}

/// Basic round-trip: capture with no optional collectors, write to disk, read
/// back, verify snapshot_bytes and schema_hash are identical.
#[test]
fn repro_bundle_roundtrip_no_collectors() {
    let mut state = SimState::new(4, 99);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 2.0, 3.0),
            hp: 80.0,
            max_hp: 100.0,
        })
        .unwrap();
    state.tick = 42;

    let events = EventRing::<Event>::with_cap(16);

    let bundle = ReproBundle::capture(&state, &events, None, None);

    // snapshot_bytes must be non-empty (engine snapshot is always > 0 bytes)
    assert!(!bundle.snapshot_bytes.is_empty(), "snapshot_bytes should be non-empty");

    // schema_hash should be the engine's current hash (non-zero)
    assert_ne!(bundle.schema_hash, [0u8; 32], "schema_hash should not be all-zeros");

    // Write then read back
    let path = tmp_path("rt");
    bundle.write_to(&path).expect("write_to failed");
    let restored = ReproBundle::read_from(&path).expect("read_from failed");

    assert_eq!(
        restored.snapshot_bytes, bundle.snapshot_bytes,
        "snapshot_bytes must survive round-trip"
    );
    assert_eq!(
        restored.schema_hash, bundle.schema_hash,
        "schema_hash must survive round-trip"
    );
    assert_eq!(
        restored.causal_tree_dump, bundle.causal_tree_dump,
        "causal_tree_dump must survive round-trip"
    );
    assert_eq!(
        restored.mask_trace_bytes, bundle.mask_trace_bytes,
        "mask_trace_bytes must survive round-trip"
    );
    assert_eq!(
        restored.agent_history_bytes, bundle.agent_history_bytes,
        "agent_history_bytes must survive round-trip"
    );

    // Cleanup
    let _ = std::fs::remove_file(&path);
}

/// Verify that the causal tree dump is correctly built from a ring with events.
#[test]
fn repro_bundle_causal_tree_populated() {
    use engine_data::events::Event;
    use engine_data::ids::AgentId;

    let state = SimState::new(4, 7);
    let mut events = EventRing::<Event>::with_cap(32);

    // Push a root event so the tree has at least one root.
    let agent = AgentId::new(1).unwrap();
    let _root = events.push(Event::AgentDied { agent_id: agent, tick: 1 });

    let bundle = ReproBundle::capture(&state, &events, None, None);

    // With one root event the dump should be non-empty.
    assert!(
        !bundle.causal_tree_dump.is_empty(),
        "causal_tree_dump should be non-empty when ring has events"
    );

    // Round-trip still works.
    let path = tmp_path("tree");
    bundle.write_to(&path).expect("write_to failed");
    let restored = ReproBundle::read_from(&path).expect("read_from failed");
    assert_eq!(restored.causal_tree_dump, bundle.causal_tree_dump);
    let _ = std::fs::remove_file(&path);
}

/// Empty ring → causal_tree_dump is empty, bundle still round-trips.
#[test]
fn repro_bundle_empty_ring() {
    let state = SimState::new(4, 0);
    let events = EventRing::<Event>::with_cap(8);

    let bundle = ReproBundle::capture(&state, &events, None, None);
    assert!(bundle.causal_tree_dump.is_empty(), "empty ring yields empty dump");

    let path = tmp_path("empty");
    bundle.write_to(&path).expect("write_to failed");
    let restored = ReproBundle::read_from(&path).expect("read_from failed");
    assert_eq!(restored.snapshot_bytes, bundle.snapshot_bytes);
    assert_eq!(restored.schema_hash, bundle.schema_hash);
    let _ = std::fs::remove_file(&path);
}
