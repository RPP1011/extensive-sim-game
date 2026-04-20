//! End-to-end emission test: parse `assets/sim/events.sim`, resolve, emit,
//! write to a tempdir, and assert the expected files appear with sane
//! content. This does not assert against a byte-exact golden — that job
//! belongs to the xtask `--check` mode running on the committed output.
//!
//! At milestone 2's integration step the seed DSL was expanded from 8 to
//! the full legacy `engine::event::Event` taxonomy (35 variants) so the
//! hand-written enum could be retired. The assertions below walk a
//! representative subset of the expanded taxonomy.

use std::fs;
use std::path::PathBuf;

use dsl_compiler::{compile, emit_with_source};

fn seed_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Walk up to the repo root, then into assets/sim/events.sim.
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("assets/sim/events.sim");
    p
}

#[test]
fn emit_from_seed_produces_expected_files() {
    let src = fs::read_to_string(seed_path()).expect("seed fixture must exist");
    let comp = compile(&src).expect("seed must compile");
    // The seed now carries every variant the pre-milestone engine::event
    // enum did. Keep this assertion loose on the high side so adding a
    // new event doesn't churn this fixture on every commit.
    assert!(
        comp.events.len() >= 30,
        "expanded seed DSL should declare >=30 events; found {}",
        comp.events.len()
    );

    let artefacts = emit_with_source(&comp, Some("assets/sim/events.sim"));

    // One struct file per event, alphabetical.
    let rust_files: Vec<&String> =
        artefacts.rust_event_structs.iter().map(|(n, _)| n).collect();
    let mut sorted = rust_files.clone();
    sorted.sort();
    assert_eq!(rust_files, sorted, "rust files must be sorted");

    // Representative subset — events that exist both pre- and post-cutover.
    let expected = [
        "agent_died.rs",
        "agent_attacked.rs",
        "agent_moved.rs",
        "effect_damage_applied.rs",
        "effect_heal_applied.rs",
        "effect_shield_applied.rs",
        "effect_slow_applied.rs",
        "slow_expired.rs",
        "stun_expired.rs",
        "chronicle_entry.rs",
    ];
    for e in expected {
        assert!(
            artefacts.rust_event_structs.iter().any(|(n, _)| n == e),
            "expected rust file {} missing",
            e
        );
        let py = e.replace(".rs", ".py");
        assert!(
            artefacts.python_event_modules.iter().any(|(n, _)| n == &py),
            "expected python file {} missing",
            py
        );
    }

    // The aggregate Rust mod should list every event as a struct-style
    // variant — `Event::AgentDied { ... }` rather than `AgentDied(AgentDied)`.
    // Engine pattern matches rely on this shape.
    for event_name in [
        "AgentDied",
        "AgentAttacked",
        "AgentMoved",
        "EffectDamageApplied",
        "ChronicleEntry",
    ] {
        assert!(
            artefacts.rust_events_mod.contains(&format!("{event_name} {{")),
            "aggregate enum missing struct-style variant {event_name}"
        );
    }

    // Helper impls exported alongside the enum.
    assert!(
        artefacts.rust_events_mod.contains("pub fn tick(&self) -> u32"),
        "Event::tick() helper missing"
    );
    assert!(
        artefacts
            .rust_events_mod
            .contains("pub fn is_replayable(&self) -> bool"),
        "Event::is_replayable() helper missing"
    );

    // ChronicleEntry is the only non-replayable variant in the seed; it
    // must return false.
    assert!(
        artefacts
            .rust_events_mod
            .contains("Event::ChronicleEntry { .. } => false"),
        "ChronicleEntry should be flagged non-replayable"
    );

    // The schema.rs must carry exactly 32 hex bytes.
    let schema = &artefacts.schema_rs;
    let hex_count = schema.matches("0x").count();
    assert_eq!(hex_count, 32, "schema.rs should have 32 hex bytes");

    // Write everything to a tempdir and sanity-check paths.
    let tmp = tempfile::tempdir().expect("tempdir");
    let events_dir = tmp.path().join("events");
    fs::create_dir_all(&events_dir).unwrap();
    for (name, content) in &artefacts.rust_event_structs {
        fs::write(events_dir.join(name), content).unwrap();
    }
    fs::write(events_dir.join("mod.rs"), &artefacts.rust_events_mod).unwrap();
    fs::write(tmp.path().join("schema.rs"), &artefacts.schema_rs).unwrap();

    assert!(events_dir.join("agent_died.rs").exists());
    assert!(events_dir.join("mod.rs").exists());
    assert!(tmp.path().join("schema.rs").exists());

    // Content spot-check: the AgentDied event struct should reference
    // AgentId via the new `crate::ids::` import path (post-cutover; the
    // old `engine::ids::` import went away when the dep cycle flipped).
    let died = fs::read_to_string(events_dir.join("agent_died.rs")).unwrap();
    assert!(died.contains("pub struct AgentDied"));
    assert!(died.contains("pub agent_id: AgentId"));
    assert!(died.contains("use crate::ids::AgentId;"));
}

#[test]
fn hash_matches_rust_const() {
    let src = fs::read_to_string(seed_path()).expect("seed fixture must exist");
    let comp = compile(&src).expect("seed must compile");
    let artefacts = emit_with_source(&comp, Some("assets/sim/events.sim"));

    // First byte of the hash should appear in the emitted schema.rs.
    let first = format!("0x{:02x}", artefacts.event_hash[0]);
    assert!(
        artefacts.schema_rs.contains(&first),
        "emitted schema.rs must contain the first byte of event_hash ({})",
        first
    );
}
