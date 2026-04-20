//! End-to-end emission test: parse `assets/sim/events.sim`, resolve, emit,
//! write to a tempdir, and assert the expected files appear with sane
//! content. This does not assert against a byte-exact golden — that job
//! belongs to the xtask `--check` mode running on the committed output.

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
    assert_eq!(
        comp.events.len(),
        8,
        "seed DSL should declare 8 events; found {}",
        comp.events.len()
    );

    let artefacts = emit_with_source(&comp, Some("assets/sim/events.sim"));

    // One struct file per event, alphabetical.
    let rust_files: Vec<&String> =
        artefacts.rust_event_structs.iter().map(|(n, _)| n).collect();
    let mut sorted = rust_files.clone();
    sorted.sort();
    assert_eq!(rust_files, sorted, "rust files must be sorted");

    let expected = [
        "agent_died.rs",
        "damage.rs",
        "heal.rs",
        "shield_applied.rs",
        "slow_applied.rs",
        "slow_expired.rs",
        "stun_applied.rs",
        "transfer_gold.rs",
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

    // The aggregate Rust mod should list every event in the enum.
    for event_name in
        ["AgentDied", "Damage", "Heal", "ShieldApplied", "SlowApplied", "SlowExpired", "StunApplied", "TransferGold"]
    {
        assert!(
            artefacts.rust_events_mod.contains(&format!("{event_name}({event_name}),")),
            "aggregate enum missing variant {event_name}"
        );
    }

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

    assert!(events_dir.join("damage.rs").exists());
    assert!(events_dir.join("mod.rs").exists());
    assert!(tmp.path().join("schema.rs").exists());

    // Content spot-check: the Damage event struct should reference AgentId.
    let damage = fs::read_to_string(events_dir.join("damage.rs")).unwrap();
    assert!(damage.contains("pub struct Damage"));
    assert!(damage.contains("pub target: AgentId"));
    assert!(damage.contains("pub amount: f32"));
    assert!(damage.contains("use engine::ids::AgentId;"));
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
