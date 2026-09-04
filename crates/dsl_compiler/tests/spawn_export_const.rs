//! `spawn <Subkind> count <N> export <NAME> { ... }` — a compile-time
//! population count a `.sim` author can ask the compiler to publish as a
//! named Rust constant, instead of host code hand-copying the literal from
//! the `.sim` source (see `ast::SpawnBlock::export`'s doc comment for why:
//! `webband`'s raid pool cap drifting from its own hardcoded mirror of a
//! `.sim` spawn count was the motivating bug).
//!
//! Two layers, matching this crate's existing test split for
//! `synthesize_runtime_core_a2` (see `init_const_type_routing.rs`):
//!   1. A synthetic-artifacts unit test drives `synthesize_runtime_core_a2`
//!      directly with hand-built `ResolvedSpawnBlock`s — independent of the
//!      full DSL parse/resolve/lower/emit stack.
//!   2. A full end-to-end test compiles a real minimal `.sim` fixture
//!      through `build_helper::emit_namespaced`, proving the grammar/AST/
//!      resolve wiring actually reaches the emitter, and that `export` on a
//!      `config.*`-driven count is rejected rather than silently ignored.

use std::path::PathBuf;
use std::sync::Mutex;

use dsl_compiler::build_helper::{self, synthesize_runtime_core_a2, ResolvedSpawnBlock};
use dsl_compiler::cg::emit::EmittedArtifacts;

// ---------------------------------------------------------------------------
// Layer 1: synthetic artifacts, `synthesize_runtime_core_a2` directly.

fn spawn(subkind: &str, count: u32, export: Option<&str>) -> ResolvedSpawnBlock {
    ResolvedSpawnBlock {
        subkind: subkind.into(),
        creature_type_ord: 0,
        count,
        export: export.map(str::to_string),
        fields: Vec::new(),
    }
}

fn synthesize_with_spawns(spawns: &[ResolvedSpawnBlock]) -> String {
    let artifacts = EmittedArtifacts::default();
    synthesize_runtime_core_a2(
        "spawn_export_test",
        &artifacts,
        &[],
        spawns,
        &std::collections::BTreeMap::new(),
        &[],
        &[],
        None,
        &[],
        false,
        false, // binds_navgrid
        &[],
        &[],
        0,
        0,
        false,
        None,
        "{\"bindings\":[]}",
        "{\"arena_radius\":0.0,\"camera\":\"Observer\",\"agents\":[],\"vfx\":[]}",
        "{\"hud\":[],\"screens\":[]}",
        dsl_compiler::cg::lower::DebugDepth::Off,
    )
}

#[test]
fn exported_spawn_counts_become_module_level_consts() {
    let core = synthesize_with_spawns(&[
        spawn("Looter", 12, Some("RAID_POOL_LOOTERS")),
        spawn("Bandit", 12, Some("RAID_POOL_BANDITS")),
        spawn("Raider", 12, Some("RAID_POOL_RAIDERS")),
        // Un-exported — must NOT appear as a const at all.
        spawn("Warlord", 4, None),
    ]);
    assert!(
        core.contains("pub const RAID_POOL_LOOTERS: u32 = 12;"),
        "missing RAID_POOL_LOOTERS const. Generated source:\n{core}",
    );
    assert!(
        core.contains("pub const RAID_POOL_BANDITS: u32 = 12;"),
        "missing RAID_POOL_BANDITS const. Generated source:\n{core}",
    );
    assert!(
        core.contains("pub const RAID_POOL_RAIDERS: u32 = 12;"),
        "missing RAID_POOL_RAIDERS const. Generated source:\n{core}",
    );
    assert!(
        !core.contains("Warlord"),
        "an un-exported spawn block must not surface in the const block: {core}",
    );
}

#[test]
fn no_exports_emits_nothing_extra() {
    let core = synthesize_with_spawns(&[spawn("Looter", 12, None)]);
    assert!(!core.contains("spawn Looter count"), "un-exported spawn leaked a comment: {core}");
}

// ---------------------------------------------------------------------------
// Layer 2: a real `.sim` fixture through the full pipeline.

fn env_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn fake_env(tmp: &tempfile::TempDir, sim_name: &str, sim_src: &str) -> PathBuf {
    let sims_dir = tmp.path().join("crates/sims");
    let assets_dir = tmp.path().join("assets/sim");
    std::fs::create_dir_all(&sims_dir).unwrap();
    std::fs::create_dir_all(&assets_dir).unwrap();
    std::fs::write(assets_dir.join(format!("{sim_name}.sim")), sim_src).unwrap();

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    std::env::set_var("CARGO_MANIFEST_DIR", &sims_dir);
    std::env::set_var("OUT_DIR", &out_dir);
    out_dir
}

#[test]
fn a_real_fixture_compiles_the_export_into_runtime_core() {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = fake_env(
        &tmp,
        "spawn_export_fixture",
        r#"
entity Grunt : Agent {
}

init {
  spawn Grunt count 7 export GRUNT_POOL_SIZE {
    alive: 1,
  }
}
"#,
    );
    build_helper::emit_namespaced("spawn_export_fixture");
    let body =
        std::fs::read_to_string(out_dir.join("spawn_export_fixture/runtime_core.rs")).unwrap();
    assert!(
        body.contains("pub const GRUNT_POOL_SIZE: u32 = 7;"),
        "missing GRUNT_POOL_SIZE const in compiled runtime_core.rs:\n{body}",
    );
}

#[test]
fn export_on_a_config_driven_count_is_rejected() {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let _out_dir = fake_env(
        &tmp,
        "spawn_export_config_rejected",
        r#"
config waves {
  cap: u32 = 10,
}

entity Grunt : Agent {
}

init {
  spawn Grunt count config.waves.cap export GRUNT_POOL_SIZE {
    alive: 1,
  }
}
"#,
    );
    let result = std::panic::catch_unwind(|| {
        build_helper::emit_namespaced("spawn_export_config_rejected");
    });
    assert!(result.is_err(), "a config-driven `export` should panic, not silently compile");
    let msg = result
        .err()
        .and_then(|e| e.downcast_ref::<String>().cloned().or_else(|| {
            e.downcast_ref::<&str>().map(|s| s.to_string())
        }))
        .unwrap_or_default();
    assert!(
        msg.contains("export") && msg.contains("compile-time constant"),
        "panic message should explain the export/config conflict, got: {msg}",
    );
}
