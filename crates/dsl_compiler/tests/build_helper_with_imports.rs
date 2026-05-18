//! Asserts emit_namespaced uses parse_with_imports under the hood: if a
//! fixture's .sim has `import ./shared.sim;` and shared.sim exists in
//! the fake workspace, the merged Program's decls are visible in the
//! emitted artifacts.

use std::path::PathBuf;
use std::sync::Mutex;
use dsl_compiler::build_helper;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn fake_env(tmp: &tempfile::TempDir, sim_name: &str, sim_src: &str) -> PathBuf {
    let sims_dir = tmp.path().join("crates/sims");
    let assets_dir = tmp.path().join("assets/sim");
    let stdlib_dir = tmp.path().join("stdlib");
    std::fs::create_dir_all(&sims_dir).unwrap();
    std::fs::create_dir_all(&assets_dir).unwrap();
    std::fs::create_dir_all(&stdlib_dir).unwrap();
    let sim_path = assets_dir.join(format!("{sim_name}.sim"));
    std::fs::write(&sim_path, sim_src).unwrap();
    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::env::set_var("CARGO_MANIFEST_DIR", &sims_dir);
    std::env::set_var("OUT_DIR", &out_dir);
    std::env::set_var("WORLDSIM_STDLIB_ROOT", &stdlib_dir);
    std::env::set_var("WORLDSIM_SANDBOX_ROOT", tmp.path());
    out_dir
}

#[test]
fn fixture_with_import_merges_shared_entity() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let assets = tmp.path().join("assets/sim");
    std::fs::create_dir_all(&assets).unwrap();
    // shared.sim defines a physics rule named WolfRule; main.sim declares the
    // Tick event + imports shared.sim + defines a physics rule named GoblinRule.
    // Rule names are emitted verbatim into generated.rs and .wgsl, so both
    // names must appear in the output for the import merge to be visible.
    std::fs::write(assets.join("shared.sim"),
        "physics WolfRule @phase(per_agent) { on Tick {} { } }\n").unwrap();
    let out_dir = fake_env(&tmp, "fixture_with_import",
        "import ./shared.sim;\nevent Tick { }\nphysics GoblinRule @phase(per_agent) { on Tick {} { } }\n");
    build_helper::emit_namespaced("fixture_with_import");
    // The generated.rs should reflect both Wolf and Goblin (or fail to
    // emit if the merge didn't happen — either failure mode signals a
    // missing wire-up).
    let generated = out_dir.join("fixture_with_import/generated.rs");
    assert!(generated.exists(), "generated.rs not written: {generated:?}");
    let body = std::fs::read_to_string(&generated).unwrap();
    // Look for both names somewhere in the emitted code.
    assert!(body.contains("Wolf"),   "Wolf missing from emitted code");
    assert!(body.contains("Goblin"), "Goblin missing from emitted code");
}
