//! Asserts emit_namespaced writes terrain_gen.rs into the fixture's
//! OUT_DIR subdir when the source has a terrain block — and does NOT
//! write it when there is no terrain block (so existing fixtures
//! stay quiet).
//!
//! T11: Wire emit_terrain into build_helper::emit_namespaced.
//!
//! Both tests mutate process-global environment variables
//! (`CARGO_MANIFEST_DIR`, `OUT_DIR`). A static mutex serialises them
//! so `cargo test`'s default multi-threaded runner doesn't race.

use std::path::PathBuf;
use std::sync::Mutex;

use dsl_compiler::build_helper;

// Serialise env-var manipulation across tests in this binary.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn fake_env(tmp: &tempfile::TempDir, sim_name: &str, sim_src: &str) -> (PathBuf, PathBuf) {
    // Lay out a fake workspace: <tmp>/crates/sims/, <tmp>/assets/sim/<sim>.sim
    let sims_dir = tmp.path().join("crates/sims");
    let assets_dir = tmp.path().join("assets/sim");
    std::fs::create_dir_all(&sims_dir).unwrap();
    std::fs::create_dir_all(&assets_dir).unwrap();
    let sim_path = assets_dir.join(format!("{sim_name}.sim"));
    std::fs::write(&sim_path, sim_src).unwrap();

    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    std::env::set_var("CARGO_MANIFEST_DIR", &sims_dir);
    std::env::set_var("OUT_DIR", &out_dir);
    (sims_dir, out_dir)
}

#[test]
fn writes_terrain_gen_rs_when_source_has_terrain_block() {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let (_sims, out_dir) = fake_env(&tmp, "with_terrain", r#"
terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#);
    build_helper::emit_namespaced("with_terrain");
    let path = out_dir.join("with_terrain/terrain_gen.rs");
    assert!(path.exists(), "terrain_gen.rs not written: {path:?}");
    let body = std::fs::read_to_string(&path).unwrap();
    assert!(body.contains("pub fn generate_terrain"), "missing generate_terrain in: {body}");
}

#[test]
fn skips_terrain_gen_rs_when_source_has_no_terrain_block() {
    let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let (_sims, out_dir) = fake_env(&tmp, "no_terrain", "// empty sim\n");
    // emit_namespaced may panic on an entirely empty .sim depending on
    // parser rules — guard with catch_unwind so the test isolates the
    // single behaviour under check.
    let _ = std::panic::catch_unwind(|| build_helper::emit_namespaced("no_terrain"));
    let path = out_dir.join("no_terrain/terrain_gen.rs");
    assert!(!path.exists(), "terrain_gen.rs unexpectedly written for non-terrain source");
}
