//! Writes the emitted source to a temp file and verifies it compiles with rustc.
//!
//! Strategy: attempt full rustc compilation against `engine_voxel`
//! (catches type-level errors). This is sufficient for validating the emitted
//! module is correct Rust.

use dsl_compiler::{emit::emit_terrain, lower::lower_terrain, parse};

fn sample_src() -> &'static str {
    r#"
terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0x1
  materials {
    grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A }
  }
  layer fill { material: grass }
}
"#
}

/// Compile the emitted source with rustc + engine_voxel rlib.
/// Panics on any failure with a diagnostic message.
fn compile_emitted_terrain(emitted: &str) {
    use std::process::Command;

    let tmp = tempfile::tempdir().unwrap();
    let src_path = tmp.path().join("terrain_gen.rs");
    std::fs::write(&src_path, emitted).unwrap();

    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/../../target/debug/deps", manifest)
    });

    // Glob for the engine_voxel rlib so we get the hash suffix right.
    // Pick the most-recently modified one so stale rlibs from earlier
    // build configurations don't shadow the current one.
    let glob_pattern = format!("{}/libengine_voxel-*.rlib", target_dir);
    let rlib_opt = glob::glob(&glob_pattern)
        .unwrap()
        .filter_map(Result::ok)
        .max_by_key(|p| {
            p.metadata()
                .and_then(|m| m.modified())
                .ok()
        });

    let Some(rlib) = rlib_opt else {
        panic!(
            "engine_voxel rlib not found in {}: cannot verify emitted terrain module",
            target_dir
        );
    };

    let out_path = tmp.path().join("libterrain_gen.rlib");
    let output = Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "-L",
            &target_dir,
            &format!("--extern=engine_voxel={}", rlib.display()),
            src_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn rustc");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "emitted terrain_gen.rs failed to compile:\n{}",
            stderr.trim()
        );
    }
}

#[test]
fn emitted_terrain_module_compiles() {
    let src = sample_src();
    let ir = lower_terrain(&parse(src).unwrap().terrain.unwrap()).unwrap();
    let emitted = emit_terrain(&ir);

    compile_emitted_terrain(&emitted);
    println!("PASS: emitted terrain_gen.rs compiled cleanly via rustc");
}
