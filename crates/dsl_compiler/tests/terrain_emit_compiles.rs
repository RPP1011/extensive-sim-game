//! Writes the emitted source to a temp file and verifies it is valid Rust.
//!
//! Strategy: first attempt full rustc compilation against `engine_voxel`
//! (catches type-level errors); if extern resolution fails (fragile in
//! CI due to Vulkan/wgpu transitive rlibs), fall back to `syn::parse_file`
//! which catches all syntactic + structural errors without needing rlibs.

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

/// Try to compile the emitted source with rustc + engine_voxel rlib.
/// Returns `Ok(())` on success, `Err(reason)` if extern resolution fails
/// (so the caller can fall back to syn-parse), or panics if rustc reports
/// an *emitter* bug (syntax / type error in the generated source).
fn try_rustc_compile(emitted: &str) -> Result<(), String> {
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
        return Err(format!(
            "engine_voxel rlib not found in {}; falling back to syn-parse",
            target_dir
        ));
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

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Distinguish "extern resolution failure" (fragile, fall back) from
    // actual emitter bugs (hard failure).
    let is_extern_error = stderr.contains("can't find crate for")
        || stderr.contains("error[E0463]")
        || stderr.contains("extern location for")
        || stderr.contains("is of an unknown type");

    if is_extern_error {
        return Err(format!(
            "extern resolution failed (expected in some envs): {}\nfalling back to syn-parse",
            stderr.trim()
        ));
    }

    // It's a real emitter bug — fail loudly.
    panic!(
        "emitted terrain_gen.rs failed to compile:\n{}",
        stderr.trim()
    );
}

/// Fallback: parse the emitted source as Rust syntax via `syn`.
/// Catches syntactic + structural errors without needing rlibs.
fn syn_parse_check(emitted: &str) {
    let parsed: syn::File =
        syn::parse_str(emitted).expect("emitted terrain_gen.rs should parse as valid Rust");
    assert!(!parsed.items.is_empty(), "emitted module is empty");
    // Spot-check: expect at least one `pub` item.
    let has_pub = parsed.items.iter().any(|item| match item {
        syn::Item::Const(c) => matches!(c.vis, syn::Visibility::Public(_)),
        syn::Item::Static(s) => matches!(s.vis, syn::Visibility::Public(_)),
        syn::Item::Fn(f) => matches!(f.vis, syn::Visibility::Public(_)),
        _ => false,
    });
    assert!(has_pub, "emitted module has no public items");
}

#[test]
fn emitted_terrain_module_compiles() {
    let src = sample_src();
    let ir = lower_terrain(&parse(src).unwrap().terrain.unwrap()).unwrap();
    let emitted = emit_terrain(&ir);

    match try_rustc_compile(&emitted) {
        Ok(()) => {
            println!("PASS: emitted terrain_gen.rs compiled cleanly via rustc");
        }
        Err(reason) => {
            println!("INFO: rustc path skipped — {}", reason);
            println!("INFO: falling back to syn::parse_file check");
            syn_parse_check(&emitted);
            println!("PASS: emitted terrain_gen.rs parses as valid Rust via syn");
        }
    }
}
