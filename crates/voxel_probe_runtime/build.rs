//! `voxel_probe_runtime` build script. Lowers
//! `assets/sim/voxel_probe.sim` through the standard DSL pipeline
//! (parse → resolve → CG lower → schedule → emit) and concatenates the
//! per-kernel WGSL + Rust files into `OUT_DIR/generated.rs` for
//! `include!` into `src/lib.rs`. Mirrors
//! `apply_ability_chronicle_consumer_runtime/build.rs` shape.
//!
//! Tolerates lower diagnostics — none expected for this fixture, but
//! the same `DriverOutcome::Err → o.program` pattern is in place so a
//! P6 well_formed nag (e.g. if a future edit adds a `@phase(post)` rule
//! with `agents.set_*`) doesn't break the build.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/voxel_probe_runtime");
    let sim_path = workspace_root.join("assets/sim/voxel_probe.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let corpus_dir = workspace_root.join("assets/ability_test/voxel_probe");
    for name in ["PlaceTestVoxel.ability", "HarvestTestVoxel.ability"] {
        println!(
            "cargo:rerun-if-changed={}",
            corpus_dir.join(name).display(),
        );
    }

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse voxel_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve voxel_probe.sim");
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[voxel_probe lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit voxel_probe CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[voxel_probe emit-stats] {} kernels, schedule has {} stages",
        artifacts.kernel_index.len(),
        schedule_result.schedule.stages.len(),
    );
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.wgsl");
        let body = match artifacts.wgsl_files.get(&key) {
            Some(b) => b,
            None => continue,
        };
        let bytes = body.len();
        let bindings = body.matches("@binding(").count();
        println!(
            "cargo:warning=[voxel_probe emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by voxel_probe_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/voxel_probe.sim and rebuilding.\n\n",
    );
    let mut wrap_module = |name: &str, content: &str| {
        generated.push_str(
            "#[allow(non_snake_case, unused_imports, unused_variables, dead_code, clippy::all)]\n",
        );
        generated.push_str(&format!("pub mod {name} {{\n"));
        generated.push_str(content);
        generated.push_str("\n}\n\n");
    };
    for kernel_name in &artifacts.kernel_index {
        let key = format!("{kernel_name}.rs");
        let content = artifacts
            .rust_files
            .get(&key)
            .unwrap_or_else(|| panic!("missing rust file {key} for kernel {kernel_name}"));
        wrap_module(kernel_name, content);
    }
    for sibling in ["schedule", "dispatch", "invariants", "metrics", "probes"] {
        let key = format!("{sibling}.rs");
        if let Some(content) = artifacts.rust_files.get(&key) {
            wrap_module(sibling, content);
        }
    }
    if let Some(lib_content) = artifacts.rust_files.get("lib.rs") {
        for line in lib_content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub mod ") || trimmed.starts_with("#![") {
                continue;
            }
            generated.push_str(line);
            generated.push('\n');
        }
    }

    fs::write(out_dir.join("generated.rs"), generated)
        .unwrap_or_else(|e| panic!("write generated.rs: {e}"));
}
