//! `tom_probe_runtime` build script. Mirrors `verb_probe_runtime`'s
//! build.rs verbatim except for the input fixture path.
//!
//! Like the verb_probe build, we tolerate lower diagnostics — this is
//! a discovery probe and the WHOLE POINT is that BeliefsAccessor and
//! `theory_of_mind.believes_knows()` are rejected at lower time today.
//! The build still succeeds; the partial program (with both producer
//! physics rules dropped) emits the admin + view-fold kernels so the
//! runtime can stand up storage and report `fact_witnesses[i] = 0.0`
//! as the OUTCOME (b) NO FIRE signal.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/tom_probe_runtime");
    let sim_path = workspace_root.join("assets/sim/tom_probe.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse tom_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve tom_probe.sim");
    // Tolerate lower diagnostics — discovery: surface what's emitted
    // even though both CheckBelief* physics rules drop out at lower
    // time (BeliefsAccessor + theory_of_mind.believes_knows are not
    // yet lowered).
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[tom_probe lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit tom_probe CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[tom_probe emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[tom_probe emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by tom_probe_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/tom_probe.sim and rebuilding.\n\n",
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
