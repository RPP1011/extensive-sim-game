//! `duel_abilities_runtime` build script. Mirrors `duel_1v1_runtime`'s
//! build.rs verbatim except for the input fixture path.
//!
//! Lowers `assets/sim/duel_abilities.sim` through the DSL compiler
//! pipeline (parse → resolve → CG lower → schedule → emit). The
//! resulting WGSL + Rust files land in OUT_DIR/<kernel>.{wgsl,rs}
//! and are concatenated into `OUT_DIR/generated.rs` for `include!`
//! into `src/lib.rs`.
//!
//! Tolerates lower diagnostics — duel_abilities inherits duel_1v1's
//! known-deferred well_formed warnings (P6 + cycle: chronicle physics
//! writes agent.hp/alive on the same ring the verb scoring reads).
//! Discovery doc parent: `docs/superpowers/notes/2026-05-04-duel_1v1.md`.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/duel_abilities_runtime");
    let sim_path = workspace_root.join("assets/sim/duel_abilities.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run if the source-of-truth .ability files change too — the
    // runtime's binding check re-parses these at startup, so a corpus
    // edit needs to surface as a stale-build trigger.
    let corpus_dir = workspace_root.join("assets/ability_test/duel_abilities");
    for name in ["Strike.ability", "ShieldUp.ability", "Mend.ability", "Bleed.ability", "Reap.ability", "Vampirize.ability", "Fortify.ability", "Daze.ability"] {
        println!("cargo:rerun-if-changed={}", corpus_dir.join(name).display());
    }

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse duel_abilities.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve duel_abilities.sim");
    // Tolerate lower diagnostics — duel_abilities inherits duel_1v1's
    // known-deferred well_formed warnings (P6 + cycle).
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[duel_abilities lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit duel_abilities CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[duel_abilities emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[duel_abilities emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by duel_abilities_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/duel_abilities.sim and rebuilding.\n\n",
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
