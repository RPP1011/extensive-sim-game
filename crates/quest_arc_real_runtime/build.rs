//! `quest_arc_real_runtime` build script. Mirrors `duel_1v1_runtime`'s
//! build.rs verbatim except for the input fixture path.
//!
//! Lowers `assets/sim/quest_arc_real.sim` through the DSL compiler
//! pipeline (parse → resolve → CG lower → schedule → emit).

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/quest_arc_real_runtime");
    let sim_path = workspace_root.join("assets/sim/quest_arc_real.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse quest_arc_real.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve quest_arc_real.sim");
    // Tolerate lower diagnostics — quest_arc_real follows duel_1v1's
    // pattern (state-write/state-read cycle on `mana`) which surfaces
    // the well_formed P6 + cycle warnings. They are advisory, not
    // blocking, and duel_1v1's build.rs uses the same tolerance.
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[quest_arc_real lower diag] {d}");
            }
            o.program
        }
    };
    // Conservative schedule = no fusion. With Default the schedule
    // synthesiser fuses ApplyStageAdvance + ApplyQuestCompleted +
    // verb_chronicle_AcceptQuest into one kernel, but those Apply*
    // ops READ events that the AcceptQuest chronicle EMITS during
    // the same dispatch — so the apply runs before the emit (in the
    // single-pass-per-event-idx semantics) and the cycle never
    // closes (mana never advances). Conservative emits one kernel
    // per op so the chronicle emits land in the ring before the
    // apply kernel reads it. (Same root cause shape as the
    // duel_1v1 design: chronicle Strike/Spell/Heal are kept
    // SEPARATE from ApplyDamage/ApplyHeal in duel_1v1's emitted
    // kernel set.)
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Conservative,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit quest_arc_real CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[quest_arc_real emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[quest_arc_real emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by quest_arc_real_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/quest_arc_real.sim and rebuilding.\n\n",
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
