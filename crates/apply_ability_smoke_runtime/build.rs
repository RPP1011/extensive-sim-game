//! `apply_ability_smoke_runtime` build script. Mirrors
//! `cooldown_probe_runtime`'s build.rs verbatim except for the input
//! fixture path.
//!
//! Lowers `assets/sim/apply_ability_smoke.sim` through the DSL compiler
//! pipeline (parse → resolve → CG lower → schedule → emit). The
//! resulting WGSL + Rust files land in `OUT_DIR/<kernel>.{wgsl,rs}`
//! and are concatenated into `OUT_DIR/generated.rs` for `include!`
//! into `src/lib.rs`.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/apply_ability_smoke_runtime");
    let sim_path = workspace_root.join("assets/sim/apply_ability_smoke.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse apply_ability_smoke.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve apply_ability_smoke.sim");
    // #121 follow-on (2026-05-07): opt the smoke runtime into AOE
    // Path B dispatch. The flag flips every `apply_ability` lowered
    // under this Compilation to `with_aoe_dispatch: true`, which
    // (a) emits the 27-cell spatial walk + per-target chronicle write
    // in the WGSL dispatcher arm, (b) surfaces `agent_pos` +
    // `spatial_grid_cells` + `spatial_grid_starts` + `area_kinds` +
    // `area_args` reads on the dispatcher op via
    // `wire_apply_ability_aoe_reads`. The smoke runtime allocates the
    // matching buffers (see `src/lib.rs::try_new_with_registry`); the
    // parity sweep + the AOE chronicle pin (`aoe_chronicle_pin.rs`)
    // exercise the new path on real GPU.
    //
    // Production runtimes (duel_abilities, tactical_squad_5v5,
    // boss_fight, mass_battle_100v100, …) keep `aoe_dispatch: false`
    // (the default) — their dispatchers stay binding-clean and the
    // spatial-build phases don't fire.
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg_with_opts(
        &comp,
        dsl_compiler::cg::lower::LowerOpts { aoe_dispatch: true, belief_state: false },
    ) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[apply_ability_smoke lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit apply_ability_smoke CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[apply_ability_smoke emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[apply_ability_smoke emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by apply_ability_smoke_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/apply_ability_smoke.sim and rebuilding.\n\n",
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
