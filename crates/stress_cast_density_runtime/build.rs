//! `stress_cast_density_runtime` build script. Mirrors
//! `spy_network_runtime`'s build.rs — lowers
//! `assets/sim/stress_cast_density.sim` through the DSL compiler
//! pipeline (parse → resolve → CG lower → schedule → emit). The
//! resulting WGSL + Rust files land in OUT_DIR/<kernel>.{wgsl,rs}
//! and are concatenated into `OUT_DIR/generated.rs` for `include!`
//! into `src/lib.rs`.
//!
//! Tolerates lower diagnostics — the no-op `@phase(post)` consumer
//! pattern matches an empty body, which trips the same well_formed
//! warnings as duel_abilities's ApplyDamage. We follow spy_network's
//! lead and emit anyway.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/stress_cast_density_runtime");
    let sim_path = workspace_root.join("assets/sim/stress_cast_density.sim");

    println!("cargo:rerun-if-changed={}", sim_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let corpus_dir = workspace_root.join("assets/ability_test/stress_cast_density");
    for name in ["AoePulse.ability"] {
        println!("cargo:rerun-if-changed={}", corpus_dir.join(name).display());
    }

    let src = fs::read_to_string(&sim_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sim_path.display()));
    let program = dsl_compiler::parse(&src).expect("parse stress_cast_density.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve stress_cast_density.sim");
    // AOE Path B opt-in (#121, 2026-05-07): the apply_ability dispatcher
    // needs `agent_pos` + `spatial_grid_*` + `area_kinds` + `area_args`
    // bindings to walk the 27-cell neighborhood per cast. Without the
    // opt-in, the WGSL dispatcher collapses to single-target shape and
    // the Spread expansion never fires — the chronicle ring stays at
    // ~agent_cap events per tick instead of agent_cap × ~256.
    let opts = dsl_compiler::cg::lower::LowerOpts {
        aoe_dispatch: true,
        // D3 — per-kernel GPU timestamps + memory-traffic accounting,
        // no .sim source-map overhead. See `DebugDepth` for the level
        // ladder.
        debug: dsl_compiler::cg::lower::DebugDepth::Kernel,
        ..dsl_compiler::cg::lower::LowerOpts::default()
    };
    let cg = match dsl_compiler::cg::lower::lower_compilation_to_cg_with_opts(&comp, opts) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                println!("cargo:warning=[stress_cast_density lower diag] {d}");
            }
            o.program
        }
    };
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    // Thread the same debug level into the emit step so the synthesised
    // dispatch.rs carries `DebugTimings`, `MemTrafficTable`, and the
    // per-kernel `record_<name>_timing` helpers the driver consumes.
    let artifacts = dsl_compiler::cg::emit::emit_cg_program_with_debug(
        &schedule_result.schedule,
        &cg,
        opts.debug,
    )
    .expect("emit stress_cast_density CG program");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:warning=[stress_cast_density emit-stats] {} kernels, schedule has {} stages",
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
            "cargo:warning=[stress_cast_density emit-stats]   {kernel_name}: {bytes} B, {bindings} bindings",
        );
    }

    for (name, body) in &artifacts.wgsl_files {
        fs::write(out_dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}: {e}", name));
    }

    let mut generated = String::new();
    generated.push_str(
        "// AUTO-CONCATENATED from compiler-emitted artifacts by stress_cast_density_runtime/build.rs.\n\
         // Do not edit. Regenerate by editing assets/sim/stress_cast_density.sim and rebuilding.\n\n",
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
