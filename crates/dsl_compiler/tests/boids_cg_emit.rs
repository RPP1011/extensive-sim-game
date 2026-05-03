//! Emit dry-run for the Boids fixture.
//!
//! Walks `assets/sim/boids.sim` through the full pipeline:
//!   parse → resolve → CG lower → schedule synthesize → emit_cg_program.
//!
//! Goal: surface any cascade on the emit side now that the lowering
//! is clean (`boids_cg_lowering.rs` confirms 5 ops, 6 stmts, 13 exprs
//! at HEAD `4bad8426`). The minimum-viable GPU pipeline rebuild needs
//! WGSL files; this test is the discovery harness for what's actually
//! producible from the boids CG IR today.

use std::fs;
use std::path::PathBuf;

use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

fn boids_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/dsl_compiler");
    workspace_root.join("assets/sim/boids.sim")
}

#[test]
fn boids_fixture_emits_wgsl() {
    let src = fs::read_to_string(boids_path()).expect("read boids.sim");
    let program = dsl_compiler::parse(&src).expect("parse boids.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve boids.sim");
    let cg = lower_compilation_to_cg(&comp).expect("lower to CG");

    let result = synthesize_schedule(&cg, ScheduleStrategy::Default);
    println!(
        "boids_cg_emit: {} stages | {} fusion diagnostics | {} schedule diagnostics",
        result.schedule.stages.len(),
        result.fusion_diagnostics.len(),
        result.schedule_diagnostics.len(),
    );
    for d in &result.fusion_diagnostics {
        println!("  fusion: {d:?}");
    }
    for d in &result.schedule_diagnostics {
        println!("  schedule: {d:?}");
    }

    let artifacts = emit_cg_program(&result.schedule, &cg)
        .expect("boids_cg_emit: emit failed");
    println!(
        "boids_cg_emit: EMIT CLEAN — {} wgsl files, {} rust files, {} kernels",
        artifacts.wgsl_files.len(),
        artifacts.rust_files.len(),
        artifacts.kernel_index.len(),
    );
    for name in &artifacts.kernel_index {
        println!("  kernel: {name}");
    }

    // Naga-validate every emitted WGSL file. naga's `wgsl-in` parser
    // is the same WGSL dialect engine_gpu_rules consumed pre-nuke
    // (pinned to `=26.0.0`), so a clean parse here is the contract
    // the GPU dispatch shim will rely on.
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, body) in &artifacts.wgsl_files {
        match naga::front::wgsl::parse_str(body) {
            Ok(_) => {}
            Err(e) => failures.push((name.clone(), e.emit_to_string(body))),
        }
    }
    if !failures.is_empty() {
        eprintln!("\n=== NAGA VALIDATION FAILURES ===");
        for (name, msg) in &failures {
            eprintln!("\n--- {name} ---\n{msg}");
        }
        // Also dump physics_MoveBoid for context.
        if let Some(body) = artifacts.wgsl_files.get("physics_MoveBoid.wgsl") {
            eprintln!("\n--- physics_MoveBoid.wgsl (for context) ---\n{body}");
        }
        panic!("{} of {} kernels failed naga validation", failures.len(), artifacts.wgsl_files.len());
    }
    println!(
        "boids_cg_emit: NAGA OK — all {} wgsl kernels parse cleanly",
        artifacts.wgsl_files.len()
    );
}
