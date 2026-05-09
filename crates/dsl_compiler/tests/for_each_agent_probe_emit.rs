//! End-to-end emit dry-run for `assets/sim/for_each_agent_probe.sim`.
//!
//! Walks the fixture through the full pipeline:
//!   parse → resolve → CG lower → schedule synthesize → emit_cg_program.
//!
//! Pins three things about the new `for_each_agent <binder>` body-shape
//! primitive (Task #229):
//!
//!   1. The fixture compiles cleanly through the full pipeline (no
//!      lower / schedule / emit diagnostics).
//!   2. The resulting per-agent rule's op shape is `OneShot` (the
//!      auto-retag triggered by `list_contains_for_each_agent_body`).
//!   3. The emitted WGSL contains the linear-scan loop with the
//!      `agent_alive[i] != 0u` gate.

use std::fs;
use std::path::PathBuf;

use dsl_compiler::cg::dispatch::DispatchShape;
use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::op::ComputeOpKind;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

fn fixture_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/dsl_compiler");
    workspace_root.join("assets/sim/for_each_agent_probe.sim")
}

#[test]
fn for_each_agent_probe_fixture_emits_wgsl() {
    let src = fs::read_to_string(fixture_path()).expect("read for_each_agent_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse for_each_agent_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve for_each_agent_probe.sim");
    let cg = lower_compilation_to_cg(&comp).expect("lower to CG");

    // (1) The per-agent rule's op shape must be OneShot — auto-retagged
    // because the body contains a ForEachAgentBody.
    let mut found_per_agent_one_shot = false;
    for op in &cg.ops {
        if let ComputeOpKind::PhysicsRule { .. } = &op.kind {
            assert_eq!(
                op.shape,
                DispatchShape::OneShot,
                "for_each_agent body must retag the per-agent rule's dispatch \
                 to OneShot — got {:?}",
                op.shape,
            );
            found_per_agent_one_shot = true;
        }
    }
    assert!(
        found_per_agent_one_shot,
        "expected at least one PhysicsRule op in the lowered fixture",
    );

    // (2) Schedule + emit must succeed (informational fusion diagnostics
    // about shape splits are fine — boids_cg_emit accepts the same).
    let schedule_result = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let artifacts =
        emit_cg_program(&schedule_result.schedule, &cg).expect("emit_cg_program succeeds");

    // (3) The WGSL for the BumpAllMana physics kernel must contain the
    // linear scan loop AND the alive gate. Pin both substrings so any
    // future emit-template drift surfaces here.
    let mut found_kernel = false;
    for (name, body) in &artifacts.wgsl_files {
        if !name.contains("BumpAllMana") {
            continue;
        }
        found_kernel = true;
        assert!(
            body.contains("for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap"),
            "WGSL kernel `{name}` must contain the linear scan loop:\n\
             {body}",
        );
        assert!(
            body.contains("if (agent_alive[per_pair_candidate] != 0u)"),
            "WGSL kernel `{name}` must gate body execution on agent_alive:\n\
             {body}",
        );
    }
    assert!(
        found_kernel,
        "expected at least one WGSL kernel containing `BumpAllMana` in its name; \
         got: {:?}",
        artifacts.wgsl_files.keys().collect::<Vec<_>>(),
    );

    // (4) Naga validation pass — every emitted WGSL file must parse
    // through naga's `wgsl-in` parser. Mirrors `boids_cg_emit`'s
    // post-emit validation: a structurally-valid emit can still
    // produce invalid WGSL (unbound identifiers, type mismatches), and
    // the engine_gpu_rules consumer pins the same dialect, so a clean
    // parse here is what the GPU dispatch shim relies on.
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, body) in &artifacts.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            failures.push((name.clone(), e.emit_to_string(body)));
        }
    }
    if !failures.is_empty() {
        for (name, msg) in &failures {
            eprintln!("\n--- naga validation failure: {name} ---\n{msg}");
        }
        if let Some(body) = artifacts.wgsl_files.get("physics_BumpAllMana.wgsl") {
            eprintln!("\n--- physics_BumpAllMana.wgsl (for context) ---\n{body}");
        }
        panic!(
            "{} of {} kernels failed naga validation",
            failures.len(),
            artifacts.wgsl_files.len()
        );
    }
}
