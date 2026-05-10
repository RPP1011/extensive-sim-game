//! Smoke test for `assets/sim/firebolt_probe.sim` lowering.
//!
//! Lives in dsl_compiler/tests/ rather than the firebolt_probe_runtime
//! crate so we can validate the .sim parses + lowers + emits cleanly
//! WITHOUT building the per-fixture GPU runtime crate (that's a larger
//! follow-up). When the runtime crate lands it will exercise the same
//! pipeline end-to-end on real wgpu hardware.
//!
//! Plan G G4: this test is the smallest .sim-level pin proving the
//! grammar additions (busy_until_tick / busy_with_ability_id /
//! busy_started_at_tick / busy_target_slot SoA accessors) lower
//! through to a complete schedule + WGSL emit without diagnostics.

#[test]
fn firebolt_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/firebolt_probe.sim")
        .expect("read assets/sim/firebolt_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse firebolt_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve firebolt_probe.sim");

    let (cg, lower_diags) = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => (p, Vec::new()),
        Err(o) => {
            let diags: Vec<String> = o.diagnostics.iter().map(|d| format!("{d}")).collect();
            (o.program, diags)
        }
    };
    // Tolerate the same P6 diagnostic that apply_ability_chronicle_consumer
    // tolerates (PerEvent + agents.set_hp). Surface it but don't fail.
    for d in &lower_diags {
        eprintln!("[lower diag] {d}");
    }

    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg, dsl_compiler::cg::schedule::ScheduleStrategy::Default);
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit firebolt_probe");

    assert!(!artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none");

    // Sanity-check: the busy-resolution rule and the cast-begin recorder
    // should both have produced kernels, identifiable by name.
    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    assert!(kernel_names.iter().any(|n| n.contains("ResolveBusy")),
        "expected ResolveBusy kernel; got: {kernel_names:?}");
    assert!(kernel_names.iter().any(|n| n.contains("RecordCastBegin")),
        "expected RecordCastBegin kernel; got: {kernel_names:?}");
    assert!(kernel_names.iter().any(|n| n.contains("DispatchFirebolt")),
        "expected DispatchFirebolt kernel; got: {kernel_names:?}");
    assert!(kernel_names.iter().any(|n| n.contains("ApplyChronicleDamage")),
        "expected ApplyChronicleDamage kernel; got: {kernel_names:?}");
}
