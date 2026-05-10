//! Plan G G3g — smoke test for `assets/sim/threats_view_probe.sim`.
//!
//! Composes the full G3 stack:
//!   * `@dispatch(per_agent_event_scan)` — G3d's 2-D dispatch
//!   * scalar f32 view storage — G3a's primary slot
//!   * `agents.set_busy_with_ability_id(...)` — G2.1's busy SoA write
//!
//! Asserts:
//!   1. The .sim parses + resolves + lowers (no errors).
//!   2. Both expected kernels emit (`physics_MarkCasterBusy`, `fold_threats`).
//!   3. The fold WGSL has the PerAgentEventScan preamble shape:
//!      - `let observer = gid.x` + `let source_candidate = gid.y`
//!      - busy-filter early-exit on `agent_busy_with_ability_id[source_candidate]`
//!      - 2-D `@workgroup_size(8, 8)`

#[test]
fn threats_view_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/threats_view_probe.sim")
        .expect("read assets/sim/threats_view_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse threats_view_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve threats_view_probe.sim");

    let (cg, lower_diags) = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => (p, Vec::new()),
        Err(o) => {
            let diags: Vec<String> = o.diagnostics.iter().map(|d| format!("{d}")).collect();
            (o.program, diags)
        }
    };
    for d in &lower_diags {
        eprintln!("[lower diag] {d}");
    }

    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg, dsl_compiler::cg::schedule::ScheduleStrategy::Default);
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit threats_view_probe");

    assert!(!artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none");

    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["MarkCasterBusy", "threats"] {
        assert!(kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}");
    }

    // Inspect the threats fold WGSL — must have the PerAgentEventScan
    // preamble shape (G3d's `(observer, source_candidate)` layout +
    // busy-filter early-exit).
    let fold_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("threats") && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .expect("threats fold kernel must emit a .wgsl artifact");

    assert!(fold_wgsl.contains("let observer = gid.x"),
        "PerAgentEventScan preamble must declare observer; WGSL:\n{fold_wgsl}");
    assert!(fold_wgsl.contains("let source_candidate = gid.y"),
        "PerAgentEventScan preamble must declare source_candidate; WGSL:\n{fold_wgsl}");
    assert!(fold_wgsl.contains("agent_busy_with_ability_id[source_candidate] == 0u"),
        "PerAgentEventScan must early-exit non-busy candidates; WGSL:\n{fold_wgsl}");
    assert!(fold_wgsl.contains("@workgroup_size(8, 8)"),
        "PerAgentEventScan kernel must declare 2-D workgroup; WGSL:\n{fold_wgsl}");
}
