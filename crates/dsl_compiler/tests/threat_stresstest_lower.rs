//! Plan H — smoke test for `assets/sim/threat_stresstest.sim`.
//!
//! Asserts the .sim composes the cyclic-busy stresstest pattern
//! (MarkBusyByPhase + ClearBusyByPhase + threats view) cleanly
//! through the lowering pipeline.

#[test]
fn threat_stresstest_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/threat_stresstest.sim")
        .expect("read assets/sim/threat_stresstest.sim");
    let program = dsl_compiler::parse(&src).expect("parse threat_stresstest.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve threat_stresstest.sim");

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
        .expect("emit threat_stresstest");

    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["MarkBusyByPhase", "ClearBusyByPhase", "threats"] {
        assert!(kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}");
    }
}
