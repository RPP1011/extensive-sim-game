//! Stage 10 lock-in: predator_prey_min.sim's `bounded_kill_count`
//! invariant referencing the materialized `kill_count` view round-trips
//! through resolve. The mask + verb + scoring rows from the design
//! target stay deferred until the from-clause spatial-query routing
//! lands.

#[test]
fn predator_prey_min_invariant_resolves() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    let inv = comp
        .invariants
        .iter()
        .find(|i| i.name == "bounded_kill_count")
        .expect("bounded_kill_count invariant resolved");
    assert_eq!(
        inv.scope.len(),
        1,
        "bounded_kill_count is per-agent — expected one scope param"
    );

    // Pipeline still emits cleanly with the invariant declared.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
}
