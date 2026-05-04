//! Stage 3 lock-in: predator_prey_min.sim's `@top_k(1)` `closest_prey`
//! spatial-query declaration round-trips through parse → resolve.
//!
//! The query is unused by any rule today (Stage 7's StrikePrey is the
//! intended consumer). This test pins the parse + resolve surface so
//! Stage 7 inherits a known-good baseline. The compiler does not emit
//! a kernel for an unused spatial-query — same shape boids.sim uses
//! for its declared-but-unused `nearby_other` query.

#[test]
fn predator_prey_min_top_k_query_resolves() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse predator_prey_min.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve predator_prey_min.sim");

    let names: Vec<&str> = comp
        .spatial_queries
        .iter()
        .map(|q| q.name.as_str())
        .collect();
    assert!(
        names.contains(&"closest_prey"),
        "closest_prey should appear in resolved spatial_queries: {:?}",
        names
    );

    // Compile pipeline still emits cleanly with the unused query
    // declared. Mirrors boids.sim's declared-but-unused `nearby_other`
    // query path.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
}
