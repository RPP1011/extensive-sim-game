//! Stage 5 lock-in: predator_prey_min.sim's `kill_count` materialized
//! view-fold round-trips through the compiler pipeline and lands as a
//! fold_kill_count kernel in the schedule.
//!
//! No event currently emits Killed (Stage 7's StrikePrey is the
//! intended emitter). The view's per-agent storage stays at the
//! initial value (0.0) at runtime; this test pins that the kernel +
//! seed-indirect dispatch land at compile time so Stage 7 inherits a
//! known-good baseline.

#[test]
fn predator_prey_min_kill_count_view_fold_emits() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    // Resolved view shows up in comp.views.
    let view_names: Vec<&str> = comp.views.iter().map(|v| v.name.as_str()).collect();
    assert!(
        view_names.contains(&"kill_count"),
        "kill_count should appear in resolved views: {:?}",
        view_names
    );

    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let arts = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");

    // The schedule emits a `fold_kill_count` kernel — that's the
    // materialized-view-fold consumer of the Killed ring.
    assert!(
        arts.kernel_index.iter().any(|n| n == "fold_kill_count"),
        "expected fold_kill_count kernel in emit; got {:?}",
        arts.kernel_index
    );
}
