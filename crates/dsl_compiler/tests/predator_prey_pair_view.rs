//! Stage 9 lock-in: predator_prey_min.sim's `predator_focus` view
//! declared with `storage = pair_map` lowers to a per-pair fold
//! kernel (`fold_predator_focus`).

#[test]
fn predator_prey_min_pair_keyed_view_emits() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    let view = comp
        .views
        .iter()
        .find(|v| v.name == "predator_focus")
        .expect("predator_focus view resolved");
    assert_eq!(
        view.params.len(),
        2,
        "predator_focus is pair-keyed — expected 2 params, got {}",
        view.params.len()
    );

    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let arts = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
    assert!(
        arts.kernel_index.iter().any(|n| n == "fold_predator_focus"),
        "expected fold_predator_focus kernel; got {:?}",
        arts.kernel_index
    );
}
