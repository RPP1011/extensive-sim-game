//! Stage 6 lock-in: predator_prey_min.sim's `kill_count` view carries
//! the `@decay(rate = 0.95, per = tick)` annotation through resolve,
//! and the resident view_storage_anchor binding is allocated in the
//! emitted fold kernel.
//!
//! The actual per-tick `view *= 0.95` multiplication in the kernel
//! body is part of the Phase-8 body-emit closure and lives outside
//! Stage 6's scope; the annotation surface and the anchor-storage
//! allocation are the parts this stage pins.

#[test]
fn predator_prey_min_kill_count_carries_decay_annotation() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    let kill_count = comp
        .views
        .iter()
        .find(|v| v.name == "kill_count")
        .expect("kill_count view");
    let names: Vec<&str> = kill_count
        .annotations
        .iter()
        .map(|a| a.name.as_str())
        .collect();
    assert!(
        names.contains(&"decay"),
        "kill_count should carry @decay annotation: {:?}",
        names
    );

    // Anchor-storage binding is part of the emitted fold kernel.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let arts = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
    let fold_wgsl = arts
        .wgsl_files
        .get("fold_kill_count.wgsl")
        .expect("fold_kill_count.wgsl emitted");
    assert!(
        fold_wgsl.contains("view_storage_anchor"),
        "expected view_storage_anchor binding in fold_kill_count kernel:\n{fold_wgsl}"
    );
}
