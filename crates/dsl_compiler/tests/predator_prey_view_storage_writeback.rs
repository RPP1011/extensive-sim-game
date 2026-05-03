//! Phase 8 lock-in: the view-fold storage write-back WGSL emit.
//!
//! Pins the `view_storage_<slot>[idx] = bitcast<u32>(bitcast<f32>(
//! load) + (rhs))` accumulator-update shape. The per-row index is
//! captured from the immediately-preceding `Let local_<N>: AgentId =
//! EventField(…)` in the fold body — the binder-extraction pattern
//! `on Killed { by: predator } { self += 1.0 }` produces. Without
//! the capture the assign would fall back to the phony `_ = (rhs);`
//! discard.
//!
//! Catches regressions in either the EmitCtx capture wiring or the
//! ViewStorage assign emit.

#[test]
fn predator_prey_kill_count_fold_emits_storage_writeback() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
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

    // The Let binder lowers to `local_0` (the by-id from the Killed
    // event field). The Assign-to-ViewStorage then writes through
    // `view_storage_primary[local_0] = bitcast<u32>(bitcast<f32>(
    // view_storage_primary[local_0]) + (1.0));`.
    assert!(
        fold_wgsl.contains("view_storage_primary[local_0] = bitcast<u32>(bitcast<f32>(view_storage_primary[local_0]) +"),
        "fold_kill_count should emit the indexed RMW with bitcast type-shim — got:\n{fold_wgsl}"
    );

    // No phony discard left over (would indicate the per-row capture
    // didn't fire for this body).
    assert!(
        !fold_wgsl.contains("_ = (1.0);"),
        "fold_kill_count should not emit the phony discard placeholder — got:\n{fold_wgsl}"
    );
}
