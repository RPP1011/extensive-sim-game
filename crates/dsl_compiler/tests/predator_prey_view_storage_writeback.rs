//! Phase 8 lock-in: the view-fold storage write-back WGSL emit.
//!
//! Pins the atomic CAS-loop accumulator-update shape:
//!
//! ```text
//! loop {
//!     let old = atomicLoad(&view_storage_<slot>[local_<N>]);
//!     let new_val = bitcast<u32>(bitcast<f32>(old) + (rhs));
//!     let result = atomicCompareExchangeWeak(&view_storage_<slot>[local_<N>], old, new_val);
//!     if (result.exchanged) { break; }
//! }
//! ```
//!
//! The per-row index is captured from the immediately-preceding
//! `Let local_<N>: AgentId = EventField(…)` in the fold body — the
//! binder-extraction pattern `on Killed { by: predator } { self +=
//! 1.0 }` produces. Without the capture the assign would fall back
//! to the phony `_ = (rhs);` discard.
//!
//! Catches regressions in either the EmitCtx capture wiring or the
//! ViewStorage assign emit. Replaces the prior non-atomic RMW which
//! lost increments under contention (B1 — P11 Reduction Determinism
//! violation surfaced by the ses_app stress fixture).
//!
//! Also pins the BGL declaration: `view_storage_primary` must be
//! declared `array<atomic<u32>>` so the CAS-loop's `atomicLoad` /
//! `atomicCompareExchangeWeak` calls type-check under naga.

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
    // event field). The Assign-to-ViewStorage then writes through a
    // CAS loop indexed by `_idx = local_0` (single-key shape — the
    // PairMap gap fix added the `_idx` binding so the 2-D pair_map
    // path can compose `_idx = (local_<k1> * cfg.second_key_pop +
    // local_<k2>)`; single-key views like kill_count keep `_idx =
    // local_<single>`).
    assert!(
        fold_wgsl.contains("let _idx = local_0;"),
        "fold_kill_count should bind the row index `_idx = local_0` for single-key shape — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("atomicLoad(&view_storage_primary[_idx])"),
        "fold_kill_count should emit atomicLoad on view_storage_primary[_idx] — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("atomicCompareExchangeWeak(&view_storage_primary[_idx]"),
        "fold_kill_count should emit atomicCompareExchangeWeak on view_storage_primary[_idx] — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("bitcast<u32>(bitcast<f32>(old) + (1.0))"),
        "fold_kill_count should bitcast through f32 for the +1.0 add — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("if (result.exchanged) { break; }"),
        "fold_kill_count CAS loop should retry until atomicCompareExchangeWeak succeeds — got:\n{fold_wgsl}"
    );

    // BGL declaration must be atomic so the CAS-loop's atomic ops
    // type-check. Lock the WGSL binding line.
    assert!(
        fold_wgsl.contains("var<storage, read_write> view_storage_primary: array<atomic<u32>>;"),
        "view_storage_primary binding must be array<atomic<u32>> — got:\n{fold_wgsl}"
    );

    // No phony discard left over (would indicate the per-row capture
    // didn't fire for this body).
    assert!(
        !fold_wgsl.contains("_ = (1.0);"),
        "fold_kill_count should not emit the phony discard placeholder — got:\n{fold_wgsl}"
    );
}
