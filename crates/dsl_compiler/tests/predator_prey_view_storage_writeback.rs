//! Phase 8 lock-in: the view-fold storage write-back WGSL emit.
//!
//! Pins the serial PerAgent scan shape for f32+Add view folds:
//!
//! ```text
//! let observer_slot = gid.x;
//! if (observer_slot >= cfg.agent_cap) { return; }
//! var accum: f32 = bitcast<f32>(atomicLoad(&view_storage_primary[observer_slot]));
//! for (var _ei = 0u; _ei < _ec; _ei = _ei + 1u) {
//!     if (event_ring[_ei * stride + 0u] == <kind>) {
//!         let local_0: u32 = event_ring[_ei * stride + <field_off>];
//!         if (local_0 == observer_slot) {
//!             accum = accum + (1.0);
//!         }
//!     }
//! }
//! atomicStore(&view_storage_primary[observer_slot], bitcast<u32>(accum));
//! ```
//!
//! After sort, events for each target slot are contiguous; one thread per
//! observer slot owns that slot, scans all events, and commits with a single
//! atomicStore.  No CAS retry loop, no multi-writer races.
//!
//! Also pins the BGL declaration: `view_storage_primary` must be
//! declared `array<atomic<u32>>` so the atomic ops type-check under naga.

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

    // Serial PerAgent scan: one thread per observer slot.
    assert!(
        fold_wgsl.contains("let observer_slot = gid.x;"),
        "fold_kill_count should use observer_slot as the outer index — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("cfg.agent_cap"),
        "fold_kill_count should bound on cfg.agent_cap — got:\n{fold_wgsl}"
    );
    assert!(
        fold_wgsl.contains("var accum"),
        "fold_kill_count should use a local accumulator — got:\n{fold_wgsl}"
    );

    // Inner loop over events with the event-ring index variable `_ei`.
    assert!(
        fold_wgsl.contains("for (var _ei = 0u;"),
        "fold_kill_count should loop over events with _ei — got:\n{fold_wgsl}"
    );

    // The condition check compares target slot against observer_slot.
    assert!(
        fold_wgsl.contains("== observer_slot"),
        "fold_kill_count inner body should filter by target == observer_slot — got:\n{fold_wgsl}"
    );

    // Commit with a plain atomicStore (not CAS).
    assert!(
        fold_wgsl.contains("atomicStore(&view_storage_primary[observer_slot]"),
        "fold_kill_count should commit with atomicStore on view_storage_primary[observer_slot] — got:\n{fold_wgsl}"
    );

    // No CAS retry loop on view_storage_primary.
    assert!(
        !fold_wgsl.contains("atomicCompareExchangeWeak(&view_storage_primary"),
        "fold_kill_count must NOT use CAS on view_storage_primary (serial scan replaced it) — got:\n{fold_wgsl}"
    );

    // BGL declaration must be atomic so the atomic ops type-check.
    assert!(
        fold_wgsl.contains("var<storage, read_write> view_storage_primary: array<atomic<u32>>;"),
        "view_storage_primary binding must be array<atomic<u32>> — got:\n{fold_wgsl}"
    );

    // No phony discard left over.
    assert!(
        !fold_wgsl.contains("_ = (1.0);"),
        "fold_kill_count should not emit the phony discard placeholder — got:\n{fold_wgsl}"
    );
}
