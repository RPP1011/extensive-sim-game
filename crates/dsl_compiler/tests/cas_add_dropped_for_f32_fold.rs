//! Pin: f32 view-fold kernels use a serial PerAgent scan (local accumulator
//! + single atomicStore) rather than a CAS+add retry loop on
//! view_storage_primary.
//!
//! After sort, events for each target slot appear contiguously in the ring.
//! The serial scan assigns one thread per observer slot; that thread iterates
//! all events, filters by target == observer_slot, and accumulates locally.
//! Because each thread is the sole writer for its slot there is no
//! contention and no CAS retry.

fn compile_forest_fire() -> dsl_compiler::cg::emit::EmittedArtifacts {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/forest_fire.sim");
    let src = std::fs::read_to_string(&path).expect("read forest_fire.sim");
    let prog = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

#[test]
fn forest_fire_f32_fold_uses_serial_sum_not_cas_add() {
    let art = compile_forest_fire();

    // Collect all ViewFold kernel WGSL bodies that write to view_storage_primary.
    let candidates: Vec<(&str, &str)> = art
        .wgsl_files
        .iter()
        .filter(|(name, _)| name.contains("fold_"))
        .filter(|(_, body)| body.contains("view_storage_primary"))
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .collect();

    assert!(
        !candidates.is_empty(),
        "expected at least one fold kernel emitted for forest_fire; kernel_index={:?}",
        art.kernel_index,
    );

    for (name, body) in &candidates {
        // The serial scan path must not emit a CAS retry loop on
        // view_storage_primary — that would be the old racy multi-writer path.
        assert!(
            !body.contains("atomicCompareExchangeWeak(&view_storage_primary"),
            "fold kernel {name} must NOT use CAS on view_storage_primary; \
             found CAS retry loop:\n{}",
            &body[..body.len().min(800)],
        );

        // The serial scan emits a local accumulator variable.
        let has_accum = body.contains("var accum")
            || body.contains("var sum")
            || body.contains("var total");
        assert!(
            has_accum,
            "fold kernel {name} must use a local accumulator (e.g. `var accum`); \
             body excerpt:\n{}",
            &body[..body.len().min(800)],
        );

        // The serial scan writes back with a plain atomicStore (not CAS).
        assert!(
            body.contains("atomicStore(&view_storage_primary"),
            "fold kernel {name} must commit with atomicStore on view_storage_primary; \
             body excerpt:\n{}",
            &body[..body.len().min(800)],
        );

        // The serial scan bounds-check uses cfg.agent_cap, not cfg.event_count,
        // since the outer index is observer_slot (not event_idx).
        assert!(
            body.contains("cfg.agent_cap"),
            "fold kernel {name} must bound on cfg.agent_cap (observer_slot check); \
             body excerpt:\n{}",
            &body[..body.len().min(800)],
        );
    }
}
