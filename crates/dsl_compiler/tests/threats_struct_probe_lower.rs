//! Plan G G3 final composition — smoke test for
//! `assets/sim/threats_struct_probe.sim`.
//!
//! Closes the design-doc gap: struct-payload `ThreatZoneCell` ring keyed
//! per-observer, populated by the PerAgentEventScan dispatch. Composes
//! G3a/b/c (struct-payload PerEntityRing + multi-stmt fold body) with
//! G3d (PerAgentEventScan dispatch).
//!
//! Asserts:
//!   1. The .sim parses + resolves + lowers (no errors).
//!   2. The view's struct-cell layout (8 fields) is registered.
//!   3. The fold WGSL has BOTH the PerAgentEventScan preamble AND the
//!      struct-cell ring-append shape:
//!      - `let observer = gid.x` + `let source_candidate = gid.y`
//!      - busy-filter early-exit
//!      - `atomicAdd(&view_storage_anchor[observer], 1u)` for cursor
//!      - per-field stores at `ring_idx * 8u + N` for N in 0..8.

#[test]
fn threats_struct_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/threats_struct_probe.sim")
        .expect("read assets/sim/threats_struct_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse threats_struct_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve threats_struct_probe.sim");

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
    assert!(
        lower_diags.is_empty(),
        "expected clean lowering; got diagnostics: {lower_diags:#?}"
    );

    // ---- ViewLayout: 8-field threat zone cell registered.
    assert!(
        !cg.view_layouts.is_empty(),
        "expected at least one registered ViewLayout for the struct-cell view"
    );
    let layout = cg
        .view_layouts
        .values()
        .next()
        .expect("at least one ViewLayout registered");
    let names: Vec<&str> = layout.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        layout.fields.len(),
        8,
        "expected 8 fields per design doc; got {}: {:?}",
        layout.fields.len(),
        names,
    );
    let expected = [
        "zone_kind",
        "center_x_q8",
        "center_y_q8",
        "radius_q8",
        "dir_x_q8",
        "dir_y_q8",
        "expires_at_tick",
        "source",
    ];
    assert_eq!(names, expected, "field names must match design doc layout");
    assert_eq!(layout.cell_stride_u32(), 8);
    assert_eq!(layout.cell_size_bytes(), 32);

    // ---- Schedule + emit.
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit threats_struct_probe");

    assert!(
        !artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel"
    );
    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["MarkCasterBusy", "threats"] {
        assert!(
            kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}"
        );
    }

    let fold_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("threats") && name.ends_with(".wgsl"))
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .expect("threats fold kernel must emit a .wgsl artifact");

    eprintln!("[G3 final] inspecting {}", fold_wgsl.0);
    eprintln!("[G3 final] body length: {} bytes", fold_wgsl.1.len());
    eprintln!("[G3 final] body:\n{}", fold_wgsl.1);

    // PerAgentEventScan preamble.
    assert!(
        fold_wgsl.1.contains("let observer = gid.x"),
        "must declare observer; WGSL:\n{}",
        fold_wgsl.1
    );
    assert!(
        fold_wgsl.1.contains("let source_candidate = gid.y"),
        "must declare source_candidate; WGSL:\n{}",
        fold_wgsl.1
    );
    assert!(
        fold_wgsl
            .1
            .contains("agent_busy_with_ability_id[source_candidate] == 0u"),
        "must early-exit non-busy candidates; WGSL:\n{}",
        fold_wgsl.1
    );

    // 2-D workgroup geometry.
    assert!(
        fold_wgsl.1.contains("@workgroup_size(8, 8)"),
        "PerAgentEventScan kernel must declare 2-D workgroup; WGSL:\n{}",
        fold_wgsl.1
    );

    // Struct-cell ring-append: cursor allocation keyed on OBSERVER (not
    // event ring's target_slot — there's no event ring read on this
    // dispatch shape).
    assert!(
        fold_wgsl
            .1
            .contains("atomicAdd(&view_storage_anchor[observer]"),
        "struct-cell ring-append must atomicAdd the observer's cursor; WGSL:\n{}",
        fold_wgsl.1
    );

    // K=4 modulo for ring slot wrap.
    assert!(
        fold_wgsl.1.contains("% 4u"),
        "must compute `ring_idx = observer * 4 + (cursor % 4)`; WGSL:\n{}",
        fold_wgsl.1
    );

    // Per-field stores at `ring_idx * 8u + N` for N in 0..8 (one per
    // ThreatZoneCell field).
    for n in 0..8 {
        let needle = format!("ring_idx * 8u + {n}u");
        assert!(
            fold_wgsl.1.contains(&needle),
            "must store field {n} at `ring_idx * 8u + {n}u`; WGSL:\n{}",
            fold_wgsl.1,
        );
    }

    // Negative pin: must NOT fall through to the scalar
    // `view_storage_primary[observer]` CAS+add (the existing
    // PerAgentEventScan-with-no-ViewLayout shape).
    assert!(
        !fold_wgsl
            .1
            .contains("atomicCompareExchangeWeak(&view_storage_primary[observer]"),
        "struct-cell PerAgentEventScan must NOT emit the scalar CAS+add; WGSL:\n{}",
        fold_wgsl.1
    );
}
