//! Plan G G3b/G3c — smoke test for `assets/sim/per_entity_ring_struct_probe.sim`.
//!
//! Lights up the struct-payload extension to the existing
//! `@per_entity_ring(K = N)` view shape (G3a's scalar-payload
//! ring-append), plus the multi-statement-fold-body admission (G3c).
//! The test asserts:
//!
//! 1. The fixture parses + resolves + lowers (no errors). The
//!    `let now = world.tick` binding inside the fold body must be
//!    accepted (the prior "Unsupported(Let)" gate is lifted).
//! 2. The `recent_damage_records` view's struct-cell layout is
//!    registered on the program (`prog.view_layouts`) with the
//!    expected (timestamp, amount) field shape.
//! 3. The view's fold WGSL contains per-field stores at indices
//!    `ring_idx * field_count + field_idx`, NOT the single-field
//!    `view_storage_primary[ring_idx] = amount_bits` shape.
//! 4. The auto-walker recorded both `view_storage_primary` and
//!    `view_storage_anchor` (cursors) writes — the same ring-append
//!    primitive surface as the scalar shape.

#[test]
fn per_entity_ring_struct_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/per_entity_ring_struct_probe.sim")
        .expect("read assets/sim/per_entity_ring_struct_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse per_entity_ring_struct_probe.sim");
    let comp = dsl_ast::resolve::resolve(program)
        .expect("resolve per_entity_ring_struct_probe.sim");

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

    // ---- ViewLayout assertions (G3b core) ----
    //
    // The lowering pass should have registered the struct-cell layout
    // for `recent_damage_records` with two fields in declaration order:
    // `(timestamp: u32, amount: f32)`. The fixture uses bit-32 types
    // for both fields so the cell stride is exactly 2 u32 words.
    assert!(
        !cg.view_layouts.is_empty(),
        "expected at least one registered ViewLayout for the struct-cell view, \
         got an empty `view_layouts` map"
    );
    let layout = cg
        .view_layouts
        .values()
        .next()
        .expect("at least one ViewLayout registered");
    assert_eq!(
        layout.fields.len(),
        2,
        "expected 2 fields (timestamp, amount); got {}: {:?}",
        layout.fields.len(),
        layout.fields
    );
    assert_eq!(layout.fields[0].name, "timestamp");
    assert_eq!(layout.fields[1].name, "amount");
    assert_eq!(layout.cell_stride_u32(), 2);
    assert_eq!(layout.cell_size_bytes(), 8);

    // ---- Schedule + emit ----
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit per_entity_ring_struct_probe");

    assert!(
        !artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none"
    );
    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["InjectDamage", "recent_damage_records"] {
        assert!(
            kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}"
        );
    }

    // ---- WGSL emit assertions (G3b struct-payload path) ----
    let fold_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("recent_damage_records") && name.ends_with(".wgsl"))
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .expect("recent_damage_records fold kernel must emit a .wgsl artifact");

    eprintln!("[G3b struct] inspecting {}", fold_wgsl.0);
    eprintln!("[G3b struct] body length: {} bytes", fold_wgsl.1.len());
    eprintln!("[G3b struct] body:\n{}", fold_wgsl.1);

    // The cursor allocation primitive is shared with G3a — same
    // `atomicAdd(&view_storage_anchor[target_slot], 1u)` surface.
    assert!(
        fold_wgsl.1.contains("atomicAdd(&view_storage_anchor"),
        "PerEntityRing struct emit must atomicAdd the cursors slot to allocate \
         a ring index. WGSL did not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // The K=4 modulo + per-field stride must both appear. The stride
    // is `field_count = 2` (timestamp + amount).
    assert!(
        fold_wgsl.1.contains("% 4u"),
        "PerEntityRing emit must compute `ring_idx = target_slot * K + (cursor_idx % K)`. \
         WGSL did not contain `% 4u` (K=4 from the .sim's @per_entity_ring(K = 4)).\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // Per-field store at index `ring_idx * field_count + field_idx`.
    // The first field (timestamp, idx=0) writes at `ring_idx * 2u + 0u`;
    // the second field (amount, idx=1) writes at `ring_idx * 2u + 1u`.
    assert!(
        fold_wgsl.1.contains("ring_idx * 2u + 0u"),
        "struct-cell emit must store field 0 (timestamp) at \
         `view_storage_primary[ring_idx * field_count + 0]`. \
         WGSL did not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );
    assert!(
        fold_wgsl.1.contains("ring_idx * 2u + 1u"),
        "struct-cell emit must store field 1 (amount) at \
         `view_storage_primary[ring_idx * field_count + 1]`. \
         WGSL did not contain the substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // Negative pin: the SCALAR-shape store
    // (`view_storage_primary[ring_idx] = amount_bits`) should NOT be in
    // the body — that path only fires when `view_layouts` is empty.
    assert!(
        !fold_wgsl.1.contains("view_storage_primary[ring_idx] = amount_bits"),
        "struct-cell emit must NOT fall back to the scalar single-field shape; \
         WGSL contained the scalar `view_storage_primary[ring_idx] = amount_bits` \
         substring.\n\nWGSL:\n{}",
        fold_wgsl.1
    );

    // The `f32 → bitcast<u32>` conversion for the amount field (since
    // `view_storage_primary` is `array<u32>` for PerEntityRing).
    assert!(
        fold_wgsl.1.contains("bitcast<u32>"),
        "struct-cell emit must bitcast f32 fields to u32 for the primary \
         array store. WGSL did not contain `bitcast<u32>`.\n\nWGSL:\n{}",
        fold_wgsl.1
    );
}
