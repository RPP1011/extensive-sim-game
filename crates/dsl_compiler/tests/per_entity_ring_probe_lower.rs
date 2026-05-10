//! Plan G G3a — smoke test for `assets/sim/per_entity_ring_probe.sim`.
//!
//! Lights up the existing-but-unused `@per_entity_ring(K = N)` view
//! storage hint by lowering a fixture that uses it on a scalar
//! payload. The test asserts:
//!
//! 1. The fixture parses + resolves + lowers (no errors).
//! 2. The expected kernels emit (InjectDamage + the recent_damages
//!    view fold).
//! 3. The view's fold WGSL contains the ring-append primitive
//!    (`atomicAdd` on a cursors-slot binding + indexed write to
//!    `view_*_primary` at `target * K + (idx % K)`). This pin
//!    catches the WGSL emit at `cg/emit/kernel.rs:2186` either:
//!    (a) producing scalar-accumulate output (silent miscompile —
//!    test fails so author surfaces the gap), or
//!    (b) producing ring-append output (G3a goal — test passes).
//!
//! The gating substring `% 4u` (= K=4 modulo) is the simplest
//! signal that the emit is ring-aware. Without G3a's WGSL emit
//! change, the body lowers as `view_*_primary[target] = ...` which
//! has no `% 4u` substring and the test fails loudly.

#[test]
fn per_entity_ring_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/per_entity_ring_probe.sim")
        .expect("read assets/sim/per_entity_ring_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse per_entity_ring_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve per_entity_ring_probe.sim");

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

    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg, dsl_compiler::cg::schedule::ScheduleStrategy::Default);
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit per_entity_ring_probe");

    assert!(!artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none");

    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();
    for expected in ["InjectDamage", "recent_damages"] {
        assert!(kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}");
    }

    // Inspect the recent_damages fold WGSL for ring-append signals.
    // The smallest signal that the emit knows about K=4 is `% 4u`
    // (the modulo on the cursor index). Without G3a's emit change,
    // the body lowers as scalar accumulate and the substring is
    // absent.
    let fold_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("recent_damages") && name.ends_with(".wgsl"))
        .map(|(name, body)| (name.as_str(), body.as_str()))
        .expect("recent_damages fold kernel must emit a .wgsl artifact");

    eprintln!("[G3a smoke] inspecting {}", fold_wgsl.0);
    eprintln!("[G3a smoke] body length: {} bytes", fold_wgsl.1.len());

    // Plan G G3a-step-2 — ring-append emit landed. The fold body
    // hand-emits the ring-append shape when the view's storage hint
    // is PerEntityRing { k }:
    //
    //   let cursor_idx = atomicAdd(&view_storage_anchor[target], 1u);
    //   let ring_idx = target * 4u + (cursor_idx % 4u);
    //   atomicStore(&view_storage_primary[ring_idx], amount_bits);
    //
    // The slot 3 binding is reused as the cursors counter (renamed
    // semantically; the binding still says `view_storage_anchor` so
    // the existing per-runtime bindings struct field name is
    // preserved). For PerEntityRing the WGSL declares slot 3 as
    // `array<atomic<u32>>` so `atomicAdd` is well-typed.
    //
    // These are HARD assertions — a regression that drops the
    // ring-append primitive surfaces here as a test failure.
    assert!(fold_wgsl.1.contains("atomicAdd(&view_storage_anchor"),
        "PerEntityRing emit must atomicAdd the cursors slot to allocate \
         a ring index. WGSL did not contain the substring.\n\nWGSL:\n{}",
         fold_wgsl.1);
    assert!(fold_wgsl.1.contains("% 4u"),
        "PerEntityRing emit must compute `ring_idx = target_slot * K + (cursor_idx % K)`. \
         WGSL did not contain `% 4u` (K=4 from the .sim's @per_entity_ring(K = 4)).\n\nWGSL:\n{}",
         fold_wgsl.1);
    assert!(fold_wgsl.1.contains("view_storage_primary[ring_idx] = amount_bits"),
        "PerEntityRing emit must store amount_bits at the computed ring_idx. \
         WGSL did not contain the substring.\n\nWGSL:\n{}",
         fold_wgsl.1);

    // Confirm the BGL declared slot 3 as an atomic array (required for
    // atomicAdd). Substring check against the WGSL declaration.
    assert!(fold_wgsl.1.contains("view_storage_anchor: array<atomic<u32>>"),
        "PerEntityRing BGL must declare slot 3 as `array<atomic<u32>>`. \
         WGSL did not match.\n\nWGSL:\n{}",
         fold_wgsl.1);

    // Negative pin: the CAS-add scalar accumulator should NOT be in the
    // body for a PerEntityRing view (the ring-append emit replaces it).
    assert!(!fold_wgsl.1.contains("atomicCompareExchangeWeak"),
        "PerEntityRing emit must NOT fall back to the scalar CAS-add \
         (CAS is for non-ring views). WGSL contained `atomicCompareExchangeWeak`.\n\nWGSL:\n{}",
         fold_wgsl.1);
}
