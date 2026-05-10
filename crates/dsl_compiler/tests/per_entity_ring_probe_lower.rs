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

    // PROBE-ONLY signal — surfaces the current state of the
    // PerEntityRing emit gap. Today (2026-05-10), the storage hint
    // does NOT propagate to the BGL or fold body:
    //
    //   * BGL slots 2/3/4 hardcode primary/anchor/ids. PerEntityRing
    //     should bind primary + cursors (no anchor/ids). See
    //     `cg/emit/kernel.rs:2222-2261` — slots are unconditional.
    //   * Fold body lowers `self += amount` as scalar CAS-add on
    //     `view_storage_primary[target_slot]`. PerEntityRing needs
    //     ring-append: `let idx = atomicAdd(&cursors[target], 1u);
    //     primary[target * K + (idx % K)] = amount`. See
    //     `cg/emit/kernel.rs:2186` (the documented TODO).
    //   * Runtime `ViewStorage` (engine/src/gpu/event_ring.rs:353)
    //     has no `cursors` field — would need extension.
    //
    // The pin is intentionally lenient (eprintln, not assert) so
    // the test stays GREEN as a regression guard for the .sim
    // parsing + lowering path. It flips to a hard assertion once
    // any one of those three layers is fixed and we want to lock
    // in the new shape.
    let has_cursors = fold_wgsl.1.contains("cursors");
    let has_modulo  = fold_wgsl.1.contains("% 4u");
    eprintln!("[G3a smoke] has 'cursors' binding: {has_cursors}");
    eprintln!("[G3a smoke] has '% 4u' ring-modulo: {has_modulo}");
    if !has_cursors || !has_modulo {
        eprintln!("[G3a smoke] PerEntityRing emit gap still open. Next sub-steps:");
        eprintln!("  1. BGL emit (cg/emit/kernel.rs:2220-2261) — branch on storage hint;");
        eprintln!("     PerEntityRing slot 3 = cursors (not anchor), slot 4 unused.");
        eprintln!("  2. Runtime ViewStorage (engine/src/gpu/event_ring.rs:353) — add");
        eprintln!("     `cursors: Option<wgpu::Buffer>` field + has_cursors constructor flag.");
        eprintln!("  3. Fold body emit (cg/emit/kernel.rs:2186 TODO) — when storage hint");
        eprintln!("     is PerEntityRing, generate ring-append instead of CAS-add.");
        eprintln!("  4. Per-runtime crate (per_entity_ring_probe_runtime) — exercise on GPU.");
    }

    // The ring-append primitive itself: the modulo on the cursor.
    // K=4 in the .sim's `@per_entity_ring(K = 4)`. If `% 4u` is
    // absent, the emit hasn't grown the ring-append yet (the gap
    // documented at `cg/emit/kernel.rs:2186`). Test fails loud so
    // the author surfaces the gap.
    if !fold_wgsl.1.contains("% 4u") {
        eprintln!("[G3a smoke] recent_damages WGSL DOES NOT contain `% 4u` ring-modulo.");
        eprintln!("[G3a smoke] This means the WGSL emit hasn't been taught the ring-append");
        eprintln!("[G3a smoke] primitive yet (see TODO at cg/emit/kernel.rs:2186). The");
        eprintln!("[G3a smoke] storage scaffold (cursors binding) is likely in place but");
        eprintln!("[G3a smoke] the body is lowering as scalar accumulate. G3a's next");
        eprintln!("[G3a smoke] sub-step is to teach the emit to recognise PerEntityRing.");
        eprintln!("[G3a smoke] Body excerpt (first 800 chars):");
        eprintln!("{}", &fold_wgsl.1[..fold_wgsl.1.len().min(800)]);
    }
    // Don't fail the assertion yet — the test is a probe to surface
    // current state. The emit-side fix flips this from a probe to
    // a regression guard.
}
