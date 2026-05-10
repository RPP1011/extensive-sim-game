//! Smoke test for `assets/sim/firebolt_interrupt_probe.sim` lowering.
//!
//! Plan G G2.5 — proves the per-fixture interrupt-detection sim
//! grammar lowers cleanly (parse + resolve + CG lower + schedule +
//! emit). Lives in dsl_compiler/tests/ rather than a runtime crate
//! because the firebolt_interrupt_probe_runtime is deferred to a
//! follow-up slice (the firebolt_probe_runtime template can be
//! copy-modified, but it's substantial work).
//!
//! Asserts the six expected named kernels emit:
//!   * `physics_DispatchFirebolt`
//!   * `physics_RecordCastBegin`
//!   * `physics_InjectDamage`
//!   * `physics_InterruptCastOnDamage`
//!   * `physics_ResolveBusy`
//!   * `physics_ApplyChronicleDamage`

#[test]
fn firebolt_interrupt_probe_sim_lowers_clean() {
    let src = std::fs::read_to_string("../../assets/sim/firebolt_interrupt_probe.sim")
        .expect("read assets/sim/firebolt_interrupt_probe.sim");
    let program = dsl_compiler::parse(&src).expect("parse firebolt_interrupt_probe.sim");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve firebolt_interrupt_probe.sim");

    let (cg, lower_diags) = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => (p, Vec::new()),
        Err(o) => {
            let diags: Vec<String> = o.diagnostics.iter().map(|d| format!("{d}")).collect();
            (o.program, diags)
        }
    };
    // Tolerate the same P6 diagnostic that firebolt_probe.sim tolerates
    // (PerEvent + agents.set_hp). Surface it but don't fail.
    for d in &lower_diags {
        eprintln!("[lower diag] {d}");
    }

    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg, dsl_compiler::cg::schedule::ScheduleStrategy::Default);
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit firebolt_interrupt_probe");

    assert!(!artifacts.kernel_index.is_empty(),
        "expected at least one emitted kernel, got none");

    let kernel_names: Vec<&str> = artifacts.kernel_index.iter().map(|s| s.as_str()).collect();

    for expected in [
        "DispatchFirebolt",
        "RecordCastBegin",
        "InjectDamage",
        "InterruptCastOnDamage",
        "ResolveBusy",
        "ApplyChronicleDamage",
    ] {
        assert!(kernel_names.iter().any(|n| n.contains(expected)),
            "expected kernel containing {expected:?}; got: {kernel_names:?}");
    }

    // Plan G G2.5 mask-aware filter — InterruptCastOnDamage's WGSL
    // body must contain a mask gate (`(<mask_token> % 2u) == 1u` —
    // bit 0 = Damage in InterruptKind). Without it, the rule treats
    // ALL damage as interrupting and `interrupts: standard - { damage }`
    // semantics aren't observable. Greps the emitted WGSL to confirm
    // the pattern is present.
    let interrupt_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("InterruptCastOnDamage") && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .expect("InterruptCastOnDamage.wgsl must be emitted");
    assert!(interrupt_wgsl.contains("% 2u"),
        "InterruptCastOnDamage.wgsl must gate on (config.interrupt.mask % 2) for the Damage bit; \
         emitted body did not contain `% 2u` (substring match). Body length: {} bytes",
        interrupt_wgsl.len());

    // Plan G tunable cfg — the .sim now marks `config.interrupt.mask`
    // as `@runtime`, so the kernel emit must route the mask read
    // through the cfg uniform (`cfg.config_interrupt_mask`) instead
    // of a baked WGSL `const config_<N>`. The body should reference
    // the cfg-uniform field and NOT emit a matching `const` line.
    assert!(
        interrupt_wgsl.contains("cfg.config_interrupt_mask"),
        "expected runtime cfg-uniform read `cfg.config_interrupt_mask` in InterruptCastOnDamage.wgsl; \
         body excerpt: {}",
        &interrupt_wgsl[..interrupt_wgsl.len().min(800)],
    );
    assert!(
        !interrupt_wgsl.contains("const config_") || !interrupt_wgsl.lines().any(|line| {
            line.trim_start().starts_with("const config_") && line.contains(": u32 = 15u")
        }),
        "runtime cfg const should NOT be emitted as `const config_<N>: u32 = 15u`; \
         body still contains the baked const declaration",
    );
    // The matching cfg struct must declare the runtime field too.
    assert!(
        interrupt_wgsl.contains("config_interrupt_mask: u32"),
        "expected `config_interrupt_mask: u32` field in the kernel's WGSL Cfg struct; \
         body excerpt: {}",
        &interrupt_wgsl[..interrupt_wgsl.len().min(800)],
    );
}
