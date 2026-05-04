//! Stage 7/8 lock-in: predator_prey_min.sim's `ChronicleDeath`
//! PerEvent physics rule lives on the host side (`@cpu_only` tag, set
//! at Stage 8 so the String `utterance` payload is allowed). The rule
//! resolves cleanly + the DeathCry event is registered, but no GPU
//! kernel emits for it (cpu_only rules are filtered out of the CG
//! lowering driver — `lower_all_physics` skips them).

#[test]
fn predator_prey_min_chronicle_death_emits_kernel() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    // Both the rule and the DeathCry event resolved.
    assert!(
        comp.physics.iter().any(|p| p.name == "ChronicleDeath"),
        "ChronicleDeath physics rule should resolve"
    );
    assert!(
        comp.events.iter().any(|e| e.name == "DeathCry"),
        "DeathCry event should resolve"
    );

    // Stage 8 made ChronicleDeath @cpu_only — its body emits a
    // String chronicle, so no GPU kernel lowers. Verify the CG
    // pipeline still emits cleanly (the rule's existence shouldn't
    // poison the GPU emit) and that no `physics_ChronicleDeath`
    // kernel appears in the artifact index.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let arts = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
    assert!(
        arts.kernel_index.iter().all(|n| n != "physics_ChronicleDeath"),
        "ChronicleDeath is @cpu_only — should NOT appear in GPU kernel index; got {:?}",
        arts.kernel_index
    );

    // Confirm the cpu_only flag is set on the resolved rule.
    let rule = comp
        .physics
        .iter()
        .find(|p| p.name == "ChronicleDeath")
        .expect("ChronicleDeath physics rule");
    assert!(rule.cpu_only, "ChronicleDeath should be cpu_only");
}
