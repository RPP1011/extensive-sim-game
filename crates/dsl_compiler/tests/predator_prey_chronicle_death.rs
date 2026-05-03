//! Stage 7 lock-in: predator_prey_min.sim's `ChronicleDeath` PerEvent
//! physics rule lowers + emits a kernel. The rule lacks a per-handler
//! `where` (so every Killed event triggers a DeathCry emit), and the
//! emit body's payload fields are all GPU-emittable (Stage 8 adds the
//! String `utterance` after the chronicle ring lands).

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

    // Schedule emits a `physics_ChronicleDeath` kernel.
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("CG lower");
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let arts = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg)
        .expect("emit predator_prey_min CG program");
    assert!(
        arts.kernel_index.iter().any(|n| n == "physics_ChronicleDeath"),
        "expected physics_ChronicleDeath kernel; got {:?}",
        arts.kernel_index
    );
}
