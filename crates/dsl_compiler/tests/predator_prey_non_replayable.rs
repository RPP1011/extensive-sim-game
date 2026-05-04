//! Stage 8 lock-in: predator_prey_min.sim's DeathCry event carries
//! `@non_replayable @traced` and a `String` payload field; the
//! ChronicleDeath rule that emits it lives @cpu_only so the host
//! runtime walks `comp.physics` to dispatch it. The GPU lowering
//! filters cpu_only rules out of `lower_all_physics`.

#[test]
fn predator_prey_min_chronicle_routes_through_cpu_only() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim/predator_prey_min.sim");
    let src = std::fs::read_to_string(&path).expect("read predator_prey_min.sim");
    let program = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");

    let death_cry = comp
        .events
        .iter()
        .find(|e| e.name == "DeathCry")
        .expect("DeathCry event resolved");
    let names: Vec<&str> = death_cry
        .annotations
        .iter()
        .map(|a| a.name.as_str())
        .collect();
    assert!(
        names.contains(&"non_replayable"),
        "DeathCry should carry @non_replayable: {:?}",
        names
    );
    assert!(
        names.contains(&"traced"),
        "DeathCry should carry @traced: {:?}",
        names
    );

    let field_names: Vec<&str> = death_cry.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        field_names.contains(&"utterance"),
        "DeathCry should carry the String utterance field: {:?}",
        field_names
    );

    // ChronicleDeath rule resolves with cpu_only set.
    let rule = comp
        .physics
        .iter()
        .find(|p| p.name == "ChronicleDeath")
        .expect("ChronicleDeath physics rule resolved");
    assert!(
        rule.cpu_only,
        "ChronicleDeath should be cpu_only — its body emits the String payload"
    );
}
