use dsl_compiler::parse;
use dsl_compiler::lower::param_rules::monomorphise;
use dsl_ast::ast::Decl;

#[test]
fn application_produces_one_concrete_rule() {
    let src = r#"
entity Wolf : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {}
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
"#;
    let mut program = parse(src).expect("parse");
    monomorphise(&mut program).expect("ok");

    let applies: Vec<&dsl_ast::ast::PhysicsApplyDecl> = program.decls.iter()
        .filter_map(|d| match d { Decl::PhysicsApply(a) => Some(a), _ => None })
        .collect();
    assert!(applies.is_empty(), "all applies should be removed after mono");

    let physics_decls: Vec<&dsl_ast::ast::PhysicsDecl> = program.decls.iter()
        .filter_map(|d| match d { Decl::Physics(p) => Some(p), _ => None })
        .collect();
    let hunter = physics_decls.iter().find(|p| p.name == "HunterChase")
        .expect("HunterChase should be emitted");
    assert!(hunter.params.is_empty(), "monomorphised rule has no params");
    assert_eq!(hunter.handlers.len(), 1, "body preserved");
}

#[test]
fn two_applications_produce_two_distinct_concrete_rules() {
    let src = r#"
entity Wolf : Agent {}
entity Sheep : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {}
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
physics WolfChase   = chase(target: Sheep, aggro: 8.0);
"#;
    let mut program = parse(src).expect("parse");
    monomorphise(&mut program).expect("ok");

    let physics_decls: Vec<&dsl_ast::ast::PhysicsDecl> = program.decls.iter()
        .filter_map(|d| match d { Decl::Physics(p) => Some(p), _ => None })
        .collect();
    let names: Vec<&str> = physics_decls.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"HunterChase"));
    assert!(names.contains(&"WolfChase"));
}

#[test]
fn parameterised_template_removed_after_mono() {
    let src = r#"
entity Wolf : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {}
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
"#;
    let mut program = parse(src).expect("parse");
    monomorphise(&mut program).expect("ok");

    // After monomorphisation:
    //  - HunterChase exists as a concrete Physics decl.
    //  - chase (the template) is REMOVED from decls.
    let physics_decls: Vec<&dsl_ast::ast::PhysicsDecl> = program.decls.iter()
        .filter_map(|d| match d { Decl::Physics(p) => Some(p), _ => None })
        .collect();
    let names: Vec<&str> = physics_decls.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"HunterChase"), "concrete rule present");
    assert!(!names.contains(&"chase"), "template should be stripped after mono; got: {:?}", names);
}
