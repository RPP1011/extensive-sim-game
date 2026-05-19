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

use dsl_ast::ast::ExprKind;

#[test]
fn monomorphise_substitutes_param_refs_in_body() {
    // The template body uses `aggro` and `target` directly. After
    // monomorphisation, the concrete rule's body should contain literal
    // 15.0 (for aggro) and Ident("Wolf") (for target), NOT the original
    // Ident("aggro") / Ident("target").
    let src = r#"
entity Wolf : Agent {}

physics chase(target: EntityKind, aggro: f32) @phase(per_agent) {
  on Tick {} {
    let aggro_squared = aggro * aggro;
    let target_kind = target;
  }
}

physics HunterChase = chase(target: Wolf, aggro: 15.0);
"#;
    let mut program = parse(src).expect("parse");
    monomorphise(&mut program).expect("ok");

    let hunter = program.decls.iter().find_map(|d| match d {
        Decl::Physics(p) if p.name == "HunterChase" => Some(p),
        _ => None,
    }).expect("HunterChase should exist");

    let body = &hunter.handlers[0].body;
    // First stmt: let aggro_squared = aggro * aggro
    let first = &body[0];
    match first {
        dsl_ast::ast::Stmt::Let { name, value, .. } => {
            assert_eq!(name, "aggro_squared");
            match &value.kind {
                ExprKind::Binary { lhs, rhs, .. } => {
                    assert!(matches!(lhs.kind, ExprKind::Float(15.0)),
                            "lhs should be Float(15.0), got {:?}", lhs.kind);
                    assert!(matches!(rhs.kind, ExprKind::Float(15.0)),
                            "rhs should be Float(15.0), got {:?}", rhs.kind);
                }
                other => panic!("expected Binary, got {other:?}"),
            }
        }
        other => panic!("expected first stmt to be Let, got {other:?}"),
    }
    // Second stmt: let target_kind = target → Ident("Wolf").
    let second = &body[1];
    match second {
        dsl_ast::ast::Stmt::Let { name, value, .. } => {
            assert_eq!(name, "target_kind");
            match &value.kind {
                ExprKind::Ident(s) => assert_eq!(s, "Wolf"),
                other => panic!("expected Ident, got {other:?}"),
            }
        }
        other => panic!("expected second stmt to be Let, got {other:?}"),
    }
}
