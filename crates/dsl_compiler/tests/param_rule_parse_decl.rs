use dsl_compiler::parse;
use dsl_ast::ast::{Decl, ParamType};

fn first_physics(src: &str) -> dsl_ast::ast::PhysicsDecl {
    let program = parse(src).expect("parse");
    program.decls.into_iter().find_map(|d| match d {
        Decl::Physics(p) => Some(p),
        _ => None,
    }).expect("physics decl present")
}

#[test]
fn parses_chase_with_three_params() {
    let src = r#"
physics chase(target: EntityKind, aggro: f32, speed: f32) @phase(per_agent) {
  on Tick {} { let _ = aggro; }
}
"#;
    let p = first_physics(src);
    assert_eq!(p.name, "chase");
    assert_eq!(p.params.len(), 3);
    assert_eq!(p.params[0].name, "target");
    assert!(matches!(p.params[0].ty, ParamType::EntityKind));
    assert_eq!(p.params[1].name, "aggro");
    assert!(matches!(p.params[1].ty, ParamType::F32));
    assert_eq!(p.params[2].name, "speed");
    assert!(matches!(p.params[2].ty, ParamType::F32));
}

#[test]
fn parses_zero_param_physics_unchanged() {
    // Existing form keeps working.
    let src = r#"
physics MoveBoid @phase(per_agent) {
  on Tick {} {}
}
"#;
    let p = first_physics(src);
    assert_eq!(p.name, "MoveBoid");
    assert!(p.params.is_empty());
}

#[test]
fn rejects_duplicate_param_name() {
    let src = r#"
physics foo(a: f32, a: f32) @phase(per_agent) {
  on Tick {} {}
}
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("duplicate") && msg.contains("a"), "got: {msg}");
}

#[test]
fn rejects_unknown_param_type() {
    let src = r#"
physics foo(x: SomeWeirdType) @phase(per_agent) {
  on Tick {} {}
}
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("type") || msg.contains("SomeWeirdType"), "got: {msg}");
}
