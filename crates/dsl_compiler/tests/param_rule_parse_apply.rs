// crates/dsl_compiler/tests/param_rule_parse_apply.rs
use dsl_compiler::parse;
use dsl_ast::ast::{Decl, ApplyArgValue};

fn first_apply(src: &str) -> dsl_ast::ast::PhysicsApplyDecl {
    let program = parse(src).expect("parse");
    program.decls.into_iter().find_map(|d| match d {
        Decl::PhysicsApply(a) => Some(a),
        _ => None,
    }).expect("apply decl present")
}

#[test]
fn parses_basic_apply() {
    let src = "physics HunterChase = chase(target: Wolf, aggro: 15.0, speed: 1.5);\n";
    let a = first_apply(src);
    assert_eq!(a.name, "HunterChase");
    assert_eq!(a.template, "chase");
    assert_eq!(a.args.len(), 3);
    assert_eq!(a.args[0].name, "target");
    matches!(a.args[0].value, ApplyArgValue::EntityKind(ref s) if s == "Wolf");
    assert_eq!(a.args[1].name, "aggro");
    matches!(a.args[1].value, ApplyArgValue::F32(15.0));
    assert_eq!(a.args[2].name, "speed");
    matches!(a.args[2].value, ApplyArgValue::F32(1.5));
}

#[test]
fn parses_apply_with_bool_and_int_args() {
    let src = "physics R = thing(flag: true, count: 42);\n";
    let a = first_apply(src);
    assert!(matches!(a.args[0].value, ApplyArgValue::Bool(true)));
    // 42 may parse as either I32 or U32; accept either.
    assert!(matches!(a.args[1].value, ApplyArgValue::I32(42) | ApplyArgValue::U32(42)));
}

#[test]
fn rejects_apply_with_no_semicolon() {
    let src = "physics HunterChase = chase(target: Wolf, aggro: 15.0)\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains(";"), "got: {msg}");
}

#[test]
fn rejects_apply_with_positional_arg() {
    // v1 is by-name only; bare expressions without `name:` are rejected.
    let src = "physics R = chase(Wolf, 15.0, 1.5);\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    // Parser will fail looking for `:` after `Wolf`. Error must mention `:` or "named".
    assert!(msg.contains(":") || msg.contains("name"), "got: {msg}");
}
