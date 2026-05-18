use dsl_ast::ast::{Decl, PhysicsApplyDecl, ApplyArg, ApplyArgValue};

#[test]
fn physics_apply_construct() {
    let apply = PhysicsApplyDecl {
        annotations: vec![],
        name: "HunterChase".into(),
        template: "chase".into(),
        args: vec![
            ApplyArg {
                name: "target".into(),
                value: ApplyArgValue::EntityKind("Wolf".into()),
                span: Default::default(),
            },
            ApplyArg {
                name: "aggro".into(),
                value: ApplyArgValue::F32(15.0),
                span: Default::default(),
            },
        ],
        span: Default::default(),
    };
    assert_eq!(apply.name, "HunterChase");
    assert_eq!(apply.template, "chase");
    assert_eq!(apply.args.len(), 2);
}

#[test]
fn decl_variant_physics_apply_exists() {
    let apply = PhysicsApplyDecl {
        annotations: vec![],
        name: "X".into(),
        template: "y".into(),
        args: vec![],
        span: Default::default(),
    };
    let _decl = Decl::PhysicsApply(apply);
}

#[test]
fn apply_arg_value_variants_all_present() {
    let _ = ApplyArgValue::F32(1.0);
    let _ = ApplyArgValue::I32(1);
    let _ = ApplyArgValue::U32(1);
    let _ = ApplyArgValue::Bool(true);
    let _ = ApplyArgValue::EntityKind("Wolf".into());
}
