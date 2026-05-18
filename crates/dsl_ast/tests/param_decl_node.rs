#[allow(unused_imports)]
use dsl_ast::ast::{ParamDecl, ParamType, PhysicsDecl, PhysicsHandler, PhysicsPattern};

#[test]
fn param_decl_construct() {
    let pd = ParamDecl {
        name: "aggro".into(),
        ty: ParamType::F32,
        span: Default::default(),
    };
    assert_eq!(pd.name, "aggro");
    assert!(matches!(pd.ty, ParamType::F32));
}

#[test]
fn param_type_variants_all_present() {
    let _ = ParamType::F32;
    let _ = ParamType::I32;
    let _ = ParamType::U32;
    let _ = ParamType::Bool;
    let _ = ParamType::EntityKind;
}

#[test]
fn physics_decl_grows_params_defaulting_empty() {
    let p = PhysicsDecl {
        annotations: vec![],
        name: "Foo".into(),
        params: vec![],
        handlers: vec![],
        cpu_only: false,
        span: Default::default(),
    };
    assert!(p.params.is_empty());
}
