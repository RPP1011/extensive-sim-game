//! Wave 1.2 `.ability` parser surface tests for `template <Name>(<params>) {
//! ... }` top-level blocks AND `ability X : TemplateName(args) { ... }`
//! instantiation syntax (per `docs/spec/ability_dsl_unified.md` §11).
//!
//! Coverage:
//!  1. Minimal template (`template Empty() { }`).
//!  2. Untyped params (`template ElementalBolt(element, radius) { ... }`).
//!  3. Typed params (`template ElementalBolt(element: Material, radius: float = 3.0) { ... }`).
//!  4. Typed params with int / bool / string defaults.
//!  5. Template body using Wave 1.5 modifier slots.
//!  6. Ability instantiating a template — `ability Fireball : ElementalBolt(fire, 4.0) { ... }`.
//!  7. Template + ability coexisting in one file.
//!  8. Unknown parameter type is a clean parse error.
//!  9. Defensive: nested `template` inside `template` errors.
//! 10. Span coverage on TemplateDecl / TemplateInstantiation / TemplateParam.

use dsl_ast::ast::{TargetMode, TemplateArg, TemplateParamTy};
use dsl_ast::parse_ability_file;

// ---------------------------------------------------------------------------
// 1. Minimal template
// ---------------------------------------------------------------------------

#[test]
fn parses_minimal_template() {
    let src = "template Empty() { }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 0);
    assert_eq!(file.passives.len(), 0);
    assert_eq!(file.templates.len(), 1);
    let t = &file.templates[0];
    assert_eq!(t.name, "Empty");
    assert!(t.params.is_empty(), "no params");
    assert!(t.effects.is_empty(), "no body effects");
}

// ---------------------------------------------------------------------------
// 2. Untyped params
// ---------------------------------------------------------------------------

#[test]
fn parses_template_with_untyped_params() {
    let src = "template ElementalBolt(element, radius) { damage 50 }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.templates.len(), 1);
    let t = &file.templates[0];
    assert_eq!(t.name, "ElementalBolt");
    assert_eq!(t.params.len(), 2);
    assert_eq!(t.params[0].name, "element");
    assert!(t.params[0].ty.is_none(), "untyped param 0");
    assert!(t.params[0].default.is_none(), "no default param 0");
    assert_eq!(t.params[1].name, "radius");
    assert!(t.params[1].ty.is_none(), "untyped param 1");
    assert!(t.params[1].default.is_none(), "no default param 1");
    assert_eq!(t.effects.len(), 1);
    assert_eq!(t.effects[0].verb, "damage");
}

// ---------------------------------------------------------------------------
// 3. Typed params with mix of typed-no-default and typed-with-default
// ---------------------------------------------------------------------------

#[test]
fn parses_template_with_typed_params() {
    let src = "template ElementalBolt(element: Material, radius: float = 3.0) { damage 50 }";
    let file = parse_ability_file(src).expect("must parse");
    let t = &file.templates[0];
    assert_eq!(t.params.len(), 2);
    // param 0: element : Material, no default
    assert_eq!(t.params[0].name, "element");
    assert_eq!(t.params[0].ty, Some(TemplateParamTy::Material));
    assert!(t.params[0].default.is_none());
    // param 1: radius : float = 3.0
    assert_eq!(t.params[1].name, "radius");
    assert_eq!(t.params[1].ty, Some(TemplateParamTy::Float));
    match t.params[1].default {
        Some(TemplateArg::Number(v)) => assert!((v - 3.0).abs() < 1e-6),
        ref other => panic!("expected default Number(3.0); got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. Typed params with int / bool / string defaults
// ---------------------------------------------------------------------------

#[test]
fn parses_template_with_int_bool_defaults() {
    let src = r#"
template Mixed(
    count: int = 5,
    on_fire: bool = true,
    label: int = -1,
    name: Structure = castle
) {
    damage 10
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let t = &file.templates[0];
    assert_eq!(t.params.len(), 4);

    // count: int = 5
    assert_eq!(t.params[0].name, "count");
    assert_eq!(t.params[0].ty, Some(TemplateParamTy::Int));
    match t.params[0].default {
        Some(TemplateArg::Number(v)) => assert_eq!(v, 5.0),
        ref other => panic!("expected Number(5); got {other:?}"),
    }

    // on_fire: bool = true
    assert_eq!(t.params[1].name, "on_fire");
    assert_eq!(t.params[1].ty, Some(TemplateParamTy::Bool));
    match t.params[1].default {
        Some(TemplateArg::Bool(b)) => assert!(b),
        ref other => panic!("expected Bool(true); got {other:?}"),
    }

    // label: int = -1  (negative number default)
    assert_eq!(t.params[2].name, "label");
    assert_eq!(t.params[2].ty, Some(TemplateParamTy::Int));
    match t.params[2].default {
        Some(TemplateArg::Number(v)) => assert_eq!(v, -1.0),
        ref other => panic!("expected Number(-1); got {other:?}"),
    }

    // name: Structure = castle  (Ident default)
    assert_eq!(t.params[3].name, "name");
    assert_eq!(t.params[3].ty, Some(TemplateParamTy::Structure));
    match t.params[3].default {
        Some(TemplateArg::Ident(ref s)) => assert_eq!(s, "castle"),
        ref other => panic!("expected Ident(\"castle\"); got {other:?}"),
    }
}

#[test]
fn parses_template_param_with_bool_false_default() {
    // Belt-and-braces — `false` follows the same path as `true`.
    let src = "template T(flag: bool = false) { damage 1 }";
    let file = parse_ability_file(src).expect("must parse");
    match file.templates[0].params[0].default {
        Some(TemplateArg::Bool(b)) => assert!(!b),
        ref other => panic!("expected Bool(false); got {other:?}"),
    }
}

#[test]
fn parses_template_param_with_string_default() {
    // String literal defaults are part of the spec §11.1 grammar.
    let src = r#"template T(label = "fireball") { damage 1 }"#;
    let file = parse_ability_file(src).expect("must parse");
    match file.templates[0].params[0].default {
        Some(TemplateArg::String(ref s)) => assert_eq!(s, "fireball"),
        ref other => panic!("expected String(\"fireball\"); got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. Body uses Wave 1.5 modifier slots — full pipeline through parse_effect
// ---------------------------------------------------------------------------

#[test]
fn parses_template_body_with_modifiers() {
    let src = "template ElementalBolt(power: float) { damage 50 in circle(2.5) [PHYSICAL: 60] }";
    let file = parse_ability_file(src).expect("must parse");
    let t = &file.templates[0];
    assert_eq!(t.effects.len(), 1);
    let e = &t.effects[0];
    assert_eq!(e.verb, "damage");
    let area = e.area.as_ref().expect("area populated");
    assert_eq!(area.shape, "circle");
    assert_eq!(area.args.len(), 1);
    assert!((area.args[0] - 2.5).abs() < 1e-6);
    assert_eq!(e.tags.len(), 1);
    assert_eq!(e.tags[0].name, "PHYSICAL");
    assert!((e.tags[0].value - 60.0).abs() < 1e-6);
}

#[test]
fn parses_template_body_with_multiple_effects() {
    let src = r#"
template DamageBuff(amt: float = 10.0) {
    damage 25
    shield 10
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let t = &file.templates[0];
    assert_eq!(t.effects.len(), 2);
    assert_eq!(t.effects[0].verb, "damage");
    assert_eq!(t.effects[1].verb, "shield");
}

// ---------------------------------------------------------------------------
// 6. Ability instantiating a template
// ---------------------------------------------------------------------------

#[test]
fn parses_ability_instantiating_template() {
    let src = "ability Fireball : ElementalBolt(fire, 4.0) { target: ground range: 8.0 cooldown: 6s }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 1);
    let a = &file.abilities[0];
    assert_eq!(a.name, "Fireball");
    let inst = a.instantiates.as_ref().expect("instantiation populated");
    assert_eq!(inst.name, "ElementalBolt");
    assert_eq!(inst.args.len(), 2);
    match &inst.args[0] {
        TemplateArg::Ident(s) => assert_eq!(s, "fire"),
        other => panic!("expected Ident(\"fire\"); got {other:?}"),
    }
    match inst.args[1] {
        TemplateArg::Number(v) => assert!((v - 4.0).abs() < 1e-6),
        ref other => panic!("expected Number(4.0); got {other:?}"),
    }
    // Headers in the body still parse.
    assert!(a.headers.iter().any(|h| matches!(
        h,
        dsl_ast::ast::AbilityHeader::Target(TargetMode::Ground)
    )));
}

#[test]
fn parses_ability_instantiating_template_empty_body() {
    // Spec §11 example shows the ability body holding only headers; an
    // entirely empty body ought to also parse when the `:` clause
    // supplies the substance.
    let src = "ability Fireball : ElementalBolt(fire, 4.0) { }";
    let file = parse_ability_file(src).expect("must parse");
    let a = &file.abilities[0];
    assert!(a.instantiates.is_some());
    assert!(a.headers.is_empty());
    assert!(a.effects.is_empty());
}

#[test]
fn parses_ability_instantiating_template_empty_args() {
    let src = "ability X : Empty() { target: enemy cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("must parse");
    let inst = file.abilities[0].instantiates.as_ref().unwrap();
    assert_eq!(inst.name, "Empty");
    assert!(inst.args.is_empty());
}

#[test]
fn parses_ability_instantiating_template_with_bool_arg() {
    let src = "ability X : T(true, false) { target: enemy cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("must parse");
    let inst = file.abilities[0].instantiates.as_ref().unwrap();
    assert_eq!(inst.args.len(), 2);
    assert_eq!(inst.args[0], TemplateArg::Bool(true));
    assert_eq!(inst.args[1], TemplateArg::Bool(false));
}

// ---------------------------------------------------------------------------
// 7. Template + ability coexist in one file
// ---------------------------------------------------------------------------

#[test]
fn template_and_ability_in_same_file() {
    let src = r#"
template ElementalBolt(element: Material, radius: float = 3.0) {
    damage 50
}

ability Fireball : ElementalBolt(fire, 4.0) {
    target: ground
    range: 8.0
    cooldown: 6s
}

ability Strike {
    target: enemy
    range: 3.0
    cooldown: 1s
    damage 25
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.templates.len(), 1);
    assert_eq!(file.abilities.len(), 2);
    assert_eq!(file.templates[0].name, "ElementalBolt");
    assert_eq!(file.abilities[0].name, "Fireball");
    assert!(file.abilities[0].instantiates.is_some());
    assert_eq!(file.abilities[1].name, "Strike");
    assert!(file.abilities[1].instantiates.is_none());
}

// ---------------------------------------------------------------------------
// 8. Unknown parameter type is a clean parse error
// ---------------------------------------------------------------------------

#[test]
fn unknown_param_type_is_error() {
    let src = "template X(p: dragon) { damage 1 }";
    let err = parse_ability_file(src).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("dragon") || msg.contains("unknown template parameter type"),
        "expected unknown-type diagnostic; got: {msg}"
    );
}

#[test]
fn empty_params_with_extra_comma_is_error() {
    // `template X(,) { ... }` — a leading comma is invalid (no first
    // param to terminate). Not strictly required by spec but a clean
    // diagnostic helps authors.
    let src = "template X(,) { damage 1 }";
    let err = parse_ability_file(src).expect_err("must error");
    let _msg = err.to_string();
    // No specific message required — just that it errors out.
}

// ---------------------------------------------------------------------------
// 9. Defensive: nested template inside template body errors
// ---------------------------------------------------------------------------

#[test]
fn nested_template_block_is_error() {
    let src = r#"
template Outer() {
    template Inner() { damage 1 }
}
"#;
    let err = parse_ability_file(src).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("nested `template`") || msg.contains("template"),
        "expected nested-template diagnostic; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 10. Span coverage
// ---------------------------------------------------------------------------

#[test]
fn template_decl_carries_span() {
    let src = "template ElementalBolt(p: int = 1) { damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let t = &file.templates[0];
    assert!(t.span.start < t.span.end, "non-empty span");
    assert!(t.span.end <= src.len(), "in-source span");
    let slice = &src[t.span.start..t.span.end];
    assert!(slice.contains("template"), "span should cover the template keyword; got `{slice}`");
    assert!(slice.contains("ElementalBolt"), "span should cover the name; got `{slice}`");
}

#[test]
fn template_instantiation_carries_span() {
    let src = "ability F : T(fire, 4.0) { target: enemy cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("must parse");
    let inst = file.abilities[0].instantiates.as_ref().unwrap();
    assert!(inst.span.start < inst.span.end);
    let slice = &src[inst.span.start..inst.span.end];
    assert!(
        slice.contains("T(") && slice.contains("fire"),
        "span should cover the `: TemplateName(args)` clause; got `{slice}`"
    );
}

#[test]
fn template_param_carries_span() {
    let src = "template T(power: float = 2.5) { damage 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let p = &file.templates[0].params[0];
    assert!(p.span.start < p.span.end);
    let slice = &src[p.span.start..p.span.end];
    assert!(slice.contains("power"), "param span should cover the name; got `{slice}`");
    assert!(slice.contains("float"), "param span should cover the type; got `{slice}`");
    assert!(slice.contains("2.5"), "param span should cover the default; got `{slice}`");
}

#[test]
fn duplicate_template_param_is_error() {
    let src = "template T(p, p) { damage 1 }";
    let err = parse_ability_file(src).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate parameter") && msg.contains("`p`"),
        "expected duplicate-param diagnostic; got: {msg}"
    );
}

// Wave 1.3 sanity: structure keyword now accepted at top level (the
// Wave 1.2 era rejection is gone — see `ability_parser_wave_1_3.rs`
// for the Wave 1.3 surface tests).
#[test]
fn structure_keyword_now_accepted_wave_1_3() {
    let src = "structure Castle() { }";
    let file = parse_ability_file(src).expect("Wave 1.3 parses structures");
    assert_eq!(file.structures.len(), 1);
    assert_eq!(file.structures[0].name, "Castle");
}
