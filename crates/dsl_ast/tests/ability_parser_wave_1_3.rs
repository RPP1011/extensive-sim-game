//! Wave 1.3 `.ability` parser surface tests for `structure <Name>(<params>) {
//! ... }` top-level blocks (per `docs/spec/ability_dsl_unified.md` §12).
//!
//! Coverage:
//!  1. Minimal structure (`structure Empty() { }`).
//!  2. Castle example from spec §12 verbatim — body captured opaquely.
//!  3. Body containing nested `if X { ... }` braces don't terminate
//!     the outer `}` early (balanced-brace test).
//!  4. Structure without parens — `structure Wall { ... }` (shorthand
//!     for empty params).
//!  5. Typed params reusing the Wave 1.2 template-param grammar
//!     (int / float / bool / Material / Structure).
//!  6. Unknown structure param type is a clean parse error (same path
//!     as Wave 1.2 templates since `parse_template_param` is reused).
//!  7. Template + structure (+ ability) coexist in one file.
//!  8. Span coverage on StructureDecl.

use dsl_ast::ast::TemplateParamTy;
use dsl_ast::parse_ability_file;

// ---------------------------------------------------------------------------
// 1. Minimal structure
// ---------------------------------------------------------------------------

#[test]
fn parses_minimal_structure() {
    let src = "structure Empty() { }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 0);
    assert_eq!(file.passives.len(), 0);
    assert_eq!(file.templates.len(), 0);
    assert_eq!(file.structures.len(), 1);
    let s = &file.structures[0];
    assert_eq!(s.name, "Empty");
    assert!(s.params.is_empty(), "no params");
    // Body is just whitespace — stored verbatim.
    assert_eq!(s.body_raw.trim(), "", "body should be empty");
}

// ---------------------------------------------------------------------------
// 2. Castle example from spec §12 verbatim
// ---------------------------------------------------------------------------

#[test]
fn parses_castle_example_from_spec() {
    // The spec §12 example, verbatim. The body holds 5 statement kinds
    // (place / harvest / transform / include / if) plus optional
    // headers (bounds / origin / rotatable / symmetry); Wave 1.3
    // captures it opaquely.
    let src = r#"
structure Castle(wall_mat: Material = stone, height: int = 8) {
    bounds: box(20, $height, 20)
    origin: (0, 0, 0)
    rotatable
    symmetry: radial(4)
    place $wall_mat in box(20, 1, 20)
    place $wall_mat in (box(20, $height, 20) diff box(18, $height, 18))
    if $height > 4 { place $wall_mat in box(20, 1, 20) at (0, $height, 0) }
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.structures.len(), 1);
    let s = &file.structures[0];
    assert_eq!(s.name, "Castle");
    // Two typed params reusing TemplateParam.
    assert_eq!(s.params.len(), 2);
    assert_eq!(s.params[0].name, "wall_mat");
    assert_eq!(s.params[0].ty, Some(TemplateParamTy::Material));
    assert_eq!(s.params[1].name, "height");
    assert_eq!(s.params[1].ty, Some(TemplateParamTy::Int));
    // Body should contain the inner text (verbatim slice between outer
    // `{` and `}`).
    assert!(s.body_raw.contains("bounds:"), "body should contain bounds: header; got `{}`", s.body_raw);
    assert!(s.body_raw.contains("rotatable"), "body should contain rotatable; got `{}`", s.body_raw);
    assert!(s.body_raw.contains("symmetry: radial(4)"), "body should contain symmetry; got `{}`", s.body_raw);
    assert!(s.body_raw.contains("place $wall_mat"), "body should contain place stmt");
    assert!(s.body_raw.contains("if $height > 4"), "body should contain if stmt");
    // Body should NOT include the outer wrapping braces — every `{` /
    // `}` in body_raw must come from the inner statements (here, the
    // `if … { … }` block + the spec-§12 `box(...)` parens are NOT
    // braces). Counting balance is the cleanest invariant.
    let open = s.body_raw.matches('{').count();
    let close = s.body_raw.matches('}').count();
    assert_eq!(open, close, "inner braces must balance — outer braces are stripped");
    assert_eq!(open, 1, "Castle body has exactly one inner `{{ }}` pair (the `if` block)");
}

// ---------------------------------------------------------------------------
// 3. Nested braces inside the body don't terminate early
// ---------------------------------------------------------------------------

#[test]
fn parses_structure_with_nested_braces() {
    // The body has nested `if X { ... } else { ... }` blocks. The
    // balanced-brace scan should walk past the inner `{` `}` pairs and
    // only stop at the outer matching `}`.
    let src = r#"
structure Tower() {
    if $height > 4 { place stone in box(5, 1, 5) }
    if $height > 8 {
        place stone in box(5, 1, 5)
        if $width > 2 { place wood in box(2, 1, 2) }
    } else {
        place wood in box(3, 1, 3)
    }
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.structures.len(), 1, "exactly one structure parsed");
    let s = &file.structures[0];
    assert_eq!(s.name, "Tower");
    // The body should contain BOTH if-blocks and the nested `if $width`.
    assert!(s.body_raw.contains("if $height > 4"));
    assert!(s.body_raw.contains("if $height > 8"));
    assert!(s.body_raw.contains("if $width > 2"));
    assert!(s.body_raw.contains("else"));
    // Body should also have closing `}` from the inner blocks (just
    // not so many that depth went negative — that's the balanced-brace
    // contract).
    let open_count = s.body_raw.matches('{').count();
    let close_count = s.body_raw.matches('}').count();
    assert_eq!(open_count, close_count, "body braces should balance internally");
}

// ---------------------------------------------------------------------------
// 4. Structure without parens — shorthand for empty params
// ---------------------------------------------------------------------------

#[test]
fn parses_structure_without_params() {
    // `structure Wall { ... }` with no parens — accepted as shorthand
    // for `structure Wall() { ... }`. Spec §12 doesn't forbid this and
    // it avoids surprising authors of zero-param structures.
    let src = "structure Wall { place stone in box(10, 5, 1) }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.structures.len(), 1);
    let s = &file.structures[0];
    assert_eq!(s.name, "Wall");
    assert!(s.params.is_empty(), "no params (shorthand form)");
    assert!(s.body_raw.contains("place stone"));
}

// ---------------------------------------------------------------------------
// 5. Typed params with all 5 types
// ---------------------------------------------------------------------------

#[test]
fn parses_structure_with_typed_params() {
    // All 5 admitted param types: int / float / bool / Material /
    // Structure. Reuses the Wave 1.2 grammar verbatim so all 5 land.
    let src = r#"
structure Mixed(
    count: int = 5,
    radius: float = 3.0,
    on_fire: bool = true,
    wall_mat: Material = stone,
    base: Structure = castle
) {
    place $wall_mat in box(10, 1, 10)
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let s = &file.structures[0];
    assert_eq!(s.params.len(), 5);
    assert_eq!(s.params[0].ty, Some(TemplateParamTy::Int));
    assert_eq!(s.params[1].ty, Some(TemplateParamTy::Float));
    assert_eq!(s.params[2].ty, Some(TemplateParamTy::Bool));
    assert_eq!(s.params[3].ty, Some(TemplateParamTy::Material));
    assert_eq!(s.params[4].ty, Some(TemplateParamTy::Structure));
}

// ---------------------------------------------------------------------------
// 6. Unknown structure param type is a clean parse error
// ---------------------------------------------------------------------------

#[test]
fn unknown_structure_param_type_is_error() {
    // Reuses `parse_template_param`, so the diagnostic mirrors the
    // Wave 1.2 template path. `dragon` isn't in the spec §11.1 type
    // list (int / float / bool / Material / Structure).
    let src = "structure X(p: dragon) { place stone in box(1, 1, 1) }";
    let err = parse_ability_file(src).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("dragon") || msg.contains("unknown template parameter type"),
        "expected unknown-type diagnostic; got: {msg}"
    );
}

#[test]
fn duplicate_structure_param_is_error() {
    let src = "structure X(p, p) { }";
    let err = parse_ability_file(src).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate parameter") && msg.contains("`p`"),
        "expected duplicate-param diagnostic; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 7. Template + structure + ability coexist in one file
// ---------------------------------------------------------------------------

#[test]
fn template_and_structure_in_same_file() {
    let src = r#"
template ElementalBolt(element: Material, radius: float = 3.0) {
    damage 50
}

structure Castle(wall_mat: Material = stone, height: int = 8) {
    bounds: box(20, $height, 20)
    place $wall_mat in box(20, 1, 20)
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
    assert_eq!(file.structures.len(), 1);
    assert_eq!(file.abilities.len(), 2);
    assert_eq!(file.templates[0].name, "ElementalBolt");
    assert_eq!(file.structures[0].name, "Castle");
    assert_eq!(file.abilities[0].name, "Fireball");
    assert_eq!(file.abilities[1].name, "Strike");
}

#[test]
fn multiple_structures_in_one_file_preserve_source_order() {
    let src = r#"
structure FirstOne() { }
structure SecondOne() { place stone in box(1, 1, 1) }
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.structures.len(), 2);
    assert_eq!(file.structures[0].name, "FirstOne");
    assert_eq!(file.structures[1].name, "SecondOne");
}

// ---------------------------------------------------------------------------
// 8. Span coverage
// ---------------------------------------------------------------------------

#[test]
fn structure_decl_carries_span() {
    let src = "structure Castle(wall_mat: Material) { place $wall_mat in box(5, 1, 5) }";
    let file = parse_ability_file(src).expect("must parse");
    let s = &file.structures[0];
    assert!(s.span.start < s.span.end, "non-empty span");
    assert!(s.span.end <= src.len(), "in-source span");
    let slice = &src[s.span.start..s.span.end];
    assert!(slice.contains("structure"), "span should cover the structure keyword; got `{slice}`");
    assert!(slice.contains("Castle"), "span should cover the name; got `{slice}`");
    assert!(slice.contains("place"), "span should cover the body; got `{slice}`");
}

#[test]
fn structure_param_carries_span() {
    // Span check on the inner TemplateParam (reused from Wave 1.2).
    let src = "structure T(power: float = 2.5) { place stone in box(1, 1, 1) }";
    let file = parse_ability_file(src).expect("must parse");
    let p = &file.structures[0].params[0];
    assert!(p.span.start < p.span.end);
    let slice = &src[p.span.start..p.span.end];
    assert!(slice.contains("power"), "param span should cover the name; got `{slice}`");
    assert!(slice.contains("float"), "param span should cover the type; got `{slice}`");
    assert!(slice.contains("2.5"), "param span should cover the default; got `{slice}`");
}

// ---------------------------------------------------------------------------
// 9. Defensive: EOF inside structure body errors cleanly
// ---------------------------------------------------------------------------

#[test]
fn unclosed_structure_body_is_error() {
    let src = "structure Castle() { place stone in box(1, 1, 1)";
    let err = parse_ability_file(src).expect_err("must error");
    let _msg = err.to_string();
    // No specific message required — just that it errors out cleanly
    // (no panic).
}

#[test]
fn unclosed_structure_param_list_is_error() {
    let src = "structure Castle(wall_mat: Material { }";
    let err = parse_ability_file(src).expect_err("must error");
    let _msg = err.to_string();
}
