//! Wave 1.2 lowering tests — the new `template` top-level block AND the
//! `ability X : TemplateName(args) { ... }` instantiation clause both
//! parse but error cleanly at the lowering boundary, and the Wave 1
//! corpus (Strike / ShieldUp / Mend / Bleed / Reap / Vampirize /
//! Fortify) still lowers without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.2 module
//! contract, lowering surfaces:
//!   * `LowerError::TemplateBlockNotImplemented { name, span }` for any
//!     non-empty `file.templates`.
//!   * `LowerError::TemplateInstantiationNotImplemented { ability,
//!     template, span }` for any `decl.instantiates.is_some()`.
//!
//! Template expansion (parameter substitution into `$ident` references
//! in the body, depth-bounded recursion per spec §11.3) is Wave 2+.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, lower_ability_file, LowerError};

// ---------------------------------------------------------------------------
// 1. `template <Name>(<params>) { ... }` top-level — TemplateBlockNotImplemented
// ---------------------------------------------------------------------------

#[test]
fn lowering_file_with_template_returns_unimplemented_error() {
    let src = r#"
template ElementalBolt(element: Material, radius: float = 3.0) {
    damage 50
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    match err {
        LowerError::TemplateBlockNotImplemented { name, span } => {
            assert_eq!(name, "ElementalBolt");
            assert!(span.start < span.end, "non-empty span");
            assert!(span.end <= src.len(), "in-source span");
        }
        other => panic!("expected TemplateBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_template_block_diagnostic_names_the_template() {
    let src = "template Empty() { damage 1 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    let msg = err.to_string();
    assert!(
        msg.contains("Empty"),
        "diagnostic must name the template; got: {msg}"
    );
    assert!(
        msg.contains("template"),
        "diagnostic must mention `template`; got: {msg}"
    );
}

#[test]
fn lowering_first_template_short_circuits() {
    // Multiple templates — the first one in source order should be the
    // surfaced diagnostic so authors get a stable error location.
    let src = r#"
template FirstOne() { damage 1 }
template SecondOne() { heal 1 }
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    match err {
        LowerError::TemplateBlockNotImplemented { name, .. } => {
            assert_eq!(name, "FirstOne", "first template wins");
        }
        other => panic!("expected TemplateBlockNotImplemented; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 2. `ability X : T(args) { ... }` instantiation — TemplateInstantiationNotImplemented
// ---------------------------------------------------------------------------

#[test]
fn lowering_ability_instantiating_template_returns_unimplemented() {
    // A standalone ability with an instantiation clause — `lower_ability_decl`
    // surfaces the error directly. Use a body without any template
    // decl in the file so we hit the per-decl path, not the file-level
    // template guard.
    let src = "ability Fireball : ElementalBolt(fire, 4.0) { target: ground range: 8.0 cooldown: 6s }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("lowering must error");
    match err {
        LowerError::TemplateInstantiationNotImplemented { ability, template, span } => {
            assert_eq!(ability, "Fireball");
            assert_eq!(template, "ElementalBolt");
            assert!(span.start < span.end, "non-empty span");
            assert!(span.end <= src.len(), "in-source span");
        }
        other => panic!("expected TemplateInstantiationNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_instantiation_short_circuits_before_target_mode_check() {
    // `target: ground` would otherwise trigger `TargetModeReserved`; the
    // instantiation diagnostic should fire first so authors don't see a
    // misleading error on a body they didn't pick to begin with.
    let src = "ability X : T(fire) { target: ground range: 8.0 cooldown: 6s }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("lowering must error");
    match err {
        LowerError::TemplateInstantiationNotImplemented { template, .. } => {
            assert_eq!(template, "T");
        }
        other => panic!("expected TemplateInstantiationNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_instantiation_diagnostic_names_template_and_ability() {
    let src = "ability F : Bolt(fire, 4.0) { target: enemy cooldown: 1s damage 1 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("lowering must error");
    let msg = err.to_string();
    assert!(msg.contains("Bolt"), "diagnostic must mention the template; got: {msg}");
    assert!(msg.contains("F"), "diagnostic must mention the ability; got: {msg}");
}

#[test]
fn lowering_file_with_instantiating_ability_surfaces_via_file_path() {
    // `lower_ability_file` walks decls in order; an instantiating ability
    // should surface the same per-decl error.
    let src = r#"
ability Fireball : ElementalBolt(fire, 4.0) {
    target: enemy cooldown: 1s damage 5
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_file(&file).expect_err("lowering must error");
    match err {
        LowerError::TemplateInstantiationNotImplemented { ability, template, .. } => {
            assert_eq!(ability, "Fireball");
            assert_eq!(template, "ElementalBolt");
        }
        other => panic!("expected TemplateInstantiationNotImplemented; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Wave 1 corpus regression — no templates / instantiations, lowers cleanly
// ---------------------------------------------------------------------------

#[test]
fn lowering_wave_1_corpus_still_works() {
    // The seven-ability Wave 1 corpus uses neither templates nor
    // instantiations, so the lowering pipeline should produce a valid
    // AbilityProgram for each.
    for (name, src) in [
        ("Strike",
         "ability Strike { target: enemy range: 5.0 cooldown: 1s hint: damage damage 15 }"),
        ("ShieldUp",
         "ability ShieldUp { target: self cooldown: 4s hint: defense shield 50 }"),
        ("Mend",
         "ability Mend { target: self cooldown: 3s hint: heal heal 20 }"),
        ("Bleed",
         "ability Bleed { target: enemy range: 3.0 cooldown: 2s hint: damage damage 5 }"),
        ("Reap",
         "ability Reap { target: enemy range: 4.0 cooldown: 5s hint: damage execute 10.0 }"),
        ("Vampirize",
         "ability Vampirize { target: self cooldown: 6s hint: utility lifesteal 0.5 4s }"),
        ("Fortify",
         "ability Fortify { target: self cooldown: 5s hint: defense damage_modify 0.5 5s }"),
    ] {
        let file = parse_ability_file(src).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        let prog = lower_ability_decl(&file.abilities[0])
            .unwrap_or_else(|e| panic!("{name} lowers: {e:?}"));
        assert_eq!(prog.effects.len(), 1, "{name} should have one effect");
    }
}
