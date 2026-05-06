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
    // #140: template defs surface as a `LowerOutcome::skipped` entry,
    // not a hard `Err`. The file has no `ability` decls, so
    // `programs` is empty.
    let outcome = lower_ability_file(&file).expect("file must lower");
    assert!(outcome.programs.is_empty(), "no abilities, no programs");
    assert_eq!(outcome.skipped.len(), 1);
    match &outcome.skipped[0] {
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
    let outcome = lower_ability_file(&file).expect("file must lower");
    assert_eq!(outcome.skipped.len(), 1);
    let msg = outcome.skipped[0].to_string();
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
fn lowering_collects_every_template_into_skipped_list_in_source_order() {
    // #140: pre-#140 the first template short-circuited and the
    // second was invisible. Now both surface as `skipped` entries in
    // source order so authors see a complete list of decls that
    // didn't lower.
    let src = r#"
template FirstOne() { damage 1 }
template SecondOne() { heal 1 }
"#;
    let file = parse_ability_file(src).expect("parser");
    let outcome = lower_ability_file(&file).expect("file must lower");
    assert!(outcome.programs.is_empty(), "no abilities");
    assert_eq!(outcome.skipped.len(), 2, "both templates must be skipped");
    assert!(
        matches!(
            &outcome.skipped[0],
            LowerError::TemplateBlockNotImplemented { name, .. } if name == "FirstOne"
        ),
        "first skipped entry must be FirstOne; got {:?}",
        outcome.skipped[0]
    );
    assert!(
        matches!(
            &outcome.skipped[1],
            LowerError::TemplateBlockNotImplemented { name, .. } if name == "SecondOne"
        ),
        "second skipped entry must be SecondOne; got {:?}",
        outcome.skipped[1]
    );
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
    // Per-decl errors (instantiation hits an unimplemented path) DO
    // still propagate as `Err` — they're genuine bugs in the source,
    // not deferred-feature gaps. Only top-level passive/template/
    // structure DEFINITIONS land in `outcome.skipped`.
    let err = lower_ability_file(&file).expect_err("instantiating ability must error");
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
