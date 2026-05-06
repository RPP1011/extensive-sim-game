//! Wave 1.1 lowering tests — verify the new parser surfaces lower
//! into the engine's IR slots (cost/charges/recharge/toggle), and that
//! `passive` top-level blocks still defer to Wave 2+ via
//! `PassiveBlockNotImplemented`. The Wave 1 corpus regression also
//! lives here.
//!
//! Wave 2 follow-on (post Delivery::Method, ea3af642+) promoted the
//! header surfaces from `HeaderNotImplemented` errors into:
//!   * `AbilityProgram.cost: Option<AbilityCost>`
//!   * `AbilityProgram.charges: Option<u32>`
//!   * `AbilityProgram.recharge_ticks: Option<u32>`
//!   * `AbilityProgram.is_toggle: bool`
//! Apply handlers debit / refill / toggle later (deferred — resource
//! SoA fields like stamina + per-agent charge state still missing).

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, lower_ability_file, LowerError};

// ---------------------------------------------------------------------------
// 1. Each Wave 1.1 ability-block header lowers into its IR slot
// ---------------------------------------------------------------------------

#[test]
fn lowering_cost_header_captures_resource_and_amount() {
    use engine::ability::program::{CostAmount, CostResource};
    let src = "ability Bolt { target: enemy range: 5.0 cost: 30 mana cooldown: 1s damage 15 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("cost header must lower");
    let cost = prog.cost.expect("cost slot populated");
    assert_eq!(cost.resource, CostResource::Mana);
    assert_eq!(cost.amount, CostAmount::Flat(30.0));
}

#[test]
fn lowering_cost_percent_form_captures_percent() {
    use engine::ability::program::{CostAmount, CostResource};
    let src = "ability Drain { target: enemy cost: 10% hp cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("percent cost must lower");
    let cost = prog.cost.expect("cost slot populated");
    assert_eq!(cost.resource, CostResource::Hp);
    assert_eq!(cost.amount, CostAmount::PercentOfMax(10.0));
}

#[test]
fn lowering_charges_header_captures_max() {
    let src = "ability Volley { target: enemy charges: 3 cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("charges must lower");
    assert_eq!(prog.charges, Some(3));
}

#[test]
fn lowering_recharge_header_captures_ticks() {
    let src = "ability Volley { target: enemy recharge: 8s cooldown: 0 damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("recharge must lower");
    // 8s = 80 ticks at the fixed 100ms cadence.
    assert_eq!(prog.recharge_ticks, Some(80));
}

#[test]
fn lowering_toggle_marker_sets_is_toggle() {
    let src = "ability Stance { target: self toggle cooldown: 1s shield 20 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("toggle must lower");
    assert!(prog.is_toggle, "is_toggle must be true after `toggle` marker");
}

#[test]
fn lowering_no_wave_1_1_headers_keeps_defaults() {
    // Wave 1 corpus shape — no cost/charges/recharge/toggle declared.
    // Defaults stay None/false so existing programs are bit-stable.
    let src = "ability Plain { target: enemy range: 5.0 cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("must lower");
    assert!(prog.cost.is_none());
    assert!(prog.charges.is_none());
    assert!(prog.recharge_ticks.is_none());
    assert!(!prog.is_toggle);
}

// ---------------------------------------------------------------------------
// 2. Passive top-level blocks surface PassiveBlockNotImplemented
// ---------------------------------------------------------------------------

#[test]
fn lowering_file_with_passive_returns_unimplemented_error() {
    let src = r#"
passive Vigilance {
    trigger: on_damage_taken
    cooldown: 5s
    heal 10
}
"#;
    let file = parse_ability_file(src).expect("parser");
    // #140: passive blocks no longer abort the file — they're collected
    // into `LowerOutcome::skipped` so any abilities alongside still
    // lower. This file has zero abilities + one passive, so the
    // outcome's `programs` is empty and `skipped` carries the passive.
    let outcome = lower_ability_file(&file).expect("file must lower");
    assert!(outcome.programs.is_empty(), "no abilities, so no programs");
    assert_eq!(outcome.skipped.len(), 1, "the lone passive is skipped");
    match &outcome.skipped[0] {
        LowerError::PassiveBlockNotImplemented { name, span } => {
            assert_eq!(name, "Vigilance");
            assert!(span.start < span.end, "span must be non-empty");
        }
        other => panic!("expected PassiveBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_file_with_ability_then_passive_lowers_ability_skips_passive() {
    // #140: when an `ability` precedes the `passive`, the ability
    // lowers cleanly into `outcome.programs` AND the passive surfaces
    // in `outcome.skipped` (so callers can warn about it without
    // losing the abilities). Pre-#140 the passive aborted the whole
    // file with `Err(_)` and the ability never reached the registry.
    let src = r#"
ability Strike {
    target: enemy
    range: 3.0
    cooldown: 1s
    damage 15
}

passive ThornArmor {
    trigger: on_damage_taken
    cooldown: 2s
    damage 5
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let outcome = lower_ability_file(&file).expect("composite file must lower");
    assert_eq!(outcome.programs.len(), 1, "Strike must lower");
    assert_eq!(outcome.skipped.len(), 1, "ThornArmor must be skipped");
    assert!(
        matches!(
            &outcome.skipped[0],
            LowerError::PassiveBlockNotImplemented { name, .. } if name == "ThornArmor"
        ),
        "expected PassiveBlockNotImplemented(ThornArmor); got {:?}",
        outcome.skipped[0]
    );
}

// ---------------------------------------------------------------------------
// 3. Sanity: Wave 1 corpus (Strike / ShieldUp / Mend) still lowers.
// ---------------------------------------------------------------------------

#[test]
fn lowering_file_with_only_legacy_abilities_still_works() {
    // Three legacy-shape abilities (Wave 1.0 headers + Wave 1.6 verbs)
    // in one file. With no Wave 1.1 surfaces in sight, lowering must
    // produce three programs — verifying we didn't break the happy
    // path with the new error arms.
    let src = r#"
ability Strike {
    target: enemy
    range: 5.0
    cooldown: 1s
    hint: damage
    damage 15
}

ability ShieldUp {
    target: self
    cooldown: 4s
    hint: defense
    shield 50
}

ability Mend {
    target: self
    cooldown: 3s
    hint: heal
    heal 25
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let outcome = lower_ability_file(&file).expect("legacy corpus must lower");
    assert_eq!(outcome.programs.len(), 3);
    assert!(outcome.skipped.is_empty(), "no top-level skips on Wave 1 corpus");
}

// Diagnostic test retired: cost/charges/recharge/toggle now lower into
// AbilityProgram fields rather than erroring. The lowering tests above
// pin the captured values; render-quality is covered by the
// PassiveBlockNotImplemented diagnostic test for the surfaces that
// still defer.

// ---------------------------------------------------------------------------
// 4. #140: composite file mixing all three skip categories — passive +
//    ability + template + structure — yields 1 program + 3 skipped
//    in source order, with the ability still reaching the registry.
// ---------------------------------------------------------------------------

#[test]
fn lowering_composite_file_yields_programs_and_skipped_in_source_order() {
    let src = r#"
passive Riposte {
    trigger: on_damage_taken
    cooldown: 2s
    damage 5
}

ability Strike {
    target: enemy
    range: 5.0
    cooldown: 1s
    damage 15
}

template Bolt(element: Material) {
    damage 50
}

structure Hut(wall_mat: Material = stone) {
    bounds: box(4, 3, 4)
}
"#;
    let file = parse_ability_file(src).expect("parser");
    let outcome = lower_ability_file(&file).expect("composite file must lower");

    assert_eq!(outcome.programs.len(), 1, "Strike must lower despite the three skipped decls");
    assert_eq!(outcome.skipped.len(), 3, "all three top-level decls land in skipped");

    // The skip order is passives → templates → structures (the
    // collection order in `lower_ability_file`). Authors get a stable
    // ordering even if the source mixes them.
    assert!(matches!(
        &outcome.skipped[0],
        LowerError::PassiveBlockNotImplemented { name, .. } if name == "Riposte"
    ));
    assert!(matches!(
        &outcome.skipped[1],
        LowerError::TemplateBlockNotImplemented { name, .. } if name == "Bolt"
    ));
    assert!(matches!(
        &outcome.skipped[2],
        LowerError::StructureBlockNotImplemented { name, .. } if name == "Hut"
    ));
}
