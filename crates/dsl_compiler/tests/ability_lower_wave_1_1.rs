//! Wave 1.1 lowering tests — verify the new parser surfaces
//! (`cost`/`charges`/`recharge`/`toggle` headers + top-level `passive`
//! blocks) error cleanly at the lowering boundary, and that the legacy
//! Wave 1 corpus still lowers without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.1 module-level
//! docs: lowering of these surfaces requires engine-side schema changes
//! (cost gates, per-agent charge SoA, toggle state, PerEvent dispatch
//! keyed on trigger kinds) and is the work of Wave 2+. Until then we
//! surface explicit `HeaderNotImplemented` / `PassiveBlockNotImplemented`
//! errors rather than silently dropping the new fields.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, lower_ability_file, LowerError};

// ---------------------------------------------------------------------------
// 1. Each Wave 1.1 ability-block header surfaces HeaderNotImplemented
// ---------------------------------------------------------------------------

#[test]
fn lowering_cost_header_returns_unimplemented_error() {
    let src = "ability Bolt { target: enemy range: 5.0 cost: 30 mana cooldown: 1s damage 15 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0])
        .expect_err("cost header must defer to Wave 2+");
    match err {
        LowerError::HeaderNotImplemented { header, span } => {
            assert_eq!(header, "cost");
            assert!(span.start < span.end, "span must be non-empty");
        }
        other => panic!("expected HeaderNotImplemented {{ header: \"cost\" }}; got {other:?}"),
    }
}

#[test]
fn lowering_charges_header_returns_unimplemented_error() {
    let src = "ability Volley { target: enemy charges: 3 cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0])
        .expect_err("charges header must defer to Wave 2+");
    match err {
        LowerError::HeaderNotImplemented { header, .. } => {
            assert_eq!(header, "charges");
        }
        other => panic!("expected HeaderNotImplemented {{ header: \"charges\" }}; got {other:?}"),
    }
}

#[test]
fn lowering_recharge_header_returns_unimplemented_error() {
    let src = "ability Volley { target: enemy recharge: 8s cooldown: 0 damage 5 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0])
        .expect_err("recharge header must defer to Wave 2+");
    match err {
        LowerError::HeaderNotImplemented { header, .. } => {
            assert_eq!(header, "recharge");
        }
        other => panic!("expected HeaderNotImplemented {{ header: \"recharge\" }}; got {other:?}"),
    }
}

#[test]
fn lowering_toggle_header_returns_unimplemented_error() {
    let src = "ability Stance { target: self toggle cooldown: 1s shield 20 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0])
        .expect_err("toggle marker must defer to Wave 2+");
    match err {
        LowerError::HeaderNotImplemented { header, .. } => {
            assert_eq!(header, "toggle");
        }
        other => panic!("expected HeaderNotImplemented {{ header: \"toggle\" }}; got {other:?}"),
    }
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
    let err = lower_ability_file(&file)
        .expect_err("passive block must defer to Wave 2+");
    match err {
        LowerError::PassiveBlockNotImplemented { name, span } => {
            assert_eq!(name, "Vigilance");
            assert!(span.start < span.end, "span must be non-empty");
        }
        other => panic!("expected PassiveBlockNotImplemented; got {other:?}"),
    }
}

#[test]
fn lowering_file_with_ability_then_passive_still_errors_on_passive() {
    // Even when an `ability` precedes the `passive`, lowering must
    // surface the unimplemented passive — silently dropping it would
    // mean the author's combat reaction logic compiled away to nothing.
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
    let err = lower_ability_file(&file).expect_err("passive must error");
    assert!(
        matches!(err, LowerError::PassiveBlockNotImplemented { ref name, .. } if name == "ThornArmor"),
        "expected PassiveBlockNotImplemented(ThornArmor); got {err:?}"
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
    let progs = lower_ability_file(&file).expect("legacy corpus must lower");
    assert_eq!(progs.len(), 3);
}

#[test]
fn lowering_diagnostic_message_mentions_wave_2() {
    // Render-quality smoke test: the LowerError Display impl should be
    // explicit about WHY the header is unimplemented so authors don't
    // file bugs against a known-deferred feature.
    let src = "ability X { target: self cost: 30 mana cooldown: 1s heal 1 }";
    let file = parse_ability_file(src).expect("parser");
    let err = lower_ability_decl(&file.abilities[0]).expect_err("must error");
    let msg = err.to_string();
    assert!(
        msg.contains("cost") && (msg.contains("Wave 2") || msg.contains("Wave 2+")),
        "diagnostic should mention `cost` and `Wave 2`; got: {msg}"
    );
}
