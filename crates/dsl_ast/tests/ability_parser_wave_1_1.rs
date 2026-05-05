//! Wave 1.1 `.ability` parser surface tests.
//!
//! Coverage by surface (per `docs/spec/ability_dsl_unified.md`):
//!
//! 1. New `ability`-block headers (§4.2):
//!     * `cost: 30 mana`           — flat amount, explicit resource
//!     * `cost: 12`                — flat amount, default-mana shorthand
//!     * `cost: 25% hp`            — percent-of-max
//!     * `charges: 3` + `recharge: 8s`
//!     * `toggle`                  — bare marker (no value)
//!
//! 2. Top-level `passive` block (§5):
//!     * minimal `passive` with `trigger:` + `cooldown:` + body
//!     * passive with only `hint:` (always-on, no trigger)
//!     * `passive` and `ability` coexisting in the same file
//!
//! 3. Span / error coverage:
//!     * cost carries a Span
//!     * unknown resource is a clean parse error
//!     * negative cost amount is rejected

use dsl_ast::ast::{
    AbilityHeader, CostAmount, CostResource, EffectArg, HintName, PassiveHeader,
};
use dsl_ast::parse_ability_file;

// ---------------------------------------------------------------------------
// 1. Ability-block header additions
// ---------------------------------------------------------------------------

#[test]
fn parses_cost_flat_mana() {
    let src = "ability X { target: enemy range: 5.0 cost: 30 mana cooldown: 1s damage 15 }";
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 1);
    let a = &file.abilities[0];
    let cost = a
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .expect("cost header present");
    assert_eq!(cost.resource, CostResource::Mana);
    match cost.amount {
        CostAmount::Flat(v) => assert!((v - 30.0).abs() < 1e-6),
        other => panic!("expected Flat(30.0); got {other:?}"),
    }
}

#[test]
fn parses_cost_default_mana_shorthand() {
    // The existing LoL hero corpus uses `cost: <int>` without a resource
    // keyword. Spec §4.2 lists `cost: int` as the mana/resource gate, so
    // the shorthand defaults to mana.
    let src = "ability X { target: enemy cost: 12 cooldown: 1s damage 5 }";
    let file = parse_ability_file(src).expect("must parse");
    let cost = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .expect("cost header");
    assert_eq!(cost.resource, CostResource::Mana);
    assert_eq!(cost.amount, CostAmount::Flat(12.0));
}

#[test]
fn parses_cost_percent_hp() {
    // Percent form preserves the percentage scalar (25% → 25.0), matching
    // the `EffectArg::Percent` convention the Wave 1.0 parser already
    // established for percent literals.
    let src = "ability Bloodcast { target: self cost: 25% hp cooldown: 1s heal 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let cost = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .expect("cost header");
    assert_eq!(cost.resource, CostResource::Hp);
    assert_eq!(cost.amount, CostAmount::PercentOfMax(25.0));
}

#[test]
fn parses_cost_stamina_and_gold_resources() {
    // All four spec-listed resources resolve.
    let src = r#"
ability A { target: self cost: 5 stamina cooldown: 1s heal 1 }
ability B { target: self cost: 100 gold  cooldown: 1s heal 1 }
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 2);
    let cost_a = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .unwrap();
    let cost_b = file.abilities[1]
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .unwrap();
    assert_eq!(cost_a.resource, CostResource::Stamina);
    assert_eq!(cost_b.resource, CostResource::Gold);
}

#[test]
fn parses_charges_and_recharge() {
    let src = r#"
ability Volley {
    target: enemy
    range: 6.0
    charges: 3
    recharge: 8s
    cooldown: 0
    damage 10
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let a = &file.abilities[0];
    let charges = a
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Charges(n) = h { Some(*n) } else { None })
        .expect("charges header");
    assert_eq!(charges, 3);
    let recharge = a
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Recharge(d) = h { Some(*d) } else { None })
        .expect("recharge header");
    assert_eq!(recharge.millis, 8_000);
}

#[test]
fn parses_toggle_marker() {
    // `toggle` is a marker — bare keyword, no `:`, no value. Body can
    // still hold the usual headers + effects.
    let src = r#"
ability Stance {
    target: self
    toggle
    cooldown: 1s
    shield 20
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let a = &file.abilities[0];
    assert!(
        a.headers.iter().any(|h| matches!(h, AbilityHeader::Toggle)),
        "expected Toggle marker; got {:?}",
        a.headers
    );
    // Body still parses normally.
    assert_eq!(a.effects.len(), 1);
    assert_eq!(a.effects[0].verb, "shield");
}

// ---------------------------------------------------------------------------
// 2. Passive top-level form (spec §5)
// ---------------------------------------------------------------------------

#[test]
fn parses_minimal_passive() {
    let src = r#"
passive Vigilance {
    trigger: on_damage_taken
    cooldown: 5s
    heal 10
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 0);
    assert_eq!(file.passives.len(), 1);
    let p = &file.passives[0];
    assert_eq!(p.name, "Vigilance");
    let trigger = p
        .headers
        .iter()
        .find_map(|h| if let PassiveHeader::Trigger(s) = h { Some(s.as_str()) } else { None })
        .expect("trigger header");
    assert_eq!(trigger, "on_damage_taken");
    let cd = p
        .headers
        .iter()
        .find_map(|h| if let PassiveHeader::Cooldown(d) = h { Some(*d) } else { None })
        .expect("cooldown header");
    assert_eq!(cd.millis, 5_000);
    assert_eq!(p.effects.len(), 1);
    assert_eq!(p.effects[0].verb, "heal");
    match p.effects[0].args[0] {
        EffectArg::Number(v) => assert!((v - 10.0).abs() < 1e-6),
        ref other => panic!("expected Number(10); got {other:?}"),
    }
}

#[test]
fn parses_passive_with_hint_only() {
    // Always-on passive — only a `hint:` header (no trigger). The parser
    // must accept it; lowering gets to decide whether triggerless
    // passives are legal (deferred).
    let src = r#"
passive AuraOfMight {
    hint: defense
    shield 5
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.passives.len(), 1);
    let p = &file.passives[0];
    let hint = p
        .headers
        .iter()
        .find_map(|h| if let PassiveHeader::Hint(h) = h { Some(*h) } else { None })
        .expect("hint header");
    assert_eq!(hint, HintName::Defense);
    assert!(
        !p.headers.iter().any(|h| matches!(h, PassiveHeader::Trigger(_))),
        "no trigger expected"
    );
}

#[test]
fn parses_passive_with_range_filter() {
    // Spec §5 lists `range:` as an optional trigger filter (mask
    // predicate clause).
    let src = r#"
passive Riposte {
    trigger: on_damage_taken
    cooldown: 5s
    range: 2.0
    damage 30
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    let p = &file.passives[0];
    let r = p
        .headers
        .iter()
        .find_map(|h| if let PassiveHeader::Range(r) = h { Some(*r) } else { None })
        .expect("range header");
    assert!((r - 2.0).abs() < 1e-6);
}

#[test]
fn passive_and_ability_in_same_file() {
    // The spec doesn't forbid mixing top-level forms in one file.
    let src = r#"
ability Strike {
    target: enemy
    range: 3.0
    cooldown: 1s
    damage 25
}

passive ThornArmor {
    trigger: on_damage_taken
    cooldown: 2s
    damage 5
}
"#;
    let file = parse_ability_file(src).expect("must parse");
    assert_eq!(file.abilities.len(), 1);
    assert_eq!(file.passives.len(), 1);
    assert_eq!(file.abilities[0].name, "Strike");
    assert_eq!(file.passives[0].name, "ThornArmor");
}

#[test]
fn empty_passive_body_is_rejected() {
    // Symmetric with the existing `empty body` rule for abilities.
    let src = "passive Empty { }";
    let err = parse_ability_file(src).expect_err("empty passive must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty body"),
        "expected empty-body diagnostic; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 3. Span + error cases
// ---------------------------------------------------------------------------

#[test]
fn cost_carries_span() {
    // The Span on `CostSpec` should cover at least the value bytes
    // (where `cost:` begins); we don't pin exact byte offsets — only
    // that the span is non-empty and points into the source.
    let src = "ability X { target: self cost: 30 mana cooldown: 1s heal 10 }";
    let file = parse_ability_file(src).expect("must parse");
    let cost = file.abilities[0]
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cost(c) = h { Some(c) } else { None })
        .unwrap();
    assert!(cost.span.start < cost.span.end, "span must be non-empty");
    assert!(cost.span.end <= src.len(), "span must be within source");
    // The span begins at the `cost` key — assert the slice contains
    // "cost".
    let slice = &src[cost.span.start..cost.span.end];
    assert!(
        slice.contains("cost"),
        "span slice should cover the `cost` token; got `{slice}`"
    );
}

#[test]
fn unknown_resource_is_parse_error() {
    let src = "ability X { target: self cost: 10 elemental_dust cooldown: 1s heal 1 }";
    let err = parse_ability_file(src).expect_err("unknown resource must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("elemental_dust") || msg.contains("unknown cost resource"),
        "expected unknown-resource diagnostic; got: {msg}"
    );
}

#[test]
fn cost_negative_amount_rejected() {
    let src = "ability X { target: self cost: -5 mana cooldown: 1s heal 1 }";
    let err = parse_ability_file(src).expect_err("negative cost must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("cost amount must be >= 0") || msg.contains("must be >= 0"),
        "expected negative-cost diagnostic; got: {msg}"
    );
}

#[test]
fn duplicate_cost_header_rejected() {
    let src = "ability X { target: self cost: 5 cost: 7 cooldown: 1s heal 1 }";
    let err = parse_ability_file(src).expect_err("duplicate cost must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate `cost:`"),
        "expected duplicate-cost diagnostic; got: {msg}"
    );
}

#[test]
fn duplicate_toggle_marker_rejected() {
    let src = "ability X { target: self toggle toggle cooldown: 1s heal 1 }";
    let err = parse_ability_file(src).expect_err("duplicate toggle must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate `toggle`"),
        "expected duplicate-toggle diagnostic; got: {msg}"
    );
}
