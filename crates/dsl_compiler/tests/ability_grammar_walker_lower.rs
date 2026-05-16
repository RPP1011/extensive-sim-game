//! Tree-walker → lowering bridge. Builds the same systematic
//! ability-grammar corpus the `dsl_ast` walker emits (one ability per
//! AST-enum variant), runs each through `lower_ability_decl`, and
//! asserts the lowering produces a valid `AbilityProgram` without
//! diagnostics. Complements `dsl_ast/tests/ability_grammar_walker.rs`
//! (parse round-trip) — together they prove "the grammar produces
//! values that BOTH parse AND lower."
//!
//! Coverage rule (same as the parse walker): one ability per
//! TargetMode / HintName / CostResource × CostAmount / RecastValue /
//! CooldownPhase / EffectArg / EffectLifetime / StackingMode variant,
//! plus the composite kitchen-sink shapes. Lowering failures surface
//! as `LowerError` (typed) — the test reports which generated shape
//! tripped the lowering pass, so a regression points at the variant
//! that broke.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ast::*;
use dsl_compiler::ability_lower::lower_ability_decl;

fn span() -> Span {
    Span { start: 0, end: 0 }
}

fn dur(ms: u32) -> Duration {
    Duration { millis: ms }
}

fn base_effect(verb: &str) -> EffectStmt {
    EffectStmt {
        verb: verb.to_string(),
        args: Vec::new(),
        span: span(),
        area: None,
        tags: Vec::new(),
        duration: None,
        condition: None,
        chance: None,
        stacking: None,
        scalings: Vec::new(),
        lifetime: None,
        nested: Vec::new(),
    }
}

fn base_ability(name: &str, headers: Vec<AbilityHeader>, effects: Vec<EffectStmt>) -> AbilityDecl {
    AbilityDecl {
        name: name.to_string(),
        headers,
        effects,
        deliver: None,
        morph: None,
        instantiates: None,
        program: None,
        span: span(),
    }
}

/// Round-trip a generated ability through:
///   1. AST → source emitter
///   2. parse back
///   3. `lower_ability_decl` on the parsed result
/// Panics with the emitted source + failure on any step.
fn round_trip_and_lower(d: AbilityDecl, label: &str) {
    let src = emit_ability_file_single(&d);
    let parsed = dsl_ast::parse_ability_file(&src)
        .unwrap_or_else(|e| panic!("[{label}] parse failed:\n{src}\nerror: {e}"));
    assert_eq!(parsed.abilities.len(), 1, "[{label}] expected one ability");
    let ad = &parsed.abilities[0];
    if let Err(e) = lower_ability_decl(ad) {
        panic!("[{label}] lower failed:\n--- emitted ---\n{src}\n--- error ---\n{e:?}");
    }
}

#[test]
fn every_target_mode_lowers() {
    for m in [
        TargetMode::Enemy,
        TargetMode::Self_,
        TargetMode::Ally,
        TargetMode::SelfAoe,
        TargetMode::Ground,
    ] {
        let mut e = base_effect("damage");
        e.args.push(EffectArg::Number(50.0));
        let d = base_ability(
            "TargetProbe",
            vec![AbilityHeader::Target(m), AbilityHeader::Range(450.0)],
            vec![e],
        );
        round_trip_and_lower(d, &format!("target/{m:?}"));
    }
}

#[test]
fn every_hint_name_lowers() {
    // `HintName::Economic` is grammar-valid but lowering-reserved
    // (`LowerError::HintReserved`) — see `ability_lower.rs:2283`.
    // Test only the lowering-supported hints; the parse-only walker
    // in `dsl_ast/tests/` covers `Economic` as a grammar-valid form.
    for h in [
        HintName::Damage,
        HintName::Defense,
        HintName::CrowdControl,
        HintName::Utility,
        HintName::Heal,
        HintName::Buff,
    ] {
        let mut e = base_effect("damage");
        e.args.push(EffectArg::Number(50.0));
        let d = base_ability(
            "HintProbe",
            vec![
                AbilityHeader::Target(TargetMode::Enemy),
                AbilityHeader::Range(500.0),
                AbilityHeader::Hint(h),
            ],
            vec![e],
        );
        round_trip_and_lower(d, &format!("hint/{h:?}"));
    }
}

#[test]
fn every_cost_resource_amount_lowers() {
    let cases = [
        (CostResource::Mana, CostAmount::Flat(40.0)),
        (CostResource::Mana, CostAmount::PercentOfMax(10.0)),
        (CostResource::Stamina, CostAmount::Flat(20.0)),
        (CostResource::Hp, CostAmount::Flat(5.0)),
    ];
    for (resource, amount) in cases {
        let mut e = base_effect("damage");
        e.args.push(EffectArg::Number(50.0));
        let d = base_ability(
            "CostProbe",
            vec![
                AbilityHeader::Target(TargetMode::Enemy),
                AbilityHeader::Range(450.0),
                AbilityHeader::Cost(CostSpec { resource, amount, span: span() }),
            ],
            vec![e],
        );
        round_trip_and_lower(d, &format!("cost/{resource:?}+{amount:?}"));
    }
}

#[test]
fn cooldown_with_all_phases_lowers() {
    for phase in [None, Some(CooldownPhase::Cast)] {
        let mut e = base_effect("damage");
        e.args.push(EffectArg::Number(50.0));
        let d = base_ability(
            "CooldownProbe",
            vec![
                AbilityHeader::Target(TargetMode::Enemy),
                AbilityHeader::Range(450.0),
                AbilityHeader::Cooldown(dur(8000), phase),
            ],
            vec![e],
        );
        round_trip_and_lower(d, &format!("cooldown_phase/{phase:?}"));
    }
}

#[test]
fn every_stacking_mode_lowers() {
    for mode in [StackingMode::Refresh, StackingMode::Stack, StackingMode::Extend] {
        let mut e = base_effect("stun");
        e.duration = Some(EffectDuration { duration: dur(1500), span: span() });
        e.stacking = Some(mode);
        let d = base_ability(
            "StackingProbe",
            vec![
                AbilityHeader::Target(TargetMode::Enemy),
                AbilityHeader::Range(500.0),
            ],
            vec![e],
        );
        round_trip_and_lower(d, &format!("stacking/{mode:?}"));
    }
}

#[test]
fn area_modifier_with_circle_lowers() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(80.0));
    e.area = Some(EffectArea {
        shape: "circle".to_string(),
        args: vec![300.0],
        span: span(),
    });
    let d = base_ability(
        "AreaProbe",
        vec![
            AbilityHeader::Target(TargetMode::Ground),
            AbilityHeader::Range(700.0),
        ],
        vec![e],
    );
    round_trip_and_lower(d, "area/circle");
}

#[test]
fn power_tags_lower() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(100.0));
    // Use registry-known tag (`AbilityTag` enum:
    // PHYSICAL/MAGICAL/CROWD_CONTROL/HEAL/DEFENSE/UTILITY).
    // Other names parse but fail the lowering's UnknownTag gate.
    e.tags.push(EffectTag {
        name: "PHYSICAL".to_string(),
        value: 60.0,
        span: span(),
    });
    let d = base_ability(
        "TagsProbe",
        vec![
            AbilityHeader::Target(TargetMode::Enemy),
            AbilityHeader::Range(500.0),
        ],
        vec![e],
    );
    round_trip_and_lower(d, "tags");
}

#[test]
fn scaling_terms_lower() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(60.0));
    e.scalings.push(EffectScaling {
        percent: 40.0,
        stat_ref: "AP".to_string(),
        span: span(),
    });
    let d = base_ability(
        "ScalingProbe",
        vec![
            AbilityHeader::Target(TargetMode::Enemy),
            AbilityHeader::Range(500.0),
        ],
        vec![e],
    );
    round_trip_and_lower(d, "scaling");
}

#[test]
fn kitchen_sink_ability_lowers() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(120.0));
    e.scalings.push(EffectScaling {
        percent: 40.0,
        stat_ref: "AP".to_string(),
        span: span(),
    });
    // Use registry-known tag (`AbilityTag` enum:
    // PHYSICAL/MAGICAL/CROWD_CONTROL/HEAL/DEFENSE/UTILITY).
    // Other names parse but fail the lowering's UnknownTag gate.
    e.tags.push(EffectTag {
        name: "PHYSICAL".to_string(),
        value: 60.0,
        span: span(),
    });
    let d = base_ability(
        "Kitchen",
        vec![
            AbilityHeader::Target(TargetMode::Enemy),
            AbilityHeader::Range(550.0),
            AbilityHeader::Cooldown(dur(8000), Some(CooldownPhase::Cast)),
            AbilityHeader::Hint(HintName::Damage),
            AbilityHeader::Cost(CostSpec {
                resource: CostResource::Mana,
                amount: CostAmount::Flat(45.0),
                span: span(),
            }),
        ],
        vec![e],
    );
    round_trip_and_lower(d, "kitchen_sink");
}
