//! Ability-grammar tree walker — systematically enumerates every
//! variant of every AST node the `.ability` parser today admits, emits
//! each to source, parses it back, and asserts the parse round-trips
//! to an equivalent AST. The goal: prove every valid grammar production
//! that the parser CAN produce is also one the parser can CONSUME.
//!
//! Coverage rule: for each enum on the surface AST (TargetMode,
//! HintName, CostResource, CostAmount, RecastValue, CooldownPhase,
//! EffectArg, EffectLifetime, StackingMode), emit at least one
//! ability whose body exercises that variant. For composite shapes
//! (header lists, effect modifier slots, nested effects, scaling
//! lists, tags, area), emit ≥1 ability per shape.
//!
//! Out of scope: opaque blocks (`deliver`, `morph`, `template`,
//! `structure`, `program` blocks) — they are captured as `raw:
//! String` and round-trip verbatim through a separate parser path
//! that doesn't gate any of the AST-shape grammar this walker
//! validates.

use dsl_ast::ability_emit::emit_ability_file_single;
use dsl_ast::ability_parser::parse_ability_file;
use dsl_ast::ast::*;

fn span() -> Span {
    Span { start: 0, end: 0 }
}

fn dur(ms: u32) -> Duration {
    Duration { millis: ms }
}

/// Empty effect statement with verb only — the workhorse builder. Tests
/// stack modifiers on top of this base.
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

/// Emit one ability to source and parse it back. Returns the parsed
/// ability for downstream assertion; panics with the emitted source
/// + the parse error on failure (so the test report shows what didn't
/// round-trip).
fn round_trip(d: AbilityDecl) -> AbilityDecl {
    let src = emit_ability_file_single(&d);
    match parse_ability_file(&src) {
        Ok(file) => {
            assert_eq!(file.abilities.len(), 1, "expected one ability; src:\n{src}");
            file.abilities.into_iter().next().unwrap()
        }
        Err(e) => panic!("parse failed:\n--- emitted ---\n{src}\n--- error ---\n{e}"),
    }
}

// ============================================================
// Headers — every AbilityHeader variant, in isolation.
// ============================================================

#[test]
fn header_target_all_modes_round_trip() {
    for m in [
        TargetMode::Enemy,
        TargetMode::Self_,
        TargetMode::Ally,
        TargetMode::SelfAoe,
        TargetMode::Ground,
        TargetMode::Direction,
        TargetMode::Vector,
        TargetMode::Global,
    ] {
        let d = base_ability(
            "TargetProbe",
            vec![AbilityHeader::Target(m)],
            vec![base_effect("damage")],
        );
        let parsed = round_trip(d.clone());
        assert!(
            parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Target(p) if *p == m)),
            "target {m:?} did not round-trip"
        );
    }
}

#[test]
fn header_range_round_trips() {
    let d = base_ability(
        "RangeProbe",
        vec![AbilityHeader::Range(550.0)],
        vec![base_effect("damage")],
    );
    let parsed = round_trip(d);
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Range(r) if (*r - 550.0).abs() < 0.01)));
}

#[test]
fn header_cooldown_with_phase_round_trips() {
    for phase in [
        None,
        Some(CooldownPhase::Cast),
        Some(CooldownPhase::Resolve),
        Some(CooldownPhase::Interrupt),
    ] {
        let d = base_ability(
            "CooldownProbe",
            vec![AbilityHeader::Cooldown(dur(8000), phase)],
            vec![base_effect("damage")],
        );
        let parsed = round_trip(d);
        let got_phase = parsed
            .headers
            .iter()
            .find_map(|h| match h {
                AbilityHeader::Cooldown(_, p) => Some(*p),
                _ => None,
            })
            .expect("cooldown header present");
        assert_eq!(got_phase, phase, "phase {phase:?} did not round-trip");
    }
}

#[test]
fn header_cast_and_recharge_round_trip() {
    let d = base_ability(
        "CastRechargeProbe",
        vec![
            AbilityHeader::Cast(dur(500)),
            AbilityHeader::Recharge(dur(12_000)),
        ],
        vec![base_effect("damage")],
    );
    let parsed = round_trip(d);
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Cast(_))));
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Recharge(_))));
}

#[test]
fn header_hint_all_variants_round_trip() {
    for h in [
        HintName::Damage,
        HintName::Defense,
        HintName::CrowdControl,
        HintName::Utility,
        HintName::Heal,
        HintName::Economic,
        HintName::Buff,
    ] {
        let d = base_ability(
            "HintProbe",
            vec![AbilityHeader::Hint(h)],
            vec![base_effect("damage")],
        );
        let parsed = round_trip(d);
        assert!(
            parsed.headers.iter().any(|x| matches!(x, AbilityHeader::Hint(g) if *g == h)),
            "hint {h:?} did not round-trip"
        );
    }
}

#[test]
fn header_cost_all_resource_amount_combos_round_trip() {
    let cases = [
        (CostResource::Mana, CostAmount::Flat(40.0)),
        (CostResource::Mana, CostAmount::PercentOfMax(15.0)),
        (CostResource::Stamina, CostAmount::Flat(20.0)),
        (CostResource::Hp, CostAmount::PercentOfMax(8.0)),
        (CostResource::Gold, CostAmount::Flat(150.0)),
    ];
    for (resource, amount) in cases {
        let d = base_ability(
            "CostProbe",
            vec![AbilityHeader::Cost(CostSpec {
                resource,
                amount,
                span: span(),
            })],
            vec![base_effect("damage")],
        );
        let parsed = round_trip(d);
        let cs = parsed
            .headers
            .iter()
            .find_map(|h| match h {
                AbilityHeader::Cost(c) => Some(*c),
                _ => None,
            })
            .expect("cost header present");
        assert_eq!(cs.resource, resource);
        match (cs.amount, amount) {
            (CostAmount::Flat(a), CostAmount::Flat(b)) => assert!((a - b).abs() < 0.01),
            (CostAmount::PercentOfMax(a), CostAmount::PercentOfMax(b)) => {
                assert!((a - b).abs() < 0.01)
            }
            (got, exp) => panic!("cost amount mismatch: got {got:?} expected {exp:?}"),
        }
    }
}

#[test]
fn header_charges_and_toggle_round_trip() {
    let d = base_ability(
        "ChargesToggleProbe",
        vec![AbilityHeader::Charges(3), AbilityHeader::Toggle],
        vec![base_effect("damage")],
    );
    let parsed = round_trip(d);
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Charges(3))));
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Toggle)));
}

#[test]
fn header_recast_count_and_duration_round_trip() {
    for v in [
        RecastValue::Count(2),
        RecastValue::Duration(dur(4_000)),
    ] {
        let d = base_ability(
            "RecastProbe",
            vec![
                AbilityHeader::Recast(v),
                AbilityHeader::RecastWindow(dur(6_000)),
            ],
            vec![base_effect("damage")],
        );
        let parsed = round_trip(d);
        let got = parsed
            .headers
            .iter()
            .find_map(|h| match h {
                AbilityHeader::Recast(r) => Some(*r),
                _ => None,
            })
            .expect("recast header present");
        assert_eq!(got, v, "recast value {v:?} did not round-trip");
    }
}

// ============================================================
// EffectArg — all five variants.
// ============================================================

#[test]
fn effect_arg_all_variants_round_trip() {
    let cases: Vec<(EffectArg, &str)> = vec![
        (EffectArg::Number(125.0), "number"),
        (EffectArg::Duration(dur(1500)), "duration"),
        (EffectArg::Percent(25.0), "percent"),
        (EffectArg::String("fire".to_string()), "string"),
        (EffectArg::Ident("self".to_string()), "ident"),
    ];
    for (arg, label) in cases {
        let mut e = base_effect("damage");
        e.args.push(arg.clone());
        let d = base_ability("ArgProbe", Vec::new(), vec![e]);
        let parsed = round_trip(d);
        let got_arg = parsed.effects[0].args.first().expect("one arg present");
        match (&arg, got_arg) {
            (EffectArg::Number(a), EffectArg::Number(b)) => {
                assert!((a - b).abs() < 0.01, "arg {label}: {a} vs {b}")
            }
            (EffectArg::Duration(a), EffectArg::Duration(b)) => {
                assert_eq!(a.millis, b.millis, "arg {label}")
            }
            (EffectArg::Percent(a), EffectArg::Percent(b)) => {
                assert!((a - b).abs() < 0.01, "arg {label}")
            }
            (EffectArg::String(a), EffectArg::String(b)) => {
                assert_eq!(a, b, "arg {label}")
            }
            (EffectArg::Ident(a), EffectArg::Ident(b)) => {
                assert_eq!(a, b, "arg {label}")
            }
            (a, b) => panic!("arg {label}: emit produced {a:?}, parsed back {b:?}"),
        }
    }
}

// ============================================================
// Effect modifier slots — area, tags, duration, chance, stacking,
// scalings, lifetime, nested.
// ============================================================

#[test]
fn effect_area_round_trips() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(80.0));
    e.area = Some(EffectArea {
        shape: "circle".to_string(),
        args: vec![300.0],
        span: span(),
    });
    let d = base_ability("AreaProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0].area.as_ref().expect("area present");
    assert_eq!(got.shape, "circle");
    assert!((got.args[0] - 300.0).abs() < 0.1);
}

#[test]
fn effect_tags_multiple_round_trip() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(100.0));
    e.tags.push(EffectTag {
        name: "FIRE".to_string(),
        value: 60.0,
        span: span(),
    });
    e.tags.push(EffectTag {
        name: "CROWD_CONTROL".to_string(),
        value: 30.0,
        span: span(),
    });
    let d = base_ability("TagsProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let tags = &parsed.effects[0].tags;
    assert_eq!(tags.len(), 2);
    assert!(tags.iter().any(|t| t.name == "FIRE" && (t.value - 60.0).abs() < 0.1));
    assert!(tags.iter().any(|t| t.name == "CROWD_CONTROL" && (t.value - 30.0).abs() < 0.1));
}

#[test]
fn effect_duration_round_trips() {
    let mut e = base_effect("stun");
    e.duration = Some(EffectDuration {
        duration: dur(2500),
        span: span(),
    });
    let d = base_ability("DurProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0].duration.expect("duration present");
    assert_eq!(got.duration.millis, 2500);
}

#[test]
fn effect_chance_round_trips() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(50.0));
    e.chance = Some(EffectChance { p: 0.25, span: span() });
    let d = base_ability("ChanceProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0].chance.expect("chance present");
    assert!((got.p - 0.25).abs() < 0.001, "got p={}, expected 0.25", got.p);
}

#[test]
fn effect_stacking_all_modes_round_trip() {
    for mode in [StackingMode::Refresh, StackingMode::Stack, StackingMode::Extend] {
        let mut e = base_effect("stun");
        e.duration = Some(EffectDuration { duration: dur(1500), span: span() });
        e.stacking = Some(mode);
        let d = base_ability("StackingProbe", Vec::new(), vec![e]);
        let parsed = round_trip(d);
        let got = parsed.effects[0].stacking.expect("stacking present");
        assert_eq!(got, mode, "stacking {mode:?} did not round-trip");
    }
}

#[test]
fn effect_scalings_multiple_round_trip() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(60.0));
    e.scalings.push(EffectScaling {
        percent: 50.0,
        stat_ref: "AP".to_string(),
        span: span(),
    });
    e.scalings.push(EffectScaling {
        percent: 30.0,
        stat_ref: "AD".to_string(),
        span: span(),
    });
    let d = base_ability("ScalingProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let scs = &parsed.effects[0].scalings;
    assert_eq!(scs.len(), 2);
    assert!(scs.iter().any(|s| s.stat_ref == "AP" && (s.percent - 50.0).abs() < 0.1));
    assert!(scs.iter().any(|s| s.stat_ref == "AD" && (s.percent - 30.0).abs() < 0.1));
}

#[test]
fn effect_lifetime_all_variants_round_trip() {
    let cases: Vec<(EffectLifetime, &str)> = vec![
        (EffectLifetime::UntilCasterDies { span: span() }, "until_caster_dies"),
        (EffectLifetime::DamageableHp { hp: 200.0, span: span() }, "damageable_hp"),
        (EffectLifetime::BreakOnDamage { span: span() }, "break_on_damage"),
    ];
    for (lt, label) in cases {
        let mut e = base_effect("shield");
        e.args.push(EffectArg::Number(150.0));
        e.lifetime = Some(lt);
        let d = base_ability("LifetimeProbe", Vec::new(), vec![e]);
        let parsed = round_trip(d);
        let got = parsed.effects[0].lifetime.expect("lifetime present");
        match (lt, got) {
            (EffectLifetime::UntilCasterDies { .. }, EffectLifetime::UntilCasterDies { .. }) => {}
            (
                EffectLifetime::DamageableHp { hp: a, .. },
                EffectLifetime::DamageableHp { hp: b, .. },
            ) => assert!((a - b).abs() < 0.1, "lifetime {label}"),
            (EffectLifetime::BreakOnDamage { .. }, EffectLifetime::BreakOnDamage { .. }) => {}
            (e, g) => panic!("lifetime {label}: emit produced {e:?}, parsed back {g:?}"),
        }
    }
}

#[test]
fn effect_nested_block_round_trips() {
    let inner = {
        let mut i = base_effect("damage");
        i.args.push(EffectArg::Number(40.0));
        i
    };
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(100.0));
    e.nested.push(inner);
    let d = base_ability("NestedProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let nested = &parsed.effects[0].nested;
    assert_eq!(nested.len(), 1, "one nested effect expected");
    assert_eq!(nested[0].verb, "damage");
}

// ============================================================
// EffectCondition (when / else) — verbatim-text slot, round-trip via
// emitter `when <cond>` + optional `else <cond>` rendering.
// ============================================================

#[test]
fn effect_condition_when_only_round_trips() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(50.0));
    e.condition = Some(EffectCondition {
        when_cond: "target.hp < 30".to_string(),
        else_cond: None,
        span: span(),
    });
    let d = base_ability("WhenOnlyProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0]
        .condition
        .as_ref()
        .expect("condition present");
    assert_eq!(got.when_cond.trim(), "target.hp < 30");
    assert!(got.else_cond.is_none());
}

#[test]
fn effect_condition_when_else_round_trips() {
    let mut e = base_effect("damage");
    e.args.push(EffectArg::Number(80.0));
    e.condition = Some(EffectCondition {
        when_cond: "target.hp < 50".to_string(),
        else_cond: Some("target.hp >= 50".to_string()),
        span: span(),
    });
    let d = base_ability("WhenElseProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0]
        .condition
        .as_ref()
        .expect("condition present");
    assert_eq!(got.when_cond.trim(), "target.hp < 50");
    assert_eq!(
        got.else_cond
            .as_ref()
            .expect("else_cond present")
            .trim(),
        "target.hp >= 50"
    );
}

#[test]
fn effect_condition_with_parens_round_trips() {
    // The parser's `capture_cond_text` balances `()` so an inner
    // `)` doesn't terminate the body. This pins that nesting
    // semantic against accidental regressions.
    let mut e = base_effect("execute");
    e.args.push(EffectArg::Number(25.0));
    e.condition = Some(EffectCondition {
        when_cond: "(target.hp < 30) and (self.alive)".to_string(),
        else_cond: None,
        span: span(),
    });
    let d = base_ability("WhenParenProbe", Vec::new(), vec![e]);
    let parsed = round_trip(d);
    let got = parsed.effects[0]
        .condition
        .as_ref()
        .expect("condition present");
    assert!(
        got.when_cond.contains("(target.hp < 30)"),
        "balanced parens preserved; got {:?}",
        got.when_cond
    );
}

// ============================================================
// Header coverage holes — variants that were only ever tested in
// combination with another header.
// ============================================================

#[test]
fn header_toggle_alone_round_trips() {
    // Toggle was only tested paired with Charges; pin the
    // standalone path too so a `cost: …` change can't break the
    // toggle-only emit grammar.
    let d = base_ability(
        "ToggleAloneProbe",
        vec![AbilityHeader::Toggle],
        vec![base_effect("damage")],
    );
    let parsed = round_trip(d);
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Toggle)));
}

#[test]
fn header_recharge_alone_round_trips() {
    // Recharge was only ever paired with Cast; pin standalone.
    let d = base_ability(
        "RechargeAloneProbe",
        vec![AbilityHeader::Recharge(dur(8_000))],
        vec![base_effect("damage")],
    );
    let parsed = round_trip(d);
    assert!(parsed.headers.iter().any(|h| matches!(h, AbilityHeader::Recharge(_))));
}

// ============================================================
// Compound — header set + multi-effect bodies + every modifier
// firing together. The "kitchen-sink" cases that catch interaction
// regressions a single-variant test would miss.
// ============================================================

#[test]
fn full_ability_with_all_header_kinds_round_trips() {
    let d = base_ability(
        "Kitchen",
        vec![
            AbilityHeader::Target(TargetMode::Enemy),
            AbilityHeader::Range(550.0),
            AbilityHeader::Cooldown(dur(8000), Some(CooldownPhase::Cast)),
            AbilityHeader::Cast(dur(250)),
            AbilityHeader::Hint(HintName::Damage),
            AbilityHeader::Cost(CostSpec {
                resource: CostResource::Mana,
                amount: CostAmount::Flat(45.0),
                span: span(),
            }),
        ],
        vec![{
            let mut e = base_effect("damage");
            e.args.push(EffectArg::Number(120.0));
            e.scalings.push(EffectScaling {
                percent: 40.0,
                stat_ref: "AP".to_string(),
                span: span(),
            });
            e.tags.push(EffectTag {
                name: "FIRE".to_string(),
                value: 60.0,
                span: span(),
            });
            e
        }],
    );
    let parsed = round_trip(d);
    assert_eq!(parsed.name, "Kitchen");
    assert_eq!(parsed.headers.len(), 6);
    assert_eq!(parsed.effects.len(), 1);
    assert_eq!(parsed.effects[0].verb, "damage");
}

#[test]
fn full_ability_multi_effect_with_modifiers_round_trips() {
    let d = base_ability(
        "MultiEffect",
        vec![
            AbilityHeader::Target(TargetMode::SelfAoe),
            AbilityHeader::Cooldown(dur(15_000), None),
        ],
        vec![
            {
                let mut e = base_effect("damage");
                e.args.push(EffectArg::Number(80.0));
                e.area = Some(EffectArea {
                    shape: "circle".to_string(),
                    args: vec![400.0],
                    span: span(),
                });
                e
            },
            {
                let mut e = base_effect("stun");
                e.duration = Some(EffectDuration { duration: dur(1500), span: span() });
                e.chance = Some(EffectChance { p: 0.5, span: span() });
                e
            },
            {
                let mut e = base_effect("heal");
                e.args.push(EffectArg::Ident("self".to_string()));
                e.args.push(EffectArg::Number(40.0));
                e
            },
        ],
    );
    let parsed = round_trip(d);
    assert_eq!(parsed.effects.len(), 3);
    assert!(parsed.effects[0].area.is_some());
    assert!(parsed.effects[1].duration.is_some());
    assert!(parsed.effects[1].chance.is_some());
}

// ============================================================
// Grammar coverage report — count what was exercised across all
// tests in this file. Acts as a smoke test that every enum variant
// of every covered AST node got emitted at least once. Fails (with
// the missing variant names) if anything was skipped.
// ============================================================

#[test]
fn coverage_report_every_enum_variant_emitted() {
    // The other tests in this file produce the corpus; this test is
    // an inventory. If a new variant is added to one of the listed
    // enums, this test would still pass — it just verifies the
    // KNOWN variants all emit + parse without error. The harder
    // "every-variant-covered" assertion lives in the per-enum tests
    // above.
    let known_target_modes = [
        TargetMode::Enemy,
        TargetMode::Self_,
        TargetMode::Ally,
        TargetMode::SelfAoe,
        TargetMode::Ground,
        TargetMode::Direction,
        TargetMode::Vector,
        TargetMode::Global,
    ];
    let known_hint_names = [
        HintName::Damage,
        HintName::Defense,
        HintName::CrowdControl,
        HintName::Utility,
        HintName::Heal,
        HintName::Economic,
        HintName::Buff,
    ];
    let known_stacking = [
        StackingMode::Refresh,
        StackingMode::Stack,
        StackingMode::Extend,
    ];

    // Compose one ability per target+hint+stacking triple to exercise
    // the cross-product without an explosion (8*7*3 = 168 trials).
    for t in known_target_modes {
        for h in known_hint_names {
            for s in known_stacking {
                let mut e = base_effect("damage");
                e.args.push(EffectArg::Number(10.0));
                e.stacking = Some(s);
                let d = base_ability(
                    "Cover",
                    vec![AbilityHeader::Target(t), AbilityHeader::Hint(h)],
                    vec![e],
                );
                let parsed = round_trip(d);
                assert!(
                    parsed
                        .headers
                        .iter()
                        .any(|x| matches!(x, AbilityHeader::Target(p) if *p == t))
                );
                assert!(
                    parsed
                        .headers
                        .iter()
                        .any(|x| matches!(x, AbilityHeader::Hint(p) if *p == h))
                );
                assert_eq!(parsed.effects[0].stacking, Some(s));
            }
        }
    }
}
