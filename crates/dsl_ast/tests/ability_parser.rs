//! Wave 1.0 `.ability` parser tests.
//!
//! Coverage:
//! 1. Inline minimal ability parses (header + one effect).
//! 2. A real ability from `dataset/hero_templates/warrior.ability` parses
//!    (`ShieldWall` — the simplest shape with a `for 5s` modifier we
//!    deliberately skip in this slice).
//! 3. All five combat-core verbs (damage / heal / shield / stun / slow)
//!    tokenise as effect statements.
//! 4. Duplicate-header (`cooldown:` twice) fails with a clean error.
//! 5. Empty body (`ability Foo { }`) fails with a clean error.
//! 6. A multi-ability file (warrior excerpt) round-trips three abilities.

use dsl_ast::ast::{AbilityHeader, EffectArg, HintName, TargetMode};
use dsl_ast::parse_ability_file;

#[test]
fn inline_minimal_ability_parses() {
    let src = "ability Foo { target: enemy, range: 5.0, cooldown: 6s, hint: damage damage 50 }";
    let file = parse_ability_file(src).expect("inline minimal must parse");
    assert_eq!(file.abilities.len(), 1);
    let a = &file.abilities[0];
    assert_eq!(a.name, "Foo");
    assert_eq!(a.headers.len(), 4);
    assert!(matches!(a.headers[0], AbilityHeader::Target(TargetMode::Enemy)));
    match a.headers[1] {
        AbilityHeader::Range(v) => assert!((v - 5.0).abs() < 1e-6),
        _ => panic!("expected Range"),
    }
    match a.headers[2] {
        AbilityHeader::Cooldown(d) => assert_eq!(d.millis, 6_000),
        _ => panic!("expected Cooldown"),
    }
    assert!(matches!(a.headers[3], AbilityHeader::Hint(HintName::Damage)));
    assert_eq!(a.effects.len(), 1);
    assert_eq!(a.effects[0].verb, "damage");
    assert_eq!(a.effects[0].args.len(), 1);
    match &a.effects[0].args[0] {
        EffectArg::Number(v) => assert!((*v - 50.0).abs() < 1e-6),
        other => panic!("expected Number(50.0); got {other:?}"),
    }
}

#[test]
fn warrior_shield_wall_parses() {
    // ShieldWall is the simplest hand-authored ability in the corpus —
    // five header lines on three source lines plus one `shield 60 for 5s`
    // effect. The `for 5s` modifier is consumed and discarded by Wave
    // 1.0's `skip_modifier_tail` (modifier capture lands in Wave 1.5).
    let src = r#"
ability ShieldWall {
    target: self
    cooldown: 15s, cast: 200ms
    hint: defense

    shield 60 for 5s
}
"#;
    let file = parse_ability_file(src).expect("ShieldWall must parse");
    assert_eq!(file.abilities.len(), 1);
    let a = &file.abilities[0];
    assert_eq!(a.name, "ShieldWall");
    assert!(matches!(a.headers[0], AbilityHeader::Target(TargetMode::Self_)));
    match a.headers[1] {
        AbilityHeader::Cooldown(d) => assert_eq!(d.millis, 15_000),
        _ => panic!("expected Cooldown header at index 1"),
    }
    match a.headers[2] {
        AbilityHeader::Cast(d) => assert_eq!(d.millis, 200),
        _ => panic!("expected Cast header at index 2"),
    }
    assert!(matches!(a.headers[3], AbilityHeader::Hint(HintName::Defense)));
    assert_eq!(a.effects.len(), 1);
    let e = &a.effects[0];
    assert_eq!(e.verb, "shield");
    assert_eq!(e.args.len(), 1);
    // Wave 1.0 captures only the leading positional `60`; the `for 5s`
    // modifier is dropped by design.
    match &e.args[0] {
        EffectArg::Number(v) => assert!((*v - 60.0).abs() < 1e-6),
        other => panic!("expected Number(60.0); got {other:?}"),
    }
}

#[test]
fn five_combat_core_verbs_tokenise() {
    // All five combat-core verbs from spec §7.1 should tokenise as effect
    // statements with the right verb name. Argument shapes are validated
    // at lowering (Wave 1.6); here we just confirm the parser walks each
    // line cleanly.
    let src = r#"
ability CombatCore {
    target: enemy, range: 4.0
    cooldown: 5s

    damage 50
    heal 30
    shield 25
    stun 1500ms
    slow 0.5 2s
}
"#;
    let file = parse_ability_file(src).expect("combat-core verbs must parse");
    assert_eq!(file.abilities.len(), 1);
    let effects = &file.abilities[0].effects;
    let verbs: Vec<&str> = effects.iter().map(|e| e.verb.as_str()).collect();
    assert_eq!(verbs, vec!["damage", "heal", "shield", "stun", "slow"]);

    // Spot-check argument lex.
    match effects[0].args[0] {
        EffectArg::Number(v) => assert!((v - 50.0).abs() < 1e-6),
        ref other => panic!("damage arg: expected Number(50.0); got {other:?}"),
    }
    match effects[3].args[0] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 1500),
        ref other => panic!("stun arg: expected Duration(1500); got {other:?}"),
    }
    // `slow 0.5 2s` — float factor + duration.
    assert_eq!(effects[4].args.len(), 2);
    match effects[4].args[0] {
        EffectArg::Number(v) => assert!((v - 0.5).abs() < 1e-6),
        ref other => panic!("slow arg 0: expected Number(0.5); got {other:?}"),
    }
    match effects[4].args[1] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 2_000),
        ref other => panic!("slow arg 1: expected Duration(2000); got {other:?}"),
    }
}

#[test]
fn duplicate_header_is_rejected() {
    let src = r#"
ability Dupe {
    target: enemy
    cooldown: 5s
    cooldown: 6s
    damage 10
}
"#;
    let err = parse_ability_file(src).expect_err("duplicate header must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate `cooldown:`"),
        "expected duplicate-header diagnostic; got: {msg}"
    );
}

#[test]
fn empty_body_is_rejected() {
    let src = "ability Empty { }";
    let err = parse_ability_file(src).expect_err("empty body must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("empty body"),
        "expected empty-body diagnostic; got: {msg}"
    );
}

#[test]
fn warrior_file_excerpt_parses_three_abilities() {
    // Three abilities from the real warrior.ability corpus. Whirlwind
    // exercises an `in circle(2.5) [PHYSICAL: 50]` modifier tail (skipped),
    // ShieldWall a `for 5s` tail, HeroicCharge a multi-effect body where
    // `dash to_target` is parsed verb-with-ident-arg.
    let src = r#"
// Warrior abilities

ability Whirlwind {
    target: self_aoe
    cooldown: 8s, cast: 400ms
    hint: damage

    damage 40 in circle(2.5) [PHYSICAL: 50]
    damage 10 in circle(2.5) when hit_count_above(2) [PHYSICAL: 50]
}

ability ShieldWall {
    target: self
    cooldown: 15s, cast: 200ms
    hint: defense

    shield 60 for 5s
}

ability HeroicCharge {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 0ms
    hint: crowd_control

    dash to_target
    damage 35 [PHYSICAL: 50]
    stun 1500ms [CROWD_CONTROL: 70]
}
"#;
    let file = parse_ability_file(src).expect("warrior excerpt must parse");
    assert_eq!(file.abilities.len(), 3);
    assert_eq!(file.abilities[0].name, "Whirlwind");
    assert_eq!(file.abilities[1].name, "ShieldWall");
    assert_eq!(file.abilities[2].name, "HeroicCharge");

    // Whirlwind: two `damage` effects, both with positional `40` / `10`
    // captured before the `in` modifier is skipped.
    let ww = &file.abilities[0];
    assert_eq!(ww.effects.len(), 2);
    assert_eq!(ww.effects[0].verb, "damage");
    assert_eq!(ww.effects[1].verb, "damage");
    match ww.effects[0].args[0] {
        EffectArg::Number(v) => assert!((v - 40.0).abs() < 1e-6),
        ref other => panic!("expected Number(40.0); got {other:?}"),
    }
    match ww.effects[1].args[0] {
        EffectArg::Number(v) => assert!((v - 10.0).abs() < 1e-6),
        ref other => panic!("expected Number(10.0); got {other:?}"),
    }

    // HeroicCharge: `dash to_target` parses as verb=`dash`, arg=Ident.
    let hc = &file.abilities[2];
    assert_eq!(hc.effects.len(), 3);
    assert_eq!(hc.effects[0].verb, "dash");
    match &hc.effects[0].args[0] {
        EffectArg::Ident(s) => assert_eq!(s, "to_target"),
        other => panic!("expected Ident(to_target); got {other:?}"),
    }
    assert_eq!(hc.effects[1].verb, "damage");
    assert_eq!(hc.effects[2].verb, "stun");
    match hc.effects[2].args[0] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 1500),
        ref other => panic!("expected Duration(1500); got {other:?}"),
    }
}

#[test]
fn template_block_unsupported_in_wave_1_1() {
    // `passive` blocks now parse (Wave 1.1); `template` and `structure`
    // remain deferred (Waves 1.2 / 1.3). This test asserts the deferred
    // surfaces still fail loudly.
    let src = r#"
template SomeTpl {
    target: enemy
    damage 10
}
"#;
    let err = parse_ability_file(src).expect_err("template must fail in Wave 1.1");
    let msg = err.to_string();
    assert!(
        msg.contains("template"),
        "expected template-not-supported diagnostic; got: {msg}"
    );
}

#[test]
fn duration_lexer_accepts_seconds_millis_and_floats() {
    // Round-trip the four duration spellings the spec supports.
    let src = r#"
ability DurForms {
    target: self
    cooldown: 5s
    cast: 1500ms

    stun 2s
    slow 0.5 1.5s
    slow 0.7 300ms
}
"#;
    let file = parse_ability_file(src).expect("duration forms must parse");
    let a = &file.abilities[0];
    let cd = a
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cooldown(d) = h { Some(d) } else { None })
        .expect("cooldown header");
    assert_eq!(cd.millis, 5_000);
    let cast = a
        .headers
        .iter()
        .find_map(|h| if let AbilityHeader::Cast(d) = h { Some(d) } else { None })
        .expect("cast header");
    assert_eq!(cast.millis, 1_500);
    // First effect: `stun 2s`.
    match a.effects[0].args[0] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 2_000),
        ref other => panic!("stun arg: expected Duration(2000); got {other:?}"),
    }
    // `slow 0.5 1.5s` — float factor + fractional second.
    match a.effects[1].args[1] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 1_500),
        ref other => panic!("slow arg 1: expected Duration(1500); got {other:?}"),
    }
    // `slow 0.7 300ms` — float factor + ms-suffix.
    match a.effects[2].args[1] {
        EffectArg::Duration(d) => assert_eq!(d.millis, 300),
        ref other => panic!("slow arg 1: expected Duration(300); got {other:?}"),
    }
}
