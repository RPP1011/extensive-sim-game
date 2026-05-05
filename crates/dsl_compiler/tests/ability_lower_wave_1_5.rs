//! Wave 1.5 lowering tests — verify the nine new effect-statement
//! modifier slots (spec §6.1) error cleanly at the lowering boundary,
//! and that Wave 1 corpus (Strike / ShieldUp / Mend) still lowers
//! without regression.
//!
//! Per `crates/dsl_compiler/src/ability_lower.rs` Wave 1.5 module-level
//! docs: lowering of these modifier slots requires distinct engine
//! schema work (area expansion, status durations, conditional gates,
//! RNG gates, stack tracking, scaling stat refs, voxel lifetimes,
//! nested dispatch). Until each lands, `lower_effect_stmt` surfaces
//! `LowerError::ModifierNotImplemented { verb, modifier, span }` for
//! the first populated modifier slot. The slot-check order mirrors
//! spec §6.1 evaluation order so the diagnostic is stable.

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_lower::{lower_ability_decl, LowerError};

/// Helper: parse one inline ability source and return the
/// `LowerError` from `lower_ability_decl`.
fn lower_inline(src: &str) -> LowerError {
    let file = parse_ability_file(src).expect("parser");
    lower_ability_decl(&file.abilities[0]).expect_err("lowering must error")
}

/// Helper: assert the error is `ModifierNotImplemented` with the given
/// modifier slot name. Returns the carried span for further checks.
fn assert_modifier(err: LowerError, expected: &'static str) -> dsl_ast::ast::Span {
    match err {
        LowerError::ModifierNotImplemented { ref modifier, span, .. } => {
            assert_eq!(
                *modifier, expected,
                "expected modifier `{expected}`; got `{modifier}`"
            );
            span
        }
        other => panic!("expected ModifierNotImplemented; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1. One test per modifier slot
// ---------------------------------------------------------------------------

#[test]
fn lowering_in_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 in circle(2.5) }");
    let span = assert_modifier(err, "in");
    assert!(span.start < span.end, "span must be non-empty");
}

#[test]
fn lowering_tags_aggregate_into_program() {
    // Wave 1.5 modifier lowering #1 (tag vocabulary) — known tag names
    // now lower into program.tags instead of erroring with
    // ModifierNotImplemented{tags}.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::AbilityTag;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 [PHYSICAL: 60] }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("known tag must lower");
    assert_eq!(prog.tags.len(), 1);
    assert_eq!(prog.tags[0].0, AbilityTag::Physical);
    assert_eq!(prog.tags[0].1, 60.0);
}

#[test]
fn lowering_tags_unknown_name_errors() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 [WIND: 30] }");
    match err {
        LowerError::UnknownTag { ref tag, .. } => assert_eq!(tag, "WIND"),
        other => panic!("expected UnknownTag(WIND); got {other:?}"),
    }
}

#[test]
fn lowering_tags_aggregate_across_effects() {
    // Multi-effect summation: PHYSICAL: 30 on damage + PHYSICAL: 40 on
    // a second damage effect → program.tags has Physical=70.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::AbilityTag;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s\n damage 30 [PHYSICAL: 30]\n damage 20 [PHYSICAL: 40]\n}"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("known tags must lower");
    assert_eq!(prog.tags.len(), 1);
    assert_eq!(prog.tags[0].0, AbilityTag::Physical);
    assert_eq!(prog.tags[0].1, 70.0);
}

#[test]
fn lowering_for_duration_on_damage_still_errors() {
    // Wave 1.5#6: `for <duration>` on instant verbs (damage/heal/shield)
    // means DoT/HoT semantics — needs a NEW EffectOp variant. Still
    // surfaces ModifierNotImplemented{for}.
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 5 for 4s }");
    assert_modifier(err, "for");
}

#[test]
fn lowering_for_duration_on_stun_lowers_cleanly() {
    // Wave 1.5#6: `for <duration>` on stateful verbs ACTS as the
    // duration source — no positional arg required.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::EffectOp;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("stun for-mod must lower");
    assert_eq!(prog.effects.len(), 1);
    match &prog.effects[0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(*duration_ticks, 20, "2s = 20 ticks"),
        other => panic!("expected Stun(20); got {other:?}"),
    }
}

#[test]
fn lowering_for_duration_on_slow_lowers_cleanly() {
    // Wave 1.5#6: `slow 0.5 for 4s` — factor positional, duration from
    // for-modifier. Used by LoL corpus.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::EffectOp;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s slow 0.5 for 4s }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("slow for-mod must lower");
    match &prog.effects[0] {
        EffectOp::Slow { duration_ticks, factor_q8 } => {
            assert_eq!(*duration_ticks, 40, "4s = 40 ticks");
            assert_eq!(*factor_q8, 128, "0.5 q8 = 128");
        }
        other => panic!("expected Slow(40, 128); got {other:?}"),
    }
}

#[test]
fn lowering_for_duration_on_lifesteal_lowers_cleanly() {
    // Wave 1.5#6: lifesteal also accepts for-modifier as duration.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::EffectOp;
    let file = parse_ability_file(
        "ability X { target: self cooldown: 8s lifesteal 0.25 for 5s }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("lifesteal for-mod must lower");
    match &prog.effects[0] {
        EffectOp::LifeSteal { duration_ticks, fraction_q8 } => {
            assert_eq!(*duration_ticks, 50, "5s = 50 ticks");
            assert_eq!(*fraction_q8, 64, "0.25 q8 = 64");
        }
        other => panic!("expected LifeSteal(50, 64); got {other:?}"),
    }
}

#[test]
fn lowering_when_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 when target.hp < 30 }");
    assert_modifier(err, "when");
}

// ---------------------------------------------------------------------------
// Wave 1.5#5 — `chance N%` modifier lowering. Effect-statement chance
// gates are captured into `program.chances`, indexed parallel to
// `program.effects`, encoded as Q16 fixed-point u16 (`p * 65535`,
// clamped to `0..=65534` so `u16::MAX = 65535` stays reserved as the
// `CHANCE_NONE_SENTINEL`). Apply handlers default to "always fires"
// for any effect that didn't carry the modifier.
// ---------------------------------------------------------------------------

#[test]
fn lowering_chance_25_percent() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 chance 25% }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("chance 25% must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(prog.chances.len(), 1, "one effect → one chances slot");
    // (0.25 * 65535).round() = 16384.
    assert_eq!(prog.chances[0], Some(16384));
}

#[test]
fn lowering_chance_100_percent_clamped_below_sentinel() {
    // `chance 100%` would naïvely encode to 65535, but that value is
    // reserved as the `CHANCE_NONE_SENTINEL` so the SoA column stays
    // unambiguous. Lowering clamps to 65534 — at 16-bit RNG resolution
    // (`per_agent_u32 & 0xFFFF < 65534`) this is indistinguishable
    // from "always fires".
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 chance 100% }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("chance 100% must lower");
    assert_eq!(prog.chances[0], Some(65534), "sentinel value 65535 reserved");
}

#[test]
fn lowering_chance_0_percent() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 chance 0% }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("chance 0% must lower");
    assert_eq!(prog.chances[0], Some(0));
}

#[test]
fn lowering_no_chance_is_empty() {
    // Bare effect with no chance modifier: the lowering pass leaves
    // `program.chances` empty (apply handlers treat empty + None
    // identically as "always fires"). This keeps Wave 1 corpus output
    // bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("bare damage must lower");
    assert!(
        prog.chances.is_empty(),
        "no chance modifier → empty chances smallvec; got {:?}",
        prog.chances,
    );
}

// ---------------------------------------------------------------------------
// Wave 1.5#3 — `stacking <mode>` modifier lowering. Effect-statement
// stacking modes are captured into `program.stackings`, indexed parallel
// to `program.effects`. Apply handlers default to `Refresh` for any
// effect that didn't carry the modifier (per project_buff_stacking_rule).
// ---------------------------------------------------------------------------

#[test]
fn lowering_stacking_refresh() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::StackingMode;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s stacking refresh }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("stacking refresh must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(prog.stackings.len(), 1, "one effect → one stackings slot");
    assert_eq!(prog.stackings[0], Some(StackingMode::Refresh));
}

#[test]
fn lowering_stacking_stack() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::StackingMode;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s stacking stack }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("stacking stack must lower");
    assert_eq!(prog.stackings[0], Some(StackingMode::Stack));
}

#[test]
fn lowering_stacking_extend() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::StackingMode;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s stacking extend }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("stacking extend must lower");
    assert_eq!(prog.stackings[0], Some(StackingMode::Extend));
}

#[test]
fn lowering_no_stacking_is_empty() {
    // Bare effect with no stacking modifier: the lowering pass leaves
    // `program.stackings` empty (apply handlers treat empty + None
    // identically as Refresh per the project memo). This keeps Wave 1
    // corpus output bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy range: 5.0 cooldown: 1s stun for 2s }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("bare stun must lower");
    assert!(
        prog.stackings.is_empty(),
        "no stacking modifier → empty stackings smallvec; got {:?}",
        prog.stackings,
    );
}

#[test]
fn lowering_stacking_with_chance_lowers_both_modifiers() {
    // Wave 1.5#5 retires the chance short-circuit too. With both
    // `chance` and `stacking` present, BOTH modifiers lower into their
    // respective per-effect SoA columns. This regression-guards that
    // the aggregators don't interfere with each other and that the
    // ordering of slot-checks in `lower_effect_stmt` doesn't
    // accidentally re-error on a now-supported slot.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::StackingMode;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s heal 10 chance 25% stacking refresh }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("chance + stacking together must lower");
    assert_eq!(prog.chances.len(), 1);
    assert_eq!(prog.chances[0], Some(16384), "chance 25% → q16 16384");
    assert_eq!(prog.stackings.len(), 1);
    assert_eq!(prog.stackings[0], Some(StackingMode::Refresh));
}

#[test]
fn lowering_scaling_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 + 30% AP }");
    assert_modifier(err, "scaling");
}

// ---------------------------------------------------------------------------
// Wave 1.5#8 — `until_caster_dies` / `damageable_hp(N)` /
// `break_on_damage` lifetime modifier lowering. Effect-statement
// lifetime modifiers are captured into `program.lifetimes`, indexed
// parallel to `program.effects`. `damageable_hp(N)` is the first
// per-effect SoA modifier with variant data — its f32 hp pool
// round-trips through `LifetimeMode::DamageableHp(hp)` straight into
// the SoA `lifetime_payloads` column at pack time. Apply handlers
// default to "effect persists indefinitely (or for the verb's own
// duration if the verb has one)" for any effect that didn't carry the
// modifier.
// ---------------------------------------------------------------------------

#[test]
fn lowering_lifetime_until_caster_dies() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::LifetimeMode;
    let file = parse_ability_file(
        "ability X { target: self cooldown: 1s shield 100 until_caster_dies }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("until_caster_dies must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(prog.lifetimes.len(), 1, "one effect → one lifetimes slot");
    assert_eq!(prog.lifetimes[0], Some(LifetimeMode::UntilCasterDies));
}

#[test]
fn lowering_lifetime_damageable_hp() {
    // Variant-with-payload — round-trips the f32 hp pool through the
    // engine `LifetimeMode::DamageableHp(hp)` shape unchanged.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::LifetimeMode;
    let file = parse_ability_file(
        "ability X { target: self cooldown: 1s shield 50 damageable_hp(100) }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("damageable_hp must lower");
    assert_eq!(prog.lifetimes[0], Some(LifetimeMode::DamageableHp(100.0)));
}

#[test]
fn lowering_lifetime_break_on_damage() {
    // LoL-style stealth idiom (Akali / Elise / MonkeyKing — see
    // EffectLifetime AST docs).
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::LifetimeMode;
    let file = parse_ability_file(
        "ability X { target: self cooldown: 6s shield 30 break_on_damage }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("break_on_damage must lower");
    assert_eq!(prog.lifetimes[0], Some(LifetimeMode::BreakOnDamage));
}

#[test]
fn lowering_no_lifetime_is_none() {
    // Bare effect with no lifetime modifier: the lowering pass leaves
    // `program.lifetimes` empty (apply handlers treat empty + None
    // identically as "effect persists indefinitely / for the verb's
    // own duration"). Keeps Wave 1 corpus output bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("bare damage must lower");
    assert!(
        prog.lifetimes.is_empty(),
        "no lifetime modifier → empty lifetimes smallvec; got {:?}",
        prog.lifetimes,
    );
}

#[test]
fn lowering_nested_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s heal 50 { stun 1s } }");
    assert_modifier(err, "nested");
}

// ---------------------------------------------------------------------------
// 2. Wave 1 corpus regression: no modifiers, lowers cleanly
// ---------------------------------------------------------------------------

#[test]
fn lowering_wave_1_corpus_still_works() {
    // The Wave 1 corpus (Strike / ShieldUp / Mend) uses no modifier
    // slots; lowering must continue to produce a valid AbilityProgram.
    for (name, src) in [
        ("Strike", "ability Strike { target: enemy range: 5.0 cooldown: 1s hint: damage damage 15 }"),
        ("ShieldUp", "ability ShieldUp { target: self cooldown: 4s hint: defense shield 50 }"),
        ("Mend",     "ability Mend { target: self cooldown: 3s hint: heal heal 20 }"),
    ] {
        let file = parse_ability_file(src).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        let prog = lower_ability_decl(&file.abilities[0])
            .unwrap_or_else(|e| panic!("{name} lowers: {e:?}"));
        assert_eq!(prog.effects.len(), 1, "{name} has one effect");
    }
}

// ---------------------------------------------------------------------------
// 3. Slot-check order: with multiple modifiers, `in` (slot 1) wins
// ---------------------------------------------------------------------------

#[test]
fn slot_order_in_takes_precedence_over_for() {
    // Per spec §6.1 the slot evaluation order is in/tags/for/when/chance/
    // stacking/scaling/lifetime/nested. With both `in` and `for`
    // populated, `in` fires first so the diagnostic points at the
    // shape modifier — stable for authors.
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 in circle(2.0) for 3s }",
    );
    assert_modifier(err, "in");
}

#[test]
fn slot_order_unknown_tag_takes_precedence_over_chance() {
    // FIRE isn't in the engine's AbilityTag vocabulary today (Wave
    // 1.5#1 fixed enum: PHYSICAL/MAGICAL/CROWD_CONTROL/HEAL/DEFENSE/
    // UTILITY). The unknown-tag error fires per-effect during the tag
    // aggregation pass BEFORE the chance aggregator (Wave 1.5#5) gets
    // a turn — so the diagnostic is "unknown tag" rather than a
    // silently-lowered chance gate riding alongside an invalid tag.
    // (Wave 1.5#5 retired the `chance` ModifierNotImplemented error;
    // this test now guards that the per-effect tag pass still
    // short-circuits before the chance aggregator runs.)
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 [FIRE: 60] chance 25% }",
    );
    match err {
        LowerError::UnknownTag { ref tag, .. } => assert_eq!(tag, "FIRE"),
        other => panic!("expected UnknownTag(FIRE) before chance aggregator; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. LoL corpus canary
// ---------------------------------------------------------------------------

#[test]
fn lowering_lol_corpus_excerpt_returns_clear_modifier_message() {
    // Real-world-shaped ability source. The brief's named canary
    // (Aatrox.ability) needs Wave 1.4 (`recast:` / `deliver` blocks),
    // and most LoL hero files mix Wave 1.1 headers (`cost:`) with
    // Wave 1.5 modifiers — those would surface `HeaderNotImplemented`
    // before the modifier check. So we lift Renekton's
    // `CulltheMeek` body verbatim into an inline test source: it uses
    // ONLY Wave 1.5 modifier surfaces (`in circle(N)`, `[TAG: N]`)
    // with Wave 1.0 headers, exercising the canonical real-world
    // shape without the unrelated Wave 1.1 surfaces tripping first.
    //
    // The brief's intent for this test is "sanity that the user gets
    // a useful error" — we assert (a) the modifier-name lands in the
    // message, and (b) per spec §6.1 slot order, `in` (slot 1) fires
    // before `tags` (slot 2).
    let src = r#"
ability CulltheMeek {
    target: self
    cooldown: 7s
    hint: damage

    damage 15 in circle(4.0) [PHYSICAL: 50]
    heal 15
}"#;
    let err = lower_inline(src);
    let span = match &err {
        LowerError::ModifierNotImplemented { modifier, span, .. } => {
            assert_eq!(*modifier, "in", "expected `in` slot to fire first; got `{modifier}`");
            *span
        }
        other => panic!("expected ModifierNotImplemented; got {other:?}"),
    };
    assert!(span.start < span.end, "span must be non-empty");
    let msg = err.to_string();
    assert!(
        msg.contains("modifier slot"),
        "expected modifier-slot diagnostic; got: {msg}"
    );
    assert!(
        msg.contains("`in`"),
        "expected diagnostic to name the `in` slot; got: {msg}"
    );
}
