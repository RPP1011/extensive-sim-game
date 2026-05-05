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

#[test]
fn lowering_chance_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 chance 25% }");
    assert_modifier(err, "chance");
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
fn lowering_stacking_with_chance_lowers_stacking_then_errors_on_chance() {
    // Wave 1.5#3 retires the stacking short-circuit. With both
    // `chance` and `stacking` present, the lowering pass surfaces the
    // chance modifier error (slot 5 in spec §6.1) — stacking (slot 6)
    // would have been the loser anyway. The point is that stacking
    // alone NO LONGER errors, but chance still does.
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s heal 10 chance 25% stacking refresh }",
    );
    assert_modifier(err, "chance");
}

#[test]
fn lowering_scaling_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: enemy cooldown: 1s damage 50 + 30% AP }");
    assert_modifier(err, "scaling");
}

#[test]
fn lowering_lifetime_modifier_returns_unimplemented() {
    let err = lower_inline("ability X { target: self cooldown: 1s shield 100 until_caster_dies }");
    assert_modifier(err, "lifetime");
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
    // aggregation pass BEFORE the chance modifier short-circuit gets
    // a chance to fire — so the diagnostic is "unknown tag", not
    // "chance modifier not implemented".
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 [FIRE: 60] chance 25% }",
    );
    match err {
        LowerError::UnknownTag { ref tag, .. } => assert_eq!(tag, "FIRE"),
        other => panic!("expected UnknownTag(FIRE) before chance check; got {other:?}"),
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
