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

// ---------------------------------------------------------------------------
// Wave 1.5#2 — `in <shape>(args)` modifier lowering. Effect-statement
// area shapes are captured into `program.per_effect_areas`, indexed
// parallel to `program.effects`. `EffectAreaShape` is the second
// per-effect SoA modifier with variant data — its 4×f32 args block
// round-trips through the engine `ShapeKind` discriminant + `args`
// straight into the SoA pair (`area_kinds` + `area_args`) at pack
// time. Apply handlers default to "single-target effect (use
// program.area)" for any effect that didn't carry the modifier.
// Unknown shape names surface as `UnknownShape` (not
// `ModifierNotImplemented`) so the diagnostic names the offending
// shape rather than the slot.
// ---------------------------------------------------------------------------

#[test]
fn lowering_in_circle() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::ShapeKind;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 in circle(2.5) }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("circle must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(prog.per_effect_areas.len(), 1, "one effect → one area slot");
    let area = prog.per_effect_areas[0].expect("slot 0 has shape");
    assert_eq!(area.kind, ShapeKind::Circle);
    assert_eq!(area.args, [2.5, 0.0, 0.0, 0.0]);
}

#[test]
fn lowering_in_cone() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::ShapeKind;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 in cone(8.0, 45.0) }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("cone must lower");
    let area = prog.per_effect_areas[0].expect("slot 0 has shape");
    assert_eq!(area.kind, ShapeKind::Cone);
    assert_eq!(area.args, [8.0, 45.0, 0.0, 0.0]);
}

#[test]
fn lowering_in_line() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::ShapeKind;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 in line(10, 1.5) }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("line must lower");
    let area = prog.per_effect_areas[0].expect("slot 0 has shape");
    assert_eq!(area.kind, ShapeKind::Line);
    assert_eq!(area.args, [10.0, 1.5, 0.0, 0.0]);
}

#[test]
fn lowering_in_unknown_shape_errors() {
    use dsl_compiler::ability_lower::LowerError;
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 in spaghetti(5) }"
    );
    match err {
        LowerError::UnknownShape { ref shape, .. } => {
            assert_eq!(shape, "spaghetti", "shape name should round-trip in error");
        }
        other => panic!("expected UnknownShape(spaghetti); got {other:?}"),
    }
}

#[test]
fn lowering_no_in_is_none() {
    // Bare effect with no `in <shape>` modifier: the lowering pass
    // leaves `program.per_effect_areas` empty (apply handlers treat
    // empty + None identically as "single-target effect"). Keeps Wave
    // 1 corpus output bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("bare damage must lower");
    assert!(
        prog.per_effect_areas.is_empty(),
        "no `in` modifier → empty per_effect_areas smallvec; got {:?}",
        prog.per_effect_areas,
    );
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

// Wave 1.5#7 — `when <cond> [else <cond>]` lowering. The predicate is
// captured into `program.when_per_effect[i]` as verbatim source text
// (the AST stores it as `EffectCondition::when_cond: String`); engine
// code does not parse it. Apply handlers wire the parse + evaluate
// later. Slot `None` = no gate, "always fires" semantics.
#[test]
fn lowering_when_simple() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 when target.hp < 30 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("when modifier must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(prog.when_per_effect.len(), 1, "one effect → one when slot");
    let cond = prog.when_per_effect[0].as_ref().expect("slot populated");
    assert_eq!(cond.when_cond, "target.hp < 30");
    assert!(cond.else_cond.is_none(), "no else clause");
}

#[test]
fn lowering_when_no_modifier_is_empty() {
    // Bare effect with no when modifier: the lowering pass leaves
    // `program.when_per_effect` empty (apply handlers treat empty +
    // None identically as "no gate, always fires"). Keeps Wave 1
    // corpus output bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("must lower");
    assert!(prog.when_per_effect.is_empty(), "no when modifier → empty slice");
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

// ---------------------------------------------------------------------------
// Wave 1.5#4 — `+ N% stat_ref` modifier lowering. Effect-statement
// scalings are captured into `program.scalings_per_effect`, OUTER-index
// parallel to `program.effects`. Each inner SmallVec is bounded at
// `MAX_SCALINGS_PER_EFFECT == 2` (LoL/MOBA convention: one or two
// scaling stats per effect). Stat-ref tokens are validated against the
// engine's `ScalingStatRef` vocabulary (8 entries: AD/AP/MaxHP/HP/
// Armor/MR/MoveSpeed/Mana). Apply handlers default to "no scaling, flat
// amount only" for any effect that didn't carry the modifier. The DSL
// parser stores `percent` as `N` (e.g. `30` for `+ 30% AP`); the
// engine stores it as a fraction (`0.30`) so apply handlers can
// multiply directly without an extra `/ 100.0`.
// ---------------------------------------------------------------------------

#[test]
fn lowering_scaling_with_ap() {
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::ScalingStatRef;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 + 30% AP }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("scaling AP must lower");
    assert_eq!(prog.effects.len(), 1);
    assert_eq!(
        prog.scalings_per_effect.len(), 1,
        "one effect → one outer scaling slot",
    );
    let inner = &prog.scalings_per_effect[0];
    assert_eq!(inner.len(), 1, "one scaling on the first effect");
    assert_eq!(inner[0].stat_ref, ScalingStatRef::AbilityPower);
    assert!(
        (inner[0].percent - 0.30).abs() < 1e-6,
        "30% AP must lower to 0.30; got {}",
        inner[0].percent,
    );
}

#[test]
fn lowering_scaling_multiple() {
    // `damage 50 + 30% AP + 20% AD` → two scalings on the first effect.
    // Validates the inner-SmallVec growth path + that aliases (AP/AD)
    // resolve to their long-form discriminants.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    use engine::ability::program::ScalingStatRef;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 + 30% AP + 20% AD }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0])
        .expect("two scalings must lower");
    let inner = &prog.scalings_per_effect[0];
    assert_eq!(inner.len(), 2, "two scalings on the first effect");
    assert_eq!(inner[0].stat_ref, ScalingStatRef::AbilityPower);
    assert!((inner[0].percent - 0.30).abs() < 1e-6);
    assert_eq!(inner[1].stat_ref, ScalingStatRef::AttackDamage);
    assert!((inner[1].percent - 0.20).abs() < 1e-6);
}

#[test]
fn lowering_scaling_unknown_stat_errors() {
    // `spaghetti` is not in the engine's `ScalingStatRef` vocabulary
    // (8 entries today). Surfaced loudly so authors don't silently
    // lose scaling on a typoed stat name.
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 + 30% spaghetti }"
    );
    match err {
        LowerError::UnknownStatRef { ref stat, .. } => {
            assert_eq!(stat, "spaghetti", "stat name should round-trip in error");
        }
        other => panic!("expected UnknownStatRef(spaghetti); got {other:?}"),
    }
}

#[test]
fn lowering_scaling_budget_exceeded_errors() {
    // 3 scalings on a single effect — exceeds `MAX_SCALINGS_PER_EFFECT
    // == 2`. Surfaced loudly so authors get a budget diagnostic.
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 + 30% AP + 20% AD + 10% MaxHP }"
    );
    match err {
        LowerError::ScalingBudgetExceeded { count, max, .. } => {
            assert_eq!(max, 2, "MAX_SCALINGS_PER_EFFECT == 2");
            assert!(count >= 3, "overflow should report at least 3 scalings; got {count}");
        }
        other => panic!("expected ScalingBudgetExceeded; got {other:?}"),
    }
}

#[test]
fn lowering_no_scaling_is_empty() {
    // Bare effect with no `+ N% stat_ref` modifier: the lowering pass
    // leaves `program.scalings_per_effect` empty (apply handlers treat
    // empty + per-slot-empty identically as "flat amount only").
    // Keeps Wave 1 corpus output bit-stable.
    use dsl_ast::parse_ability_file;
    use dsl_compiler::ability_lower::lower_ability_decl;
    let file = parse_ability_file(
        "ability X { target: enemy cooldown: 1s damage 50 }"
    ).expect("parser");
    let prog = lower_ability_decl(&file.abilities[0]).expect("bare damage must lower");
    assert!(
        prog.scalings_per_effect.is_empty(),
        "no scaling modifier → empty scalings_per_effect smallvec; got {:?}",
        prog.scalings_per_effect,
    );
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
fn slot_order_in_lowers_alongside_for_error_on_damage() {
    // Wave 1.5#2 retired the `in` short-circuit. With both `in` and
    // `for` populated on a non-duration-bearing verb (damage), the
    // `in` aggregator runs to completion (storing the shape in
    // `program.per_effect_areas`) and THEN the `for` slot trips with
    // ModifierNotImplemented{for} — DoT semantics need a new EffectOp
    // variant that hasn't landed yet. Stable diagnostic for authors:
    // the offending slot is named, not "in".
    let err = lower_inline(
        "ability X { target: enemy cooldown: 1s damage 50 in circle(2.0) for 3s }",
    );
    assert_modifier(err, "for");
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
fn lowering_lol_corpus_excerpt_lowers_with_in_and_tags() {
    // Real-world-shaped ability source. The brief's named canary
    // (Aatrox.ability) needs Wave 1.4 (`recast:` / `deliver` blocks),
    // and most LoL hero files mix Wave 1.1 headers (`cost:`) with
    // Wave 1.5 modifiers. Renekton's `CulltheMeek` body uses ONLY
    // Wave 1.5 modifier surfaces (`in circle(N)`, `[TAG: N]`) with
    // Wave 1.0 headers, exercising the canonical real-world shape.
    //
    // Wave 1.5#2 retired the `in` short-circuit; the `[TAG]` slot was
    // retired in Wave 1.5#1. Both modifiers now lower cleanly into
    // their respective per-effect SoA columns, so the corpus excerpt
    // round-trips end-to-end. This guards that the per-ability
    // aggregators handle a multi-effect program without interfering.
    use engine::ability::program::{AbilityTag, ShapeKind};
    let src = r#"
ability CulltheMeek {
    target: self
    cooldown: 7s
    hint: damage

    damage 15 in circle(4.0) [PHYSICAL: 50]
    heal 15
}"#;
    let file = dsl_ast::parse_ability_file(src).expect("parser");
    let prog = dsl_compiler::ability_lower::lower_ability_decl(&file.abilities[0])
        .expect("CulltheMeek excerpt must lower");
    assert_eq!(prog.effects.len(), 2, "damage + heal");
    // First effect: in circle(4.0) — second effect (heal) has no shape.
    assert_eq!(prog.per_effect_areas.len(), 2, "areas vec parallel to effects");
    let area0 = prog.per_effect_areas[0].expect("damage has shape");
    assert_eq!(area0.kind, ShapeKind::Circle);
    assert_eq!(area0.args[0], 4.0, "circle radius");
    assert!(prog.per_effect_areas[1].is_none(), "heal has no shape");
    // Tag aggregation: PHYSICAL: 50.
    assert_eq!(prog.tags.len(), 1);
    assert_eq!(prog.tags[0].0, AbilityTag::Physical);
    assert_eq!(prog.tags[0].1, 50.0);
}
