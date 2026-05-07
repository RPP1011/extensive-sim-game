//! #136 dispatcher-discriminant guard.
//!
//! The WGSL dispatcher kernel emitted by
//! `dsl_compiler::cg::emit::wgsl_body::lower_cg_stmt_to_wgsl`'s
//! `CgStmt::ApplyAbility` arm (slice β, commit `04a71d94`) hardcodes
//! the per-variant discriminants in the WGSL switch:
//!
//!     if      (kind == 0u) { /* Damage */ }
//!     else if (kind == 1u) { /* Heal   */ }
//!     else if (kind == 3u) { /* Stun   */ }
//!     else if (kind == 4u) { /* Slow   */ }
//!
//! These constants must agree with the engine's `#[repr(u8)]`
//! discriminants on `EffectOp`. If a future commit renumbers a
//! variant (re-ordering the enum body, inserting a new variant in the
//! middle, etc.), the WGSL switch silently mis-routes effects to the
//! wrong apply path. The pack-corpus test (`packed_registry_lol_corpus.rs`)
//! catches missing pack arms but NOT discriminant drift on already-
//! covered variants.
//!
//! This test loops `EffectOp::pack`'s output for the four duel_abilities
//! staples and asserts each maps to the discriminant the dispatcher
//! emits. A renumbering would surface here as a single legible diff.

use engine::ability::{
    AbilityProgram, EFFECT_KIND_EMPTY, EffectOp, Gate, MAX_EFFECTS_PER_PROGRAM,
    PackedAbilityRegistry,
};

fn pack_one(effect: EffectOp) -> u32 {
    let prog = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
        [effect],
    );
    let mut builder = engine::ability::AbilityRegistryBuilder::new();
    builder.register(prog);
    let registry = builder.build();
    let packed = PackedAbilityRegistry::pack(&registry);
    let kind = packed.effect_kinds[0];
    assert_ne!(
        kind, EFFECT_KIND_EMPTY,
        "pack must produce a non-empty discriminant for a non-empty effect"
    );
    kind
}

#[test]
fn damage_packs_to_discriminant_zero_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Damage { amount: 30.0 }),
        0,
        "Damage discriminant — WGSL dispatcher arm `kind == 0u` depends on this"
    );
}

#[test]
fn heal_packs_to_discriminant_one_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Heal { amount: 25.0 }),
        1,
        "Heal discriminant — WGSL dispatcher arm `kind == 1u` depends on this"
    );
}

#[test]
fn stun_packs_to_discriminant_three_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Stun { duration_ticks: 10 }),
        3,
        "Stun discriminant — WGSL dispatcher arm `kind == 3u` depends on this"
    );
}

#[test]
fn slow_packs_to_discriminant_four_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Slow { duration_ticks: 10, factor_q8: 128 }),
        4,
        "Slow discriminant — WGSL dispatcher arm `kind == 4u` depends on this"
    );
}

#[test]
fn shield_packs_to_discriminant_two_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Shield { amount: 50.0 }),
        2,
        "Shield discriminant — WGSL dispatcher arm `kind == 2u` depends on this"
    );
}

#[test]
fn root_packs_to_discriminant_eight_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Root { duration_ticks: 10 }),
        8,
        "Root discriminant — WGSL dispatcher arm `kind == 8u` depends on this"
    );
}

#[test]
fn silence_packs_to_discriminant_nine_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Silence { duration_ticks: 10 }),
        9,
        "Silence discriminant — WGSL dispatcher arm `kind == 9u` depends on this"
    );
}

#[test]
fn fear_packs_to_discriminant_ten_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Fear { duration_ticks: 10 }),
        10,
        "Fear discriminant — WGSL dispatcher arm `kind == 10u` depends on this"
    );
}

#[test]
fn taunt_packs_to_discriminant_eleven_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Taunt { duration_ticks: 10 }),
        11,
        "Taunt discriminant — WGSL dispatcher arm `kind == 11u` depends on this"
    );
}

#[test]
fn empty_sentinel_byte_matches_wgsl_dispatcher() {
    // The dispatcher loop's `if (kind == 0xFFu) { continue; }` early-out
    // depends on the sentinel value. Pin it so a renaming of
    // `EFFECT_KIND_EMPTY` doesn't silently break the loop.
    assert_eq!(
        EFFECT_KIND_EMPTY, 0xFF,
        "WGSL dispatcher continues on `kind == 0xFFu`; sentinel must stay 0xFF"
    );
}

#[test]
fn max_effects_per_program_matches_wgsl_dispatcher_loop_bound() {
    // The dispatcher's `for (var i = 0u; i < 6u; ...)` loop bound
    // hardcodes 6u; this test pins the engine const that justifies it.
    assert_eq!(
        MAX_EFFECTS_PER_PROGRAM, 6,
        "WGSL dispatcher loops `i < 6u`; engine const must stay 6"
    );
}
