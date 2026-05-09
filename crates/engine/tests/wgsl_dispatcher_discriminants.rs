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

// Wave 2 piece 2 — movement verbs (`Damage`-shape with f32 distance).
#[test]
fn dash_packs_to_discriminant_twelve_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Dash { distance: 5.0 }),
        12,
        "Dash discriminant — WGSL dispatcher arm `kind == 12u` depends on this"
    );
}

#[test]
fn blink_packs_to_discriminant_thirteen_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Blink { distance: 5.0 }),
        13,
        "Blink discriminant — WGSL dispatcher arm `kind == 13u` depends on this"
    );
}

#[test]
fn knockback_packs_to_discriminant_fourteen_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Knockback { distance: 5.0 }),
        14,
        "Knockback discriminant — WGSL dispatcher arm `kind == 14u` depends on this"
    );
}

#[test]
fn pull_packs_to_discriminant_fifteen_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Pull { distance: 5.0 }),
        15,
        "Pull discriminant — WGSL dispatcher arm `kind == 15u` depends on this"
    );
}

// Wave 2 pieces 7 + 8 — single-duration CC verbs.
#[test]
fn stealth_packs_to_discriminant_27_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Stealth { duration_ticks: 30 }),
        27,
        "Stealth discriminant — WGSL dispatcher arm `kind == 27u` depends on this"
    );
}

#[test]
fn charm_packs_to_discriminant_28_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Charm { duration_ticks: 12 }),
        28,
        "Charm discriminant — WGSL dispatcher arm `kind == 28u` depends on this"
    );
}

#[test]
fn grounded_packs_to_discriminant_29_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Grounded { duration_ticks: 20 }),
        29,
        "Grounded discriminant — WGSL dispatcher arm `kind == 29u` depends on this"
    );
}

#[test]
fn suppress_packs_to_discriminant_30_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Suppress { duration_ticks: 15 }),
        30,
        "Suppress discriminant — WGSL dispatcher arm `kind == 30u` depends on this"
    );
}

// Wave 2 piece 3 — Execute / SelfDamage (Damage-shape, f32 payload).
#[test]
fn execute_packs_to_discriminant_16_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Execute { hp_threshold: 25.0 }),
        16,
        "Execute discriminant — WGSL dispatcher arm `kind == 16u` depends on this"
    );
}

#[test]
fn self_damage_packs_to_discriminant_17_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::SelfDamage { amount: 5.0 }),
        17,
        "SelfDamage discriminant — WGSL dispatcher arm `kind == 17u` depends on this"
    );
}

// Wave 1 economy — TransferGold / ModifyStanding (sign-extended i32).
#[test]
fn transfer_gold_packs_to_discriminant_5_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::TransferGold { amount: -50 }),
        5,
        "TransferGold discriminant — WGSL dispatcher arm `kind == 5u` depends on this"
    );
}

#[test]
fn modify_standing_packs_to_discriminant_6_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::ModifyStanding { delta: -10 }),
        6,
        "ModifyStanding discriminant — WGSL dispatcher arm `kind == 6u` depends on this"
    );
}

// Wave 2 piece 4 — LifeSteal / DamageModify (Slow-shape, dual u32+i16).
#[test]
fn life_steal_packs_to_discriminant_18_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::LifeSteal { duration_ticks: 30, fraction_q8: 76 }),
        18,
        "LifeSteal discriminant — WGSL dispatcher arm `kind == 18u` depends on this"
    );
}

#[test]
fn damage_modify_packs_to_discriminant_19_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::DamageModify { duration_ticks: 30, multiplier_q8: 128 }),
        19,
        "DamageModify discriminant — WGSL dispatcher arm `kind == 19u` depends on this"
    );
}

// DoT / HoT / TimedShield (dual f32 + u32).
#[test]
fn damage_over_time_packs_to_discriminant_20_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::DamageOverTime { amount: 5.0, duration_ticks: 10 }),
        20,
        "DamageOverTime discriminant — WGSL dispatcher arm `kind == 20u` depends on this"
    );
}

#[test]
fn heal_over_time_packs_to_discriminant_21_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::HealOverTime { amount: 5.0, duration_ticks: 10 }),
        21,
        "HealOverTime discriminant — WGSL dispatcher arm `kind == 21u` depends on this"
    );
}

#[test]
fn timed_shield_packs_to_discriminant_22_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::TimedShield { amount: 50.0, duration_ticks: 30 }),
        22,
        "TimedShield discriminant — WGSL dispatcher arm `kind == 22u` depends on this"
    );
}

// Buff / Summon — packed payloads with bit unpacking.
#[test]
fn buff_packs_to_discriminant_23_matching_wgsl_dispatcher() {
    use engine::ability::program::BuffStat;
    assert_eq!(
        pack_one(EffectOp::Buff {
            stat: BuffStat::MoveSpeed,
            magnitude_q8: 76,
            duration_ticks: 50,
        }),
        23,
        "Buff discriminant — WGSL dispatcher arm `kind == 23u` depends on this"
    );
}

#[test]
fn summon_packs_to_discriminant_24_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Summon {
            template_hash: 0xCAFE_BABE,
            count: 3,
            lifetime_ticks: 600,
        }),
        24,
        "Summon discriminant — WGSL dispatcher arm `kind == 24u` depends on this"
    );
}

// Non-combat verbs (Wave 2 phase 1).
#[test]
fn harvest_packs_to_discriminant_25_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Harvest { kind_hash: 0xDEAD_BEEF, amount: 5 }),
        25,
        "Harvest discriminant — WGSL dispatcher arm `kind == 25u` depends on this"
    );
}

#[test]
fn place_voxel_packs_to_discriminant_26_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::PlaceVoxel { kind_hash: 0xFEED_FACE }),
        26,
        "PlaceVoxel discriminant — WGSL dispatcher arm `kind == 26u` depends on this"
    );
}

// Wave 2 piece 8 — Reflect (Slow-shape, dual u32+i16).
#[test]
fn reflect_packs_to_discriminant_31_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Reflect { duration_ticks: 30, fraction_q8: 76 }),
        31,
        "Reflect discriminant — WGSL dispatcher arm `kind == 31u` depends on this"
    );
}

// Wave 3 ToM Phase 1 — `plant_belief` bit-flag belief verb.
#[test]
fn plant_belief_packs_to_discriminant_32_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::PlantBelief { subject_idx: 7, fact_bit: 5 }),
        32,
        "PlantBelief discriminant — WGSL dispatcher arm `kind == 32u` depends on this"
    );
}

// Wave 3 ToM Phase 3 — `observe` self-observe-target verb.
#[test]
fn observe_packs_to_discriminant_33_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Observe { target_observer: 0 }),
        33,
        "Observe discriminant — WGSL dispatcher arm `kind == 33u` depends on this"
    );
}

// Wave 3 ToM Phase 3.5 — `scry` cross-observer access verb.
#[test]
fn scry_packs_to_discriminant_34_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Scry { target_observer: 3, subject_idx: 4 }),
        34,
        "Scry discriminant — WGSL dispatcher arm `kind == 34u` depends on this"
    );
}

// Wave 3 ToM Phase 3.5 — `reveal` one-to-many propagation verb.
#[test]
fn reveal_packs_to_discriminant_35_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Reveal { subject_idx: 4 }),
        35,
        "Reveal discriminant — WGSL dispatcher arm `kind == 35u` depends on this"
    );
}

// Wave 3 ToM Phase 4 — `disguise` deception verb.
#[test]
fn disguise_packs_to_discriminant_36_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Disguise { fake_type: 7, duration_ticks: 200 }),
        36,
        "Disguise discriminant — WGSL dispatcher arm `kind == 36u` depends on this"
    );
}

// Wave 3 ToM Phase 4 — `decoy` deception verb.
#[test]
fn decoy_packs_to_discriminant_37_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Decoy { subject_idx: 4, fake_pos: 0xDEADBEEF }),
        37,
        "Decoy discriminant — WGSL dispatcher arm `kind == 37u` depends on this"
    );
}

// Wave 3 ToM Phase 4 — `erase_belief` deception verb.
#[test]
fn erase_belief_packs_to_discriminant_38_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::EraseBelief { subject_idx: 4, fields: 0b00111111 }),
        38,
        "EraseBelief discriminant — WGSL dispatcher arm `kind == 38u` depends on this"
    );
}

// Lift A — `travel_to` multi-tick travel verb.
#[test]
fn travel_to_packs_to_discriminant_39_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::TravelTo { dest_x_q8: 1280, dest_y_q8: 1280, eta_ticks: 50 }),
        39,
        "TravelTo discriminant — WGSL dispatcher arm `kind == 39u` depends on this"
    );
}

// Lift B — `cast_recipe` production-recipe verb.
#[test]
fn recipe_packs_to_discriminant_40_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::Recipe { recipe_id: 7, target_tool: 0xFF }),
        40,
        "Recipe discriminant — WGSL dispatcher arm `kind == 40u` depends on this"
    );
}

// Lift B — `wear_tool` capital-goods wear verb.
#[test]
fn wear_tool_packs_to_discriminant_41_matching_wgsl_dispatcher() {
    assert_eq!(
        pack_one(EffectOp::WearTool { tool_kind: 4, amount: 64 }),
        41,
        "WearTool discriminant — WGSL dispatcher arm `kind == 41u` depends on this"
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
