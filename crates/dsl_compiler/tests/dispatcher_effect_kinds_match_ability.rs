//! Regression: the `apply_ability <id>` dispatcher reads
//! `effect_kinds[ability_slot * MAX_EFFECTS_PER_PROGRAM + i]`, so the
//! registry's slot order is load-bearing for which chronicle kind each
//! ability emits.
//!
//! `dsl_compiler::ability_registry::build_registry` registers programs
//! in the caller-provided iteration order, and the runtime `build_helper`
//! collects `.ability` filenames via `read_dir().sort()` (alphabetical).
//! When a fixture's `.sim` hand-codes ability ids (e.g. `apply_ability 1
//! by self target target` from a verb named `Strike`), the literal id
//! must therefore match the alphabetical slot of `Strike.ability`, not
//! the AST/source order of the verbs.
//!
//! squad_skirmish caught this the hard way: verb bodies passed
//! `apply_ability 1..4` matching the verb declaration order
//! (`Strike → 1, Volley → 2, Rally → 3, Daze → 4`), but the alphabetical
//! .ability filenames assigned `Daze → 1, Rally → 2, Strike → 3, Volley → 4`.
//! The dispatcher then sent Strike's id=1 through Daze's `Stun` effect
//! slot — symptoms: zero damage, 2520 heal events at the Volley cooldown
//! tick, stray stun events at tick 0. See the fix commit message for the
//! pin's chronicle readback.
//!
//! This regression locks the build-registry / packed-registry alignment:
//! a two-ability registry built in alphabetical order (Damage program
//! first, Heal program second) packs `effect_kinds[0] == 0` (Damage) and
//! `effect_kinds[MAX_EFFECTS_PER_PROGRAM] == 1` (Heal). If the registry
//! ever reorders or the packer's slot↔id arithmetic drifts, the
//! GPU dispatcher's `kind == 0u → emit Damage` arm fires the wrong
//! chronicle record and this test fails.

use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, Gate, PackedAbilityRegistry,
    packed::EFFECT_KIND_EMPTY,
    program::{EffectOp, MAX_EFFECTS_PER_PROGRAM},
};

/// Build a 2-ability registry, alphabetical-style:
///   - slot 0 (AbilityId 1) — single Damage{12.0} program ("DamageAbility")
///   - slot 1 (AbilityId 2) — single Heal{20.0} program ("HealAbility")
/// Pack it and inspect the SoA columns. The GPU dispatcher reads
/// `effect_kinds[ability_slot * MAX_EFFECTS_PER_PROGRAM + i]` after
/// computing `ability_slot = ability_id__u32 - 1u`; this test pins that
/// chain end-to-end so a regression surfaces here BEFORE the squad_skirmish
/// pin's chronicle readback catches the wrong-kind emission.
#[test]
fn two_ability_registry_effect_kinds_match_slot_order() {
    let mut builder = AbilityRegistryBuilder::new();
    let id_damage = builder.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount: 12.0 }],
    ));
    let id_heal = builder.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 20.0 }],
    ));
    assert_eq!(id_damage.raw(), 1, "first register → AbilityId 1 (slot 0)");
    assert_eq!(id_heal.raw(), 2,   "second register → AbilityId 2 (slot 1)");

    let registry = builder.build();
    let packed = PackedAbilityRegistry::pack(&registry);

    let stride = MAX_EFFECTS_PER_PROGRAM;
    // Slot 0 (Damage program): first effect kind is Damage (0); the
    // remaining (stride-1) slots are EFFECT_KIND_EMPTY.
    assert_eq!(
        packed.effect_kinds[0], 0,
        "slot 0 effect 0: expected Damage discriminant 0, got {}",
        packed.effect_kinds[0],
    );
    for i in 1..stride {
        assert_eq!(
            packed.effect_kinds[i], EFFECT_KIND_EMPTY,
            "slot 0 effect {i}: expected EFFECT_KIND_EMPTY 0xFF",
        );
    }

    // Slot 1 (Heal program): first effect kind is Heal (1); the
    // remaining (stride-1) slots are EFFECT_KIND_EMPTY.
    let heal_base = stride;
    assert_eq!(
        packed.effect_kinds[heal_base], 1,
        "slot 1 effect 0: expected Heal discriminant 1, got {}",
        packed.effect_kinds[heal_base],
    );
    for i in 1..stride {
        assert_eq!(
            packed.effect_kinds[heal_base + i], EFFECT_KIND_EMPTY,
            "slot 1 effect {i}: expected EFFECT_KIND_EMPTY 0xFF",
        );
    }
}

/// Mirror of the squad_skirmish corpus shape: register Daze (Stun), Rally
/// (Heal), Strike (Damage), Volley (Damage) in that alphabetical filename
/// order and verify each ability's slot in `effect_kinds` carries its
/// own program's effect — NOT a neighbour's. Pre-fix the .sim's verb
/// bodies passed `apply_ability 1` from the Strike verb, which routed
/// through Daze's `Stun` slot at index 0; this test asserts the
/// alphabetical mapping is what every consumer of `apply_ability <id>`
/// must use.
#[test]
fn squad_skirmish_shape_alphabetical_slot_assignment() {
    let mut builder = AbilityRegistryBuilder::new();
    let id_daze = builder.register(AbilityProgram::new_single_target(
        6.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Stun { duration_ticks: 8 }],
    ));
    let id_rally = builder.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 20.0 }],
    ));
    let id_strike = builder.register(AbilityProgram::new_single_target(
        8.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount: 12.0 }],
    ));
    let id_volley = builder.register(AbilityProgram::new_single_target(
        12.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount: 6.0 }],
    ));
    assert_eq!(id_daze.raw(),   1, "Daze sorts first → slot 0");
    assert_eq!(id_rally.raw(),  2, "Rally sorts second → slot 1");
    assert_eq!(id_strike.raw(), 3, "Strike sorts third → slot 2");
    assert_eq!(id_volley.raw(), 4, "Volley sorts fourth → slot 3");

    let registry = builder.build();
    let packed = PackedAbilityRegistry::pack(&registry);

    let stride = MAX_EFFECTS_PER_PROGRAM;
    // EffectOp discriminants (pinned by the schema hash + program.rs):
    //   0 = Damage, 1 = Heal, 3 = Stun.
    // Each ability's first effect slot must carry its own discriminant.
    assert_eq!(packed.effect_kinds[0 * stride],     3, "Daze[0]   = Stun");
    assert_eq!(packed.effect_kinds[1 * stride],     1, "Rally[0]  = Heal");
    assert_eq!(packed.effect_kinds[2 * stride],     0, "Strike[0] = Damage");
    assert_eq!(packed.effect_kinds[3 * stride],     0, "Volley[0] = Damage");

    // GPU dispatcher arithmetic: `ability_slot = id - 1`, `effect_base =
    // ability_slot * MAX_EFFECTS_PER_PROGRAM`, `kind = effect_kinds[
    // effect_base + i]`. So `apply_ability 3` (id_strike) reads
    // effect_kinds[2 * stride] = 0 (Damage) — the Strike-verb-correct
    // routing. Pre-fix, the .sim's Strike verb passed
    // `apply_ability 1`, which would read effect_kinds[0 * stride] = 3
    // (Stun) — the wrong program entirely.
    let strike_slot = (id_strike.raw() - 1) as usize;
    let strike_kind = packed.effect_kinds[strike_slot * stride];
    assert_eq!(
        strike_kind, 0,
        "the literal id passed to apply_ability MUST resolve to its own \
         program's slot — Strike.id=3 → slot 2 → kind Damage(0)"
    );
}
