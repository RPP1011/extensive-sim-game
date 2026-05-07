//! mass_battle_100v100 apply_ability binding check.
//!
//! Builds the runtime's two-program AbilityRegistry (Strike + Snipe)
//! and asserts the registered slot IDs match the `apply_ability 1`
//! and `apply_ability 2` literals hardcoded in
//! `assets/sim/mass_battle_100v100.sim`'s Strike + Snipe verb bodies.
//! If the registered slots drift (e.g. someone reorders the
//! registration in `build_mass_battle_100v100_registry`), the panic
//! here surfaces at fixture-construction time rather than as silent
//! wrong-ability dispatch.
//!
//! Mirrors `crates/duel_25v25_runtime/src/binding_check.rs` — same
//! one-program-per-ability hand-built shape, just two programs
//! (Strike and Snipe) instead of one. No `.ability` files involved;
//! the source of truth is this file's program builders.

use engine::ability::program::{EffectOp, Gate};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};

/// Strike is registered first so it always lands at AbilityId(1).
/// The `apply_ability 1` literal in
/// `assets/sim/mass_battle_100v100.sim::Strike` pins this slot. Any
/// drift trips `assert_ability_registry_matches_sim_constants` at
/// startup.
pub const STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// Snipe is registered second so it lands at AbilityId(2). The
/// `apply_ability 2` literal in
/// `assets/sim/mass_battle_100v100.sim::Snipe` pins this slot.
pub const SNIPE_EXPECTED_ABILITY_ID: u32 = 2;

/// mass_battle_100v100's Strike registry-resident program.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim's verb-
/// style `world.tick % 2 == 0` clause (the GPU dispatcher does not
/// consult program.cooldown_ticks at the apply_ability arm today).
/// `hostile_only: true` matches the .sim's level-encoded enemy
/// predicate (Tank Red→Blue, Tank Blue→Red); the .sim's body-side
/// level-pair check is the load-bearing team gate today (predicate
/// dispatch can't reference level encoding).
///
/// `range: 999.0` matches the .sim's `perception_radius = 999.0`
/// (mass_battle uses pair-field scoring, not spatial narrowing — the
/// argmax loops over all candidates regardless of distance). The
/// registry's `range` is metadata-only at the apply_ability arm
/// today.
///
/// Effect: one `Damage { amount: 30.0 }` — matches
/// `config.combat.strike_damage = 30.0` in the .sim.
fn build_strike_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 999.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    )
}

/// mass_battle_100v100's Snipe registry-resident program.
///
/// Same shape as Strike but at the DPS damage tier.
/// `cooldown_ticks: 0` (gate stays in the .sim's `world.tick % 3 == 0`
/// clause), `hostile_only: true` (DPS attacks enemies), `range:
/// 999.0` (pair-field, no spatial narrowing), `Damage { amount:
/// 22.0 }` matching `config.combat.snipe_damage = 22.0`.
fn build_snipe_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 999.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 22.0 }],
    )
}

/// Build the mass_battle_100v100 AbilityRegistry — two programs
/// (Strike at AbilityId(1), Snipe at AbilityId(2)). Returns the frozen
/// registry; callers pack + upload via `PackedAbilityRegistry::pack`
/// + `PackedAbilityRegistryGpu::upload`.
pub fn build_mass_battle_100v100_registry() -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    let strike_id = builder.register(build_strike_program());
    debug_assert_eq!(
        strike_id,
        AbilityId::new(STRIKE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "first registered program must land at AbilityId(1)",
    );
    let snipe_id = builder.register(build_snipe_program());
    debug_assert_eq!(
        snipe_id,
        AbilityId::new(SNIPE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "second registered program must land at AbilityId(2)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `MassBattle100v100State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly two programs and that each
/// slot, gate, area, and effect match the .sim's hand-mirrored Strike
/// and Snipe verb behaviour. If anything diverges the panic message
/// points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_mass_battle_100v100_registry();
    assert_eq!(
        registry.len(),
        2,
        "mass_battle_100v100 registry must contain exactly two programs \
         (Strike + Snipe); got {}",
        registry.len(),
    );

    use engine::ability::program::Area;

    // ---- Strike ----
    let strike_id = AbilityId::new(STRIKE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let strike = registry
        .get(strike_id)
        .expect("Strike resolves to a program at AbilityId(1)");
    assert_eq!(
        strike.gate.cooldown_ticks, 0,
        "Strike cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 2 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        strike.gate.hostile_only,
        "Strike must be hostile_only — .sim's body-side level-pair \
         check enforces the actual gate, but the registry metadata \
         should still record `target enemy` semantics for future \
         predicate dispatch",
    );
    match strike.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 999.0,
            "Strike range must be 999.0 — matches \
             config.combat.perception_radius in mass_battle_100v100.sim \
             (pair-field scoring, no spatial narrowing)",
        ),
    }
    assert_eq!(
        strike.effects.len(), 1,
        "Strike must have exactly one effect (Damage 30.0)",
    );
    match &strike.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 30.0,
            "Strike damage must be 30.0 — matches \
             config.combat.strike_damage in mass_battle_100v100.sim",
        ),
        other => panic!(
            "Strike effect[0]: expected Damage(30.0), got {other:?}",
        ),
    }

    // ---- Snipe ----
    let snipe_id = AbilityId::new(SNIPE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let snipe = registry
        .get(snipe_id)
        .expect("Snipe resolves to a program at AbilityId(2)");
    assert_eq!(
        snipe.gate.cooldown_ticks, 0,
        "Snipe cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 3 == 0`)",
    );
    assert!(
        snipe.gate.hostile_only,
        "Snipe must be hostile_only — DPS attacks enemies",
    );
    match snipe.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 999.0,
            "Snipe range must be 999.0 — matches \
             config.combat.perception_radius in mass_battle_100v100.sim",
        ),
    }
    assert_eq!(
        snipe.effects.len(), 1,
        "Snipe must have exactly one effect (Damage 22.0)",
    );
    match &snipe.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 22.0,
            "Snipe damage must be 22.0 — matches \
             config.combat.snipe_damage in mass_battle_100v100.sim",
        ),
        other => panic!(
            "Snipe effect[0]: expected Damage(22.0), got {other:?}",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: two abilities (Strike at
    /// slot 1, Snipe at slot 2), each with the expected gate/area/
    /// effect. Catches drift before construction-time panics surface
    /// in viz_tests / behavioural tests.
    #[test]
    fn registry_contains_strike_and_snipe_at_expected_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
