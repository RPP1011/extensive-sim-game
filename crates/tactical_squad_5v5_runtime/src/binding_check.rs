//! tactical_squad_5v5 apply_ability binding check.
//!
//! Builds the runtime's two-program AbilityRegistry (TankAttack +
//! DpsAttack) and asserts that the registered slot IDs match the
//! `apply_ability 1` / `apply_ability 2` literals hardcoded in
//! `assets/sim/tactical_squad_5v5.sim`'s Strike (Tank) and Snipe (DPS)
//! verb bodies. If a registered slot drifts (e.g. someone later
//! reorders the registration calls or inserts a placeholder ahead of
//! TankAttack), the panic here surfaces at fixture-construction time
//! rather than as silent wrong-ability dispatch.
//!
//! Mirrors `crates/duel_25v25_runtime/src/binding_check.rs` — both
//! fixtures hand-build their programs (no `.ability` files involved)
//! and pin slot IDs via constants. tactical_squad_5v5 is a small step
//! up: TWO abilities (rather than one), demonstrating that the
//! AbilityRegistryBuilder's slot-assignment ordering survives multiple
//! registrations.
//!
//! CONFIG → REGISTRY BRIDGE: the .sim's `config.combat.tank_damage =
//! 10.0` and `config.combat.dps_damage = 22.0` are mirrored as
//! literals in the registry programs below. apply_ability's
//! pack-time literal API doesn't accept config refs today, so the
//! runtime hardcodes the matching values; if the .sim config drifts,
//! the assertions below flag the divergence.

use engine::ability::program::{EffectOp, Gate};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};

/// TankAttack is registered first → AbilityId(1). The
/// `apply_ability 1` literal in
/// `assets/sim/tactical_squad_5v5.sim::Strike` pins this slot.
pub const TANK_ATTACK_EXPECTED_ABILITY_ID: u32 = 1;

/// DpsAttack is registered second → AbilityId(2). The
/// `apply_ability 2` literal in
/// `assets/sim/tactical_squad_5v5.sim::Snipe` pins this slot.
pub const DPS_ATTACK_EXPECTED_ABILITY_ID: u32 = 2;

/// Hand-mirrored Tank base damage — matches `config.combat.tank_damage
/// = 10.0` in `assets/sim/tactical_squad_5v5.sim`. The .sim's verb
/// body used to read this config field directly via
/// `config.combat.tank_damage`; after the apply_ability swap the
/// dispatcher pulls the literal from the program's effect op (set
/// here), so any drift between the .sim config and this constant
/// surfaces at the binding-check panic, not as silent wrong-damage
/// dispatch.
pub const TANK_DAMAGE: f32 = 10.0;

/// Hand-mirrored DPS base damage — matches `config.combat.dps_damage
/// = 22.0`. Same bridge caveat as `TANK_DAMAGE`.
pub const DPS_DAMAGE: f32 = 22.0;

/// TankAttack registry-resident program.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim's verb-style
/// `world.tick % 3 == 0` clause — the GPU dispatcher doesn't consult
/// program.cooldown_ticks at the apply_ability arm today (same caveat
/// as duel_25v25). `hostile_only: true` matches the .sim's `target
/// enemy` semantic; the .sim's body-side `target.level != self.level`
/// + `self.creature_type == Tank` checks are the load-bearing role +
/// team gates today (apply_ability `when` predicates can't reference
/// creature_type).
///
/// `range: 0.0` — the .sim doesn't declare a range/spatial filter for
/// Strike (the verb is a flat per-pair scoring kernel, not a spatial
/// walk), so the registry's range is metadata-only. Set to 0.0 to
/// signal "any range; verb-gated by creature_type/level instead".
fn build_tank_attack_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 0.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: TANK_DAMAGE }],
    )
}

/// DpsAttack registry-resident program. Same shape as TankAttack but
/// 22.0 damage (matches `config.combat.dps_damage`).
fn build_dps_attack_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 0.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: DPS_DAMAGE }],
    )
}

/// Build the tactical_squad_5v5 AbilityRegistry — TankAttack at
/// AbilityId(1), DpsAttack at AbilityId(2). Returns the frozen
/// registry; callers pack + upload via `PackedAbilityRegistry::pack`
/// + `PackedAbilityRegistryGpu::upload`.
pub fn build_tactical_squad_5v5_registry() -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    let tank_id = builder.register(build_tank_attack_program());
    debug_assert_eq!(
        tank_id,
        AbilityId::new(TANK_ATTACK_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "TankAttack must register first → AbilityId(1)",
    );
    let dps_id = builder.register(build_dps_attack_program());
    debug_assert_eq!(
        dps_id,
        AbilityId::new(DPS_ATTACK_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "DpsAttack must register second → AbilityId(2)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `TacticalSquad5v5State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly two programs and that each
/// program's slot, gate, area, and effect match the .sim's
/// hand-mirrored Strike/Snipe verb constants. If anything diverges
/// the panic message points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_tactical_squad_5v5_registry();
    assert_eq!(
        registry.len(),
        2,
        "tactical_squad_5v5 registry must contain exactly two programs \
         (TankAttack, DpsAttack); got {}",
        registry.len(),
    );

    use engine::ability::program::Area;

    // ---- TankAttack at AbilityId(1) — 10.0 damage ----
    let tank_id = AbilityId::new(TANK_ATTACK_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let tank = registry
        .get(tank_id)
        .expect("TankAttack resolves to a program at AbilityId(1)");
    assert_eq!(
        tank.gate.cooldown_ticks, 0,
        "TankAttack cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 3 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        tank.gate.hostile_only,
        "TankAttack must be hostile_only — .sim's body-side team check \
         (target.level != self.level) enforces the actual gate, but the \
         registry metadata records `target enemy` semantics for future \
         predicate dispatch",
    );
    match tank.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 0.0,
            "TankAttack range must be 0.0 — Strike has no spatial filter \
             in tactical_squad_5v5.sim (verb-gated, not spatial)",
        ),
    }
    assert_eq!(
        tank.effects.len(), 1,
        "TankAttack must have exactly one effect (Damage 10.0)",
    );
    match &tank.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, TANK_DAMAGE,
            "TankAttack damage must be {} — matches \
             config.combat.tank_damage in tactical_squad_5v5.sim",
            TANK_DAMAGE,
        ),
        other => panic!(
            "TankAttack effect[0]: expected Damage({}), got {other:?}",
            TANK_DAMAGE,
        ),
    }

    // ---- DpsAttack at AbilityId(2) — 22.0 damage ----
    let dps_id = AbilityId::new(DPS_ATTACK_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let dps = registry
        .get(dps_id)
        .expect("DpsAttack resolves to a program at AbilityId(2)");
    assert_eq!(
        dps.gate.cooldown_ticks, 0,
        "DpsAttack cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 4 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        dps.gate.hostile_only,
        "DpsAttack must be hostile_only — same role/team gate caveat as \
         TankAttack",
    );
    match dps.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 0.0,
            "DpsAttack range must be 0.0 — Snipe has no spatial filter \
             in tactical_squad_5v5.sim (verb-gated, not spatial)",
        ),
    }
    assert_eq!(
        dps.effects.len(), 1,
        "DpsAttack must have exactly one effect (Damage 22.0)",
    );
    match &dps.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, DPS_DAMAGE,
            "DpsAttack damage must be {} — matches \
             config.combat.dps_damage in tactical_squad_5v5.sim",
            DPS_DAMAGE,
        ),
        other => panic!(
            "DpsAttack effect[0]: expected Damage({}), got {other:?}",
            DPS_DAMAGE,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: two abilities, TankAttack at
    /// slot 1 (Damage 10.0) and DpsAttack at slot 2 (Damage 22.0) at
    /// the expected gates/areas. Catches drift before construction-time
    /// panics surface in viz_tests / behavioural tests.
    #[test]
    fn registry_contains_tank_and_dps_at_expected_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
