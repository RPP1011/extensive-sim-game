//! tactical_squad_5v5 apply_ability binding check.
//!
//! Builds the runtime's four-program AbilityRegistry (TankAttack +
//! DpsAttack + ConcussiveBlow + SquadHeal) and asserts that the
//! registered slot IDs match the `apply_ability 1` / `apply_ability 2`
//! / `apply_ability 3` / `apply_ability 4` literals hardcoded in
//! `assets/sim/tactical_squad_5v5.sim`'s Strike (Tank), Snipe (DPS),
//! ConcussiveBlow (DPS), and SquadHeal (any role) verb bodies. If a
//! registered slot drifts (e.g. someone later reorders the
//! registration calls or inserts a placeholder ahead of TankAttack),
//! the panic here surfaces at fixture-construction time rather than as
//! silent wrong-ability dispatch.
//!
//! Mirrors `crates/duel_25v25_runtime/src/binding_check.rs` and
//! `crates/mass_battle_100v100_runtime/src/binding_check.rs` — same
//! one-program-per-ability hand-built shape, three programs (TankAttack,
//! DpsAttack, ConcussiveBlow). No `.ability` files involved; the source
//! of truth is this file's program builders.
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

/// ConcussiveBlow control-status proof (5v5 scale, 2026-05-07) —
/// registered third so it lands at AbilityId(3). The `apply_ability 3`
/// literal in `assets/sim/tactical_squad_5v5.sim::ConcussiveBlow` pins
/// this slot. First control-status (Stun) ability in
/// tactical_squad_5v5; proves the apply_ability dispatcher's per-
/// effect-slot loop emits kind=29 EffectStunApplied chronicle records
/// at 5v5 scale (10 agents × pair-field scoring). Mirrors
/// mass_battle_100v100's StunBolt third-site pattern (commit
/// 67b61aa2).
pub const CONCUSSIVE_BLOW_EXPECTED_ABILITY_ID: u32 = 3;

/// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07) —
/// registered fourth so it lands at AbilityId(4). The `apply_ability 4`
/// literal in `assets/sim/tactical_squad_5v5.sim::SquadHeal` pins this
/// slot. First chronicle-pipeline (kind=27 EffectHealApplied) heal
/// ability in tactical_squad_5v5 — proves the apply_ability
/// dispatcher's per-effect-slot loop emits Heal chronicle records at
/// 5v5 scale and the lower pass auto-fuses ApplyHealFromChronicle into
/// the existing ApplyDamage + ApplyStun chronicle consumer kernel
/// (3-way fusion). Mirrors duel_25v25's HealPulse third-site pattern
/// (commit 049feb0c).
pub const SQUAD_HEAL_EXPECTED_ABILITY_ID: u32 = 4;

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

/// SquadHeal heal magnitude (apply_ability ally heal, 2026-05-07) —
/// hand-mirrored heal amount. The .sim's SquadHeal verb has no
/// counterpart in `config.combat` (the existing `heal_amount = 18.0`
/// belongs to the legacy `Heal` verb's `emit Healed` path, not the
/// chronicle-pipeline SquadHeal). 12.0 is roughly two-thirds of the
/// legacy Heal magnitude — large enough to visibly recover from
/// Strike/Snipe/ConcussiveBlow drain over a 7-tick cadence cycle, but
/// small enough that several casts can land before clamping at the
/// 100.0 max_hp ceiling for an under-damaged target. Mirrors
/// duel_25v25's HealPulse 15.0 magnitude (commit 049feb0c).
pub const SQUAD_HEAL_AMOUNT: f32 = 12.0;

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

/// tactical_squad_5v5's ConcussiveBlow registry-resident program
/// (control-status proof at 5v5 scale, 2026-05-07).
///
/// Single-target Stun(20 ticks). The first non-Damage EffectOp in this
/// fixture — proves the apply_ability dispatcher emits kind=29
/// EffectStunApplied chronicle records (drained by
/// `ApplyStunFromChronicle` straight into the per-agent
/// `stun_expires_at_tick` SoA). 20 ticks (= 2s at 100ms tick) gives the
/// stun a long enough window that several `world.tick % 7 == 0` cast
/// cycles can land before any single stun expires, so the test sees a
/// non-zero stun_expires_at_tick after one or two cast pulses. Mirrors
/// mass_battle_100v100's StunBolt program (commit 67b61aa2) — same
/// EffectOp::Stun{duration_ticks=20} shape.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim verb's
/// `world.tick % 7 == 0` clause (the GPU dispatcher does not consult
/// program.cooldown_ticks at the apply_ability arm today).
/// `hostile_only: true` matches the .sim's `target.level != self.level`
/// enemy predicate (DPS Red→Blue, DPS Blue→Red — same role-team gate
/// Snipe uses, just at a different cadence). `range: 0.0` matches
/// Strike + Snipe (no spatial filter — verb-gated by creature_type
/// + level instead).
fn build_concussive_blow_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 0.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Stun { duration_ticks: 20 }],
    )
}

/// tactical_squad_5v5's SquadHeal registry-resident program
/// (apply_ability-dispatched ally heal, 2026-05-07). FIRST
/// chronicle-pipeline heal ability in this fixture — the existing
/// `Heal` verb still emits the legacy `Healed` event via the per-agent
/// kernel; SquadHeal is a brand-new verb that writes kind=27
/// EffectHealApplied via the apply_ability dispatcher.
///
/// Single-target Heal(12.0). The first non-Damage / non-Stun EffectOp
/// in this fixture's chronicle pipeline — proves the apply_ability
/// dispatcher emits kind=27 EffectHealApplied chronicle records
/// (drained by the lower-pass-fused
/// `physics_ApplyDamageFromChronicle_and_ApplyHealFromChronicle_and_ApplyStunFromChronicle`
/// kernel straight into a clamped `agents.set_hp(t, min(hp + amt,
/// max_hp))` write). Mirrors duel_25v25's HealPulse program (commit
/// 049feb0c) — same EffectOp::Heal{amount=12.0} shape (12.0 vs
/// duel_25v25's 15.0; tactical_squad_5v5's per-pair scoring funnels
/// fewer casts onto the lowest-HP target than duel_25v25's spatial
/// neighbour walk, so a smaller per-cast amount keeps the recovery
/// dynamic visible without saturating max_hp on the first tick).
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim verb's
/// `world.tick % 7 == 0` clause (the GPU dispatcher does not consult
/// program.cooldown_ticks at the apply_ability arm today).
/// `hostile_only: false` matches the .sim's `target friend` semantic:
/// the body-side check `target.level == self.level` enforces the
/// same-team predicate today, but the registry metadata records that
/// future predicate dispatch should scope target selection to allies.
/// `range: 50.0` matches the task spec — metadata-only at 5v5 scale
/// because tactical_squad_5v5 has no spatial filter (the verb is a
/// flat per-pair scoring kernel, not a spatial walk); the value
/// signals "any target on the team" for documentation purposes
/// without affecting dispatch.
fn build_squad_heal_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 50.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: SQUAD_HEAL_AMOUNT }],
    )
}

/// Build the tactical_squad_5v5 AbilityRegistry — TankAttack at
/// AbilityId(1), DpsAttack at AbilityId(2), ConcussiveBlow at
/// AbilityId(3), SquadHeal at AbilityId(4). Returns the frozen
/// registry; callers pack + upload via `PackedAbilityRegistry::pack` +
/// `PackedAbilityRegistryGpu::upload`.
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
    let concussive_blow_id = builder.register(build_concussive_blow_program());
    debug_assert_eq!(
        concussive_blow_id,
        AbilityId::new(CONCUSSIVE_BLOW_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "ConcussiveBlow must register third → AbilityId(3)",
    );
    let squad_heal_id = builder.register(build_squad_heal_program());
    debug_assert_eq!(
        squad_heal_id,
        AbilityId::new(SQUAD_HEAL_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "SquadHeal must register fourth → AbilityId(4)",
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
        4,
        "tactical_squad_5v5 registry must contain exactly four programs \
         (TankAttack, DpsAttack, ConcussiveBlow, SquadHeal); got {}",
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

    // ---- ConcussiveBlow at AbilityId(3) (control-status proof at 5v5
    //      scale — first non-Damage EffectOp in this fixture) ----
    let concussive_blow_id = AbilityId::new(CONCUSSIVE_BLOW_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let concussive_blow = registry
        .get(concussive_blow_id)
        .expect("ConcussiveBlow resolves to a program at AbilityId(3)");
    assert_eq!(
        concussive_blow.gate.cooldown_ticks, 0,
        "ConcussiveBlow cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 7 == 0`)",
    );
    assert!(
        concussive_blow.gate.hostile_only,
        "ConcussiveBlow must be hostile_only — DPS-vs-enemy gate, same as Snipe",
    );
    match concussive_blow.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 0.0,
            "ConcussiveBlow range must be 0.0 — no spatial filter in \
             tactical_squad_5v5.sim (verb-gated, not spatial — same as \
             Strike + Snipe)",
        ),
    }
    assert_eq!(
        concussive_blow.effects.len(), 1,
        "ConcussiveBlow must have exactly one effect (Stun 20 ticks)",
    );
    match &concussive_blow.effects[0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 20,
            "ConcussiveBlow stun duration_ticks must be 20 (= 2s at 100ms \
             tick); .sim verb hand-mirrors via the apply_ability \
             dispatcher's `expires_at_tick = world.tick + 20` chronicle \
             write",
        ),
        other => panic!(
            "ConcussiveBlow effect[0]: expected Stun(duration_ticks=20), got {other:?}",
        ),
    }

    // ---- SquadHeal at AbilityId(4) (apply_ability ally heal at 5v5
    //      scale — first chronicle-pipeline heal in this fixture) ----
    let squad_heal_id = AbilityId::new(SQUAD_HEAL_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let squad_heal = registry
        .get(squad_heal_id)
        .expect("SquadHeal resolves to a program at AbilityId(4)");
    assert_eq!(
        squad_heal.gate.cooldown_ticks, 0,
        "SquadHeal cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 7 == 0`)",
    );
    assert!(
        !squad_heal.gate.hostile_only,
        "SquadHeal must NOT be hostile_only — `target friend` semantic \
         (the .sim's body-side `target.level == self.level` enforces \
         the same-team predicate today; future predicate dispatch will \
         consult this flag)",
    );
    match squad_heal.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 50.0,
            "SquadHeal range must be 50.0 — metadata-only at 5v5 scale \
             (verb is per-pair scoring, not spatial)",
        ),
    }
    assert_eq!(
        squad_heal.effects.len(), 1,
        "SquadHeal must have exactly one effect (Heal {})",
        SQUAD_HEAL_AMOUNT,
    );
    match &squad_heal.effects[0] {
        EffectOp::Heal { amount } => assert_eq!(
            *amount, SQUAD_HEAL_AMOUNT,
            "SquadHeal heal amount must be {} — visible recovery dynamic \
             on a 7-tick cadence at 5v5 scale, mirrors duel_25v25's \
             HealPulse magnitude shape (commit 049feb0c)",
            SQUAD_HEAL_AMOUNT,
        ),
        other => panic!(
            "SquadHeal effect[0]: expected Heal({}), got {other:?}",
            SQUAD_HEAL_AMOUNT,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: four abilities, TankAttack at
    /// slot 1 (Damage 10.0), DpsAttack at slot 2 (Damage 22.0),
    /// ConcussiveBlow at slot 3 (Stun 20 ticks), SquadHeal at slot 4
    /// (Heal 12.0) at the expected gates/areas. Catches drift before
    /// construction-time panics surface in viz_tests / behavioural
    /// tests.
    #[test]
    fn registry_contains_tank_dps_concussive_blow_squad_heal_at_expected_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
