//! boss_fight apply_ability binding check.
//!
//! Builds the runtime's three-program AbilityRegistry (BossStrike at
//! AbilityId(1), HeroAttack at AbilityId(2), HeroStun at AbilityId(3))
//! and asserts that the registered slot IDs match the
//! `apply_ability 1 ...` / `apply_ability 2 ...` / `apply_ability 3 ...`
//! literals hardcoded in `assets/sim/boss_fight.sim`'s BossStrike +
//! HeroAttack + HeroStun verb bodies. If a slot drifts (e.g. someone
//! later registers a fourth ability ahead of HeroStun), the panic here
//! surfaces at fixture-construction time rather than as silent
//! wrong-ability dispatch.
//!
//! Mirrors `crates/duel_25v25_runtime/src/binding_check.rs` shape but
//! with three hand-built programs so we can pin the asymmetric per-side
//! damage values (Boss 50.0 vs. Hero 35.0) plus HeroStun's 15-tick
//! Stun. Source of truth is the runtime's hand-built programs (no
//! `.ability` files involved). Keeps the same naming conventions the
//! larger duel_abilities binding-check uses so a future port that grows
//! ability variety can crib the pattern straight from there.

use engine::ability::program::{EffectOp, Gate};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};

/// BossStrike is registered first so it lands at AbilityId(1).
/// The `apply_ability 1` literal in `assets/sim/boss_fight.sim::BossStrike`
/// pins this slot. Drift trips
/// `assert_ability_registry_matches_sim_constants` at startup.
pub const BOSS_STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// HeroAttack is registered second so it lands at AbilityId(2).
/// The `apply_ability 2` literal in `assets/sim/boss_fight.sim::HeroAttack`
/// pins this slot.
pub const HERO_ATTACK_EXPECTED_ABILITY_ID: u32 = 2;

/// HeroStun control-status proof (boss_fight, 2026-05-07) — registered
/// third so it lands at AbilityId(3). The `apply_ability 3` literal in
/// `assets/sim/boss_fight.sim::HeroStun` pins this slot. First control-
/// status (Stun) ability in boss_fight; proves the apply_ability
/// dispatcher's per-effect-slot loop emits kind=29 EffectStunApplied
/// chronicle records in the asymmetric 1-vs-N RPG combat shape.
pub const HERO_STUN_EXPECTED_ABILITY_ID: u32 = 3;

/// boss_fight's BossStrike registry-resident program.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim's verb-style
/// `world.tick % 10 == 0` clause (the GPU dispatcher does not consult
/// program.cooldown_ticks at the apply_ability arm today — same caveat
/// as duel_abilities's chance gating). `hostile_only: true` matches the
/// .sim's score expression that picks Hero-creature_type targets; the
/// .sim's body-side `self.creature_type == Boss` mask gate is the
/// load-bearing creature-type gate today (predicate dispatch can't
/// reference creature_type).
///
/// `range: 1.5` matches the duel_25v25 convention — boss_fight has no
/// @spatial annotation today (the verb's score expression iterates all
/// candidates and argmaxes Hero matches), so range is metadata-only at
/// the apply_ability arm. The single-target dispatcher routes one
/// effect per (caster, target) pair regardless of distance.
///
/// Effect: one `Damage { amount: 50.0 }` — matches
/// `config.combat.boss_strike_damage = 50.0` in the .sim. The chronicle
/// dispatcher writes EffectDamageApplied records carrying this amount;
/// `ApplyDamageFromChronicle` re-emits as `Damaged` so the existing
/// `ApplyDamage` cascade keeps draining HP unchanged.
fn build_boss_strike_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 1.5,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 50.0 }],
    )
}

/// boss_fight's HeroAttack registry-resident program.
///
/// Same shape as BossStrike but with the Hero-side damage value. The
/// .sim's body-side `self.creature_type == Hero` mask gate routes this
/// verb to hero slots (1..=5) only.
///
/// Effect: one `Damage { amount: 35.0 }` — matches
/// `config.combat.hero_attack_damage = 35.0` in the .sim.
fn build_hero_attack_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 1.5,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 35.0 }],
    )
}

/// boss_fight's HeroStun registry-resident program (HeroStun control-
/// status proof, 2026-05-07).
///
/// Single-target Stun(15 ticks). The first non-Damage EffectOp in this
/// fixture — proves the apply_ability dispatcher emits kind=29
/// EffectStunApplied chronicle records (drained by
/// `ApplyStunFromChronicle` straight into the per-agent
/// `stun_expires_at_tick` SoA). 15 ticks (= 1.5s at 100ms tick) gives
/// the stun a long enough window that several `world.tick % 7 == 0`
/// cast cycles can land before any single stun expires, so the test
/// sees a non-zero `stun_expires_at_tick` after one or two cast pulses.
/// Matches duel_25v25's ConcussiveCleave Stun duration verbatim.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim verb's
/// `world.tick % 7 == 0` clause (the GPU dispatcher does not consult
/// program.cooldown_ticks at the apply_ability arm today). `hostile_only:
/// true` matches the .sim's score expression that picks Boss-
/// creature_type targets; the .sim's body-side `self.creature_type ==
/// Hero` mask gate routes this verb to hero slots (1..=5) only.
/// `range: 1.5` matches BossStrike + HeroAttack — pair-field scoring
/// argmaxes the Boss as the only valid candidate.
fn build_hero_stun_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 1.5,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Stun { duration_ticks: 15 }],
    )
}

/// Build the boss_fight AbilityRegistry — three programs (BossStrike at
/// AbilityId(1), HeroAttack at AbilityId(2), HeroStun at AbilityId(3)).
/// Returns the frozen registry; callers pack + upload via
/// `PackedAbilityRegistry::pack` + `PackedAbilityRegistryGpu::upload`.
pub fn build_boss_fight_registry() -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    let bs_id = builder.register(build_boss_strike_program());
    debug_assert_eq!(
        bs_id,
        AbilityId::new(BOSS_STRIKE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "first registered program must land at AbilityId(1)",
    );
    let ha_id = builder.register(build_hero_attack_program());
    debug_assert_eq!(
        ha_id,
        AbilityId::new(HERO_ATTACK_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "second registered program must land at AbilityId(2)",
    );
    let hs_id = builder.register(build_hero_stun_program());
    debug_assert_eq!(
        hs_id,
        AbilityId::new(HERO_STUN_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "third registered program must land at AbilityId(3)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `BossFightState::new` at fixture-construction time.
///
/// Asserts the registry contains exactly three programs and that each
/// slot, gate, area, and effect matches the .sim's hand-mirrored
/// BossStrike + HeroAttack + HeroStun behaviour. If anything diverges
/// the panic message points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_boss_fight_registry();
    assert_eq!(
        registry.len(),
        3,
        "boss_fight registry must contain exactly three programs \
         (BossStrike + HeroAttack + HeroStun); got {}",
        registry.len(),
    );

    // ---- BossStrike at slot 1 ----
    let bs_id = AbilityId::new(BOSS_STRIKE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let bs = registry
        .get(bs_id)
        .expect("BossStrike resolves to a program at AbilityId(1)");
    assert_eq!(
        bs.gate.cooldown_ticks, 0,
        "BossStrike cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 10 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        bs.gate.hostile_only,
        "BossStrike must be hostile_only — .sim score picks Hero targets, \
         but the registry metadata should still record `target enemy` \
         semantics for future predicate dispatch",
    );
    use engine::ability::program::Area;
    match bs.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 1.5,
            "BossStrike range must be 1.5 — single-target dispatch metadata",
        ),
    }
    assert_eq!(
        bs.effects.len(), 1,
        "BossStrike must have exactly one effect (Damage 50.0)",
    );
    match &bs.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 50.0,
            "BossStrike damage must be 50.0 — matches \
             config.combat.boss_strike_damage in boss_fight.sim",
        ),
        other => panic!(
            "BossStrike effect[0]: expected Damage(50.0), got {other:?}",
        ),
    }

    // ---- HeroAttack at slot 2 ----
    let ha_id = AbilityId::new(HERO_ATTACK_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let ha = registry
        .get(ha_id)
        .expect("HeroAttack resolves to a program at AbilityId(2)");
    assert_eq!(
        ha.gate.cooldown_ticks, 0,
        "HeroAttack cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 3 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        ha.gate.hostile_only,
        "HeroAttack must be hostile_only — .sim score picks Boss target",
    );
    match ha.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 1.5,
            "HeroAttack range must be 1.5 — single-target dispatch metadata",
        ),
    }
    assert_eq!(
        ha.effects.len(), 1,
        "HeroAttack must have exactly one effect (Damage 35.0)",
    );
    match &ha.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 35.0,
            "HeroAttack damage must be 35.0 — matches \
             config.combat.hero_attack_damage in boss_fight.sim",
        ),
        other => panic!(
            "HeroAttack effect[0]: expected Damage(35.0), got {other:?}",
        ),
    }

    // ---- HeroStun at slot 3 (control-status proof — first non-Damage
    //      EffectOp in this fixture) ----
    let hs_id = AbilityId::new(HERO_STUN_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let hs = registry
        .get(hs_id)
        .expect("HeroStun resolves to a program at AbilityId(3)");
    assert_eq!(
        hs.gate.cooldown_ticks, 0,
        "HeroStun cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 7 == 0`)",
    );
    assert!(
        hs.gate.hostile_only,
        "HeroStun must be hostile_only — .sim score picks Boss target, \
         same as HeroAttack",
    );
    match hs.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 1.5,
            "HeroStun range must be 1.5 — single-target dispatch metadata, \
             same as BossStrike + HeroAttack",
        ),
    }
    assert_eq!(
        hs.effects.len(), 1,
        "HeroStun must have exactly one effect (Stun 15 ticks)",
    );
    match &hs.effects[0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 15,
            "HeroStun stun duration_ticks must be 15 (= 1.5s at 100ms \
             tick); .sim verb hand-mirrors via the apply_ability \
             dispatcher's `expires_at_tick = world.tick + 15` chronicle \
             write",
        ),
        other => panic!(
            "HeroStun effect[0]: expected Stun(duration_ticks=15), got {other:?}",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: three abilities at slots 1 +
    /// 2 + 3 (BossStrike, HeroAttack, HeroStun), with the asymmetric
    /// Damage amounts (Boss 50, Hero 35) and HeroStun's 15-tick Stun
    /// at the expected gate/area. Catches drift before construction-
    /// time panics surface in viz_tests / behavioural tests.
    #[test]
    fn registry_contains_bossstrike_heroattack_herostun_at_pinned_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
