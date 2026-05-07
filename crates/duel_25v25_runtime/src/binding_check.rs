//! duel_25v25 apply_ability binding check.
//!
//! Builds the runtime's one-program AbilityRegistry (a single Strike
//! ability) and asserts that the registered slot ID matches the
//! `apply_ability 1` literal hardcoded in `assets/sim/duel_25v25.sim`'s
//! ScanAndStrike body. If the registered slot drifts (e.g. someone
//! later registers a second ability ahead of Strike), the panic here
//! surfaces at fixture-construction time rather than as silent
//! wrong-ability dispatch.
//!
//! Mirrors `crates/duel_abilities_runtime/src/binding_check.rs` but is
//! much smaller — duel_25v25 only registers ONE ability and the source
//! of truth is the runtime's hand-built program (no `.ability` files
//! involved). Keeps the same naming conventions so a future port that
//! grows ability variety (or moves to .ability sources) can crib the
//! pattern straight from duel_abilities.

use engine::ability::program::{EffectOp, Gate};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};

/// Strike is registered first — and as the only ability — so it always
/// lands at AbilityId(1). The `apply_ability 1` literal in
/// `assets/sim/duel_25v25.sim::ScanAndStrike` pins this slot. Any
/// drift (e.g. inserting a placeholder ability ahead of Strike during
/// future expansion) trips `assert_ability_registry_matches_sim_constants`
/// at startup.
pub const STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// duel_25v25's Strike registry-resident program.
///
/// `cooldown_ticks: 0` keeps the per-tick gate in the .sim's verb-style
/// `world.tick % 2 == 0` clause (the GPU dispatcher does not consult
/// program.cooldown_ticks at the apply_ability arm today — same caveat
/// as duel_abilities's chance gating). `hostile_only: true` matches the
/// .sim's `target enemy` semantic; the .sim's body-side
/// `other.creature_type != self.creature_type` check is the load-bearing
/// team gate today (predicate dispatch can't reference creature_type).
///
/// `range: 1.5` matches the @spatial annotation's radius — the spatial
/// grid filter is what actually scopes targets; the registry's `range`
/// is metadata-only at the apply_ability arm today (single-target
/// dispatcher routes one effect per (caster, target) pair regardless
/// of distance). Set so the registry's metadata stays consistent with
/// the .sim's actual neighbour radius rather than carrying a phantom
/// 5.0-cell range like duel_abilities's Strike (where 5.0 is the
/// .ability source's declared range).
///
/// Effect: one `Damage { amount: 5.0 }` — matches
/// `config.combat.strike_damage = 5.0` in the .sim. The chronicle
/// dispatcher writes EffectDamageApplied records carrying this amount;
/// `ApplyDamageFromChronicle` re-emits as `Damaged` so the existing
/// `ApplyDamage` cascade keeps draining HP unchanged.
fn build_strike_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 1.5,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 5.0 }],
    )
}

/// Build the duel_25v25 AbilityRegistry — a single Strike program at
/// AbilityId(1). Returns the frozen registry; callers pack + upload via
/// `PackedAbilityRegistry::pack` + `PackedAbilityRegistryGpu::upload`.
pub fn build_duel_25v25_registry() -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(build_strike_program());
    debug_assert_eq!(
        id,
        AbilityId::new(STRIKE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "first registered program must land at AbilityId(1)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `Duel25v25State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly one program and that its
/// slot, gate, area, and effect match the .sim's hand-mirrored
/// ScanAndStrike behaviour. If anything diverges the panic message
/// points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_duel_25v25_registry();
    assert_eq!(
        registry.len(),
        1,
        "duel_25v25 registry must contain exactly one program (Strike); \
         got {}",
        registry.len(),
    );
    let strike_id = AbilityId::new(STRIKE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let strike = registry
        .get(strike_id)
        .expect("Strike resolves to a program at AbilityId(1)");

    // Gate.
    assert_eq!(
        strike.gate.cooldown_ticks, 0,
        "Strike cooldown_ticks must be 0 (cadence is in the .sim verb gate \
         `world.tick % 2 == 0`; dispatcher doesn't consult cooldown_ticks today)",
    );
    assert!(
        strike.gate.hostile_only,
        "Strike must be hostile_only — .sim's body-side team check enforces \
         the actual gate, but the registry metadata should still record \
         `target enemy` semantics for future predicate dispatch",
    );

    // Area + effect.
    use engine::ability::program::Area;
    match strike.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 1.5,
            "Strike range must be 1.5 — matches the @spatial annotation \
             radius in duel_25v25.sim",
        ),
    }
    assert_eq!(
        strike.effects.len(), 1,
        "Strike must have exactly one effect (Damage 5.0)",
    );
    match &strike.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 5.0,
            "Strike damage must be 5.0 — matches \
             config.combat.strike_damage in duel_25v25.sim",
        ),
        other => panic!(
            "Strike effect[0]: expected Damage(5.0), got {other:?}",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: one ability, slot 1, Damage 5.0
    /// at the expected gate/area. Catches drift before construction-time
    /// panics surface in viz_tests / behavioural tests.
    #[test]
    fn registry_contains_strike_at_slot_one() {
        assert_ability_registry_matches_sim_constants();
    }
}
