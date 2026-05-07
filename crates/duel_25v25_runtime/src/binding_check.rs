//! duel_25v25 apply_ability binding check.
//!
//! Builds the runtime's two-program AbilityRegistry (Strike at slot 1,
//! Cleave at slot 2) and asserts that the registered slot IDs match
//! the `apply_ability 1` and `apply_ability 2` literals hardcoded in
//! `assets/sim/duel_25v25.sim`'s ScanAndStrike + ScanAndCleave bodies.
//! If either slot drifts (e.g. someone reorders registration), the
//! panic here surfaces at fixture-construction time rather than as
//! silent wrong-ability dispatch.
//!
//! Mirrors `crates/duel_abilities_runtime/src/binding_check.rs` but is
//! still smaller — duel_25v25 only registers TWO abilities and the
//! source of truth is the runtime's hand-built programs (no `.ability`
//! files involved). Keeps the same naming conventions so a future port
//! that grows ability variety (or moves to .ability sources) can crib
//! the pattern straight from duel_abilities.

use engine::ability::program::{EffectAreaShape, EffectOp, Gate, ShapeKind};
use engine::ability::{AbilityId, AbilityProgram, AbilityRegistry, AbilityRegistryBuilder};

/// Strike is registered first — so it lands at AbilityId(1). The
/// `apply_ability 1` literal in `assets/sim/duel_25v25.sim::ScanAndStrike`
/// pins this slot.
pub const STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// Cleave is registered second — so it lands at AbilityId(2). The
/// `apply_ability 2` literal in `assets/sim/duel_25v25.sim::ScanAndCleave`
/// pins this slot. AOE Cleave (Path B production proof, 2026-05-07).
pub const CLEAVE_EXPECTED_ABILITY_ID: u32 = 2;

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

/// duel_25v25's Cleave registry-resident program (AOE Path B production
/// proof, 2026-05-07).
///
/// Shape: single-target program (so `Area::SingleTarget { range }`
/// metadata is preserved) with a per-effect `EffectAreaShape::Circle`
/// override at slot 0. The dispatcher reads `area_kinds[effect_slot]`
/// at the apply_ability arm; when non-sentinel (here `Circle = 0u`),
/// the WGSL kernel walks the 27-cell neighborhood around
/// `agent_pos[target_slot]` (i.e. the dispatched target's world
/// position) and emits one chronicle record per candidate within
/// `radius` (here 1.0). Radius 1.0 ≤ `SPATIAL_CELL_SIZE = 6.0` so the
/// single 27-cell walk covers the full circle (no extended-neighborhood
/// dispatch needed).
///
/// Damage 2.0 is intentionally lower than Strike's 5.0: each Cleave
/// cast can hit multiple in-radius targets, so the per-cast total
/// scales with neighbor density. 2.0 × 1-3 neighbours keeps the
/// battle's HP drain shape similar to Strike-only (with a richer
/// fan-out pattern) instead of collapsing in 2 ticks.
///
/// Cadence is gated in the .sim by `world.tick % 5 == 0` (vs Strike's
/// `% 2`); both rules' gates are independent, so tick 10, 20, …
/// overlap and trigger an HP-drop bump.
fn build_cleave_program() -> AbilityProgram {
    let mut cleave = AbilityProgram::new_single_target(
        /*range*/ 3.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 2.0 }],
    );
    cleave.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        // [radius, _, _, _]; radius=1.0 stays within the 6.0 cell_size
        // so the single 27-cell walk in the WGSL dispatcher covers the
        // full circle.
        args: [1.0, 0.0, 0.0, 0.0],
    }));
    cleave
}

/// Build the duel_25v25 AbilityRegistry — Strike at AbilityId(1) and
/// Cleave at AbilityId(2). Returns the frozen registry; callers pack +
/// upload via `PackedAbilityRegistry::pack` +
/// `PackedAbilityRegistryGpu::upload`.
pub fn build_duel_25v25_registry() -> AbilityRegistry {
    let mut builder = AbilityRegistryBuilder::new();
    let strike_id = builder.register(build_strike_program());
    debug_assert_eq!(
        strike_id,
        AbilityId::new(STRIKE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "first registered program must land at AbilityId(1)",
    );
    let cleave_id = builder.register(build_cleave_program());
    debug_assert_eq!(
        cleave_id,
        AbilityId::new(CLEAVE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "second registered program must land at AbilityId(2)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `Duel25v25State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly two programs (Strike +
/// Cleave) at the expected slots, with the gate / area / effect shape
/// each rule's body in the .sim mirrors. If anything diverges the
/// panic message points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_duel_25v25_registry();
    assert_eq!(
        registry.len(),
        2,
        "duel_25v25 registry must contain exactly two programs (Strike + \
         Cleave); got {}",
        registry.len(),
    );

    // ---- Strike at AbilityId(1) ----
    let strike_id = AbilityId::new(STRIKE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let strike = registry
        .get(strike_id)
        .expect("Strike resolves to a program at AbilityId(1)");

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
    assert!(
        strike.per_effect_areas.is_empty()
            || strike.per_effect_areas[0].is_none(),
        "Strike must NOT have a per-effect area (single-target Damage); \
         got {:?}",
        strike.per_effect_areas,
    );

    // ---- Cleave at AbilityId(2) (AOE Path B production proof) ----
    let cleave_id = AbilityId::new(CLEAVE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let cleave = registry
        .get(cleave_id)
        .expect("Cleave resolves to a program at AbilityId(2)");

    assert_eq!(
        cleave.gate.cooldown_ticks, 0,
        "Cleave cooldown_ticks must be 0 (cadence is in the .sim verb gate \
         `world.tick % 5 == 0`; dispatcher doesn't consult cooldown_ticks today)",
    );
    assert!(
        cleave.gate.hostile_only,
        "Cleave must be hostile_only — same body-side team check as Strike",
    );
    match cleave.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 3.0,
            "Cleave range must be 3.0 — single-target metadata; the AOE \
             walk reads radius from per_effect_areas, not Area::range",
        ),
    }
    assert_eq!(
        cleave.effects.len(), 1,
        "Cleave must have exactly one effect (Damage 2.0)",
    );
    match &cleave.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 2.0,
            "Cleave damage must be 2.0 — kept low because each cast hits \
             multiple in-radius targets",
        ),
        other => panic!(
            "Cleave effect[0]: expected Damage(2.0), got {other:?}",
        ),
    }
    assert_eq!(
        cleave.per_effect_areas.len(), 1,
        "Cleave must have exactly one per-effect area entry (Circle at \
         slot 0); got {} entries",
        cleave.per_effect_areas.len(),
    );
    let area = cleave.per_effect_areas[0]
        .as_ref()
        .expect("Cleave per_effect_areas[0] must be Some(EffectAreaShape)");
    assert!(
        matches!(area.kind, ShapeKind::Circle),
        "Cleave per_effect_areas[0].kind must be Circle; got {:?}",
        area.kind,
    );
    assert_eq!(
        area.args[0], 1.0,
        "Cleave per_effect_areas[0].args[0] (radius) must be 1.0 (≤ \
         SPATIAL_CELL_SIZE=6.0 so the 27-cell walk covers the full \
         circle); got {}",
        area.args[0],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: two abilities at slots 1 + 2,
    /// Strike Damage 5.0 single-target, Cleave Damage 2.0 in Circle(1.0).
    /// Catches drift before construction-time panics surface in
    /// viz_tests / behavioural tests.
    #[test]
    fn registry_contains_strike_and_cleave_at_slots_one_and_two() {
        assert_ability_registry_matches_sim_constants();
    }
}
