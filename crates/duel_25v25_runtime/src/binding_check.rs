//! duel_25v25 apply_ability binding check.
//!
//! Builds the runtime's four-program AbilityRegistry (Strike at slot 1,
//! Cleave at slot 2, ConcussiveCleave at slot 3, HealPulse at slot 4)
//! and asserts that the registered slot IDs match the `apply_ability 1`,
//! `apply_ability 2`, `apply_ability 3`, and `apply_ability 4` literals
//! hardcoded in `assets/sim/duel_25v25.sim`'s ScanAndStrike + ScanAndCleave
//! + ScanAndConcussiveCleave + ScanAndHeal bodies. If any slot drifts
//! (e.g. someone reorders registration), the panic here surfaces at
//! fixture-construction time rather than as silent wrong-ability
//! dispatch.
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

/// ConcussiveCleave is registered third — so it lands at AbilityId(3).
/// The `apply_ability 3` literal in
/// `assets/sim/duel_25v25.sim::ScanAndConcussiveCleave` pins this slot.
/// Multi-effect AOE Cleave+Stun (Path B production proof, 2026-05-07).
pub const CONCUSSIVE_CLEAVE_EXPECTED_ABILITY_ID: u32 = 3;

/// HealPulse is registered fourth — so it lands at AbilityId(4). The
/// `apply_ability 4` literal in `assets/sim/duel_25v25.sim::ScanAndHeal`
/// pins this slot. Single-target healing for chronicle-pipeline recovery
/// dynamics in 50-agent combat (2026-05-07).
pub const HEAL_PULSE_EXPECTED_ABILITY_ID: u32 = 4;

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

/// duel_25v25's ConcussiveCleave registry-resident program (multi-effect
/// AOE Path B production proof, 2026-05-07).
///
/// Shape: TWO-effect program with BOTH effects sharing Circle(1.0) — the
/// dispatcher's per-effect-slot loop walks effects[0]=Damage(3.0) and
/// effects[1]=Stun(15 ticks) in order. For each AOE-equipped slot the
/// 27-cell walk emits ONE chronicle record per in-radius candidate, so
/// each ConcussiveCleave cast on a target produces:
///   - kind=26 EffectDamageApplied per in-radius agent (Damage slot)
///   - kind=29 EffectStunApplied per in-radius agent (Stun slot)
///
/// duel_25v25's `ApplyDamageFromChronicle` rule already drains kind=26
/// records into `Damaged → ApplyDamage`. The new `ApplyStunFromChronicle`
/// rule (added alongside this slice) drains kind=29 records straight into
/// the per-agent `stun_expires_at_tick` SoA via
/// `agents.set_stun_expires_at_tick(t, e)`.
///
/// Damage 3.0 is a midpoint between Strike's 5.0 and Cleave's 2.0 —
/// each ConcussiveCleave cast hits multiple in-radius targets (like
/// Cleave) AND stuns each, so the per-cast effective output is hefty;
/// keeping per-target damage at 3.0 prevents the seam from collapsing.
///
/// Stun duration_ticks=15 (= 1.5s at 100ms tick) gives the engine a
/// long enough window that the cast-gate observably suppresses
/// subsequent verb dispatches BEFORE the stun expires (a tighter 5-tick
/// duration would race with the % 7 cadence). The dispatcher
/// pre-computes `expires_at_tick = world.tick + duration_ticks` at
/// chronicle-write time.
///
/// Cadence is gated in the .sim by `world.tick % 7 == 0` (coprime with
/// Strike's % 2 and Cleave's % 5 so all three cadences interleave).
fn build_concussive_cleave_program() -> AbilityProgram {
    let mut concussive = AbilityProgram::new_single_target(
        /*range*/ 3.0,
        Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
        [
            EffectOp::Damage { amount: 3.0 },
            EffectOp::Stun { duration_ticks: 15 },
        ],
    );
    // Effect 0 (Damage): Circle(1.0). Each AOE target receives one
    // EffectDamageApplied chronicle record.
    concussive.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [1.0, 0.0, 0.0, 0.0],
    }));
    // Effect 1 (Stun): Circle(1.0) — SAME shape as effect 0 so each
    // in-radius target receives BOTH the damage AND the stun. The
    // per-effect-slot loop in the dispatcher walks both slots
    // independently; when both share Circle(1.0), the same set of
    // candidates lands in each slot's chronicle write pass.
    concussive.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [1.0, 0.0, 0.0, 0.0],
    }));
    concussive
}

/// duel_25v25's HealPulse registry-resident program (single-target
/// recovery for chronicle-pipeline healing in 50-agent combat,
/// 2026-05-07).
///
/// Shape: single-target program (no `per_effect_areas` entries; the
/// dispatcher reads sentinel 0xFFu and falls through to the
/// single-target chain — same shape as Strike). One `EffectOp::Heal {
/// amount: 15.0 }` lowered to chronicle kind=27 EffectHealApplied
/// records. The fused `ApplyDamageFromChronicle_and_ApplyStunFromChronicle`
/// kernel grows a third arm (kind=27) that writes
/// `agents.set_hp(t, min(agents.hp(t) + amt, agents.max_hp(t)))` directly,
/// clamping overflow to the per-agent max_hp SoA column (init 100.0).
///
/// `range: 1.5` matches Cleave's @spatial annotation radius — the
/// .sim's ScanAndHeal walks `nearby_enemies(self)` (informational
/// filter; the actual neighbour set is the 27-cell ring) then inverts
/// the team check body-side (`if (other.creature_type ==
/// self.creature_type ...)`) so the dispatch targets a SAME-TEAM ally.
/// `hostile_only: false` matches the `target friend` semantic — when
/// future predicate dispatch starts consulting program metadata, this
/// flag will scope target selection to allies; today the body-side
/// `if` enforces the gate.
///
/// Heal amount 15.0 is intentionally generous: with strike_damage=5.0
/// firing every 2 ticks at the seam and Cleave AOE adding 2.0 every
/// 5 ticks, an ally taking sustained damage drops 5-7 hp/tick. A
/// 15.0 heal every 5 ticks (~3 hp/tick averaged) partially offsets
/// the seam pressure without making fights unwinnable — visible
/// recovery dynamic on the HP curves.
fn build_heal_pulse_program() -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 1.5,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Heal { amount: 15.0 }],
    )
}

/// Build the duel_25v25 AbilityRegistry — Strike at AbilityId(1),
/// Cleave at AbilityId(2), ConcussiveCleave at AbilityId(3),
/// HealPulse at AbilityId(4). Returns the frozen registry; callers
/// pack + upload via `PackedAbilityRegistry::pack` +
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
    let concussive_id = builder.register(build_concussive_cleave_program());
    debug_assert_eq!(
        concussive_id,
        AbilityId::new(CONCUSSIVE_CLEAVE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "third registered program must land at AbilityId(3)",
    );
    let heal_pulse_id = builder.register(build_heal_pulse_program());
    debug_assert_eq!(
        heal_pulse_id,
        AbilityId::new(HEAL_PULSE_EXPECTED_ABILITY_ID).expect("non-zero AbilityId"),
        "fourth registered program must land at AbilityId(4)",
    );
    builder.build()
}

/// Single binding-check entry point. Called once from
/// `Duel25v25State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly four programs (Strike +
/// Cleave + ConcussiveCleave + HealPulse) at the expected slots, with
/// the gate / area / effect shape each rule's body in the .sim
/// mirrors. If anything diverges the panic message points at the exact
/// divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let registry = build_duel_25v25_registry();
    assert_eq!(
        registry.len(),
        4,
        "duel_25v25 registry must contain exactly four programs \
         (Strike + Cleave + ConcussiveCleave + HealPulse); got {}",
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

    // ---- ConcussiveCleave at AbilityId(3) (Multi-effect AOE Path B
    //      production proof — Damage + Stun in same Circle(1.0)) ----
    let concussive_id = AbilityId::new(CONCUSSIVE_CLEAVE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let concussive = registry
        .get(concussive_id)
        .expect("ConcussiveCleave resolves to a program at AbilityId(3)");

    assert_eq!(
        concussive.gate.cooldown_ticks, 0,
        "ConcussiveCleave cooldown_ticks must be 0 (cadence is in the \
         .sim verb gate `world.tick % 7 == 0`; dispatcher doesn't \
         consult cooldown_ticks today)",
    );
    assert!(
        concussive.gate.hostile_only,
        "ConcussiveCleave must be hostile_only — same body-side team \
         check as Strike + Cleave",
    );
    match concussive.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 3.0,
            "ConcussiveCleave range must be 3.0 — single-target metadata; \
             both AOE walks read radius from per_effect_areas[i].args[0], \
             not Area::range",
        ),
    }
    assert_eq!(
        concussive.effects.len(), 2,
        "ConcussiveCleave must have exactly two effects (Damage 3.0 + \
         Stun 15 ticks); got {}",
        concussive.effects.len(),
    );
    match &concussive.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 3.0,
            "ConcussiveCleave effect[0] (Damage) amount must be 3.0",
        ),
        other => panic!(
            "ConcussiveCleave effect[0]: expected Damage(3.0), got {other:?}",
        ),
    }
    match &concussive.effects[1] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 15,
            "ConcussiveCleave effect[1] (Stun) duration_ticks must be \
             15 (= 1.5s at 100ms tick); got {}",
            duration_ticks,
        ),
        other => panic!(
            "ConcussiveCleave effect[1]: expected Stun(duration_ticks=15), \
             got {other:?}",
        ),
    }
    assert_eq!(
        concussive.per_effect_areas.len(), 2,
        "ConcussiveCleave must have exactly two per-effect area entries \
         (Circle(1.0) at slots 0 and 1, so both Damage and Stun expand \
         across the same in-radius candidates); got {} entries",
        concussive.per_effect_areas.len(),
    );
    for (i, slot_label) in [(0usize, "Damage"), (1usize, "Stun")] {
        let area_i = concussive.per_effect_areas[i]
            .as_ref()
            .unwrap_or_else(|| panic!(
                "ConcussiveCleave per_effect_areas[{i}] ({slot_label} \
                 slot) must be Some(EffectAreaShape) — the dispatcher's \
                 multi-effect AOE walk requires per-slot shape entries \
                 for both effects",
            ));
        assert!(
            matches!(area_i.kind, ShapeKind::Circle),
            "ConcussiveCleave per_effect_areas[{i}] ({slot_label}) kind \
             must be Circle; got {:?}",
            area_i.kind,
        );
        assert_eq!(
            area_i.args[0], 1.0,
            "ConcussiveCleave per_effect_areas[{i}] ({slot_label}) \
             radius must be 1.0 (matches Cleave's slot 0); got {}",
            area_i.args[0],
        );
    }

    // ---- HealPulse at AbilityId(4) (single-target ally healing for
    //      chronicle-pipeline recovery dynamics, 2026-05-07) ----
    let heal_pulse_id = AbilityId::new(HEAL_PULSE_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let heal_pulse = registry
        .get(heal_pulse_id)
        .expect("HealPulse resolves to a program at AbilityId(4)");

    assert_eq!(
        heal_pulse.gate.cooldown_ticks, 0,
        "HealPulse cooldown_ticks must be 0 (cadence is in the .sim \
         verb gate `world.tick % 5 == 0`; dispatcher doesn't consult \
         cooldown_ticks today)",
    );
    assert!(
        !heal_pulse.gate.hostile_only,
        "HealPulse must NOT be hostile_only — `target friend` semantic \
         (the .sim's body-side check inverts the team test, dispatching \
         on same-team agents)",
    );
    match heal_pulse.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 1.5,
            "HealPulse range must be 1.5 — matches the @spatial \
             annotation radius in duel_25v25.sim",
        ),
    }
    assert_eq!(
        heal_pulse.effects.len(), 1,
        "HealPulse must have exactly one effect (Heal 15.0); got {}",
        heal_pulse.effects.len(),
    );
    match &heal_pulse.effects[0] {
        EffectOp::Heal { amount } => assert_eq!(
            *amount, 15.0,
            "HealPulse heal amount must be 15.0 — generous enough that \
             one cast visibly recovers the seam-tick HP drain (~5-7 \
             dmg/tick)",
        ),
        other => panic!(
            "HealPulse effect[0]: expected Heal(15.0), got {other:?}",
        ),
    }
    assert!(
        heal_pulse.per_effect_areas.is_empty()
            || heal_pulse.per_effect_areas[0].is_none(),
        "HealPulse must NOT have a per-effect area (single-target \
         Heal); got {:?}",
        heal_pulse.per_effect_areas,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: four abilities at slots 1..4:
    /// Strike Damage 5.0 single-target, Cleave Damage 2.0 in Circle(1.0),
    /// ConcussiveCleave [Damage 3.0, Stun 15 ticks] both in Circle(1.0),
    /// HealPulse Heal 15.0 single-target. Catches drift before
    /// construction-time panics surface in viz_tests / behavioural tests.
    #[test]
    fn registry_contains_strike_cleave_concussive_heal_pulse_at_slots_one_through_four() {
        assert_ability_registry_matches_sim_constants();
    }
}
