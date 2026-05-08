//! duel_25v25 apply_ability binding check.
//!
//! The registry source-of-truth lives in
//! `assets/ability_test/duel_25v25/*.ability`; this module re-parses the
//! files via `dsl_ast::parse_ability_file` + lowers + builds the
//! registry through `dsl_compiler::ability_registry::build_registry` at
//! fixture-construction time, then asserts each program lowered to the
//! constants the .sim verb bodies hand-mirror. The four programs land
//! at:
//!
//!   - slot 1 / AbilityId(1) — Strike (single-target Damage 5.0)
//!   - slot 2 / AbilityId(2) — Cleave (AOE Damage 2.0 in Circle(1.0))
//!   - slot 3 / AbilityId(3) — ConcussiveCleave (AOE Damage 3.0 + Stun 15)
//!   - slot 4 / AbilityId(4) — HealPulse (single-target Heal 15.0)
//!
//! `apply_ability 1..=4` literals in `assets/sim/duel_25v25.sim`
//! (ScanAndStrike + ScanAndCleave + ScanAndConcussiveCleave +
//! ScanAndHeal bodies) pin each slot. If any drifts the panic here
//! surfaces at fixture-construction time rather than as silent
//! wrong-ability dispatch.
//!
//! Pre-port this module loaded the registry via
//! `AbilityRegistry::from_toml(...)`. The TOML loader was retired in
//! favour of the `.ability` DSL surface, which is more expressive
//! (per-effect predicates, scalings, modifiers, deliver-block hooks).
//! The slot-pin assertions stayed verbatim — they're the load-bearing
//! contract between the registry's slot order and the .sim's
//! `apply_ability N` literals.

use std::path::PathBuf;

use engine::ability::program::{EffectOp, ShapeKind};
use engine::ability::AbilityId;

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

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/duel_25v25/`. Shared by the binding check
/// AND the GPU upload site in `lib.rs`.
///
/// Slot order is the source-order names array literal here, pinned by
/// the slot-pin assertions in
/// `assert_ability_registry_matches_sim_constants`. Drift surfaces as a
/// panic at fixture-construction time. Mirrors the canonical
/// `duel_abilities_runtime` pattern at `binding_check.rs::
/// build_duel_abilities_registry`.
///
/// Panics on any read/parse/build error — these would also fire from
/// the binding check at fixture-construction time, so any failure here
/// points at the same .ability source defect.
pub fn build_duel_25v25_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("duel_25v25");

    let read = |name: &str| {
        let path = corpus.join(name);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    };
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    let names = [
        "Strike.ability",
        "Cleave.ability",
        "ConcussiveCleave.ability",
        "HealPulse.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over duel_25v25 corpus")
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
    let built = build_duel_25v25_registry();
    let registry = &built.registry;
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
        "HealPulse must NOT be hostile_only — `target ally` semantic \
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
