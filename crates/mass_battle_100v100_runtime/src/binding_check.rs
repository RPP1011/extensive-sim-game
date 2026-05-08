//! mass_battle_100v100 apply_ability binding check.
//!
//! The registry source-of-truth lives in
//! `assets/ability_test/mass_battle_100v100/*.ability`; this module
//! re-parses the files via `dsl_ast::parse_ability_file` + lowers +
//! builds the registry through
//! `dsl_compiler::ability_registry::build_registry` at fixture-
//! construction time, then asserts each program lowered to the
//! constants the .sim verb bodies hand-mirror. The four programs land at:
//!
//!   - slot 1 / AbilityId(1) — Strike (Damage 30.0)
//!   - slot 2 / AbilityId(2) — Snipe (Damage 22.0)
//!   - slot 3 / AbilityId(3) — StunBolt (Stun 20 ticks)
//!   - slot 4 / AbilityId(4) — MassHeal (Heal 18.0)
//!
//! `apply_ability 1..=4` literals in
//! `assets/sim/mass_battle_100v100.sim`'s Strike + Snipe + StunBolt +
//! MassHeal verb bodies pin each slot. If any drifts the panic here
//! surfaces at fixture-construction time rather than as silent
//! wrong-ability dispatch.
//!
//! Pre-port this module hand-rolled four `AbilityProgram` builders. The
//! port to `.ability` files mirrors the canonical
//! `duel_abilities_runtime` pattern (no engine-side TOML loader; the
//! `.ability` DSL is the only authoring surface for ability programs).

use std::path::PathBuf;

use engine::ability::program::EffectOp;
use engine::ability::AbilityId;

/// Strike is registered first so it always lands at AbilityId(1).
/// The `apply_ability 1` literal in
/// `assets/sim/mass_battle_100v100.sim::Strike` pins this slot.
pub const STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// Snipe is registered second so it lands at AbilityId(2). The
/// `apply_ability 2` literal in
/// `assets/sim/mass_battle_100v100.sim::Snipe` pins this slot.
pub const SNIPE_EXPECTED_ABILITY_ID: u32 = 2;

/// StunBolt control-status proof (200-agent scale, 2026-05-07) —
/// registered third so it lands at AbilityId(3).
pub const STUN_BOLT_EXPECTED_ABILITY_ID: u32 = 3;

/// MassHeal recovery-dynamics proof (200-agent scale, 2026-05-07) —
/// registered fourth so it lands at AbilityId(4).
pub const MASS_HEAL_EXPECTED_ABILITY_ID: u32 = 4;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/mass_battle_100v100/`. Shared by the
/// binding check AND the GPU upload site in `lib.rs`.
///
/// Slot order is the source-order names array literal here. Mirrors
/// the canonical `duel_abilities_runtime::binding_check::
/// build_duel_abilities_registry` pattern.
pub fn build_mass_battle_100v100_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("mass_battle_100v100");

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
        "Snipe.ability",
        "StunBolt.ability",
        "MassHeal.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over mass_battle_100v100 corpus")
}

/// Single binding-check entry point. Called once from
/// `MassBattle100v100State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly four programs (Strike +
/// Snipe + StunBolt + MassHeal) and that each slot, gate, area, and
/// effect match the .sim's hand-mirrored verb behaviour. If anything
/// diverges the panic message points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_mass_battle_100v100_registry();
    let registry = &built.registry;
    assert_eq!(
        registry.len(),
        4,
        "mass_battle_100v100 registry must contain exactly four programs \
         (Strike + Snipe + StunBolt + MassHeal); got {}",
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

    // ---- StunBolt at AbilityId(3) (control-status proof at 200-agent
    //      scale — first non-Damage EffectOp in this fixture) ----
    let stun_bolt_id = AbilityId::new(STUN_BOLT_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let stun_bolt = registry
        .get(stun_bolt_id)
        .expect("StunBolt resolves to a program at AbilityId(3)");
    assert_eq!(
        stun_bolt.gate.cooldown_ticks, 0,
        "StunBolt cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 7 == 0`)",
    );
    assert!(
        stun_bolt.gate.hostile_only,
        "StunBolt must be hostile_only — DPS-vs-enemy gate, same as Snipe",
    );
    match stun_bolt.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 999.0,
            "StunBolt range must be 999.0 — matches \
             config.combat.perception_radius (pair-field scoring, no \
             spatial narrowing — same as Strike + Snipe)",
        ),
    }
    assert_eq!(
        stun_bolt.effects.len(), 1,
        "StunBolt must have exactly one effect (Stun 20 ticks)",
    );
    match &stun_bolt.effects[0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 20,
            "StunBolt stun duration_ticks must be 20 (= 2s at 100ms \
             tick); .sim verb hand-mirrors via the apply_ability \
             dispatcher's `expires_at_tick = world.tick + 20` chronicle \
             write",
        ),
        other => panic!(
            "StunBolt effect[0]: expected Stun(duration_ticks=20), got {other:?}",
        ),
    }

    // ---- MassHeal at AbilityId(4) (recovery-dynamics proof at
    //      200-agent scale — first friendly-targeted EffectOp in this
    //      fixture) ----
    let mass_heal_id = AbilityId::new(MASS_HEAL_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let mass_heal = registry
        .get(mass_heal_id)
        .expect("MassHeal resolves to a program at AbilityId(4)");
    assert_eq!(
        mass_heal.gate.cooldown_ticks, 0,
        "MassHeal cooldown_ticks must be 0 (cadence is in the .sim verb \
         gate `world.tick % 11 == 0`)",
    );
    assert!(
        !mass_heal.gate.hostile_only,
        "MassHeal must NOT be hostile_only — `target ally` semantic \
         (the .sim's body-side level-pair check uses the ally \
         predicate, dispatching on same-team agents)",
    );
    match mass_heal.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 999.0,
            "MassHeal range must be 999.0 — matches \
             config.combat.perception_radius (pair-field scoring, no \
             spatial narrowing — same as Strike + Snipe + StunBolt)",
        ),
    }
    assert_eq!(
        mass_heal.effects.len(), 1,
        "MassHeal must have exactly one effect (Heal 18.0); got {}",
        mass_heal.effects.len(),
    );
    match &mass_heal.effects[0] {
        EffectOp::Heal { amount } => assert_eq!(
            *amount, 18.0,
            "MassHeal heal amount must be 18.0 — sized so a single cast \
             visibly recovers HP on a damaged target without instantly \
             capping at max_hp=100",
        ),
        other => panic!(
            "MassHeal effect[0]: expected Heal(18.0), got {other:?}",
        ),
    }
    assert!(
        mass_heal.per_effect_areas.is_empty()
            || mass_heal.per_effect_areas[0].is_none(),
        "MassHeal must NOT have a per-effect area (single-target \
         Heal); got {:?}",
        mass_heal.per_effect_areas,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: four abilities (Strike at
    /// slot 1, Snipe at slot 2, StunBolt at slot 3, MassHeal at slot
    /// 4), each with the expected gate/area/effect. Catches drift
    /// before construction-time panics surface in viz_tests /
    /// behavioural tests.
    #[test]
    fn registry_contains_strike_snipe_stun_bolt_mass_heal_at_expected_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
