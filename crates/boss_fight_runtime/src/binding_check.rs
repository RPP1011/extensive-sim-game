//! boss_fight apply_ability binding check.
//!
//! The registry source-of-truth lives in
//! `assets/ability_test/boss_fight/*.ability`; this module re-parses
//! the files via `dsl_ast::parse_ability_file` + lowers + builds the
//! registry through `dsl_compiler::ability_registry::build_registry` at
//! fixture-construction time, then asserts each program lowered to the
//! constants the .sim verb bodies hand-mirror. The four programs land at:
//!
//!   - slot 1 / AbilityId(1) — BossStrike (Damage 50.0)
//!   - slot 2 / AbilityId(2) — HeroAttack (Damage 35.0)
//!   - slot 3 / AbilityId(3) — HeroStun (Stun 15 ticks)
//!   - slot 4 / AbilityId(4) — HeroHeal (Heal 25.0)
//!
//! `apply_ability 1..=4` literals in `assets/sim/boss_fight.sim`'s
//! BossStrike + HeroAttack + HeroStun + HeroHeal verb bodies pin each
//! slot. If any drifts the panic here surfaces at fixture-construction
//! time rather than as silent wrong-ability dispatch.
//!
//! Pre-port this module hand-rolled four `AbilityProgram` builders. The
//! port to `.ability` files mirrors the canonical
//! `duel_abilities_runtime` pattern (no engine-side TOML loader; the
//! `.ability` DSL is the only authoring surface for ability programs).

use std::path::PathBuf;

use engine::ability::program::EffectOp;
use engine::ability::AbilityId;

/// BossStrike is registered first so it lands at AbilityId(1).
pub const BOSS_STRIKE_EXPECTED_ABILITY_ID: u32 = 1;

/// HeroAttack is registered second so it lands at AbilityId(2).
pub const HERO_ATTACK_EXPECTED_ABILITY_ID: u32 = 2;

/// HeroStun control-status proof (boss_fight, 2026-05-07) — registered
/// third so it lands at AbilityId(3).
pub const HERO_STUN_EXPECTED_ABILITY_ID: u32 = 3;

/// HeroHeal apply_ability proof (boss_fight, 2026-05-07) — registered
/// fourth so it lands at AbilityId(4).
pub const HERO_HEAL_EXPECTED_ABILITY_ID: u32 = 4;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/boss_fight/`. Shared by the binding check
/// AND the GPU upload site in `lib.rs`.
///
/// Slot order is the source-order names array literal here. Mirrors
/// the canonical `duel_abilities_runtime::binding_check::
/// build_duel_abilities_registry` pattern.
pub fn build_boss_fight_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("boss_fight");

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
        "BossStrike.ability",
        "HeroAttack.ability",
        "HeroStun.ability",
        "HeroHeal.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over boss_fight corpus")
}

/// Single binding-check entry point. Called once from
/// `BossFightState::new` at fixture-construction time.
///
/// Asserts the registry contains exactly four programs and that each
/// slot, gate, area, and effect matches the .sim's hand-mirrored
/// BossStrike + HeroAttack + HeroStun + HeroHeal behaviour. If anything
/// diverges the panic message points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_boss_fight_registry();
    let registry = &built.registry;
    assert_eq!(
        registry.len(),
        4,
        "boss_fight registry must contain exactly four programs \
         (BossStrike + HeroAttack + HeroStun + HeroHeal); got {}",
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

    // ---- HeroHeal at slot 4 (HeroHeal apply_ability proof — first
    //      Heal EffectOp in this fixture) ----
    let hh_id = AbilityId::new(HERO_HEAL_EXPECTED_ABILITY_ID)
        .expect("non-zero AbilityId");
    let hh = registry
        .get(hh_id)
        .expect("HeroHeal resolves to a program at AbilityId(4)");
    assert_eq!(
        hh.gate.cooldown_ticks, 0,
        "HeroHeal cooldown_ticks must be 0 (cadence is in the .sim \
         verb gate `world.tick % 7 == 0`)",
    );
    assert!(
        !hh.gate.hostile_only,
        "HeroHeal must NOT be hostile_only — `target ally` semantic \
         (the .sim's score expression picks Hero allies via \
         `target.creature_type == Hero && target != self`)",
    );
    match hh.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 50.0,
            "HeroHeal range must be 50.0 — wide single-target metadata, \
             boss_fight scores per-pair with no spatial filter",
        ),
    }
    assert_eq!(
        hh.effects.len(), 1,
        "HeroHeal must have exactly one effect (Heal 25.0)",
    );
    match &hh.effects[0] {
        EffectOp::Heal { amount } => assert_eq!(
            *amount, 25.0,
            "HeroHeal heal amount must be 25.0 — generous enough that \
             one cast on a hp=50 ally visibly recovers HP (50 + 25 = \
             75) under the per-agent max_hp=100 clamp",
        ),
        other => panic!(
            "HeroHeal effect[0]: expected Heal(25.0), got {other:?}",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the registry build pattern: four abilities at slots 1..4
    /// (BossStrike, HeroAttack, HeroStun, HeroHeal), with the asymmetric
    /// Damage amounts (Boss 50, Hero 35), HeroStun's 15-tick Stun, and
    /// HeroHeal's 25.0 Heal at the expected gates/areas. Catches drift
    /// before construction-time panics surface in viz_tests /
    /// behavioural tests.
    #[test]
    fn registry_contains_bossstrike_heroattack_herostun_heroheal_at_pinned_slots() {
        assert_ability_registry_matches_sim_constants();
    }
}
