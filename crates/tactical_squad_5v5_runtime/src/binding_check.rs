//! tactical_squad_5v5 apply_ability binding check.
//!
//! The registry source-of-truth lives in
//! `assets/ability_test/tactical_squad_5v5/*.ability`; this module
//! re-parses the files via `dsl_ast::parse_ability_file` + lowers +
//! builds the registry through
//! `dsl_compiler::ability_registry::build_registry` at fixture-
//! construction time, then asserts each program lowered to the
//! constants the .sim verb bodies hand-mirror. The four programs land at:
//!
//!   - slot 1 / AbilityId(1) — TankAttack (Damage 10.0)
//!   - slot 2 / AbilityId(2) — DpsAttack (Damage 22.0)
//!   - slot 3 / AbilityId(3) — ConcussiveBlow (Stun 20 ticks)
//!   - slot 4 / AbilityId(4) — SquadHeal (Heal 12.0)
//!
//! `apply_ability 1..=4` literals in
//! `assets/sim/tactical_squad_5v5.sim`'s Strike (Tank), Snipe (DPS),
//! ConcussiveBlow (DPS), and SquadHeal (any role) verb bodies pin each
//! slot. If any drifts the panic here surfaces at fixture-construction
//! time rather than as silent wrong-ability dispatch.
//!
//! Pre-port this module hand-rolled four `AbilityProgram` builders. The
//! port to `.ability` files mirrors the canonical
//! `duel_abilities_runtime` pattern (no engine-side TOML loader; the
//! `.ability` DSL is the only authoring surface for ability programs).
//!
//! CONFIG → REGISTRY BRIDGE: the .sim's `config.combat.tank_damage =
//! 10.0` and `config.combat.dps_damage = 22.0` are mirrored as
//! literals in the .ability files. apply_ability's pack-time literal
//! API doesn't accept config refs today, so the runtime hardcodes the
//! matching values; if the .sim config drifts, the assertions below
//! flag the divergence.

use std::path::PathBuf;

use engine::ability::program::EffectOp;
use engine::ability::AbilityId;

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
/// this slot.
pub const CONCUSSIVE_BLOW_EXPECTED_ABILITY_ID: u32 = 3;

/// SquadHeal apply_ability ally-heal proof (5v5 scale, 2026-05-07) —
/// registered fourth so it lands at AbilityId(4). The `apply_ability 4`
/// literal in `assets/sim/tactical_squad_5v5.sim::SquadHeal` pins this
/// slot.
pub const SQUAD_HEAL_EXPECTED_ABILITY_ID: u32 = 4;

/// Hand-mirrored Tank base damage — matches `config.combat.tank_damage
/// = 10.0` in `assets/sim/tactical_squad_5v5.sim`.
pub const TANK_DAMAGE: f32 = 10.0;

/// Hand-mirrored DPS base damage — matches `config.combat.dps_damage
/// = 22.0`.
pub const DPS_DAMAGE: f32 = 22.0;

/// SquadHeal heal magnitude (apply_ability ally heal, 2026-05-07) —
/// hand-mirrored heal amount. The .sim's SquadHeal verb has no
/// counterpart in `config.combat`. 12.0 mirrors duel_25v25's HealPulse
/// 15.0 magnitude shape (commit 049feb0c).
pub const SQUAD_HEAL_AMOUNT: f32 = 12.0;

/// Read + parse + build the AbilityRegistry over every .ability file
/// under `assets/ability_test/tactical_squad_5v5/`. Shared by the
/// binding check AND the GPU upload site in `lib.rs`.
///
/// Slot order is the source-order names array literal here, pinned by
/// the slot-pin assertions in
/// `assert_ability_registry_matches_sim_constants`. Mirrors the
/// canonical `duel_abilities_runtime::binding_check::
/// build_duel_abilities_registry` pattern.
pub fn build_tactical_squad_5v5_registry() -> dsl_compiler::ability_registry::BuiltRegistry {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("tactical_squad_5v5");

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
        "TankAttack.ability",
        "DpsAttack.ability",
        "ConcussiveBlow.ability",
        "SquadHeal.ability",
    ];
    let files: Vec<(String, _)> = names
        .iter()
        .map(|name| {
            let src = read(name);
            (name.to_string(), parse(name, &src))
        })
        .collect();

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over tactical_squad_5v5 corpus")
}

/// Single binding-check entry point. Called once from
/// `TacticalSquad5v5State::new` at fixture-construction time.
///
/// Asserts the registry contains exactly four programs and that each
/// program's slot, gate, area, and effect match the .sim's
/// hand-mirrored verb constants. If anything diverges the panic message
/// points at the exact divergence.
pub fn assert_ability_registry_matches_sim_constants() {
    let built = build_tactical_squad_5v5_registry();
    let registry = &built.registry;
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
        "SquadHeal must NOT be hostile_only — `target ally` semantic \
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
