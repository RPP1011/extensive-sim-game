//! Wave 1 acceptance binding check — the heart of the duel_abilities
//! fixture.
//!
//! Re-parses the source-of-truth `.ability` files from
//! `assets/ability_test/duel_abilities/` at fixture-construction time,
//! lowers them through the Wave 1.0 + 1.6 + 1.7 + 1.9 pipeline
//! (parser → AbilityProgram → AbilityRegistry → PackedAbilityRegistry),
//! and asserts every program lowered to constants that MATCH the
//! hand-mirrored .sim verb constants in
//! `assets/sim/duel_abilities.sim`.
//!
//! If any assertion here fires the panic message points at the exact
//! .sim/.ability divergence. This is the entire acceptance contract for
//! Wave 1: the .ability files lower to the values the runtime
//! hand-mirrored, end-to-end through every pipeline stage.
//!
//! The check pays a 3-file read + parse + lower + pack on each fixture
//! construction. That cost is trivial relative to GPU init and provides
//! a hard guarantee that any drift between the .ability files and the
//! .sim verb constants surfaces at startup, not later in a stale
//! integration test. Per task spec we run the check unconditionally
//! (not behind `cfg(debug_assertions)`).

use std::path::PathBuf;

use engine::ability::{
    program::{Area, EffectOp},
    PackedAbilityRegistry,
};

/// The single binding-check entry point. Called once from
/// `DuelAbilitiesState::new` at fixture-construction time.
///
/// Panics with a descriptive message on any divergence between the
/// .ability files and the .sim hand-mirrored constants.
pub fn assert_ability_registry_matches_sim_constants() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set by cargo");
    let corpus = PathBuf::from(manifest)
        .join("..")
        .join("..")
        .join("assets")
        .join("ability_test")
        .join("duel_abilities");

    let strike_src = std::fs::read_to_string(corpus.join("Strike.ability"))
        .expect("read Strike.ability");
    let shieldup_src = std::fs::read_to_string(corpus.join("ShieldUp.ability"))
        .expect("read ShieldUp.ability");
    let mend_src = std::fs::read_to_string(corpus.join("Mend.ability"))
        .expect("read Mend.ability");

    let files = vec![
        (
            "Strike.ability".to_string(),
            dsl_ast::parse_ability_file(&strike_src)
                .expect("parse Strike.ability"),
        ),
        (
            "ShieldUp.ability".to_string(),
            dsl_ast::parse_ability_file(&shieldup_src)
                .expect("parse ShieldUp.ability"),
        ),
        (
            "Mend.ability".to_string(),
            dsl_ast::parse_ability_file(&mend_src)
                .expect("parse Mend.ability"),
        ),
    ];
    let built = dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over duel_abilities corpus");

    // ---- Strike: cooldown 10 ticks, range 5.0, hostile_only, damage 30.0 ----
    let strike_id = *built.names.get("Strike")
        .expect("Strike registered in name table");
    let strike = built.registry.get(strike_id)
        .expect("Strike resolves to a program");
    assert_eq!(
        strike.gate.cooldown_ticks, 10,
        "Strike cooldown must be 10 ticks (1s) — .sim verb gate \
         (world.tick % 10 == 0) and .ability `cooldown: 1s` must agree",
    );
    assert!(
        strike.gate.hostile_only,
        "Strike must be hostile_only — .ability `target: enemy`",
    );
    match strike.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 5.0,
            "Strike range must be 5.0 — .ability `range: 5.0`, .sim \
             config.combat.strike_range",
        ),
    }
    assert_eq!(
        strike.effects.len(), 1,
        "Strike must have exactly one effect (damage 30)",
    );
    match &strike.effects[0] {
        EffectOp::Damage { amount } => assert_eq!(
            *amount, 30.0,
            "Strike damage must be 30.0 — .ability `damage 30`, .sim \
             config.combat.strike_damage",
        ),
        other => panic!(
            "Strike effect[0]: expected Damage(30.0), got {other:?}",
        ),
    }

    // ---- ShieldUp: cooldown 40 ticks, self-target, shield 50.0 ----
    let shieldup_id = *built.names.get("ShieldUp")
        .expect("ShieldUp registered in name table");
    let shieldup = built.registry.get(shieldup_id)
        .expect("ShieldUp resolves to a program");
    assert_eq!(
        shieldup.gate.cooldown_ticks, 40,
        "ShieldUp cooldown must be 40 ticks (4s) — .sim verb gate \
         (world.tick % 40 == 0) and .ability `cooldown: 4s` must agree",
    );
    assert!(
        !shieldup.gate.hostile_only,
        "ShieldUp must NOT be hostile_only — .ability `target: self`",
    );
    assert_eq!(
        shieldup.effects.len(), 1,
        "ShieldUp must have exactly one effect (shield 50)",
    );
    match &shieldup.effects[0] {
        EffectOp::Shield { amount } => assert_eq!(
            *amount, 50.0,
            "ShieldUp shield must be 50.0 — .ability `shield 50`, .sim \
             config.combat.shieldup_amount",
        ),
        other => panic!(
            "ShieldUp effect[0]: expected Shield(50.0), got {other:?}",
        ),
    }

    // ---- Mend: cooldown 30 ticks, self-target, heal 25.0 ----
    let mend_id = *built.names.get("Mend")
        .expect("Mend registered in name table");
    let mend = built.registry.get(mend_id)
        .expect("Mend resolves to a program");
    assert_eq!(
        mend.gate.cooldown_ticks, 30,
        "Mend cooldown must be 30 ticks (3s) — .sim verb gate \
         (world.tick % 30 == 0) and .ability `cooldown: 3s` must agree",
    );
    assert!(
        !mend.gate.hostile_only,
        "Mend must NOT be hostile_only — .ability `target: self`",
    );
    assert_eq!(
        mend.effects.len(), 1,
        "Mend must have exactly one effect (heal 25)",
    );
    match &mend.effects[0] {
        EffectOp::Heal { amount } => assert_eq!(
            *amount, 25.0,
            "Mend heal must be 25.0 — .ability `heal 25`, .sim \
             config.combat.mend_amount",
        ),
        other => panic!(
            "Mend effect[0]: expected Heal(25.0), got {other:?}",
        ),
    }

    // ---- Smoke-pack: prove the GPU SoA layout works on this corpus ----
    // PackedAbilityRegistry::pack runs its own per-program packing
    // walk; if the registry contains anything pack can't encode this
    // panics. Confirms the lowering output is consumable by the
    // Wave 1.9 GPU buffer producer.
    let packed = PackedAbilityRegistry::pack(&built.registry);
    assert_eq!(
        packed.n_abilities, 3,
        "PackedAbilityRegistry must contain exactly 3 abilities",
    );
}
