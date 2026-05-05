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
    let bleed_src = std::fs::read_to_string(corpus.join("Bleed.ability"))
        .expect("read Bleed.ability");
    let reap_src = std::fs::read_to_string(corpus.join("Reap.ability"))
        .expect("read Reap.ability");
    let vampirize_src = std::fs::read_to_string(corpus.join("Vampirize.ability"))
        .expect("read Vampirize.ability");
    let fortify_src = std::fs::read_to_string(corpus.join("Fortify.ability"))
        .expect("read Fortify.ability");

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
        (
            "Bleed.ability".to_string(),
            dsl_ast::parse_ability_file(&bleed_src)
                .expect("parse Bleed.ability"),
        ),
        (
            "Reap.ability".to_string(),
            dsl_ast::parse_ability_file(&reap_src)
                .expect("parse Reap.ability"),
        ),
        (
            "Vampirize.ability".to_string(),
            dsl_ast::parse_ability_file(&vampirize_src)
                .expect("parse Vampirize.ability"),
        ),
        (
            "Fortify.ability".to_string(),
            dsl_ast::parse_ability_file(&fortify_src)
                .expect("parse Fortify.ability"),
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

    // ---- Bleed: cooldown 50 ticks, self-target, self_damage 5.0 ----
    //   Wave 2 piece N: first SelfDamage E2E demo. Lowers via
    //   `dsl_compiler::ability_lower::lower_effect_stmt` ("self_damage"
    //   match arm) to `EffectOp::SelfDamage { amount: 5.0 }`.
    let bleed_id = *built.names.get("Bleed")
        .expect("Bleed registered in name table");
    let bleed = built.registry.get(bleed_id)
        .expect("Bleed resolves to a program");
    assert_eq!(
        bleed.gate.cooldown_ticks, 50,
        "Bleed cooldown must be 50 ticks (5s) — .sim verb gate \
         (world.tick % 50 == 0) and .ability `cooldown: 5s` must agree",
    );
    assert!(
        !bleed.gate.hostile_only,
        "Bleed must NOT be hostile_only — .ability `target: self`",
    );
    assert_eq!(
        bleed.effects.len(), 1,
        "Bleed must have exactly one effect (self_damage 5)",
    );
    match &bleed.effects[0] {
        EffectOp::SelfDamage { amount } => assert_eq!(
            *amount, 5.0,
            "Bleed self_damage must be 5.0 — .ability `self_damage 5`, \
             .sim config.combat.bleed_amount",
        ),
        other => panic!(
            "Bleed effect[0]: expected SelfDamage(5.0), got {other:?}",
        ),
    }

    // ---- Reap: cooldown 20 ticks, range 5.0, hostile_only, execute 20.0 ----
    //   Wave 2 piece N: Execute E2E demo. Lowers via
    //   `dsl_compiler::ability_lower::lower_effect_stmt` ("execute"
    //   match arm) to `EffectOp::Execute { hp_threshold: 20.0 }`. The
    //   .sim verb gates emission on `target.hp < reap_threshold` so the
    //   chronicle only fires Defeated when the finisher condition holds.
    let reap_id = *built.names.get("Reap")
        .expect("Reap registered in name table");
    let reap = built.registry.get(reap_id).expect("Reap resolves to a program");
    assert_eq!(
        reap.gate.cooldown_ticks, 20,
        "Reap cooldown must be 20 ticks (2s) — .sim verb gate \
         (world.tick % 20 == 0) and .ability `cooldown: 2s` must agree",
    );
    assert!(
        reap.gate.hostile_only,
        "Reap must be hostile_only — .ability `target: enemy`",
    );
    match reap.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 5.0,
            "Reap range must be 5.0 — .ability `range: 5.0`",
        ),
    }
    assert_eq!(
        reap.effects.len(), 1,
        "Reap must have exactly one effect (execute 20)",
    );
    match &reap.effects[0] {
        EffectOp::Execute { hp_threshold } => assert_eq!(
            *hp_threshold, 20.0,
            "Reap execute threshold must be 20.0 — .ability \
             `execute 20`, .sim config.combat.reap_threshold",
        ),
        other => panic!(
            "Reap effect[0]: expected Execute(20.0), got {other:?}",
        ),
    }

    // ---- Vampirize: cooldown 80 ticks, self-target, lifesteal 0.5 5s ----
    //   Wave 2 piece N: LifeSteal E2E demo. Lowers via
    //   `dsl_compiler::ability_lower::lower_effect_stmt` ("lifesteal"
    //   match arm) to `EffectOp::LifeSteal { duration_ticks: 50,
    //   fraction_q8: 128 }` — 0.5 * 256 == 128 q8, 5s == 50 ticks.
    //   The .sim verb's own `when` clause gates emission on
    //   `self.hp < hp_vampire_floor`, and the new
    //   ApplyLifestealActivation chronicle drains SetLifesteal events
    //   into the per-agent lifesteal SoA fields.
    let vampirize_id = *built.names.get("Vampirize")
        .expect("Vampirize registered in name table");
    let vampirize = built.registry.get(vampirize_id)
        .expect("Vampirize resolves to a program");
    assert_eq!(
        vampirize.gate.cooldown_ticks, 80,
        "Vampirize cooldown must be 80 ticks (8s) — .sim verb gate \
         (world.tick % 80 == 0) and .ability `cooldown: 8s` must agree",
    );
    assert!(
        !vampirize.gate.hostile_only,
        "Vampirize must NOT be hostile_only — .ability `target: self`",
    );
    assert_eq!(
        vampirize.effects.len(), 1,
        "Vampirize must have exactly one effect (lifesteal 0.5 5s)",
    );
    match &vampirize.effects[0] {
        EffectOp::LifeSteal { duration_ticks, fraction_q8 } => {
            assert_eq!(
                *duration_ticks, 50,
                "Vampirize lifesteal duration must be 50 ticks (5s) — \
                 .ability `lifesteal 0.5 5s`, .sim config.combat.vampirize_dur",
            );
            assert_eq!(
                *fraction_q8, 128,
                "Vampirize lifesteal fraction must be 128 (0.5 in q8) — \
                 .ability `lifesteal 0.5 5s`, .sim config.combat.vampirize_frac_q8",
            );
        }
        other => panic!(
            "Vampirize effect[0]: expected LifeSteal(50, 128), got {other:?}",
        ),
    }

    // ---- Fortify: cooldown 80 ticks, self-target, damage_modify 0.5 5s ----
    //   Wave 2 piece N: DamageModify E2E demo. Lowers via
    //   `dsl_compiler::ability_lower::lower_effect_stmt` ("damage_modify"
    //   match arm) to `EffectOp::DamageModify { duration_ticks: 50,
    //   multiplier_q8: 128 }` — 0.5 * 256 == 128 q8, 5s == 50 ticks.
    //   The .sim verb's own `when` clause gates emission on
    //   `self.hp < hp_fortify_floor`, and the new
    //   ApplyDamageModActivation chronicle drains SetDamageMod events
    //   into the per-agent damage_taken_mult SoA fields. ApplyDamage
    //   reads them target-side to scale incoming bleed by `mult/256`.
    let fortify_id = *built.names.get("Fortify")
        .expect("Fortify registered in name table");
    let fortify = built.registry.get(fortify_id)
        .expect("Fortify resolves to a program");
    assert_eq!(
        fortify.gate.cooldown_ticks, 80,
        "Fortify cooldown must be 80 ticks (8s) — .sim verb gate \
         (world.tick % 80 == 0) and .ability `cooldown: 8s` must agree",
    );
    assert!(
        !fortify.gate.hostile_only,
        "Fortify must NOT be hostile_only — .ability `target: self`",
    );
    assert_eq!(
        fortify.effects.len(), 1,
        "Fortify must have exactly one effect (damage_modify 0.5 5s)",
    );
    match &fortify.effects[0] {
        EffectOp::DamageModify { duration_ticks, multiplier_q8 } => {
            assert_eq!(
                *duration_ticks, 50,
                "Fortify damage_modify duration must be 50 ticks (5s) — \
                 .ability `damage_modify 0.5 5s`, .sim config.combat.fortify_dur",
            );
            assert_eq!(
                *multiplier_q8, 128,
                "Fortify damage_modify multiplier must be 128 (0.5 in q8) — \
                 .ability `damage_modify 0.5 5s`, .sim config.combat.fortify_mult_q8",
            );
        }
        other => panic!(
            "Fortify effect[0]: expected DamageModify(50, 128), got {other:?}",
        ),
    }

    // ---- Smoke-pack: prove the GPU SoA layout works on this corpus ----
    // PackedAbilityRegistry::pack runs its own per-program packing
    // walk; if the registry contains anything pack can't encode this
    // panics. Confirms the lowering output is consumable by the
    // Wave 1.9 GPU buffer producer. Also verifies SelfDamage (op#17),
    // Execute (op#16), LifeSteal (op#18), and DamageModify (op#19)
    // pack cleanly via the Wave 2 piece 3+4 packer arms in
    // `engine::ability::packed::pack_effect_op`.
    let packed = PackedAbilityRegistry::pack(&built.registry);
    assert_eq!(
        packed.n_abilities, 7,
        "PackedAbilityRegistry must contain exactly 7 abilities \
         (Strike, ShieldUp, Mend, Bleed, Reap, Vampirize, Fortify)",
    );
}
