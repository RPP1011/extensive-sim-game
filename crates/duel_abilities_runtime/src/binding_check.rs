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
    program::{
        AbilityTag, Area, EffectAreaShape, EffectOp, EffectScaling, EffectWhenCondition,
        LifetimeMode, ScalingStatRef, ShapeKind, StackingMode,
    },
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
    let daze_src = std::fs::read_to_string(corpus.join("Daze.ability"))
        .expect("read Daze.ability");

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
        (
            "Daze.ability".to_string(),
            dsl_ast::parse_ability_file(&daze_src)
                .expect("parse Daze.ability"),
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
    // Wave 1.5#2 in-shape modifier — Strike.ability declares
    // `in circle(0.5)`, lowered to program.per_effect_areas[0] =
    // Some(EffectAreaShape { kind: Circle, args: [0.5, 0, 0, 0] }).
    // The 0.5 radius is small enough to be effectively single-target
    // in the duel_abilities 2-agent fixture (no second enemy can be
    // inside the disc), so the .sim's existing single-target verb
    // behavior is correct. Real AOE spatial dispatch — verb iterating
    // multiple candidates via spatial.within / for_each_neighbor —
    // lands in a multi-agent fixture later. The binding check pins
    // the lowered metadata so corpus drift surfaces immediately.
    assert_eq!(
        strike.per_effect_areas.len(), 1,
        "Strike must have one area slot (parallel to one effect)",
    );
    assert_eq!(
        strike.per_effect_areas[0],
        Some(EffectAreaShape { kind: ShapeKind::Circle, args: [0.5, 0.0, 0.0, 0.0] }),
        "Strike area must be Circle(0.5) — .ability `in circle(0.5)`; \
         args use the shape's positional vocabulary (radius for circle), \
         zero-padded to 4 slots",
    );

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
    // Wave 1.5#3 stacking modifier — ShieldUp.ability declares
    // `stacking stack`, lowered to program.stackings[0] =
    // Some(StackingMode::Stack). The .sim's ApplyShield does
    // `new_shield = agents.shield_hp(t) + a` (additive accumulation),
    // which IS stack semantics. The binding check proves the lowering
    // captures the modifier; registry-driven stacking dispatch is
    // later infrastructure.
    assert_eq!(
        shieldup.stackings.len(), 1,
        "ShieldUp must have one stacking slot (parallel to one effect)",
    );
    assert_eq!(
        shieldup.stackings[0], Some(StackingMode::Stack),
        "ShieldUp stacking must be Stack — .ability `stacking stack`; \
         .sim ApplyShield does additive accumulation",
    );
    // Wave 1.5#8 lifetime modifier — ShieldUp.ability declares
    // `damageable_hp(50)`, lowered to program.lifetimes[0] =
    // Some(LifetimeMode::DamageableHp(50.0)). The .sim's ApplyDamage
    // drains shield_hp by `absorbed = min(s, d_scaled); s = s - absorbed`
    // and the buff is "broken" when shield_hp settles at 0 — that IS
    // DamageableHp(N) semantics. The binding-check proves the lowering
    // captures the modifier; registry-driven lifetime dispatch is
    // later infrastructure.
    assert_eq!(
        shieldup.lifetimes.len(), 1,
        "ShieldUp must have one lifetime slot (parallel to one effect)",
    );
    assert_eq!(
        shieldup.lifetimes[0], Some(LifetimeMode::DamageableHp(50.0)),
        "ShieldUp lifetime must be DamageableHp(50.0) — .ability \
         `damageable_hp(50)` mirrors the 50 shield_hp absorption pool; \
         .sim ApplyDamage drains shield_hp wholesale",
    );

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
            "Bleed self_damage base must be 5.0 — .ability \
             `self_damage 5 + 5% max_hp`; the base lands in EffectOp, \
             the +5% lives in scalings_per_effect[0]",
        ),
        other => panic!(
            "Bleed effect[0]: expected SelfDamage(5.0), got {other:?}",
        ),
    }
    // Wave 1.5#4 scaling modifier — Bleed.ability declares
    // `+ 5% max_hp`, lowered to scalings_per_effect[0] = [
    // EffectScaling { stat_ref: MaxHp, percent: 0.05 }]. The .sim
    // verb hand-mirrors the scaled total via config.combat.bleed_amount
    // = 10 (fixture pins MaxHp = 100, so 5 + 0.05·100 = 10). Registry-
    // driven scaling dispatch (per-effect kernel reads the scalings and
    // multiplies by the agent's stat field) is later infrastructure;
    // this binding check proves the lowering captures the modifier.
    assert_eq!(
        bleed.scalings_per_effect.len(), 1,
        "Bleed must have one scaling slot (parallel to one effect)",
    );
    assert_eq!(
        bleed.scalings_per_effect[0].len(), 1,
        "Bleed effect[0] must have exactly one scaling entry — \
         .ability declares `+ 5% max_hp` once",
    );
    assert_eq!(
        bleed.scalings_per_effect[0][0],
        EffectScaling { stat_ref: ScalingStatRef::MaxHp, percent: 0.05 },
        "Bleed scaling must be (MaxHp, 0.05) — .ability \
         `+ 5% max_hp`; lowering converts N% → N/100 for direct \
         multiplication by the agent's stat field",
    );

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
    // Wave 1.5#2 in-shape — Reap declares `in cone(45deg, 5.0)`,
    // lowered to per_effect_areas[0] = Some(EffectAreaShape {
    // kind: Cone, args: [45.0, 5.0, 0, 0] }). Cone is the canonical
    // 2-arg shape (angle_deg, radius); arg ordering matches the spec
    // §8.1 constructor signature. Like Strike's circle, this is
    // metadata-only in the 2-agent fixture — the .sim's single-target
    // verb behavior is unchanged. Adding shape variety to the corpus
    // so packed-args ordering surfaces if shape parsing ever drifts.
    assert_eq!(
        reap.per_effect_areas.len(), 1,
        "Reap must have one area slot (parallel to one effect)",
    );
    assert_eq!(
        reap.per_effect_areas[0],
        Some(EffectAreaShape { kind: ShapeKind::Cone, args: [45.0, 5.0, 0.0, 0.0] }),
        "Reap area must be Cone(45deg, 5.0) — .ability \
         `in cone(45deg, 5.0)`; spec §8.1 cone(angle_deg, radius)",
    );
    // Wave 1.5#7 when-condition modifier — Reap.ability declares
    // `when target.hp < 20`, lowered to program.when_per_effect[0] =
    // Some(EffectWhenCondition { when_cond: "target.hp < 20",
    // else_cond: None }). Predicate body is captured as verbatim
    // source text; engine code does not parse it. The .sim verb
    // already gates emit on the same predicate via its own `when`
    // clause (`target.hp < config.combat.reap_threshold`), so this
    // is the canonical executor pattern: the .ability declares
    // intent, the .sim hand-mirrors the gate. Apply handlers wire
    // registry-driven predicate evaluation later.
    assert_eq!(
        reap.when_per_effect.len(), 1,
        "Reap must have one when slot (parallel to one effect)",
    );
    assert_eq!(
        reap.when_per_effect[0],
        Some(EffectWhenCondition {
            when_cond: "target.hp < 20".to_string(),
            else_cond: None,
        }),
        "Reap when must be `target.hp < 20` — .ability \
         `when target.hp < 20`; .sim Reap verb gates on \
         `target.hp < config.combat.reap_threshold`",
    );
    // Wave 1.5#9 nested-effect modifier — Reap.ability declares
    // `{ stun 1s }`, lowered to program.nested_per_effect[0] =
    // [EffectOp::Stun{duration_ticks: 10}]. Lowering recursively
    // calls lower_effect_stmt on each inner stmt; inner-stmt
    // modifiers (e.g. `chance 50%`) are silently dropped today —
    // recursive aggregator capture is later infrastructure. The
    // .sim verb's existing emit fires Defeated only (no Stunned)
    // — apply-handler dispatch for the nested op is deferred. The
    // binding check pins the lowered IR so corpus drift surfaces.
    assert_eq!(
        reap.nested_per_effect.len(), 1,
        "Reap must have one nested slot (parallel to one effect)",
    );
    assert_eq!(
        reap.nested_per_effect[0].len(), 1,
        "Reap effect[0] must have exactly one nested op — \
         .ability declares `{{ stun 1s }}` once",
    );
    match &reap.nested_per_effect[0][0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 10,
            "Reap nested stun must be 10 ticks (1s) — .ability \
             `{{ stun 1s }}`",
        ),
        other => panic!(
            "Reap nested[0]: expected Stun(10), got {other:?}",
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
    // Wave 1.5#3 stacking modifier — Fortify.ability declares
    // `stacking refresh`, lowered to program.stackings[0] =
    // Some(StackingMode::Refresh). The .sim's ApplyDamageModActivation
    // overwrites damage_taken_mult_expires_at_tick wholesale (no
    // max-with-existing), which IS refresh semantics. The binding
    // check proves the lowering captures the modifier; registry-
    // driven stacking dispatch is later infrastructure.
    assert_eq!(
        fortify.stackings.len(), 1,
        "Fortify must have one stacking slot (parallel to one effect)",
    );
    assert_eq!(
        fortify.stackings[0], Some(StackingMode::Refresh),
        "Fortify stacking must be Refresh — .ability `stacking refresh`; \
         .sim ApplyDamageModActivation overwrites expires_at wholesale",
    );

    // ---- Daze: cooldown 40 ticks, range 5.0, hostile_only, stun 10 ticks ----
    //   Wave 2 piece N: Stun E2E demo + FIRST verb-status cast-gate.
    //   Lowers via `dsl_compiler::ability_lower::lower_effect_stmt`
    //   ("stun" match arm) to `EffectOp::Stun { duration_ticks: 10 }`
    //   — `1s` == 10 ticks at the fixed 100ms tick. The .sim verb's own
    //   `when` clause gates emission on `target.hp > hp_daze_floor`
    //   (don't waste a stun on a finisher candidate), and the new
    //   ApplyStun chronicle drains Stunned events into the per-agent
    //   hot_stun_expires_at_tick SoA field. Every offensive verb's
    //   `when` clause (Strike/ShieldUp/Mend/Bleed/Reap/Vampirize/
    //   Fortify/Daze) reads that field via
    //   `agents.stun_expires_at_tick(self) <= world.tick` and skips
    //   casting while the window is active.
    let daze_id = *built.names.get("Daze")
        .expect("Daze registered in name table");
    let daze = built.registry.get(daze_id)
        .expect("Daze resolves to a program");
    assert_eq!(
        daze.gate.cooldown_ticks, 40,
        "Daze cooldown must be 40 ticks (4s) — .sim verb gate \
         (world.tick % 40 == 0) and .ability `cooldown: 4s` must agree",
    );
    assert!(
        daze.gate.hostile_only,
        "Daze must be hostile_only — .ability `target: enemy`",
    );
    match daze.area {
        Area::SingleTarget { range } => assert_eq!(
            range, 5.0,
            "Daze range must be 5.0 — .ability `range: 5.0`",
        ),
    }
    assert_eq!(
        daze.effects.len(), 1,
        "Daze must have exactly one effect (stun 1s)",
    );
    match &daze.effects[0] {
        EffectOp::Stun { duration_ticks } => assert_eq!(
            *duration_ticks, 10,
            "Daze stun duration must be 10 ticks (1s) — .ability \
             `stun 1s`, .sim config.combat.daze_dur",
        ),
        other => panic!(
            "Daze effect[0]: expected Stun(10), got {other:?}",
        ),
    }
    // Wave 1.5#5 chance modifier — Daze.ability declares `chance 50%`,
    // lowered to program.chances[0] = Some(32768) (q16 fixed-point;
    // (0.5 * 65535).round() = 32768 because 32767.5 rounds half-up).
    // The .sim verb's `when` clause hand-mirrors this as
    // `(rng.action() % 65536) < 32768`, matching the runtime gate's
    // `(per_agent_u32(...) & 0xFFFF) < q16` shape. Registry-driven
    // chance dispatch is later infrastructure; for now the binding
    // check proves the lowering produces the expected q16 value.
    assert_eq!(
        daze.chances.len(), 1,
        "Daze must have one chance entry (parallel to one effect)",
    );
    assert_eq!(
        daze.chances[0], Some(32768),
        "Daze chance must be Some(32768) — .ability `chance 50%`, \
         q16 = (0.5 * 65535).round() = 32768, clamped below the \
         CHANCE_NONE_SENTINEL=0xFFFF reservation",
    );

    // ---- Wave 1.5#1 tag-vocabulary E2E ----
    //   Each .ability declares a single primary tag (Vampirize is the
    //   one multi-tag entry: UTILITY + HEAL). Tags lower to
    //   program.tags via the per-effect aggregator in
    //   `dsl_compiler::ability_lower` — same-tag values across
    //   multiple effects sum into one (tag, weight) pair. The .sim
    //   doesn't read tags today; they're metadata for AI/UI filtering
    //   and registry-driven dispatch (later). The binding check pins
    //   the lowered values so corpus drift surfaces immediately.
    let tag_value = |program: &engine::ability::AbilityProgram, t: AbilityTag| -> f32 {
        program.tags.iter().find(|(tt, _)| *tt == t).map(|(_, v)| *v).unwrap_or(0.0)
    };
    assert_eq!(
        tag_value(strike, AbilityTag::Physical), 100.0,
        "Strike must be tagged [PHYSICAL: 100]",
    );
    assert_eq!(
        tag_value(mend, AbilityTag::Heal), 100.0,
        "Mend must be tagged [HEAL: 100]",
    );
    assert_eq!(
        tag_value(reap, AbilityTag::Physical), 100.0,
        "Reap must be tagged [PHYSICAL: 100]",
    );
    assert_eq!(
        tag_value(bleed, AbilityTag::Physical), 50.0,
        "Bleed must be tagged [PHYSICAL: 50] (self-cost half-weight)",
    );
    assert_eq!(
        tag_value(shieldup, AbilityTag::Defense), 100.0,
        "ShieldUp must be tagged [DEFENSE: 100]",
    );
    assert_eq!(
        tag_value(fortify, AbilityTag::Defense), 100.0,
        "Fortify must be tagged [DEFENSE: 100]",
    );
    assert_eq!(
        tag_value(daze, AbilityTag::CrowdControl), 100.0,
        "Daze must be tagged [CROWD_CONTROL: 100]",
    );
    // Vampirize is the multi-tag canary — the lowering must keep both
    // entries distinct (no accidental same-tag aggregation).
    assert_eq!(
        tag_value(vampirize, AbilityTag::Utility), 50.0,
        "Vampirize must be tagged [UTILITY: 50]",
    );
    assert_eq!(
        tag_value(vampirize, AbilityTag::Heal), 50.0,
        "Vampirize must also be tagged [HEAL: 50] — multi-tag canary",
    );
    assert_eq!(
        vampirize.tags.len(), 2,
        "Vampirize must have exactly two distinct tags (UTILITY, HEAL)",
    );

    // ---- Smoke-pack: prove the GPU SoA layout works on this corpus ----
    // PackedAbilityRegistry::pack runs its own per-program packing
    // walk; if the registry contains anything pack can't encode this
    // panics. Confirms the lowering output is consumable by the
    // Wave 1.9 GPU buffer producer. Also verifies SelfDamage (op#17),
    // Execute (op#16), LifeSteal (op#18), DamageModify (op#19), and
    // Stun (op#3) pack cleanly via the Wave 2 piece 3+4 packer arms in
    // `engine::ability::packed::pack_effect_op`.
    let packed = PackedAbilityRegistry::pack(&built.registry);
    assert_eq!(
        packed.n_abilities, 8,
        "PackedAbilityRegistry must contain exactly 8 abilities \
         (Strike, ShieldUp, Mend, Bleed, Reap, Vampirize, Fortify, Daze)",
    );
}
