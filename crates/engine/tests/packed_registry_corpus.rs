//! Wave 1.9 integration test — drive the Wave 1 `.ability` corpus end-to-end:
//! parse (`dsl_ast::parse_ability_file`) -> lower + registry build
//! (`dsl_compiler::ability_registry::build_registry`) -> SoA pack
//! (`engine::ability::PackedAbilityRegistry::pack`).
//!
//! The corpus lives at `assets/ability_test/duel_abilities/{Strike,
//! ShieldUp, Mend, Bleed, Reap, Vampirize, Fortify, Daze}.ability` and
//! exercises the eight header / effect shapes Wave 1+2 ships:
//!   * `Strike`    -> enemy-target + range + cooldown + damage hint
//!   * `ShieldUp`  -> self-target + cooldown + defense hint + shield effect
//!   * `Mend`      -> self-target + cooldown + heal hint + heal effect
//!   * `Bleed`     -> self-target + cooldown + self_damage (Wave 2 piece N
//!                    SelfDamage E2E demo, EffectOp::SelfDamage discriminant 17)
//!   * `Reap`      -> enemy-target + range + cooldown + execute (Wave 2
//!                    piece N Execute E2E demo, EffectOp::Execute
//!                    discriminant 16)
//!   * `Vampirize` -> self-target + cooldown + lifesteal (Wave 2 piece N
//!                    LifeSteal E2E demo, EffectOp::LifeSteal discriminant 18)
//!   * `Fortify`   -> self-target + cooldown + damage_modify (Wave 2 piece N
//!                    DamageModify E2E demo, EffectOp::DamageModify
//!                    discriminant 19)
//!   * `Daze`      -> enemy-target + range + cooldown + crowd_control hint +
//!                    stun (Wave 2 piece N Stun E2E demo + FIRST verb-status
//!                    cast-gate, EffectOp::Stun discriminant 3)
//!
//! This test guards the boundary between the DSL pipeline and the GPU-
//! facing layout: any drift in slot ordering, payload encoding, or
//! gate-flag bit layout that the unit tests inside `packed.rs` somehow
//! miss should still surface here, because the corpus is a real input.

use std::fs;
use std::path::PathBuf;

use dsl_ast::parse_ability_file;
use dsl_compiler::ability_registry::build_registry;
use engine::ability::{
    EFFECT_KIND_EMPTY, MAX_EFFECTS_PER_PROGRAM, PackedAbilityRegistry,
};

fn corpus_root() -> PathBuf {
    // Walk from this test file's location up to the repo root, then into
    // the corpus dir. Avoids a `CARGO_MANIFEST_DIR`-relative hack that
    // breaks when the engine crate is moved.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("assets/ability_test/duel_abilities")
}

fn load(name: &str) -> (String, dsl_ast::AbilityFile) {
    let path = corpus_root().join(format!("{name}.ability"));
    let src = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let file = parse_ability_file(&src)
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    (name.to_string(), file)
}

#[test]
fn pack_duel_corpus_end_to_end() {
    // Load the eight corpus files; preserve the canonical input order so
    // the resulting `AbilityId` slots are deterministic.
    let strike = load("Strike");
    let shield_up = load("ShieldUp");
    let mend = load("Mend");
    let bleed = load("Bleed");
    let reap = load("Reap");
    let vampirize = load("Vampirize");
    let fortify = load("Fortify");
    let daze = load("Daze");

    let files = vec![strike, shield_up, mend, bleed, reap, vampirize, fortify, daze];
    let built = build_registry(&files).expect("registry build must succeed");

    // Eight abilities; name table covers all eight.
    assert_eq!(built.registry.len(), 8);
    assert!(built.names.contains_key("Strike"));
    assert!(built.names.contains_key("ShieldUp"));
    assert!(built.names.contains_key("Mend"));
    assert!(built.names.contains_key("Bleed"));
    assert!(built.names.contains_key("Reap"));
    assert!(built.names.contains_key("Vampirize"));
    assert!(built.names.contains_key("Fortify"));
    assert!(built.names.contains_key("Daze"));

    let strike_slot = built.names["Strike"].slot();
    let shield_slot = built.names["ShieldUp"].slot();
    let mend_slot = built.names["Mend"].slot();
    let bleed_slot = built.names["Bleed"].slot();
    let reap_slot = built.names["Reap"].slot();
    let vampirize_slot = built.names["Vampirize"].slot();
    let fortify_slot = built.names["Fortify"].slot();
    let daze_slot = built.names["Daze"].slot();

    let packed = PackedAbilityRegistry::pack(&built.registry);

    assert_eq!(packed.n_abilities, 8);

    // -- Strike: range 5.0, cooldown 1s == 10 ticks, hostile_only set. --
    // Damage value is 30.0 since the #85 retune (re-enabled shield_hp
    // absorption with rebalanced damage so combat still resolves).
    assert_eq!(packed.cooldown_ticks[strike_slot], 10);
    assert_eq!(packed.range[strike_slot], 5.0);
    assert_eq!(packed.gate_flags[strike_slot], 0b01,
        "Strike has target: enemy -> hostile_only bit set");
    // Effect[0] is Damage(30.0); other slots are empty.
    let strike_eff_base = strike_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[strike_eff_base], 0,
        "Strike effect 0 is Damage (discriminant 0)");
    assert_eq!(
        packed.effect_payload_a[strike_eff_base],
        30.0_f32.to_bits(),
        "Strike Damage amount packs to f32::to_bits(30.0)",
    );
    for i in 1..MAX_EFFECTS_PER_PROGRAM {
        assert_eq!(
            packed.effect_kinds[strike_eff_base + i],
            EFFECT_KIND_EMPTY,
            "Strike effect slot {i} must be empty sentinel",
        );
    }

    // -- ShieldUp: cooldown 4s == 40 ticks, self-target -> hostile_only=false. --
    assert_eq!(packed.cooldown_ticks[shield_slot], 40);
    assert_eq!(packed.range[shield_slot], 0.0,
        "ShieldUp has no range header -> Wave 1.6 default 0.0");
    assert_eq!(packed.gate_flags[shield_slot], 0,
        "ShieldUp is target: self -> no gate flag bits");
    let shield_eff_base = shield_slot * MAX_EFFECTS_PER_PROGRAM;
    // Shield discriminant == 2.
    assert_eq!(packed.effect_kinds[shield_eff_base], 2);
    assert_eq!(packed.effect_payload_a[shield_eff_base], 50.0_f32.to_bits());

    // -- Mend: cooldown 3s == 30 ticks; self-target heal. --
    assert_eq!(packed.cooldown_ticks[mend_slot], 30);
    let mend_eff_base = mend_slot * MAX_EFFECTS_PER_PROGRAM;
    // Heal discriminant == 1.
    assert_eq!(packed.effect_kinds[mend_eff_base], 1,
        "Mend effect 0 is Heal (discriminant 1)");
    assert_eq!(packed.effect_payload_a[mend_eff_base], 25.0_f32.to_bits());

    // -- Bleed: cooldown 5s == 50 ticks; self-target SelfDamage. --
    //   Wave 2 piece N first SelfDamage E2E demo. Discriminant 17 per
    //   `engine::ability::packed::pack_effect_op` (matches the schema
    //   hash entry `SelfDamage=17{amount=f32}`).
    assert_eq!(packed.cooldown_ticks[bleed_slot], 50);
    assert_eq!(packed.range[bleed_slot], 0.0,
        "Bleed has no range header -> Wave 1.6 default 0.0");
    assert_eq!(packed.gate_flags[bleed_slot], 0,
        "Bleed is target: self -> no gate flag bits");
    let bleed_eff_base = bleed_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[bleed_eff_base], 17,
        "Bleed effect 0 is SelfDamage (discriminant 17)");
    assert_eq!(packed.effect_payload_a[bleed_eff_base], 5.0_f32.to_bits());

    // -- Reap: cooldown 2s == 20 ticks, range 5.0; enemy-target Execute. --
    //   Wave 2 piece N Execute E2E demo. Discriminant 16 per
    //   `engine::ability::packed::pack_effect_op` (matches the schema
    //   hash entry `Execute=16{hp_threshold=f32}`).
    assert_eq!(packed.cooldown_ticks[reap_slot], 20);
    assert_eq!(packed.range[reap_slot], 5.0);
    assert_eq!(packed.gate_flags[reap_slot], 0b01,
        "Reap has target: enemy -> hostile_only bit set");
    let reap_eff_base = reap_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[reap_eff_base], 16,
        "Reap effect 0 is Execute (discriminant 16)");
    assert_eq!(packed.effect_payload_a[reap_eff_base], 20.0_f32.to_bits());

    // -- Vampirize: cooldown 8s == 80 ticks; self-target LifeSteal. --
    //   Wave 2 piece N LifeSteal E2E demo. Discriminant 18 per
    //   `engine::ability::packed::pack_effect_op` — same shape as Slow:
    //   payload_a = duration_ticks (50), payload_b = fraction_q8 cast to
    //   `i16 as u32` (128). The `Vampirize.ability` source is
    //   `lifesteal 0.5 5s`, so 0.5 * 256 == 128 q8 and 5s == 50 ticks.
    assert_eq!(packed.cooldown_ticks[vampirize_slot], 80);
    assert_eq!(packed.range[vampirize_slot], 0.0,
        "Vampirize has no range header -> Wave 1.6 default 0.0");
    assert_eq!(packed.gate_flags[vampirize_slot], 0,
        "Vampirize is target: self -> no gate flag bits");
    let vampirize_eff_base = vampirize_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[vampirize_eff_base], 18,
        "Vampirize effect 0 is LifeSteal (discriminant 18)");
    assert_eq!(packed.effect_payload_a[vampirize_eff_base], 50_u32,
        "Vampirize LifeSteal payload_a = duration_ticks (50)");
    assert_eq!(packed.effect_payload_b[vampirize_eff_base], 128_u32,
        "Vampirize LifeSteal payload_b = fraction_q8 (128 = 0.5 q8)");

    // -- Fortify: cooldown 8s == 80 ticks; self-target DamageModify. --
    //   Wave 2 piece N DamageModify E2E demo. Discriminant 19 per
    //   `engine::ability::packed::pack_effect_op` — same shape as
    //   LifeSteal/Slow: payload_a = duration_ticks (50), payload_b =
    //   multiplier_q8 cast to `i16 as u32` (128). The `Fortify.ability`
    //   source is `damage_modify 0.5 5s`, so 0.5 * 256 == 128 q8 and
    //   5s == 50 ticks.
    assert_eq!(packed.cooldown_ticks[fortify_slot], 80);
    assert_eq!(packed.range[fortify_slot], 0.0,
        "Fortify has no range header -> Wave 1.6 default 0.0");
    assert_eq!(packed.gate_flags[fortify_slot], 0,
        "Fortify is target: self -> no gate flag bits");
    let fortify_eff_base = fortify_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[fortify_eff_base], 19,
        "Fortify effect 0 is DamageModify (discriminant 19)");
    assert_eq!(packed.effect_payload_a[fortify_eff_base], 50_u32,
        "Fortify DamageModify payload_a = duration_ticks (50)");
    assert_eq!(packed.effect_payload_b[fortify_eff_base], 128_u32,
        "Fortify DamageModify payload_b = multiplier_q8 (128 = 0.5 q8)");

    // -- Daze: cooldown 4s == 40 ticks, range 5.0; enemy-target Stun. --
    //   Wave 2 piece N Stun E2E demo + FIRST verb-status cast-gate.
    //   Discriminant 3 per `engine::ability::packed::pack_effect_op` —
    //   payload_a = duration_ticks (10), payload_b = 0. The
    //   `Daze.ability` source is `stun 1s`, so 1s == 10 ticks. The
    //   .sim verb writes hot_stun_expires_at_tick = world.tick + 10;
    //   every offensive verb's `when` clause now reads that field via
    //   `agents.stun_expires_at_tick(self) <= world.tick`.
    assert_eq!(packed.cooldown_ticks[daze_slot], 40);
    assert_eq!(packed.range[daze_slot], 5.0);
    assert_eq!(packed.gate_flags[daze_slot], 0b01,
        "Daze has target: enemy -> hostile_only bit set");
    let daze_eff_base = daze_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[daze_eff_base], 3,
        "Daze effect 0 is Stun (discriminant 3)");
    assert_eq!(packed.effect_payload_a[daze_eff_base], 10_u32,
        "Daze Stun payload_a = duration_ticks (10)");

    // Delivery is Instant (=0) for all eight — no other variant exists.
    for slot in [strike_slot, shield_slot, mend_slot, bleed_slot, reap_slot, vampirize_slot, fortify_slot, daze_slot] {
        assert_eq!(packed.delivery_kind[slot], 0);
    }
}
