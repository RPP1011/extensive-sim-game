//! Wave 1.9 integration test — drive the Wave 1 `.ability` corpus end-to-end:
//! parse (`dsl_ast::parse_ability_file`) -> lower + registry build
//! (`dsl_compiler::ability_registry::build_registry`) -> SoA pack
//! (`engine::ability::PackedAbilityRegistry::pack`).
//!
//! The corpus lives at `assets/ability_test/duel_abilities/{Strike,
//! ShieldUp, Mend, Bleed, Reap}.ability` and exercises the five header /
//! effect shapes Wave 1+2 ships:
//!   * `Strike`   -> enemy-target + range + cooldown + damage hint
//!   * `ShieldUp` -> self-target + cooldown + defense hint + shield effect
//!   * `Mend`     -> self-target + cooldown + heal hint + heal effect
//!   * `Bleed`    -> self-target + cooldown + self_damage (Wave 2 piece N
//!                   SelfDamage E2E demo, EffectOp::SelfDamage discriminant 17)
//!   * `Reap`     -> enemy-target + range + cooldown + execute (Wave 2
//!                   piece N Execute E2E demo, EffectOp::Execute
//!                   discriminant 16)
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
    // Load the five corpus files; preserve the canonical input order so
    // the resulting `AbilityId` slots are deterministic.
    let strike = load("Strike");
    let shield_up = load("ShieldUp");
    let mend = load("Mend");
    let bleed = load("Bleed");
    let reap = load("Reap");

    let files = vec![strike, shield_up, mend, bleed, reap];
    let built = build_registry(&files).expect("registry build must succeed");

    // Five abilities; name table covers all five.
    assert_eq!(built.registry.len(), 5);
    assert!(built.names.contains_key("Strike"));
    assert!(built.names.contains_key("ShieldUp"));
    assert!(built.names.contains_key("Mend"));
    assert!(built.names.contains_key("Bleed"));
    assert!(built.names.contains_key("Reap"));

    let strike_slot = built.names["Strike"].slot();
    let shield_slot = built.names["ShieldUp"].slot();
    let mend_slot = built.names["Mend"].slot();
    let bleed_slot = built.names["Bleed"].slot();
    let reap_slot = built.names["Reap"].slot();

    let packed = PackedAbilityRegistry::pack(&built.registry);

    assert_eq!(packed.n_abilities, 5);

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

    // Delivery is Instant (=0) for all five — no other variant exists.
    for slot in [strike_slot, shield_slot, mend_slot, bleed_slot, reap_slot] {
        assert_eq!(packed.delivery_kind[slot], 0);
    }
}
