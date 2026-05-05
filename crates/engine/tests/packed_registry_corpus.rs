//! Wave 1.9 integration test — drive the Wave 1 `.ability` corpus end-to-end:
//! parse (`dsl_ast::parse_ability_file`) -> lower + registry build
//! (`dsl_compiler::ability_registry::build_registry`) -> SoA pack
//! (`engine::ability::PackedAbilityRegistry::pack`).
//!
//! The corpus lives at `assets/ability_test/duel_abilities/{Strike,
//! ShieldUp, Mend}.ability` and exercises the three header / effect
//! shapes Wave 1 ships:
//!   * `Strike`   -> enemy-target + range + cooldown + damage hint
//!   * `ShieldUp` -> self-target + cooldown + defense hint + shield effect
//!   * `Mend`     -> self-target + cooldown + heal hint + heal effect
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
    // Load the three corpus files; preserve the canonical input order so
    // the resulting `AbilityId` slots are deterministic.
    let strike = load("Strike");
    let shield_up = load("ShieldUp");
    let mend = load("Mend");

    let files = vec![strike, shield_up, mend];
    let built = build_registry(&files).expect("registry build must succeed");

    // Three abilities; name table covers all three.
    assert_eq!(built.registry.len(), 3);
    assert!(built.names.contains_key("Strike"));
    assert!(built.names.contains_key("ShieldUp"));
    assert!(built.names.contains_key("Mend"));

    let strike_slot = built.names["Strike"].slot();
    let shield_slot = built.names["ShieldUp"].slot();
    let mend_slot = built.names["Mend"].slot();

    let packed = PackedAbilityRegistry::pack(&built.registry);

    assert_eq!(packed.n_abilities, 3);

    // -- Strike: range 5.0, cooldown 1s == 10 ticks, hostile_only set. --
    assert_eq!(packed.cooldown_ticks[strike_slot], 10);
    assert_eq!(packed.range[strike_slot], 5.0);
    assert_eq!(packed.gate_flags[strike_slot], 0b01,
        "Strike has target: enemy -> hostile_only bit set");
    // Effect[0] is Damage(15.0); other slots are empty.
    let strike_eff_base = strike_slot * MAX_EFFECTS_PER_PROGRAM;
    assert_eq!(packed.effect_kinds[strike_eff_base], 0,
        "Strike effect 0 is Damage (discriminant 0)");
    assert_eq!(
        packed.effect_payload_a[strike_eff_base],
        15.0_f32.to_bits(),
        "Strike Damage amount packs to f32::to_bits(15.0)",
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

    // Delivery is Instant (=0) for all three — no other variant exists.
    for slot in [strike_slot, shield_slot, mend_slot] {
        assert_eq!(packed.delivery_kind[slot], 0);
    }
}
