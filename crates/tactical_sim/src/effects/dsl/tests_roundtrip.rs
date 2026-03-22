//! Roundtrip comparison tests (TOML ↔ DSL).

use super::*;

// ---------------------------------------------------------------------------
// Multi-hero porting tests
// ---------------------------------------------------------------------------

/// Helper to compare key fields between TOML and DSL parsed abilities.
fn assert_roundtrip(toml_file: &str, dsl_file: &str, hero_name: &str) {
    let toml_hero: crate::effects::defs::HeroToml =
        toml::from_str(toml_file).unwrap_or_else(|e| panic!("{hero_name} TOML parse failed: {e}"));
    let (dsl_abilities, dsl_passives) =
        parse_abilities(dsl_file).unwrap_or_else(|e| panic!("{hero_name} DSL parse failed: {e}"));

    assert_eq!(toml_hero.abilities.len(), dsl_abilities.len(),
        "{hero_name}: ability count mismatch");
    assert_eq!(toml_hero.passives.len(), dsl_passives.len(),
        "{hero_name}: passive count mismatch");

    for (i, (t, d)) in toml_hero.abilities.iter().zip(&dsl_abilities).enumerate() {
        assert_eq!(t.name, d.name, "{hero_name} ability {i} name");
        assert_eq!(t.cooldown_ms, d.cooldown_ms, "{hero_name} {} cooldown", t.name);
        assert_eq!(t.cast_time_ms, d.cast_time_ms, "{hero_name} {} cast_time", t.name);
        assert_eq!(t.range, d.range, "{hero_name} {} range", t.name);
        assert_eq!(t.ai_hint, d.ai_hint, "{hero_name} {} hint", t.name);
        assert_eq!(t.delivery.is_some(), d.delivery.is_some(),
            "{hero_name} {} delivery presence", t.name);
    }

    for (i, (t, d)) in toml_hero.passives.iter().zip(&dsl_passives).enumerate() {
        assert_eq!(t.name, d.name, "{hero_name} passive {i} name");
        assert_eq!(t.cooldown_ms, d.cooldown_ms, "{hero_name} {} cooldown", t.name);
    }
}

#[test]
fn roundtrip_mage_matches_toml() {
    // Parse both TOML and DSL versions, compare key fields
    let toml_str = include_str!("../../../../../assets/hero_templates/mage.toml");
    let toml_hero: crate::effects::defs::HeroToml =
        toml::from_str(toml_str).expect("TOML parse failed");

    let dsl_str = include_str!("../../../../../assets/hero_templates/mage.ability");
    let (dsl_abilities, dsl_passives) = parse_abilities(dsl_str).unwrap();

    assert_eq!(toml_hero.abilities.len(), dsl_abilities.len(),
        "ability count mismatch");
    assert_eq!(toml_hero.passives.len(), dsl_passives.len(),
        "passive count mismatch");

    // Compare each ability's key properties
    for (i, (toml_ab, dsl_ab)) in toml_hero.abilities.iter().zip(&dsl_abilities).enumerate() {
        assert_eq!(toml_ab.name, dsl_ab.name, "ability {i} name mismatch");
        assert_eq!(toml_ab.cooldown_ms, dsl_ab.cooldown_ms,
            "ability {} ({}) cooldown mismatch", i, toml_ab.name);
        assert_eq!(toml_ab.cast_time_ms, dsl_ab.cast_time_ms,
            "ability {} ({}) cast_time mismatch", i, toml_ab.name);
        assert_eq!(toml_ab.range, dsl_ab.range,
            "ability {} ({}) range mismatch", i, toml_ab.name);
        assert_eq!(toml_ab.ai_hint, dsl_ab.ai_hint,
            "ability {} ({}) ai_hint mismatch", i, toml_ab.name);
        // Check delivery presence matches
        assert_eq!(toml_ab.delivery.is_some(), dsl_ab.delivery.is_some(),
            "ability {} ({}) delivery presence mismatch", i, toml_ab.name);
    }
}

#[test]
fn roundtrip_warrior() {
    assert_roundtrip(
        include_str!("../../../../../assets/hero_templates/warrior.toml"),
        include_str!("../../../../../assets/hero_templates/warrior.ability"),
        "Warrior",
    );
}

#[test]
fn roundtrip_ranger() {
    assert_roundtrip(
        include_str!("../../../../../assets/hero_templates/ranger.toml"),
        include_str!("../../../../../assets/hero_templates/ranger.ability"),
        "Ranger",
    );
}

#[test]
fn roundtrip_necromancer() {
    assert_roundtrip(
        include_str!("../../../../../assets/hero_templates/necromancer.toml"),
        include_str!("../../../../../assets/hero_templates/necromancer.ability"),
        "Necromancer",
    );
}

#[test]
fn roundtrip_arcanist() {
    assert_roundtrip(
        include_str!("../../../../../assets/hero_templates/arcanist.toml"),
        include_str!("../../../../../assets/hero_templates/arcanist.ability"),
        "Arcanist",
    );
}
