//! Tests for the ability DSL parser and lowering.

use super::*;
use crate::ai::effects::defs::AbilityTargeting;
use crate::ai::effects::types::{Area, Delivery};

#[test]
fn parse_simple_ability() {
    let input = r#"
ability Fireball {
    target: enemy
    range: 5.0
    cooldown: 5s
    cast: 300ms
    hint: damage

    damage 55 [FIRE: 60]
}
"#;
    let (abilities, passives) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 1);
    assert_eq!(passives.len(), 0);

    let fb = &abilities[0];
    assert_eq!(fb.name, "Fireball");
    assert!(matches!(fb.targeting, AbilityTargeting::TargetEnemy));
    assert_eq!(fb.range, 5.0);
    assert_eq!(fb.cooldown_ms, 5000);
    assert_eq!(fb.cast_time_ms, 300);
    assert_eq!(fb.ai_hint, "damage");
    assert_eq!(fb.effects.len(), 1);
}

#[test]
fn parse_comma_separated_props() {
    let input = r#"
ability Test {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    damage 10
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities[0].range, 5.0);
    assert_eq!(abilities[0].cooldown_ms, 5000);
    assert_eq!(abilities[0].cast_time_ms, 300);
}

#[test]
fn parse_projectile_delivery() {
    let input = r#"
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit {
            damage 55 [FIRE: 60]
        }
        on_arrival {
            damage 15 in circle(2.0)
        }
    }
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let fb = &abilities[0];
    assert!(fb.delivery.is_some());

    if let Some(Delivery::Projectile { speed, width, on_hit, on_arrival, .. }) = &fb.delivery {
        assert_eq!(*speed, 8.0);
        assert_eq!(*width, 0.3);
        assert_eq!(on_hit.len(), 1);
        assert_eq!(on_arrival.len(), 1);
    } else {
        panic!("expected projectile delivery");
    }
}

#[test]
fn parse_chain_delivery() {
    let input = r#"
ability ArcaneMissiles {
    target: enemy, range: 5.0
    cooldown: 4s, cast: 200ms
    hint: damage

    deliver chain { bounces: 3, range: 3.0, falloff: 0.8 } {
        on_hit {
            damage 35 [MAGIC: 50]
        }
    }
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    if let Some(Delivery::Chain { bounces, bounce_range, falloff, on_hit }) = &abilities[0].delivery {
        assert_eq!(*bounces, 3);
        assert_eq!(*bounce_range, 3.0);
        assert_eq!(*falloff, 0.8);
        assert_eq!(on_hit.len(), 1);
    } else {
        panic!("expected chain delivery");
    }
}

#[test]
fn parse_zone_delivery() {
    let input = r#"
ability Blizzard {
    target: ground, range: 6.0
    cooldown: 12s, cast: 400ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            damage 15 in circle(3.0) [ICE: 50]
            slow 0.3 for 1.5s in circle(3.0)
        }
    }
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    if let Some(Delivery::Zone { duration_ms, tick_interval_ms }) = &abilities[0].delivery {
        assert_eq!(*duration_ms, 4000);
        assert_eq!(*tick_interval_ms, 1000);
    } else {
        panic!("expected zone delivery");
    }
}

#[test]
fn parse_aoe_effects() {
    let input = r#"
ability FrostNova {
    target: self_aoe
    cooldown: 10s, cast: 300ms
    hint: crowd_control

    damage 20 in circle(3.0)
    stun 2s in circle(3.0) [CROWD_CONTROL: 80, ICE: 60]
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let fn_ = &abilities[0];
    assert!(matches!(fn_.targeting, AbilityTargeting::SelfAoe));
    assert_eq!(fn_.effects.len(), 2);

    // Check first effect has area
    assert!(fn_.effects[0].area.is_some());
    if let Some(Area::Circle { radius }) = &fn_.effects[0].area {
        assert_eq!(*radius, 3.0);
    }

    // Check second effect has tags
    assert!(fn_.effects[1].tags.contains_key("CROWD_CONTROL"));
    assert_eq!(fn_.effects[1].tags["CROWD_CONTROL"], 80.0);
}

#[test]
fn parse_passive() {
    let input = r#"
passive ArcaneShield {
    trigger: on_hp_below(50%)
    cooldown: 30s

    shield 40 for 4s
}
"#;
    let (_, passives) = parse_abilities(input).unwrap();
    assert_eq!(passives.len(), 1);
    let p = &passives[0];
    assert_eq!(p.name, "ArcaneShield");
    assert_eq!(p.cooldown_ms, 30000);
    assert_eq!(p.effects.len(), 1);
}

#[test]
fn parse_passive_on_ability_used() {
    let input = r#"
passive ArcaneMastery {
    trigger: on_ability_used
    cooldown: 8s

    buff cooldown_reduction 0.15 for 3s
}
"#;
    let (_, passives) = parse_abilities(input).unwrap();
    assert_eq!(passives.len(), 1);
    let p = &passives[0];
    assert_eq!(p.name, "ArcaneMastery");
    assert_eq!(p.effects.len(), 1);
}

#[test]
fn parse_multiple_blocks() {
    let input = r#"
ability A {
    target: enemy, range: 2.0
    cooldown: 3s, cast: 0ms
    hint: damage
    damage 10
}

ability B {
    target: self
    cooldown: 5s, cast: 0ms
    hint: utility
    dash 3.0
}

passive C {
    trigger: on_kill
    cooldown: 5s
    heal 20
}
"#;
    let (abilities, passives) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 2);
    assert_eq!(passives.len(), 1);
}

#[test]
fn parse_comments() {
    let input = r#"
// This is a comment
# This is also a comment
ability Test {
    target: enemy // inline comment after properties not on their own line
    cooldown: 3s
    cast: 0ms
    hint: damage

    damage 10
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 1);
}

#[test]
fn parse_condition_when() {
    let input = r#"
ability VampStrike {
    target: enemy, range: 2.0
    cooldown: 8s, cast: 200ms
    hint: damage

    damage 40
    heal 20 when caster_hp_below(30%)
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities[0].effects.len(), 2);
    assert!(abilities[0].effects[1].condition.is_some());
}

#[test]
fn parse_shield_with_for_duration() {
    let input = r#"
ability ManaShield {
    target: self
    cooldown: 14s, cast: 200ms
    hint: defense

    shield 50 for 5s
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    // The effect should be a Shield with duration
    match &eff.effect {
        crate::ai::effects::Effect::Shield { amount, duration_ms } => {
            assert_eq!(*amount, 50);
            assert_eq!(*duration_ms, 5000);
        }
        other => panic!("expected Shield, got {other:?}"),
    }
}

#[test]
fn parse_full_mage_kit() {
    let input = include_str!("../../../../assets/hero_templates/mage.ability");
    let (abilities, passives) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 8, "mage should have 8 abilities");
    assert_eq!(passives.len(), 2, "mage should have 2 passives");

    // Verify names
    assert_eq!(abilities[0].name, "Fireball");
    assert_eq!(abilities[1].name, "FrostNova");
    assert_eq!(abilities[2].name, "ArcaneMissiles");
    assert_eq!(abilities[3].name, "Blizzard");
    assert_eq!(abilities[4].name, "Meteor");
    assert_eq!(abilities[5].name, "Blink");
    assert_eq!(abilities[6].name, "Polymorph");
    assert_eq!(abilities[7].name, "ManaShield");
    assert_eq!(passives[0].name, "ArcaneShield");
    assert_eq!(passives[1].name, "ArcaneMastery");
}

#[test]
fn roundtrip_mage_matches_toml() {
    // Parse both TOML and DSL versions, compare key fields
    let toml_str = include_str!("../../../../assets/hero_templates/mage.toml");
    let toml_hero: crate::ai::effects::defs::HeroToml =
        toml::from_str(toml_str).expect("TOML parse failed");

    let dsl_str = include_str!("../../../../assets/hero_templates/mage.ability");
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

// ---------------------------------------------------------------------------
// Multi-hero porting tests
// ---------------------------------------------------------------------------

/// Helper to compare key fields between TOML and DSL parsed abilities.
fn assert_roundtrip(toml_file: &str, dsl_file: &str, hero_name: &str) {
    let toml_hero: crate::ai::effects::defs::HeroToml =
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
fn roundtrip_warrior() {
    assert_roundtrip(
        include_str!("../../../../assets/hero_templates/warrior.toml"),
        include_str!("../../../../assets/hero_templates/warrior.ability"),
        "Warrior",
    );
}

#[test]
fn roundtrip_ranger() {
    assert_roundtrip(
        include_str!("../../../../assets/hero_templates/ranger.toml"),
        include_str!("../../../../assets/hero_templates/ranger.ability"),
        "Ranger",
    );
}

#[test]
fn roundtrip_necromancer() {
    assert_roundtrip(
        include_str!("../../../../assets/hero_templates/necromancer.toml"),
        include_str!("../../../../assets/hero_templates/necromancer.ability"),
        "Necromancer",
    );
}

#[test]
fn roundtrip_arcanist() {
    assert_roundtrip(
        include_str!("../../../../assets/hero_templates/arcanist.toml"),
        include_str!("../../../../assets/hero_templates/arcanist.ability"),
        "Arcanist",
    );
}

#[test]
fn parse_zone_tag_property() {
    let input = r#"
ability FireRing {
    target: ground, range: 6.0
    cooldown: 6s, cast: 300ms
    hint: damage
    zone_tag: "fire"

    damage 15 in circle(2.5) [FIRE: 50]
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities[0].zone_tag, Some("fire".to_string()));
}

#[test]
fn parse_trap_delivery() {
    let input = r#"
ability BearTrap {
    target: ground, range: 5.0
    cooldown: 10s, cast: 200ms
    hint: crowd_control

    deliver trap { duration: 15s, trigger_radius: 1.5, arm_time: 500ms } {
        on_hit {
            damage 25 [PHYSICAL: 40]
            root 2s [CROWD_CONTROL: 50]
        }
    }
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let trap = &abilities[0];
    match &trap.delivery {
        Some(crate::ai::effects::types::Delivery::Trap { duration_ms, trigger_radius, arm_time_ms }) => {
            assert_eq!(*duration_ms, 15000);
            assert_eq!(*trigger_radius, 1.5);
            assert_eq!(*arm_time_ms, 500);
        }
        other => panic!("expected Trap delivery, got {other:?}"),
    }
}

#[test]
fn parse_summon_with_count() {
    let input = r#"
ability Raise {
    target: self
    cooldown: 18s, cast: 500ms
    hint: utility

    summon "skeleton" x2
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    match &abilities[0].effects[0].effect {
        crate::ai::effects::Effect::Summon { template, count, .. } => {
            assert_eq!(template, "skeleton");
            assert_eq!(*count, 2);
        }
        other => panic!("expected Summon, got {other:?}"),
    }
}

#[test]
fn parse_spread_area() {
    let input = r#"
ability MultiShot {
    target: enemy, range: 6.0
    cooldown: 5s, cast: 200ms
    hint: damage

    damage 30 in spread(4.0, 3) [PHYSICAL: 40]
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    match &abilities[0].effects[0].area {
        Some(crate::ai::effects::types::Area::Spread { radius, max_targets }) => {
            assert_eq!(*radius, 4.0);
            assert_eq!(*max_targets, 3);
        }
        other => panic!("expected Spread area, got {other:?}"),
    }
}
