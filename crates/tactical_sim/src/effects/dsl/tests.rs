//! Tests for the ability DSL parser and lowering.

use super::*;
use crate::effects::defs::AbilityTargeting;
use crate::effects::types::{Area, Delivery, TargetFilter};

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
        crate::effects::Effect::Shield { amount, duration_ms } => {
            assert_eq!(*amount, 50);
            assert_eq!(*duration_ms, 5000);
        }
        other => panic!("expected Shield, got {other:?}"),
    }
}

#[test]
fn parse_full_mage_kit() {
    let input = include_str!("../../../../../dataset/abilities/hero_templates/mage.ability");
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
        Some(crate::effects::types::Delivery::Trap { duration_ms, trigger_radius, arm_time_ms }) => {
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
        crate::effects::Effect::Summon { template, count, .. } => {
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
        Some(crate::effects::types::Area::Spread { radius, max_targets }) => {
            assert_eq!(*radius, 4.0);
            assert_eq!(*max_targets, 3);
        }
        other => panic!("expected Spread area, got {other:?}"),
    }
}

#[test]
fn parse_all_lol_heroes() {
    let lol_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../dataset/abilities/lol_heroes");
    let mut count = 0;
    let mut errors = Vec::new();

    for entry in std::fs::read_dir(&lol_dir).expect("cannot read lol_heroes dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "ability") {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap();
        match parse_abilities(&content) {
            Ok(_) => count += 1,
            Err(e) => errors.push(format!("{}: {e}", path.file_name().unwrap().to_string_lossy())),
        }
    }

    if !errors.is_empty() {
        panic!("{} / {} LoL hero .ability files failed to parse:\n{}",
            errors.len(), count + errors.len(), errors.join("\n"));
    }
    assert!(count > 0, "no .ability files found in lol_heroes");
    eprintln!("Successfully parsed {count} LoL hero .ability files");
}

/// Roundtrip test for programmatically generated AbilityDefs covering all
/// delivery types, effect types, and area types that ability_gen.rs can produce.
#[test]
fn roundtrip_generated_abilities() {
    use crate::effects::defs::AbilityDef;
    use crate::effects::effect_enum::Effect;
    use crate::effects::types::*;
    use crate::effects::dsl::emit::emit_ability_dsl;

    fn xrng(rng: &mut u64) -> u64 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    }
    fn rf(rng: &mut u64) -> f32 {
        (xrng(rng) as f32) / (u64::MAX as f32)
    }

    // All effect types the generator can produce
    let make_effects: Vec<Box<dyn Fn(&mut u64) -> Effect>> = vec![
        Box::new(|rng| Effect::Damage {
            amount: (10.0 + rf(rng) * 40.0) as i32,
            amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0,
            scaling_stat: None, scaling_percent: 0.0,
            damage_type: DamageType::Physical, bonus: vec![],
        }),
        Box::new(|rng| Effect::Heal {
            amount: (8.0 + rf(rng) * 30.0) as i32,
            amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0,
            scaling_stat: None, scaling_percent: 0.0, bonus: vec![],
        }),
        Box::new(|rng| Effect::Shield {
            amount: (15.0 + rf(rng) * 25.0) as i32,
            duration_ms: 3000 + (rf(rng) * 5000.0) as u32,
        }),
        Box::new(|rng| Effect::Stun { duration_ms: 500 + (rf(rng) * 2000.0) as u32 }),
        Box::new(|rng| Effect::Root { duration_ms: 500 + (rf(rng) * 2000.0) as u32 }),
        Box::new(|rng| Effect::Silence { duration_ms: 500 + (rf(rng) * 2000.0) as u32 }),
        Box::new(|rng| Effect::Slow {
            factor: ((0.2 + rf(rng) * 0.5) * 20.0).round() / 20.0,
            duration_ms: ((1000 + (rf(rng) * 3000.0) as u32) / 250) * 250,
        }),
        Box::new(|rng| Effect::Knockback {
            distance: ((1.0 + rf(rng) * 3.0) * 2.0).round() / 2.0,
        }),
        Box::new(|rng| Effect::Dash {
            to_target: rf(rng) < 0.5,
            distance: ((3.0 + rf(rng) * 4.0) * 2.0).round() / 2.0,
            to_position: false,
            is_blink: rf(rng) < 0.2,
        }),
        Box::new(|rng| Effect::Buff {
            stat: "attack_damage".into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: ((3000 + (rf(rng) * 7000.0) as u32) / 500) * 500,
        }),
        Box::new(|rng| Effect::Debuff {
            stat: "defense".into(),
            factor: ((0.1 + rf(rng) * 0.25) * 20.0).round() / 20.0,
            duration_ms: ((2000 + (rf(rng) * 5000.0) as u32) / 500) * 500,
        }),
        Box::new(|rng| Effect::Summon {
            template: "minion".into(),
            count: 1 + (rf(rng) * 2.0) as u32,
            hp_percent: 0.3 + rf(rng) * 0.5,
            clone: false, clone_damage_percent: 0.0, directed: false,
        }),
        Box::new(|rng| Effect::Stealth {
            duration_ms: 2000 + (rf(rng) * 4000.0) as u32,
            break_on_damage: rf(rng) < 0.6,
            break_on_ability: rf(rng) < 0.3,
        }),
        Box::new(|rng| Effect::Lifesteal {
            percent: 0.1 + rf(rng) * 0.3,
            duration_ms: 3000 + (rf(rng) * 5000.0) as u32,
        }),
        Box::new(|rng| Effect::Execute {
            hp_threshold_percent: 0.15 + rf(rng) * 0.25,
        }),
        Box::new(|rng| Effect::Resurrect {
            hp_percent: 0.3 + rf(rng) * 0.4,
        }),
        // Campaign buffs
        Box::new(|rng| Effect::Buff {
            stat: "travel_speed".into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: 0,
        }),
    ];

    // All delivery types
    let deliveries: Vec<Option<Delivery>> = vec![
        None, // instant
        Some(Delivery::Projectile {
            speed: 10.0, pierce: false, width: 0.75,
            on_hit: vec![], on_arrival: vec![],
        }),
        Some(Delivery::Projectile {
            speed: 8.0, pierce: true, width: 0.5,
            on_hit: vec![], on_arrival: vec![],
        }),
        Some(Delivery::Channel { duration_ms: 2000, tick_interval_ms: 500 }),
        Some(Delivery::Zone { duration_ms: 5000, tick_interval_ms: 500 }),
        Some(Delivery::Tether { max_range: 6.0, tick_interval_ms: 500, on_complete: vec![] }),
        Some(Delivery::Trap { duration_ms: 15000, trigger_radius: 2.0, arm_time_ms: 500 }),
        Some(Delivery::Chain { bounces: 3, bounce_range: 5.0, falloff: 0.8, on_hit: vec![] }),
    ];

    // Areas
    let areas: Vec<Option<Area>> = vec![
        None,
        Some(Area::SingleTarget),
        Some(Area::Circle { radius: 3.0 }),
        Some(Area::Cone { radius: 4.5, angle_deg: 60.0 }),
        Some(Area::Line { length: 6.0, width: 1.0 }),
        Some(Area::Spread { radius: 4.0, max_targets: 3 }),
    ];

    let mut total = 0;
    let mut failures = Vec::new();
    let mut rng = 12345u64;

    for (ei, make_eff) in make_effects.iter().enumerate() {
        for (di, delivery) in deliveries.iter().enumerate() {
            for (ai, area) in areas.iter().enumerate() {
                let effect = make_eff(&mut rng);
                let ce = ConditionalEffect {
                    effect,
                    condition: None,
                    area: area.clone(),
                    tags: Tags::new(),
                    stacking: Stacking::Refresh,
                    chance: 1.0,
                    else_effects: vec![],
                    targeting_filter: None,
                };

                let def = AbilityDef {
                    name: format!("Test_e{ei}_d{di}_a{ai}"),
                    targeting: AbilityTargeting::TargetEnemy,
                    range: 5.0,
                    cooldown_ms: 8000,
                    cast_time_ms: 300,
                    ai_hint: "damage".into(),
                    effects: vec![ce],
                    delivery: delivery.clone(),
                    ..Default::default()
                };

                let dsl = emit_ability_dsl(&def);
                total += 1;

                match parse_abilities(&dsl) {
                    Ok((parsed, _)) => {
                        if parsed.len() != 1 {
                            failures.push(format!(
                                "e{}:d{}:a{} — got {} abilities:\n{}",
                                ei, di, ai, parsed.len(), dsl
                            ));
                        }
                    }
                    Err(e) => {
                        failures.push(format!(
                            "e{}:d{}:a{} — {}\nDSL:\n{}",
                            ei, di, ai, e, dsl
                        ));
                    }
                }
            }
        }
    }

    // Also test with unrounded float values and edge cases
    let edge_case_effects: Vec<ConditionalEffect> = vec![
        // Blink with to_target
        ConditionalEffect {
            effect: Effect::Dash { to_target: true, distance: 7.0, to_position: false, is_blink: true },
            condition: None, area: None, tags: Tags::new(),
            stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
        },
        // Lifesteal with imprecise float
        ConditionalEffect {
            effect: Effect::Lifesteal { percent: 0.27341, duration_ms: 4523 },
            condition: None, area: None, tags: Tags::new(),
            stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
        },
        // Execute with fractional threshold
        ConditionalEffect {
            effect: Effect::Execute { hp_threshold_percent: 0.35 },
            condition: None, area: None, tags: Tags::new(),
            stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
        },
        // Stealth with break flags
        ConditionalEffect {
            effect: Effect::Stealth { duration_ms: 3000, break_on_damage: true, break_on_ability: true },
            condition: None, area: None, tags: Tags::new(),
            stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
        },
        // Resurrect
        ConditionalEffect {
            effect: Effect::Resurrect { hp_percent: 0.55 },
            condition: None, area: None, tags: Tags::new(),
            stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
        },
    ];

    // Test edge cases with all delivery types
    for (ei, ce) in edge_case_effects.iter().enumerate() {
        for (di, delivery) in deliveries.iter().enumerate() {
            let def = AbilityDef {
                name: format!("Edge_e{ei}_d{di}"),
                targeting: AbilityTargeting::TargetEnemy,
                range: 5.0,
                cooldown_ms: 8000,
                cast_time_ms: 300,
                ai_hint: "damage".into(),
                effects: vec![ce.clone()],
                delivery: delivery.clone(),
                ..Default::default()
            };

            let dsl = emit_ability_dsl(&def);
            total += 1;

            match parse_abilities(&dsl) {
                Ok((parsed, _)) => {
                    if parsed.len() != 1 {
                        failures.push(format!(
                            "edge:e{}:d{} — got {} abilities:\n{}",
                            ei, di, parsed.len(), dsl
                        ));
                    }
                }
                Err(e) => {
                    failures.push(format!(
                        "edge:e{}:d{} — {}\nDSL:\n{}",
                        ei, di, e, dsl
                    ));
                }
            }
        }
    }

    // Test with unrounded delivery parameters (like ability_gen produces)
    let messy_deliveries: Vec<Option<Delivery>> = vec![
        Some(Delivery::Tether { max_range: 5.7942, tick_interval_ms: 500, on_complete: vec![] }),
        Some(Delivery::Trap { duration_ms: 17965, trigger_radius: 1.7942, arm_time_ms: 1357 }),
        Some(Delivery::Chain { bounces: 4, bounce_range: 4.698, falloff: 0.754, on_hit: vec![] }),
        Some(Delivery::Channel { duration_ms: 2743, tick_interval_ms: 387 }),
        Some(Delivery::Zone { duration_ms: 5234, tick_interval_ms: 723 }),
    ];

    for (di, delivery) in messy_deliveries.iter().enumerate() {
        let def = AbilityDef {
            name: format!("Messy_d{di}"),
            targeting: AbilityTargeting::TargetEnemy,
            range: 5.0,
            cooldown_ms: 8000,
            cast_time_ms: 300,
            ai_hint: "damage".into(),
            effects: vec![ConditionalEffect {
                effect: Effect::Damage {
                    amount: 25, amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0,
                    scaling_stat: None, scaling_percent: 0.0,
                    damage_type: DamageType::Physical, bonus: vec![],
                },
                condition: None, area: None, tags: Tags::new(),
                stacking: Stacking::Refresh, chance: 1.0, else_effects: vec![], targeting_filter: None,
            }],
            delivery: delivery.clone(),
            ..Default::default()
        };

        let dsl = emit_ability_dsl(&def);
        total += 1;

        match parse_abilities(&dsl) {
            Ok((parsed, _)) => {
                if parsed.len() != 1 {
                    failures.push(format!(
                        "messy:d{} — got {} abilities:\n{}",
                        di, parsed.len(), dsl
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "messy:d{} — {}\nDSL:\n{}",
                    di, e, dsl
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n=== ROUNDTRIP FAILURES: {}/{} ===\n{}",
            failures.len(), total,
            failures.join("\n---\n")
        );
    }
    eprintln!("All {total} generated abilities roundtripped successfully");
}

// ---------------------------------------------------------------------------
// Aura (while_alive) and scales_with tests
// ---------------------------------------------------------------------------

#[test]
fn parse_while_alive_aura() {
    let input = r#"
ability AuraOfTheBrave {
    target: self_aoe
    cooldown: 0s
    hint: defense

    immunity "fear" in circle(10.0) while_alive
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 1);
    let a = &abilities[0];
    assert_eq!(a.name, "AuraOfTheBrave");
    assert_eq!(a.effects.len(), 1);
    let eff = &a.effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Immunity { duration_ms, immune_to } => {
            assert_eq!(*duration_ms, u32::MAX, "while_alive should set duration to u32::MAX");
            assert_eq!(immune_to, &["fear"]);
        }
        other => panic!("expected Immunity, got: {other:?}"),
    }
    assert!(matches!(eff.area.as_ref().unwrap(), Area::Circle { radius } if (*radius - 10.0).abs() < 0.01));
}

#[test]
fn parse_while_alive_buff() {
    let input = r#"
ability InspiringAura {
    target: self_aoe
    cooldown: 0s
    hint: utility

    buff damage_output 0.1 in circle(8.0) while_alive
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Buff { stat, factor, duration_ms } => {
            assert_eq!(stat, "damage_output");
            assert!((*factor - 0.1).abs() < 0.01);
            assert_eq!(*duration_ms, u32::MAX);
        }
        other => panic!("expected Buff, got: {other:?}"),
    }
}

#[test]
fn parse_scales_with() {
    let input = r#"
ability KingsBlessing {
    target: self_aoe
    cooldown: 0s
    hint: utility

    buff armor 2 for 10s in circle(8.0) scales_with party_size
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Buff { stat, factor, duration_ms } => {
            assert_eq!(stat, "armor");
            assert!((*factor - 2.0).abs() < 0.01);
            assert_eq!(*duration_ms, 10000);
        }
        other => panic!("expected Buff, got: {other:?}"),
    }
    // Scaling won't be on Buff (only on Damage/Heal), but it should parse without error
}

#[test]
fn parse_scales_with_on_damage() {
    let input = r#"
ability PowerStrike {
    target: enemy
    range: 3.0
    cooldown: 5s
    hint: damage

    damage 10 scales_with caster_level
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Damage { amount, bonus, .. } => {
            assert_eq!(*amount, 10);
            assert_eq!(bonus.len(), 1);
            assert!(matches!(bonus[0].stat, crate::effects::types::StatRef::CasterLevel));
            assert!((bonus[0].percent - 100.0).abs() < 0.01);
        }
        other => panic!("expected Damage, got: {other:?}"),
    }
}

#[test]
fn parse_while_alive_and_scales_with_combined() {
    let input = r#"
ability KingdomAura {
    target: self_aoe
    cooldown: 0s
    hint: utility

    buff damage_output 0.01 while_alive scales_with kingdom_size
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Buff { duration_ms, .. } => {
            assert_eq!(*duration_ms, u32::MAX, "should be while_alive");
        }
        other => panic!("expected Buff, got: {other:?}"),
    }
    // Scaling is parsed but only applies to Damage/Heal effects during lowering
}

#[test]
fn roundtrip_while_alive() {
    // Parse, emit, re-parse, and verify while_alive survives the roundtrip
    let input = r#"
ability AuraOfCourage {
    target: self_aoe
    cooldown: 0s
    hint: defense

    immunity "fear" in circle(10.0) while_alive
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let emitted = crate::effects::dsl::emit::emit_ability_dsl(&abilities[0]);
    eprintln!("Emitted:\n{emitted}");
    let (reparsed, _) = parse_abilities(&emitted).unwrap();
    let eff = &reparsed[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Immunity { duration_ms, .. } => {
            assert_eq!(*duration_ms, u32::MAX, "while_alive should survive roundtrip");
        }
        other => panic!("expected Immunity, got: {other:?}"),
    }
}

#[test]
fn roundtrip_scales_with() {
    let input = r#"
ability LevelStrike {
    target: enemy
    range: 3.0
    cooldown: 5s
    hint: damage

    damage 10 scales_with caster_level
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let emitted = crate::effects::dsl::emit::emit_ability_dsl(&abilities[0]);
    eprintln!("Emitted:\n{emitted}");
    let (reparsed, _) = parse_abilities(&emitted).unwrap();
    let eff = &reparsed[0].effects[0];
    match &eff.effect {
        crate::effects::effect_enum::Effect::Damage { bonus, .. } => {
            assert_eq!(bonus.len(), 1);
            assert!(matches!(bonus[0].stat, crate::effects::types::StatRef::CasterLevel));
        }
        other => panic!("expected Damage, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Targeting filter tests
// ---------------------------------------------------------------------------

#[test]
fn parse_targeting_under_command() {
    let input = r#"
ability CommandBuff {
    target: self_aoe
    cooldown: 10s
    hint: utility

    buff damage_output 0.3 for 10s in circle(8.0) targeting under_command
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 1);
    let eff = &abilities[0].effects[0];
    assert!(matches!(&eff.targeting_filter, Some(TargetFilter::UnderCommand)));
}

#[test]
fn parse_targeting_has_class() {
    let input = r#"
ability KnightBless {
    target: ally
    range: 6.0
    cooldown: 8s
    hint: utility

    buff defense 0.2 for 5s in circle(6.0) targeting has_class("knight")
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.targeting_filter {
        Some(TargetFilter::HasClass { class_name }) => assert_eq!(class_name, "knight"),
        other => panic!("expected HasClass, got: {other:?}"),
    }
}

#[test]
fn parse_targeting_loyalty_above() {
    let input = r#"
ability LoyalRally {
    target: self_aoe
    cooldown: 15s
    hint: leadership

    heal 20 in circle(10.0) targeting loyalty_above(75)
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    match &eff.targeting_filter {
        Some(TargetFilter::LoyaltyAbove { threshold }) => assert!((*threshold - 75.0).abs() < 0.01),
        other => panic!("expected LoyaltyAbove, got: {other:?}"),
    }
}

#[test]
fn parse_targeting_injured() {
    let input = r#"
ability BattlefieldMedic {
    target: self_aoe
    cooldown: 12s
    hint: heal

    heal 30 in circle(8.0) targeting injured
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    assert!(matches!(&eff.targeting_filter, Some(TargetFilter::Injured)));
}

#[test]
fn parse_targeting_with_condition_and_tags() {
    let input = r#"
ability HolySmite {
    target: enemy
    range: 5.0
    cooldown: 6s
    hint: damage

    damage 40 in circle(4.0) [HOLY: 80] when target_hp_below(50%) targeting has_status("undead")
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    assert!(eff.condition.is_some());
    assert!(!eff.tags.is_empty());
    match &eff.targeting_filter {
        Some(TargetFilter::HasStatus { status }) => assert_eq!(status, "undead"),
        other => panic!("expected HasStatus, got: {other:?}"),
    }
}

#[test]
fn roundtrip_targeting_filter() {
    let input = r#"
ability CommandBuff {
    target: self_aoe
    cooldown: 10s
    hint: utility

    buff damage_output 0.3 for 10s in circle(8.0) targeting under_command
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let emitted = crate::effects::dsl::emit::emit_ability_dsl(&abilities[0]);
    eprintln!("Emitted:\n{emitted}");
    let (reparsed, _) = parse_abilities(&emitted).unwrap();
    let eff = &reparsed[0].effects[0];
    assert!(matches!(&eff.targeting_filter, Some(TargetFilter::UnderCommand)));
}

#[test]
fn roundtrip_targeting_has_class() {
    let input = r#"
ability KnightBless {
    target: ally
    range: 6.0
    cooldown: 8s
    hint: utility

    buff defense 0.2 for 5s in circle(6.0) targeting has_class("knight")
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let emitted = crate::effects::dsl::emit::emit_ability_dsl(&abilities[0]);
    eprintln!("Emitted:\n{emitted}");
    let (reparsed, _) = parse_abilities(&emitted).unwrap();
    match &reparsed[0].effects[0].targeting_filter {
        Some(TargetFilter::HasClass { class_name }) => assert_eq!(class_name, "knight"),
        other => panic!("expected HasClass after roundtrip, got: {other:?}"),
    }
}

#[test]
fn no_targeting_filter_when_absent() {
    let input = r#"
ability Fireball {
    target: enemy
    range: 5.0
    cooldown: 5s
    hint: damage

    damage 55
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    let eff = &abilities[0].effects[0];
    assert!(eff.targeting_filter.is_none());
}

#[test]
fn parse_new_dataset_abilities() {
    let base_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../dataset/abilities");

    let new_files = &[
        "tier8_passives/aura_passives.ability",
        "tier7_complex/targeting_filter.ability",
        "tier7_complex/scaling_abilities.ability",
        "tier7_complex/banish_abilities.ability",
        "tier9_ultimates/combination_ultimates.ability",
    ];

    let mut count = 0;
    let mut errors = Vec::new();

    for file in new_files {
        let path = base_dir.join(file);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("cannot read {}", path.display()));
        match parse_abilities(&content) {
            Ok((a, p)) => {
                count += 1;
                eprintln!("  OK: {file} — {} abilities, {} passives", a.len(), p.len());
            }
            Err(e) => errors.push(format!("{file}: {e}")),
        }
    }

    if !errors.is_empty() {
        panic!("{} / {} new .ability files failed to parse:\n{}",
            errors.len(), count + errors.len(), errors.join("\n"));
    }
    eprintln!("Successfully parsed all {count} new .ability files");
}
