use super::*;

// --- ScalingTerm (bonus) Tests ---

#[test]
fn parse_damage_with_bonus_scaling_terms() {
    let toml_str = r#"
type = "damage"
amount = 0

[[bonus]]
stat = "target_max_hp"
percent = 10.0
max = 200

[[bonus]]
stat = "caster_attack_damage"
percent = 50.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Damage { amount, ref bonus, .. } => {
            assert_eq!(amount, 0);
            assert_eq!(bonus.len(), 2);
            assert!(matches!(bonus[0].stat, StatRef::TargetMaxHp));
            assert!((bonus[0].percent - 10.0).abs() < 0.01);
            assert_eq!(bonus[0].max, 200);
            assert!(matches!(bonus[1].stat, StatRef::CasterAttackDamage));
            assert!((bonus[1].percent - 50.0).abs() < 0.01);
        }
        _ => panic!("expected Damage"),
    }
}

#[test]
fn parse_heal_with_bonus_scaling_terms() {
    let toml_str = r#"
type = "heal"
amount = 0

[[bonus]]
stat = "target_missing_hp"
percent = 50.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Heal { amount, ref bonus, .. } => {
            assert_eq!(amount, 0);
            assert_eq!(bonus.len(), 1);
            assert!(matches!(bonus[0].stat, StatRef::TargetMissingHp));
            assert!((bonus[0].percent - 50.0).abs() < 0.01);
        }
        _ => panic!("expected Heal"),
    }
}

#[test]
fn parse_damage_with_stack_scaling_and_consume() {
    let toml_str = r#"
type = "damage"
amount = 20

[[bonus]]
stat.target_stacks = { name = "venom" }
percent = 1500.0
consume = true
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Damage { amount, ref bonus, .. } => {
            assert_eq!(amount, 20);
            assert_eq!(bonus.len(), 1);
            assert!(matches!(&bonus[0].stat, StatRef::TargetStacks { name } if name == "venom"));
            assert!(bonus[0].consume);
        }
        _ => panic!("expected Damage"),
    }
}

// --- Distance & Resource Condition Tests ---

#[test]
fn parse_condition_target_distance_below() {
    let toml_str = r#"
type = "target_distance_below"
range = 3.0
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::TargetDistanceBelow { range } => assert!((range - 3.0).abs() < 0.01),
        _ => panic!("expected TargetDistanceBelow"),
    }
}

#[test]
fn parse_condition_target_distance_above() {
    let toml_str = r#"
type = "target_distance_above"
range = 5.0
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::TargetDistanceAbove { range } => assert!((range - 5.0).abs() < 0.01),
        _ => panic!("expected TargetDistanceAbove"),
    }
}

#[test]
fn parse_condition_caster_resource_below() {
    let toml_str = r#"
type = "caster_resource_below"
percent = 20.0
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::CasterResourceBelow { percent } => assert!((percent - 20.0).abs() < 0.01),
        _ => panic!("expected CasterResourceBelow"),
    }
}

#[test]
fn parse_condition_caster_resource_above() {
    let toml_str = r#"
type = "caster_resource_above"
percent = 80.0
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::CasterResourceAbove { percent } => assert!((percent - 80.0).abs() < 0.01),
        _ => panic!("expected CasterResourceAbove"),
    }
}

// --- ConditionalEffect chance & else_effects Tests ---

#[test]
fn parse_conditional_effect_with_chance() {
    let toml_str = r#"
type = "stun"
duration_ms = 1500
chance = 0.3
"#;
    let ce: ConditionalEffect = toml::from_str(toml_str).unwrap();
    assert!((ce.chance - 0.3).abs() < 0.01);
    assert!(matches!(ce.effect, Effect::Stun { duration_ms: 1500 }));
}

#[test]
fn parse_conditional_effect_with_else_effects() {
    let toml_str = r#"
type = "damage"
amount = 60

[condition]
type = "target_hp_below"
percent = 30.0

[[else_effects]]
type = "damage"
amount = 25
"#;
    let ce: ConditionalEffect = toml::from_str(toml_str).unwrap();
    assert!(ce.condition.is_some());
    assert_eq!(ce.else_effects.len(), 1);
    match &ce.else_effects[0].effect {
        Effect::Damage { amount, .. } => assert_eq!(*amount, 25),
        _ => panic!("expected Damage in else_effects"),
    }
}

#[test]
fn parse_conditional_effect_default_chance_is_zero() {
    let toml_str = r#"
type = "damage"
amount = 10
"#;
    let ce: ConditionalEffect = toml::from_str(toml_str).unwrap();
    assert!((ce.chance - 0.0).abs() < 0.001);
    assert!(ce.else_effects.is_empty());
}
