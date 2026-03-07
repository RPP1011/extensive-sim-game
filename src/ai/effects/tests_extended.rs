use super::*;

#[test]
fn parse_damage_modify_effect() {
    let toml_str = r#"type = "damage_modify"
factor = 0.75
duration_ms = 6000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::DamageModify { factor, duration_ms } => {
            assert!((factor - 0.75).abs() < 0.01);
            assert_eq!(duration_ms, 6000);
        }
        _ => panic!("expected DamageModify"),
    }
}

#[test]
fn parse_execute_effect() {
    let toml_str = r#"type = "execute"
hp_threshold_percent = 20.0"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Execute { hp_threshold_percent } => {
            assert!((hp_threshold_percent - 20.0).abs() < 0.01);
        }
        _ => panic!("expected Execute"),
    }
}

#[test]
fn parse_resurrect_effect() {
    let toml_str = r#"type = "resurrect"
hp_percent = 50.0"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Resurrect { hp_percent } => {
            assert!((hp_percent - 50.0).abs() < 0.01);
        }
        _ => panic!("expected Resurrect"),
    }
}

#[test]
fn parse_immunity_effect() {
    let toml_str = r#"type = "immunity"
immune_to = ["stun", "silence"]
duration_ms = 5000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Immunity { immune_to, duration_ms } => {
            assert_eq!(immune_to, vec!["stun", "silence"]);
            assert_eq!(duration_ms, 5000);
        }
        _ => panic!("expected Immunity"),
    }
}

#[test]
fn parse_polymorph_effect() {
    let toml_str = r#"type = "polymorph"
duration_ms = 3000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Polymorph { duration_ms: 3000 }));
}

#[test]
fn parse_stealth_effect() {
    let toml_str = r#"type = "stealth"
duration_ms = 5000
break_on_damage = true
break_on_ability = true"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Stealth { duration_ms, break_on_damage, break_on_ability } => {
            assert_eq!(duration_ms, 5000);
            assert!(break_on_damage);
            assert!(break_on_ability);
        }
        _ => panic!("expected Stealth"),
    }
}

#[test]
fn parse_delivery_chain() {
    let toml_str = r#"
method = "chain"
bounces = 3
bounce_range = 4.0
falloff = 0.15
"#;
    let delivery: Delivery = toml::from_str(toml_str).unwrap();
    match delivery {
        Delivery::Chain { bounces, bounce_range, falloff, .. } => {
            assert_eq!(bounces, 3);
            assert!((bounce_range - 4.0).abs() < 0.01);
            assert!((falloff - 0.15).abs() < 0.01);
        }
        _ => panic!("expected Chain"),
    }
}

#[test]
fn parse_area_spread() {
    let toml_str = r#"
shape = "spread"
radius = 3.0
max_targets = 4
"#;
    let area: Area = toml::from_str(toml_str).unwrap();
    match area {
        Area::Spread { radius, max_targets } => {
            assert!((radius - 3.0).abs() < 0.01);
            assert_eq!(max_targets, 4);
        }
        _ => panic!("expected Spread"),
    }
}

#[test]
fn parse_damage_with_scaling() {
    let toml_str = r#"
type = "damage"
amount = 30
scaling_stat = "caster_max_hp"
scaling_percent = 10.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Damage { amount, scaling_stat, scaling_percent, .. } => {
            assert_eq!(amount, 30);
            assert_eq!(scaling_stat.as_deref(), Some("caster_max_hp"));
            assert!((scaling_percent - 10.0).abs() < 0.01);
        }
        _ => panic!("expected Damage"),
    }
}

#[test]
fn parse_summon_with_hp_percent() {
    let toml_str = r#"
type = "summon"
template = "self"
count = 1
hp_percent = 50.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Summon { template, hp_percent, .. } => {
            assert_eq!(template, "self");
            assert!((hp_percent - 50.0).abs() < 0.01);
        }
        _ => panic!("expected Summon"),
    }
}

#[test]
fn parse_new_conditions() {
    let toml_str = r#"type = "target_is_rooted""#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    assert!(matches!(cond, Condition::TargetIsRooted));

    let toml_str = r#"type = "target_is_silenced""#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    assert!(matches!(cond, Condition::TargetIsSilenced));

    let toml_str = r#"type = "target_debuff_count"
min_count = 3"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    assert!(matches!(cond, Condition::TargetDebuffCount { min_count: 3 }));
}

#[test]
fn parse_new_triggers() {
    let toml_str = r#"type = "on_heal_received""#;
    let trigger: Trigger = toml::from_str(toml_str).unwrap();
    assert!(matches!(trigger, Trigger::OnHealReceived));

    let toml_str = r#"type = "on_auto_attack""#;
    let trigger: Trigger = toml::from_str(toml_str).unwrap();
    assert!(matches!(trigger, Trigger::OnAutoAttack));

    let toml_str = r#"type = "on_ally_killed"
range = 8.0"#;
    let trigger: Trigger = toml::from_str(toml_str).unwrap();
    match trigger {
        Trigger::OnAllyKilled { range } => assert!((range - 8.0).abs() < 0.01),
        _ => panic!("expected OnAllyKilled"),
    }
}

#[test]
fn parse_cooldown_modify_effect() {
    let toml_str = r#"type = "cooldown_modify"
amount_ms = -2000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::CooldownModify { amount_ms, .. } => assert_eq!(amount_ms, -2000),
        _ => panic!("expected CooldownModify"),
    }
}

#[test]
fn parse_death_mark_effect() {
    let toml_str = r#"type = "death_mark"
duration_ms = 5000
damage_percent = 40.0"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::DeathMark { duration_ms, damage_percent } => {
            assert_eq!(duration_ms, 5000);
            assert!((damage_percent - 40.0).abs() < 0.01);
        }
        _ => panic!("expected DeathMark"),
    }
}

#[test]
fn parse_blind_effect() {
    let toml_str = r#"type = "blind"
miss_chance = 0.5
duration_ms = 3000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Blind { miss_chance, duration_ms } => {
            assert!((miss_chance - 0.5).abs() < 0.01);
            assert_eq!(duration_ms, 3000);
        }
        _ => panic!("expected Blind"),
    }
}

#[test]
fn parse_lifesteal_effect() {
    let toml_str = r#"type = "lifesteal"
percent = 25.0
duration_ms = 6000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Lifesteal { percent, duration_ms } => {
            assert!((percent - 25.0).abs() < 0.01);
            assert_eq!(duration_ms, 6000);
        }
        _ => panic!("expected Lifesteal"),
    }
}

#[test]
fn parse_link_effect() {
    let toml_str = r#"type = "link"
duration_ms = 8000
share_percent = 30.0"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Link { duration_ms, share_percent } => {
            assert_eq!(duration_ms, 8000);
            assert!((share_percent - 30.0).abs() < 0.01);
        }
        _ => panic!("expected Link"),
    }
}

#[test]
fn parse_rewind_effect() {
    let toml_str = r#"type = "rewind"
lookback_ms = 5000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Rewind { lookback_ms } => assert_eq!(lookback_ms, 5000),
        _ => panic!("expected Rewind"),
    }
}

// --- Compound Condition Tests ---

#[test]
fn parse_compound_condition_and() {
    let toml_str = r#"
type = "and"
[[conditions]]
type = "target_hp_below"
percent = 50.0
[[conditions]]
type = "target_is_stunned"
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::And { conditions } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0], Condition::TargetHpBelow { percent } if (percent - 50.0).abs() < 0.01));
            assert!(matches!(conditions[1], Condition::TargetIsStunned));
        }
        _ => panic!("expected And"),
    }
}

#[test]
fn parse_compound_condition_or() {
    let toml_str = r#"
type = "or"
[[conditions]]
type = "caster_hp_below"
percent = 30.0
[[conditions]]
type = "ally_count_below"
count = 2
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::Or { conditions } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0], Condition::CasterHpBelow { .. }));
            assert!(matches!(conditions[1], Condition::AllyCountBelow { .. }));
        }
        _ => panic!("expected Or"),
    }
}

#[test]
fn parse_compound_condition_not() {
    let toml_str = r#"
type = "not"
[condition]
type = "target_is_stunned"
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::Not { condition } => {
            assert!(matches!(*condition, Condition::TargetIsStunned));
        }
        _ => panic!("expected Not"),
    }
}

// --- Percent HP Effect Tests ---

#[test]
fn parse_percent_hp_damage() {
    let toml_str = r#"
type = "percent_hp_damage"
percent = 10.0
damage_type = "magic"
max_damage = 200
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::PercentHpDamage { percent, damage_type, max_damage } => {
            assert!((percent - 10.0).abs() < 0.01);
            assert!(matches!(damage_type, DamageType::Magic));
            assert_eq!(max_damage, 200);
        }
        _ => panic!("expected PercentHpDamage"),
    }
}

#[test]
fn parse_percent_missing_hp_heal() {
    let toml_str = r#"
type = "percent_missing_hp_heal"
percent = 50.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::PercentMissingHpHeal { percent } => {
            assert!((percent - 50.0).abs() < 0.01);
        }
        _ => panic!("expected PercentMissingHpHeal"),
    }
}

#[test]
fn parse_percent_max_hp_heal() {
    let toml_str = r#"
type = "percent_max_hp_heal"
percent = 15.0
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::PercentMaxHpHeal { percent } => {
            assert!((percent - 15.0).abs() < 0.01);
        }
        _ => panic!("expected PercentMaxHpHeal"),
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

// --- DamagePerStack Effect Test ---

#[test]
fn parse_damage_per_stack() {
    let toml_str = r#"
type = "damage_per_stack"
base = 20
per_stack = 15
stack_name = "venom"
damage_type = "magic"
consume = true
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::DamagePerStack { base, per_stack, stack_name, damage_type, consume } => {
            assert_eq!(base, 20);
            assert_eq!(per_stack, 15);
            assert_eq!(stack_name, "venom");
            assert!(matches!(damage_type, DamageType::Magic));
            assert!(consume);
        }
        _ => panic!("expected DamagePerStack"),
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
