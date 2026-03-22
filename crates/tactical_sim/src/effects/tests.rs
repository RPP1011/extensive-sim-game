use super::*;
use serde::Deserialize;

#[test]
fn parse_damage_effect() {
    let toml_str = r#"
type = "damage"
amount = 50
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Damage { amount, .. } => assert_eq!(amount, 50),
        _ => panic!("expected Damage"),
    }
}

#[test]
fn parse_dot_effect() {
    let toml_str = r#"
type = "damage"
amount_per_tick = 10
duration_ms = 4000
tick_interval_ms = 1000
"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Damage {
            amount_per_tick,
            duration_ms,
            tick_interval_ms,
            ..
        } => {
            assert_eq!(amount_per_tick, 10);
            assert_eq!(duration_ms, 4000);
            assert_eq!(tick_interval_ms, 1000);
        }
        _ => panic!("expected Damage DoT"),
    }
}

#[test]
fn parse_area_circle() {
    let toml_str = r#"
shape = "circle"
radius = 2.5
"#;
    let area: Area = toml::from_str(toml_str).unwrap();
    match area {
        Area::Circle { radius } => assert!((radius - 2.5).abs() < 0.01),
        _ => panic!("expected Circle"),
    }
}

#[test]
fn parse_delivery_projectile() {
    let toml_str = r#"
method = "projectile"
speed = 12.0
pierce = true
width = 0.3
"#;
    let delivery: Delivery = toml::from_str(toml_str).unwrap();
    match delivery {
        Delivery::Projectile {
            speed, pierce, width, ..
        } => {
            assert!((speed - 12.0).abs() < 0.01);
            assert!(pierce);
            assert!((width - 0.3).abs() < 0.01);
        }
        _ => panic!("expected Projectile"),
    }
}

#[test]
fn parse_condition_target_hp_below() {
    let toml_str = r#"
type = "target_hp_below"
percent = 30.0
"#;
    let cond: Condition = toml::from_str(toml_str).unwrap();
    match cond {
        Condition::TargetHpBelow { percent } => assert!((percent - 30.0).abs() < 0.01),
        _ => panic!("expected TargetHpBelow"),
    }
}

#[test]
fn parse_trigger_on_damage_taken() {
    let toml_str = r#"
type = "on_damage_taken"
"#;
    let trigger: Trigger = toml::from_str(toml_str).unwrap();
    assert!(matches!(trigger, Trigger::OnDamageTaken));
}

#[test]
fn parse_stacking() {
    #[derive(Deserialize)]
    struct W {
        s: Stacking,
    }
    let toml_str = r#"s = "strongest""#;
    let w: W = toml::from_str(toml_str).unwrap();
    assert_eq!(w.s, Stacking::Strongest);
}

#[test]
fn default_stacking_is_refresh() {
    assert_eq!(Stacking::default(), Stacking::Refresh);
}

#[test]
fn ability_slot_starts_ready() {
    let slot = AbilitySlot::new(AbilityDef {
        name: "Test".into(),
        targeting: AbilityTargeting::TargetEnemy,
        range: 5.0,
        cooldown_ms: 3000,
        cast_time_ms: 200,
        ai_hint: "damage".into(),
        effects: vec![],
        delivery: None,
        resource_cost: 0,
        morph_into: None,
        morph_duration_ms: 0,
        zone_tag: None,
        ..Default::default()
    });
    assert!(slot.is_ready());
}

#[test]
fn passive_slot_starts_ready() {
    let slot = PassiveSlot::new(PassiveDef {
        name: "Test Passive".into(),
        trigger: Trigger::OnDamageTaken,
        cooldown_ms: 5000,
        effects: vec![],
        range: 0.0,
    });
    assert!(slot.is_ready());
}

#[test]
fn parse_hero_toml_warrior() {
    let toml_str = r#"
[hero]
name = "Warrior"

[stats]
hp = 180
move_speed = 2.2

[stats.tags]
TENACITY = 60.0

[attack]
damage = 18
range = 1.5
cooldown = 1000
cast_time = 300

[[abilities]]
name = "Whirlwind"
targeting = "self_aoe"
range = 0.0
cooldown_ms = 8000
cast_time_ms = 300
ai_hint = "damage"

[[abilities.effects]]
type = "damage"
amount = 40

[abilities.effects.area]
shape = "circle"
radius = 2.5

[[abilities]]
name = "Shield Wall"
targeting = "self_cast"
range = 0.0
cooldown_ms = 15000
cast_time_ms = 200
ai_hint = "defense"

[[abilities.effects]]
type = "shield"
amount = 60
duration_ms = 5000

[[passives]]
name = "Iron Skin"
cooldown_ms = 5000

[passives.trigger]
type = "on_damage_taken"

[[passives.effects]]
type = "shield"
amount = 20
duration_ms = 3000
"#;
    let hero: HeroToml = toml::from_str(toml_str).unwrap();
    assert_eq!(hero.hero.name, "Warrior");
    assert_eq!(hero.stats.hp, 180);
    assert_eq!(hero.abilities.len(), 2);
    assert_eq!(hero.abilities[0].name, "Whirlwind");
    assert_eq!(hero.abilities[1].name, "Shield Wall");
    assert_eq!(hero.passives.len(), 1);
    assert_eq!(hero.passives[0].name, "Iron Skin");
}

// --- New effect parsing tests ---

#[test]
fn parse_root_effect() {
    let toml_str = r#"type = "root"
duration_ms = 2000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Root { duration_ms: 2000 }));
}

#[test]
fn parse_silence_effect() {
    let toml_str = r#"type = "silence"
duration_ms = 3000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Silence { duration_ms: 3000 }));
}

#[test]
fn parse_fear_effect() {
    let toml_str = r#"type = "fear"
duration_ms = 1500"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Fear { duration_ms: 1500 }));
}

#[test]
fn parse_taunt_effect() {
    let toml_str = r#"type = "taunt"
duration_ms = 4000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Taunt { duration_ms: 4000 }));
}

#[test]
fn parse_pull_effect() {
    let toml_str = r#"type = "pull"
distance = 3.5"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Pull { distance } => assert!((distance - 3.5).abs() < 0.01),
        _ => panic!("expected Pull"),
    }
}

#[test]
fn parse_swap_effect() {
    let toml_str = r#"type = "swap""#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    assert!(matches!(effect, Effect::Swap));
}

#[test]
fn parse_reflect_effect() {
    let toml_str = r#"type = "reflect"
percent = 30.0
duration_ms = 5000"#;
    let effect: Effect = toml::from_str(toml_str).unwrap();
    match effect {
        Effect::Reflect { percent, duration_ms } => {
            assert!((percent - 30.0).abs() < 0.01);
            assert_eq!(duration_ms, 5000);
        }
        _ => panic!("expected Reflect"),
    }
}
