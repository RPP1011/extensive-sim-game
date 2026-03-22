//! God ability tests, edge case regression tests, realistic generators,
//! and the dataset generator for the ability DSL fuzzer.

use proptest::prelude::*;

use crate::effects::dsl::parse_abilities;

use super::fuzz_generators::*;
use super::fuzz::tests::Coverage;

// -----------------------------------------------------------------------
// God ability — single ability coverage test
// -----------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn fuzz_god_ability(input in god_ability_block()) {
        let result = parse_abilities(&input);
        prop_assert!(
            result.is_ok(),
            "God ability failed:\n---INPUT---\n{}\n---ERROR---\n{}",
            input,
            result.unwrap_err()
        );
        let (abilities, _) = result.unwrap();
        prop_assert_eq!(abilities.len(), 1);
    }
}

#[test]
fn god_ability_coverage() {
    use proptest::test_runner::{TestRunner, Config};
    use proptest::strategy::ValueTree;
    use std::collections::BTreeSet;

    let config = Config { cases: 1, .. Config::default() };
    let mut runner = TestRunner::new(config);

    let tree = god_ability_block().new_tree(&mut runner).unwrap();
    let input = tree.current();

    let (abilities, _) = parse_abilities(&input)
        .unwrap_or_else(|e| panic!("God ability failed to parse:\n{input}\n---\n{e}"));

    assert_eq!(abilities.len(), 1);

    let mut cov = Coverage::default();
    cov.record_abilities(&abilities, &[]);

    let report = cov.report();
    eprintln!("\n=== God Ability Coverage ===\n{report}");

    let mut hit_features = BTreeSet::new();
    if cov.has_charges { hit_features.insert("charges"); }
    if cov.has_recast { hit_features.insert("recast"); }
    if cov.has_cost { hit_features.insert("cost"); }
    if cov.has_tags { hit_features.insert("tags"); }
    if cov.has_scaling { hit_features.insert("scaling"); }
    if cov.has_toggle { hit_features.insert("toggle"); }
    if cov.has_unstoppable { hit_features.insert("unstoppable"); }
    if cov.has_form { hit_features.insert("form"); }

    let total = cov.effects.len()
        + cov.deliveries.len()
        + cov.areas.len()
        + cov.conditions.len()
        + cov.targetings.len()
        + hit_features.len();

    eprintln!("Single ability feature count: {total}");
    eprintln!("  effects: {} {:?}", cov.effects.len(), cov.effects);
    eprintln!("  deliveries: {} {:?}", cov.deliveries.len(), cov.deliveries);
    eprintln!("  areas: {} {:?}", cov.areas.len(), cov.areas);
    eprintln!("  conditions: {} {:?}", cov.conditions.len(), cov.conditions);
    eprintln!("  targeting: {} {:?}", cov.targetings.len(), cov.targetings);
    eprintln!("  features: {} {:?}", hit_features.len(), hit_features);

    assert!(total >= 45,
        "God ability should cover >=45 features, got {total}. Report:\n{report}");
}

// -----------------------------------------------------------------------
// Edge case regression tests
// -----------------------------------------------------------------------

#[test]
fn fuzz_empty_effects_in_delivery() {
    let input = r#"
ability Test {
    target: ground, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            damage 10
        }
    }
}
"#;
    parse_abilities(input).unwrap();
}

#[test]
fn fuzz_multiple_tags() {
    let input = r#"
ability Test {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    damage 50 [FIRE: 60, MAGIC: 40, CROWD_CONTROL: 20]
}
"#;
    parse_abilities(input).unwrap();
}

#[test]
fn fuzz_projectile_with_arrival() {
    let input = r#"
ability Test {
    target: enemy, range: 8.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 10.0, pierce, width: 0.5 } {
        on_hit {
            damage 30 [PHYSICAL: 50]
            slow 0.3 for 2s
        }
        on_arrival {
            damage 15 in circle(2.5)
        }
    }
}
"#;
    parse_abilities(input).unwrap();
}

#[test]
fn fuzz_many_effects() {
    let input = r#"
ability KitchenSink {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 300ms
    hint: damage
    cost: 15
    charges: 2
    recharge: 8s

    damage 50 [FIRE: 60]
    heal 20
    shield 30 for 4s
    stun 1s
    slow 0.3 for 2s
    dash to_target
    buff move_speed 0.2 for 3s
    apply_stacks "combo" 1 max 5
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities[0].effects.len(), 8);
}

#[test]
fn fuzz_mixed_abilities_and_passives() {
    let input = r#"
ability A {
    target: enemy, range: 2.0
    cooldown: 3s
    hint: damage
    damage 10
}

passive B {
    trigger: on_kill
    cooldown: 5s
    heal 20
}

ability C {
    target: self
    cooldown: 10s
    hint: utility
    dash 4.0
}

passive D {
    trigger: on_hp_below(25%)
    cooldown: 30s
    shield 50 for 5s
}
"#;
    let (abilities, passives) = parse_abilities(input).unwrap();
    assert_eq!(abilities.len(), 2);
    assert_eq!(passives.len(), 2);
}

#[test]
fn fuzz_toggle_and_unstoppable() {
    let input = r#"
ability Spin {
    target: self_aoe
    cooldown: 1s
    hint: damage
    toggle
    toggle_cost: 5

    damage 10 in circle(2.5)
}

ability Charge {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 0ms
    hint: crowd_control
    unstoppable

    dash to_target
    stun 1s
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert!(abilities[0].is_toggle);
    assert!(abilities[1].unstoppable);
}

#[test]
fn fuzz_form_swap() {
    let input = r#"
ability Transform {
    target: self
    cooldown: 3s
    hint: utility
    swap_form: "dragon"

    buff move_speed 0.3 for 10s
}
"#;
    let (abilities, _) = parse_abilities(input).unwrap();
    assert_eq!(abilities[0].swap_form, Some("dragon".to_string()));
}

// -----------------------------------------------------------------------
// Realistic parameter strategies
// -----------------------------------------------------------------------

fn realistic_cooldown() -> impl Strategy<Value = String> {
    prop_oneof![
        2 => (3u32..8).prop_map(|s| format!("{s}s")),
        4 => (8u32..15).prop_map(|s| format!("{s}s")),
        2 => (15u32..30).prop_map(|s| format!("{s}s")),
        1 => (30u32..60).prop_map(|s| format!("{s}s")),
    ]
}

fn realistic_cast() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => Just("0ms".to_string()),
        3 => (100u32..500).prop_map(|ms| format!("{ms}ms")),
        2 => (500u32..1500).prop_map(|ms| format!("{ms}ms")),
        1 => (2u32..4).prop_map(|s| format!("{s}s")),
    ]
}

fn realistic_effect_duration() -> impl Strategy<Value = String> {
    prop_oneof![
        2 => (300u32..1000).prop_map(|ms| format!("{ms}ms")),
        4 => (1u32..4).prop_map(|s| format!("{s}s")),
        2 => (4u32..8).prop_map(|s| format!("{s}s")),
        1 => (8u32..15).prop_map(|s| format!("{s}s")),
    ]
}

fn realistic_range() -> impl Strategy<Value = String> {
    prop_oneof![
        2 => (1u32..3).prop_map(|r| format!("{r}.0")),
        4 => (3u32..7).prop_map(|r| format!("{r}.0")),
        2 => (7u32..10).prop_map(|r| format!("{r}.0")),
    ]
}

fn realistic_damage() -> impl Strategy<Value = String> {
    (15i32..80).prop_map(|d| d.to_string())
}

fn realistic_heal() -> impl Strategy<Value = String> {
    (15i32..60).prop_map(|d| d.to_string())
}

fn realistic_shield() -> impl Strategy<Value = String> {
    (20i32..60).prop_map(|d| d.to_string())
}

fn realistic_area() -> impl Strategy<Value = String> {
    prop_oneof![
        (2u32..6).prop_map(|r| format!("in circle({r}.0)")),
        (3u32..6, 30u32..120).prop_map(|(r, a)| format!("in cone({r}.0, {a}.0)")),
        (3u32..8, 1u32..3).prop_map(|(l, w)| format!("in line({l}.0, {w}.0)")),
        (1u32..3, 3u32..7).prop_map(|(i, o)| format!("in ring({i}.0, {o}.0)")),
        (2u32..5, 2u32..5).prop_map(|(r, t)| format!("in spread({r}.0, {t})")),
    ]
}

fn realistic_opt_area() -> impl Strategy<Value = String> {
    prop_oneof![3 => Just(String::new()), 1 => realistic_area()]
}

// -----------------------------------------------------------------------
// Archetype-based ability generators
// -----------------------------------------------------------------------

fn damage_archetype_effects() -> impl Strategy<Value = String> {
    prop_oneof![
        (realistic_damage(), opt_tags()).prop_map(|(d, tags)|
            format!("    damage {d} {tags}")),
        (realistic_damage(), realistic_area(), opt_tags()).prop_map(|(d, area, tags)|
            format!("    damage {d} {area} {tags}")),
        (realistic_damage(), 2u32..6, realistic_effect_duration(), opt_tags())
            .prop_map(|(d, slow10, dur, tags)| {
                let slow = slow10 as f32 / 10.0;
                format!("    damage {d} {tags}\n    slow {slow} for {dur}")
            }),
        (realistic_damage(), 5u32..30, scaling_stat(), opt_tags())
            .prop_map(|(d, pct, stat, tags)|
                format!("    damage {d} + {pct}% {stat} {tags}")),
        (realistic_damage(), 2u32..5, opt_tags()).prop_map(|(d, kb, tags)|
            format!("    damage {d} {tags}\n    knockback {kb}.0")),
    ]
}

fn heal_archetype_effects() -> impl Strategy<Value = String> {
    prop_oneof![
        realistic_heal().prop_map(|h| format!("    heal {h}")),
        (realistic_heal(), realistic_area()).prop_map(|(h, area)|
            format!("    heal {h} {area}")),
        (realistic_heal(), realistic_shield(), realistic_effect_duration())
            .prop_map(|(h, s, dur)|
                format!("    heal {h}\n    shield {s} for {dur}")),
        (realistic_heal(), 1u32..4, realistic_effect_duration()).prop_map(|(h, f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    heal {h}\n    buff move_speed {f} for {dur}")
        }),
        (realistic_heal(), simple_condition()).prop_map(|(h, cond)|
            format!("    heal {h} when {cond}")),
    ]
}

fn cc_archetype_effects() -> impl Strategy<Value = String> {
    let cc_type = prop_oneof![
        3 => Just("stun"), 3 => Just("root"), 2 => Just("silence"),
        1 => Just("fear"), 1 => Just("taunt"),
    ];
    prop_oneof![
        (cc_type.clone(), realistic_effect_duration()).prop_map(|(cc, dur)|
            format!("    {cc} {dur}")),
        (cc_type.clone(), realistic_effect_duration(), realistic_area())
            .prop_map(|(cc, dur, area)|
                format!("    {cc} {dur} {area}")),
        (cc_type.clone(), realistic_effect_duration(), realistic_damage())
            .prop_map(|(cc, dur, d)|
                format!("    {cc} {dur}\n    damage {d}")),
        (2u32..6, realistic_effect_duration(), realistic_damage(), realistic_opt_area())
            .prop_map(|(slow10, dur, d, area)| {
                let slow = slow10 as f32 / 10.0;
                format!("    slow {slow} for {dur} {area}\n    damage {d}")
            }),
    ]
}

fn buff_archetype_effects() -> impl Strategy<Value = String> {
    let buff_stat = prop_oneof![
        Just("move_speed"), Just("attack_speed"),
        Just("damage_output"), Just("cooldown_reduction"),
    ];
    prop_oneof![
        (buff_stat.clone(), 1u32..5, realistic_effect_duration())
            .prop_map(|(s, f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    buff {s} {f} for {dur}")
            }),
        (buff_stat.clone(), 1u32..5, realistic_effect_duration(), realistic_shield(), realistic_effect_duration())
            .prop_map(|(s, f10, dur, sh, sh_dur)| {
                let f = f10 as f32 / 10.0;
                format!("    buff {s} {f} for {dur}\n    shield {sh} for {sh_dur}")
            }),
        (buff_stat.clone(), 1u32..4, realistic_effect_duration(), realistic_area())
            .prop_map(|(s, f10, dur, area)| {
                let f = f10 as f32 / 10.0;
                format!("    buff {s} {f} for {dur} {area}")
            }),
        (10u32..20, realistic_effect_duration()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    damage_modify {f} for {dur}")
        }),
        (2u32..5, realistic_effect_duration()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    lifesteal {f} for {dur}")
        }),
    ]
}

fn defense_archetype_effects() -> impl Strategy<Value = String> {
    prop_oneof![
        (realistic_shield(), realistic_effect_duration()).prop_map(|(s, dur)|
            format!("    shield {s} for {dur}")),
        (realistic_shield(), realistic_effect_duration(), realistic_heal())
            .prop_map(|(s, dur, h)|
                format!("    shield {s} for {dur}\n    heal {h}")),
        (2u32..6, realistic_effect_duration()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    reflect {f} for {dur}")
        }),
        realistic_effect_duration().prop_map(|dur|
            format!("    stealth for {dur} break_on_damage")),
        (realistic_shield(), realistic_effect_duration(), 1u32..4, realistic_effect_duration())
            .prop_map(|(s, s_dur, f10, b_dur)| {
                let f = f10 as f32 / 10.0;
                format!("    shield {s} for {s_dur}\n    buff armor {f} for {b_dur}")
            }),
    ]
}

fn utility_archetype_effects() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("    dash to_target".to_string()),
        Just("    dash to_position".to_string()),
        (2u32..6).prop_map(|d| format!("    blink {d}.0")),
        (2u32..4, realistic_effect_duration()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    dash to_target\n    buff move_speed {f} for {dur}")
        }),
        Just("    swap".to_string()),
        Just("    dispel".to_string()),
    ]
}

fn archetype_effects(hint: &'static str) -> proptest::strategy::BoxedStrategy<String> {
    match hint {
        "damage" => damage_archetype_effects().boxed(),
        "heal" => heal_archetype_effects().boxed(),
        "crowd_control" => cc_archetype_effects().boxed(),
        "buff" => buff_archetype_effects().boxed(),
        "defense" => defense_archetype_effects().boxed(),
        "utility" => utility_archetype_effects().boxed(),
        _ => damage_archetype_effects().boxed(),
    }
}

fn realistic_delivery() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => (5u32..15, proptest::bool::ANY, effect_list(1, 2))
            .prop_map(|(speed, pierce, effects)| {
                let pierce_str = if pierce { ", pierce" } else { "" };
                format!(
                    "    deliver projectile {{ speed: {speed}.0{pierce_str} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            }),
        1 => (2u32..5, (3u32..6).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
            .prop_map(|(bounces, range, effects)|
                format!(
                    "    deliver chain {{ bounces: {bounces}, range: {range} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            ),
        2 => ((3u32..10).prop_map(|s| format!("{s}s")), (500u32..2000).prop_map(|ms| format!("{ms}ms")), effect_list(1, 2))
            .prop_map(|(dur, tick, effects)|
                format!(
                    "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            ),
        1 => ((2u32..5).prop_map(|s| format!("{s}s")), (300u32..1000).prop_map(|ms| format!("{ms}ms")), effect_list(1, 2))
            .prop_map(|(dur, tick, effects)|
                format!(
                    "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            ),
        1 => ((3u32..7).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
            .prop_map(|(range, effects)|
                format!(
                    "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effects}\n        }}\n    }}"
                )
            ),
        1 => ((5u32..15).prop_map(|s| format!("{s}s")), (1u32..3).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
            .prop_map(|(dur, radius, effects)|
                format!(
                    "    deliver trap {{ duration: {dur}, trigger_radius: {radius} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            ),
    ]
}

pub(super) fn coherent_ability() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => Just("damage"),
        2 => Just("heal"),
        2 => Just("crowd_control"),
        2 => Just("buff"),
        2 => Just("defense"),
        1 => Just("utility"),
    ]
    .prop_flat_map(|hint| {
        (
            ident_strategy(),
            match hint {
                "heal" => prop_oneof![3 => Just("ally"), 1 => Just("self"), 1 => Just("self_aoe")].boxed(),
                "buff" | "defense" => prop_oneof![2 => Just("self"), 2 => Just("ally"), 1 => Just("self_aoe")].boxed(),
                "utility" => prop_oneof![Just("self"), Just("enemy"), Just("ground")].boxed(),
                _ => prop_oneof![3 => Just("enemy"), 1 => Just("ground"), 1 => Just("direction")].boxed(),
            },
            realistic_range(),
            realistic_cooldown(),
            realistic_cast(),
            Just(hint),
            proptest::option::weighted(0.3, 5u32..25),
            proptest::option::weighted(0.2, (1u32..4, (5u32..15).prop_map(|s| format!("{s}s")))),
            proptest::option::weighted(0.15, (1u32..3, (2u32..5).prop_map(|s| format!("{s}s")))),
            prop_oneof![
                3 => archetype_effects(hint).prop_map(|e| (e, None)),
                2 => (archetype_effects(hint), realistic_delivery().prop_map(Some))
                    .prop_map(|(e, d)| (e, d)),
            ],
        )
            .prop_map(
                |(name, target, range, cd, cast, hint, cost, charges, recast, (effects, delivery))| {
                    let mut lines = vec![format!("ability {name} {{")];
                    lines.push(format!("    target: {target}, range: {range}"));
                    lines.push(format!("    cooldown: {cd}, cast: {cast}"));
                    lines.push(format!("    hint: {hint}"));

                    if let Some(c) = cost {
                        lines.push(format!("    cost: {c}"));
                    }
                    if let Some((ch, rech)) = charges {
                        lines.push(format!("    charges: {ch}"));
                        lines.push(format!("    recharge: {rech}"));
                    }
                    if let Some((rc, rw)) = recast {
                        lines.push(format!("    recast: {rc}"));
                        lines.push(format!("    recast_window: {rw}"));
                    }

                    lines.push(String::new());

                    if let Some(del) = delivery {
                        lines.push(del);
                    }
                    if !effects.is_empty() {
                        lines.push(effects);
                    }

                    lines.push("}".to_string());
                    lines.join("\n")
                },
            )
    })
}

pub(super) fn realistic_passive() -> impl Strategy<Value = String> {
    (
        ident_strategy(),
        trigger_strategy(),
        realistic_cooldown(),
        effect_list(1, 2),
    )
        .prop_map(|(name, trigger, cd, effects)| {
            format!(
                "passive {name} {{\n    trigger: {trigger}\n    cooldown: {cd}\n\n{effects}\n}}"
            )
        })
}

// -----------------------------------------------------------------------
// Dataset generator
// -----------------------------------------------------------------------

#[test]
#[ignore]
fn generate_ability_dataset() {
    use proptest::test_runner::{TestRunner, Config, TestRng, RngAlgorithm};
    use proptest::strategy::ValueTree;
    use std::collections::HashSet;
    use std::fs;
    use std::path::Path;

    let out_dir = Path::new("generated/ability_dataset");
    if out_dir.exists() {
        fs::remove_dir_all(out_dir).unwrap();
    }
    fs::create_dir_all(out_dir).unwrap();

    let mut all_dsl: Vec<String> = Vec::with_capacity(80_000);
    let mut seen_names: HashSet<String> = HashSet::new();

    fn make_seed(n: u64) -> [u8; 32] {
        let mut seed = [0u8; 32];
        for (i, chunk) in seed.chunks_exact_mut(8).enumerate() {
            let val = n.wrapping_add(i as u64);
            chunk.copy_from_slice(&val.to_le_bytes());
        }
        seed
    }

    fn make_runner(seed_val: u64) -> TestRunner {
        let config = Config { cases: 1, .. Config::default() };
        TestRunner::new_with_rng(config, TestRng::from_seed(
            RngAlgorithm::ChaCha, &make_seed(seed_val),
        ))
    }

    let batch = 250;
    for seed in 0..288u64 {
        let mut runner = make_runner(seed);
        let strat = proptest::collection::vec(coherent_ability(), batch..=batch);
        let tree = strat.new_tree(&mut runner).unwrap();
        all_dsl.extend(tree.current());
    }
    eprintln!("Generated {} coherent abilities", all_dsl.len());

    let normal_start = all_dsl.len();
    for seed in 0..6u64 {
        let mut runner = make_runner(seed + 1000);
        let strat = proptest::collection::vec(ability_block(), batch..=batch);
        let tree = strat.new_tree(&mut runner).unwrap();
        all_dsl.extend(tree.current());
    }
    eprintln!("Generated {} normal abilities", all_dsl.len() - normal_start);

    let abom_start = all_dsl.len();
    for seed in 0..4u64 {
        let mut runner = make_runner(seed + 2000);
        let strat = proptest::collection::vec(abomination_block(), batch..=batch);
        let tree = strat.new_tree(&mut runner).unwrap();
        all_dsl.extend(tree.current());
    }
    eprintln!("Generated {} abomination abilities", all_dsl.len() - abom_start);

    let god_start = all_dsl.len();
    for seed in 0..2u64 {
        let mut runner = make_runner(seed + 3000);
        let strat = proptest::collection::vec(god_ability_block(), batch..=batch);
        let tree = strat.new_tree(&mut runner).unwrap();
        all_dsl.extend(tree.current());
    }
    eprintln!("Generated {} god abilities", all_dsl.len() - god_start);

    eprintln!("Total generated: {}", all_dsl.len());

    let mut written = 0usize;
    let mut parse_failures = 0usize;
    let target = 75_000;

    for (i, dsl) in all_dsl.iter().enumerate() {
        let result = parse_abilities(dsl);
        let (abilities, _) = match result {
            Ok(v) => v,
            Err(_) => { parse_failures += 1; continue; }
        };
        if abilities.is_empty() { continue; }

        let base_name = &abilities[0].name;
        let unique_name = if seen_names.contains(base_name) {
            format!("{base_name}_{i}")
        } else {
            seen_names.insert(base_name.clone());
            base_name.clone()
        };

        let fixed_dsl = if unique_name != *base_name {
            let header = format!("ability {base_name} {{");
            let new_header = format!("ability {unique_name} {{");
            dsl.replacen(&header, &new_header, 1)
        } else {
            dsl.clone()
        };

        let path = out_dir.join(format!("{unique_name}.ability"));
        fs::write(&path, &fixed_dsl).unwrap();
        written += 1;

        if written >= target { break; }
    }

    eprintln!("\nDataset: {written} abilities written to {}", out_dir.display());
    if parse_failures > 0 {
        eprintln!("  ({parse_failures} parse failures skipped)");
    }

    let entries: Vec<_> = fs::read_dir(out_dir).unwrap()
        .filter_map(|e| e.ok())
        .collect();
    let sample_size = 200.min(entries.len());
    let mut sample_ok = 0;
    for entry in entries.iter().take(sample_size) {
        let content = fs::read_to_string(entry.path()).unwrap();
        if parse_abilities(&content).is_ok() {
            sample_ok += 1;
        }
    }
    eprintln!("  Sample validation: {sample_ok}/{sample_size} parsed OK");
    assert_eq!(sample_ok, sample_size, "all sampled abilities should parse");
    assert!(written >= target, "expected >={target} abilities, got {written}");
}
