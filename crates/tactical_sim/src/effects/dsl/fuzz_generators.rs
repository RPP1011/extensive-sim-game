//! Proptest strategy generators for the ability DSL fuzzer.

use proptest::prelude::*;

// -----------------------------------------------------------------------
// Building blocks — strategies that produce DSL string fragments
// -----------------------------------------------------------------------

pub(super) fn ident_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z_][a-zA-Z0-9_]{0,15}"
}

pub(super) fn duration_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        (1u32..60_000).prop_map(|ms| format!("{ms}ms")),
        (1u32..120).prop_map(|s| format!("{s}s")),
    ]
}

pub(super) fn number_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        (0i32..999).prop_map(|n| n.to_string()),
        (0u32..999, 0u32..99).prop_map(|(i, f)| format!("{i}.{f}")),
    ]
}

pub(super) fn targeting_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("enemy"),
        Just("ally"),
        Just("self"),
        Just("self_aoe"),
        Just("ground"),
        Just("direction"),
        Just("vector"),
        Just("global"),
    ]
}

pub(super) fn hint_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("damage"),
        Just("heal"),
        Just("buff"),
        Just("utility"),
        Just("defense"),
        Just("crowd_control"),
    ]
}

pub(super) fn area_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        (1u32..100).prop_map(|r| format!("in circle({}.0)", r)),
        (1u32..100, 10u32..180).prop_map(|(r, a)| format!("in cone({}.0, {}.0)", r, a)),
        (1u32..100, 1u32..50).prop_map(|(l, w)| format!("in line({}.0, {}.0)", l, w)),
        (1u32..50, 2u32..100).prop_map(|(i, o)| format!("in ring({}.0, {}.0)", i, o)),
        (1u32..100, 1u32..10).prop_map(|(r, t)| format!("in spread({}.0, {})", r, t)),
    ]
}

fn tag_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("FIRE"),
        Just("ICE"),
        Just("LIGHTNING"),
        Just("POISON"),
        Just("DARK"),
        Just("HOLY"),
        Just("PHYSICAL"),
        Just("MAGIC"),
        Just("TRUE"),
        Just("CROWD_CONTROL"),
    ]
    .prop_flat_map(|tag| (Just(tag), 1u32..100).prop_map(|(t, v)| format!("[{t}: {v}]")))
}

pub(super) fn opt_area() -> impl Strategy<Value = String> {
    prop_oneof![3 => Just(String::new()), 1 => area_strategy()]
}

pub(super) fn opt_tags() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => Just(String::new()),
        1 => tag_strategy(),
    ]
}

pub(super) fn multi_tag_strategy() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        prop_oneof![
            Just("FIRE"), Just("ICE"), Just("LIGHTNING"), Just("POISON"),
            Just("DARK"), Just("HOLY"), Just("PHYSICAL"), Just("MAGIC"),
            Just("CROWD_CONTROL"),
        ].prop_flat_map(|tag| (Just(tag), 1u32..100).prop_map(|(t, v)| format!("{t}: {v}"))),
        1..=4,
    ).prop_map(|pairs| format!("[{}]", pairs.join(", ")))
}

// -----------------------------------------------------------------------
// Effect generators
// -----------------------------------------------------------------------

fn damage_effect() -> impl Strategy<Value = String> {
    (1i32..200, opt_area(), opt_tags()).prop_map(|(amt, area, tags)| {
        format!("    damage {amt} {area} {tags}")
    })
}

fn heal_effect() -> impl Strategy<Value = String> {
    (1i32..200).prop_map(|amt| format!("    heal {amt}"))
}

fn shield_effect() -> impl Strategy<Value = String> {
    (1i32..200, duration_strategy()).prop_map(|(amt, dur)| {
        format!("    shield {amt} for {dur}")
    })
}

fn cc_effect() -> impl Strategy<Value = String> {
    let cc_type = prop_oneof![
        Just("stun"),
        Just("root"),
        Just("silence"),
        Just("fear"),
        Just("taunt"),
        Just("charm"),
        Just("suppress"),
        Just("grounded"),
        Just("confuse"),
        Just("banish"),
        Just("polymorph"),
    ];
    (cc_type, duration_strategy(), opt_area()).prop_map(|(cc, dur, area)| {
        format!("    {cc} {dur} {area}")
    })
}

fn slow_effect() -> impl Strategy<Value = String> {
    (1u32..9, duration_strategy(), opt_area()).prop_map(|(factor_10, dur, area)| {
        let factor = factor_10 as f32 / 10.0;
        format!("    slow {factor} for {dur} {area}")
    })
}

fn knockback_effect() -> impl Strategy<Value = String> {
    (1u32..10).prop_map(|d| format!("    knockback {d}.0"))
}

fn pull_effect() -> impl Strategy<Value = String> {
    (1u32..10).prop_map(|d| format!("    pull {d}.0"))
}

fn dash_effect() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("    dash to_target".to_string()),
        Just("    dash to_position".to_string()),
        (1u32..10).prop_map(|d| format!("    dash {d}.0")),
    ]
}

fn blink_effect() -> impl Strategy<Value = String> {
    (1u32..10).prop_map(|d| format!("    blink {d}.0"))
}

fn buff_effect() -> impl Strategy<Value = String> {
    let stat = prop_oneof![
        Just("move_speed"),
        Just("attack_speed"),
        Just("damage_output"),
        Just("cooldown_reduction"),
    ];
    (stat, 1u32..9, duration_strategy()).prop_map(|(s, f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    buff {s} {f} for {dur}")
    })
}

fn debuff_effect() -> impl Strategy<Value = String> {
    let stat = prop_oneof![Just("move_speed"), Just("attack_speed"), Just("damage_output")];
    (stat, 1u32..9, duration_strategy()).prop_map(|(s, f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    debuff {s} {f} for {dur}")
    })
}

fn damage_modify_effect() -> impl Strategy<Value = String> {
    (5u32..20, duration_strategy()).prop_map(|(f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    damage_modify {f} for {dur}")
    })
}

fn lifesteal_effect() -> impl Strategy<Value = String> {
    (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    lifesteal {f} for {dur}")
    })
}

fn reflect_effect() -> impl Strategy<Value = String> {
    (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    reflect {f} for {dur}")
    })
}

fn stealth_effect() -> impl Strategy<Value = String> {
    (duration_strategy(), proptest::bool::ANY).prop_map(|(dur, bod)| {
        let suffix = if bod { " break_on_damage" } else { "" };
        format!("    stealth for {dur}{suffix}")
    })
}

fn summon_effect() -> impl Strategy<Value = String> {
    let templates = prop_oneof![
        Just("skeleton"),
        Just("minion"),
        Just("soldier"),
        Just("clone"),
    ];
    (templates, 1u32..4).prop_map(|(t, count)| {
        if count > 1 {
            format!("    summon \"{t}\" x{count}")
        } else {
            format!("    summon \"{t}\"")
        }
    })
}

fn apply_stacks_effect() -> impl Strategy<Value = String> {
    (ident_strategy(), 1u32..5, 2u32..10).prop_map(|(name, count, max)| {
        format!("    apply_stacks \"{name}\" {count} max {max}")
    })
}

fn blind_effect() -> impl Strategy<Value = String> {
    (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
        let f = f10 as f32 / 10.0;
        format!("    blind {f} for {dur}")
    })
}

fn swap_effect() -> impl Strategy<Value = String> {
    Just("    swap".to_string())
}

fn self_damage_effect() -> impl Strategy<Value = String> {
    (1i32..100).prop_map(|amt| format!("    self_damage {amt}"))
}

fn execute_effect() -> impl Strategy<Value = String> {
    (5u32..50).prop_map(|pct| format!("    execute {pct}.0"))
}

fn resurrect_effect() -> impl Strategy<Value = String> {
    (10u32..100).prop_map(|pct| format!("    resurrect {pct}.0"))
}

// -----------------------------------------------------------------------
// Condition generators (for `when` clauses)
// -----------------------------------------------------------------------

pub(super) fn simple_condition() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("target_is_stunned".to_string()),
        Just("target_is_slowed".to_string()),
        Just("target_is_rooted".to_string()),
        Just("target_is_silenced".to_string()),
        Just("target_is_feared".to_string()),
        (1u32..100).prop_map(|pct| format!("target_hp_below({pct}%)")),
        (1u32..100).prop_map(|pct| format!("caster_hp_below({pct}%)")),
        (1u32..100).prop_map(|pct| format!("target_hp_above({pct}%)")),
        (1u32..100).prop_map(|pct| format!("caster_hp_above({pct}%)")),
        (1u32..10).prop_map(|n| format!("hit_count_above({n})")),
    ]
}

fn conditional_damage() -> impl Strategy<Value = String> {
    (1i32..200, simple_condition()).prop_map(|(amt, cond)| {
        format!("    damage {amt} when {cond}")
    })
}

fn conditional_heal() -> impl Strategy<Value = String> {
    (1i32..200, simple_condition()).prop_map(|(amt, cond)| {
        format!("    heal {amt} when {cond}")
    })
}

// -----------------------------------------------------------------------
// Scaling generators
// -----------------------------------------------------------------------

pub(super) fn scaling_stat() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("target_max_hp"),
        Just("target_current_hp"),
        Just("target_missing_hp"),
        Just("caster_max_hp"),
        Just("caster_current_hp"),
        Just("caster_missing_hp"),
        Just("caster_attack_damage"),
    ]
}

fn damage_with_scaling() -> impl Strategy<Value = String> {
    (1i32..200, 1u32..100, scaling_stat()).prop_map(|(amt, pct, stat)| {
        format!("    damage {amt} + {pct}% {stat}")
    })
}

/// Pick a random effect from any category.
pub(super) fn any_effect() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => damage_effect(),
        3 => heal_effect(),
        2 => shield_effect(),
        3 => cc_effect(),
        2 => slow_effect(),
        1 => knockback_effect(),
        1 => pull_effect(),
        2 => dash_effect(),
        1 => blink_effect(),
        2 => buff_effect(),
        1 => debuff_effect(),
        1 => damage_modify_effect(),
        1 => lifesteal_effect(),
        1 => reflect_effect(),
        1 => stealth_effect(),
        1 => summon_effect(),
        1 => apply_stacks_effect(),
        1 => blind_effect(),
        1 => swap_effect(),
        1 => self_damage_effect(),
        1 => execute_effect(),
        1 => resurrect_effect(),
        2 => conditional_damage(),
        2 => conditional_heal(),
        2 => damage_with_scaling(),
    ]
}

pub(super) fn effect_list(min: usize, max: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(any_effect(), min..=max)
        .prop_map(|effects| effects.join("\n"))
}

// -----------------------------------------------------------------------
// Rich effects
// -----------------------------------------------------------------------

fn bare_area_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        (1u32..100).prop_map(|r| format!("circle({}.0)", r)),
        (1u32..100, 10u32..180).prop_map(|(r, a)| format!("cone({}.0, {}.0)", r, a)),
        (1u32..100, 1u32..50).prop_map(|(l, w)| format!("line({}.0, {}.0)", l, w)),
        (1u32..50, 2u32..100).prop_map(|(i, o)| format!("ring({}.0, {}.0)", i, o)),
        (1u32..100, 1u32..10).prop_map(|(r, t)| format!("spread({}.0, {})", r, t)),
    ]
}

fn rich_damage() -> impl Strategy<Value = String> {
    (
        1i32..200,
        bare_area_strategy(),
        multi_tag_strategy(),
        simple_condition(),
        1u32..100,
        scaling_stat(),
    ).prop_map(|(amt, area, tags, cond, pct, stat)| {
        format!("    damage {amt} in {area} {tags} + {pct}% {stat} when {cond}")
    })
}

fn rich_heal() -> impl Strategy<Value = String> {
    (1i32..200, simple_condition(), 1u32..100, scaling_stat()).prop_map(|(amt, cond, pct, stat)| {
        format!("    heal {amt} + {pct}% {stat} when {cond}")
    })
}

fn rich_cc() -> impl Strategy<Value = String> {
    let cc = prop_oneof![
        Just("stun"), Just("root"), Just("silence"), Just("fear"),
        Just("suppress"), Just("grounded"), Just("charm"), Just("banish"),
    ];
    (cc, duration_strategy(), bare_area_strategy(), multi_tag_strategy(), simple_condition())
        .prop_map(|(cc, dur, area, tags, cond)| {
            format!("    {cc} {dur} in {area} {tags} when {cond}")
        })
}

fn rich_slow() -> impl Strategy<Value = String> {
    (1u32..9, duration_strategy(), bare_area_strategy(), multi_tag_strategy())
        .prop_map(|(f10, dur, area, tags)| {
            let f = f10 as f32 / 10.0;
            format!("    slow {f} for {dur} in {area} {tags}")
        })
}

fn rich_buff() -> impl Strategy<Value = String> {
    let stat = prop_oneof![
        Just("move_speed"), Just("attack_speed"), Just("damage_output"),
    ];
    (stat, 1u32..9, duration_strategy(), bare_area_strategy())
        .prop_map(|(s, f10, dur, area)| {
            let f = f10 as f32 / 10.0;
            format!("    buff {s} {f} for {dur} in {area}")
        })
}

fn any_rich_effect() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => rich_damage(),
        2 => rich_heal(),
        2 => rich_cc(),
        2 => rich_slow(),
        2 => rich_buff(),
        1 => dash_effect(),
        1 => shield_effect(),
        1 => stealth_effect(),
        1 => summon_effect(),
        1 => lifesteal_effect(),
        1 => apply_stacks_effect(),
        1 => blind_effect(),
        1 => reflect_effect(),
        1 => execute_effect(),
    ]
}

pub(super) fn rich_effect_list(min: usize, max: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(any_rich_effect(), min..=max)
        .prop_map(|effects| effects.join("\n"))
}

pub(super) fn rich_delivery() -> impl Strategy<Value = String> {
    prop_oneof![
        (
            (1u32..30).prop_map(|s| format!("{s}.0")),
            rich_effect_list(2, 4),
            rich_effect_list(1, 3),
        ).prop_map(|(speed, hit_effs, arrival_effs)| {
            format!(
                "    deliver projectile {{ speed: {speed}, pierce, width: 0.5 }} {{\n        on_hit {{\n{hit_effs}\n        }}\n        on_arrival {{\n{arrival_effs}\n        }}\n    }}"
            )
        }),
        (duration_strategy(), duration_strategy(), rich_effect_list(2, 4))
            .prop_map(|(dur, tick, effs)| {
                format!(
                    "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effs}\n        }}\n    }}"
                )
            }),
        (duration_strategy(), duration_strategy(), rich_effect_list(2, 3))
            .prop_map(|(dur, tick, effs)| {
                format!(
                    "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effs}\n        }}\n    }}"
                )
            }),
        ((2u32..10).prop_map(|r| format!("{r}.0")), rich_effect_list(2, 4))
            .prop_map(|(range, effs)| {
                format!(
                    "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effs}\n        }}\n    }}"
                )
            }),
    ]
}

// -----------------------------------------------------------------------
// Delivery generators
// -----------------------------------------------------------------------

fn projectile_delivery() -> impl Strategy<Value = String> {
    (
        (1u32..30).prop_map(|s| format!("{s}.0")),
        proptest::bool::ANY,
        effect_list(1, 3),
    )
        .prop_map(|(speed, pierce, effects)| {
            let pierce_str = if pierce { ", pierce" } else { "" };
            format!(
                "    deliver projectile {{ speed: {speed}{pierce_str} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
            )
        })
}

fn chain_delivery() -> impl Strategy<Value = String> {
    (1u32..6, (2u32..8).prop_map(|r| format!("{r}.0")), effect_list(1, 2)).prop_map(
        |(bounces, range, effects)| {
            format!(
                "    deliver chain {{ bounces: {bounces}, range: {range} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
            )
        },
    )
}

fn zone_delivery() -> impl Strategy<Value = String> {
    (duration_strategy(), duration_strategy(), effect_list(1, 2)).prop_map(
        |(dur, tick, effects)| {
            format!(
                "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
            )
        },
    )
}

fn channel_delivery() -> impl Strategy<Value = String> {
    (duration_strategy(), duration_strategy(), effect_list(1, 2)).prop_map(
        |(dur, tick, effects)| {
            format!(
                "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
            )
        },
    )
}

fn tether_delivery() -> impl Strategy<Value = String> {
    ((2u32..10).prop_map(|r| format!("{r}.0")), effect_list(1, 2)).prop_map(
        |(range, effects)| {
            format!(
                "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effects}\n        }}\n    }}"
            )
        },
    )
}

fn trap_delivery() -> impl Strategy<Value = String> {
    (
        duration_strategy(),
        (1u32..5).prop_map(|r| format!("{r}.0")),
        effect_list(1, 2),
    )
        .prop_map(|(dur, radius, effects)| {
            format!(
                "    deliver trap {{ duration: {dur}, trigger_radius: {radius} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
            )
        })
}

pub(super) fn any_delivery() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => projectile_delivery(),
        1 => chain_delivery(),
        1 => zone_delivery(),
        1 => channel_delivery(),
        1 => tether_delivery(),
        1 => trap_delivery(),
    ]
}

// -----------------------------------------------------------------------
// Ability and passive block generators
// -----------------------------------------------------------------------

pub(super) fn ability_block() -> impl Strategy<Value = String> {
    (
        ident_strategy(),
        targeting_strategy(),
        (0u32..100).prop_map(|r| format!("{r}.0")),
        duration_strategy(),
        duration_strategy(),
        hint_strategy(),
        proptest::option::of(1u32..30),
        proptest::option::of((1u32..5, duration_strategy())),
        proptest::option::of((1u32..4, duration_strategy())),
        prop_oneof![
            3 => effect_list(1, 5).prop_map(|e| (e, None)),
            2 => (effect_list(0, 2), any_delivery().prop_map(Some)).prop_map(|(e, d)| (e, d)),
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
}

pub(super) fn trigger_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("on_damage_dealt".to_string()),
        Just("on_damage_taken".to_string()),
        Just("on_kill".to_string()),
        Just("on_death".to_string()),
        Just("on_ability_used".to_string()),
        Just("on_auto_attack".to_string()),
        Just("on_shield_broken".to_string()),
        Just("on_heal_received".to_string()),
        Just("on_dodge".to_string()),
        Just("on_reflect".to_string()),
        (1u32..100).prop_map(|pct| format!("on_hp_below({pct}%)")),
        (1u32..100).prop_map(|pct| format!("on_hp_above({pct}%)")),
        duration_strategy().prop_map(|d| format!("periodic({d})")),
    ]
}

pub(super) fn passive_block() -> impl Strategy<Value = String> {
    (
        ident_strategy(),
        trigger_strategy(),
        duration_strategy(),
        effect_list(1, 3),
    )
        .prop_map(|(name, trigger, cd, effects)| {
            format!(
                "passive {name} {{\n    trigger: {trigger}\n    cooldown: {cd}\n\n{effects}\n}}"
            )
        })
}

pub(super) fn ability_file() -> impl Strategy<Value = String> {
    (
        proptest::collection::vec(ability_block(), 1..=4),
        proptest::collection::vec(passive_block(), 0..=2),
    )
        .prop_map(|(abilities, passives)| {
            let mut parts: Vec<String> = abilities;
            parts.extend(passives);
            parts.join("\n\n")
        })
}

pub(super) fn god_ability_block() -> impl Strategy<Value = String> {
    (
        ident_strategy(),
        (1u32..100).prop_map(|r| format!("{r}.0")),
        duration_strategy(),
        duration_strategy(),
    ).prop_map(|(name, range, cd, cast)| {
        let lines = vec![
            format!("ability {name} {{"),
            format!("    target: enemy, range: {range}"),
            format!("    cooldown: {cd}, cast: {cast}"),
            "    hint: damage".to_string(),
            "    cost: 15".to_string(),
            "    charges: 3".to_string(),
            "    recharge: 8s".to_string(),
            "    recast: 2".to_string(),
            "    recast_window: 3s".to_string(),
            "    unstoppable".to_string(),
            "".to_string(),
            "    deliver projectile { speed: 10.0, pierce, width: 0.5 } {".to_string(),
            "        on_hit {".to_string(),
            "            damage 50 [FIRE: 60, ICE: 20]".to_string(),
            "            heal 30".to_string(),
            "            stun 1s in circle(3.0)".to_string(),
            "            slow 0.4 for 2s in cone(4.0, 90.0)".to_string(),
            "            root 1500ms in line(5.0, 1.5)".to_string(),
            "            silence 2s".to_string(),
            "            fear 1s".to_string(),
            "            shield 40 for 4s in ring(1.0, 5.0)".to_string(),
            "            knockback 3.0".to_string(),
            "            pull 2.0".to_string(),
            "        }".to_string(),
            "        on_arrival {".to_string(),
            "            damage 20 in spread(3.0, 3) [LIGHTNING: 40]".to_string(),
            "            blind 0.5 for 2s".to_string(),
            "            taunt 1s".to_string(),
            "            charm 1500ms".to_string(),
            "            suppress 1s".to_string(),
            "            grounded 2s".to_string(),
            "            confuse 1s".to_string(),
            "            banish 1s".to_string(),
            "            polymorph 2s".to_string(),
            "        }".to_string(),
            "    }".to_string(),
            "".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_hp_below(50%) }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_hp_above(80%) }".to_string(),
            "    on_hit_buff for 1s { heal 10 when caster_hp_below(30%) }".to_string(),
            "    on_hit_buff for 1s { heal 10 when caster_hp_above(60%) }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_is_stunned }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_is_slowed }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_is_rooted }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_is_silenced }".to_string(),
            "    on_hit_buff for 1s { damage 10 when target_is_feared }".to_string(),
            "    on_hit_buff for 1s { damage 10 when hit_count_above(3) }".to_string(),
            "    on_hit_buff for 1s { damage 10 + 15% target_max_hp }".to_string(),
            "".to_string(),
            "    dash to_target".to_string(),
            "    blink 4.0".to_string(),
            "    buff move_speed 0.3 for 5s".to_string(),
            "    debuff attack_speed 0.4 for 3s".to_string(),
            "    damage_modify 1.5 for 4s".to_string(),
            "    lifesteal 0.3 for 5s".to_string(),
            "    reflect 0.5 for 3s".to_string(),
            "    stealth for 5s break_on_damage".to_string(),
            "    summon \"golem\" x2".to_string(),
            "    apply_stacks \"doom\" 2 max 5".to_string(),
            "    swap".to_string(),
            "    self_damage 25".to_string(),
            "    execute 20.0".to_string(),
            "    resurrect 50.0".to_string(),
            "    on_hit_buff for 6s {".to_string(),
            "        damage 10 [DARK: 30]".to_string(),
            "        slow 0.2 for 1s".to_string(),
            "    }".to_string(),
            "}".to_string(),
        ];
        lines.join("\n")
    })
}

pub(super) fn abomination_block() -> impl Strategy<Value = String> {
    (
        ident_strategy(),
        targeting_strategy(),
        (1u32..100).prop_map(|r| format!("{r}.0")),
        duration_strategy(),
        duration_strategy(),
        hint_strategy(),
        1u32..30,
        (1u32..5, duration_strategy()),
        (1u32..4, duration_strategy()),
        rich_delivery(),
        rich_effect_list(3, 8),
    )
        .prop_map(
            |(name, target, range, cd, cast, hint, cost, (ch, rech), (rc, rw), delivery, effects)| {
                let mut lines = vec![format!("ability {name} {{")];
                lines.push(format!("    target: {target}, range: {range}"));
                lines.push(format!("    cooldown: {cd}, cast: {cast}"));
                lines.push(format!("    hint: {hint}"));
                lines.push(format!("    cost: {cost}"));
                lines.push(format!("    charges: {ch}"));
                lines.push(format!("    recharge: {rech}"));
                lines.push(format!("    recast: {rc}"));
                lines.push(format!("    recast_window: {rw}"));
                lines.push(String::new());
                lines.push(delivery);
                lines.push(effects);
                lines.push("}".to_string());
                lines.join("\n")
            },
        )
}
