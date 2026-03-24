//! Heuristic ability generator — walks the combat DSL grammar tree with
//! archetype-conditioned probability distributions to produce thematically
//! coherent, syntactically valid AbilityDef structs.
//!
//! Each grammar node decision (active/passive, targeting, effect type, etc.)
//! is sampled from distributions that depend on the archetype's semantic tags.
//! This produces unlimited diverse training data with 100% parser validity.

use std::collections::HashMap;

use tactical_sim::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
use tactical_sim::effects::effect_enum::Effect;
use tactical_sim::effects::types::DamageType;
use tactical_sim::effects::types::*;

/// Simple xorshift RNG.
fn xrng(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

/// Sample float in [0, 1).
fn rf(rng: &mut u64) -> f32 {
    (xrng(rng) as f32) / (u64::MAX as f32)
}

/// Sample int in [lo, hi].
fn ri(rng: &mut u64, lo: i32, hi: i32) -> i32 {
    lo + (xrng(rng) as i32).unsigned_abs() as i32 % (hi - lo + 1)
}

/// Weighted categorical sample. Returns index.
fn sample_weighted(rng: &mut u64, weights: &[f32]) -> usize {
    let total: f32 = weights.iter().sum();
    let mut target = rf(rng) * total;
    for (i, &w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Archetype profiles — probability weights for each grammar decision
// ---------------------------------------------------------------------------

struct ArchetypeProfile {
    /// P(passive) vs P(active)
    passive_chance: f32,
    /// Weights for targeting: [enemy, ally, self, self_aoe, ground, direction]
    targeting: [f32; 6],
    /// Range: [min, max]
    range: [f32; 2],
    /// Cooldown ms: [min, max]
    cooldown: [u32; 2],
    /// Hint weights: [damage, cc, defense, utility, heal]
    hint: [f32; 5],
    /// Delivery weights: [instant, projectile, channel, zone, tether, trap, chain]
    delivery: [f32; 7],
    /// Effect type weights (17 categories)
    effect_weights: [f32; 17],
    /// P(second effect)
    second_effect_chance: f32,
    /// P(has area)
    area_chance: f32,
    /// Trigger weights for passives: [on_dmg_dealt, on_dmg_taken, on_kill, on_ability_used, on_hp_below, periodic]
    trigger: [f32; 6],
}

fn profile_for_archetype(archetype: &str) -> ArchetypeProfile {
    match archetype {
        "knight" | "guardian" | "tank" | "paladin" => ArchetypeProfile {
            passive_chance: 0.3,
            targeting: [3.0, 1.0, 2.0, 1.0, 0.5, 0.5],
            range: [1.5, 3.0],
            cooldown: [8000, 18000],
            hint: [1.0, 3.0, 4.0, 1.0, 0.5],
            delivery: [4.0, 1.0, 0.5, 0.5, 1.0, 0.3, 0.2],
            effect_weights: [2.0, 0.5, 3.0, 2.0, 1.0, 0.3, 1.0, 1.5, 0.5, 2.0, 0.5, 0.3, 0.1, 0.3, 0.1, 0.1, 0.5],
            second_effect_chance: 0.4,
            area_chance: 0.3,
            trigger: [0.5, 3.0, 0.3, 0.5, 1.0, 0.5],
        },
        "ranger" | "archer" | "scout" => ArchetypeProfile {
            passive_chance: 0.25,
            targeting: [4.0, 0.5, 1.0, 0.5, 1.5, 1.5],
            range: [5.0, 8.0],
            cooldown: [5000, 12000],
            hint: [4.0, 1.0, 0.5, 2.0, 0.3],
            delivery: [2.0, 4.0, 0.5, 0.5, 0.3, 1.5, 1.0],
            effect_weights: [4.0, 0.3, 0.5, 0.5, 0.5, 0.3, 2.0, 0.5, 1.5, 1.0, 1.0, 0.3, 1.5, 0.3, 0.5, 0.1, 0.5],
            second_effect_chance: 0.35,
            area_chance: 0.2,
            trigger: [2.0, 0.5, 1.5, 1.0, 0.5, 0.5],
        },
        "mage" | "sorcerer" | "enchanter" | "caster" => ArchetypeProfile {
            passive_chance: 0.2,
            targeting: [3.0, 0.5, 1.0, 1.5, 3.0, 1.0],
            range: [5.0, 8.0],
            cooldown: [6000, 15000],
            hint: [3.0, 2.0, 0.5, 2.0, 0.3],
            delivery: [1.5, 2.0, 2.0, 3.0, 0.5, 0.3, 1.0],
            effect_weights: [3.0, 0.3, 1.0, 1.0, 0.5, 1.0, 1.5, 0.5, 0.5, 2.0, 1.5, 1.0, 0.3, 0.3, 0.3, 0.1, 1.0],
            second_effect_chance: 0.5,
            area_chance: 0.5,
            trigger: [1.0, 1.0, 0.5, 2.0, 1.0, 0.5],
        },
        "cleric" | "healer" | "druid" => ArchetypeProfile {
            passive_chance: 0.35,
            targeting: [1.0, 3.0, 2.0, 1.5, 0.5, 0.3],
            range: [3.0, 6.0],
            cooldown: [8000, 20000],
            hint: [0.5, 0.5, 1.0, 1.0, 5.0],
            delivery: [3.0, 0.5, 1.0, 1.5, 1.0, 0.2, 0.3],
            effect_weights: [0.5, 5.0, 3.0, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 3.0, 0.3, 0.3, 0.1, 0.5, 0.1, 1.0, 0.5],
            second_effect_chance: 0.4,
            area_chance: 0.35,
            trigger: [0.3, 2.0, 0.2, 1.0, 1.5, 2.0],
        },
        "rogue" | "assassin" => ArchetypeProfile {
            passive_chance: 0.3,
            targeting: [4.0, 0.3, 1.0, 0.3, 0.5, 0.5],
            range: [1.5, 4.0],
            cooldown: [4000, 10000],
            hint: [4.0, 1.0, 0.3, 1.5, 0.2],
            delivery: [3.0, 1.5, 0.3, 0.3, 0.5, 1.5, 0.5],
            effect_weights: [4.0, 0.2, 0.3, 1.0, 0.3, 0.5, 1.0, 0.5, 3.0, 0.5, 2.0, 0.2, 3.0, 1.0, 1.5, 0.1, 0.5],
            second_effect_chance: 0.45,
            area_chance: 0.15,
            trigger: [3.0, 0.5, 2.0, 1.0, 0.5, 0.3],
        },
        "berserker" | "warrior" | "fighter" => ArchetypeProfile {
            passive_chance: 0.25,
            targeting: [4.0, 0.3, 1.5, 1.0, 0.5, 0.5],
            range: [1.5, 3.0],
            cooldown: [5000, 12000],
            hint: [5.0, 1.0, 1.0, 0.5, 0.3],
            delivery: [4.0, 0.5, 0.5, 0.3, 0.3, 0.2, 0.5],
            effect_weights: [5.0, 0.3, 0.5, 1.0, 0.3, 0.3, 0.5, 2.0, 2.0, 1.5, 0.5, 0.3, 0.3, 2.0, 1.0, 0.1, 0.3],
            second_effect_chance: 0.35,
            area_chance: 0.25,
            trigger: [2.0, 1.5, 1.5, 0.5, 2.0, 0.3],
        },
        "necromancer" | "warlock" => ArchetypeProfile {
            passive_chance: 0.3,
            targeting: [3.0, 0.5, 1.0, 1.0, 2.0, 0.5],
            range: [4.0, 7.0],
            cooldown: [8000, 18000],
            hint: [2.0, 2.0, 0.5, 2.0, 1.0],
            delivery: [2.0, 1.0, 1.5, 2.0, 2.0, 0.5, 1.0],
            effect_weights: [2.0, 1.0, 0.5, 0.5, 1.0, 1.5, 0.5, 0.3, 0.3, 0.5, 2.5, 3.0, 0.5, 2.0, 0.5, 0.3, 1.0],
            second_effect_chance: 0.5,
            area_chance: 0.35,
            trigger: [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        },
        "bard" | "shaman" | "monk" | "artificer" => ArchetypeProfile {
            passive_chance: 0.35,
            targeting: [2.0, 2.0, 2.0, 1.5, 1.0, 0.5],
            range: [3.0, 6.0],
            cooldown: [6000, 15000],
            hint: [1.0, 1.0, 1.0, 4.0, 1.5],
            delivery: [3.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            effect_weights: [1.5, 2.0, 1.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 3.0, 1.0, 0.5, 1.0, 0.5, 0.3, 0.3, 1.5],
            second_effect_chance: 0.5,
            area_chance: 0.3,
            trigger: [1.0, 1.0, 0.5, 2.0, 0.5, 2.0],
        },
        _ => profile_for_archetype("warrior"),
    }
}

// ---------------------------------------------------------------------------
// History tag biasing
// ---------------------------------------------------------------------------

/// Apply history tag biases to an archetype profile.
/// Each tag shifts specific probability weights to make abilities reflect gameplay.
fn apply_history_biases(p: &mut ArchetypeProfile, history: &std::collections::HashMap<String, u32>) {
    let get = |key: &str| -> f32 { *history.get(key).unwrap_or(&0) as f32 };

    // Solo play → boost self-targeting, sustain effects
    let solo = get("solo");
    if solo > 0.0 {
        let s = (solo / 10.0).min(1.5); // caps at 15 solo quests
        p.targeting[2] += s * 2.0;      // self-cast
        p.effect_weights[1] += s;        // heal
        p.effect_weights[2] += s;        // shield
        p.effect_weights[13] += s * 1.5; // lifesteal
        p.passive_chance += s * 0.05;
    }

    // Party combat → boost ally targeting, aura/buff effects
    let party = get("party_combat");
    if party > 0.0 {
        let s = (party / 10.0).min(1.5);
        p.targeting[1] += s * 2.0;       // ally
        p.targeting[3] += s;             // self_aoe
        p.effect_weights[9] += s * 2.0;  // buff
        p.hint[3] += s;                  // utility
    }

    // Near death → boost defensive/survival
    let near_death = get("near_death");
    if near_death > 0.0 {
        let s = (near_death / 5.0).min(2.0);
        p.effect_weights[2] += s * 2.0;  // shield
        p.effect_weights[1] += s;         // heal
        p.effect_weights[8] += s * 0.5;  // dash (escape)
        p.hint[2] += s;                   // defense
        p.passive_chance += s * 0.1;
        p.trigger[4] += s * 2.0;          // on_hp_below
    }

    // High threat quests → boost damage scaling and CC
    let high_threat = get("high_threat");
    if high_threat > 0.0 {
        let s = (high_threat / 8.0).min(1.5);
        p.effect_weights[0] += s;         // damage
        p.effect_weights[3] += s * 0.5;   // stun
        p.effect_weights[14] += s * 0.5;  // execute
    }

    // Exploration → boost mobility and utility
    let explore = get("exploration");
    if explore > 0.0 {
        let s = (explore / 8.0).min(1.5);
        p.effect_weights[8] += s * 2.0;   // dash
        p.effect_weights[12] += s;         // stealth
        p.hint[3] += s;                    // utility
        p.range[1] += s;                   // longer range
    }

    // Diplomatic → boost crowd control and debuffs
    let diplo = get("diplomatic");
    if diplo > 0.0 {
        let s = (diplo / 5.0).min(1.5);
        p.effect_weights[3] += s;          // stun
        p.effect_weights[6] += s;          // slow
        p.effect_weights[10] += s * 1.5;   // debuff
        p.hint[1] += s;                    // crowd_control
    }

    // Crisis: blight → boost nature/purification/resistance
    let blight = get("crisis_blight_prevention");
    if blight > 0.0 {
        let s = (blight / 3.0).min(2.0);
        p.effect_weights[1] += s;          // heal
        p.effect_weights[2] += s;          // shield
        p.effect_weights[9] += s;          // buff (resistance)
        p.hint[2] += s;                    // defense
        p.delivery[3] += s;               // zone (purification area)
    }

    // Crisis: breach defense → boost AoE, fortification
    let breach = get("crisis_breach_defense");
    if breach > 0.0 {
        let s = (breach / 3.0).min(2.0);
        p.area_chance += s * 0.15;
        p.effect_weights[0] += s;          // damage (AoE)
        p.effect_weights[3] += s;          // stun (crowd control)
        p.delivery[3] += s;               // zone
        p.hint[1] += s;                    // crowd_control
    }

    // Crisis: sleeping king → boost leadership, rally
    let king = get("crisis_sleeping_king");
    if king > 0.0 {
        let s = (king / 3.0).min(2.0);
        p.effect_weights[9] += s * 2.0;   // buff (rally)
        p.targeting[3] += s;              // self_aoe (aura)
        p.hint[3] += s;                   // utility
    }

    // Region defense → boost fortification effects
    let region_def = get("region_defense");
    if region_def > 0.0 {
        let s = (region_def / 8.0).min(1.5);
        p.effect_weights[2] += s;          // shield
        p.effect_weights[7] += s;          // knockback (repel)
        p.delivery[5] += s;               // trap
    }

    // Rescue quests → boost healing
    let rescue = get("rescue");
    if rescue > 0.0 {
        let s = (rescue / 5.0).min(2.0);
        p.effect_weights[1] += s * 2.0;   // heal
        p.effect_weights[15] += s;         // resurrect
        p.targeting[1] += s;              // ally
        p.hint[4] += s;                   // heal hint
    }
}

// ---------------------------------------------------------------------------
// Ability generation
// ---------------------------------------------------------------------------

/// Generate a random ability for the given archetype, level, and history.
/// History tags bias the probability distributions so abilities reflect
/// the adventurer's actual journey.
pub fn generate_ability(
    archetype: &str,
    level: u32,
    rng: &mut u64,
) -> (AbilityDef, bool) {
    generate_ability_with_history(archetype, level, rng, &std::collections::HashMap::new())
}

/// Generate with history tag biasing.
pub fn generate_ability_with_history(
    archetype: &str,
    level: u32,
    rng: &mut u64,
    history: &std::collections::HashMap<String, u32>,
) -> (AbilityDef, bool) {
    let mut p = profile_for_archetype(archetype);
    apply_history_biases(&mut p, history);
    let is_passive = rf(rng) < p.passive_chance;

    // Scale params by level
    let level_scale = 1.0 + (level as f32 - 1.0) * 0.1;

    let targeting_idx = sample_weighted(rng, &p.targeting);
    let targeting = match targeting_idx {
        0 => AbilityTargeting::TargetEnemy,
        1 => AbilityTargeting::TargetAlly,
        2 => AbilityTargeting::SelfCast,
        3 => AbilityTargeting::SelfAoe,
        4 => AbilityTargeting::GroundTarget,
        _ => AbilityTargeting::Direction,
    };

    let range = ((p.range[0] + rf(rng) * (p.range[1] - p.range[0])) * 2.0).round() / 2.0; // round to 0.5
    let cooldown_ms = p.cooldown[0] + (rf(rng) * (p.cooldown[1] - p.cooldown[0]) as f32) as u32;
    let cooldown_ms = (cooldown_ms / 1000) * 1000; // round to seconds

    let hint_idx = sample_weighted(rng, &p.hint);
    let hint = ["damage", "crowd_control", "defense", "utility", "heal"][hint_idx];

    // Generate 1-3 effects
    let mut effects = vec![gen_effect(&p, level_scale, rng)];
    if rf(rng) < p.second_effect_chance {
        effects.push(gen_effect(&p, level_scale, rng));
    }
    if rf(rng) < p.second_effect_chance * 0.3 {
        effects.push(gen_effect(&p, level_scale, rng));
    }

    // Delivery
    let delivery = if !is_passive {
        let di = sample_weighted(rng, &p.delivery);
        match di {
            0 => None, // instant
            1 => Some(Delivery::Projectile {
                speed: (6.0 + rf(rng) * 8.0).round(),
                pierce: rf(rng) < 0.2,
                width: ((0.3 + rf(rng) * 0.7) * 4.0).round() / 4.0,
                on_hit: vec![],
                on_arrival: vec![],
            }),
            2 => Some(Delivery::Channel {
                duration_ms: 1500 + (rf(rng) * 3000.0) as u32,
                tick_interval_ms: 250 + (rf(rng) * 500.0) as u32,
            }),
            3 => Some(Delivery::Zone {
                duration_ms: 3000 + (rf(rng) * 5000.0) as u32,
                tick_interval_ms: 500 + (rf(rng) * 500.0) as u32,
            }),
            4 => Some(Delivery::Tether {
                max_range: 4.0 + rf(rng) * 4.0,
                tick_interval_ms: 500,
                on_complete: vec![],
            }),
            5 => Some(Delivery::Trap {
                duration_ms: 10000 + (rf(rng) * 20000.0) as u32,
                trigger_radius: 1.5 + rf(rng) * 2.0,
                arm_time_ms: 500 + (rf(rng) * 1000.0) as u32,
            }),
            _ => Some(Delivery::Chain {
                bounces: 2 + (rf(rng) * 4.0) as u32,
                bounce_range: 4.0 + rf(rng) * 2.0,
                falloff: 0.7 + rf(rng) * 0.2,
                on_hit: vec![],
            }),
        }
    } else {
        None
    };

    let def = AbilityDef {
        name: format!("gen_{}", xrng(rng) % 100000),
        targeting,
        range,
        cooldown_ms,
        cast_time_ms: if is_passive { 0 } else { (rf(rng) * 500.0) as u32 },
        ai_hint: hint.into(),
        effects,
        delivery,
        resource_cost: 0,
        morph_into: None,
        morph_duration_ms: 0,
        zone_tag: None,
        max_charges: 0,
        charge_recharge_ms: 0,
        is_toggle: false,
        toggle_cost_per_sec: 0.0,
        recast_count: 0,
        recast_window_ms: 0,
        recast_effects: vec![],
        unstoppable: rf(rng) < 0.05,
        swap_form: None,
        form: None,
        evolve_into: None,
    };

    (def, is_passive)
}

/// Generate a single ConditionalEffect from the archetype profile.
fn gen_effect(p: &ArchetypeProfile, level_scale: f32, rng: &mut u64) -> ConditionalEffect {
    let etype = sample_weighted(rng, &p.effect_weights);

    let effect = match etype {
        0 => Effect::Damage {
            amount: (10.0 * level_scale + rf(rng) * 40.0 * level_scale) as i32,
            amount_per_tick: 0,
            duration_ms: 0,
            tick_interval_ms: 0,
            scaling_stat: None,
            scaling_percent: 0.0,
            damage_type: DamageType::Physical,
            bonus: vec![],
        },
        1 => Effect::Heal {
            amount: (8.0 * level_scale + rf(rng) * 30.0 * level_scale) as i32,
            amount_per_tick: 0,
            duration_ms: 0,
            tick_interval_ms: 0,
            scaling_stat: None,
            scaling_percent: 0.0,
            bonus: vec![],
        },
        2 => Effect::Shield {
            amount: (15.0 * level_scale + rf(rng) * 25.0 * level_scale) as i32,
            duration_ms: 3000 + (rf(rng) * 5000.0) as u32,
        },
        3 => Effect::Stun { duration_ms: 500 + (rf(rng) * 2000.0) as u32 },
        4 => Effect::Root { duration_ms: 500 + (rf(rng) * 2000.0) as u32 },
        5 => Effect::Silence { duration_ms: 500 + (rf(rng) * 2000.0) as u32 },
        6 => Effect::Slow {
            factor: ((0.2 + rf(rng) * 0.5) * 20.0).round() / 20.0, // round to 0.05
            duration_ms: ((1000 + (rf(rng) * 3000.0) as u32) / 250) * 250, // round to 250ms
        },
        7 => Effect::Knockback { distance: ((1.0 + rf(rng) * 3.0) * 2.0).round() / 2.0 },
        8 => Effect::Dash {
            to_target: rf(rng) < 0.5,
            distance: ((3.0 + rf(rng) * 4.0) * 2.0).round() / 2.0,
            to_position: false,
            is_blink: rf(rng) < 0.2,
        },
        9 => Effect::Buff {
            stat: ["attack_damage", "defense", "attack_speed", "move_speed", "damage_output"]
                [(xrng(rng) as usize) % 5].into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: ((3000 + (rf(rng) * 7000.0) as u32) / 500) * 500,
        },
        10 => Effect::Debuff {
            stat: ["attack_damage", "defense", "attack_speed", "move_speed"]
                [(xrng(rng) as usize) % 4].into(),
            factor: ((0.1 + rf(rng) * 0.25) * 20.0).round() / 20.0,
            duration_ms: ((2000 + (rf(rng) * 5000.0) as u32) / 500) * 500,
        },
        11 => Effect::Summon {
            template: "minion".into(),
            count: 1 + (rf(rng) * 2.0) as u32,
            hp_percent: 0.3 + rf(rng) * 0.5,
            clone: false,
            clone_damage_percent: 0.0,
            directed: false,
        },
        12 => Effect::Stealth {
            duration_ms: 2000 + (rf(rng) * 4000.0) as u32,
            break_on_damage: rf(rng) < 0.6,
            break_on_ability: rf(rng) < 0.3,
        },
        13 => Effect::Lifesteal {
            percent: 0.1 + rf(rng) * 0.3,
            duration_ms: 3000 + (rf(rng) * 5000.0) as u32,
        },
        14 => Effect::Execute { hp_threshold_percent: 0.15 + rf(rng) * 0.25 },
        15 => Effect::Resurrect { hp_percent: 0.3 + rf(rng) * 0.4 },
        _ => Effect::Damage {
            amount: (15.0 * level_scale) as i32,
            amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0,
            scaling_stat: None, scaling_percent: 0.0,
            damage_type: DamageType::Physical, bonus: vec![],
        },
    };

    // Optional area (round to 0.5 for clean DSL output)
    let area = if rf(rng) < p.area_chance {
        let ai = (xrng(rng) as usize) % 5;
        let r2 = |v: f32| (v * 2.0).round() / 2.0;
        Some(match ai {
            0 => Area::SingleTarget,
            1 => Area::Circle { radius: r2(1.5 + rf(rng) * 3.0) },
            2 => Area::Cone { radius: r2(3.0 + rf(rng) * 3.0), angle_deg: (40.0 + rf(rng) * 50.0).round() },
            3 => Area::Line { length: r2(4.0 + rf(rng) * 6.0), width: r2(0.5 + rf(rng) * 1.0) },
            _ => Area::Spread { radius: r2(3.0 + rf(rng) * 3.0), max_targets: 2 + (rf(rng) * 3.0) as u32 },
        })
    } else {
        None
    };

    ConditionalEffect {
        effect,
        condition: None,
        area,
        tags: Tags::new(),
        stacking: Stacking::Refresh,
        chance: 1.0,
        else_effects: vec![],
    }
}

/// Generate N abilities for dataset creation.
/// Returns Vec of (slots, archetype, level).
pub fn generate_batch(
    archetypes: &[&str],
    max_level: u32,
    count: usize,
    base_seed: u64,
) -> Vec<(Vec<f32>, String, u32, bool)> {
    use super::vae_slots;

    let mut rng = base_seed;
    let mut results = Vec::with_capacity(count);

    for _ in 0..count {
        let arch_idx = (xrng(&mut rng) as usize) % archetypes.len();
        let archetype = archetypes[arch_idx];
        let level = 1 + (xrng(&mut rng) % max_level as u64) as u32;

        let (def, is_passive) = generate_ability(archetype, level, &mut rng);

        let slots = if is_passive {
            // Convert to PassiveDef for slot extraction
            let p = &profile_for_archetype(archetype);
            let trigger_idx = sample_weighted(&mut rng, &p.trigger);
            let trigger = match trigger_idx {
                0 => Trigger::OnDamageDealt,
                1 => Trigger::OnDamageTaken,
                2 => Trigger::OnKill,
                3 => Trigger::OnAbilityUsed,
                4 => Trigger::OnHpBelow { percent: 0.3 + rf(&mut rng) * 0.3 },
                _ => Trigger::Periodic { interval_ms: 3000 + (rf(&mut rng) * 5000.0) as u32 },
            };
            let passive = PassiveDef {
                name: def.name.clone(),
                trigger,
                cooldown_ms: def.cooldown_ms,
                effects: def.effects.clone(),
                range: def.range,
            };
            vae_slots::passive_to_slots(&passive)
        } else {
            vae_slots::ability_to_slots(&def)
        };

        results.push((slots, archetype.to_string(), level, is_passive));
    }

    results
}

/// Generate synthetic abilities and write to stdout.
/// With `emit_dsl=true`, outputs the actual DSL text instead of slot vectors.
pub fn dump_synthetic(count: usize, seed: u64, emit_dsl: bool) {
    use std::io::Write;
    use tactical_sim::effects::dsl::emit::emit_ability_dsl;

    let archetypes = [
        "knight", "ranger", "mage", "cleric", "rogue",
        "paladin", "berserker", "necromancer", "bard", "druid",
        "warlock", "monk", "assassin", "guardian", "shaman",
        "artificer", "tank", "warrior", "fighter",
    ];

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut rng = seed;

    for _ in 0..count {
        let arch_idx = (xrng(&mut rng) as usize) % archetypes.len();
        let archetype = archetypes[arch_idx];
        let level = 1 + (xrng(&mut rng) % 40) as u32;

        let (mut def, is_passive) = generate_ability(archetype, level, &mut rng);
        def.name = format!("{}_L{}", archetype, level);

        if emit_dsl {
            let dsl = if is_passive {
                let p = &profile_for_archetype(archetype);
                let trigger_idx = sample_weighted(&mut rng, &p.trigger);
                let trigger = match trigger_idx {
                    0 => Trigger::OnDamageDealt,
                    1 => Trigger::OnDamageTaken,
                    2 => Trigger::OnKill,
                    3 => Trigger::OnAbilityUsed,
                    4 => Trigger::OnHpBelow { percent: 0.3 + rf(&mut rng) * 0.3 },
                    _ => Trigger::Periodic { interval_ms: 3000 + (rf(&mut rng) * 5000.0) as u32 },
                };
                let passive = tactical_sim::effects::defs::PassiveDef {
                    name: def.name.clone(),
                    trigger,
                    cooldown_ms: def.cooldown_ms,
                    effects: def.effects.clone(),
                    range: def.range,
                };
                // No emit_passive_dsl in the crate, format manually
                let trigger_str = match &passive.trigger {
                    Trigger::OnDamageDealt => "on_damage_dealt".into(),
                    Trigger::OnDamageTaken => "on_damage_taken".into(),
                    Trigger::OnKill => "on_kill".into(),
                    Trigger::OnAbilityUsed => "on_ability_used".into(),
                    Trigger::OnHpBelow { percent } => format!("on_hp_below({:.0}%)", percent * 100.0),
                    Trigger::Periodic { interval_ms } => format!("periodic({}ms)", interval_ms),
                    _ => "on_damage_taken".into(),
                };
                // Emit as ability but change header to passive
                let mut dsl = emit_ability_dsl(&def);
                // Replace "ability Name {" with "passive Name {"
                if let Some(brace) = dsl.find('{') {
                    let header = format!("passive {} {{\n    trigger: {}\n    cooldown: {}s\n",
                        def.name, trigger_str, passive.cooldown_ms / 1000);
                    // Find end of original header (after hint line)
                    let lines: Vec<&str> = dsl.lines().collect();
                    let body_start = lines.iter().position(|l| l.trim().is_empty())
                        .unwrap_or(4);
                    let body = lines[body_start..].join("\n");
                    dsl = format!("{}{}", header, body);
                }
                dsl
            } else {
                emit_ability_dsl(&def)
            };
            writeln!(out, "// {} L{}", archetype, level).ok();
            writeln!(out, "{}", dsl).ok();
            writeln!(out).ok();
        } else {
            let slots = if is_passive {
                let passive = tactical_sim::effects::defs::PassiveDef {
                    name: def.name.clone(),
                    trigger: Trigger::OnDamageTaken,
                    cooldown_ms: def.cooldown_ms,
                    effects: def.effects,
                    range: def.range,
                };
                super::vae_slots::passive_to_slots(&passive)
            } else {
                super::vae_slots::ability_to_slots(&def)
            };
            let record = serde_json::json!({
                "slots": slots,
                "archetype": archetype,
                "level": level,
                "is_passive": is_passive,
            });
            writeln!(out, "{}", record).ok();
        }
    }

    eprintln!("Generated {} synthetic abilities", count);
}
