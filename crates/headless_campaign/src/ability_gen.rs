//! Heuristic ability generator — walks the combat DSL grammar tree with
//! archetype-conditioned probability distributions to produce thematically
//! coherent, syntactically valid AbilityDef structs.
//!
//! Each grammar node decision (active/passive, targeting, effect type, etc.)
//! is sampled from distributions that depend on the archetype's semantic tags.
//! This produces unlimited diverse training data with 100% parser validity.


// ---------------------------------------------------------------------------
// Thematic naming
// ---------------------------------------------------------------------------

/// Generate a thematic ability name from mechanics + archetype.
pub fn generate_name(archetype: &str, effect_type: usize, is_passive: bool, rng: &mut u64) -> String {
    // Archetype-flavored prefixes
    let prefix = match archetype {
        "knight" | "guardian" | "paladin" | "tank" => {
            let opts = ["iron", "steel", "sentinel", "bastion", "fortress", "vanguard", "aegis", "bulwark"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "ranger" | "archer" | "scout" => {
            let opts = ["wild", "swift", "keen", "stalker", "hawk", "wind", "frontier", "pathfinder"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "mage" | "sorcerer" | "enchanter" | "caster" => {
            let opts = ["arcane", "mystic", "elder", "astral", "void", "primal", "ether", "rune"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "cleric" | "healer" | "druid" => {
            let opts = ["sacred", "blessed", "divine", "gentle", "radiant", "serene", "verdant", "dawn"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "rogue" | "assassin" => {
            let opts = ["shadow", "venom", "silent", "phantom", "dusk", "ghost", "razor", "whisper"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "berserker" | "warrior" | "fighter" => {
            let opts = ["blood", "fury", "war", "savage", "brutal", "storm", "rage", "titan"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "necromancer" | "warlock" => {
            let opts = ["dark", "soul", "grave", "blight", "hollow", "death", "hex", "curse"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        "bard" | "shaman" | "monk" | "artificer" => {
            let opts = ["echo", "spirit", "harmony", "pulse", "mantra", "chant", "rhythm", "craft"];
            opts[(xrng(rng) as usize) % opts.len()]
        }
        _ => "battle",
    };

    // Effect-derived suffixes
    let suffix = match effect_type {
        0 => { let o = ["strike", "bolt", "blast", "barrage", "rend", "cleave"]; o[(xrng(rng) as usize) % o.len()] }
        1 => { let o = ["mend", "grace", "restoration", "prayer", "salve", "balm"]; o[(xrng(rng) as usize) % o.len()] }
        2 => { let o = ["ward", "barrier", "bulwark", "aegis", "shell", "guard"]; o[(xrng(rng) as usize) % o.len()] }
        3 => { let o = ["lock", "hold", "shackle", "bind", "stasis", "freeze"]; o[(xrng(rng) as usize) % o.len()] }
        4 => { let o = ["root", "snare", "anchor", "grip", "tangle", "vine"]; o[(xrng(rng) as usize) % o.len()] }
        5 => { let o = ["silence", "hush", "void", "mute", "seal", "suppress"]; o[(xrng(rng) as usize) % o.len()] }
        6 => { let o = ["chill", "drag", "weight", "bog", "mire", "crawl"]; o[(xrng(rng) as usize) % o.len()] }
        7 => { let o = ["push", "repel", "thrust", "slam", "impact", "shove"]; o[(xrng(rng) as usize) % o.len()] }
        8 => { let o = ["rush", "leap", "charge", "dash", "surge", "flash"]; o[(xrng(rng) as usize) % o.len()] }
        9 => { let o = ["rally", "empower", "inspire", "fortify", "bolster", "charge"]; o[(xrng(rng) as usize) % o.len()] }
        10 => { let o = ["curse", "weaken", "corrode", "sap", "drain", "wither"]; o[(xrng(rng) as usize) % o.len()] }
        11 => { let o = ["call", "summon", "conjure", "invoke", "manifest", "beckon"]; o[(xrng(rng) as usize) % o.len()] }
        12 => { let o = ["veil", "cloak", "vanish", "fade", "shroud", "eclipse"]; o[(xrng(rng) as usize) % o.len()] }
        13 => { let o = ["feast", "siphon", "leech", "devour", "absorb", "thirst"]; o[(xrng(rng) as usize) % o.len()] }
        14 => { let o = ["execute", "reap", "cull", "finish", "sever", "end"]; o[(xrng(rng) as usize) % o.len()] }
        15 => { let o = ["revive", "rebirth", "renewal", "rise", "resurrect", "awaken"]; o[(xrng(rng) as usize) % o.len()] }
        17 => { let o = ["stride", "swiftness", "haste", "march", "trek", "passage"]; o[(xrng(rng) as usize) % o.len()] }
        18 => { let o = ["provision", "sustenance", "ration", "bounty", "harvest", "reserve"]; o[(xrng(rng) as usize) % o.len()] }
        19 => { let o = ["insight", "fortune", "boon", "windfall", "reward", "tithe"]; o[(xrng(rng) as usize) % o.len()] }
        20 => { let o = ["sight", "vigil", "watchtower", "survey", "recon", "farsight"]; o[(xrng(rng) as usize) % o.len()] }
        21 => { let o = ["pact", "accord", "envoy", "parley", "decree", "embassy"]; o[(xrng(rng) as usize) % o.len()] }
        22 => { let o = ["veil", "subtlety", "concealment", "discretion", "cover", "evasion"]; o[(xrng(rng) as usize) % o.len()] }
        23 => { let o = ["aura", "rite", "mandate", "charter", "doctrine", "edict"]; o[(xrng(rng) as usize) % o.len()] }
        _ => "technique",
    };

    // Passive prefix
    let passive_prefix = if is_passive {
        let opts = ["inner", "latent", "dormant", "instinct", "reflex"];
        format!("{}_", opts[(xrng(rng) as usize) % opts.len()])
    } else {
        String::new()
    };

    format!("{}{}_{}", passive_prefix, prefix, suffix)
}

/// Map an Effect to its category index for naming.
fn categorize_for_name(effect: &Effect) -> usize {
    match effect {
        Effect::Damage { .. } => 0,
        Effect::Heal { .. } => 1,
        Effect::Shield { .. } => 2,
        Effect::Stun { .. } => 3,
        Effect::Root { .. } => 4,
        Effect::Silence { .. } => 5,
        Effect::Slow { .. } => 6,
        Effect::Knockback { .. } | Effect::Pull { .. } => 7,
        Effect::Dash { .. } => 8,
        Effect::Buff { stat, .. } => {
            // Distinguish combat buffs from campaign buffs by stat name
            match stat.as_str() {
                "travel_speed" => 17,
                "supply_efficiency" => 18,
                "quest_gold_bonus" | "quest_rep_bonus" => 19,
                "scout_range" => 20,
                "diplomacy_bonus" => 21,
                "threat_reduction" => 22,
                "morale_aura" | "training_boost" | "recruit_bonus" | "passive_income" => 23,
                _ => 9, // combat buff
            }
        }
        Effect::Debuff { .. } => 10,
        Effect::Summon { .. } => 11,
        Effect::Stealth { .. } => 12,
        Effect::Lifesteal { .. } => 13,
        Effect::Execute { .. } => 14,
        Effect::Resurrect { .. } => 15,
        _ => 0,
    }
}

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
    /// Effect type weights (24 categories: 17 combat + 7 non-combat campaign)
    /// 17=travel_speed, 18=supply_efficiency, 19=quest_reward,
    /// 20=scout_range, 21=diplomacy_bonus, 22=threat_reduction,
    /// 23=passive_income/morale_aura/training_boost/recruit_bonus (campaign utility)
    effect_weights: [f32; 24],
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
            //                combat(17)                                                  | non-combat(7)
            effect_weights: [2.0, 0.5, 3.0, 2.0, 1.0, 0.3, 1.0, 1.5, 0.5, 2.0, 0.5, 0.3, 0.1, 0.3, 0.1, 0.1, 0.5,
                             0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.6],
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
            effect_weights: [4.0, 0.3, 0.5, 0.5, 0.5, 0.3, 2.0, 0.5, 1.5, 1.0, 1.0, 0.3, 1.5, 0.3, 0.5, 0.1, 0.5,
                             1.5, 0.3, 0.2, 1.5, 0.2, 0.3, 0.3],
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
            effect_weights: [3.0, 0.3, 1.0, 1.0, 0.5, 1.0, 1.5, 0.5, 0.5, 2.0, 1.5, 1.0, 0.3, 0.3, 0.3, 0.1, 1.0,
                             0.3, 0.2, 0.3, 0.5, 0.3, 0.2, 0.3],
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
            effect_weights: [0.5, 5.0, 3.0, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 3.0, 0.3, 0.3, 0.1, 0.5, 0.1, 1.0, 0.5,
                             0.2, 0.2, 0.2, 0.1, 0.3, 0.1, 1.0],
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
            effect_weights: [4.0, 0.2, 0.3, 1.0, 0.3, 0.5, 1.0, 0.5, 3.0, 0.5, 2.0, 0.2, 3.0, 1.0, 1.5, 0.1, 0.5,
                             0.3, 0.2, 1.2, 0.5, 0.2, 1.0, 0.3],
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
            effect_weights: [5.0, 0.3, 0.5, 1.0, 0.3, 0.3, 0.5, 2.0, 2.0, 1.5, 0.5, 0.3, 0.3, 2.0, 1.0, 0.1, 0.3,
                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],
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
            effect_weights: [2.0, 1.0, 0.5, 0.5, 1.0, 1.5, 0.5, 0.3, 0.3, 0.5, 2.5, 3.0, 0.5, 2.0, 0.5, 0.3, 1.0,
                             0.1, 0.2, 0.2, 0.3, 0.2, 0.5, 0.3],
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
            effect_weights: [1.5, 2.0, 1.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 3.0, 1.0, 0.5, 1.0, 0.5, 0.3, 0.3, 1.5,
                             0.5, 1.0, 0.5, 0.3, 1.5, 0.3, 1.5],
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

    // Solo play → boost self-targeting, sustain, threat reduction
    let solo = get("solo");
    if solo > 0.0 {
        let s = (solo / 10.0).min(1.5); // caps at 15 solo quests
        p.targeting[2] += s * 2.0;      // self-cast
        p.effect_weights[1] += s;        // heal
        p.effect_weights[2] += s;        // shield
        p.effect_weights[13] += s * 1.5; // lifesteal
        p.effect_weights[22] += s * 1.0; // threat_reduction
        p.passive_chance += s * 0.05;
    }

    // Party combat → boost ally targeting, aura/buff, morale
    let party = get("party_combat");
    if party > 0.0 {
        let s = (party / 10.0).min(1.5);
        p.targeting[1] += s * 2.0;       // ally
        p.targeting[3] += s;             // self_aoe
        p.effect_weights[9] += s * 2.0;  // buff
        p.effect_weights[23] += s * 1.5; // morale_aura/training (campaign utility)
        p.hint[3] += s;                  // utility
    }

    // Gather quests → boost supply efficiency
    let gather = get("gather");
    if gather > 0.0 {
        let s = (gather / 8.0).min(1.5);
        p.effect_weights[18] += s * 2.0; // supply_efficiency
        p.effect_weights[23] += s * 0.5; // passive_income (campaign utility)
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

    // Exploration → boost mobility, utility, and campaign travel/scout
    let explore = get("exploration");
    if explore > 0.0 {
        let s = (explore / 8.0).min(1.5);
        p.effect_weights[8] += s * 2.0;   // dash
        p.effect_weights[12] += s;         // stealth
        p.effect_weights[17] += s * 1.5;  // travel_speed
        p.effect_weights[20] += s * 1.5;  // scout_range
        p.hint[3] += s;                    // utility
        p.range[1] += s;                   // longer range
    }

    // Diplomatic → boost crowd control, debuffs, and diplomacy bonus
    let diplo = get("diplomatic");
    if diplo > 0.0 {
        let s = (diplo / 5.0).min(1.5);
        p.effect_weights[3] += s;          // stun
        p.effect_weights[6] += s;          // slow
        p.effect_weights[10] += s * 1.5;   // debuff
        p.effect_weights[21] += s * 2.0;   // diplomacy_bonus
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

    // Determine primary effect type for naming
    let primary_effect_type = effects.first().map(|ce| categorize_for_name(&ce.effect)).unwrap_or(0);
    let name = generate_name(archetype, primary_effect_type, is_passive, rng);

    let def = AbilityDef {
        name,
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
        requires_participants: 0,
        requires_class: String::new(),
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

        // --- Non-combat campaign effects (encoded as Buff with campaign stat names) ---
        17 => Effect::Buff {
            stat: "travel_speed".into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: if rf(rng) < 0.4 { 0 } else { ((15000 + (rf(rng) * 45000.0) as u32) / 5000) * 5000 },
        },
        18 => Effect::Buff {
            stat: "supply_efficiency".into(),
            factor: ((0.1 + rf(rng) * 0.25) * 20.0).round() / 20.0,
            duration_ms: if rf(rng) < 0.5 { 0 } else { ((20000 + (rf(rng) * 40000.0) as u32) / 5000) * 5000 },
        },
        19 => Effect::Buff {
            stat: ["quest_gold_bonus", "quest_rep_bonus"][(xrng(rng) as usize) % 2].into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: 0, // permanent (passive quest reward)
        },
        20 => Effect::Buff {
            stat: "scout_range".into(),
            factor: ((0.15 + rf(rng) * 0.35) * 20.0).round() / 20.0,
            duration_ms: if rf(rng) < 0.3 { 0 } else { ((20000 + (rf(rng) * 40000.0) as u32) / 5000) * 5000 },
        },
        21 => Effect::Buff {
            stat: "diplomacy_bonus".into(),
            factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
            duration_ms: if rf(rng) < 0.5 { 0 } else { ((30000 + (rf(rng) * 30000.0) as u32) / 5000) * 5000 },
        },
        22 => Effect::Buff {
            stat: "threat_reduction".into(),
            factor: ((0.1 + rf(rng) * 0.25) * 20.0).round() / 20.0,
            duration_ms: if rf(rng) < 0.4 { 0 } else { ((20000 + (rf(rng) * 40000.0) as u32) / 5000) * 5000 },
        },
        23 => {
            // Campaign utility grab bag: morale, training, recruit, passive income
            let stat = ["morale_aura", "training_boost", "recruit_bonus", "passive_income"]
                [(xrng(rng) as usize) % 4];
            Effect::Buff {
                stat: stat.into(),
                factor: ((0.1 + rf(rng) * 0.3) * 20.0).round() / 20.0,
                duration_ms: if rf(rng) < 0.5 { 0 } else { ((20000 + (rf(rng) * 40000.0) as u32) / 5000) * 5000 },
            }
        },

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
        targeting_filter: None,
    }
}

/// Generate N abilities for dataset creation.
/// Returns Vec of (slots, archetype, level).
///
/// NOTE: The `vae_slots` module has been removed. This function is no longer
/// functional — it returns an empty vec. Callers that need slot vectors should
/// use `grammar_space::encode` instead.
pub fn generate_batch(
    _archetypes: &[&str],
    _max_level: u32,
    _count: usize,
    _base_seed: u64,
) -> Vec<(Vec<f32>, String, u32, bool)> {
    Vec::new()
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

        let (def, is_passive) = generate_ability(archetype, level, &mut rng);
        // Name is already set by generate_ability_with_history

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
                if let Some(_brace) = dsl.find('{') {
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
            // vae_slots module removed — emit empty slots (use grammar_space::encode instead)
            let slots: Vec<f32> = Vec::new();
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

// ---------------------------------------------------------------------------
// Tiered ability generation for TWI class system (Phase 10)
// ---------------------------------------------------------------------------

/// Tier-based power multipliers for combat abilities.
/// T1 (novice) through T7 (mythic capstone).
/// Tier power multipliers. These scale base damage/heal/shield values.
/// T1 (novice) abilities do base damage. T7 (mythic/endgame) abilities
/// should one-shot or nearly one-shot level 50 heroes (HP ~1000-2000).
/// Base ability damage is ~20-30, so T7 at 50x = 1000-1500 per ability.
const TIER_POWER: [f32; 8] = [0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 50.0];
const TIER_COOLDOWN_MULT: [f32; 8] = [0.0, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0];
const TIER_MAX_EFFECTS: [usize; 8] = [0, 1, 1, 2, 2, 3, 3, 4];

/// Generate a combat ability scaled to a class tier (1-7).
///
/// Higher tiers produce:
/// - More damage/heal/shield values (scaled by TIER_POWER)
/// - Longer cooldowns (scaled by TIER_COOLDOWN_MULT)
/// - More effects per ability (TIER_MAX_EFFECTS)
/// - More area/delivery variety
///
/// The archetype determines the ability style (ranged, melee, heal, etc.)
/// and the tier scales the power budget.
pub fn generate_tiered_ability(
    archetype: &str,
    tier: u32,
    rng: &mut u64,
    history: &std::collections::HashMap<String, u32>,
) -> (AbilityDef, String) {
    let tier_idx = (tier as usize).clamp(1, 7);
    let power = TIER_POWER[tier_idx];
    let cd_mult = TIER_COOLDOWN_MULT[tier_idx];
    let max_effects = TIER_MAX_EFFECTS[tier_idx];

    // Use the base generator with a synthetic "level" that maps to tier power
    let synthetic_level = match tier_idx {
        1 => 3,
        2 => 7,
        3 => 15,
        4 => 25,
        5 => 40,
        6 => 60,
        7 => 80,
        _ => 3,
    };

    let (mut ability, _is_passive) = generate_ability_with_history(archetype, synthetic_level, rng, history);

    // Scale damage/heal/shield values by tier power multiplier
    use tactical_sim::effects::types::Effect;
    for cond_effect in &mut ability.effects {
        match &mut cond_effect.effect {
            Effect::Damage { amount, .. } => *amount = (*amount as f32 * power) as i32,
            Effect::Heal { amount, .. } => *amount = (*amount as f32 * power) as i32,
            Effect::Shield { amount, .. } => *amount = (*amount as f32 * power) as i32,
            _ => {}
        }
    }

    // Apply tier-based cooldown scaling (higher tier = longer CD = more dramatic).
    ability.cooldown_ms = (ability.cooldown_ms as f32 * cd_mult) as u32;

    // Limit effects to tier budget (higher tiers allowed more effects).
    if ability.effects.len() > max_effects {
        ability.effects.truncate(max_effects);
    }

    // Generate DSL text for the ability
    let dsl = tactical_sim::effects::dsl::emit::emit_ability_dsl(&ability);

    (ability, dsl)
}

// Roundtrip test lives in tactical_sim crate:
// crates/tactical_sim/src/effects/dsl/tests.rs::roundtrip_generated_abilities
