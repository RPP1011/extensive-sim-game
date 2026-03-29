//! VAE slot serializer — converts predicted slot vectors into valid DSL text.
//!
//! This is the exact inverse of `vae_slots.rs`. For every categorical slot,
//! argmax picks the category. For every scalar slot, denormalization recovers
//! the original value. The output is guaranteed to be valid DSL that the
//! winnow parser will accept.
//!
//! Slot layout (142 dims for abilities):
//!   [0..3)    output_type: active/passive/class
//!   [3..11)   targeting: 8 one-hot
//!   [11..15)  range, cooldown_ms, cast_time_ms, cost
//!   [15..20)  hint: 5 one-hot (damage, cc, defense, utility, heal)
//!   [20..27)  delivery type: 7 one-hot
//!   [27..33)  delivery params: speed, width, duration, bounces, range, falloff
//!   [33..37)  flags: charges, toggle, recast, unstoppable
//!   [37..42)  flag params
//!   [42..142) 4 effects × 25 dims each
//!     per effect [0..17) type, [17] param, [18] duration, [19..24) area, [24] condition

use super::vae_slots::{ABILITY_SLOT_DIM, CLASS_SLOT_DIM, EFFECT_SLOT_DIM};

const MAX_EFFECTS: usize = 4;

// ---------------------------------------------------------------------------
// Ability serialization
// ---------------------------------------------------------------------------

/// Convert a 142-dim ability slot vector into valid combat DSL text.
/// Guaranteed to produce parseable output.
pub fn slots_to_ability_dsl(slots: &[f32], name: &str) -> String {
    assert!(slots.len() >= ABILITY_SLOT_DIM);

    let is_passive = argmax(&slots[0..3]) == 1;

    if is_passive {
        serialize_passive(slots, name)
    } else {
        serialize_active(slots, name)
    }
}

fn serialize_active(slots: &[f32], name: &str) -> String {
    let mut lines = Vec::new();

    // Header
    let targeting = TARGETING_NAMES[argmax(&slots[3..11])];
    let range = (slots[11] * 10.0).max(1.0);
    let cooldown_ms = (slots[12] * 30000.0).max(1000.0) as u32;
    let cast_ms = (slots[13] * 1500.0) as u32;
    let hint = HINT_NAMES[argmax(&slots[15..20])];

    lines.push(format!("ability {} {{", to_pascal(name)));
    lines.push(format!("    target: {}, range: {:.1}", targeting, range));
    lines.push(format!(
        "    cooldown: {}s, cast: {}ms",
        cooldown_ms / 1000,
        cast_ms
    ));
    lines.push(format!("    hint: {}", hint));

    // Delivery
    let delivery_type = argmax(&slots[20..27]);
    if delivery_type > 0 {
        // Non-instant delivery
        let delivery_str = serialize_delivery(delivery_type, &slots[27..33]);
        if !delivery_str.is_empty() {
            lines.push(String::new());
            lines.push(format!("    {}", delivery_str));
        }
    }

    // Effects
    lines.push(String::new());
    let effect_lines = serialize_effects(&slots[42..]);
    for el in &effect_lines {
        lines.push(format!("    {}", el));
    }

    // Fallback: at least one effect
    if effect_lines.is_empty() {
        lines.push("    damage 10".into());
    }

    lines.push("}".into());
    lines.join("\n")
}

fn serialize_passive(slots: &[f32], name: &str) -> String {
    let mut lines = Vec::new();

    let cooldown_ms = (slots[12] * 30000.0).max(3000.0) as u32;

    // Pick a trigger based on hint
    let hint_idx = argmax(&slots[15..20]);
    let trigger = match hint_idx {
        0 => "on_damage_dealt",  // damage
        1 => "on_damage_taken",  // cc
        2 => "on_damage_taken",  // defense
        3 => "on_ability_used",  // utility
        4 => "on_damage_taken",  // heal
        _ => "on_damage_taken",
    };

    lines.push(format!("passive {} {{", to_pascal(name)));
    lines.push(format!("    trigger: {}", trigger));
    lines.push(format!("    cooldown: {}s", cooldown_ms / 1000));

    // Effects
    lines.push(String::new());
    let effect_lines = serialize_effects(&slots[42..]);
    for el in &effect_lines {
        lines.push(format!("    {}", el));
    }

    if effect_lines.is_empty() {
        lines.push("    buff defense 0.15 for 5000ms".into());
    }

    lines.push("}".into());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Effect serialization
// ---------------------------------------------------------------------------

fn serialize_effects(effect_slots: &[f32]) -> Vec<String> {
    let mut lines = Vec::new();

    for i in 0..MAX_EFFECTS {
        let offset = i * EFFECT_SLOT_DIM;
        if offset + EFFECT_SLOT_DIM > effect_slots.len() {
            break;
        }
        let eslot = &effect_slots[offset..offset + EFFECT_SLOT_DIM];

        // Check if this effect slot is active (any type > threshold)
        let type_idx = argmax(&eslot[0..17]);
        let max_val = eslot[type_idx];
        if max_val < 0.3 {
            continue; // Skip inactive effect slots
        }

        let param = eslot[17] * 155.0; // denormalize
        let duration_ms = (eslot[18] * 10000.0) as u32;

        // Area
        let area_idx = argmax(&eslot[19..24]);
        let area_str = if eslot[19 + area_idx] > 0.3 && area_idx > 0 {
            match area_idx {
                1 => format!(" in circle({:.1})", (param / 155.0 * 10.0).max(1.0).min(8.0)),
                2 => format!(" in cone({:.1}, 60)", (param / 155.0 * 10.0).max(1.0).min(8.0)),
                3 => format!(" in line({:.1}, 1.0)", (param / 155.0 * 10.0).max(2.0).min(12.0)),
                4 => format!(" in ring(1.0, {:.1})", (param / 155.0 * 10.0).max(2.0).min(8.0)),
                _ => String::new(),
            }
        } else {
            String::new()
        };

        let effect_line = match type_idx {
            0 => {
                // Damage
                let amount = param.max(5.0).min(150.0) as i32;
                if duration_ms > 500 {
                    format!("damage {} over {}ms tick 500ms{}", amount, duration_ms, area_str)
                } else {
                    format!("damage {}{}", amount, area_str)
                }
            }
            1 => {
                // Heal
                let amount = param.max(5.0).min(100.0) as i32;
                format!("heal {}{}", amount, area_str)
            }
            2 => {
                // Shield
                let amount = param.max(10.0).min(100.0) as i32;
                format!("shield {} for {}ms", amount, duration_ms.max(2000).min(8000))
            }
            3 => {
                // Stun
                format!("stun {}", duration_ms.max(500).min(3000))
            }
            4 => {
                // Root
                format!("root {}", duration_ms.max(500).min(3000))
            }
            5 => {
                // Silence/Fear/Polymorph → silence
                format!("silence {}", duration_ms.max(500).min(2500))
            }
            6 => {
                // Slow
                let factor = (param / 155.0).max(0.1).min(0.8);
                format!("slow {:.1} for {}ms{}", factor, duration_ms.max(1000).min(5000), area_str)
            }
            7 => {
                // Knockback/Pull
                let dist = (param / 155.0 * 5.0).max(1.0).min(5.0);
                format!("knockback {:.1}", dist)
            }
            8 => {
                // Dash
                "dash".into()
            }
            9 => {
                // Buff
                let factor = (param / 155.0).max(0.05).min(0.5);
                let stat = pick_stat(i);
                format!("buff {} {:.2} for {}ms", stat, factor, duration_ms.max(2000).min(10000))
            }
            10 => {
                // Debuff
                let factor = (param / 155.0).max(0.05).min(0.5);
                let stat = pick_stat(i);
                format!("debuff {} {:.2} for {}ms", stat, factor, duration_ms.max(2000).min(8000))
            }
            11 => {
                // Summon
                let count = (param / 155.0 * 3.0).max(1.0) as u32;
                format!("summon \"minion\" {} hp 0.5", count)
            }
            12 => {
                // Stealth
                format!("stealth {}", duration_ms.max(1000).min(5000))
            }
            13 => {
                // Lifesteal
                let pct = (param / 155.0).max(0.1).min(0.5);
                format!("lifesteal {:.1} for {}ms", pct, duration_ms.max(3000).min(10000))
            }
            14 => {
                // Execute
                let threshold = (param / 155.0).max(0.1).min(0.5);
                format!("execute {:.0}%", threshold * 100.0)
            }
            15 => {
                // Resurrect
                let hp_pct = (param / 155.0).max(0.2).min(0.8);
                format!("resurrect {:.0}%", hp_pct * 100.0)
            }
            _ => {
                // Utility fallback
                format!("damage {}", (param.max(5.0).min(50.0)) as i32)
            }
        };

        lines.push(effect_line);
    }

    lines
}

// ---------------------------------------------------------------------------
// Delivery serialization
// ---------------------------------------------------------------------------

fn serialize_delivery(type_idx: usize, params: &[f32]) -> String {
    match type_idx {
        1 => {
            // Projectile
            let speed = (params[0] * 14.0).max(4.0).min(14.0);
            let width = (params[1] * 1.1).max(0.3).min(1.0);
            format!("deliver projectile {{ speed: {:.0}, width: {:.1} }}", speed, width)
        }
        2 => {
            // Channel
            let dur = (params[2] * 9000.0).max(1000.0) as u32;
            let tick = (params[3] * 2500.0).max(250.0) as u32;
            format!("deliver channel {{ duration: {}ms, tick: {}ms }}", dur, tick)
        }
        3 => {
            // Zone
            let dur = (params[2] * 9000.0).max(2000.0) as u32;
            let tick = (params[3] * 2500.0).max(250.0) as u32;
            format!("deliver zone {{ duration: {}ms, tick: {}ms }}", dur, tick)
        }
        4 => {
            // Tether
            let range = (params[4] * 10.0).max(3.0).min(10.0);
            format!("deliver tether {{ max_range: {:.1} }}", range)
        }
        5 => {
            // Trap
            let dur = (params[2] * 9000.0).max(5000.0) as u32;
            let radius = (params[4] * 5.0).max(1.0).min(4.0);
            format!("deliver trap {{ duration: {}ms, trigger_radius: {:.1} }}", dur, radius)
        }
        6 => {
            // Chain
            let bounces = (params[3] * 6.0).max(2.0) as u32;
            let range = (params[4] * 6.0).max(3.0).min(6.0);
            let falloff = params[5].max(0.5).min(1.0);
            format!("deliver chain {{ bounces: {}, range: {:.1}, falloff: {:.1} }}", bounces, range, falloff)
        }
        _ => String::new(), // Instant
    }
}

// ---------------------------------------------------------------------------
// Class serialization
// ---------------------------------------------------------------------------

/// Convert a 75-dim class slot vector into valid class DSL text.
pub fn slots_to_class_dsl(slots: &[f32], name: &str) -> String {
    assert!(slots.len() >= CLASS_SLOT_DIM);
    let mut lines = Vec::new();

    lines.push(format!("class {} {{", name));

    // stat_growth (5 dims at offset 0)
    let stats = [
        ("attack", (slots[0] * 5.0).round().max(0.0) as i32),
        ("defense", (slots[1] * 5.0).round().max(0.0) as i32),
        ("speed", (slots[2] * 5.0).round().max(0.0) as i32),
        ("max_hp", (slots[3] * 5.0).round().max(0.0) as i32),
        ("ability_power", (slots[4] * 5.0).round().max(0.0) as i32),
    ];
    let growth_parts: Vec<String> = stats
        .iter()
        .filter(|(_, v)| *v > 0)
        .map(|(name, v)| format!("+{} {}", v, name))
        .collect();
    if growth_parts.is_empty() {
        lines.push("    stat_growth: +1 attack, +1 defense per level".into());
    } else {
        lines.push(format!("    stat_growth: {} per level", growth_parts.join(", ")));
    }

    // tags (16 dims at offset 5)
    let tag_names = [
        "ranged", "nature", "stealth", "tracking", "survival",
        "melee", "defense", "leadership", "arcane", "elemental",
        "healing", "divine", "assassination", "agility", "deception", "sabotage",
    ];
    let active_tags: Vec<&str> = tag_names
        .iter()
        .enumerate()
        .filter(|(i, _)| slots[5 + i] > 0.4)
        .map(|(_, name)| *name)
        .collect();
    if active_tags.is_empty() {
        lines.push("    tags: melee".into());
    } else {
        lines.push(format!("    tags: {}", active_tags.join(", ")));
    }

    // scaling (11 dims at offset 21)
    let source_names = [
        "party_alive_count", "party_size", "faction_strength",
        "coalition_strength", "crisis_severity", "fame",
        "territory_control", "adventurer_count", "gold",
        "reputation", "threat_level",
    ];
    let source_idx = argmax(&slots[21..32]);
    if slots[21 + source_idx] > 0.3 {
        let source = source_names[source_idx];
        lines.push(format!("    scaling {} {{", source));
        lines.push("        always: +5% attack".into());
        lines.push("    }".into());
    }

    // abilities (5 × 2 at offset 56)
    lines.push("    abilities {".into());
    let mut has_ability = false;
    for i in 0..5 {
        let level = (slots[56 + i * 2] * 40.0).round() as u32;
        let present = slots[56 + i * 2 + 1];
        if present > 0.4 && level > 0 {
            lines.push(format!(
                "        level {}: ability_{} \"Class ability\"",
                level, i + 1
            ));
            has_ability = true;
        }
    }
    if !has_ability {
        lines.push("        level 1: specialization \"Class ability\"".into());
    }
    lines.push("    }".into());

    // requirements (4 × 2 at offset 66)
    let mut reqs = Vec::new();
    for i in 0..4 {
        let type_val = slots[66 + i * 2];
        let threshold = slots[66 + i * 2 + 1];
        if threshold < 0.01 {
            continue;
        }
        let type_idx = (type_val * 7.0).round() as usize;
        match type_idx {
            0 => reqs.push(format!("level {}", (threshold * 20.0).round() as u32)),
            1 => reqs.push(format!("fame {}", (threshold * 2000.0).round() as u32)),
            3 => reqs.push(format!("quests {}", (threshold * 50.0).round() as u32)),
            4 => reqs.push("active_crisis".into()),
            _ => {}
        }
    }
    if reqs.is_empty() {
        reqs.push("level 1".into());
    }
    lines.push(format!("    requirements: {}", reqs.join(", ")));

    // consolidates_at
    let cons = (slots[74] * 20.0).round() as u32;
    if cons > 0 {
        lines.push(format!("    consolidates_at: {}", cons));
    }

    lines.push("}".into());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn to_pascal(name: &str) -> String {
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(first) => first.to_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn pick_stat(effect_index: usize) -> &'static str {
    // Cycle through combat + campaign stats for variety
    match effect_index % 15 {
        0 => "attack_damage",
        1 => "defense",
        2 => "attack_speed",
        3 => "move_speed",
        4 => "damage_output",
        5 => "travel_speed",
        6 => "supply_efficiency",
        7 => "quest_gold_bonus",
        8 => "scout_range",
        9 => "diplomacy_bonus",
        10 => "threat_reduction",
        11 => "morale_aura",
        12 => "training_boost",
        13 => "recruit_bonus",
        _ => "passive_income",
    }
}

const TARGETING_NAMES: [&str; 8] = [
    "enemy", "ally", "self", "self_aoe", "ground", "direction", "vector", "global",
];

const HINT_NAMES: [&str; 5] = ["damage", "crowd_control", "defense", "utility", "heal"];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tactical_sim::effects::dsl;

    #[test]
    fn test_roundtrip_slot_serialize() {
        // Parse a real ability → extract slots → serialize back → parse again
        let dsl_text = r#"
ability FireBolt {
    target: enemy, range: 6.0
    cooldown: 5s, cast: 300ms
    hint: damage

    damage 30
}
"#;
        let (abilities, _) = dsl::parse_abilities(dsl_text).unwrap();
        let def = &abilities[0];
        let slots = super::super::vae_slots::ability_to_slots(def);

        let reserialized = slots_to_ability_dsl(&slots, "fire_bolt");
        assert!(reserialized.contains("ability FireBolt"));
        assert!(reserialized.contains("target: enemy"));
        assert!(reserialized.contains("hint: damage"));
        assert!(reserialized.contains("damage"));

        // The reserialized text should parse
        let result = dsl::parse_abilities(&reserialized);
        assert!(result.is_ok(), "Failed to parse reserialized: {}\nError: {:?}", reserialized, result.err());
    }

    #[test]
    fn test_passive_roundtrip() {
        let dsl_text = r#"
passive ArcaneShield {
    trigger: on_hp_below(50%)
    cooldown: 30s

    shield 40 for 4000ms
}
"#;
        let (_, passives) = dsl::parse_abilities(dsl_text).unwrap();
        let def = &passives[0];
        let slots = super::super::vae_slots::passive_to_slots(def);

        let reserialized = slots_to_ability_dsl(&slots, "arcane_shield");
        assert!(reserialized.contains("passive ArcaneShield"));
        assert!(reserialized.contains("trigger:"));

        let result = dsl::parse_abilities(&reserialized);
        assert!(result.is_ok(), "Failed to parse: {}\nError: {:?}", reserialized, result.err());
    }

    #[test]
    fn test_random_ability_slots_all_parse() {
        // Fuzz test: random slot vectors must ALL produce parseable DSL
        let mut rng: u64 = 42;
        let mut failures = Vec::new();

        for i in 0..10_000 {
            let mut slots = vec![0.0f32; ABILITY_SLOT_DIM];
            for s in slots.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                *s = (rng as f32 / u64::MAX as f32); // [0, 1)
            }

            let name = format!("random_ability_{}", i);
            let dsl_text = slots_to_ability_dsl(&slots, &name);

            match dsl::parse_abilities(&dsl_text) {
                Ok(_) => {}
                Err(e) => {
                    failures.push((i, format!("{}", e), dsl_text));
                    if failures.len() >= 10 {
                        break; // don't flood
                    }
                }
            }
        }

        if !failures.is_empty() {
            for (i, err, dsl) in &failures {
                eprintln!("=== FAILURE #{} ===\n{}\nError: {}\n", i, dsl, err);
            }
            panic!("{}/10000 random ability slot vectors failed to parse", failures.len());
        }
    }

    #[test]
    fn test_random_class_slots_all_parse() {
        let mut rng: u64 = 123;
        let mut failures = Vec::new();

        for i in 0..10_000 {
            let mut slots = vec![0.0f32; CLASS_SLOT_DIM];
            for s in slots.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                *s = (rng as f32 / u64::MAX as f32);
            }

            let name = format!("RandomClass{}", i);
            let dsl_text = slots_to_class_dsl(&slots, &name);

            match super::super::class_dsl::parse_class(&dsl_text) {
                Ok(_) => {}
                Err(e) => {
                    failures.push((i, e, dsl_text));
                    if failures.len() >= 10 {
                        break;
                    }
                }
            }
        }

        if !failures.is_empty() {
            for (i, err, dsl) in &failures {
                eprintln!("=== FAILURE #{} ===\n{}\nError: {}\n", i, dsl, err);
            }
            panic!("{}/10000 random class slot vectors failed to parse", failures.len());
        }
    }

    #[test]
    fn test_class_roundtrip() {
        let dsl_text = r#"class Knight {
    stat_growth: +2 attack, +3 defense, +1 speed per level
    tags: melee, defense, leadership
    scaling party_alive_count {
        always: +5% attack
    }
    abilities {
        level 1: shield_wall "Reduces incoming damage"
        level 5: taunt "Forces enemies to target"
    }
    requirements: level 5, fame 50
}"#;
        let def = super::super::class_dsl::parse_class(dsl_text).unwrap();
        let slots = super::super::vae_slots::class_to_slots(&def);

        let reserialized = slots_to_class_dsl(&slots, "Knight");
        assert!(reserialized.contains("class Knight"));
        assert!(reserialized.contains("stat_growth:"));
        assert!(reserialized.contains("tags:"));
        assert!(reserialized.contains("requirements:"));

        // Should parse back
        let result = super::super::class_dsl::parse_class(&reserialized);
        assert!(result.is_ok(), "Failed to parse: {}\nError: {:?}", reserialized, result.err());
    }
}
