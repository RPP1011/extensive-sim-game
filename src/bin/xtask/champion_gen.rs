//! Generate Sleeping King champion candidates with grammar-walked abilities.
//!
//! Usage: cargo run --bin xtask -- champion-gen [--seed N] [--candidates N]
//!
//! Produces 3 candidates per slot (4 combat + 3 non-combat = 7 slots = 21 candidates)
//! with 4-5 procedurally generated abilities each at tier 6-7.

use game::world_sim::ability_gen::generate_tiered_ability;
use std::collections::HashMap;

pub struct ChampionGenArgs {
    pub seed: u64,
    pub candidates_per_slot: usize,
}

struct SlotDef {
    role: &'static str,
    archetype: &'static str,
    level: u32,
    buff_type: &'static str,
    buff_desc: &'static str,
}

/// Generate the ability count and tier list for a champion at a given level.
/// Standard thresholds up to 50, then one per level after that.
fn ability_tiers_for_level(level: u32) -> Vec<u32> {
    let standard_thresholds: &[u32] = &[2, 3, 4, 5, 6, 7, 10, 13, 17, 21, 25, 30, 35, 40, 45, 50];
    let mut tiers = Vec::new();
    for &threshold in standard_thresholds {
        if threshold <= level {
            let tier = match threshold {
                0..=4 => 1,
                5..=7 => 2,
                8..=17 => 3,
                18..=35 => 4,
                36..=50 => 5,
                _ => 5,
            };
            tiers.push(tier);
        }
    }
    // One skill per level from 51 to level (TWI: every level past 50)
    for lv in 51..=level {
        let tier = match lv {
            51..=60 => 5,  // Master
            61..=70 => 6,  // Legendary
            71..=99 => 6,
            100 => 7,      // Mythic
            _ => 6,
        };
        tiers.push(tier);
    }
    tiers
}

const SLOTS: &[SlotDef] = &[
    // 4 combat slots
    SlotDef {
        role: "Combat Frontline",
        archetype: "berserker",
        level: 70,
        buff_type: "military_strength",
        buff_desc: "Raw destructive power",
    },
    SlotDef {
        role: "Combat Tank",
        archetype: "knight",
        level: 68,
        buff_type: "defense_bonus",
        buff_desc: "Unbreakable defense",
    },
    SlotDef {
        role: "Combat Assassin",
        archetype: "rogue",
        level: 72,
        buff_type: "attack_multiplier",
        buff_desc: "Lethal precision",
    },
    SlotDef {
        role: "Combat Caster",
        archetype: "mage",
        level: 69,
        buff_type: "military_strength",
        buff_desc: "Arcane devastation",
    },
    // 3 non-combat slots
    SlotDef {
        role: "Support Healer",
        archetype: "cleric",
        level: 65,
        buff_type: "recovery_rate",
        buff_desc: "Heals the king's army faster than you can damage it",
    },
    SlotDef {
        role: "Non-Combat Economy",
        archetype: "bard",
        level: 67,
        buff_type: "gold_income",
        buff_desc: "Drains regional economies",
    },
    SlotDef {
        role: "Non-Combat Political",
        archetype: "bard",
        level: 70,
        buff_type: "diplomacy_bonus",
        buff_desc: "Turns factions against the guild",
    },
];

pub fn run(args: ChampionGenArgs) {
    let mut rng = args.seed;
    let candidates = args.candidates_per_slot;

    println!("=== SLEEPING KING CHAMPION CANDIDATES ===");
    println!("Seed: {}, {} candidates per slot\n", args.seed, candidates);

    for slot in SLOTS {
        let tiers = ability_tiers_for_level(slot.level);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("SLOT: {} ({} lv{})", slot.role, slot.archetype, slot.level);
        println!("Buff: {} — {}", slot.buff_type, slot.buff_desc);
        println!("Abilities: {} (standard thresholds + 1/level past 50)", tiers.len());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        for c in 0..candidates {
            // Different history tags per candidate for variety
            let mut history = HashMap::new();
            match c % 3 {
                0 => {
                    history.insert("melee".to_string(), 500u32);
                    history.insert("tanking".to_string(), 300);
                }
                1 => {
                    history.insert("solo".to_string(), 400);
                    history.insert("near_death".to_string(), 200);
                }
                _ => {
                    history.insert("party_combat".to_string(), 500);
                    history.insert("leadership".to_string(), 300);
                }
            }

            println!("\n  ┌─ Candidate {} ({} abilities) ─────────────────", c + 1, tiers.len());

            let mut ability_names = Vec::new();
            let mut tier_counts = [0u32; 8]; // count per tier

            for (i, &tier) in tiers.iter().enumerate() {
                let (ability_def, dsl_text) = generate_tiered_ability(
                    slot.archetype,
                    tier,
                    &mut rng,
                    &history,
                );

                let name = &ability_def.name;
                ability_names.push(name.clone());
                tier_counts[tier as usize] += 1;

                // Print compactly — full DSL for T5+, just name+stats for lower tiers
                if tier >= 5 {
                    println!("  │");
                    println!("  │  ★ [{}] (T{}, cd={}s, range={:.1})",
                        name, tier,
                        ability_def.cooldown_ms / 1000,
                        ability_def.range,
                    );
                    for line in dsl_text.lines() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            println!("  │    {}", trimmed);
                        }
                    }
                } else if i < 16 {
                    // Print lower tier abilities as one-liners
                    println!("  │  [{}] (T{}, cd={}s)", name, tier, ability_def.cooldown_ms / 1000);
                }
            }

            // Summary
            println!("  │");
            print!("  │  Tier breakdown:");
            for t in 1..=7 {
                if tier_counts[t] > 0 {
                    print!(" T{}={}", t, tier_counts[t]);
                }
            }
            println!();
            println!("  │  High-tier abilities (T5+): {}",
                ability_names.iter().enumerate()
                    .filter(|(i, _)| tiers[*i] >= 5)
                    .map(|(_, n)| format!("[{}]", n))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("  └─────────────────────────────────────────────────");
        }
        println!();
    }
}
