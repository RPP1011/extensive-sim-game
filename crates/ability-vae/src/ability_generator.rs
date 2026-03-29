//! Ability generation from structured game state.
//!
//! Takes class, level, behavior profile, archetype, and context — produces
//! a valid ability that fits the character. No NL encoding needed.

use super::grammar_space::{self, GRAMMAR_DIM};

/// Output from ability generation.
pub struct GeneratedAbility {
    /// The grammar space vector (for further manipulation if needed).
    pub grammar_vec: [f32; GRAMMAR_DIM],
    /// The generated DSL text.
    pub dsl: String,
    /// Quality score from heuristics.
    pub quality: f32,
}

/// Structured ability generator.
///
/// Maps game state directly to grammar space vectors using learned mappings
/// for categorical decisions and deterministic rules for continuous params.
pub struct AbilityGenerator {
    // Could hold model weights in the future.
    // For now, uses rule-based mapping + grammar space.
}

impl AbilityGenerator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn generate_ability(
        &self,
        class_name_hash: u32,
        class_level: u16,
        tier: u32, // 1-7

        behavior_tags: &[u32],
        behavior_values: &[f32],

        archetype: &str,

        settlement_context: &[(u32, f32)],

        existing_abilities: &[u32],

        seed: u64,
    ) -> GeneratedAbility {
        let mut v = [0.0f32; GRAMMAR_DIM];
        let mut rng = seed;

        // --- Type: active vs passive ---
        // Higher tiers more likely to be active, but some passives at all tiers
        let passive_chance = match tier {
            1..=2 => 0.3,
            3..=4 => 0.2,
            5..=6 => 0.15,
            _ => 0.1,
        };
        v[0] = if rng_f32(&mut rng) < passive_chance { 0.75 } else { 0.25 };

        // --- Domain: combat vs campaign ---
        // Determined by archetype
        let is_campaign = matches!(archetype,
            "merchant" | "diplomat" | "spy" | "commander" | "scholar" |
            "innkeeper" | "farmer" | "blacksmith" | "alchemist" | "navigator" |
            "bard" | "cook" | "doctor" | "builder" | "thief" | "scout" |
            "priest" | "noble" | "hunter" | "sailor" | "miner"
        );
        v[1] = if is_campaign { 0.75 } else { 0.25 };

        // --- Targeting ---
        v[2] = archetype_to_targeting(archetype, is_campaign, &mut rng);

        // --- Range ---
        v[3] = archetype_to_range(archetype, &mut rng);

        // --- Cooldown: scales with tier (higher tier = longer CD = more powerful) ---
        let base_cd = match tier {
            1 => rng_range(&mut rng, 0.05, 0.25),  // 1-4s
            2 => rng_range(&mut rng, 0.15, 0.35),  // 3-8s
            3 => rng_range(&mut rng, 0.25, 0.50),  // 5-15s
            4 => rng_range(&mut rng, 0.35, 0.60),  // 8-20s
            5 => rng_range(&mut rng, 0.50, 0.75),  // 15-35s
            6 => rng_range(&mut rng, 0.65, 0.85),  // 25-50s
            _ => rng_range(&mut rng, 0.75, 0.95),  // 35-55s
        };
        v[4] = base_cd;

        // --- Cast time ---
        v[5] = rng_range(&mut rng, 0.0, 0.4);

        // --- Hint: from behavior tags ---
        v[6] = behavior_to_hint(behavior_tags, behavior_values, is_campaign, &mut rng);

        // --- Cost: scales with tier ---
        v[7] = (tier as f32 / 7.0) * rng_range(&mut rng, 0.3, 0.8);

        // --- Delivery: from archetype ---
        v[8] = archetype_to_delivery(archetype, is_campaign, &mut rng);

        // --- Delivery params ---
        v[9] = rng_range(&mut rng, 0.2, 0.8);
        v[10] = rng_range(&mut rng, 0.2, 0.8);

        // --- Number of effects: scales with tier ---
        v[11] = match tier {
            1 => 0.1,       // 1 effect
            2 => rng_range(&mut rng, 0.1, 0.4),  // 1-2
            3 => rng_range(&mut rng, 0.3, 0.5),  // 1-2
            4 => rng_range(&mut rng, 0.4, 0.7),  // 2-3
            5 => rng_range(&mut rng, 0.5, 0.8),  // 2-3
            6 => rng_range(&mut rng, 0.6, 0.9),  // 3-4
            _ => rng_range(&mut rng, 0.7, 1.0),  // 3-4
        };

        // --- Effects: from archetype + behavior + tier ---
        let effect_bases = [12, 20, 28, 36]; // D_E0_TYPE etc
        let n_effects = ((v[11] * 4.0) as usize + 1).min(4);
        for i in 0..n_effects {
            let base = effect_bases[i];
            // Effect type
            v[base] = archetype_to_effect(archetype, behavior_tags, behavior_values,
                is_campaign, i, &mut rng);
            // Param: scales with tier and level
            let power = (tier as f32 / 7.0) * (class_level as f32 / 100.0).sqrt();
            v[base + 1] = power.clamp(0.05, 0.95) * rng_range(&mut rng, 0.7, 1.3);
            v[base + 1] = v[base + 1].clamp(0.0, 1.0);
            // Duration
            v[base + 2] = rng_range(&mut rng, 0.2, 0.7);
            // Area: some archetypes prefer AoE
            v[base + 3] = if matches!(archetype, "mage" | "sorcerer" | "elementalist" | "druid")
                && rng_f32(&mut rng) < 0.4 {
                rng_range(&mut rng, 0.6, 0.9) // AoE
            } else {
                rng_range(&mut rng, 0.0, 0.4) // mostly single target
            };
            // Area param
            v[base + 4] = rng_range(&mut rng, 0.2, 0.7);
            // Element tag: from archetype/settlement context
            v[base + 5] = archetype_to_element(archetype, settlement_context, &mut rng);
            // Tag power: scales with level
            v[base + 6] = (class_level as f32 / 100.0).clamp(0.1, 0.9);
            // Condition
            v[base + 7] = if tier >= 3 && rng_f32(&mut rng) < 0.2 {
                rng_range(&mut rng, 0.6, 0.9) // conditional effect
            } else {
                rng_range(&mut rng, 0.0, 0.3) // no condition
            };
        }

        // --- Passive trigger ---
        if v[0] > 0.5 {
            v[44] = archetype_to_trigger(archetype, is_campaign, &mut rng);
            v[45] = rng_range(&mut rng, 0.2, 0.8);
        }

        // --- Scaling ---
        if tier >= 4 && rng_f32(&mut rng) < 0.3 {
            v[46] = rng_range(&mut rng, 0.3, 0.9); // some scaling stat
            v[47] = rng_range(&mut rng, 0.1, 0.5);
        }

        // --- Avoid duplicates ---
        // Use seed + existing_abilities to shift the vector slightly
        // This ensures repeated calls with same input but different existing_abilities
        // produce different results
        for &existing_hash in existing_abilities {
            let shift_dim = (existing_hash as usize) % GRAMMAR_DIM;
            v[shift_dim] = (v[shift_dim] + 0.1).fract();
        }

        // Decode and score
        let dsl = grammar_space::decode(&v);
        let quality = super::quality::score_ability(&v);

        GeneratedAbility {
            grammar_vec: v,
            dsl,
            quality,
        }
    }

    /// Generate multiple candidates and return the best one.
    pub fn generate_best(
        &self,
        class_name_hash: u32,
        class_level: u16,
        tier: u32,
        behavior_tags: &[u32],
        behavior_values: &[f32],
        archetype: &str,
        settlement_context: &[(u32, f32)],
        existing_abilities: &[u32],
        seed: u64,
        candidates: usize,
    ) -> GeneratedAbility {
        let mut best: Option<GeneratedAbility> = None;

        for i in 0..candidates {
            let candidate_seed = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            let result = self.generate_ability(
                class_name_hash, class_level, tier,
                behavior_tags, behavior_values, archetype,
                settlement_context, existing_abilities, candidate_seed,
            );

            if best.as_ref().map_or(true, |b| result.quality > b.quality) {
                best = Some(result);
            }
        }

        best.unwrap()
    }
}

// ---------------------------------------------------------------------------
// Mapping functions: game state → grammar space dimensions
// ---------------------------------------------------------------------------

fn archetype_to_targeting(archetype: &str, is_campaign: bool, rng: &mut u64) -> f32 {
    if is_campaign {
        match archetype {
            "merchant" | "alchemist" => rng_pick(rng, &[0.31, 0.56]), // market, guild
            "diplomat" | "noble" => rng_pick(rng, &[0.07, 0.81]),     // faction, global
            "spy" | "thief" | "scout" => rng_pick(rng, &[0.69, 0.07]), // self, faction
            "commander" => rng_pick(rng, &[0.44, 0.81]),               // party, global
            "doctor" | "cook" => rng_pick(rng, &[0.44, 0.69]),         // party, self
            "builder" | "miner" => rng_pick(rng, &[0.19, 0.56]),       // region, guild
            _ => rng_range(rng, 0.0, 1.0),
        }
    } else {
        match archetype {
            "warrior" | "knight" | "berserker" | "duelist" | "paladin" =>
                rng_pick(rng, &[0.07, 0.50]), // enemy, self_aoe
            "mage" | "sorcerer" | "elementalist" | "necromancer" =>
                rng_pick(rng, &[0.07, 0.64, 0.50]), // enemy, ground, self_aoe
            "ranger" | "archer" =>
                rng_pick(rng, &[0.07, 0.78]), // enemy, direction
            "cleric" | "healer" =>
                rng_pick(rng, &[0.21, 0.36, 0.50]), // ally, self, self_aoe
            "rogue" | "assassin" =>
                rng_pick(rng, &[0.07, 0.36]), // enemy, self
            "guardian" | "warden" =>
                rng_pick(rng, &[0.21, 0.50, 0.07]), // ally, self_aoe, enemy
            _ => rng_range(rng, 0.0, 0.7),
        }
    }
}

fn archetype_to_range(archetype: &str, rng: &mut u64) -> f32 {
    match archetype {
        "warrior" | "knight" | "berserker" | "duelist" | "rogue" | "assassin" | "monk" =>
            rng_range(rng, 0.05, 0.2), // melee
        "mage" | "sorcerer" | "elementalist" | "necromancer" | "archer" | "ranger" =>
            rng_range(rng, 0.5, 0.9),  // ranged
        "cleric" | "healer" | "paladin" =>
            rng_range(rng, 0.2, 0.5),  // mid
        _ => rng_range(rng, 0.1, 0.7),
    }
}

fn archetype_to_delivery(archetype: &str, is_campaign: bool, rng: &mut u64) -> f32 {
    if is_campaign { return rng_range(rng, 0.0, 0.3); } // no delivery for campaign

    match archetype {
        "mage" | "sorcerer" | "elementalist" =>
            rng_pick(rng, &[0.14, 0.5, 0.78, 0.64]), // none, projectile, zone, chain
        "ranger" | "archer" =>
            rng_pick(rng, &[0.14, 0.5]),               // none, projectile
        "rogue" | "assassin" =>
            rng_pick(rng, &[0.14, 0.92]),               // none, trap
        _ => rng_pick(rng, &[0.14, 0.14, 0.14]),        // mostly none
    }
}

fn behavior_to_hint(tags: &[u32], values: &[f32], is_campaign: bool, rng: &mut u64) -> f32 {
    // Find the dominant behavior tag
    let max_idx = values.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let n_hints = if is_campaign { 7.0 } else { 5.0 };

    // Map behavior to hint based on dominant tag
    if let Some(&tag) = tags.get(max_idx) {
        let tag_mod = tag % 10;
        match tag_mod {
            0 | 1 => (0.0 + 0.5) / n_hints, // damage/attack → damage
            2 => (1.0 + 0.5) / n_hints,       // heal
            3 => (2.0 + 0.5) / n_hints,       // cc
            4 => (3.0 + 0.5) / n_hints,       // defense
            _ => (4.0 + 0.5) / n_hints,       // utility
        }
    } else {
        rng_range(rng, 0.0, 1.0)
    }
}

fn archetype_to_effect(
    archetype: &str, _tags: &[u32], _values: &[f32],
    is_campaign: bool, effect_idx: usize, rng: &mut u64,
) -> f32 {
    if is_campaign {
        return rng_range(rng, 0.0, 1.0); // full campaign effect range
    }

    // Primary effect based on archetype
    if effect_idx == 0 {
        match archetype {
            "warrior" | "berserker" | "duelist" =>
                rng_pick(rng, &[0.08, 0.14]),           // damage, knockback
            "knight" | "guardian" | "paladin" =>
                rng_pick(rng, &[0.12, 0.08, 0.38]),     // shield, damage, taunt
            "mage" | "sorcerer" | "elementalist" =>
                rng_pick(rng, &[0.08, 0.88]),            // damage, DoT
            "cleric" | "healer" =>
                rng_pick(rng, &[0.10, 0.12, 0.92]),     // heal, shield, HoT
            "rogue" | "assassin" =>
                rng_pick(rng, &[0.08, 0.18, 0.40]),     // damage, dash, stealth
            "ranger" | "archer" =>
                rng_pick(rng, &[0.08, 0.58]),            // damage, slow
            "necromancer" =>
                rng_pick(rng, &[0.08, 0.88, 0.62]),     // damage, DoT, lifesteal
            _ => rng_range(rng, 0.0, 0.5),
        }
    } else {
        // Secondary effects: CC, buffs, mobility
        rng_pick(rng, &[0.30, 0.34, 0.58, 0.78, 0.82]) // stun, root, slow, buff, debuff
    }
}

fn archetype_to_element(
    archetype: &str, settlement: &[(u32, f32)], rng: &mut u64,
) -> f32 {
    // Base element from archetype
    let base = match archetype {
        "paladin" | "cleric" | "priest" => 0.85,         // holy
        "necromancer" => 0.75,                            // dark
        "elementalist" | "sorcerer" => rng_pick(rng, &[0.45, 0.55, 0.65, 0.75, 0.95]), // random element
        "rogue" | "assassin" => rng_pick(rng, &[0.15, 0.75, 0.95]), // none, dark, poison
        _ => rng_pick(rng, &[0.15, 0.35, 0.45]),          // none, physical, magic
    };

    // Settlement can influence element
    if let Some(&(_, influence)) = settlement.first() {
        let blend = base * 0.7 + influence * 0.3;
        blend.clamp(0.0, 1.0)
    } else {
        base
    }
}

fn archetype_to_trigger(archetype: &str, is_campaign: bool, rng: &mut u64) -> f32 {
    if is_campaign {
        return rng_pick(rng, &[0.55, 0.65, 0.75, 0.85]); // on_trade, on_quest, on_level, on_crisis
    }

    match archetype {
        "warrior" | "berserker" | "duelist" | "rogue" =>
            rng_pick(rng, &[0.05, 0.25]),    // on_damage_dealt, on_kill
        "knight" | "guardian" | "paladin" =>
            rng_pick(rng, &[0.15, 0.35]),    // on_damage_taken, on_heal
        "cleric" | "healer" =>
            rng_pick(rng, &[0.35, 0.45]),    // on_heal, on_ability_use
        "mage" | "sorcerer" =>
            rng_pick(rng, &[0.05, 0.45]),    // on_damage_dealt, on_ability_use
        _ => rng_range(rng, 0.0, 0.5),
    }
}

// ---------------------------------------------------------------------------
// RNG helpers (deterministic, fast)
// ---------------------------------------------------------------------------

fn rng_next(rng: &mut u64) -> u64 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    *rng
}

fn rng_f32(rng: &mut u64) -> f32 {
    (rng_next(rng) >> 33) as f32 / (1u64 << 31) as f32
}

fn rng_range(rng: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + rng_f32(rng) * (hi - lo)
}

fn rng_pick(rng: &mut u64, options: &[f32]) -> f32 {
    let idx = (rng_next(rng) >> 33) as usize % options.len();
    options[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tactical_sim::effects::dsl::parse_abilities;

    #[test]
    fn test_generate_validity() {
        let gen = AbilityGenerator::new();
        let archetypes = &["warrior", "mage", "cleric", "rogue", "merchant", "diplomat"];

        for (i, archetype) in archetypes.iter().enumerate() {
            for tier in 1..=7 {
                let result = gen.generate_ability(
                    i as u32 * 100 + tier,
                    (tier * 10) as u16,
                    tier,
                    &[0, 1, 2],
                    &[0.8, 0.1, 0.1],
                    archetype,
                    &[],
                    &[],
                    42 + i as u64 * 7 + tier as u64,
                );
                assert!(parse_abilities(&result.dsl).is_ok(),
                    "Failed for {} tier {}: {}", archetype, tier, result.dsl);
            }
        }
    }

    #[test]
    fn test_show_examples() {
        let gen = AbilityGenerator::new();
        let cases: &[(&str, u16, u32)] = &[
            ("warrior", 10, 2),
            ("mage", 30, 4),
            ("cleric", 20, 3),
            ("rogue", 40, 5),
            ("merchant", 25, 3),
            ("knight", 60, 6),
            ("necromancer", 50, 5),
            ("diplomat", 35, 4),
        ];
        for &(arch, level, tier) in cases {
            let result = gen.generate_best(
                0, level, tier,
                &[0, 3], &[0.7, 0.3],
                arch, &[], &[], 42, 10,
            );
            println!("=== {} L{} T{} (q={:.2}) ===", arch, level, tier, result.quality);
            println!("{}\n", result.dsl);
        }
    }

    #[test]
    fn test_generate_best() {
        let gen = AbilityGenerator::new();
        let result = gen.generate_best(
            100, 30, 4,
            &[0, 3], &[0.7, 0.3],
            "knight",
            &[],
            &[],
            42,
            20,
        );
        assert!(parse_abilities(&result.dsl).is_ok());
        assert!(result.quality > 0.0);
    }
}
