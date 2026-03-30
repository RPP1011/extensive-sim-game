//! Ability quality scoring heuristics.
//!
//! Scores a grammar space vector on coherence, balance, purpose, and variety.
//! Used to filter generated abilities — sample from grammar space, score, keep the good ones.

use super::grammar_space::*;

/// Quality score for a generated ability. Higher = better.
/// Range: 0.0 (garbage) to 1.0 (excellent).
pub fn score_ability(v: &[f32; GRAMMAR_DIM]) -> f32 {
    let mut score = 0.0f32;
    let mut penalties = 0.0f32;

    let is_passive = v[0] > 0.5;
    let is_campaign = v[1] > 0.5;

    // Targeting indices (approximate)
    let combat_targets = ["enemy", "ally", "self", "self_aoe", "ground", "direction", "global"];
    let tgt_idx = ((v[2] * 8.0) as usize).min(7);

    // Hint indices
    let hint_idx = ((v[6] * 7.0) as usize).min(6);

    // Effect type (approximate region)
    let eff_type = v[12];

    // Element
    let elem_idx = ((v[17] * 10.0) as usize).min(9);
    let has_element = elem_idx >= 3;

    // Area
    let has_area = v[15] > 0.5;

    // Delivery
    let delivery_idx = ((v[8] * 7.0) as usize).min(6);
    let has_delivery = delivery_idx >= 3; // projectile, chain, zone, trap

    // Param intensity
    let param = v[13];

    // Cooldown
    let cooldown = v[4];

    // Number of effects
    let n_effects = ((v[11] * 4.0) as usize + 1).min(4);

    // =========================================================================
    // Coherence: do the parts make sense together?
    // =========================================================================

    // Heal targeting enemy = bad
    if !is_campaign && hint_idx == 1 && tgt_idx == 0 {
        penalties += 0.3; // heal targeting enemy
    }
    // Damage targeting ally = bad
    if !is_campaign && hint_idx == 0 && tgt_idx == 1 {
        penalties += 0.3; // damage targeting ally
    }
    // Campaign effects with combat delivery = bad
    if is_campaign && has_delivery {
        penalties += 0.4;
    }
    // Element on campaign effects = questionable
    if is_campaign && has_element {
        penalties += 0.1;
    }
    // Self-target with AoE = fine (self_aoe is correct)
    // Ground target without AoE = weird
    if !is_campaign && tgt_idx == 4 && !has_area && !has_delivery {
        penalties += 0.15;
    }

    // =========================================================================
    // Balance: are the numbers reasonable?
    // =========================================================================

    // High damage + low cooldown = overpowered (penalty)
    if param > 0.8 && cooldown < 0.15 {
        penalties += 0.2;
    }
    // Low damage + high cooldown = underpowered (penalty)
    if param < 0.2 && cooldown > 0.7 && n_effects == 1 {
        penalties += 0.15;
    }
    // Many effects should have longer cooldown
    if n_effects >= 3 && cooldown < 0.2 {
        penalties += 0.1;
    }
    // Single simple effect with very long cooldown = boring
    if n_effects == 1 && cooldown > 0.8 && !is_campaign {
        penalties += 0.1;
    }
    // Good power budget: param roughly proportional to cooldown
    let power_balance = 1.0 - (param - cooldown).abs();
    score += power_balance * 0.15;

    // =========================================================================
    // Purpose: does it have a clear identity?
    // =========================================================================

    // Hint matches effect type
    if !is_campaign {
        let hint_matches = match hint_idx {
            0 => eff_type < 0.25, // damage hint → damage/shield/knockback effects
            1 => (0.08..0.15).contains(&eff_type), // heal hint → heal/shield effects
            2 => (0.28..0.55).contains(&eff_type), // cc hint → stun/root/silence/fear
            3 => (0.10..0.20).contains(&eff_type) || eff_type > 0.70, // defense → shield/buff
            4 => true, // utility = anything
            _ => true,
        };
        if hint_matches {
            score += 0.2;
        }
    }

    // Passive trigger should relate to effects
    if is_passive {
        let trigger_val = v[44];
        // on_kill (0.25) + damage effects = coherent
        // on_damage_taken (0.15) + defensive effects = coherent
        // on_heal (0.35) + heal effects = coherent
        if trigger_val < 0.2 && hint_idx == 0 { score += 0.1; } // on_damage_dealt + damage
        if (0.1..0.2).contains(&trigger_val) && hint_idx == 3 { score += 0.1; } // on_damage_taken + defense
        if (0.2..0.3).contains(&trigger_val) && hint_idx == 0 { score += 0.1; } // on_kill + damage
        if (0.3..0.4).contains(&trigger_val) && hint_idx == 1 { score += 0.1; } // on_heal + heal

        // Campaign triggers (on_trade=0.55, on_quest=0.65, on_level_up=0.75)
        // should NOT produce combat CC effects
        if trigger_val > 0.5 && !is_campaign {
            // Campaign trigger on combat ability = incoherent
            penalties += 0.3;
        }

        // Hard CC (stun, suppress, banish, charm, polymorph) on passives is OP
        if eff_type > 0.28 && eff_type < 0.55 {
            // CC effect on passive — only acceptable with long cooldown
            if cooldown < 0.3 {
                penalties += 0.2; // short-CD passive CC = broken
            }
        }
    }

    // =========================================================================
    // Tag consistency: effects in one ability should share an element theme
    // =========================================================================

    // Check if multiple effects have conflicting tags
    let effect_bases = [12usize, 20, 28, 36]; // D_E0_TYPE, D_E1_BASE, D_E2_BASE, D_E3_BASE
    let mut elem_indices = Vec::new();
    for i in 0..n_effects {
        let tag_dim = effect_bases[i] + 5; // tag dim within each effect block
        if tag_dim < GRAMMAR_DIM {
            let ei = ((v[tag_dim] * 10.0) as usize).min(9);
            if ei >= 3 { elem_indices.push(ei); } // has an element
        }
    }
    // If multiple effects have elements, they should match (or be close)
    if elem_indices.len() >= 2 {
        let all_same = elem_indices.iter().all(|&e| e == elem_indices[0]);
        if !all_same {
            penalties += 0.15; // mixed elements in one ability
        }
    }

    // Campaign effects with combat-only triggers/delivery
    if is_campaign && !is_passive {
        // Campaign cooldowns should be long (>10s at minimum)
        if cooldown < 0.1 {
            penalties += 0.1; // spammable campaign ability is weird
        }
    }

    // =========================================================================
    // Variety bonuses: reward interesting/underrepresented abilities
    // =========================================================================

    // Non-damage combat abilities
    if !is_campaign && hint_idx != 0 {
        score += 0.1;
    }
    // Campaign abilities
    if is_campaign {
        score += 0.05;
    }
    // Passives
    if is_passive {
        score += 0.05;
    }
    // Has element tag
    if has_element && !is_campaign {
        score += 0.05;
    }
    // Has area
    if has_area {
        score += 0.05;
    }
    // Multi-effect combos
    if n_effects >= 2 {
        score += 0.05;
    }
    // Has delivery
    if has_delivery && !is_campaign {
        score += 0.05;
    }
    // Reasonable range for targeting
    if !is_campaign {
        let range = v[3];
        if tgt_idx == 0 && range > 0.1 { score += 0.05; } // enemy needs some range
        if tgt_idx == 2 && range < 0.3 { score += 0.05; } // self doesn't need range
    }

    // =========================================================================
    // Conditions and advanced features (bonus)
    // =========================================================================

    let has_condition = v[19] > 0.6; // D_E0_COND
    if has_condition { score += 0.05; }

    // Clamp and return
    (score - penalties).clamp(0.0, 1.0)
}

/// Generate N abilities from grammar space, score them, return the top K.
pub fn generate_quality_abilities(
    n_candidates: usize,
    top_k: usize,
    seed: u64,
) -> Vec<([f32; GRAMMAR_DIM], f32)> {
    let mut rng = seed;
    let mut candidates: Vec<([f32; GRAMMAR_DIM], f32)> = Vec::with_capacity(n_candidates);

    for _ in 0..n_candidates {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in &mut v {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *d = (rng >> 33) as f32 / (1u64 << 31) as f32;
        }
        let s = score_ability(&v);
        candidates.push((v, s));
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);
    candidates
}

/// Generate abilities with specific feature requirements.
/// `require_feature` pins certain dimensions, then samples the rest randomly.
pub fn generate_with_feature(
    feature_dims: &[(usize, f32)], // (dim_index, value) pairs to fix
    n_candidates: usize,
    top_k: usize,
    seed: u64,
) -> Vec<([f32; GRAMMAR_DIM], f32)> {
    let mut rng = seed;
    let mut candidates: Vec<([f32; GRAMMAR_DIM], f32)> = Vec::with_capacity(n_candidates);

    for _ in 0..n_candidates {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in &mut v {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *d = (rng >> 33) as f32 / (1u64 << 31) as f32;
        }
        // Pin required features
        for &(dim, val) in feature_dims {
            if dim < GRAMMAR_DIM {
                v[dim] = val;
            }
        }
        let s = score_ability(&v);
        candidates.push((v, s));
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherent_ability_scores_higher() {
        // Coherent: damage ability targeting enemy
        let mut good = [0.3f32; GRAMMAR_DIM];
        good[0] = 0.25; // active
        good[1] = 0.25; // combat
        good[2] = 0.07; // enemy
        good[6] = 0.1;  // damage hint
        good[12] = 0.08; // damage effect

        // Incoherent: heal targeting enemy with fire tag
        let mut bad = [0.3f32; GRAMMAR_DIM];
        bad[0] = 0.25;
        bad[1] = 0.25;
        bad[2] = 0.07; // enemy
        bad[6] = 0.3;  // heal hint
        bad[12] = 0.10; // heal effect
        bad[17] = 0.55; // fire tag on heal??

        let good_score = score_ability(&good);
        let bad_score = score_ability(&bad);
        assert!(good_score > bad_score,
            "coherent ability ({:.3}) should score higher than incoherent ({:.3})",
            good_score, bad_score);
    }

    #[test]
    fn test_generate_quality() {
        let abilities = generate_quality_abilities(1000, 50, 42);
        assert_eq!(abilities.len(), 50);
        // All scores should be positive
        assert!(abilities.iter().all(|(_, s)| *s > 0.0));
        // Top score should be decent
        assert!(abilities[0].1 > 0.2, "top score too low: {}", abilities[0].1);
        // All should parse
        for (v, _) in &abilities {
            let dsl = decode(v);
            assert!(
                tactical_sim::effects::dsl::parse_abilities(&dsl).is_ok(),
                "quality ability should parse: {}", dsl
            );
        }
    }
}
