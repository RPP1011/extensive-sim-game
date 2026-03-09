use serde::{Deserialize, Serialize};

use crate::ai::core::{is_alive, step, IntentAction, SimState, Team, FIXED_TICK_MS};
use crate::ai::core::oracle::{run_rollout, score_rollout};
use crate::ai::effects::AbilityTarget;
use crate::ai::squad::{generate_intents, SquadAiState};

use crate::ai::effects::dsl::emit::emit_ability_dsl;

use super::categories::AbilityCategory;
use super::game_state::extract_game_state;
use super::features::{extract_damage_unit_features, extract_cc_unit_features, extract_heal_unit_features};
use super::features_aoe::{extract_damage_aoe_features, extract_simple_features, extract_summon_features, extract_obstacle_features};
use super::oracle_scoring::oracle_score_ability;

// ---------------------------------------------------------------------------
// Training sample for ability evaluator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityEvalSample {
    pub category: String,
    pub features: Vec<f32>,
    pub urgency: f32,
    /// Index into the candidate targets list (0, 1, or 2 for top-3)
    pub target_idx: u8,
    pub ability_hint: String,
    pub scenario: String,
    pub tick: u64,
    pub unit_id: u32,
    /// DSL text representation of the ability (for transformer training).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ability_dsl: Option<String>,
    /// Uniform game state features for cross-attention (70-dim entity vector).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game_state: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

/// Generate ability evaluation training samples from a simulation run.
/// At each tick, for each hero unit with ready abilities:
/// 1. Get baseline score (Hold action rollout)
/// 2. Score each ready ability via oracle rollout
/// 3. Extract per-category features + urgency label
pub fn generate_ability_eval_dataset(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    rollout_ticks: u64,
) -> Vec<AbilityEvalSample> {
    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut samples = Vec::new();

    let hero_ids: Vec<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    for _ in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        for &uid in &hero_ids {
            let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                Some(u) => u,
                None => continue,
            };

            // Skip if unit is busy
            if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                continue;
            }

            // Get baseline score (Hold)
            let (hold_ehl, hold_ahl, hold_kills, hold_cc, hold_heal) =
                run_rollout(&sim, &squad_ai, uid, &IntentAction::Hold, rollout_ticks, None);
            let baseline = score_rollout(hold_ehl, hold_ahl, hold_kills, hold_cc, hold_heal);

            // Score each ready ability
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms > 0 {
                    continue;
                }
                if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
                    continue;
                }

                let category = AbilityCategory::from_ability_full(
                    &slot.def.ai_hint,
                    &slot.def.targeting,
                    &slot.def.effects,
                    slot.def.delivery.as_ref(),
                );

                // Oracle score this ability
                let result = oracle_score_ability(
                    &sim, &squad_ai, uid, idx, baseline, rollout_ticks,
                );

                let (urgency, best_action) = match result {
                    Some((u, a)) => (u, a),
                    None => continue,
                };

                // Extract category-specific features and determine target_idx
                let (features, target_idx) = match category {
                    AbilityCategory::DamageUnit => {
                        let (feats, target_ids) = extract_damage_unit_features(&sim, unit, idx);
                        let tidx = match &best_action {
                            IntentAction::UseAbility { target: AbilityTarget::Unit(tid), .. } => {
                                target_ids.iter().position(|&id| id == *tid).unwrap_or(0) as u8
                            }
                            _ => 0,
                        };
                        (feats, tidx)
                    }
                    AbilityCategory::CcUnit => {
                        let (feats, target_ids) = extract_cc_unit_features(&sim, unit, idx);
                        let tidx = match &best_action {
                            IntentAction::UseAbility { target: AbilityTarget::Unit(tid), .. } => {
                                target_ids.iter().position(|&id| id == *tid).unwrap_or(0) as u8
                            }
                            _ => 0,
                        };
                        (feats, tidx)
                    }
                    AbilityCategory::HealUnit => {
                        let (feats, target_ids) = extract_heal_unit_features(&sim, unit, idx);
                        let tidx = match &best_action {
                            IntentAction::UseAbility { target: AbilityTarget::Unit(tid), .. } => {
                                target_ids.iter().position(|&id| id == *tid).unwrap_or(0) as u8
                            }
                            _ => 0,
                        };
                        (feats, tidx)
                    }
                    AbilityCategory::DamageAoe => {
                        let (feats, _positions) = extract_damage_aoe_features(&sim, unit, idx);
                        (feats, 0u8)
                    }
                    AbilityCategory::Summon => {
                        let feats = extract_summon_features(&sim, unit, idx);
                        (feats, 0u8)
                    }
                    AbilityCategory::Obstacle => {
                        let (feats, _positions) = extract_obstacle_features(&sim, unit, idx);
                        (feats, 0u8)
                    }
                    AbilityCategory::HealAoe | AbilityCategory::Defense | AbilityCategory::Utility => {
                        let feats = extract_simple_features(&sim, unit, idx);
                        (feats, 0u8)
                    }
                };

                samples.push(AbilityEvalSample {
                    category: category.name().to_string(),
                    features,
                    urgency,
                    target_idx,
                    ability_hint: slot.def.ai_hint.clone(),
                    scenario: scenario_name.to_string(),
                    tick: sim.tick,
                    unit_id: uid,
                    ability_dsl: Some(emit_ability_dsl(&slot.def)),
                    game_state: Some(extract_game_state(&sim, unit)),
                });
            }
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    samples
}

/// Write ability eval samples as JSONL.
pub fn write_ability_eval_dataset(
    samples: &[AbilityEvalSample],
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for sample in samples {
        serde_json::to_writer(&mut writer, sample).unwrap();
        writeln!(writer)?;
    }
    Ok(())
}
