use crate::ai::core::{distance, is_alive, SimState, IntentAction, UnitState};
use crate::ai::effects::{AbilityTarget, AbilityTargeting};
use crate::ai::squad::SquadAiState;

use super::categories::{AbilityCategory, URGENCY_THRESHOLD};
use super::features::{extract_damage_unit_features, extract_cc_unit_features, extract_heal_unit_features, is_healer};
use super::features_aoe::{extract_damage_aoe_features, extract_simple_features, extract_summon_features, extract_obstacle_features};
use super::weights::AbilityEvalWeights;

// ---------------------------------------------------------------------------
// Runtime: evaluate all ready abilities for a unit
// ---------------------------------------------------------------------------

/// Evaluate all ready abilities for a unit and return the best one above threshold.
pub fn evaluate_abilities(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    weights: &AbilityEvalWeights,
) -> Option<(IntentAction, f32)> {
    evaluate_abilities_with_encoder(state, squad_ai, unit_id, weights, None)
}

/// Evaluate abilities with an optional encoder for embedding-enriched features.
pub fn evaluate_abilities_with_encoder(
    state: &SimState,
    _squad_ai: &SquadAiState,
    unit_id: u32,
    weights: &AbilityEvalWeights,
    encoder: Option<&crate::ai::core::ability_encoding::AbilityEncoder>,
) -> Option<(IntentAction, f32)> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;

    let mut best_urgency = 0.0f32;
    let mut best_action = None;

    for (idx, slot) in unit.abilities.iter().enumerate() {
        // Skip unavailable abilities
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

        let eval_weights = match weights.evaluators.get(&category) {
            Some(w) => w,
            None => continue,
        };

        // Optionally compute ability embedding
        let embedding: Option<[f32; 32]> = encoder.map(|enc| enc.encode_def(&slot.def));

        // Extract features and predict
        let (urgency, action) = match category {
            AbilityCategory::DamageUnit => {
                let (mut features, target_ids) = extract_damage_unit_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                let target_idx = output[1..].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let target_id = target_ids.get(target_idx).copied().unwrap_or(0);
                if target_id == 0 { continue; }
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target: AbilityTarget::Unit(target_id),
                })
            }
            AbilityCategory::CcUnit => {
                let (mut features, target_ids) = extract_cc_unit_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                let target_idx = output[1..].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let target_id = target_ids.get(target_idx).copied().unwrap_or(0);
                if target_id == 0 { continue; }
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target: AbilityTarget::Unit(target_id),
                })
            }
            AbilityCategory::HealUnit => {
                let (mut features, target_ids) = extract_heal_unit_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                let target_idx = output[1..].iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let target_id = target_ids.get(target_idx).copied().unwrap_or(0);
                if target_id == 0 { continue; }
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target: AbilityTarget::Unit(target_id),
                })
            }
            AbilityCategory::DamageAoe => {
                let (mut features, positions) = extract_damage_aoe_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                let pos_idx = if output.len() > 1 {
                    output[1..].iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                } else { 0 };
                let target = if let Some(&pos) = positions.get(pos_idx) {
                    AbilityTarget::Position(pos)
                } else if matches!(slot.def.targeting, AbilityTargeting::SelfAoe | AbilityTargeting::SelfCast) {
                    AbilityTarget::None
                } else {
                    continue;
                };
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target,
                })
            }
            AbilityCategory::Obstacle => {
                let (mut features, positions) = extract_obstacle_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                // Pick best wall placement position
                let pos_idx = if output.len() > 1 {
                    output[1..].iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                } else { 0 };
                let target = if let Some(&pos) = positions.get(pos_idx) {
                    AbilityTarget::Position(pos)
                } else {
                    AbilityTarget::None
                };
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target,
                })
            }
            AbilityCategory::Summon => {
                let mut features = extract_summon_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                // Summons are typically self-cast or ground-targeted
                let target = match slot.def.targeting {
                    AbilityTargeting::GroundTarget | AbilityTargeting::Direction => {
                        // Place near nearest enemy
                        let nearest_enemy = state.units.iter()
                            .filter(|u| u.team != unit.team && is_alive(u))
                            .min_by(|a, b| {
                                let da = distance(unit.position, a.position);
                                let db = distance(unit.position, b.position);
                                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        match nearest_enemy {
                            Some(e) => AbilityTarget::Position(e.position),
                            None => AbilityTarget::None,
                        }
                    }
                    _ => AbilityTarget::None,
                };
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target,
                })
            }
            AbilityCategory::HealAoe | AbilityCategory::Defense | AbilityCategory::Utility => {
                let mut features = extract_simple_features(state, unit, idx);
                if let Some(ref emb) = embedding { features.extend_from_slice(emb); }
                let output = eval_weights.predict(&features);
                let urgency = sigmoid(output[0]);
                let target = match slot.def.targeting {
                    AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe | AbilityTargeting::Global => {
                        AbilityTarget::None
                    }
                    AbilityTargeting::TargetAlly => {
                        let best_ally = state.units.iter()
                            .filter(|u| u.team == unit.team && is_alive(u))
                            .min_by(|a, b| {
                                let ha = a.hp as f32 / a.max_hp.max(1) as f32;
                                let hb = b.hp as f32 / b.max_hp.max(1) as f32;
                                ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        match best_ally {
                            Some(a) => AbilityTarget::Unit(a.id),
                            None => continue,
                        }
                    }
                    _ => AbilityTarget::None,
                };
                (urgency, IntentAction::UseAbility {
                    ability_index: idx,
                    target,
                })
            }
        };

        // Apply post-prediction urgency modifiers
        let urgency = apply_heal_saturation(urgency, category, state, unit);
        let urgency = apply_cleanup_boost(urgency, category, state, unit);

        // In late-game cleanup, suppress non-offensive abilities entirely
        let urgency = apply_cleanup_suppress(urgency, category, state, unit);

        if urgency > best_urgency {
            best_urgency = urgency;
            best_action = Some(action);
        }
    }

    if best_urgency >= URGENCY_THRESHOLD {
        best_action.and_then(|a| fix_out_of_range(a, best_urgency, unit, state))
    } else {
        None
    }
}

/// If the chosen ability targets a unit that is out of range, either:
/// - In cleanup (advantage + late game): move toward the target to close the gap
/// - Otherwise: return None so the default AI / student model handles movement
///
/// This prevents stalemates where the ability eval keeps picking out-of-range
/// abilities and the hero stands still doing nothing.
fn fix_out_of_range(
    action: IntentAction,
    urgency: f32,
    unit: &UnitState,
    state: &SimState,
) -> Option<(IntentAction, f32)> {
    if let IntentAction::UseAbility { ability_index, target: AbilityTarget::Unit(tid) } = action {
        let range = unit.abilities.get(ability_index)
            .map(|s| s.def.range)
            .unwrap_or(0.0);
        if range > 0.0 {
            if let Some(target) = state.units.iter().find(|u| u.id == tid) {
                let dist = distance(unit.position, target.position);
                if dist > range {
                    // Check if we're in a cleanup situation
                    let ally_count = state.units.iter()
                        .filter(|u| u.team == unit.team && is_alive(u)).count();
                    let enemy_count = state.units.iter()
                        .filter(|u| u.team != unit.team && is_alive(u)).count();
                    let in_cleanup = enemy_count > 0 && ally_count > enemy_count && state.tick > 2000;

                    if in_cleanup {
                        // Move toward target to close gap in cleanup
                        let desired = crate::ai::core::position_at_range(
                            unit.position, target.position, range * 0.9,
                        );
                        let step = unit.move_speed_per_sec * 0.1;
                        let next = crate::ai::core::move_towards(unit.position, desired, step);
                        return Some((IntentAction::MoveTo { position: next }, urgency));
                    } else {
                        // Out of range but not cleanup: fall through to default AI
                        return None;
                    }
                }
            }
        }
    }
    Some((action, urgency))
}

/// Adjust urgency based on team healing saturation.
///
/// When the team has surplus healing capacity (multiple healers, allies healthy),
/// heal urgency is dampened so the unit considers damage abilities instead.
/// Conversely, damage urgency gets a small boost when the healer is redundant.
fn apply_heal_saturation(
    urgency: f32,
    category: AbilityCategory,
    state: &SimState,
    unit: &UnitState,
) -> f32 {
    // Only apply to heal/damage categories
    let is_heal_cat = matches!(category,
        AbilityCategory::HealUnit | AbilityCategory::HealAoe);
    let is_damage_cat = matches!(category,
        AbilityCategory::DamageUnit | AbilityCategory::DamageAoe);

    if !is_heal_cat && !is_damage_cat {
        return urgency;
    }

    // Only modulate for units that ARE healers
    if !is_healer(unit) {
        return urgency;
    }

    let allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();

    let healer_count = allies.iter().filter(|a| is_healer(a)).count();
    let ally_count = allies.len().max(1) as f32;

    // Team HP deficit: 0 = everyone full, 1 = everyone dead
    let hp_deficit = allies.iter()
        .map(|a| 1.0 - (a.hp as f32 / a.max_hp.max(1) as f32))
        .sum::<f32>() / ally_count;

    // Critical allies: anyone below 30% HP
    let critical_count = allies.iter()
        .filter(|a| (a.hp as f32 / a.max_hp.max(1) as f32) < 0.3)
        .count();

    if is_heal_cat {
        // If there's a critical ally, never dampen healing
        if critical_count > 0 {
            return urgency;
        }
        // Dampen healing when: multiple healers AND team is healthy
        if healer_count >= 2 && hp_deficit < 0.2 {
            // Scale dampen by how many extra healers there are
            let surplus = (healer_count - 1) as f32;
            let dampen = (1.0 - hp_deficit) * surplus * 0.15; // up to ~0.45 reduction for 3 extra healers
            return (urgency - dampen).max(0.0);
        }
        urgency
    } else {
        // Boost damage urgency for redundant healers when team is healthy
        if healer_count >= 2 && hp_deficit < 0.3 && critical_count == 0 {
            let boost = (1.0 - hp_deficit) * 0.1; // small boost, up to 0.1
            return (urgency + boost).min(1.0);
        }
        urgency
    }
}

/// Boost damage/CC urgency in late-game cleanup situations.
///
/// Only activates when:
/// - Team has 2:1+ numeric advantage
/// - Fight has been going for a while (tick > 2000)
/// - Dampens heal/defense urgency to prevent passive stalling
fn apply_cleanup_boost(
    urgency: f32,
    category: AbilityCategory,
    state: &SimState,
    unit: &UnitState,
) -> f32 {
    // Only activate in late game
    if state.tick < 2000 {
        return urgency;
    }

    let ally_count = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .count();
    let enemy_count = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .count();

    if enemy_count == 0 || ally_count <= enemy_count {
        return urgency;
    }

    let advantage = ally_count as f32 / enemy_count as f32;

    // Need at least 2:1 advantage
    if advantage < 2.0 {
        return urgency;
    }

    let is_offensive = matches!(category,
        AbilityCategory::DamageUnit | AbilityCategory::DamageAoe | AbilityCategory::CcUnit);
    let is_defensive = matches!(category,
        AbilityCategory::HealUnit | AbilityCategory::HealAoe | AbilityCategory::Defense);

    if is_defensive {
        // Dampen healing/defense in cleanup — focus on killing
        let dampen = ((advantage - 1.0) * 0.1).min(0.15);
        (urgency - dampen).max(0.0)
    } else if is_offensive {
        // Small boost to damage in cleanup
        let boost = ((advantage - 1.0) * 0.05).min(0.1);
        (urgency + boost).min(1.0)
    } else {
        urgency
    }
}

/// Hard suppress non-offensive abilities in late-game cleanup.
///
/// When heroes have a clear numeric advantage and the fight is dragging on,
/// completely zero out heal/defense/utility urgency. This forces the ability
/// evaluator to either pick an offensive ability or return None (falling through
/// to basic attacks), preventing infinite stalemates where heroes waste time
/// healing instead of finishing off remaining enemies.
fn apply_cleanup_suppress(
    urgency: f32,
    category: AbilityCategory,
    state: &SimState,
    unit: &UnitState,
) -> f32 {
    if state.tick < 5000 {
        return urgency;
    }

    let ally_count = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .count();
    let enemy_count = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .count();

    if enemy_count == 0 || ally_count <= enemy_count {
        return urgency;
    }

    let is_defensive = matches!(category,
        AbilityCategory::HealUnit | AbilityCategory::HealAoe |
        AbilityCategory::Defense | AbilityCategory::Utility);

    if is_defensive {
        0.0
    } else {
        urgency
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
