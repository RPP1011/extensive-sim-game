use crate::ai::core::{distance, is_alive, SimState, UnitState, IntentAction};
use crate::ai::effects::{AbilityTarget, AbilityTargeting};
use crate::ai::squad::SquadAiState;

// ---------------------------------------------------------------------------
// Oracle: per-ability urgency scoring
// ---------------------------------------------------------------------------

/// Score a single ability's value via oracle rollout comparison.
/// Returns (urgency, best_target_action) where urgency is the sigmoid of
/// (ability_score - baseline_score) / scale.
pub fn oracle_score_ability(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    ability_index: usize,
    baseline_score: f64,
    rollout_ticks: u64,
) -> Option<(f32, IntentAction)> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let slot = &unit.abilities.get(ability_index)?;

    // Skip if on cooldown or can't afford
    if slot.cooldown_remaining_ms > 0 {
        return None;
    }
    if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
        return None;
    }

    let unit_team = unit.team;

    // Generate candidate actions for this ability
    let mut candidates: Vec<IntentAction> = Vec::new();
    let enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit_team && is_alive(u))
        .collect();
    let allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit_team && is_alive(u) && u.id != unit_id)
        .collect();

    match slot.def.targeting {
        AbilityTargeting::TargetEnemy => {
            let mut sorted: Vec<_> = enemies.iter()
                .map(|e| (distance(unit.position, e.position), e.id))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(_, eid) in sorted.iter().take(3) {
                candidates.push(IntentAction::UseAbility {
                    ability_index,
                    target: AbilityTarget::Unit(eid),
                });
            }
        }
        AbilityTargeting::TargetAlly => {
            // Sort allies by HP (lowest first)
            let mut sorted: Vec<_> = allies.iter()
                .map(|a| (a.hp as f32 / a.max_hp.max(1) as f32, a.id))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(_, aid) in sorted.iter().take(3) {
                candidates.push(IntentAction::UseAbility {
                    ability_index,
                    target: AbilityTarget::Unit(aid),
                });
            }
            // Self-target
            candidates.push(IntentAction::UseAbility {
                ability_index,
                target: AbilityTarget::Unit(unit_id),
            });
        }
        AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe | AbilityTargeting::Global => {
            candidates.push(IntentAction::UseAbility {
                ability_index,
                target: AbilityTarget::None,
            });
        }
        AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
            // Generate candidate positions
            if !enemies.is_empty() {
                // Enemy centroid
                let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
                let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
                candidates.push(IntentAction::UseAbility {
                    ability_index,
                    target: AbilityTarget::Position(crate::ai::core::SimVec2 { x: cx, y: cy }),
                });

                // Top 3 nearest enemy positions
                let mut sorted: Vec<_> = enemies.iter()
                    .map(|e| (distance(unit.position, e.position), e.position))
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                for &(_, pos) in sorted.iter().take(3) {
                    candidates.push(IntentAction::UseAbility {
                        ability_index,
                        target: AbilityTarget::Position(pos),
                    });
                }
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // Rollout each candidate and find the best
    let mut best_score = f64::NEG_INFINITY;
    let mut best_action = candidates[0];

    for candidate in &candidates {
        let (ehl, ahl, kills, cc_ms, heal_sup) = crate::ai::core::oracle::run_rollout(
            state, squad_ai, unit_id, candidate, rollout_ticks, None,
        );
        let score = crate::ai::core::oracle::score_rollout(ehl, ahl, kills, cc_ms, heal_sup);
        if score > best_score {
            best_score = score;
            best_action = *candidate;
        }
    }

    // Urgency = sigmoid((ability_score - baseline) / scale)
    let delta = best_score - baseline_score;
    let scale = 10.0; // tunable: higher = more conservative
    let urgency = 1.0 / (1.0 + (-delta / scale).exp());

    Some((urgency as f32, best_action))
}
