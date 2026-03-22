//! MCTS rollout policy.
//!
//! Simple heuristic policy for rollout simulation during MCTS tree search.
//! Only needs to be "not terrible" — MCTS corrects for policy quality
//! through exploration.

use crate::headless_campaign::actions::*;
use crate::headless_campaign::state::*;
use crate::headless_campaign::step::step_campaign;

/// Run a heuristic rollout from the current state for up to `horizon_ticks`.
///
/// Returns a value estimate in [0, 1] based on campaign state at rollout end.
pub fn heuristic_rollout(
    state: &mut CampaignState,
    horizon_ticks: u64,
    discount: f64,
) -> f64 {
    let start_tick = state.tick;
    let mut accumulated_reward = 0.0;
    let mut discount_factor = 1.0;

    for _ in 0..horizon_ticks {
        let action = heuristic_rollout_policy(state);
        let result = step_campaign(state, action);

        // Shaped reward
        let step_reward = compute_step_reward(state, &result);
        accumulated_reward += discount_factor * step_reward;
        discount_factor *= discount;

        if let Some(outcome) = result.outcome {
            // Terminal reward
            let terminal = match outcome {
                CampaignOutcome::Victory => 1.0,
                CampaignOutcome::Defeat => -0.5,
                CampaignOutcome::Timeout => 0.0,
            };
            accumulated_reward += discount_factor * terminal;
            break;
        }
    }

    // Normalize to roughly [0, 1]
    (accumulated_reward + 1.0) / 2.0
}

/// Heuristic policy for MCTS rollouts.
///
/// - Accept quests with decent reward/threat ratio
/// - Assign idle adventurers to preparing quests
/// - Dispatch when pool ≥ 2
/// - Send runner when supply low
/// - Rescue when battle going badly
/// - Wait otherwise
pub fn heuristic_rollout_policy(state: &CampaignState) -> Option<CampaignAction> {
    // Accept quests with reward/threat > 1.0
    for req in &state.request_board {
        let ratio = req.reward.gold / req.threat_level.max(1.0);
        if ratio > 1.0 && state.active_quests.len() < state.guild.active_quest_capacity {
            return Some(CampaignAction::AcceptQuest { request_id: req.id });
        }
    }

    // Assign idle adventurers to preparing quests
    for quest in &state.active_quests {
        if quest.status == ActiveQuestStatus::Preparing {
            for adv in &state.adventurers {
                if adv.status == AdventurerStatus::Idle
                    && !quest.assigned_pool.contains(&adv.id)
                {
                    return Some(CampaignAction::AssignToPool {
                        adventurer_id: adv.id,
                        quest_id: quest.id,
                    });
                }
            }
            if quest.assigned_pool.len() >= 2 {
                return Some(CampaignAction::DispatchQuest { quest_id: quest.id });
            }
        }
    }

    // Send runner when supply < 25%
    for party in &state.parties {
        if party.supply_level < 25.0
            && matches!(
                party.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            )
            && state.guild.gold >= RUNNER_COST
        {
            return Some(CampaignAction::SendRunner {
                party_id: party.id,
                payload: RunnerPayload::Supplies(25.0),
            });
        }
    }

    // Rescue when health < 0.3
    for battle in &state.active_battles {
        if battle.status == BattleStatus::Active && battle.party_health_ratio < 0.3 {
            let can_rescue = state.guild.gold >= RESCUE_BRIBE_COST
                || state.npc_relationships.iter().any(|r| r.rescue_available);
            if can_rescue {
                return Some(CampaignAction::CallRescue {
                    battle_id: battle.id,
                });
            }
        }
    }

    // Accept any remaining quest
    if let Some(req) = state.request_board.first() {
        if state.active_quests.len() < state.guild.active_quest_capacity {
            return Some(CampaignAction::AcceptQuest { request_id: req.id });
        }
    }

    None // Wait
}

/// Compute a shaped reward for a single step.
fn compute_step_reward(state: &CampaignState, result: &CampaignStepResult) -> f64 {
    let mut reward = 0.0;

    // Gold changes (normalized)
    let gold_delta = result.deltas.gold_after - result.deltas.gold_before;
    reward += (gold_delta / 100.0) as f64;

    // Quest completions
    reward += result.deltas.quests_completed as f64 * 0.3;
    reward -= result.deltas.quests_failed as f64 * 0.2;

    // Battle results
    for event in &result.events {
        match event {
            WorldEvent::BattleEnded {
                result: BattleStatus::Victory,
                ..
            } => reward += 0.2,
            WorldEvent::BattleEnded {
                result: BattleStatus::Defeat,
                ..
            } => reward -= 0.3,
            WorldEvent::AdventurerDied { .. } => reward -= 0.5,
            WorldEvent::AdventurerDeserted { .. } => reward -= 0.3,
            WorldEvent::ProgressionUnlocked { .. } => reward += 0.4,
            _ => {}
        }
    }

    // Reputation progress
    let rep_delta = result.deltas.reputation_after - result.deltas.reputation_before;
    reward += (rep_delta / 50.0) as f64;

    // Campaign progress
    reward += state.overworld.campaign_progress as f64 * 0.01;

    reward
}
