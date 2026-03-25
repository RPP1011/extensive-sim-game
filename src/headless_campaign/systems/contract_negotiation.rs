//! Contract negotiation system — every 200 ticks.
//!
//! When quests appear on the guild board, the guild can negotiate terms
//! (reward, deadline, party size) with the offering faction. Negotiation
//! success depends on guild reputation, faction relationship, and quest urgency.
//! Each counter-offer reduces faction patience; if patience reaches zero the
//! quest is withdrawn entirely.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, CampaignState, NegotiationState, QuestResult,
};

/// How often to tick negotiations (in ticks).
const NEGOTIATION_TICK_INTERVAL: u64 = 200;

/// Minimum guild reputation with a faction to initiate negotiation.
const MIN_REPUTATION_TO_NEGOTIATE: f32 = 30.0;

/// Maximum negotiation rounds before auto-resolution.
const DEFAULT_MAX_ROUNDS: u32 = 3;

/// Run the contract negotiation system. Called every tick; gates internally.
pub fn tick_contract_negotiation(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % NEGOTIATION_TICK_INTERVAL != 0 {
        return;
    }

    // --- Start new negotiations for eligible quests on the board ---
    start_new_negotiations(state, events);

    // --- Advance existing negotiations ---
    advance_negotiations(state, events);
}

/// Start negotiations for quests on the board that don't yet have one.
fn start_new_negotiations(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect quest IDs already being negotiated
    let negotiating_ids: Vec<u32> = state
        .negotiation_rounds
        .iter()
        .map(|n| n.quest_id)
        .collect();

    // Find quests on the board eligible for negotiation
    let eligible: Vec<(u32, f32, Option<usize>)> = state
        .request_board
        .iter()
        .filter(|q| !negotiating_ids.contains(&q.id))
        .map(|q| (q.id, q.reward.gold, q.source_faction_id))
        .collect();

    for (quest_id, reward_gold, faction_id) in eligible {
        // Must have a faction to negotiate with
        let fid = match faction_id {
            Some(fid) => fid,
            None => continue,
        };

        // Check faction relationship meets minimum threshold
        let faction_rel = state
            .factions
            .iter()
            .find(|f| f.id == fid)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0);

        if faction_rel < MIN_REPUTATION_TO_NEGOTIATE {
            continue;
        }

        // Roll to decide if negotiation starts (50% base chance)
        let roll = lcg_f32(&mut state.rng);
        if roll > 0.5 {
            continue;
        }

        let negotiation = NegotiationState {
            quest_id,
            original_reward: reward_gold,
            current_offer: reward_gold,
            rounds: 0,
            max_rounds: DEFAULT_MAX_ROUNDS,
            faction_patience: 1.0,
            faction_id: fid,
        };

        events.push(WorldEvent::NegotiationStarted {
            quest_id,
            original_reward: reward_gold,
        });

        state.negotiation_rounds.push(negotiation);
    }
}

/// Advance each active negotiation by one round.
fn advance_negotiations(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Process negotiations; collect indices to remove (resolved ones)
    let mut to_remove: Vec<usize> = Vec::new();

    for i in 0..state.negotiation_rounds.len() {
        let neg = &state.negotiation_rounds[i];
        let quest_id = neg.quest_id;
        let faction_id = neg.faction_id;

        // Check if the quest is still on the board
        let quest_still_available = state.request_board.iter().any(|q| q.id == quest_id);
        if !quest_still_available {
            to_remove.push(i);
            continue;
        }

        // Already exhausted patience
        if neg.faction_patience <= 0.0 {
            to_remove.push(i);
            continue;
        }

        // Already at max rounds — resolve
        if neg.rounds >= neg.max_rounds {
            to_remove.push(i);
            // Final offer accepted — apply the current offer
            if let Some(quest) = state.request_board.iter_mut().find(|q| q.id == quest_id) {
                quest.reward.gold = neg.current_offer;
            }
            events.push(WorldEvent::NegotiationAccepted {
                quest_id,
                final_reward: neg.current_offer,
            });
            continue;
        }

        // --- Calculate acceptance modifiers ---
        let faction_rel = state
            .factions
            .iter()
            .find(|f| f.id == faction_id)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0);

        // High reputation bonus: +20% acceptance if > 60
        let rep_bonus = if faction_rel > 60.0 { 0.20 } else { 0.0 };

        // Quest completion history bonus: +10% if completed 5+ quests for faction
        let completed_for_faction = state
            .completed_quests
            .iter()
            .filter(|cq| {
                cq.result == QuestResult::Victory
                    && cq.reward_applied.relation_faction_id == Some(faction_id)
            })
            .count();
        let history_bonus = if completed_for_faction >= 5 { 0.10 } else { 0.0 };

        // Urgency bonus: faction in crisis makes them more willing to pay premium
        let faction_in_crisis = state
            .overworld
            .regions
            .iter()
            .any(|r| r.threat_level > 70.0);
        let urgency_bonus = if faction_in_crisis { 0.15 } else { 0.0 };

        // --- Guild counter-offer: request higher reward ---
        // Counter-offer amount: +10-50% of original reward
        let increase_pct = 0.10 + lcg_f32(&mut state.rng) * 0.40;
        let proposed_reward = neg.original_reward * (1.0 + increase_pct);

        // Patience cost: 0.1-0.3 per counter-offer
        let patience_cost = 0.1 + lcg_f32(&mut state.rng) * 0.2;

        // Base acceptance: 30% + modifiers
        let acceptance_chance = 0.30 + rep_bonus + history_bonus + urgency_bonus;
        let roll = lcg_f32(&mut state.rng);

        let neg = &mut state.negotiation_rounds[i];
        neg.rounds += 1;
        neg.faction_patience = (neg.faction_patience - patience_cost).max(0.0);

        if neg.faction_patience <= 0.0 {
            // Faction patience exhausted — withdraw quest
            to_remove.push(i);
            state.request_board.retain(|q| q.id != quest_id);
            events.push(WorldEvent::NegotiationFailed {
                quest_id,
                reason: "Faction patience exhausted — quest withdrawn".into(),
            });
            continue;
        }

        if roll < acceptance_chance {
            // Faction accepts the counter-offer
            neg.current_offer = proposed_reward;
            if let Some(quest) = state.request_board.iter_mut().find(|q| q.id == quest_id) {
                quest.reward.gold = proposed_reward;
            }
            to_remove.push(i);
            events.push(WorldEvent::NegotiationAccepted {
                quest_id,
                final_reward: proposed_reward,
            });
        } else {
            // Faction makes a counter — split the difference
            let faction_counter = neg.current_offer + (proposed_reward - neg.current_offer) * 0.5;
            neg.current_offer = faction_counter;

            events.push(WorldEvent::NegotiationCounterOffer {
                quest_id,
                new_reward: faction_counter,
                round: neg.rounds,
            });
        }
    }

    // Remove resolved negotiations (reverse order to preserve indices)
    to_remove.sort_unstable();
    to_remove.dedup();
    for idx in to_remove.into_iter().rev() {
        state.negotiation_rounds.swap_remove(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;
        state.tick = 200;
        state
    }

    #[test]
    fn no_negotiation_without_faction() {
        let mut state = make_test_state();
        // Add a quest without a faction
        state.request_board.push(crate::headless_campaign::state::QuestRequest {
            id: 1,
            source_faction_id: None,
            source_area_id: None,
            quest_type: crate::headless_campaign::state::QuestType::Combat,
            threat_level: 20.0,
            reward: crate::headless_campaign::state::QuestReward {
                gold: 50.0,
                ..Default::default()
            },
            distance: 10.0,
            target_position: (0.0, 0.0),
            deadline_ms: 99999,
            description: "Test quest".into(),
            arrived_at_ms: 0,
        });

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_contract_negotiation(&mut state, &mut deltas, &mut events);

        assert!(
            state.negotiation_rounds.is_empty(),
            "Should not negotiate quests without a faction"
        );
    }

    #[test]
    fn no_negotiation_low_reputation() {
        let mut state = make_test_state();
        // Ensure faction relationship is below threshold
        if let Some(f) = state.factions.first_mut() {
            f.relationship_to_guild = 10.0;
        }
        let faction_id = state.factions.first().map(|f| f.id);
        state.request_board.push(crate::headless_campaign::state::QuestRequest {
            id: 1,
            source_faction_id: faction_id,
            source_area_id: None,
            quest_type: crate::headless_campaign::state::QuestType::Combat,
            threat_level: 20.0,
            reward: crate::headless_campaign::state::QuestReward {
                gold: 50.0,
                ..Default::default()
            },
            distance: 10.0,
            target_position: (0.0, 0.0),
            deadline_ms: 99999,
            description: "Test quest".into(),
            arrived_at_ms: 0,
        });

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_contract_negotiation(&mut state, &mut deltas, &mut events);

        assert!(
            state.negotiation_rounds.is_empty(),
            "Should not negotiate with low-reputation faction"
        );
    }

    #[test]
    fn negotiation_starts_with_high_rep_faction() {
        let mut state = make_test_state();
        // Set high reputation
        if let Some(f) = state.factions.first_mut() {
            f.relationship_to_guild = 50.0;
        }
        let faction_id = state.factions.first().map(|f| f.id);

        // Add multiple quests to increase chance of at least one negotiation starting
        for i in 0..10 {
            state.request_board.push(crate::headless_campaign::state::QuestRequest {
                id: 100 + i,
                source_faction_id: faction_id,
                source_area_id: None,
                quest_type: crate::headless_campaign::state::QuestType::Combat,
                threat_level: 20.0,
                reward: crate::headless_campaign::state::QuestReward {
                    gold: 50.0,
                    ..Default::default()
                },
                distance: 10.0,
                target_position: (0.0, 0.0),
                deadline_ms: 99999,
                description: "Test quest".into(),
                arrived_at_ms: 0,
            });
        }

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_contract_negotiation(&mut state, &mut deltas, &mut events);

        // With 10 quests and 50% chance each, very likely at least one starts
        let started_count = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::NegotiationStarted { .. }))
            .count();
        assert!(
            started_count > 0,
            "Should start at least one negotiation with 10 eligible quests"
        );
    }

    #[test]
    fn patience_exhaustion_withdraws_quest() {
        let mut state = make_test_state();
        let faction_id = state.factions.first().map(|f| f.id).unwrap_or(0);

        state.request_board.push(crate::headless_campaign::state::QuestRequest {
            id: 1,
            source_faction_id: Some(faction_id),
            source_area_id: None,
            quest_type: crate::headless_campaign::state::QuestType::Combat,
            threat_level: 20.0,
            reward: crate::headless_campaign::state::QuestReward {
                gold: 50.0,
                ..Default::default()
            },
            distance: 10.0,
            target_position: (0.0, 0.0),
            deadline_ms: 99999,
            description: "Test quest".into(),
            arrived_at_ms: 0,
        });

        // Manually add a negotiation with near-zero patience
        state.negotiation_rounds.push(NegotiationState {
            quest_id: 1,
            original_reward: 50.0,
            current_offer: 60.0,
            rounds: 0,
            max_rounds: DEFAULT_MAX_ROUNDS,
            faction_patience: 0.05,
            faction_id,
        });

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();

        // Advance negotiation
        advance_negotiations(&mut state, &mut events);

        // Quest should be withdrawn (patience will hit 0 after the cost)
        let failed = events.iter().any(|e| matches!(e, WorldEvent::NegotiationFailed { .. }));
        assert!(failed, "Quest should be withdrawn when patience exhausted");
        assert!(
            state.request_board.iter().all(|q| q.id != 1),
            "Quest should be removed from board"
        );
    }

    #[test]
    fn max_rounds_auto_accepts() {
        let mut state = make_test_state();
        let faction_id = state.factions.first().map(|f| f.id).unwrap_or(0);

        state.request_board.push(crate::headless_campaign::state::QuestRequest {
            id: 1,
            source_faction_id: Some(faction_id),
            source_area_id: None,
            quest_type: crate::headless_campaign::state::QuestType::Combat,
            threat_level: 20.0,
            reward: crate::headless_campaign::state::QuestReward {
                gold: 50.0,
                ..Default::default()
            },
            distance: 10.0,
            target_position: (0.0, 0.0),
            deadline_ms: 99999,
            description: "Test quest".into(),
            arrived_at_ms: 0,
        });

        // Add negotiation at max rounds with modified offer
        state.negotiation_rounds.push(NegotiationState {
            quest_id: 1,
            original_reward: 50.0,
            current_offer: 65.0,
            rounds: DEFAULT_MAX_ROUNDS,
            max_rounds: DEFAULT_MAX_ROUNDS,
            faction_patience: 0.5,
            faction_id,
        });

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        advance_negotiations(&mut state, &mut events);

        let accepted = events.iter().any(|e| matches!(e, WorldEvent::NegotiationAccepted { .. }));
        assert!(accepted, "Should auto-accept at max rounds");

        // Quest reward should be updated
        let quest = state.request_board.iter().find(|q| q.id == 1).unwrap();
        assert_eq!(quest.reward.gold, 65.0, "Quest reward should be updated to current offer");
    }
}
