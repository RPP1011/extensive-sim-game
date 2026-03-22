//! Economy tick — every tick.
//!
//! Applies rewards from completed quests, passive income, and costs.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::CampaignState;

/// Passive gold income per second.
const PASSIVE_GOLD_PER_SEC: f32 = 0.5;

pub fn tick_economy(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let dt_sec = 0.1; // 100ms tick
    deltas.gold_before = state.guild.gold;
    deltas.supplies_before = state.guild.supplies;
    deltas.reputation_before = state.guild.reputation;

    // Passive income
    state.guild.gold += PASSIVE_GOLD_PER_SEC * dt_sec;

    // Apply completed quest rewards (process newly completed quests)
    // Quest rewards are applied by quest_lifecycle when quests complete,
    // but the actual gold/rep/supply changes happen here.
    let newly_completed: Vec<_> = state
        .completed_quests
        .iter()
        .filter(|q| q.completed_at_ms == state.elapsed_ms)
        .cloned()
        .collect();

    for quest in &newly_completed {
        let reward = &quest.reward_applied;
        if reward.gold > 0.0 {
            state.guild.gold += reward.gold;
            events.push(WorldEvent::GoldChanged {
                amount: reward.gold,
                reason: format!("Quest {} reward", quest.id),
            });
        }
        if reward.reputation > 0.0 {
            state.guild.reputation =
                (state.guild.reputation + reward.reputation).min(100.0);
        }
        if reward.supply_reward > 0.0 {
            state.guild.supplies += reward.supply_reward;
            events.push(WorldEvent::SupplyChanged {
                amount: reward.supply_reward,
                reason: format!("Quest {} reward", quest.id),
            });
        }
        // Apply faction relation change
        if let Some(fid) = reward.relation_faction_id {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == fid) {
                let old = faction.relationship_to_guild;
                faction.relationship_to_guild =
                    (faction.relationship_to_guild + reward.relation_change).clamp(-100.0, 100.0);
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id: fid,
                    old,
                    new: faction.relationship_to_guild,
                });
            }
        }
    }

    deltas.gold_after = state.guild.gold;
    deltas.supplies_after = state.guild.supplies;
    deltas.reputation_after = state.guild.reputation;
}
