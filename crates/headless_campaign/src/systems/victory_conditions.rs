//! Victory conditions system — checks multiple win conditions every 200 ticks.
//!
//! Different victory conditions create different strategic landscapes:
//! - **SurviveCrisis**: defeat the endgame crisis (default, checked in step.rs)
//! - **EconomicDominance**: accumulate 5000 total gold earned
//! - **DiplomaticUnity**: achieve relation > 60 with all factions simultaneously
//! - **MilitaryConquest**: control all regions
//! - **CulturalHegemony**: reach Legend tier + all buildings at tier 3
//! - **QuestMaster**: complete 50 quests
//!
//! Near-victory (>75%) escalation increases crisis pressure so the world fights back.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to check victory progress (in turns, ~21 seconds).
const VICTORY_CHECK_INTERVAL: u64 = 7;

/// Tick the victory conditions system.
///
/// Every 200 ticks:
/// 1. Update progress metrics for the active victory condition
/// 2. Emit milestone events at 25%, 50%, 75%
/// 3. Check if the condition is fully met
/// 4. Apply near-victory escalation when progress > 75%
pub fn tick_victory_conditions(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % VICTORY_CHECK_INTERVAL != 0 {
        return;
    }

    let condition = state.victory_condition;
    let old_percent = state.victory_progress.percent;

    // Update progress for the active condition
    let new_percent = compute_progress(state, condition);
    state.victory_progress.percent = new_percent;

    // Update sub-metrics
    update_sub_metrics(state);

    // Emit milestone events at 25%, 50%, 75% thresholds
    for &milestone in &[25.0_f32, 50.0, 75.0] {
        if old_percent < milestone && new_percent >= milestone {
            events.push(WorldEvent::VictoryProgress {
                condition: format!("{:?}", condition),
                percent: new_percent,
            });
        }
    }

    // Near-victory escalation: when progress > 75%, increase crisis pressure
    if new_percent >= 75.0 && old_percent < 75.0 {
        events.push(WorldEvent::NearVictoryEscalation);
        apply_near_victory_escalation(state);
    }

    // Check for victory — requires minimum 15,000 turns (~12 hours game time)
    // and at least one adventurer at level 30+ (mid-late game progression)
    let min_turns_met = state.tick >= 15_000;
    let has_high_level = state.adventurers.iter().any(|a| {
        a.status != AdventurerStatus::Dead
            && a.classes.iter().any(|c| c.level >= 30)
    });
    if new_percent >= 100.0 && min_turns_met && has_high_level {
        events.push(WorldEvent::VictoryAchieved { condition: format!("{:?}", condition) });
        state.overworld.campaign_progress = 1.0;
    }
}

/// Compute the percentage progress toward a given victory condition.
fn compute_progress(state: &CampaignState, condition: VictoryCondition) -> f32 {
    match condition {
        VictoryCondition::SurviveCrisis => {
            // Progress mirrors campaign_progress (driven by crisis system).
            (state.overworld.campaign_progress * 100.0).clamp(0.0, 100.0)
        }
        VictoryCondition::EconomicDominance => {
            let target = 50_000.0_f32; // 10x harder for longer campaigns
            (state.victory_progress.total_gold_earned / target * 100.0).clamp(0.0, 100.0)
        }
        VictoryCondition::DiplomaticUnity => {
            if state.factions.is_empty() {
                return 0.0;
            }
            // All non-guild factions must have relation > 75 (stricter for longer campaign).
            let qualifying = state
                .factions
                .iter()
                .filter(|f| f.id != state.diplomacy.guild_faction_id)
                .filter(|f| f.relationship_to_guild > 75.0)
                .count();
            let total_non_guild = state
                .factions
                .iter()
                .filter(|f| f.id != state.diplomacy.guild_faction_id)
                .count();
            if total_non_guild == 0 {
                return 100.0;
            }
            (qualifying as f32 / total_non_guild as f32 * 100.0).clamp(0.0, 100.0)
        }
        VictoryCondition::MilitaryConquest => {
            if state.overworld.regions.is_empty() {
                return 100.0;
            }
            let guild_regions = state
                .overworld
                .regions
                .iter()
                .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
                .count();
            (guild_regions as f32 / state.overworld.regions.len() as f32 * 100.0)
                .clamp(0.0, 100.0)
        }
        VictoryCondition::CulturalHegemony => {
            // Three sub-goals:
            // 1. Any adventurer at Legend tier (tier 5) — 30%
            // 2. All 6 buildings at tier 3 — 30%
            // 3. Any adventurer with a class at level 40+ — 40%
            let has_legend = state
                .adventurers
                .iter()
                .any(|a| a.tier_status.tier >= 5 && a.status != AdventurerStatus::Dead);
            let legend_pct = if has_legend { 30.0 } else {
                // Partial: highest tier / 5 * 30
                let max_tier = state
                    .adventurers
                    .iter()
                    .filter(|a| a.status != AdventurerStatus::Dead)
                    .map(|a| a.tier_status.tier)
                    .max()
                    .unwrap_or(0);
                (max_tier as f32 / 5.0 * 30.0).clamp(0.0, 30.0)
            };

            let max_class_level = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .flat_map(|a| a.classes.iter().map(|c| c.level))
                .max()
                .unwrap_or(0);
            let class_pct = (max_class_level as f32 / 40.0 * 40.0).clamp(0.0, 40.0);

            let b = &state.guild_buildings;
            let total_tiers = (b.training_grounds + b.watchtower + b.trade_post
                + b.barracks + b.infirmary + b.war_room) as f32;
            let max_total = 18.0; // 6 buildings * 3 max tier
            let building_pct = (total_tiers / max_total * 30.0).clamp(0.0, 30.0);

            (legend_pct + building_pct + class_pct).clamp(0.0, 100.0)
        }
        VictoryCondition::QuestMaster => {
            let target = 150.0_f32; // 3x harder for longer campaigns
            let completed = state.victory_progress.quests_completed as f32;
            (completed / target * 100.0).clamp(0.0, 100.0)
        }
    }
}

/// Update the sub-metric trackers on VictoryProgress each check.
fn update_sub_metrics(state: &mut CampaignState) {
    // Track total gold earned: gold currently held + gold spent on everything.
    // We approximate by tracking the maximum gold seen + cumulative spending.
    // A simpler and more reliable approach: track it cumulatively in economy ticks.
    // For now, we use `gold` plus completed quest gold rewards as a proxy.
    let quest_gold: f32 = state
        .completed_quests
        .iter()
        .map(|q| q.reward_applied.gold)
        .sum();
    let trade_income = state.guild.total_trade_income;
    // Total gold earned = quest rewards + trade income + current holdings as floor
    let earned = quest_gold + trade_income + state.guild.gold;
    if earned > state.victory_progress.total_gold_earned {
        state.victory_progress.total_gold_earned = earned;
    }

    // Track quests completed
    state.victory_progress.quests_completed = state.completed_quests.len() as u32;

    // Track regions controlled
    state.victory_progress.regions_controlled = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .count() as u32;

    // Track max adventurer tier
    state.victory_progress.max_adventurer_tier = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| a.tier_status.tier)
        .max()
        .unwrap_or(0);

    // Track building sum
    let b = &state.guild_buildings;
    state.victory_progress.total_building_tiers = (b.training_grounds
        + b.watchtower
        + b.trade_post
        + b.barracks
        + b.infirmary
        + b.war_room) as u32;

    // Track faction relations above threshold
    state.victory_progress.factions_above_60 = state
        .factions
        .iter()
        .filter(|f| f.id != state.diplomacy.guild_faction_id)
        .filter(|f| f.relationship_to_guild > 60.0)
        .count() as u32;
}

/// When any victory condition crosses 75%, make the world fight back.
fn apply_near_victory_escalation(state: &mut CampaignState) {
    // Increase global threat — crisis systems will react to this
    state.overworld.global_threat_level =
        (state.overworld.global_threat_level + 15.0).min(100.0);

    // Hostile factions get a military boost
    for faction in &mut state.factions {
        if faction.id != state.diplomacy.guild_faction_id
            && faction.relationship_to_guild < 0.0
        {
            faction.military_strength += 10.0;
        }
    }

    // Increase unrest in guild-controlled regions
    for region in &mut state.overworld.regions {
        if region.owner_faction_id == state.diplomacy.guild_faction_id {
            region.unrest = (region.unrest + 10.0).min(100.0);
        }
    }
}

/// Select a victory condition deterministically from the campaign RNG.
///
/// Uses the campaign seed to pick one of the 6 conditions, or cycles
/// through them for BFS diversity when `cycle_index` is provided.
pub fn select_victory_condition(rng: u64, cycle_index: Option<u32>) -> VictoryCondition {
    let variants = VictoryCondition::ALL;
    let idx = if let Some(ci) = cycle_index {
        ci as usize % variants.len()
    } else {
        (rng as usize) % variants.len()
    };
    variants[idx]
}

/// Adjust the BFS value estimate based on the active victory condition.
///
/// Different conditions value different state dimensions, so the value
/// function should weight them accordingly.
pub fn victory_condition_value_adjustment(
    state: &CampaignState,
    base_value: f32,
) -> f32 {
    let condition = state.victory_condition;
    let progress = state.victory_progress.percent / 100.0;

    // Bonus for making progress toward the active victory condition
    let progress_bonus = progress * 1.5;

    // Condition-specific value adjustments
    let specific_bonus = match condition {
        VictoryCondition::SurviveCrisis => {
            // Default behavior — no extra adjustment
            0.0
        }
        VictoryCondition::EconomicDominance => {
            // Extra weight on gold and trade income
            let gold_value = (state.guild.gold / 500.0).min(1.0) * 0.5;
            let trade_value = (state.guild.total_trade_income / 200.0).min(1.0) * 0.3;
            gold_value + trade_value
        }
        VictoryCondition::DiplomaticUnity => {
            // Extra weight on faction relations
            let mean_relation = if state.factions.is_empty() {
                0.0
            } else {
                let non_guild: Vec<_> = state
                    .factions
                    .iter()
                    .filter(|f| f.id != state.diplomacy.guild_faction_id)
                    .collect();
                if non_guild.is_empty() {
                    0.0
                } else {
                    non_guild
                        .iter()
                        .map(|f| f.relationship_to_guild)
                        .sum::<f32>()
                        / non_guild.len() as f32
                }
            };
            (mean_relation / 100.0).clamp(0.0, 1.0) * 0.8
        }
        VictoryCondition::MilitaryConquest => {
            // Extra weight on territory control and military strength
            let territory = if state.overworld.regions.is_empty() {
                1.0
            } else {
                let guild_regions = state
                    .overworld
                    .regions
                    .iter()
                    .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
                    .count() as f32;
                guild_regions / state.overworld.regions.len() as f32
            };
            territory * 0.8
        }
        VictoryCondition::CulturalHegemony => {
            // Extra weight on tier progression and buildings
            let max_tier = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .map(|a| a.tier_status.tier)
                .max()
                .unwrap_or(0) as f32;
            let tier_value = (max_tier / 5.0).min(1.0) * 0.4;
            let b = &state.guild_buildings;
            let building_total = (b.training_grounds + b.watchtower + b.trade_post
                + b.barracks + b.infirmary + b.war_room) as f32;
            let building_value = (building_total / 18.0).min(1.0) * 0.4;
            tier_value + building_value
        }
        VictoryCondition::QuestMaster => {
            // Extra weight on quest throughput
            let quests = state.completed_quests.len() as f32;
            (quests / 50.0).min(1.0) * 0.8
        }
    };

    base_value + progress_bonus + specific_bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_victory_condition_cycles() {
        // Cycle index should deterministically rotate through all conditions
        let conditions: Vec<_> = (0..6)
            .map(|i| select_victory_condition(0, Some(i)))
            .collect();
        assert_eq!(conditions[0], VictoryCondition::SurviveCrisis);
        assert_eq!(conditions[1], VictoryCondition::EconomicDominance);
        assert_eq!(conditions[5], VictoryCondition::QuestMaster);
        // Wraps around
        assert_eq!(
            select_victory_condition(0, Some(6)),
            VictoryCondition::SurviveCrisis
        );
    }

    #[test]
    fn test_select_victory_condition_from_rng() {
        // Without cycle index, uses rng seed
        let c1 = select_victory_condition(0, None);
        let c2 = select_victory_condition(1, None);
        assert_eq!(c1, VictoryCondition::SurviveCrisis); // 0 % 6 = 0
        assert_eq!(c2, VictoryCondition::EconomicDominance); // 1 % 6 = 1
    }

    #[test]
    fn test_compute_progress_economic() {
        let mut state = CampaignState::default_test_campaign(42);
        state.victory_condition = VictoryCondition::EconomicDominance;
        state.victory_progress.total_gold_earned = 2500.0;
        let pct = compute_progress(&state, VictoryCondition::EconomicDominance);
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_progress_quest_master() {
        let mut state = CampaignState::default_test_campaign(42);
        state.victory_condition = VictoryCondition::QuestMaster;
        state.victory_progress.quests_completed = 25;
        let pct = compute_progress(&state, VictoryCondition::QuestMaster);
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_near_victory_escalation_increases_threat() {
        let mut state = CampaignState::default_test_campaign(42);
        let threat_before = state.overworld.global_threat_level;
        apply_near_victory_escalation(&mut state);
        assert!(state.overworld.global_threat_level > threat_before);
    }
}
