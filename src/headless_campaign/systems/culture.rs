//! Regional culture system — fires every 500 ticks (~50s game time).
//!
//! Guild actions in a region shift the local culture along four axes:
//! martial, trade, scholarly, diplomatic. Each axis ranges 0–100 and
//! decays toward 25 (neutral) when the guild is not active. Rival
//! factions push their own culture in regions they control.
//!
//! Culture effects:
//! - High martial (>50): recruits have +attack/defense, more combat quests
//! - High trade (>50): +20% trade income, merchant NPCs
//! - High scholarly (>50): +XP bonus, discovery quests
//! - High diplomatic (>50): +faction relation gains, diplomatic quests

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the culture system ticks (in ticks).
const CULTURE_INTERVAL: u64 = 17;

/// Influence gain per completed quest of a matching type in the region.
const QUEST_INFLUENCE_GAIN: f32 = 3.0;

/// How fast culture decays toward neutral per culture tick.
const DECAY_RATE: f32 = 0.5;

/// Neutral resting point for all culture axes.
const NEUTRAL: f32 = 25.0;

/// Influence pushed by rival factions in regions they control, per tick.
const RIVAL_PUSH_RATE: f32 = 1.0;

/// Milestone thresholds that emit events when crossed.
const MILESTONES: &[u32] = &[50, 75];

pub fn tick_culture(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CULTURE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Lazily initialize culture entries for all regions.
    let num_regions = state.overworld.regions.len();
    if state.regional_cultures.len() < num_regions {
        state.regional_cultures.resize_with(num_regions, || RegionalCulture::default());
        for (i, culture) in state.regional_cultures.iter_mut().enumerate() {
            culture.region_id = i;
        }
    }

    // --- Phase 1: Tally guild activity per region from recent completed quests ---
    // We look at quests completed in the last CULTURE_INTERVAL ticks.
    let window_start_ms = state.elapsed_ms.saturating_sub(CULTURE_INTERVAL * CAMPAIGN_TURN_SECS as u64 * 1000);
    let mut region_activity: Vec<[f32; 4]> = vec![[0.0; 4]; num_regions];

    for quest in &state.completed_quests {
        if quest.completed_at_ms < window_start_ms {
            continue;
        }
        if quest.result != QuestResult::Victory {
            continue;
        }
        // Determine which region this quest was in via source_area_id from
        // the original request. Since CompletedQuest doesn't store area_id,
        // we approximate: look up active quests first, then fall back to
        // distributing across guild-adjacent regions.
        // For now, use a deterministic hash of the quest id to pick a region.
        let region_idx = (quest.id as usize) % num_regions;

        match quest.quest_type {
            QuestType::Combat | QuestType::Rescue => {
                region_activity[region_idx][0] += QUEST_INFLUENCE_GAIN;
            }
            QuestType::Gather | QuestType::Escort => {
                region_activity[region_idx][1] += QUEST_INFLUENCE_GAIN;
            }
            QuestType::Exploration => {
                region_activity[region_idx][2] += QUEST_INFLUENCE_GAIN;
            }
            QuestType::Diplomatic => {
                region_activity[region_idx][3] += QUEST_INFLUENCE_GAIN;
            }
        }
    }

    // Active quests also contribute (partial influence while in progress).
    for quest in &state.active_quests {
        if let Some(area_id) = quest.request.source_area_id {
            let region_idx = area_id % num_regions;
            let partial = QUEST_INFLUENCE_GAIN * 0.5;
            match quest.request.quest_type {
                QuestType::Combat | QuestType::Rescue => {
                    region_activity[region_idx][0] += partial;
                }
                QuestType::Gather | QuestType::Escort => {
                    region_activity[region_idx][1] += partial;
                }
                QuestType::Exploration => {
                    region_activity[region_idx][2] += partial;
                }
                QuestType::Diplomatic => {
                    region_activity[region_idx][3] += partial;
                }
            }
        }
    }

    // --- Phase 2: Apply guild influence and decay ---
    // Snapshot old dominant cultures for shift detection.
    let old_dominants: Vec<Option<CultureAxis>> = state
        .regional_cultures
        .iter()
        .map(|c| c.dominant())
        .collect();

    // Snapshot old milestone levels for milestone detection.
    let old_levels: Vec<[f32; 4]> = state
        .regional_cultures
        .iter()
        .map(|c| [c.martial_influence, c.trade_influence, c.scholarly_influence, c.diplomatic_influence])
        .collect();

    for (i, culture) in state.regional_cultures.iter_mut().enumerate() {
        let activity = &region_activity[i];
        let has_activity = activity.iter().any(|&v| v > 0.0);

        // Apply guild influence gains.
        culture.martial_influence = (culture.martial_influence + activity[0]).min(100.0);
        culture.trade_influence = (culture.trade_influence + activity[1]).min(100.0);
        culture.scholarly_influence = (culture.scholarly_influence + activity[2]).min(100.0);
        culture.diplomatic_influence = (culture.diplomatic_influence + activity[3]).min(100.0);

        // Decay toward neutral if no guild activity in this region.
        if !has_activity {
            culture.martial_influence = decay_toward(culture.martial_influence, NEUTRAL, DECAY_RATE);
            culture.trade_influence = decay_toward(culture.trade_influence, NEUTRAL, DECAY_RATE);
            culture.scholarly_influence = decay_toward(culture.scholarly_influence, NEUTRAL, DECAY_RATE);
            culture.diplomatic_influence = decay_toward(culture.diplomatic_influence, NEUTRAL, DECAY_RATE);
        }
    }

    // --- Phase 3: Rival faction cultural push ---
    // Each faction pushes its preferred culture axis in regions it controls.
    for region in &state.overworld.regions {
        let faction_id = region.owner_faction_id;
        if faction_id >= state.factions.len() {
            continue;
        }
        let culture = &mut state.regional_cultures[region.id];

        // Faction cultural preference based on diplomatic stance.
        let faction = &state.factions[faction_id];
        match faction.diplomatic_stance {
            DiplomaticStance::Hostile | DiplomaticStance::AtWar => {
                culture.martial_influence = (culture.martial_influence + RIVAL_PUSH_RATE).min(100.0);
            }
            DiplomaticStance::Neutral => {
                culture.trade_influence = (culture.trade_influence + RIVAL_PUSH_RATE).min(100.0);
            }
            DiplomaticStance::Friendly | DiplomaticStance::Coalition => {
                culture.diplomatic_influence = (culture.diplomatic_influence + RIVAL_PUSH_RATE).min(100.0);
            }
        }
    }

    // --- Phase 4: Emit events for culture shifts and milestones ---
    for (i, culture) in state.regional_cultures.iter().enumerate() {
        let new_dominant = culture.dominant();
        if new_dominant != old_dominants[i] {
            if let Some(axis) = new_dominant {
                events.push(WorldEvent::CultureShift {
                    region_id: i,
                    dominant_culture: format!("{:?}", axis),
                });
            }
        }

        // Check milestones.
        let new_levels = [
            culture.martial_influence,
            culture.trade_influence,
            culture.scholarly_influence,
            culture.diplomatic_influence,
        ];
        let axes = [
            CultureAxis::Martial,
            CultureAxis::Trade,
            CultureAxis::Scholarly,
            CultureAxis::Diplomatic,
        ];
        for (j, axis) in axes.iter().enumerate() {
            for &threshold in MILESTONES {
                let t = threshold as f32;
                if old_levels[i][j] < t && new_levels[j] >= t {
                    events.push(WorldEvent::CulturalMilestone {
                        region_id: i,
                        culture: format!("{:?}", axis),
                        level: threshold as f32,
                    });
                }
            }
        }
    }

    // --- Phase 5: Apply culture effects to game state ---
    apply_culture_effects(state);
}

/// Decay a value toward `target` by `rate`, clamping to [0, 100].
fn decay_toward(value: f32, target: f32, rate: f32) -> f32 {
    if value > target {
        (value - rate).max(target)
    } else if value < target {
        (value + rate).min(target)
    } else {
        value
    }
}

/// Apply culture bonuses to game state based on current regional cultures.
fn apply_culture_effects(state: &mut CampaignState) {
    // Accumulate bonuses across all regions.
    let mut trade_income_bonus: f32 = 0.0;
    let mut xp_bonus_regions: Vec<usize> = Vec::new();
    let mut diplomatic_regions: Vec<usize> = Vec::new();

    for culture in &state.regional_cultures {
        // High trade (>50): +20% trade income bonus (per region).
        if culture.trade_influence > 50.0 {
            trade_income_bonus += 0.20;
        }

        // High scholarly (>50): XP bonus for adventurers in that region.
        if culture.scholarly_influence > 50.0 {
            xp_bonus_regions.push(culture.region_id);
        }

        // High diplomatic (>50): faction relation bonus.
        if culture.diplomatic_influence > 50.0 {
            diplomatic_regions.push(culture.region_id);
        }
    }

    // Apply trade income bonus to guild gold (small passive income).
    if trade_income_bonus > 0.0 {
        let bonus = trade_income_bonus * 0.5; // 0.1 gold per high-trade region per tick
        state.guild.gold += bonus;
    }

    // Apply diplomatic bonus: slight faction relation improvement for
    // factions that own diplomatically-cultured regions.
    for &region_id in &diplomatic_regions {
        if region_id < state.overworld.regions.len() {
            let faction_id = state.overworld.regions[region_id].owner_faction_id;
            if faction_id < state.factions.len() {
                let current = state.factions[faction_id].relationship_to_guild;
                state.factions[faction_id].relationship_to_guild = (current + 0.1).min(100.0);
            }
        }
    }

    // Apply scholarly bonus: small XP bump for all adventurers
    // (simplified: we don't track per-adventurer region, so apply globally
    // scaled by how many scholarly regions exist).
    if !xp_bonus_regions.is_empty() {
        let xp_bump = xp_bonus_regions.len() as u32;
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.xp += xp_bump;
            }
        }
    }

    // High martial regions: recruitment bonus is applied in the recruitment
    // system by reading regional_cultures (passive — no mutation needed here).
}

/// Check if any region with high martial culture exists, providing a stat
/// bonus multiplier for newly recruited adventurers.
pub fn martial_recruit_bonus(state: &CampaignState) -> f32 {
    let martial_regions = state
        .regional_cultures
        .iter()
        .filter(|c| c.martial_influence > 50.0)
        .count();
    if martial_regions > 0 {
        // +5% attack/defense per martial region, capped at +25%.
        (martial_regions as f32 * 0.05).min(0.25)
    } else {
        0.0
    }
}
