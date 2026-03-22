//! Global threat and campaign progress — every 600 ticks (~60s).
//!
//! Updates global threat level based on world state.
//! Checks for endgame calamity conditions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

pub fn tick_threat(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let tcfg = &state.config.threat;
    if state.tick % tcfg.update_interval_ticks != 0 || state.tick == 0 {
        return;
    }

    let pcfg = &state.config.campaign_progress;

    // Global threat rises with unrest and hostile factions
    let avg_unrest = if state.overworld.regions.is_empty() {
        0.0
    } else {
        state.overworld.regions.iter().map(|r| r.unrest).sum::<f32>()
            / state.overworld.regions.len() as f32
    };

    let hostile_factions = state
        .factions
        .iter()
        .filter(|f| matches!(f.diplomatic_stance, DiplomaticStance::Hostile | DiplomaticStance::AtWar))
        .count() as f32;

    state.overworld.global_threat_level =
        (avg_unrest * tcfg.unrest_weight + hostile_factions * tcfg.hostile_faction_multiplier
            + state.overworld.campaign_progress * tcfg.progress_threat_weight)
            .clamp(0.0, 100.0);

    // Campaign progress based on quests completed, reputation, and territory
    let _quests_done = state.completed_quests.len() as f32;
    let victories = state
        .completed_quests
        .iter()
        .filter(|q| q.result == crate::headless_campaign::state::QuestResult::Victory)
        .count() as f32;
    let guild_territory = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .count() as f32;
    let total_regions = state.overworld.regions.len().max(1) as f32;

    let quest_progress = (victories / pcfg.victory_quest_count).min(1.0);
    let territory_progress = guild_territory / total_regions;
    let rep_progress = (state.guild.reputation / pcfg.reputation_cap).min(1.0);

    state.overworld.campaign_progress =
        (quest_progress * pcfg.quest_weight + rep_progress * pcfg.reputation_weight
            + territory_progress * pcfg.territory_weight).min(1.0);

    // Quest victories also flip territory
    if victories > 0.0 && (victories as u32) % pcfg.territory_flip_interval == 0 {
        if let Some(region) = state
            .overworld
            .regions
            .iter_mut()
            .find(|r| r.owner_faction_id != state.diplomacy.guild_faction_id)
        {
            region.owner_faction_id = state.diplomacy.guild_faction_id;
            region.control = pcfg.capture_control;
            region.unrest = (region.unrest - pcfg.capture_unrest_reduction).max(0.0);
        }
    }

    // Endgame calamity warning
    if state.overworld.campaign_progress > pcfg.calamity_warning_threshold && state.overworld.endgame_calamity.is_none() {
        // Select a calamity based on world state
        let strongest_hostile = state
            .factions
            .iter()
            .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
            .max_by(|a, b| a.military_strength.partial_cmp(&b.military_strength).unwrap_or(std::cmp::Ordering::Equal));

        let calamity = if let Some(faction) = strongest_hostile {
            CalamityType::AggressiveFaction {
                faction_id: faction.id,
            }
        } else if avg_unrest > pcfg.crisis_flood_unrest_threshold {
            CalamityType::CrisisFlood
        } else {
            CalamityType::Conquest
        };

        state.overworld.endgame_calamity = Some(calamity);
        events.push(WorldEvent::CalamityWarning {
            description: "Endgame calamity approaching!".into(),
        });
    }
}
