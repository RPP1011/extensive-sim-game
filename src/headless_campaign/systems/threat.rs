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
    if state.tick % 600 != 0 || state.tick == 0 {
        return;
    }

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
        (avg_unrest * 0.5 + hostile_factions * 10.0 + state.overworld.campaign_progress * 20.0)
            .clamp(0.0, 100.0);

    // Campaign progress based on quests completed, reputation, and territory
    let quests_done = state.completed_quests.len() as f32;
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

    // 25 victories = campaign win
    // Quest victories 60%, reputation 25%, territory 15%
    let quest_progress = (victories / 25.0).min(1.0);
    let territory_progress = guild_territory / total_regions;
    let rep_progress = (state.guild.reputation / 70.0).min(1.0);

    state.overworld.campaign_progress =
        (quest_progress * 0.6 + rep_progress * 0.25 + territory_progress * 0.15).min(1.0);

    // Quest victories also flip territory: every 5 victories, take a region
    if victories > 0.0 && (victories as u32) % 5 == 0 {
        if let Some(region) = state
            .overworld
            .regions
            .iter_mut()
            .find(|r| r.owner_faction_id != state.diplomacy.guild_faction_id)
        {
            region.owner_faction_id = state.diplomacy.guild_faction_id;
            region.control = 60.0;
            region.unrest = (region.unrest - 10.0).max(0.0);
        }
    }

    // Endgame calamity warning
    if state.overworld.campaign_progress > 0.7 && state.overworld.endgame_calamity.is_none() {
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
        } else if avg_unrest > 60.0 {
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
