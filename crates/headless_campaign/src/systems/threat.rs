//! Global threat and campaign progress — every 600 ticks (~60s).
//!
//! Updates global threat level based on world state.
//! Checks for endgame crisis conditions using data-driven templates
//! loaded from `dataset/campaign/crises/*.toml`.

use crate::actions::{StepDeltas, WorldEvent};
use crate::crisis_templates::get_or_load_crises;
use crate::state::*;

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
        .filter(|q| q.result == crate::state::QuestResult::Victory)
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

    // --- Data-driven crisis activation from templates ---
    let templates = get_or_load_crises();

    // Templates are already sorted by priority (highest first).
    // Activate each template whose threshold is met, respecting allow_simultaneous.
    for template in templates {
        if state.overworld.campaign_progress < template.trigger_threshold {
            continue;
        }

        // Check world_template filter (empty = any world)
        // We match against faction names as a proxy for world template identity.
        if !template.world_templates.is_empty() {
            let faction_names: Vec<&str> = state.factions.iter().map(|f| f.name.as_str()).collect();
            let world_match = template.world_templates.iter().any(|wt| {
                faction_names.iter().any(|fn_| fn_.contains(wt.as_str()))
            });
            if !world_match {
                continue;
            }
        }

        // Check allow_simultaneous: if false, skip if any crisis is already active
        if !template.allow_simultaneous && !state.overworld.active_crises.is_empty() {
            continue;
        }

        // Activate the crisis (activate_crisis_from_template handles dedup)
        super::crisis::activate_crisis_from_template(state, template, events);

        // Track the first crisis as the endgame_calamity for backward compat
        if state.overworld.endgame_calamity.is_none() {
            let calamity = match template.crisis_type.as_str() {
                "sleeping_king" => {
                    let fid = state
                        .factions
                        .iter()
                        .find(|f| {
                            matches!(
                                f.diplomatic_stance,
                                DiplomaticStance::Hostile | DiplomaticStance::AtWar
                            )
                        })
                        .map(|f| f.id)
                        .unwrap_or(1);
                    Some(CalamityType::AggressiveFaction { faction_id: fid })
                }
                "breach" => Some(CalamityType::MajorMonster {
                    name: template.name.clone(),
                    strength: state.overworld.global_threat_level,
                }),
                "corruption" => Some(CalamityType::CrisisFlood),
                "decline" | "unifier" => Some(CalamityType::Conquest),
                _ => None,
            };
            if let Some(c) = calamity {
                state.overworld.endgame_calamity = Some(c);
            }
        }
    }
}
