//! Faction AI — every 600 ticks (~60s).
//!
//! Each faction takes a strategic action based on its personality and world state.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

pub fn tick_faction_ai(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.faction_ai;
    if state.tick % cfg.decision_interval_ticks != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions == 0 {
        return;
    }

    let attack_power_fraction = cfg.attack_power_fraction;
    let territory_capture_control = cfg.territory_capture_control;
    let hostile_strength_gain = cfg.hostile_strength_gain;
    let war_declaration_threshold = cfg.war_declaration_threshold;
    let war_declaration_penalty = cfg.war_declaration_penalty;
    let neutral_control_gain = cfg.neutral_control_gain;
    let friendly_relationship_gain = cfg.friendly_relationship_gain;
    let max_recent_actions = cfg.max_recent_actions;

    for fi in 0..n_factions {
        let faction = &state.factions[fi];
        let strength = faction.military_strength;
        let stance = faction.diplomatic_stance;

        // Simple AI: action depends on stance
        let action_desc = match stance {
            DiplomaticStance::AtWar => {
                // Attack guild territory aggressively
                if let Some(region) = state
                    .overworld
                    .regions
                    .iter_mut()
                    .find(|r| r.owner_faction_id == state.diplomacy.guild_faction_id && r.control > 5.0)
                {
                    let attack_power = strength * attack_power_fraction;
                    region.control = (region.control - attack_power).max(0.0);
                    region.unrest = (region.unrest + attack_power * 0.5).min(100.0);
                    // If control drops to 0, take the region
                    if region.control <= 0.0 {
                        region.owner_faction_id = fi;
                        region.control = territory_capture_control;
                    }
                    format!("Attacked region {}", region.name)
                } else {
                    "No valid targets".into()
                }
            }
            DiplomaticStance::Hostile => {
                // Build up military, then declare war
                state.factions[fi].military_strength += hostile_strength_gain;
                if state.factions[fi].military_strength > war_declaration_threshold {
                    state.factions[fi].diplomatic_stance = DiplomaticStance::AtWar;
                    state.factions[fi].relationship_to_guild =
                        (state.factions[fi].relationship_to_guild - war_declaration_penalty).max(-100.0);
                    "Declared war on the guild!".into()
                } else {
                    "Recruited forces".into()
                }
            }
            DiplomaticStance::Neutral => {
                // Defend territory
                for region in &mut state.overworld.regions {
                    if region.owner_faction_id == fi {
                        region.control = (region.control + neutral_control_gain).min(100.0);
                    }
                }
                "Fortified borders".into()
            }
            DiplomaticStance::Friendly => {
                // Improve relations with guild
                state.factions[fi].relationship_to_guild =
                    (state.factions[fi].relationship_to_guild + friendly_relationship_gain).min(100.0);
                "Sent diplomatic envoy to guild".into()
            }
        };

        state.factions[fi].recent_actions.push(FactionActionRecord {
            tick: state.tick,
            action: action_desc.clone(),
        });
        // Keep recent actions bounded
        if state.factions[fi].recent_actions.len() > max_recent_actions {
            state.factions[fi].recent_actions.remove(0);
        }

        events.push(WorldEvent::FactionActionTaken {
            faction_id: fi,
            action: action_desc,
        });
    }
}
