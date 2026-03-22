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
    if state.tick % 600 != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions == 0 {
        return;
    }

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
                    let attack_power = strength * 0.1;
                    region.control = (region.control - attack_power).max(0.0);
                    region.unrest = (region.unrest + attack_power * 0.5).min(100.0);
                    // If control drops to 0, take the region
                    if region.control <= 0.0 {
                        region.owner_faction_id = fi;
                        region.control = 30.0;
                    }
                    format!("Attacked region {}", region.name)
                } else {
                    "No valid targets".into()
                }
            }
            DiplomaticStance::Hostile => {
                // Build up military, then declare war
                state.factions[fi].military_strength += 3.0;
                if state.factions[fi].military_strength > 80.0 {
                    state.factions[fi].diplomatic_stance = DiplomaticStance::AtWar;
                    state.factions[fi].relationship_to_guild =
                        (state.factions[fi].relationship_to_guild - 30.0).max(-100.0);
                    "Declared war on the guild!".into()
                } else {
                    "Recruited forces".into()
                }
            }
            DiplomaticStance::Neutral => {
                // Defend territory
                for region in &mut state.overworld.regions {
                    if region.owner_faction_id == fi {
                        region.control = (region.control + 1.0).min(100.0);
                    }
                }
                "Fortified borders".into()
            }
            DiplomaticStance::Friendly => {
                // Improve relations with guild
                state.factions[fi].relationship_to_guild =
                    (state.factions[fi].relationship_to_guild + 2.0).min(100.0);
                "Sent diplomatic envoy to guild".into()
            }
        };

        state.factions[fi].recent_actions.push(FactionActionRecord {
            tick: state.tick,
            action: action_desc.clone(),
        });
        // Keep recent actions bounded
        if state.factions[fi].recent_actions.len() > 10 {
            state.factions[fi].recent_actions.remove(0);
        }

        events.push(WorldEvent::FactionActionTaken {
            faction_id: fi,
            action: action_desc,
        });
    }
}
