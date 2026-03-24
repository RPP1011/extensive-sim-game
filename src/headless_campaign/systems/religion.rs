//! Religious orders and temples system — fires every 500 ticks.
//!
//! Temples in regions provide blessings and quests. Religious orders are
//! faction-like entities that offer unique services based on order type.
//! Devotion drifts toward 0 over time and is raised through donations
//! and quest completions in the temple's region.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the religion system ticks (in ticks).
const RELIGION_INTERVAL: u64 = 500;

/// Devotion cost to request a blessing.
pub const BLESSING_DEVOTION_COST: f32 = 30.0;

/// How long a blessing lasts (in ticks).
pub const BLESSING_DURATION: u64 = 1000;

/// Devotion decay per religion tick.
const DEVOTION_DECAY: f32 = 1.0;

/// Devotion gained per quest completion in the temple's region.
pub const QUEST_COMPLETION_DEVOTION: f32 = 3.0;

/// Devotion threshold below which the temple becomes hostile.
const HOSTILITY_THRESHOLD: f32 = -20.0;

/// Tick the religion system: devotion drift, blessing expiry, and bonuses.
pub fn tick_religion(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % RELIGION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let tick = state.tick;

    for i in 0..state.temples.len() {
        // Devotion drifts toward 0
        let old_devotion = state.temples[i].devotion;
        if state.temples[i].devotion > 0.0 {
            state.temples[i].devotion = (state.temples[i].devotion - DEVOTION_DECAY).max(0.0);
        } else if state.temples[i].devotion < 0.0 {
            state.temples[i].devotion = (state.temples[i].devotion + DEVOTION_DECAY).min(0.0);
        }

        // Check blessing expiry
        if state.temples[i].blessing_active && tick >= state.temples[i].blessing_expires_tick {
            state.temples[i].blessing_active = false;
            let name = state.temples[i].name.clone();
            events.push(WorldEvent::BlessingExpired {
                temple_name: name,
            });
        }

        // Apply active blessing bonuses
        if state.temples[i].blessing_active {
            let region_id = state.temples[i].region_id;
            let order = state.temples[i].order;
            apply_blessing_bonuses(state, order, region_id);
        }

        // Low devotion penalty: emit warning
        if state.temples[i].devotion < HOSTILITY_THRESHOLD && old_devotion >= HOSTILITY_THRESHOLD {
            let name = state.temples[i].name.clone();
            events.push(WorldEvent::TempleDevotion {
                temple_name: name.clone(),
                change: state.temples[i].devotion - old_devotion,
            });
        }
    }
}

/// Apply blessing bonuses for a given order in a region.
fn apply_blessing_bonuses(state: &mut CampaignState, order: ReligiousOrder, region_id: usize) {
    match order {
        ReligiousOrder::OrderOfLight => {
            // +5 morale to adventurers in parties traveling in this region
            for party in &state.parties {
                if party.status == PartyStatus::Traveling || party.status == PartyStatus::OnMission {
                    // Check if party is in the blessed region (approximate by faction owner)
                    if let Some(region) = state.overworld.regions.get(region_id) {
                        if region.owner_faction_id == state.diplomacy.guild_faction_id {
                            for &mid in &party.member_ids {
                                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                                    adv.morale = (adv.morale + 5.0).min(100.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        ReligiousOrder::BrotherhoodOfSteel => {
            // +10% combat power handled via battle prediction modifier
            // Crafting quality bonus applied through guild market
            // (Applied passively when battles resolve in tick_battles)
        }
        ReligiousOrder::CircleOfNature => {
            // Disease immunity in region — reduce injury for adventurers
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead && adv.injury > 0.0 {
                    adv.injury = (adv.injury - 1.0).max(0.0);
                }
            }
        }
        ReligiousOrder::ShadowCovenant => {
            // +20% spy cover — boost visibility of enemy regions
            if let Some(region) = state.overworld.regions.get_mut(region_id) {
                region.visibility = (region.visibility + 0.02).min(1.0);
            }
        }
        ReligiousOrder::ScholarsGuild => {
            // +15% XP for idle adventurers (studying)
            for adv in &mut state.adventurers {
                if adv.status == AdventurerStatus::Idle {
                    adv.xp += 5;
                }
            }
        }
    }
}

/// Check if a given religious order has an active blessing in any temple.
pub fn has_active_blessing(state: &CampaignState, order: ReligiousOrder) -> bool {
    state.temples.iter().any(|t| t.order == order && t.blessing_active)
}

/// Get the combat power multiplier from Brotherhood of Steel blessing.
pub fn brotherhood_combat_bonus(state: &CampaignState) -> f32 {
    if has_active_blessing(state, ReligiousOrder::BrotherhoodOfSteel) {
        1.10
    } else {
        1.0
    }
}

/// Get the XP multiplier from Scholars Guild blessing.
pub fn scholars_xp_bonus(state: &CampaignState) -> f32 {
    if has_active_blessing(state, ReligiousOrder::ScholarsGuild) {
        1.15
    } else {
        1.0
    }
}

/// Boost devotion for quest completions in a temple's region.
pub fn on_quest_completed_in_region(state: &mut CampaignState, region_id: usize) {
    for temple in &mut state.temples {
        if temple.region_id == region_id {
            temple.devotion = (temple.devotion + QUEST_COMPLETION_DEVOTION).min(100.0);
        }
    }
}
