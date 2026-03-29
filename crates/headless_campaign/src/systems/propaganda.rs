//! Propaganda and public relations system.
//!
//! Allows the guild to spend gold on influence campaigns that boost reputation,
//! counter rival guilds, discredit hostile factions, recruit adventurers, or
//! raise morale during wartime.
//!
//! Ticks every 200 ticks. Max 2 active campaigns at once.
//! Campaign effectiveness decays over time (diminishing returns).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Cadence: propaganda effects apply every 200 ticks.
const TICK_CADENCE: u64 = 7;

/// Run the propaganda system for one tick.
///
/// - Active campaigns apply their effects with decaying effectiveness.
/// - Expired campaigns are removed and generate events.
pub fn tick_propaganda(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    if state.propaganda_campaigns.is_empty() {
        return;
    }

    // Collect effects to apply (avoid borrowing state mutably in the loop).
    struct CampaignEffect {
        id: u32,
        campaign_type: PropagandaType,
        target_region_id: Option<usize>,
        target_faction_id: Option<usize>,
        effectiveness: f32,
    }

    let mut effects = Vec::new();
    let mut expired_ids = Vec::new();

    for campaign in &state.propaganda_campaigns {
        let elapsed = state.tick.saturating_sub(campaign.started_tick);
        if elapsed >= campaign.duration {
            expired_ids.push((campaign.id, campaign.campaign_type));
        } else {
            effects.push(CampaignEffect {
                id: campaign.id,
                campaign_type: campaign.campaign_type,
                target_region_id: campaign.target_region_id,
                target_faction_id: campaign.target_faction_id,
                effectiveness: campaign.effectiveness,
            });
        }
    }

    // Apply effects
    for effect in &effects {
        apply_campaign_effect(state, effect.id, effect.campaign_type, effect.target_region_id, effect.target_faction_id, effect.effectiveness, events);
    }

    // Decay effectiveness on active campaigns
    for campaign in &mut state.propaganda_campaigns {
        let elapsed = state.tick.saturating_sub(campaign.started_tick);
        if elapsed < campaign.duration {
            // Effectiveness decays linearly from 1.0 to 0.2 over the campaign duration
            let progress = elapsed as f32 / campaign.duration as f32;
            campaign.effectiveness = 1.0 - 0.8 * progress;
        }
    }

    // Remove expired campaigns and emit events
    for (id, ctype) in &expired_ids {
        events.push(WorldEvent::PropagandaExpired {
            campaign_id: *id,
            campaign_type: format!("{:?}", ctype),
        });
    }
    state.propaganda_campaigns.retain(|c| {
        let elapsed = state.tick.saturating_sub(c.started_tick);
        elapsed < c.duration
    });
}

/// Apply a single campaign's effect for this tick.
fn apply_campaign_effect(
    state: &mut CampaignState,
    campaign_id: u32,
    campaign_type: PropagandaType,
    target_region_id: Option<usize>,
    target_faction_id: Option<usize>,
    effectiveness: f32,
    events: &mut Vec<WorldEvent>,
) {
    match campaign_type {
        PropagandaType::BoostReputation => {
            // +0.5 reputation per tick (scaled by effectiveness) in target region
            let boost = 0.5 * effectiveness;
            state.guild.reputation = (state.guild.reputation + boost).min(100.0);

            // Also reduce unrest in target region
            if let Some(rid) = target_region_id {
                if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
                    region.unrest = (region.unrest - 0.2 * effectiveness).max(0.0);
                }
            }

            events.push(WorldEvent::PropagandaEffect {
                campaign_id,
                description: format!(
                    "Reputation campaign boosted guild standing by {:.1}",
                    boost
                ),
            });
        }

        PropagandaType::CounterRival => {
            // -1.0 rival guild reputation per tick (scaled by effectiveness)
            let reduction = 1.0 * effectiveness;
            state.rival_guild.reputation =
                (state.rival_guild.reputation - reduction).max(0.0);

            events.push(WorldEvent::PropagandaEffect {
                campaign_id,
                description: format!(
                    "Counter-propaganda reduced rival reputation by {:.1}",
                    reduction
                ),
            });
        }

        PropagandaType::DiscreditFaction => {
            // -2.0 to target faction's relations with all other factions (scaled)
            let penalty = 2.0 * effectiveness;
            if let Some(fid) = target_faction_id {
                // Reduce faction-to-faction relations in the diplomacy matrix
                let n = state.diplomacy.relations.len();
                if fid < n {
                    for other in 0..n {
                        if other != fid {
                            state.diplomacy.relations[fid][other] =
                                (state.diplomacy.relations[fid][other] as f32 - penalty)
                                    as i32;
                            state.diplomacy.relations[other][fid] =
                                (state.diplomacy.relations[other][fid] as f32 - penalty)
                                    as i32;
                        }
                    }
                }

                events.push(WorldEvent::PropagandaEffect {
                    campaign_id,
                    description: format!(
                        "Discredit campaign damaged faction {} relations by {:.1}",
                        fid, penalty
                    ),
                });
            }
        }

        PropagandaType::RecruitmentDrive => {
            // +20% recruitment chance in target region — implemented as a
            // morale and reputation bump that makes recruitment system more
            // likely to fire. The recruitment system reads guild.reputation.
            let boost = 0.3 * effectiveness;
            state.guild.reputation = (state.guild.reputation + boost).min(100.0);

            // Also slightly improve faction relations in the region's owning faction
            if let Some(rid) = target_region_id {
                if let Some(region) = state.overworld.regions.iter().find(|r| r.id == rid) {
                    let owner_id = region.owner_faction_id;
                    if let Some(faction) = state.factions.iter_mut().find(|f| f.id == owner_id) {
                        faction.relationship_to_guild =
                            (faction.relationship_to_guild + 0.3 * effectiveness).min(100.0);
                    }
                }
            }

            events.push(WorldEvent::PropagandaEffect {
                campaign_id,
                description: "Recruitment drive increased guild visibility".into(),
            });
        }

        PropagandaType::WarPropaganda => {
            // +5 morale for guild adventurers, +5 civilian morale (via unrest reduction)
            // in guild-owned regions (all scaled by effectiveness)
            let morale_boost = 5.0 * effectiveness;
            let unrest_reduction = 5.0 * effectiveness;

            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead && adv.faction_id.is_none() {
                    adv.morale = (adv.morale + morale_boost).min(100.0);
                }
            }

            let guild_fid = state.diplomacy.guild_faction_id;
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == guild_fid {
                    region.unrest = (region.unrest - unrest_reduction).max(0.0);
                }
            }

            events.push(WorldEvent::PropagandaEffect {
                campaign_id,
                description: format!(
                    "War propaganda boosted morale by {:.1} and reduced unrest by {:.1}",
                    morale_boost, unrest_reduction
                ),
            });
        }
    }
}
