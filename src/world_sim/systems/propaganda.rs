#![allow(unused)]
//! Propaganda and public relations system — every 7 ticks.
//!
//! Allows the guild to spend gold on influence campaigns that boost reputation,
//! counter rival guilds, discredit hostile factions, recruit adventurers, or
//! raise morale during wartime.
//!
//! Max 2 active campaigns at once. Effectiveness decays over time.
//!
//! Ported from `crates/headless_campaign/src/systems/propaganda.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: propaganda_campaigns: Vec<PropagandaCampaign> on WorldState
//   PropagandaCampaign { id, campaign_type: PropagandaType, started_tick, duration,
//                        effectiveness, target_region_id: Option<u32>,
//                        target_faction_id: Option<u32> }
// NEEDS STATE: PropagandaType enum { BoostReputation, CounterRival, DiscreditFaction,
//              RecruitmentDrive, WarPropaganda }
// NEEDS STATE: guild: GuildState { reputation } on WorldState
// NEEDS STATE: rival_guild: RivalGuild { reputation } on WorldState
// NEEDS STATE: factions: Vec<FactionState> on WorldState
// NEEDS STATE: diplomacy: DiplomacyState { relations, guild_faction_id } on WorldState
// NEEDS STATE: regions with owner_faction_id, unrest fields
// NEEDS STATE: adventurers with morale, faction_id, status fields

// NEEDS DELTA: AdjustGuildReputation { delta: f32 }
// NEEDS DELTA: AdjustRivalReputation { delta: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: AdjustDiplomacyRelation { faction_a: u32, faction_b: u32, delta: i32 }
// NEEDS DELTA: AdjustRegionUnrest { region_id: u32, delta: f32 }
// NEEDS DELTA: AdjustMorale { adventurer_id: u32, delta: f32 }
// NEEDS DELTA: SetCampaignEffectiveness { campaign_id: u32, value: f32 }
// NEEDS DELTA: RemoveCampaign { campaign_id: u32 }

/// Cadence: propaganda effects apply every 7 ticks.
const TICK_CADENCE: u64 = 7;

pub fn compute_propaganda(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    // Once state.propaganda_campaigns, state.guild, state.rival_guild,
    // state.factions, state.diplomacy, state.regions, state.adventurers exist,
    // enable this.

    /*
    if state.propaganda_campaigns.is_empty() {
        return;
    }

    for campaign in &state.propaganda_campaigns {
        let elapsed = state.tick.saturating_sub(campaign.started_tick);

        // Check expiry.
        if elapsed >= campaign.duration {
            out.push(WorldDelta::RemoveCampaign {
                campaign_id: campaign.id,
            });
            continue;
        }

        // Compute decaying effectiveness: 1.0 -> 0.2 over duration.
        let progress = elapsed as f32 / campaign.duration as f32;
        let effectiveness = 1.0 - 0.8 * progress;

        out.push(WorldDelta::SetCampaignEffectiveness {
            campaign_id: campaign.id,
            value: effectiveness,
        });

        // Apply campaign type effects.
        match campaign.campaign_type {
            PropagandaType::BoostReputation => {
                let boost = 0.5 * effectiveness;
                out.push(WorldDelta::AdjustGuildReputation { delta: boost });

                // Reduce unrest in target region.
                if let Some(rid) = campaign.target_region_id {
                    out.push(WorldDelta::AdjustRegionUnrest {
                        region_id: rid,
                        delta: -0.2 * effectiveness,
                    });
                }
            }

            PropagandaType::CounterRival => {
                let reduction = 1.0 * effectiveness;
                out.push(WorldDelta::AdjustRivalReputation {
                    delta: -reduction,
                });
            }

            PropagandaType::DiscreditFaction => {
                let penalty = 2.0 * effectiveness;
                if let Some(fid) = campaign.target_faction_id {
                    // Reduce target faction's relations with all other factions.
                    let n = state.diplomacy.relations.len();
                    if (fid as usize) < n {
                        for other in 0..n {
                            if other != fid as usize {
                                out.push(WorldDelta::AdjustDiplomacyRelation {
                                    faction_a: fid,
                                    faction_b: other as u32,
                                    delta: -(penalty as i32),
                                });
                                out.push(WorldDelta::AdjustDiplomacyRelation {
                                    faction_a: other as u32,
                                    faction_b: fid,
                                    delta: -(penalty as i32),
                                });
                            }
                        }
                    }
                }
            }

            PropagandaType::RecruitmentDrive => {
                let boost = 0.3 * effectiveness;
                out.push(WorldDelta::AdjustGuildReputation { delta: boost });

                // Improve relations with the region's owning faction.
                if let Some(rid) = campaign.target_region_id {
                    if let Some(region) = state.regions.iter().find(|r| r.id == rid) {
                        if let Some(owner_id) = region.faction_id {
                            out.push(WorldDelta::AdjustRelationship {
                                faction_id: owner_id,
                                delta: 0.3 * effectiveness,
                            });
                        }
                    }
                }
            }

            PropagandaType::WarPropaganda => {
                let morale_boost = 5.0 * effectiveness;
                let unrest_reduction = 5.0 * effectiveness;

                // Boost morale for guild adventurers (faction_id == None).
                for adv in &state.adventurers {
                    if adv.status != AdventurerStatus::Dead && adv.faction_id.is_none() {
                        out.push(WorldDelta::AdjustMorale {
                            adventurer_id: adv.id,
                            delta: morale_boost,
                        });
                    }
                }

                // Reduce unrest in guild-owned regions.
                let guild_fid = state.diplomacy.guild_faction_id;
                for region in &state.regions {
                    if region.faction_id == Some(guild_fid) {
                        out.push(WorldDelta::AdjustRegionUnrest {
                            region_id: region.id,
                            delta: -unrest_reduction,
                        });
                    }
                }
            }
        }
    }
    */
}
