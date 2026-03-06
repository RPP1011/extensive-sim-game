use bevy::prelude::*;

use super::overworld_types::OverworldMap;
use super::roster_types::*;
use super::types::RunState;

pub fn sync_campaign_parties_with_roster_system(
    overworld: Res<OverworldMap>,
    roster: Res<CampaignRoster>,
    mut parties: ResMut<CampaignParties>,
) {
    normalize_campaign_parties(&mut parties, &roster, &overworld);
}

pub fn campaign_party_orders_system(
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    mut parties: ResMut<CampaignParties>,
) {
    if run_state.global_turn == 0 || run_state.global_turn % 6 != 0 || overworld.regions.is_empty()
    {
        return;
    }
    let player_region = parties
        .parties
        .iter()
        .find(|p| p.is_player_controlled)
        .map(|p| p.region_id)
        .unwrap_or(
            overworld
                .current_region
                .min(overworld.regions.len().saturating_sub(1)),
        );
    for party in parties
        .parties
        .iter_mut()
        .filter(|p| !p.is_player_controlled)
    {
        let old_region = party.region_id;
        let next = match party.order {
            PartyOrderKind::HoldPosition => None,
            PartyOrderKind::PatrolNearby => {
                if let Some(target) = party.order_target_region_id {
                    if target != party.region_id {
                        next_region_step(&overworld, party.region_id, target)
                    } else {
                        let neighbors = &overworld.regions[party.region_id].neighbors;
                        if neighbors.is_empty() {
                            None
                        } else {
                            let idx = ((run_state.global_turn as usize) + (party.id as usize))
                                % neighbors.len();
                            Some(neighbors[idx])
                        }
                    }
                } else {
                    let neighbors = &overworld.regions[party.region_id].neighbors;
                    if neighbors.is_empty() {
                        None
                    } else {
                        let idx = ((run_state.global_turn as usize) + (party.id as usize))
                            % neighbors.len();
                        Some(neighbors[idx])
                    }
                }
            }
            PartyOrderKind::ReinforceFront => party
                .order_target_region_id
                .and_then(|target| next_region_step(&overworld, party.region_id, target)),
            PartyOrderKind::RecruitAndTrain => {
                let target = party.order_target_region_id.unwrap_or(player_region);
                next_region_step(&overworld, party.region_id, target)
            }
        };
        if let Some(next_region) = next {
            party.region_id = next_region.min(overworld.regions.len().saturating_sub(1));
        }
        party.supply = if party.region_id == old_region {
            (party.supply + 1.5).min(100.0)
        } else {
            (party.supply - 3.0).max(0.0)
        };
    }
}

pub(crate) fn ensure_faction_mission_slots(overworld: &mut OverworldMap) {
    for region in &mut overworld.regions {
        region.mission_slot = None;
    }
    for slot in 0..overworld.factions.len() {
        if let Some((idx, _)) = overworld
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.owner_faction_id == slot)
            .max_by(|(_, a), (_, b)| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
        {
            overworld.regions[idx].mission_slot = Some(slot);
        }
    }
}
