//! Party travel — every tick.
//!
//! Moves parties toward their destinations at `speed` tiles/second.
//! Emits `PartyArrived` when a party reaches its destination.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState, PartyStatus, CAMPAIGN_TURN_SECS};

pub fn tick_travel(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let dt_sec = CAMPAIGN_TURN_SECS as f32;

    for party in &mut state.parties {
        if party.status != PartyStatus::Traveling && party.status != PartyStatus::Returning {
            continue;
        }

        let dest = match party.destination {
            Some(d) => d,
            None => continue,
        };

        let old_pos = party.position;
        let dx = dest.0 - party.position.0;
        let dy = dest.1 - party.position.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.01 {
            // Already at destination
            party.destination = None;
            if party.status == PartyStatus::Returning {
                party.status = PartyStatus::Idle;
                events.push(WorldEvent::PartyReturned { party_id: party.id });
            } else {
                party.status = PartyStatus::OnMission;
                events.push(WorldEvent::PartyArrived {
                    party_id: party.id,
                    location: dest,
                });
            }
            continue;
        }

        let move_dist = party.speed * dt_sec;

        if move_dist >= dist {
            // Arrive this tick
            party.position = dest;
            party.destination = None;
            if party.status == PartyStatus::Returning {
                party.status = PartyStatus::Idle;
                events.push(WorldEvent::PartyReturned { party_id: party.id });
            } else {
                party.status = PartyStatus::OnMission;
                events.push(WorldEvent::PartyArrived {
                    party_id: party.id,
                    location: dest,
                });
            }
        } else {
            // Move toward destination
            let nx = dx / dist;
            let ny = dy / dist;
            party.position.0 += nx * move_dist;
            party.position.1 += ny * move_dist;
        }

        deltas
            .party_position_changes
            .push((party.id, old_pos, party.position));
    }

    // Clean up idle parties: release adventurers and remove empty/idle parties
    let idle_party_ids: Vec<u32> = state
        .parties
        .iter()
        .filter(|p| p.status == PartyStatus::Idle)
        .map(|p| p.id)
        .collect();

    for pid in &idle_party_ids {
        for adv in &mut state.adventurers {
            if adv.party_id == Some(*pid)
                && adv.status != AdventurerStatus::Dead
                && adv.status != AdventurerStatus::Injured
            {
                adv.status = AdventurerStatus::Idle;
                adv.party_id = None;
            }
        }
    }

    state.parties.retain(|p| p.status != PartyStatus::Idle);
}
