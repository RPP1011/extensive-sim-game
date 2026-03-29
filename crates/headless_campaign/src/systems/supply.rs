//! Supply drain — every tick.
//!
//! Parties in the field consume supplies proportional to their size.
//! Emits `PartySupplyLow` when supply drops below 20%.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{CampaignState, PartyStatus, CAMPAIGN_TURN_SECS};

pub fn tick_supply(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let dt_sec = CAMPAIGN_TURN_SECS as f32;
    let drain_rate = state.config.supply.drain_per_member_per_sec;
    let low_threshold = state.config.supply.low_threshold;

    for party in &mut state.parties {
        if matches!(party.status, PartyStatus::Idle) {
            continue;
        }

        let member_count = party.member_ids.len() as f32;
        let drain = drain_rate * member_count * dt_sec;
        let old_supply = party.supply_level;
        party.supply_level = (party.supply_level - drain).max(0.0);

        // Emit warning on crossing threshold
        if old_supply >= low_threshold && party.supply_level < low_threshold {
            events.push(WorldEvent::PartySupplyLow {
                party_id: party.id,
                supply_level: party.supply_level,
            });
        }
    }
}
