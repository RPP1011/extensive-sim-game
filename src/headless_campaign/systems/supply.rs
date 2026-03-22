//! Supply drain — every tick.
//!
//! Parties in the field consume supplies proportional to their size.
//! Emits `PartySupplyLow` when supply drops below 20%.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{CampaignState, PartyStatus, CAMPAIGN_TICK_MS};

/// Supply drain rate: units per member per second.
const SUPPLY_DRAIN_PER_MEMBER_PER_SEC: f32 = 0.02;
/// Low supply warning threshold.
const LOW_SUPPLY_THRESHOLD: f32 = 20.0;

pub fn tick_supply(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let dt_sec = CAMPAIGN_TICK_MS as f32 / 1000.0;

    for party in &mut state.parties {
        if matches!(party.status, PartyStatus::Idle) {
            continue;
        }

        let member_count = party.member_ids.len() as f32;
        let drain = SUPPLY_DRAIN_PER_MEMBER_PER_SEC * member_count * dt_sec;
        let old_supply = party.supply_level;
        party.supply_level = (party.supply_level - drain).max(0.0);

        // Emit warning on crossing threshold
        if old_supply >= LOW_SUPPLY_THRESHOLD && party.supply_level < LOW_SUPPLY_THRESHOLD {
            events.push(WorldEvent::PartySupplyLow {
                party_id: party.id,
                supply_level: party.supply_level,
            });
        }
    }
}
