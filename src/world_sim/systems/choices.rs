#![allow(unused)]
//! Choice event generation — fires every 10 ticks.
//!
//! Generates branching decisions from templates. In the delta architecture,
//! choices are primarily narrative/UI events. The only mechanical effect is
//! gold or treasury changes from choice outcomes.
//!
//! Original: `crates/headless_campaign/src/systems/choices.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::{entity_hash_f32};

/// World event choices check cadence.
const WORLD_EVENT_INTERVAL: u64 = 10;


pub fn compute_choices(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick == 0 || state.tick % WORLD_EVENT_INTERVAL != 0 {
        return;
    }

    // Without choice templates and pending_choices on WorldState, we approximate:
    // Occasionally generate small gold windfalls or costs to settlements as
    // "resolved choice outcomes".

    for settlement in &state.settlements {
        let roll = entity_hash_f32(settlement.id, state.tick, 0xC401CE);
        if roll < 0.05 {
            // Random event: small gold change (positive or negative)
            let amount_roll = entity_hash_f32(settlement.id, state.tick, 0xA3B7);
            let amount = if amount_roll < 0.5 { 10.0 } else { -5.0 };
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: amount,
            });
        }
    }
}
