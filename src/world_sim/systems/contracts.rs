#![allow(unused)]
//! Guild contracts / commissions — fires every 10 ticks.
//!
//! NPCs and factions commission work with deadlines and penalties. Completed
//! contracts reward gold; failed contracts penalize. Maps to UpdateTreasury
//! deltas for rewards and penalties.
//!
//! Original: `crates/headless_campaign/src/systems/contracts.rs`
//!
//! NEEDS STATE: `contracts: Vec<Contract>` on WorldState
//! NEEDS DELTA: OfferContract, CompleteContract, FailContract

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::{entity_hash_f32};

/// Contract tick interval.
const CONTRACT_TICK_INTERVAL: u64 = 10;

/// Contract refresh interval.
const CONTRACT_REFRESH_INTERVAL: u64 = 33;

/// Reward for contract completion (distributed to settlement).
const CONTRACT_REWARD: f32 = 25.0;

/// Penalty for contract failure.
const CONTRACT_PENALTY: f32 = 10.0;


pub fn compute_contracts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CONTRACT_TICK_INTERVAL != 0 {
        return;
    }

    // Without contract tracking state, we approximate:
    // Settlements periodically receive contract rewards or penalties based
    // on their overall health (treasury, population).

    for settlement in &state.settlements {
        let roll = entity_hash_f32(settlement.id, state.tick, 0xC0EEACB);

        // Well-functioning settlements complete contracts
        if settlement.treasury > 30.0 && settlement.population > 50 && roll < 0.08 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: CONTRACT_REWARD,
            });
        }

        // Struggling settlements fail contracts (don't pile on if already deep in debt)
        if settlement.treasury < 10.0 && settlement.treasury > -100.0 && roll > 0.95 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -CONTRACT_PENALTY,
            });
        }
    }
}
