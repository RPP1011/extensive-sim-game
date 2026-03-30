#![allow(unused)]
//! Contract negotiation — fires every 7 ticks.
//!
//! When contracts appear, factions negotiate terms. Success depends on
//! reputation and relationship. Counter-offers reduce patience; exhausted
//! patience withdraws the contract. In the delta architecture, negotiation
//! outcomes map to UpdateTreasury (adjusted rewards).
//!
//! Original: `crates/headless_campaign/src/systems/contract_negotiation.rs`
//!
//! NEEDS STATE: `negotiation_rounds: Vec<NegotiationState>` on WorldState
//! NEEDS DELTA: NegotiationStarted, NegotiationAccepted, NegotiationFailed

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::{entity_hash_f32};

/// Negotiation tick interval.
const NEGOTIATION_TICK_INTERVAL: u64 = 7;

/// Bonus gold from successful negotiation.
const NEGOTIATION_BONUS: f32 = 8.0;


pub fn compute_contract_negotiation(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % NEGOTIATION_TICK_INTERVAL != 0 {
        return;
    }

    // Without negotiation tracking, we approximate:
    // Settlements with high treasury (good reputation) have a chance of
    // receiving negotiation bonuses (better contract terms).

    for settlement in &state.settlements {
        if settlement.treasury < 40.0 {
            continue;
        }

        let roll = entity_hash_f32(settlement.id, state.tick, 0xAE60);
        if roll < 0.05 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: NEGOTIATION_BONUS,
            });
        }
    }
}
