#![allow(unused)]
//! Messenger system — fires every tick.
//!
//! Orders to distant entities are delayed by distance. In the delta
//! architecture, messengers are modeled as entities moving toward a target.
//! Delivery is represented by SharePriceReport (information transfer) or
//! Move deltas for the messenger entity.
//!
//! Original: `crates/headless_campaign/src/systems/messengers.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, EntityKind, WorldState};

/// Messenger check interval (every tick for responsiveness).
const MESSENGER_INTERVAL: u64 = 1;

/// Base messenger speed in world units per tick.
const MESSENGER_SPEED: f32 = 2.0;

pub fn compute_messengers(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MESSENGER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without pending_orders tracking, we approximate:
    // NPCs that are trading share price reports with their home settlement.
    // This represents the information-carrying role of messengers.

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Trading NPCs share price knowledge with their destination
        if let EconomicIntent::Trade {
            destination_settlement_id,
        } = &npc.economic_intent
        {
            // Share the NPC's price knowledge with the destination settlement
            for report in &npc.price_knowledge {
                out.push(WorldDelta::SharePriceReport {
                    from_id: entity.id,
                    to_id: *destination_settlement_id,
                    report: crate::world_sim::state::PriceReport {
                        settlement_id: report.settlement_id,
                        prices: report.prices,
                        tick_observed: report.tick_observed,
                    },
                });
            }
        }
    }
}
