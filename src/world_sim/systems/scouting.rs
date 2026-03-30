//! Scouting and fog-of-war system — every tick.
//!
//! NPC entities traveling near settlements generate price reports,
//! sharing market information. High-fidelity grids with friendlies
//! get visibility benefits (modeled as price report freshness).
//!
//! Original: `crates/headless_campaign/src/systems/scouting.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, EntityKind, PriceReport, WorldState};

/// Cadence: runs every tick.
const SCOUTING_INTERVAL: u64 = 1;

/// Distance at which an NPC "scouts" a settlement and learns its prices.
const SCOUT_RANGE_SQ: f32 = 225.0; // 15 units

/// Staleness: price reports older than this many ticks are ignored.
const PRICE_REPORT_STALENESS: u64 = 200;

pub fn compute_scouting(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SCOUTING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Scouting in the delta architecture: NPCs near settlements share
    // price reports. This mirrors the original system's visibility-based
    // scout reports but uses the SharePriceReport delta.

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Only traveling NPCs perform scouting.
        let is_traveling = matches!(
            npc.economic_intent,
            EconomicIntent::Travel { .. } | EconomicIntent::Trade { .. }
        );
        if !is_traveling {
            continue;
        }

        // Check proximity to each settlement.
        for settlement in &state.settlements {
            let dx = entity.pos.0 - settlement.pos.0;
            let dy = entity.pos.1 - settlement.pos.1;
            if dx * dx + dy * dy > SCOUT_RANGE_SQ {
                continue;
            }

            // NPC already knows this settlement's prices at current tick?
            let already_fresh = npc.price_knowledge.iter().any(|pr| {
                pr.settlement_id == settlement.id
                    && state.tick.saturating_sub(pr.tick_observed) < PRICE_REPORT_STALENESS
            });
            if already_fresh {
                continue;
            }

            // Share a price report from this settlement to the NPC.
            out.push(WorldDelta::SharePriceReport {
                from_id: settlement.id,
                to_id: entity.id,
                report: PriceReport {
                    settlement_id: settlement.id,
                    prices: settlement.prices,
                    tick_observed: state.tick,
                },
            });
        }
    }
}
