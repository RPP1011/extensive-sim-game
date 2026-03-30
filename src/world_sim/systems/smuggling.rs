#![allow(unused)]
//! Smuggling routes system — fires every 10 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/smuggling.rs`.
//! Secret trade routes between settlements for high profit but high risk.
//! Routes can be busted, causing gold penalties and reputation loss.
//!
//!   (id, start_settlement_id, end_settlement_id, route_type, profit_per_trip,
//!    risk, active, trips_completed, busted_count, suspended_until)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// How often smuggling routes tick.
const SMUGGLING_INTERVAL: u64 = 10;

/// Minimum tick before activation.
const ACTIVATION_TICK: u64 = 1500;

/// Profit fraction that flows as treasury delta at each endpoint.
const PROFIT_TREASURY_FRACTION: f32 = 0.15;

/// Transit loss fraction when moving goods between settlements.
const TRANSIT_LOSS: f32 = 0.25;

/// Minimum price ratio between two settlements to justify smuggling.
const SMUGGLING_PRICE_RATIO: f32 = 2.0;

/// Compute smuggling deltas.
///
/// Without explicit smuggling-route state, this system finds pairs of
/// settlements where price differentials are large enough to justify
/// illicit trade, and moves goods from the cheap source to the expensive
/// destination (with transit losses and treasury effects).
pub fn compute_smuggling(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SMUGGLING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < ACTIVATION_TICK {
        return;
    }

    if state.settlements.len() < 2 {
        return;
    }

    // For each commodity, find the cheapest and most expensive settlements.
    for c in 0..NUM_COMMODITIES {
        let source = state
            .settlements
            .iter()
            .filter(|s| s.stockpile[c] > 3.0)
            .min_by(|a, b| {
                a.prices[c]
                    .partial_cmp(&b.prices[c])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let dest = state
            .settlements
            .iter()
            .max_by(|a, b| {
                a.prices[c]
                    .partial_cmp(&b.prices[c])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let (src, dst) = match (source, dest) {
            (Some(s), Some(d)) if s.id != d.id => (s, d),
            _ => continue,
        };

        let price_ratio = dst.prices[c] / src.prices[c].max(0.01);
        if price_ratio < SMUGGLING_PRICE_RATIO {
            continue;
        }

        // Move goods (smuggling bypasses normal caravan rules).
        let amount = 1.0_f32.min(src.stockpile[c] * 0.1);
        if amount < 0.01 {
            continue;
        }

        // Drain from source.
        out.push(WorldDelta::UpdateStockpile {
            location_id: src.id,
            commodity: c,
            delta: -amount,
        });

        // Deliver to destination (minus transit loss).
        let delivered = amount * (1.0 - TRANSIT_LOSS);
        out.push(WorldDelta::UpdateStockpile {
            location_id: dst.id,
            commodity: c,
            delta: delivered,
        });

        // Profit flows into source settlement treasury.
        let profit = (dst.prices[c] - src.prices[c]) * amount * PROFIT_TREASURY_FRACTION;
        if profit > 0.0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: src.id,
                delta: profit,
            });
        }
    }
}
