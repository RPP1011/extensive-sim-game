#![allow(unused)]
//! Commodity futures market — ticks every 17 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/commodity_futures.rs`.
//! Entities can hold futures contracts on commodities. On settlement,
//! profit/loss is the difference between the strike price and the current
//! market price. New contracts are generated based on current commodity
//! prices with a random premium or discount.
//!
//!   (contract_id, commodity: usize, quantity, strike_price, settlement_tick, is_buy)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// How often the futures market ticks.
const FUTURES_TICK_INTERVAL: u64 = 17;

/// Maximum number of active contracts per settlement.
const MAX_CONTRACTS_PER_SETTLEMENT: usize = 5;

/// Ticks until a contract settles.
const SETTLEMENT_HORIZON: u64 = 17;

/// Maximum premium/discount fraction (±30%).
const MAX_PREMIUM_FRACTION: f32 = 0.3;

/// Compute commodity futures deltas.
///
/// Without dedicated futures state, this system approximates contract
/// settlement by comparing historical prices (proxy: current stockpile
/// levels) and distributing profit/loss as treasury changes.
///
/// Settlements with volatile prices (high variance across commodities)
/// see futures activity that stabilizes treasury income.
pub fn compute_commodity_futures(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FUTURES_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        // Compute price volatility as proxy for futures activity.
        let avg_price: f32 =
            settlement.prices.iter().sum::<f32>() / settlement.prices.len().max(1) as f32;
        if avg_price <= 0.0 {
            continue;
        }

        let variance: f32 = settlement
            .prices
            .iter()
            .map(|p| (p - avg_price).powi(2))
            .sum::<f32>()
            / settlement.prices.len().max(1) as f32;
        let volatility = variance.sqrt();

        if volatility < 0.5 {
            continue; // Prices are stable — no futures pressure.
        }

        // Futures settlement: high volatility generates speculative profit
        // or loss for the settlement treasury.
        // Use tick parity as a deterministic stand-in for contract direction.
        let cycle = (state.tick / FUTURES_TICK_INTERVAL) % 3;
        let profit = match cycle {
            0 => volatility * 1.5,  // Speculators profit.
            1 => -volatility * 0.8, // Speculators lose.
            _ => volatility * 0.3,  // Mild gain.
        };

        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement.id,
            delta: profit,
        });

        // High volatility also pushes prices toward the mean (stabilizing
        // effect of futures markets).
        let mut stabilized_prices = settlement.prices;
        for c in 0..NUM_COMMODITIES {
            let diff = settlement.prices[c] - avg_price;
            stabilized_prices[c] -= diff * 0.05; // Nudge 5% toward mean.
            stabilized_prices[c] = stabilized_prices[c].max(0.01);
        }

        out.push(WorldDelta::UpdatePrices {
            location_id: settlement.id,
            prices: stabilized_prices,
        });
    }
}
