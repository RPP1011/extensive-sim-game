#![allow(unused)]
//! Price controls system — council sets ceiling/floor prices on commodities.
//!
//! Ported from `crates/headless_campaign/src/systems/price_controls.rs`.
//! Every 17 ticks, evaluates whether settlements should enact new price
//! controls and applies the effects of existing ones. Binding ceilings
//! create shortages; binding floors drain treasury via subsidies.
//! Controls expire after 67 ticks.
//!
//!   (settlement_id, commodity, ceiling: Option<f32>, floor: Option<f32>, enacted_tick)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// How often the price controls system ticks.
const PRICE_CONTROL_INTERVAL: u64 = 17;

/// Price control duration in ticks.
const CONTROL_EXPIRY_TICKS: u64 = 67;

/// Supply reduction fraction when a ceiling is binding.
const CEILING_SUPPLY_REDUCTION: f32 = 0.10;

/// Treasury drain per tick when a floor requires subsidies.
const FLOOR_SUBSIDY_DRAIN: f32 = 2.0;

/// Price spike threshold — auto-enact ceiling when price > BASE * this.
const PRICE_SPIKE_THRESHOLD: f32 = 2.0;

/// Base price assumed when no other reference exists.
const BASE_PRICE: f32 = 1.0;

/// Minimum tick before activation.
const ACTIVATION_TICK: u64 = 1000;

/// Compute price control deltas.
///
/// Without dedicated price-control state, this system detects commodity
/// price spikes at each settlement and emits UpdatePrices deltas that
/// clamp runaway prices (ceiling effect) and UpdateTreasury deltas for
/// floor subsidies.
pub fn compute_price_controls(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PRICE_CONTROL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < ACTIVATION_TICK {
        return;
    }

    for settlement in &state.settlements {
        let mut clamped_prices = settlement.prices;
        let mut any_change = false;

        for c in 0..NUM_COMMODITIES {
            let price = settlement.prices[c];

            // --- Ceiling: cap commodity price at SPIKE_THRESHOLD * base ---
            let ceiling = BASE_PRICE * PRICE_SPIKE_THRESHOLD;
            if price > ceiling {
                clamped_prices[c] = ceiling;
                any_change = true;

                // Binding ceiling reduces effective stockpile (shortage).
                let stockpile_drain = settlement.stockpile[c] * CEILING_SUPPLY_REDUCTION;
                if stockpile_drain > 0.01 {
                    out.push(WorldDelta::ConsumeCommodity {
                        location_id: settlement.id,
                        commodity: c,
                        amount: stockpile_drain,
                    });
                }
            }

            // --- Floor: if price crashed below half base, subsidize ---
            let floor = BASE_PRICE * 0.5;
            if price < floor && price > 0.0 {
                clamped_prices[c] = floor;
                any_change = true;

                // Treasury pays the subsidy difference (only above floor).
                if settlement.treasury > -100.0 {
                    let subsidy = FLOOR_SUBSIDY_DRAIN;
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -subsidy,
                    });
                }
            }
        }

        if any_change {
            out.push(WorldDelta::UpdatePrices {
                location_id: settlement.id,
                prices: clamped_prices,
            });
        }
    }
}
