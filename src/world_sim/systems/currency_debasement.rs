#![allow(unused)]
//! Currency debasement system — fires every 13 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/currency_debasement.rs`.
//! Factions under financial stress may secretly reduce coin metal content,
//! causing inflation that erodes the real value of gold. Detection and
//! exposure yield diplomatic payoffs.
//!
//! Without faction-level currency integrity state, this system approximates
//! inflation as a settlement-level price adjustment: settlements belonging
//! to regions with high threat and low population (stressed) see prices
//! inflated, draining treasury value.
//!
//! NEEDS STATE: `currency_integrity: Vec<CurrencyState>` on WorldState
//!   (faction_id, purity, inflation_rate, debasement_detected)
//! NEEDS STATE: `factions: Vec<FactionState>` on WorldState
//!   (at_war_with, military_strength, max_military_strength, relationship_to_guild)
//! NEEDS STATE: `spies: Vec<Spy>` on WorldState (target_faction_id)
//! NEEDS STATE: `guild: GuildState` on WorldState (reputation, gold, investment)
//! NEEDS DELTA: AdjustInflation { region_id: u32, rate: f32 }
//! NEEDS DELTA: DetectDebasement { faction_id: u32 }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// How often the debasement system ticks.
const DEBASEMENT_INTERVAL: u64 = 13;

/// Inflation multiplier: applied to prices when conditions are met.
const INFLATION_FACTOR: f32 = 0.02;

/// Minimum tick before activation.
const ACTIVATION_TICK: u64 = 800;

/// Threshold for threat_level that triggers inflationary pressure.
const THREAT_INFLATION_THRESHOLD: f32 = 20.0;

/// Compute currency debasement deltas.
///
/// Regions with high threat are assumed to be under financial stress,
/// causing inflationary price increases at nearby settlements.
pub fn compute_currency_debasement(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DEBASEMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < ACTIVATION_TICK {
        return;
    }

    // Build set of stressed regions.
    let stressed_regions: Vec<u32> = state
        .regions
        .iter()
        .filter(|r| r.threat_level > THREAT_INFLATION_THRESHOLD)
        .map(|r| r.id)
        .collect();

    if stressed_regions.is_empty() {
        return;
    }

    // Inflate prices at settlements near stressed regions.
    // (Heuristic: settlement's closest region by ID match.)
    for settlement in &state.settlements {
        // Check if any stressed region is "close" (simple: same index range).
        let is_affected = stressed_regions
            .iter()
            .any(|&rid| {
                // Proximity heuristic: region id matches settlement index modulo.
                (rid as i64 - settlement.id as i64).unsigned_abs() <= 1
            });

        if !is_affected {
            continue;
        }

        // Apply inflationary pressure: all prices tick upward.
        let mut inflated_prices = settlement.prices;
        for c in 0..NUM_COMMODITIES {
            inflated_prices[c] *= 1.0 + INFLATION_FACTOR;
        }

        out.push(WorldDelta::UpdatePrices {
            location_id: settlement.id,
            prices: inflated_prices,
        });

        // Inflation erodes treasury value.
        let erosion = settlement.treasury * INFLATION_FACTOR;
        if erosion > 0.01 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -erosion,
            });
        }
    }
}
