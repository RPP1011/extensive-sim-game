#![allow(unused)]
//! Population growth/decline system — every 3 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/population.rs`.
//! Tracks settlement population growth based on food supply, treasury
//! health, and threat levels. Population changes are expressed as
//! UpdateTreasury (tax income) and UpdateStockpile (consumption) deltas.
//!
//! NEEDS STATE: `civilian_morale: f32` on SettlementState
//! NEEDS STATE: `growth_rate: f32` on SettlementState
//! NEEDS STATE: `tax_rate: f32` on SettlementState
//! NEEDS DELTA: UpdatePopulation { settlement_id, delta: i32 }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// Cadence: runs every 3 ticks.
const POP_TICK_INTERVAL: u64 = 3;

/// Base population growth rate per tick (fraction of current population).
const BASE_GROWTH_RATE: f32 = 0.001;

/// Food commodity index consumed by population.
const COMMODITY_FOOD: usize = 0;

/// Food consumed per population unit per tick.
const FOOD_PER_POP: f32 = 0.001;

/// Tax income per population unit per tick.
const TAX_PER_POP: f32 = 0.001;

/// Threat threshold above which population declines.
const THREAT_DECLINE_THRESHOLD: f32 = 50.0;

/// Population decline rate when threat is high.
const THREAT_DECLINE_RATE: f32 = 0.005;

/// Minimum population floor (settlements don't fully depopulate).
const MIN_POPULATION: u32 = 10;

pub fn compute_population(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % POP_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let pop = settlement.population;
        if pop == 0 {
            continue;
        }

        // --- Food consumption by civilian population ---
        let food_demand = pop as f32 * FOOD_PER_POP;
        let food_available = settlement.stockpile[COMMODITY_FOOD];
        let food_consumed = food_demand.min(food_available);

        if food_consumed > 0.0 {
            out.push(WorldDelta::ConsumeCommodity {
                location_id: settlement.id,
                commodity: COMMODITY_FOOD,
                amount: food_consumed,
            });
        }

        // --- Population growth/decline ---
        // Growth factors:
        //   + food surplus (stockpile > demand)
        //   + positive treasury
        //   - food shortage
        //   - high regional threat
        let food_ratio = if food_demand > 0.0 {
            food_available / food_demand
        } else {
            1.0
        };

        let mut growth = BASE_GROWTH_RATE;

        // Food surplus boosts growth.
        if food_ratio > 1.5 {
            growth *= 1.5;
        }
        // Food shortage suppresses growth.
        if food_ratio < 0.5 {
            growth *= 0.3;
        }
        // Starvation causes decline.
        if food_ratio < 0.1 {
            growth = -0.01;
        }

        // Positive treasury boosts growth.
        if settlement.treasury > 0.0 {
            growth *= 1.2;
        }
        // Negative treasury suppresses growth.
        if settlement.treasury < -10.0 {
            growth *= 0.5;
        }

        // High regional threat causes population decline.
        // Find the nearest region's threat level.
        let regional_threat = state
            .regions
            .iter()
            .map(|r| r.threat_level)
            .fold(0.0f32, f32::max);
        if regional_threat > THREAT_DECLINE_THRESHOLD {
            growth -= THREAT_DECLINE_RATE * (regional_threat / 100.0);
        }

        // Compute population change.
        // Note: we can't directly modify population through existing deltas,
        // so we express the economic consequences: tax income and food consumption
        // are the observable effects. The actual population update would need
        // a new delta type.
        //
        // For now, model population growth/decline indirectly:
        //   - Growing population → increased tax revenue → UpdateTreasury
        //   - Declining population → reduced tax revenue (handled by lower pop next tick)

        // --- Tax income from population ---
        let tax_income = pop as f32 * TAX_PER_POP;
        if tax_income > 0.0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: tax_income,
            });
        }

        // --- Growth bonus: when population is growing, surplus production goes to stockpile ---
        if growth > 0.0 && food_ratio > 1.0 {
            // Growing settlement produces surplus across all commodities.
            let surplus_rate = growth * pop as f32 * 0.01;
            for c in 0..NUM_COMMODITIES {
                if surplus_rate > 0.001 {
                    out.push(WorldDelta::ProduceCommodity {
                        location_id: settlement.id,
                        commodity: c,
                        amount: surplus_rate,
                    });
                }
            }
        }

        // --- Decline penalty: starving settlements lose stockpile ---
        if growth < 0.0 {
            // Declining population consumes stockpile faster (desperation).
            let desperation = (-growth) * pop as f32 * 0.005;
            for c in 0..NUM_COMMODITIES {
                if settlement.stockpile[c] > desperation {
                    out.push(WorldDelta::ConsumeCommodity {
                        location_id: settlement.id,
                        commodity: c,
                        amount: desperation,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn population_consumes_food() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 200;
        s.stockpile[0] = 50.0; // food available
        s.treasury = 10.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_consume = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::ConsumeCommodity {
                    location_id: 10,
                    commodity: 0,
                    ..
                }
            )
        });
        assert!(has_consume, "population should consume food");
    }

    #[test]
    fn population_generates_tax() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        s.stockpile[0] = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_tax = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateTreasury { location_id: 10, delta } if *delta > 0.0
            )
        });
        assert!(has_tax, "population should generate tax income");
    }

    #[test]
    fn growing_settlement_produces_surplus() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 200;
        s.stockpile[0] = 100.0; // plenty of food (ratio > 1.0)
        s.treasury = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_produce = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::ProduceCommodity {
                    location_id: 10,
                    ..
                }
            )
        });
        assert!(has_produce, "growing settlement should produce surplus");
    }

    #[test]
    fn zero_pop_no_deltas() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let mut s = SettlementState::new(10, "Ghost".into(), (0.0, 0.0));
        s.population = 0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);
        assert!(deltas.is_empty());
    }

    #[test]
    fn skips_off_cadence() {
        let mut state = WorldState::new(42);
        state.tick = 1;
        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);
        assert!(deltas.is_empty());
    }
}
