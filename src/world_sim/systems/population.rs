#![allow(unused)]
//! Population growth/decline system — every 3 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/population.rs`.
//! Tracks settlement population growth based on food supply, treasury
//! health, and threat levels. Population changes are expressed as
//! UpdateTreasury (tax income) and UpdateStockpile (consumption) deltas.
//!
//!
//! Population changes are emitted via `UpdateSettlementField { field: Population, .. }`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, SettlementField, WorldState};
use crate::world_sim::NUM_COMMODITIES;

/// Cadence: runs every 50 ticks.
const POP_TICK_INTERVAL: u64 = 50;

/// Base population growth rate per interval (fraction of current population).
const BASE_GROWTH_RATE: f32 = 0.005;

/// Maximum population per settlement (carrying capacity).
const MAX_POPULATION: u32 = 500;

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
        compute_population_for_settlement(state, settlement.id, &[], out);
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Population operates at settlement level (not entity level), so the
/// `entities` slice is accepted for interface consistency but unused.
pub fn compute_population_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % POP_TICK_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    let pop = settlement.population;
    if pop == 0 {
        return;
    }

    // --- Food consumption by civilian population ---
    let food_demand = pop as f32 * FOOD_PER_POP;
    let food_available = settlement.stockpile[COMMODITY_FOOD];
    let food_consumed = food_demand.min(food_available);

    if food_consumed > 0.0 {
        out.push(WorldDelta::ConsumeCommodity {
            settlement_id: settlement_id,
            commodity: COMMODITY_FOOD,
            amount: food_consumed,
        });
    }

    // --- Population growth/decline ---
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
    let regional_threat = state
        .regions
        .iter()
        .map(|r| r.threat_level)
        .fold(0.0f32, f32::max);
    if regional_threat > THREAT_DECLINE_THRESHOLD {
        growth -= THREAT_DECLINE_RATE * (regional_threat / 100.0);
    }

    // Logistic growth: slow down as population approaches carrying capacity.
    if pop >= MAX_POPULATION {
        growth = growth.min(0.0); // only decline at cap
    } else {
        let headroom = 1.0 - (pop as f32 / MAX_POPULATION as f32);
        growth *= headroom;
    }

    // --- Apply population growth/decline ---
    let pop_delta = (growth * pop as f32).round();
    if pop_delta.abs() >= 1.0 {
        // Don't shrink below MIN_POPULATION.
        let clamped = if pop_delta < 0.0 {
            let max_loss = -((pop.saturating_sub(MIN_POPULATION)) as f32);
            pop_delta.max(max_loss)
        } else {
            pop_delta.min((MAX_POPULATION - pop) as f32)
        };
        if clamped.abs() >= 1.0 {
            out.push(WorldDelta::UpdateSettlementField {
                settlement_id,
                field: SettlementField::Population,
                value: clamped,
            });
        }
    }

    // --- Tax income from population ---
    let tax_income = pop as f32 * TAX_PER_POP;
    if tax_income > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: tax_income,
        });
    }

    // --- Growth bonus: when population is growing, surplus production goes to stockpile ---
    if growth > 0.0 && food_ratio > 1.0 {
        let surplus_rate = growth * pop as f32 * 0.01;
        for c in 0..NUM_COMMODITIES {
            if surplus_rate > 0.001 {
                out.push(WorldDelta::ProduceCommodity {
                    settlement_id: settlement_id,
                    commodity: c,
                    amount: surplus_rate,
                });
            }
        }
    }

    // --- Decline penalty: starving settlements lose stockpile ---
    if growth < 0.0 {
        let desperation = (-growth) * pop as f32 * 0.005;
        for c in 0..NUM_COMMODITIES {
            if settlement.stockpile[c] > desperation {
                out.push(WorldDelta::ConsumeCommodity {
                    settlement_id: settlement_id,
                    commodity: c,
                    amount: desperation,
                });
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
        state.tick = 50;
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
                    settlement_id: 10,
                    commodity: crate::world_sim::commodity::FOOD,
                    ..
                }
            )
        });
        assert!(has_consume, "population should consume food");
    }

    #[test]
    fn population_generates_tax() {
        let mut state = WorldState::new(42);
        state.tick = 50;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        s.stockpile[0] = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_tax = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateTreasury { settlement_id: 10, delta } if *delta > 0.0
            )
        });
        assert!(has_tax, "population should generate tax income");
    }

    #[test]
    fn growing_settlement_produces_surplus() {
        let mut state = WorldState::new(42);
        state.tick = 50;
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
                    settlement_id: 10,
                    ..
                }
            )
        });
        assert!(has_produce, "growing settlement should produce surplus");
    }

    #[test]
    fn zero_pop_no_deltas() {
        let mut state = WorldState::new(42);
        state.tick = 50;
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

    #[test]
    fn growing_settlement_emits_population_delta() {
        let mut state = WorldState::new(42);
        state.tick = 50;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 200; // large enough that growth rounds to >= 1, below MAX_POPULATION
        s.stockpile[0] = 500.0; // plenty of food (ratio >> 1.5)
        s.treasury = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_pop_delta = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::UpdateSettlementField {
                    settlement_id: 10,
                    field: SettlementField::Population,
                    value,
                } if *value > 0.0
            )
        });
        assert!(has_pop_delta, "growing settlement should emit positive population delta");
    }

    #[test]
    fn starving_settlement_loses_population() {
        let mut state = WorldState::new(42);
        state.tick = 50;
        let mut s = SettlementState::new(10, "Famine".into(), (0.0, 0.0));
        s.population = 500;
        s.stockpile[0] = 0.01; // almost no food (ratio < 0.1 → starvation)
        s.treasury = -20.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_population(&state, &mut deltas);

        let has_pop_decline = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::UpdateSettlementField {
                    settlement_id: 10,
                    field: SettlementField::Population,
                    value,
                } if *value < 0.0
            )
        });
        assert!(has_pop_decline, "starving settlement should emit negative population delta");
    }
}
