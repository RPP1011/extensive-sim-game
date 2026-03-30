//! Infrastructure system — every 7 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/infrastructure.rs`.
//! Infrastructure (roads, bridges, waypoints, trade posts, watchtowers)
//! provides passive bonuses to settlements. Maintained infrastructure grows;
//! unmaintained infrastructure degrades. Effects on production are modeled
//! as UpdateStockpile deltas; maintenance costs as UpdateTreasury.
//!
//! Since WorldState does not have an infrastructure list, this system
//! approximates infrastructure effects through settlement-level bonuses
//! based on treasury and population (proxy for infrastructure investment).
//!
//!   where InfrastructureEntry { id, infra_type, settlement_a_id, settlement_b_id, level, maintenance_cost }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::NUM_COMMODITIES;

/// Cadence: runs every 7 ticks.
const INFRA_TICK_INTERVAL: u64 = 7;

/// Maintenance cost per settlement per tick, proportional to population.
const MAINTENANCE_COST_PER_POP: f32 = 0.002;

/// Production bonus per commodity when settlement treasury is positive
/// (proxy for maintained infrastructure).
const MAINTAINED_PRODUCTION_BONUS: f32 = 0.02;

/// Production penalty per commodity when settlement treasury is negative
/// (proxy for crumbling infrastructure).
const UNMAINTAINED_PRODUCTION_PENALTY: f32 = 0.01;

/// Trade post bonus: extra stockpile regeneration when two settlements
/// are both well-funded (treasury > 10).
const TRADE_POST_BONUS: f32 = 0.03;

pub fn compute_infrastructure(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % INFRA_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_infrastructure_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // --- Inter-settlement trade bonuses ---
    let funded_settlements: Vec<u32> = state
        .settlements
        .iter()
        .filter(|s| s.treasury > 10.0)
        .map(|s| s.id)
        .collect();

    if funded_settlements.len() >= 2 {
        for &sid in &funded_settlements {
            for c in 0..NUM_COMMODITIES {
                out.push(WorldDelta::UpdateStockpile {
                    settlement_id: sid,
                    commodity: c,
                    delta: TRADE_POST_BONUS,
                });
            }
        }
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_infrastructure_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % INFRA_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- Maintenance cost (only drain if treasury above floor) ---
    let maintenance = settlement.population as f32 * MAINTENANCE_COST_PER_POP;
    if maintenance > 0.0 && settlement.treasury > -100.0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: -maintenance,
        });
    }

    // --- Infrastructure effects on production ---
    if settlement.treasury > 0.0 {
        let bonus_scale = (settlement.treasury / 100.0).sqrt().min(1.0);
        for c in 0..NUM_COMMODITIES {
            let bonus = MAINTAINED_PRODUCTION_BONUS
                * bonus_scale
                * (settlement.population as f32 / 50.0).max(0.1);
            out.push(WorldDelta::UpdateStockpile {
                settlement_id: settlement_id,
                commodity: c,
                delta: bonus,
            });
        }
    } else {
        for c in 0..NUM_COMMODITIES {
            let penalty = UNMAINTAINED_PRODUCTION_PENALTY
                * (settlement.population as f32 / 50.0).max(0.1);
            if settlement.stockpile[c] > penalty {
                out.push(WorldDelta::UpdateStockpile {
                    settlement_id: settlement_id,
                    commodity: c,
                    delta: -penalty,
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
    fn maintenance_drains_treasury() {
        let mut state = WorldState::new(42);
        state.tick = 7;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        s.treasury = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_infrastructure(&state, &mut deltas);

        let has_drain = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateTreasury { settlement_id: 10, delta } if *delta < 0.0
            )
        });
        assert!(has_drain, "maintenance should drain treasury");
    }

    #[test]
    fn positive_treasury_boosts_production() {
        let mut state = WorldState::new(42);
        state.tick = 7;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        s.treasury = 50.0;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_infrastructure(&state, &mut deltas);

        let has_stockpile_boost = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateStockpile { settlement_id: 10, delta, .. } if *delta > 0.0
            )
        });
        assert!(
            has_stockpile_boost,
            "positive treasury should boost stockpile"
        );
    }

    #[test]
    fn negative_treasury_penalizes_production() {
        let mut state = WorldState::new(42);
        state.tick = 7;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        s.treasury = -10.0;
        s.stockpile = [20.0; NUM_COMMODITIES];
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_infrastructure(&state, &mut deltas);

        let has_penalty = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateStockpile { settlement_id: 10, delta, .. } if *delta < 0.0
            )
        });
        assert!(has_penalty, "negative treasury should penalize stockpile");
    }

    #[test]
    fn skips_tick_zero() {
        let mut state = WorldState::new(42);
        state.tick = 0;
        let mut deltas = Vec::new();
        compute_infrastructure(&state, &mut deltas);
        assert!(deltas.is_empty());
    }

    #[test]
    fn skips_off_cadence() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let mut deltas = Vec::new();
        compute_infrastructure(&state, &mut deltas);
        assert!(deltas.is_empty());
    }
}
