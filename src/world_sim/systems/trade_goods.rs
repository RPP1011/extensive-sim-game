//! Regional trade goods system — every 7 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/trade_goods.rs`.
//! Settlements produce commodities based on their production profile,
//! prices adjust via supply/demand, and automated caravans move goods
//! between settlements when profitable. All mutations expressed as
//! TransferCommodity, UpdateStockpile, and UpdatePrices deltas.
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, WorldState, tags};
use crate::world_sim::NUM_COMMODITIES;

/// How often the trade goods system ticks.
const TRADE_TICK_INTERVAL: u64 = 7;

/// Maximum price multiplier (price = base * demand/supply, capped here).
const MAX_PRICE_MULTIPLIER: f32 = 5.0;

/// Minimum supply floor to prevent division by zero in price calc.
const MIN_SUPPLY: f32 = 0.1;

/// Base supply regeneration per commodity per tick.
const SUPPLY_REGEN_RATE: f32 = 0.05;

/// Demand drift rate per tick (demand gravitates toward equilibrium).
const DEMAND_DRIFT_RATE: f32 = 0.03;

/// Caravan profit threshold — only move goods if sell/buy price ratio exceeds this.
const CARAVAN_PROFIT_THRESHOLD: f32 = 1.5;

/// Amount moved per caravan run.
const CARAVAN_AMOUNT: f32 = 2.0;

/// Transit loss fraction (20% lost in transit).
const TRANSIT_LOSS: f32 = 0.2;

/// Base price for commodity pricing when no settlement-level override exists.
const BASE_PRICE: f32 = 1.0;

/// Compute the current price of a commodity at a settlement.
fn commodity_price(supply: f32, demand: f32) -> f32 {
    let ratio = demand / supply.max(MIN_SUPPLY);
    (BASE_PRICE * ratio).min(BASE_PRICE * MAX_PRICE_MULTIPLIER)
}

pub fn compute_trade_goods(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TRADE_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_trade_goods_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // --- Automated caravans: move goods between settlements for profit ---
    // (Cross-settlement logic remains in the top-level function.)
    if state.settlements.len() < 2 {
        return;
    }

    for c in 0..NUM_COMMODITIES {
        let source = state
            .settlements
            .iter()
            .filter(|s| s.stockpile[c] > CARAVAN_AMOUNT)
            .min_by(|a, b| {
                a.prices[c]
                    .partial_cmp(&b.prices[c])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let dest = state.settlements.iter().max_by(|a, b| {
            a.prices[c]
                .partial_cmp(&b.prices[c])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let (Some(src), Some(dst)) = (source, dest) {
            if src.id == dst.id {
                continue;
            }
            let price_ratio = dst.prices[c] / src.prices[c].max(0.01);
            if price_ratio > CARAVAN_PROFIT_THRESHOLD {
                out.push(WorldDelta::UpdateStockpile {
                    settlement_id: src.id,
                    commodity: c,
                    delta: -CARAVAN_AMOUNT,
                });
                let delivered = CARAVAN_AMOUNT * (1.0 - TRANSIT_LOSS);
                out.push(WorldDelta::UpdateStockpile {
                    settlement_id: dst.id,
                    commodity: c,
                    delta: delivered,
                });
                let profit = (dst.prices[c] - src.prices[c]) * CARAVAN_AMOUNT * 0.1;
                if profit > 0.0 {
                    out.push(WorldDelta::UpdateTreasury {
                        settlement_id: src.id,
                        delta: profit,
                    });
                }

                // Behavior tags: NPCs at source settlement earn trade/negotiation.
                let range = state.group_index.settlement_entities(src.id);
                for entity in &state.entities[range] {
                    if entity.alive && entity.kind == crate::world_sim::state::EntityKind::Npc {
                        let mut action = ActionTags::empty();
                        action.add(tags::TRADE, 1.0);
                        action.add(tags::NEGOTIATION, 0.5);
                        let action = crate::world_sim::action_context::with_context(&action, entity, state);
                        out.push(WorldDelta::AddBehaviorTags { entity_id: entity.id, tags: action.tags, count: action.count });
                        break; // Only tag one NPC per caravan run.
                    }
                }
            }
        }
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Handles supply regeneration and price updates for a single settlement.
pub fn compute_trade_goods_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % TRADE_TICK_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- Supply regeneration ---
    for c in 0..NUM_COMMODITIES {
        let regen = SUPPLY_REGEN_RATE * (settlement.population as f32 / 100.0).max(0.1);
        out.push(WorldDelta::UpdateStockpile {
            settlement_id: settlement_id,
            commodity: c,
            delta: regen,
        });
    }

    // --- Price updates based on supply/demand ---
    let mut new_prices = [0.0f32; NUM_COMMODITIES];
    for c in 0..NUM_COMMODITIES {
        let supply = settlement.stockpile[c].max(MIN_SUPPLY);
        let demand = (settlement.population as f32 * 0.01).max(0.5);
        new_prices[c] = commodity_price(supply, demand);
    }
    out.push(WorldDelta::UpdatePrices {
        settlement_id: settlement_id,
        prices: new_prices,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn stockpile_regenerates() {
        let mut state = WorldState::new(42);
        state.tick = 7;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 100;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_trade_goods(&state, &mut deltas);

        let has_regen = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateStockpile { settlement_id: 10, delta, .. } if *delta > 0.0
            )
        });
        assert!(has_regen, "stockpile should regenerate");
    }

    #[test]
    fn prices_update() {
        let mut state = WorldState::new(42);
        state.tick = 7;
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.population = 50;
        state.settlements.push(s);

        let mut deltas = Vec::new();
        compute_trade_goods(&state, &mut deltas);

        let has_prices = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::UpdatePrices {
                    settlement_id: 10,
                    ..
                }
            )
        });
        assert!(has_prices, "prices should be updated");
    }

    #[test]
    fn caravan_moves_goods() {
        let mut state = WorldState::new(42);
        state.tick = 7;

        // Source: lots of commodity 1, low price.
        let mut src = SettlementState::new(10, "Source".into(), (0.0, 0.0));
        src.population = 50;
        src.stockpile[1] = 100.0;
        src.prices[1] = 0.5;
        state.settlements.push(src);

        // Dest: no commodity 1, high price.
        let mut dst = SettlementState::new(20, "Dest".into(), (50.0, 0.0));
        dst.population = 50;
        dst.stockpile[1] = 0.0;
        dst.prices[1] = 5.0;
        state.settlements.push(dst);

        let mut deltas = Vec::new();
        compute_trade_goods(&state, &mut deltas);

        // Should have negative stockpile delta on source and positive on dest for commodity 1.
        let src_drain = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateStockpile { settlement_id: 10, commodity: crate::world_sim::commodity::IRON, delta }
                    if *delta < 0.0
            )
        });
        let dst_gain = deltas.iter().any(|d| {
            matches!(d,
                WorldDelta::UpdateStockpile { settlement_id: 20, commodity: crate::world_sim::commodity::IRON, delta }
                    if *delta > 0.0
            )
        });
        assert!(src_drain, "source should lose goods");
        assert!(dst_gain, "destination should gain goods");
    }

    #[test]
    fn skips_off_cadence() {
        let mut state = WorldState::new(42);
        state.tick = 1;
        let mut deltas = Vec::new();
        compute_trade_goods(&state, &mut deltas);
        assert!(deltas.is_empty());
    }
}
