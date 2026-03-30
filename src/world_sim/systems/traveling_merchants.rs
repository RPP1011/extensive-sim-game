#![allow(unused)]
//! Seasonal traveling merchant system — fires every 17 ticks.
//!
//! NPC merchant entities arrive at settlements, share price reports, and
//! transfer goods. In the delta architecture, merchants are modeled as
//! NPC entities with Sell intent who bring goods from distant settlements.
//!
//! Original: `crates/headless_campaign/src/systems/traveling_merchants.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, EntityKind, PriceReport, WorldState};
use crate::world_sim::state::{entity_hash_f32};
use crate::world_sim::NUM_COMMODITIES;

/// How often to process merchant activity (in ticks).
const MERCHANT_INTERVAL: u64 = 17;

/// Distance at which a merchant interacts with a settlement.
const MERCHANT_RANGE_SQ: f32 = 100.0; // 10 units


pub fn compute_traveling_merchants(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MERCHANT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // In the delta architecture, traveling merchants are NPC entities with
    // Sell intent. When near a settlement, they transfer goods and share
    // price reports from distant settlements.

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Only entities with Sell intent act as merchants.
        let sell_commodity = match &npc.economic_intent {
            EconomicIntent::Sell { commodity } => *commodity,
            _ => continue,
        };

        // Find the nearest settlement.
        let nearest = state.settlements.iter().min_by(|a, b| {
            let da = (a.pos.0 - entity.pos.0).powi(2) + (a.pos.1 - entity.pos.1).powi(2);
            let db = (b.pos.0 - entity.pos.0).powi(2) + (b.pos.1 - entity.pos.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        let settlement = match nearest {
            Some(s) => s,
            None => continue,
        };

        let dx = entity.pos.0 - settlement.pos.0;
        let dy = entity.pos.1 - settlement.pos.1;
        if dx * dx + dy * dy > MERCHANT_RANGE_SQ {
            // Move toward the settlement.
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 0.5 {
                let speed = entity.move_speed * 0.1;
                out.push(WorldDelta::Move {
                    entity_id: entity.id,
                    force: (-dx / dist * speed, -dy / dist * speed),
                });
            }
            continue;
        }

        // At the settlement: sell goods.
        let sell_stock = entity.inv_commodity(sell_commodity);
        if sell_commodity < NUM_COMMODITIES && sell_stock > 0.001 {
            let amount = sell_stock;
            out.push(WorldDelta::TransferCommodity {
                from_entity: entity.id,
                to_entity: settlement.id,
                commodity: sell_commodity,
                amount,
            });

            // Receive gold based on settlement prices.
            let gold = settlement.prices[sell_commodity] * amount;
            out.push(WorldDelta::TransferGold {
                from_entity: settlement.id,
                to_entity: entity.id,
                amount: gold,
            });
        }

        // Share price knowledge from the merchant's home settlement.
        if let Some(home_id) = npc.home_settlement_id {
            if let Some(home) = state.settlement(home_id) {
                out.push(WorldDelta::SharePriceReport {
                    from_id: entity.id,
                    to_id: settlement.id,
                    report: PriceReport {
                        settlement_id: home_id,
                        prices: home.prices,
                        tick_observed: state.tick,
                    },
                });
            }
        }
    }
}
