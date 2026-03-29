#![allow(unused)]
//! NPC economic agent decisions — the core AI loop.
//!
//! Every DECISION_INTERVAL ticks, each idle/working NPC evaluates:
//! 1. Stay and work (produce at current settlement)
//! 2. Trade (buy cheap, carry to expensive settlement, sell)
//! 3. Relocate (move to a settlement with better prospects)
//!
//! Decisions use utility scoring based on:
//! - Expected income from production at current vs other settlements
//! - Price differentials from price knowledge (information economy)
//! - Safety (settlement threat level)
//! - Risk aversion (stress, fears, resolve)
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;
use crate::world_sim::commodity;

const DECISION_INTERVAL: u64 = 100;

/// Minimum profit margin to initiate a trade run.
const MIN_TRADE_MARGIN: f32 = 2.0;

/// Price staleness discount: reports older than this lose confidence.
const STALE_TICKS: f32 = 200.0;

/// Hysteresis: relocation must be 30% better than staying.
const RELOCATION_HYSTERESIS: f32 = 1.3;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_npc_decisions(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DECISION_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_npc_decisions_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_npc_decisions_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % DECISION_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Handle NPCs already on trade runs — check for arrival + sell.
        if let EconomicIntent::Trade { destination_settlement_id } = &npc.economic_intent {
            let dest_id = *destination_settlement_id;
            if let Some(dest) = state.settlement(dest_id) {
                let dx = dest.pos.0 - entity.pos.0;
                let dy = dest.pos.1 - entity.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 3.0 {
                    // Arrived! Sell carried goods at destination prices.
                    for c in 0..crate::world_sim::NUM_COMMODITIES {
                        let amount = npc.carried_goods[c];
                        if amount > 0.1 {
                            let revenue = amount * dest.prices[c];

                            // Add goods to destination stockpile.
                            out.push(WorldDelta::UpdateStockpile {
                                location_id: dest_id,
                                commodity: c,
                                delta: amount,
                            });

                            // NPC earns gold from the sale.
                            out.push(WorldDelta::TransferGold {
                                from_id: dest_id, // destination pays
                                to_id: entity.id,
                                amount: revenue,
                            });

                            // Clear carried goods (consume from self).
                            out.push(WorldDelta::TransferGoods {
                                from_id: entity.id,
                                to_id: entity.id,
                                commodity: c,
                                amount,
                            });
                        }
                    }

                    // Trade behavior tags + XP.
                    let mut action = ActionTags::empty();
                    action.add(tags::TRADE, 2.0);
                    action.add(tags::NEGOTIATION, 1.0);
                    let action = crate::world_sim::action_context::with_context(&action, entity, state);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                    out.push(WorldDelta::AddXp { entity_id: entity.id, amount: 5 });

                    // Return to producing.
                    out.push(WorldDelta::SetIntent {
                        entity_id: entity.id,
                        intent: EconomicIntent::Produce,
                    });
                }
            }
            continue; // don't re-evaluate while trading
        }

        // Only re-evaluate idle or producing NPCs.
        match &npc.economic_intent {
            EconomicIntent::Idle | EconomicIntent::Produce => {}
            _ => continue,
        }

        // --- Utility: Stay and work ---
        let production_income = npc.behavior_production.iter()
            .map(|&(c, rate)| rate * settlement.prices[c])
            .sum::<f32>();
        let safety = 1.0 / (1.0 + settlement.threat_level * 0.05);
        let stay_utility = production_income + 2.0 * safety;

        // --- Utility: Trade run (if NPC has price knowledge) ---
        let mut best_trade_utility = f32::NEG_INFINITY;
        let mut best_trade_dest: Option<u32> = None;
        let mut best_trade_commodity: usize = 0;

        if npc.gold > 5.0 && !npc.price_knowledge.is_empty() {
            for report in &npc.price_knowledge {
                if report.settlement_id == settlement_id { continue; }
                let staleness = (state.tick.saturating_sub(report.tick_observed)) as f32 / STALE_TICKS;
                if staleness > 3.0 { continue; } // too old
                let confidence = (1.0 - staleness * 0.3).max(0.1);

                // Find best commodity to trade.
                for c in 0..crate::world_sim::NUM_COMMODITIES {
                    let local_price = settlement.prices[c];
                    let remote_price = report.prices[c];
                    let margin = remote_price - local_price;
                    if margin < MIN_TRADE_MARGIN { continue; }

                    // How much can we carry? Simplified: level * 5 units.
                    let carry = (entity.level as f32 * 5.0).max(5.0);
                    let afford = if local_price > 0.01 { npc.gold / local_price } else { carry };
                    let units = carry.min(afford).min(settlement.stockpile[c]);
                    if units < 1.0 { continue; }

                    let profit = margin * units * confidence;
                    let trade_utility = profit * 0.5 + safety; // trade value minus risk

                    if trade_utility > best_trade_utility {
                        best_trade_utility = trade_utility;
                        best_trade_dest = Some(report.settlement_id);
                        best_trade_commodity = c;
                    }
                }
            }
        }

        // --- Utility: Relocate to a better settlement ---
        let mut best_reloc_utility = f32::NEG_INFINITY;
        let mut best_reloc_dest: Option<u32> = None;

        // Risk aversion from NPC state.
        let risk_aversion = (1.0 + npc.stress * 0.005 - npc.resolve * 0.003).clamp(0.3, 3.0);

        for other in &state.settlements {
            if other.id == settlement_id { continue; }
            // Estimate income at other settlement.
            let other_income = npc.behavior_production.iter()
                .map(|&(c, rate)| rate * other.prices[c])
                .sum::<f32>();
            let other_safety = 1.0 / (1.0 + other.threat_level * 0.05);
            let travel_risk = settlement.threat_level * 0.02 * risk_aversion;
            let reloc_utility = other_income + 2.0 * other_safety - travel_risk;

            if reloc_utility > best_reloc_utility {
                best_reloc_utility = reloc_utility;
                best_reloc_dest = Some(other.id);
            }
        }

        // --- Pick best option ---
        // Default: keep working.
        let mut chosen_intent = EconomicIntent::Produce;

        // Trade if significantly better than staying.
        if best_trade_utility > stay_utility * 1.2 {
            if let Some(dest_id) = best_trade_dest {
                // Buy goods at local settlement, set intent to trade.
                let local_price = settlement.prices[best_trade_commodity];
                let carry = (entity.level as f32 * 5.0).max(5.0);
                let afford = if local_price > 0.01 { npc.gold / local_price } else { carry };
                let units = carry.min(afford).min(settlement.stockpile[best_trade_commodity]);

                if units > 0.0 {
                    // Buy from settlement.
                    out.push(WorldDelta::ConsumeCommodity {
                        location_id: settlement_id,
                        commodity: best_trade_commodity,
                        amount: units,
                    });
                    // Pay gold.
                    let cost = units * local_price;
                    out.push(WorldDelta::TransferGold {
                        from_id: entity.id,
                        to_id: settlement_id, // gold goes to settlement treasury
                        amount: cost,
                    });
                    // Set intent to travel to destination.
                    if let Some(dest) = state.settlement(dest_id) {
                        chosen_intent = EconomicIntent::Trade {
                            destination_settlement_id: dest_id,
                        };
                        // Behavior: trade action.
                        let mut action = ActionTags::empty();
                        action.add(crate::world_sim::state::tags::TRADE, 1.0);
                        action.add(crate::world_sim::state::tags::NEGOTIATION, 0.5);
                        let action = crate::world_sim::action_context::with_context(&action, entity, state);
                        out.push(WorldDelta::AddBehaviorTags {
                            entity_id: entity.id,
                            tags: action.tags,
                            count: action.count,
                        });
                    }
                }
            }
        }

        // Relocate if significantly better than staying (with hysteresis).
        if best_reloc_utility > stay_utility * RELOCATION_HYSTERESIS
            && best_reloc_utility > best_trade_utility
        {
            if let Some(dest_id) = best_reloc_dest {
                if let Some(dest) = state.settlement(dest_id) {
                    chosen_intent = EconomicIntent::Travel {
                        destination: dest.pos,
                    };
                }
            }
        }

        // Emit intent change + movement toward destination.
        // Skip if intent didn't change from current.
        let intent_changed = std::mem::discriminant(&chosen_intent) != std::mem::discriminant(&npc.economic_intent);
        if intent_changed {
            out.push(WorldDelta::SetIntent {
                entity_id: entity.id,
                intent: chosen_intent.clone(),
            });
        }

        // Movement toward destination (regardless of intent change — keep moving if already traveling).
        let dest_pos = match &chosen_intent {
            EconomicIntent::Trade { destination_settlement_id } => {
                state.settlement(*destination_settlement_id).map(|s| s.pos)
            }
            EconomicIntent::Travel { destination } => Some(*destination),
            _ => None,
        };

        if let Some(dest) = dest_pos {
            let dx = dest.0 - entity.pos.0;
            let dy = dest.1 - entity.pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 1.0 {
                let speed = entity.move_speed * crate::world_sim::DT_SEC;
                out.push(WorldDelta::Move {
                    entity_id: entity.id,
                    force: (dx / dist * speed, dy / dist * speed),
                });
            }
        }
    }
}
