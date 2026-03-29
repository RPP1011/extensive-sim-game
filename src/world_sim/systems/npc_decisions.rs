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

    // --- Barter: NPCs at the same settlement swap commodities ---
    // Find pairs where A has surplus of X and B has surplus of Y.
    // They trade at local prices without gold changing hands.
    // This enables gold-poor settlements to still have functioning economies.
    if state.tick % DECISION_INTERVAL != 0 { return; }
    barter_at_settlement(state, settlement_id, entities, out);
}

/// Carrying capacity for an NPC: level-scaled plus a base allowance.
fn carry_capacity(entity: &Entity) -> f32 {
    entity.level as f32 * 5.0 + 10.0
}

/// Total weight of goods currently carried by an NPC.
fn total_carried(npc: &NpcData) -> f32 {
    npc.carried_goods.iter().sum()
}

fn barter_at_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Collect NPC IDs with their primary production commodity.
    // Stack-allocated, max 64 pairs.
    let mut producers: [(u32, usize, f32); 64] = [(0, 0, 0.0); 64]; // (entity_id, commodity, amount_available)
    let mut count = 0usize;

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if count >= 64 { break; }

        // Find this NPC's primary production commodity.
        if let Some(&(commodity, rate)) = npc.behavior_production.first() {
            if rate > 0.0 {
                producers[count] = (entity.id, commodity, rate);
                count += 1;
            }
        }
    }

    if count < 2 { return; }

    // Find barter pairs: A produces X, B produces Y, X ≠ Y.
    // Each pair swaps a small amount at local prices.
    let mut swaps_done = 0;
    let max_swaps = 3; // cap per settlement per decision cycle

    for i in 0..count {
        if swaps_done >= max_swaps { break; }
        for j in (i + 1)..count {
            if swaps_done >= max_swaps { break; }
            let (a_id, a_commodity, a_rate) = producers[i];
            let (b_id, b_commodity, b_rate) = producers[j];

            if a_commodity == b_commodity { continue; } // same product, no point

            // Swap amount: smaller of the two rates.
            let swap_amount = a_rate.min(b_rate) * 0.5;
            if swap_amount < 0.01 { continue; }

            // Value equivalence at local prices.
            let a_value = swap_amount * settlement.prices[a_commodity];
            let b_value = swap_amount * settlement.prices[b_commodity];
            let value_ratio = a_value / b_value.max(0.01);

            // Only barter if roughly fair (within 3:1 ratio).
            if value_ratio < 0.33 || value_ratio > 3.0 { continue; }

            // Adjust amounts so values match.
            let (a_give, b_give) = if value_ratio > 1.0 {
                // A's goods worth more — A gives less.
                (swap_amount / value_ratio, swap_amount)
            } else {
                (swap_amount, swap_amount * value_ratio)
            };

            // --- Carrying capacity check ---
            // After the swap, A loses a_give of a_commodity and gains b_give of b_commodity.
            // B loses b_give of b_commodity and gains a_give of a_commodity.
            // Reject the trade if either NPC would exceed their capacity.
            let entity_a = entities.iter().find(|e| e.id == a_id);
            let entity_b = entities.iter().find(|e| e.id == b_id);
            if let (Some(ea), Some(eb)) = (entity_a, entity_b) {
                let npc_a = ea.npc.as_ref().unwrap();
                let npc_b = eb.npc.as_ref().unwrap();
                let a_after = total_carried(npc_a) - a_give + b_give;
                let b_after = total_carried(npc_b) - b_give + a_give;
                if a_after > carry_capacity(ea) || b_after > carry_capacity(eb) {
                    continue;
                }
            }

            // A gives commodity_a to B.
            out.push(WorldDelta::TransferGoods {
                from_id: a_id,
                to_id: b_id,
                commodity: a_commodity,
                amount: a_give,
            });
            // B gives commodity_b to A.
            out.push(WorldDelta::TransferGoods {
                from_id: b_id,
                to_id: a_id,
                commodity: b_commodity,
                amount: b_give,
            });

            // Both earn trade behavior tags.
            let mut action = ActionTags::empty();
            action.add(tags::TRADE, 0.5);
            action.add(tags::NEGOTIATION, 0.3);
            out.push(WorldDelta::AddBehaviorTags {
                entity_id: a_id,
                tags: action.tags,
                count: action.count,
            });
            out.push(WorldDelta::AddBehaviorTags {
                entity_id: b_id,
                tags: action.tags,
                count: action.count,
            });

            swaps_done += 1;
        }
    }
}
