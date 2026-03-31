//! NPC needs-driven decisions — the core AI loop.
//!
//! Every DECISION_INTERVAL ticks, each idle/working NPC evaluates possible
//! actions scored by how well they satisfy the NPC's most urgent need, modulated
//! by emotions, beliefs, and personality.
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const DECISION_INTERVAL: u64 = 20;



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

    if state.settlement(settlement_id).is_none() { return; }

    for entity in entities {
        if !entity.alive { continue; }
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
                    let mut total_revenue = 0.0f32;
                    for c in 0..crate::world_sim::NUM_COMMODITIES {
                        let amount = entity.inv_commodity(c);
                        if amount > 0.1 {
                            let revenue = amount * dest.prices[c];
                            total_revenue += revenue;

                            // Add goods to destination stockpile.
                            out.push(WorldDelta::UpdateStockpile {
                                settlement_id: dest_id,
                                commodity: c,
                                delta: amount,
                            });

                            // NPC earns gold from the sale.
                            out.push(WorldDelta::TransferGold {
                                from_entity: dest_id, // destination pays
                                to_entity: entity.id,
                                amount: revenue,
                            });

                            // Clear carried goods (consume from self).
                            out.push(WorldDelta::TransferCommodity {
                                from_entity: entity.id,
                                to_entity: entity.id,
                                commodity: c,
                                amount,
                            });
                        }
                    }

                    // Record profitable trade for route tracking.
                    if total_revenue > 0.0 {
                        let home_sid = npc.home_settlement_id.unwrap_or(settlement_id);
                        out.push(WorldDelta::RecordTradeCompletion {
                            entity_id: entity.id,
                            home_settlement_id: home_sid,
                            dest_settlement_id: dest_id,
                            profit: total_revenue,
                        });
                    }

                    // Chronicle: notable traders completing runs.
                    if entity.level >= 30 {
                        let trader_name = crate::world_sim::naming::entity_display_name(entity);
                        let dest_name = state.settlement(dest_id)
                            .map(|s| s.name.as_str()).unwrap_or("unknown");
                        let home_name = npc.home_settlement_id
                            .and_then(|sid| state.settlement(sid))
                            .map(|s| s.name.as_str()).unwrap_or("unknown");
                        out.push(WorldDelta::RecordChronicle {
                            entry: ChronicleEntry {
                                tick: state.tick,
                                category: ChronicleCategory::Economy,
                                text: format!("{} completed a trade run from {} to {}",
                                    trader_name, home_name, dest_name),
                                entity_ids: vec![entity.id],
                            },
                        });
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
                    // Return to producing.
                    out.push(WorldDelta::SetIntent {
                        entity_id: entity.id,
                        intent: EconomicIntent::Produce,
                    });
                }
            }
            continue; // don't re-evaluate while trading
        }

        // NPC action scoring now handled by action_eval.rs.
        // Only trade arrival and barter remain here.
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
fn total_carried(entity: &Entity) -> f32 {
    entity.inv_total_commodities()
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
        if !entity.alive { continue; }
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
            let entity_a = state.entity(a_id);
            let entity_b = state.entity(b_id);
            if let (Some(ea), Some(eb)) = (entity_a, entity_b) {
                let a_after = total_carried(ea) - a_give + b_give;
                let b_after = total_carried(eb) - b_give + a_give;
                if a_after > carry_capacity(ea) || b_after > carry_capacity(eb) {
                    continue;
                }
            }

            // A gives commodity_a to B.
            out.push(WorldDelta::TransferCommodity {
                from_entity: a_id,
                to_entity: b_id,
                commodity: a_commodity,
                amount: a_give,
            });
            // B gives commodity_b to A.
            out.push(WorldDelta::TransferCommodity {
                from_entity: b_id,
                to_entity: a_id,
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
