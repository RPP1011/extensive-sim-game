#![allow(unused)]
//! NPC needs-driven decisions — the core AI loop.
//!
//! Every DECISION_INTERVAL ticks, each idle/working NPC evaluates possible
//! actions scored by how well they satisfy the NPC's most urgent need, modulated
//! by emotions, beliefs, and personality.
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

        // --- Commuting: NPCs with home and work buildings follow schedules ---
        // Skip commuting if the NPC is actively in the work state machine (non-Idle).
        // The work system handles movement for traveling/working/carrying states.
        if let (Some(home_bid), Some(work_bid)) = (npc.home_building_id, npc.work_building_id) {
            if !matches!(npc.work_state, WorkState::Idle) {
                continue; // work system handles movement
            }

            // Look up building positions from real entity IDs.
            let home_pos = state.entity(home_bid).map(|e| e.pos);
            let work_pos = state.entity(work_bid).map(|e| e.pos);

            if let (Some(home_pos), Some(work_pos)) = (home_pos, work_pos) {
                // Commute schedule: first half of cycle at work, second half at home.
                let target = if state.tick % 200 < 100 {
                    work_pos
                } else {
                    home_pos
                };

                let dx = target.0 - entity.pos.0;
                let dy = target.1 - entity.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 1.0 {
                    let speed = entity.move_speed * crate::world_sim::DT_SEC;
                    out.push(WorldDelta::Move {
                        entity_id: entity.id,
                        force: (dx / dist * speed, dy / dist * speed),
                    });
                }
                continue; // structured schedule, skip normal decision tree
            }
        }

        // Only re-evaluate idle or producing NPCs.
        match &npc.economic_intent {
            EconomicIntent::Idle => {
                // Idle (combat-ready) NPCs stand down when threat is low.
                if settlement.threat_level < 0.4 {
                    out.push(WorldDelta::SetIntent {
                        entity_id: entity.id,
                        intent: EconomicIntent::Produce,
                    });
                    continue;
                }
            }
            EconomicIntent::Produce => {}
            _ => continue,
        }

        // =================================================================
        // Needs-driven decision scoring
        // =================================================================

        // 1. Compute need urgencies.
        let (urgent_need, _urgency) = npc.needs.most_urgent();

        // 2. Emotion modifiers (transient bias from current emotional state).
        let emo = &npc.emotions;
        let anger_mod   = if emo.anger > 0.3 { emo.anger } else { 0.0 };
        let fear_mod    = if emo.fear > 0.3 { emo.fear } else { 0.0 };
        let grief_mod   = if emo.grief > 0.3 { emo.grief } else { 0.0 };
        let anxiety_mod = if emo.anxiety > 0.5 { 0.1 } else { 0.0 };
        let pride_mod   = if emo.pride > 0.3 { emo.pride } else { 0.0 };

        // 3. Personality shortcuts.
        let pers = &npc.personality;

        // Helper: does this NPC produce food?
        let produces_food = npc.behavior_production.iter().any(|&(c, _)| c == commodity::FOOD);

        // ----- Candidate action: Work/Produce -----
        let work_base = match urgent_need {
            "purpose" => 0.7,
            "hunger" if produces_food => 0.5,
            "esteem" => 0.3,
            _ => 0.15,
        };
        let work_emotion = pride_mod * 0.2 - anxiety_mod;
        let work_personality = if pers.ambition > 0.6 { 0.2 } else { 0.0 };
        let work_utility = work_base * (1.0 + work_emotion * 0.3 + work_personality * 0.2);

        // ----- Candidate action: Trade run -----
        let mut best_trade_utility = f32::NEG_INFINITY;
        let mut best_trade_dest: Option<u32> = None;
        let mut best_trade_commodity: usize = 0;

        if npc.gold > 5.0 && !npc.price_knowledge.is_empty() {
            let trade_base = match urgent_need {
                "purpose" => 0.5,
                "esteem" => 0.3,
                _ => 0.2,
            };
            let trade_emotion = -fear_mod * 0.2 - anxiety_mod;
            let trade_personality =
                  (if pers.curiosity > 0.6 { 0.2 } else { 0.0 })
                + (if pers.ambition > 0.6 { 0.1 } else { 0.0 });

            for report in &npc.price_knowledge {
                if report.settlement_id == settlement_id { continue; }
                let staleness = (state.tick.saturating_sub(report.tick_observed)) as f32 / STALE_TICKS;
                if staleness > 3.0 { continue; }
                let confidence = (1.0 - staleness * 0.3).max(0.1);

                // Belief modifier: prosperous destination boosts trade.
                let belief_mod = npc.memory.has_belief(
                    &BeliefType::SettlementProsperous(report.settlement_id)
                ).unwrap_or(0.0) * 0.2;

                for c in 0..crate::world_sim::NUM_COMMODITIES {
                    let local_price = settlement.prices[c];
                    let remote_price = report.prices[c];
                    let margin = remote_price - local_price;
                    if margin < MIN_TRADE_MARGIN { continue; }

                    let carry = (entity.level as f32 * 5.0).max(5.0);
                    let afford = if local_price > 0.01 { npc.gold / local_price } else { carry };
                    let units = carry.min(afford).min(settlement.stockpile[c]);
                    if units < 1.0 { continue; }

                    let profit_factor = (margin * units * confidence).min(20.0) / 20.0;
                    let effective_base = trade_base + profit_factor * 0.4;

                    let trade_util = effective_base
                        * (1.0 + trade_emotion * 0.3 + belief_mod + trade_personality * 0.2);

                    if trade_util > best_trade_utility {
                        best_trade_utility = trade_util;
                        best_trade_dest = Some(report.settlement_id);
                        best_trade_commodity = c;
                    }
                }
            }
        }

        // ----- Candidate action: Combat mobilize -----
        let combat_aptitude = npc.behavior_value(tags::COMBAT) + npc.behavior_value(tags::MELEE)
            + npc.behavior_value(tags::DEFENSE);
        let threat_high = settlement.threat_level > 0.6;

        let mobilize_base = if threat_high {
            match urgent_need {
                "safety" => 0.6,
                "esteem" => 0.4,
                _ => 0.25,
            }
        } else {
            0.05
        };
        let mobilize_emotion = anger_mod * 0.3 - fear_mod * 0.2;
        let mobilize_personality =
              (if pers.risk_tolerance > 0.6 { 0.2 } else if pers.risk_tolerance < 0.3 { -0.2 } else { 0.0 })
            + (if pers.compassion > 0.6 { 0.1 } else { 0.0 });
        let mobilize_aptitude = (combat_aptitude * 0.01).min(0.3);
        let mobilize_utility = mobilize_base
            * (1.0 + mobilize_emotion * 0.3 + mobilize_personality * 0.2)
            + mobilize_aptitude;

        // ----- Candidate action: Relocate -----
        let mut best_reloc_utility = f32::NEG_INFINITY;
        let mut best_reloc_dest: Option<u32> = None;

        let reloc_base = match urgent_need {
            "safety" if threat_high => 0.7,
            "safety" => 0.4,
            "shelter" => 0.4,
            _ => 0.1,
        };
        // Belief: current location dangerous boosts relocation.
        let loc_danger_belief = npc.memory.has_belief(
            &BeliefType::LocationDangerous(settlement_id)
        ).unwrap_or(0.0) * 0.3;
        let reloc_emotion = fear_mod * 0.3 - anxiety_mod;
        let reloc_personality = if pers.curiosity > 0.6 { 0.2 } else { 0.0 };

        for other in &state.settlements {
            if other.id == settlement_id { continue; }
            let other_safety = 1.0 / (1.0 + other.threat_level * 0.05);
            // Slightly prefer settlements with better safety.
            let dest_bonus = (other_safety - 0.5).max(0.0) * 0.3;
            let dest_prosperous = npc.memory.has_belief(
                &BeliefType::SettlementProsperous(other.id)
            ).unwrap_or(0.0) * 0.2;

            let reloc_util = (reloc_base + dest_bonus + dest_prosperous + loc_danger_belief)
                * (1.0 + reloc_emotion * 0.3 + reloc_personality * 0.2);

            if reloc_util > best_reloc_utility {
                best_reloc_utility = reloc_util;
                best_reloc_dest = Some(other.id);
            }
        }

        // ----- Candidate action: Seek companionship (social) -----
        let social_base = match urgent_need {
            "social" => 0.7,
            _ => 0.1,
        };
        let social_emotion = grief_mod * 0.4 - anxiety_mod;
        let social_personality = if pers.social_drive > 0.6 { 0.2 } else { 0.0 };
        let social_utility = social_base * (1.0 + social_emotion * 0.3 + social_personality * 0.2);

        // ----- Candidate action: Quest/Adventure -----
        let quest_base = match urgent_need {
            "purpose" => 0.8,
            "esteem" => 0.6,
            _ => 0.15,
        };
        let quest_emotion = pride_mod * 0.2 + anger_mod * 0.1 - fear_mod * 0.15;
        let quest_personality =
              (if pers.risk_tolerance > 0.6 { 0.2 } else if pers.risk_tolerance < 0.3 { -0.2 } else { 0.0 })
            + (if pers.ambition > 0.6 { 0.15 } else { 0.0 })
            + (if pers.curiosity > 0.6 { 0.1 } else { 0.0 });
        let quest_utility = quest_base * (1.0 + quest_emotion * 0.3 + quest_personality * 0.2);

        // ----- Candidate action: Go home (shelter) -----
        let home_base = match urgent_need {
            "shelter" => 0.8,
            "hunger" if !produces_food => 0.3,
            _ => 0.1,
        };
        let home_utility = home_base * (1.0 - anxiety_mod * 0.1);

        // =================================================================
        // Pick best action
        // =================================================================
        // Collect (label, utility) and pick highest.
        let candidates: [(&str, f32); 7] = [
            ("work",      work_utility),
            ("trade",     best_trade_utility.max(0.0)),
            ("mobilize",  mobilize_utility),
            ("relocate",  best_reloc_utility.max(0.0)),
            ("social",    social_utility),
            ("quest",     quest_utility),
            ("home",      home_utility),
        ];

        let (best_action, _best_score) = candidates.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let mut chosen_intent = EconomicIntent::Produce;

        match *best_action {
            "work" => {
                chosen_intent = EconomicIntent::Produce;
            }
            "trade" => {
                if let Some(dest_id) = best_trade_dest {
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
                        let cost = units * local_price;
                        out.push(WorldDelta::TransferGold {
                            from_id: entity.id,
                            to_id: settlement_id,
                            amount: cost,
                        });
                        if state.settlement(dest_id).is_some() {
                            chosen_intent = EconomicIntent::Trade {
                                destination_settlement_id: dest_id,
                            };
                            let mut action = ActionTags::empty();
                            action.add(tags::TRADE, 1.0);
                            action.add(tags::NEGOTIATION, 0.5);
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
            "mobilize" => {
                // Only ~20% mobilize per cycle (deterministic hash).
                let mob_roll = entity_hash_f32(entity.id, state.tick, 0xF16B7);
                if threat_high && mob_roll < 0.2 {
                    chosen_intent = EconomicIntent::Idle;
                    let mut action = ActionTags::empty();
                    action.add(tags::COMBAT, 1.0);
                    action.add(tags::DEFENSE, 0.5);
                    action.add(tags::TACTICS, 0.3);
                    let action = crate::world_sim::action_context::with_context(&action, entity, state);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                }
            }
            "relocate" => {
                if let Some(dest_id) = best_reloc_dest {
                    if let Some(dest) = state.settlement(dest_id) {
                        chosen_intent = EconomicIntent::Travel {
                            destination: dest.pos,
                        };
                    }
                }
            }
            "social" => {
                // Social seeking: stay in settlement, gain social tags.
                chosen_intent = EconomicIntent::Produce; // stays local
                let mut action = ActionTags::empty();
                action.add(tags::DIPLOMACY, 0.5);
                action.add(tags::NEGOTIATION, 0.3);
                let action = crate::world_sim::action_context::with_context(&action, entity, state);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: entity.id,
                    tags: action.tags,
                    count: action.count,
                });
            }
            "quest" => {
                // Quest/adventure: for now, treated as producing with purpose.
                // When quest system integration is ready, this will set Adventuring intent.
                chosen_intent = EconomicIntent::Produce;
                let mut action = ActionTags::empty();
                action.add(tags::EXPLORATION, 1.0);
                action.add(tags::COMBAT, 0.3);
                let action = crate::world_sim::action_context::with_context(&action, entity, state);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: entity.id,
                    tags: action.tags,
                    count: action.count,
                });
            }
            "home" => {
                // Go home: travel to home settlement if away, else stay local.
                if let Some(home_sid) = npc.home_settlement_id {
                    if home_sid != settlement_id {
                        if let Some(home_s) = state.settlement(home_sid) {
                            chosen_intent = EconomicIntent::Travel {
                                destination: home_s.pos,
                            };
                        }
                    }
                }
                // If already home, just stay (Produce).
            }
            _ => {}
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
            let entity_a = state.entity(a_id);
            let entity_b = state.entity(b_id);
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
