//! Food consumption system — every 3 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/food.rs`.
//! Parties (groups of NPCs at a settlement) consume food (commodity 0)
//! proportional to their member count. Food is drawn from the settlement
//! stockpile. Starvation is modeled as status-effect damage.
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, Entity, EntityField, WorldState, tags};

/// How often the food system ticks.
const FOOD_TICK_INTERVAL: u64 = 3;

/// Food consumed per meal (one NPC eats this much per food tick).
/// A farmer produces 0.15 × level_mult(0.6) = 0.09 food per food tick.
/// One farmer feeds ~3 NPCs at this meal size. Sustainable with ~33% farmers.
const MEAL_SIZE: f32 = 0.03;

/// HP healed per meal. Eating restores a small amount of health.
const MEAL_HEAL: f32 = 0.5;

/// Small morale boost from eating a meal (contentment).
const MEAL_MORALE_BOOST: f32 = 0.5;

/// Morale penalty per food tick when starving.
const STARVATION_MORALE_PENALTY: f32 = -2.0;

/// Commodity index for food.
const COMMODITY_FOOD: usize = 0;

/// HP damage per tick when starving (no food available).
/// Starvation damage per food tick when food runs out.
/// At 100 HP, this takes ~100 ticks of complete starvation to kill.
const STARVATION_DAMAGE: f32 = 0.3;

pub fn compute_food(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FOOD_TICK_INTERVAL != 0 {
        return;
    }

    // For each settlement, count resident NPCs and consume food from stockpile.
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_food_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // Traveling NPCs (not at any settlement) consume carried food.
    // NOTE: this loop is not settlement-scoped and remains in the top-level function.
    for entity in &state.entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        // Skip NPCs that are homed at a settlement (handled above).
        if npc.home_settlement_id.is_some() {
            continue;
        }
        // Traveling NPC eats from carried goods.
        let carried_food = entity.inv_commodity(COMMODITY_FOOD);
        let consume = MEAL_SIZE.min(carried_food);
        if consume > 0.0 {
            // Model carried food consumption as TransferCommodity from self to self
            // (the apply phase will clamp to available).
            out.push(WorldDelta::TransferCommodity {
                from_entity: entity.id,
                to_entity: entity.id,
                commodity: COMMODITY_FOOD,
                amount: consume,
            });
        }
        // Starvation for homeless NPCs with no food.
        if carried_food < MEAL_SIZE {
            let severity = (1.0 - carried_food / MEAL_SIZE).clamp(0.0, 1.0);
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: STARVATION_DAMAGE * severity,
                source_id: 0,
            });
        }
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_food_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % FOOD_TICK_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // --- NPC-driven production ---
    // Each NPC produces commodities based on their behavior_production tags.
    // Production only happens for NPCs with Working or Produce intent.
    // Output scaled by NPC level. This is the core economic engine.
    let mut resident_count = 0u32;
    let mut resident_ids = [0u32; 512];

    // Batch per-commodity totals so we push ONE Produce/Consume delta per
    // settlement-commodity pair instead of one per NPC per commodity.
    // At 20K NPCs this collapses ~100K deltas/tick into ~20.
    const NUM_COMMODITIES: usize = 8; // FOOD..MEDICINE
    let mut produce_totals = [0.0f32; NUM_COMMODITIES];
    let mut consume_totals = [0.0f32; NUM_COMMODITIES];
    let mut total_earnings_paid = 0.0f32;

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        if resident_count < 512 {
            resident_ids[resident_count as usize] = entity.id;
        }
        resident_count += 1;

        // Only working NPCs produce.
        let working = matches!(npc.economic_intent,
            crate::world_sim::state::EconomicIntent::Produce
            | crate::world_sim::state::EconomicIntent::Idle
        );
        if !working { continue; }

        let level_mult = 0.5 + entity.level as f32 * 0.1;
        let mut produced_anything = false;

        // Fallback: NPCs with no production assignments forage for food.
        if npc.behavior_production.is_empty() {
            let forage = 0.02 * level_mult;
            produce_totals[COMMODITY_FOOD] += forage;
            produced_anything = true;
        }

        for &(commodity, rate) in &npc.behavior_production {
            if rate <= 0.0 { continue; }
            let amount = rate * level_mult;

            // Recipe check: processed goods require raw material inputs.
            // Raw materials (food, iron, wood, herbs, hide, crystal) need no inputs.
            // Equipment: 2 iron + 1 wood per unit.
            // Medicine: 2 herbs per unit.
            use crate::world_sim::commodity as c;

            // inputs: [(commodity, amount_per_unit); 2] — stack allocated, no Vec.
            let (recipe, recipe_count): ([(usize, f32); 2], usize) = match commodity {
                c::EQUIPMENT => ([(c::IRON, 2.0), (c::WOOD, 1.0)], 2),
                c::MEDICINE => ([(c::HERBS, 2.0), (0, 0.0)], 1),
                _ => ([(0, 0.0); 2], 0), // raw material, no inputs
            };

            // Check input availability and compute production scale.
            let mut scale = 1.0f32;
            for i in 0..recipe_count {
                let (input_c, per_unit) = recipe[i];
                let needed = amount * per_unit;
                let avail = settlement.stockpile[input_c];
                if needed > 0.0 {
                    scale = scale.min(avail / needed);
                }
            }
            scale = scale.clamp(0.0, 1.0);

            if recipe_count > 0 && scale < 0.1 {
                continue; // not enough inputs to produce anything meaningful
            }

            let actual_amount = amount * scale;

            // Consume inputs (batch into per-commodity total).
            for i in 0..recipe_count {
                let (input_c, per_unit) = recipe[i];
                let consume = actual_amount * per_unit;
                if consume > 0.0 {
                    consume_totals[input_c] += consume;
                }
            }

            if actual_amount > 0.0 {
                produce_totals[commodity] += actual_amount;
                produced_anything = true;
            }
        }

        // NPC sells produced goods to settlement at local price.
        // The aggregate treasury drain is batched; per-NPC gold updates
        // remain because they target individual entities.
        if produced_anything {
            let mut earnings = 0.0f32;
            for &(commodity, rate) in &npc.behavior_production {
                if rate > 0.0 {
                    let actual = rate * level_mult;
                    earnings += actual * settlement.prices[commodity];
                }
            }
            if earnings > 0.01 && settlement.treasury > 0.0 {
                // NOTE: individual gold update still per-NPC; aggregate
                // treasury drain is pushed once after the loop. Earnings
                // are capped against a per-NPC treasury share (approx),
                // using settlement.treasury since we can't see the running
                // total inside the parallel variant.
                let pay = earnings.min(settlement.treasury);
                total_earnings_paid += pay;
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: crate::world_sim::state::EntityField::Gold,
                    value: pay,
                });
            }
        }

        // Behavior tags: earned by doing the work.
        if produced_anything {
            for &(commodity, rate) in &npc.behavior_production {
                if rate <= 0.0 { continue; }
                let mut action = ActionTags::empty();
                use crate::world_sim::commodity;
                let commodity_tag = match commodity {
                    commodity::FOOD => tags::FARMING,
                    commodity::IRON => tags::MINING,
                    commodity::WOOD => tags::WOODWORK,
                    commodity::HERBS => tags::HERBALISM,
                    commodity::HIDE => tags::SURVIVAL,
                    commodity::CRYSTAL => tags::ALCHEMY,
                    commodity::EQUIPMENT => tags::SMITHING,
                    commodity::MEDICINE => tags::MEDICINE,
                    _ => tags::LABOR,
                };
                action.add(commodity_tag, 1.0);
                action.add(tags::LABOR, 0.5);
                // Crafting-specific tags for processed goods.
                if commodity == commodity::EQUIPMENT || commodity == commodity::MEDICINE {
                    action.add(tags::CRAFTING, 0.8);
                }
                let action = crate::world_sim::action_context::with_context(&action, entity, state);
                out.push(WorldDelta::AddBehaviorTags { entity_id: entity.id, tags: action.tags, count: action.count });
            }
        }
    }

    // Flush per-commodity aggregates as single deltas per settlement.
    for i in 0..NUM_COMMODITIES {
        if produce_totals[i] > 0.0 {
            out.push(WorldDelta::ProduceCommodity {
                settlement_id,
                commodity: i,
                amount: produce_totals[i],
            });
        }
        if consume_totals[i] > 0.0 {
            out.push(WorldDelta::ConsumeCommodity {
                settlement_id,
                commodity: i,
                amount: consume_totals[i],
            });
        }
    }
    if total_earnings_paid > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id,
            delta: -total_earnings_paid,
        });
    }

    if resident_count == 0 {
        return;
    }

    // --- Eating: NPCs consume food as an action ---
    // Each NPC eats one meal per food tick if food is available.
    // Eating heals a small amount. No food = slow starvation.
    let food_available = settlement.stockpile[COMMODITY_FOOD];
    let count = (resident_count as usize).min(512);

    if food_available > 0.0 {
        // Food available — NPCs eat. Each meal is a small amount.
        let meal_size = MEAL_SIZE;
        let meals_possible = (food_available / meal_size) as usize;
        let eaters = count.min(meals_possible);

        if eaters > 0 {
            // Consume food from stockpile.
            out.push(WorldDelta::ConsumeCommodity {
                settlement_id: settlement_id,
                commodity: COMMODITY_FOOD,
                amount: meal_size * eaters as f32,
            });

            // Eating heals, boosts morale, and grants farming behavior (food preparation).
            for i in 0..eaters {
                out.push(WorldDelta::Heal {
                    target_id: resident_ids[i],
                    amount: MEAL_HEAL,
                    source_id: 0,
                });
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: resident_ids[i],
                    field: EntityField::Morale,
                    value: MEAL_MORALE_BOOST,
                });
            }
        }

        // NPCs who didn't get food: slow starvation + morale drop.
        if eaters < count {
            for i in eaters..count {
                out.push(WorldDelta::Damage {
                    target_id: resident_ids[i],
                    amount: STARVATION_DAMAGE,
                    source_id: 0,
                });
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: resident_ids[i],
                    field: EntityField::Morale,
                    value: STARVATION_MORALE_PENALTY,
                });
            }
        }
    } else {
        // No food at all — everyone starves slowly.
        for i in 0..count {
            out.push(WorldDelta::Damage {
                target_id: resident_ids[i],
                amount: STARVATION_DAMAGE,
                source_id: 0,
            });
            out.push(WorldDelta::UpdateEntityField {
                entity_id: resident_ids[i],
                field: EntityField::Morale,
                value: STARVATION_MORALE_PENALTY,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn consumes_food_from_settlement() {
        let mut state = WorldState::new(42);
        state.tick = 3; // food tick
        let mut s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        s.stockpile[0] = 100.0; // plenty of food
        state.settlements.push(s);

        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().home_settlement_id = Some(10);
        state.entities.push(npc);
        state.rebuild_group_index();

        let mut deltas = Vec::new();
        compute_food(&state, &mut deltas);

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
        assert!(has_consume, "should consume food from settlement stockpile");
    }

    #[test]
    fn starvation_when_no_food() {
        let mut state = WorldState::new(42);
        state.tick = 3;
        let s = SettlementState::new(10, "Town".into(), (0.0, 0.0));
        // stockpile[0] is 0.0 by default
        state.settlements.push(s);

        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().home_settlement_id = Some(10);
        state.entities.push(npc);
        state.rebuild_group_index();

        let mut deltas = Vec::new();
        compute_food(&state, &mut deltas);

        let has_damage = deltas
            .iter()
            .any(|d| matches!(d, WorldDelta::Damage { target_id: 1, .. }));
        assert!(
            has_damage,
            "NPCs should take starvation damage when no food"
        );
    }

    #[test]
    fn skips_off_cadence() {
        let mut state = WorldState::new(42);
        state.tick = 1; // not a food tick
        let mut deltas = Vec::new();
        compute_food(&state, &mut deltas);
        assert!(deltas.is_empty());
    }
}
