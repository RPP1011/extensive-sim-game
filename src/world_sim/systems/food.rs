#![allow(unused)]
//! Food consumption system — every 3 ticks.
//!
//! Ported from `crates/headless_campaign/src/systems/food.rs`.
//! Parties (groups of NPCs at a settlement) consume food (commodity 0)
//! proportional to their member count. Food is drawn from the settlement
//! stockpile. Starvation is modeled as status-effect damage.
//!
//! NEEDS STATE: `party_id: Option<u32>` on NpcData (to group NPCs into parties)
//! NEEDS DELTA: ApplyFatigueAndMorale { entity_id, fatigue_delta, morale_delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

/// How often the food system ticks.
const FOOD_TICK_INTERVAL: u64 = 3;

/// Food (commodity index 0) consumed per NPC per food tick.
const FOOD_PER_NPC: f32 = 1.0;

/// Commodity index for food.
const COMMODITY_FOOD: usize = 0;

/// HP damage per tick when starving (no food available).
const STARVATION_DAMAGE: f32 = 2.0;

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
        if entity.kind != EntityKind::Npc || !entity.alive {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };
        // Skip NPCs that are homed at a settlement (handled above).
        if npc.home_settlement_id.is_some() {
            continue;
        }
        // Traveling NPC eats from carried goods.
        let carried_food = npc.carried_goods[COMMODITY_FOOD];
        let consume = FOOD_PER_NPC.min(carried_food);
        if consume > 0.0 {
            // Model carried food consumption as TransferGoods from self to self
            // (the apply phase will clamp to available).
            out.push(WorldDelta::TransferGoods {
                from_id: entity.id,
                to_id: entity.id,
                commodity: COMMODITY_FOOD,
                amount: consume,
            });
        }
        // Starvation for homeless NPCs with no food.
        if carried_food < FOOD_PER_NPC {
            let severity = (1.0 - carried_food / FOOD_PER_NPC).clamp(0.0, 1.0);
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

    for entity in entities {
        if entity.kind != EntityKind::Npc || !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        if resident_count < 512 {
            resident_ids[resident_count as usize] = entity.id;
        }
        resident_count += 1;

        // Only working NPCs produce.
        let working = matches!(npc.economic_intent,
            crate::world_sim::state::EconomicIntent::Produce
            | crate::world_sim::state::EconomicIntent::Idle // idle NPCs default to basic labor
        );
        if !working { continue; }

        let level_mult = 0.5 + entity.level as f32 * 0.1; // level 1 = 0.6x, level 10 = 1.5x
        for &(commodity, rate) in &npc.behavior_production {
            if rate > 0.0 {
                out.push(WorldDelta::ProduceCommodity {
                    location_id: settlement_id,
                    commodity,
                    amount: rate * level_mult,
                });
            }
        }
    }

    if resident_count == 0 {
        return;
    }

    // --- Consumption ---
    let food_needed = FOOD_PER_NPC * resident_count as f32;
    let food_available = settlement.stockpile[COMMODITY_FOOD];

    // Consume food from settlement stockpile.
    let consumed = food_needed.min(food_available);
    if consumed > 0.0 {
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: COMMODITY_FOOD,
            amount: consumed,
        });
    }

    // If food was insufficient, apply starvation damage to residents.
    let shortfall = food_needed - consumed;
    if shortfall > 0.0 {
        // Damage proportional to shortfall ratio.
        let severity = (shortfall / food_needed).clamp(0.0, 1.0);
        let damage = STARVATION_DAMAGE * severity;
        let count = (resident_count as usize).min(512);
        for i in 0..count {
            out.push(WorldDelta::Damage {
                target_id: resident_ids[i],
                amount: damage,
                source_id: 0, // environmental
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;
    use crate::world_sim::NUM_COMMODITIES;

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
                    location_id: 10,
                    commodity: 0,
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
