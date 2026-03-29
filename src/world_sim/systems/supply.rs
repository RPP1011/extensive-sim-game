#![allow(unused)]
//! Supply depletion for traveling parties — every tick.
//!
//! Ported from `crates/headless_campaign/src/systems/supply.rs`.
//! NPCs traveling away from their home settlement consume food
//! (commodity 0) from their carried goods proportional to distance.
//! Expressed as ConsumeCommodity and Damage deltas.
//!
//! NEEDS STATE: `party_id: Option<u32>` on NpcData (to group traveling NPCs)
//! NEEDS DELTA: ApplyFatigueAndMorale { entity_id, fatigue_delta, morale_delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, Entity, EntityKind, WorldState};

/// Food commodity index.
const COMMODITY_FOOD: usize = 0;

/// Base supply drain per NPC per tick while traveling.
const BASE_DRAIN_PER_TICK: f32 = 0.01;

/// Extra drain multiplier per unit of distance from home settlement.
const DISTANCE_DRAIN_FACTOR: f32 = 0.0001;

/// Damage per tick when completely out of supplies.
const OUT_OF_SUPPLY_DAMAGE: f32 = 1.0;

pub fn compute_supply(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_supply_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_supply_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    let settlement_pos = state
        .settlement(settlement_id)
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    for entity in entities {
        if entity.kind != EntityKind::Npc || !entity.alive {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Only traveling NPCs consume supply.
        let is_traveling = matches!(
            npc.economic_intent,
            EconomicIntent::Travel { .. } | EconomicIntent::Trade { .. }
        );
        if !is_traveling {
            continue;
        }

        // Calculate distance from home settlement.
        let home_dist = {
            let dx = entity.pos.0 - settlement_pos.0;
            let dy = entity.pos.1 - settlement_pos.1;
            (dx * dx + dy * dy).sqrt()
        };

        // Total drain: base + distance-proportional.
        let drain = BASE_DRAIN_PER_TICK + home_dist * DISTANCE_DRAIN_FACTOR;

        let carried_food = npc.carried_goods[COMMODITY_FOOD];
        if carried_food > 0.0 {
            let consumed = drain.min(carried_food);
            out.push(WorldDelta::TransferGoods {
                from_id: entity.id,
                to_id: entity.id,
                commodity: COMMODITY_FOOD,
                amount: consumed,
            });
        }

        // Out-of-supply: NPC has no food left and is still traveling.
        if carried_food < drain {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: OUT_OF_SUPPLY_DAMAGE,
                source_id: 0,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn traveling_npc_consumes_food() {
        let mut state = WorldState::new(42);
        state
            .settlements
            .push(SettlementState::new(10, "Home".into(), (0.0, 0.0)));

        let mut npc = Entity::new_npc(1, (50.0, 0.0)); // far from home
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(10);
        npc_data.economic_intent = EconomicIntent::Travel {
            destination: (100.0, 0.0),
        };
        npc_data.carried_goods[0] = 10.0; // has food
        state.entities.push(npc);
        state.rebuild_group_index();

        let mut deltas = Vec::new();
        compute_supply(&state, &mut deltas);

        let has_consume = deltas.iter().any(|d| {
            matches!(
                d,
                WorldDelta::TransferGoods {
                    from_id: 1,
                    commodity: crate::world_sim::commodity::FOOD,
                    ..
                }
            )
        });
        assert!(has_consume, "traveling NPC should consume carried food");
    }

    #[test]
    fn out_of_supply_damage() {
        let mut state = WorldState::new(42);
        state
            .settlements
            .push(SettlementState::new(10, "Home".into(), (0.0, 0.0)));

        let mut npc = Entity::new_npc(1, (50.0, 0.0));
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(10);
        npc_data.economic_intent = EconomicIntent::Travel {
            destination: (100.0, 0.0),
        };
        npc_data.carried_goods[0] = 0.0; // no food
        state.entities.push(npc);
        state.rebuild_group_index();

        let mut deltas = Vec::new();
        compute_supply(&state, &mut deltas);

        let has_damage = deltas
            .iter()
            .any(|d| matches!(d, WorldDelta::Damage { target_id: 1, .. }));
        assert!(has_damage, "out-of-supply NPC should take damage");
    }

    #[test]
    fn idle_npc_no_drain() {
        let mut state = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().economic_intent = EconomicIntent::Idle;
        state.entities.push(npc);

        let mut deltas = Vec::new();
        compute_supply(&state, &mut deltas);
        assert!(deltas.is_empty(), "idle NPC should not consume supply");
    }
}
