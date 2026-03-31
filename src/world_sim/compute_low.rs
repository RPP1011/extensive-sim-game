//! Low-fidelity entity compute: overworld movement as deltas.
//!
//! NPCs travel along the overworld toward destinations.
//! Monsters move toward nearest settlement.

use super::delta::WorldDelta;
use super::state::{Entity, EntityKind, EconomicIntent, WorldState};

/// Compute deltas for an entity at Low fidelity (overworld).
pub fn compute_entity_deltas(entity: &Entity, state: &WorldState) -> Vec<WorldDelta> {
    let mut out = Vec::new();
    compute_entity_deltas_into(entity, state, &mut out);
    out
}

/// Push deltas into `out` without allocating.
pub fn compute_entity_deltas_into(entity: &Entity, state: &WorldState, out: &mut Vec<WorldDelta>) {
    match entity.kind {
        EntityKind::Npc => compute_npc_overworld_into(entity, state, out),
        EntityKind::Monster => compute_monster_overworld_into(entity, state, out),
        _ => {}
    }
}

fn compute_npc_overworld_into(entity: &Entity, state: &WorldState, out: &mut Vec<WorldDelta>) {
    let npc = match &entity.npc {
        Some(n) => n,
        None => return,
    };

    let destination = match &npc.economic_intent {
        EconomicIntent::Travel { destination } => Some(*destination),
        EconomicIntent::Trade { destination_settlement_id } => {
            state.settlement(*destination_settlement_id).map(|s| s.pos)
        }
        _ => {
            npc.home_settlement_id
                .and_then(|id| state.settlement(id))
                .map(|s| s.pos)
        }
    };

    // NPC overworld movement is handled by goal_eval setting entity.move_target,
    // which advance_movement() processes each tick.
    let _ = destination;

    let carried_food = entity.inv_commodity(0);
    if carried_food > 0.0 {
        let travel_food: f32 = 0.005;
        out.push(WorldDelta::TransferCommodity {
            from_entity: entity.id,
            to_entity: entity.id,
            commodity: 0,
            amount: travel_food.min(carried_food),
        });
    }
}

fn compute_monster_overworld_into(_entity: &Entity, _state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Monster overworld drift is handled by move_target set in monster_ecology
    // and processed by advance_movement() each tick.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn monster_no_move_delta() {
        let mut state = WorldState::new(42);
        state.entities.push(Entity::new_monster(1, (20.0, 0.0), 1));
        state.settlements.push(SettlementState::new(10, "Town".into(), (0.0, 0.0)));

        // Monster movement now handled by move_target + advance_movement.
        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(!deltas.iter().any(|d| matches!(d, WorldDelta::Move { .. })));
    }

    #[test]
    fn npc_trade_no_move_delta() {
        let mut state = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().economic_intent = EconomicIntent::Trade {
            destination_settlement_id: 10,
        };
        state.entities.push(npc);
        state.settlements.push(SettlementState::new(10, "Market".into(), (50.0, 0.0)));

        // NPC movement now handled by move_target + advance_movement.
        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(!deltas.iter().any(|d| matches!(d, WorldDelta::Move { .. })));
    }
}
