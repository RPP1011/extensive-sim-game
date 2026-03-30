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

    if let Some(dest) = destination {
        let dx = dest.0 - entity.pos.0;
        let dy = dest.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 0.5 {
            let speed = entity.move_speed * crate::world_sim::DT_SEC;
            out.push(WorldDelta::Move {
                entity_id: entity.id,
                force: (dx / dist * speed, dy / dist * speed),
            });
        }
    }

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

fn compute_monster_overworld_into(entity: &Entity, state: &WorldState, out: &mut Vec<WorldDelta>) {
    let nearest = state.settlements.iter()
        .min_by(|a, b| {
            let da = dist_sq_pos(entity.pos, a.pos);
            let db = dist_sq_pos(entity.pos, b.pos);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(settlement) = nearest {
        let dx = settlement.pos.0 - entity.pos.0;
        let dy = settlement.pos.1 - entity.pos.1;
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

fn dist_sq_pos(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn monster_moves_toward_settlement() {
        let mut state = WorldState::new(42);
        state.entities.push(Entity::new_monster(1, (20.0, 0.0), 1));
        state.settlements.push(SettlementState::new(10, "Town".into(), (0.0, 0.0)));

        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(deltas.iter().any(|d| matches!(d, WorldDelta::Move { entity_id: 1, force }
            if force.0 < 0.0 // moving left toward settlement at origin
        )));
    }

    #[test]
    fn npc_moves_toward_trade_destination() {
        let mut state = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.npc.as_mut().unwrap().economic_intent = EconomicIntent::Trade {
            destination_settlement_id: 10,
        };
        state.entities.push(npc);
        state.settlements.push(SettlementState::new(10, "Market".into(), (50.0, 0.0)));

        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(deltas.iter().any(|d| matches!(d, WorldDelta::Move { entity_id: 1, force }
            if force.0 > 0.0 // moving right toward market
        )));
    }
}
