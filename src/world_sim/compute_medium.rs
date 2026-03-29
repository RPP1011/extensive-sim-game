//! Medium-fidelity entity compute: settlement activity as deltas.
//!
//! NPCs in settlements produce commodities, consume food/equipment,
//! and make economic decisions — all expressed as deltas.

use super::delta::WorldDelta;
use super::state::{Entity, EntityKind, EconomicIntent, WorldState};

/// Compute deltas for an entity at Medium fidelity (settlement).
pub fn compute_entity_deltas(entity: &Entity, state: &WorldState) -> Vec<WorldDelta> {
    let mut out = Vec::new();
    compute_entity_deltas_into(entity, state, &mut out);
    out
}

/// Push deltas into `out` without allocating.
pub fn compute_entity_deltas_into(entity: &Entity, _state: &WorldState, out: &mut Vec<WorldDelta>) {
    match entity.kind {
        EntityKind::Npc => {}
        _ => return,
    }
    let npc = match &entity.npc {
        Some(n) => n,
        None => return,
    };

    let settlement_id = npc.home_settlement_id.unwrap_or(0);

    for &(commodity, rate) in &npc.behavior_production {
        if rate > 0.0 {
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement_id,
                commodity,
                amount: rate,
            });
        }
    }

    out.push(WorldDelta::ConsumeCommodity {
        location_id: settlement_id,
        commodity: 0,
        amount: 0.01,
    });

    if let EconomicIntent::Travel { destination } = &npc.economic_intent {
        let dx = destination.0 - entity.pos.0;
        let dy = destination.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 0.1 {
            let speed = entity.move_speed * 0.1;
            out.push(WorldDelta::Move {
                entity_id: entity.id,
                force: (dx / dist * speed, dy / dist * speed),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;

    #[test]
    fn npc_produces_and_consumes() {
        let mut state = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(10);
        npc_data.behavior_production = vec![(1, 0.5)]; // produces iron
        state.entities.push(npc);
        state.settlements.push(SettlementState::new(10, "Town".into(), (0.0, 0.0)));

        let deltas = compute_entity_deltas(&state.entities[0], &state);
        // Should have production + consumption deltas.
        assert!(deltas.iter().any(|d| matches!(d,
            WorldDelta::ProduceCommodity { commodity: 1, .. }
        )));
        assert!(deltas.iter().any(|d| matches!(d,
            WorldDelta::ConsumeCommodity { commodity: 0, .. }
        )));
    }
}
