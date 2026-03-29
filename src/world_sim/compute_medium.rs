//! Medium-fidelity entity compute: settlement-level NPC behavior.
//!
//! Handles movement only. All economic behavior (production, consumption,
//! trade) is driven by NPC agent decisions in the campaign systems
//! (systems/economy.rs, systems/food.rs, systems/trade_goods.rs, etc.)
//! which use price signals and economic intent.

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

    // Movement toward economic destination.
    if let EconomicIntent::Travel { destination } = &npc.economic_intent {
        let dx = destination.0 - entity.pos.0;
        let dy = destination.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 0.1 {
            let speed = entity.move_speed * crate::world_sim::DT_SEC;
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
    fn npc_moves_toward_destination() {
        let mut state = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.economic_intent = EconomicIntent::Travel { destination: (10.0, 0.0) };
        state.entities.push(npc);

        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(deltas.iter().any(|d| matches!(d, WorldDelta::Move { entity_id: 1, .. })));
    }

    #[test]
    fn idle_npc_no_deltas() {
        let mut state = WorldState::new(42);
        let npc = Entity::new_npc(1, (0.0, 0.0));
        state.entities.push(npc);

        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(deltas.is_empty());
    }
}
