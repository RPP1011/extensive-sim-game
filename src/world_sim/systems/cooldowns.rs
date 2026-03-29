#![allow(unused)]
//! Cooldown tick — every tick.
//!
//! Decrements active ability cooldowns on all entities. Maps directly to
//! the TickCooldown delta.
//!
//! Original: `crates/headless_campaign/src/systems/cooldowns.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Campaign turn length in milliseconds (used as dt per tick).
const DT_MS: u32 = 1000;

pub fn compute_cooldowns(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_cooldowns_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_cooldowns_for_settlement(
    _state: &WorldState,
    _settlement_id: u32,
    entities: &[super::super::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if !entity.alive {
            continue;
        }
        out.push(WorldDelta::TickCooldown {
            entity_id: entity.id,
            dt_ms: DT_MS,
        });
    }
}
