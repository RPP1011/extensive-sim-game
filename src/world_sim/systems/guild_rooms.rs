#![allow(unused)]
//! Guild room building system.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

pub fn compute_guild_rooms(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % 17 != 0 { return; }
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_guild_rooms_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_guild_rooms_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % 17 != 0 { return; }
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };
    if settlement.treasury <= 0.0 {
        return;
    }
    let room_count = (settlement.treasury / 200.0).floor().min(5.0) as u32;
    if room_count > 0 {
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: -(room_count as f32 * 0.5),
        });
    }
}
