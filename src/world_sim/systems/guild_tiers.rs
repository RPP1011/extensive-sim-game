#![allow(unused)]
//! Guild tier progression system.
//! NEEDS STATE: guild_tier, guild_reputation on EconomyState
//! NEEDS DELTA: AdvanceGuildTier

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

pub fn compute_guild_tiers(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % 17 != 0 { return; }
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_guild_tiers_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_guild_tiers_for_settlement(
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
    let tier = ((settlement.treasury / 500.0).floor() as u32).min(5);
    if tier > 0 {
        out.push(WorldDelta::ProduceCommodity {
            location_id: settlement_id,
            commodity: crate::world_sim::commodity::FOOD,
            amount: tier as f32 * 0.02,
        });
    }
}
