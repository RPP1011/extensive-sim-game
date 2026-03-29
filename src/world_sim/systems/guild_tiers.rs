#![allow(unused)]
//! Guild tier progression system.
//! NEEDS STATE: guild_tier, guild_reputation on EconomyState
//! NEEDS DELTA: AdvanceGuildTier

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

pub fn compute_guild_tiers(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % 17 != 0 { return; }
    for settlement in &state.settlements {
        let tier = ((settlement.treasury / 500.0).floor() as u32).min(5);
        if tier > 0 {
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement.id,
                commodity: 0,
                amount: tier as f32 * 0.02,
            });
        }
    }
}
