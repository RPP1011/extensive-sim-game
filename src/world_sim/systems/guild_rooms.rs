#![allow(unused)]
//! Guild room building system.
//! NEEDS STATE: guild_rooms on SettlementState
//! NEEDS DELTA: UpgradeRoom

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

pub fn compute_guild_rooms(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % 17 != 0 { return; }
    for settlement in &state.settlements {
        let room_count = (settlement.treasury / 200.0).floor().min(5.0) as u32;
        if room_count > 0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -(room_count as f32 * 0.5),
            });
        }
    }
}
