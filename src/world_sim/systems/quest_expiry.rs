#![allow(unused)]
//! Quest expiry — every tick.
//!
//! Removes quest requests that have passed their deadline tick.
//!
//! Ported from `crates/headless_campaign/src/systems/quest_expiry.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: request_board: Vec<QuestRequest> on WorldState
// NEEDS STATE: QuestRequest { id: u32, deadline_tick: u64, ... }
// NEEDS DELTA: ExpireQuestRequest { quest_id: u32 }

pub fn compute_quest_expiry(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // The original system checks each quest request on the board against the
    // current elapsed time. If now >= deadline, the request is removed.
    //
    // In the delta architecture:
    //   for req in &state.request_board {
    //       if state.tick >= req.deadline_tick {
    //           out.push(WorldDelta::ExpireQuestRequest { quest_id: req.id });
    //       }
    //   }
    //
    // Since `request_board` does not yet exist on WorldState and there is no
    // ExpireQuestRequest delta variant, this system is currently a no-op.
    // The logic is trivial: compare tick against deadline, emit removal delta.
}
