//! Quest expiry — every tick.
//!
//! Removes quest requests that have passed their deadline.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::CampaignState;

pub fn tick_quest_expiry(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let now = state.elapsed_ms;
    let mut expired_ids = Vec::new();

    for req in &state.request_board {
        if now >= req.deadline_ms {
            expired_ids.push(req.id);
        }
    }

    for id in &expired_ids {
        events.push(WorldEvent::QuestRequestExpired { request_id: *id });
        deltas.quests_expired += 1;
    }

    state.request_board.retain(|r| !expired_ids.contains(&r.id));
}
