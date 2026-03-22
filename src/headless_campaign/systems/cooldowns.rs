//! Cooldown tick — every tick.
//!
//! Decrements active ability cooldowns by `CAMPAIGN_TICK_MS`.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{CampaignState, CAMPAIGN_TICK_MS};

pub fn tick_cooldowns(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    _events: &mut Vec<WorldEvent>,
) {
    let dt = CAMPAIGN_TICK_MS as u64;
    for unlock in &mut state.unlocks {
        if unlock.cooldown_remaining_ms > 0 {
            unlock.cooldown_remaining_ms = unlock.cooldown_remaining_ms.saturating_sub(dt);
        }
    }
}
