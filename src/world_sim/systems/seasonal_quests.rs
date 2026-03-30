#![allow(unused)]
//! Seasonal quest variants — fires every 17 ticks.
//!
//! Season-themed quests appear on season change and expire at season end.
//! Completing all seasonal quests earns a "Season Champion" bonus. In the
//! delta architecture, quest rewards map to UpdateTreasury and gold transfers.
//!
//! Original: `crates/headless_campaign/src/systems/seasonal_quests.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

use super::seasons::{current_season, Season, TICKS_PER_SEASON};

/// Seasonal quest check interval.
const CHECK_INTERVAL: u64 = 17;

/// Gold reward for completing seasonal quests (distributed to settlements).
const SEASONAL_REWARD_GOLD: f32 = 15.0;

pub fn compute_seasonal_quests(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without seasonal quest tracking, we apply a seasonal bonus:
    // At the start of each season (tick within first CHECK_INTERVAL of season cycle),
    // distribute a small gold bonus to all settlements as "seasonal quest rewards".

    let season_tick = state.tick % TICKS_PER_SEASON;
    let just_changed = season_tick < CHECK_INTERVAL;

    if just_changed {
        for settlement in &state.settlements {
            out.push(WorldDelta::UpdateTreasury {
                settlement_id: settlement.id,
                delta: SEASONAL_REWARD_GOLD,
            });
        }
    }
}
