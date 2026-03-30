#![allow(unused)]
//! Propaganda and public relations system — every 7 ticks.
//!
//! Allows the guild to spend gold on influence campaigns that boost reputation,
//! counter rival guilds, discredit hostile factions, recruit adventurers, or
//! raise morale during wartime.
//!
//! Max 2 active campaigns at once. Effectiveness decays over time.
//!
//! Ported from `crates/headless_campaign/src/systems/propaganda.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

//   PropagandaCampaign { id, campaign_type: PropagandaType, started_tick, duration,
//                        effectiveness, target_region_id: Option<u32>,
//                        target_faction_id: Option<u32> }
//              RecruitmentDrive, WarPropaganda }


/// Cadence: propaganda effects apply every 7 ticks.
const TICK_CADENCE: u64 = 7;

pub fn compute_propaganda(_state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Stub: propaganda campaign state not yet tracked. See git history for planned design.
}
