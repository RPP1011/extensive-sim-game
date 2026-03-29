#![allow(unused)]
//! Seasonal festivals — fires every 17 ticks.
//!
//! Settlements host festivals that provide economic and morale effects.
//! Festival types depend on season. In the delta architecture, festivals
//! map to UpdateTreasury (trade fairs), Heal (morale), and commodity
//! production bonuses.
//!
//! Original: `crates/headless_campaign/src/systems/festivals.rs`
//!
//! NEEDS STATE: `active_festivals: Vec<Festival>` on WorldState
//! NEEDS STATE: `season` on WorldState
//! NEEDS DELTA: StartFestival, EndFestival

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

use super::seasons::{current_season, Season};

/// Festival check interval.
const FESTIVAL_INTERVAL: u64 = 17;

/// Treasury bonus from a trade fair.
const TRADE_FAIR_BONUS: f32 = 10.0;

/// Food production bonus from a harvest feast.
const HARVEST_FEAST_FOOD: f32 = 5.0;

pub fn compute_festivals(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FESTIVAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_festivals_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_festivals_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % FESTIVAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let season = current_season(state.tick);

    match season {
        Season::Autumn => {
            out.push(WorldDelta::ProduceCommodity {
                location_id: settlement_id,
                commodity: 0,
                amount: HARVEST_FEAST_FOOD,
            });
        }
        Season::Summer => {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement_id,
                delta: TRADE_FAIR_BONUS,
            });
        }
        Season::Spring | Season::Winter => {}
    }
}
