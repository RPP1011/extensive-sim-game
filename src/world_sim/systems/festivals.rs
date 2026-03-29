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

    let season = current_season(state.tick);

    // Without festival tracking, we apply seasonal bonuses directly:
    // Each season provides a different passive benefit to settlements.

    for settlement in &state.settlements {
        match season {
            Season::Autumn => {
                // Harvest Feast: food production bonus
                out.push(WorldDelta::ProduceCommodity {
                    location_id: settlement.id,
                    commodity: 0, // food
                    amount: HARVEST_FEAST_FOOD,
                });
            }
            Season::Summer => {
                // Trade Fair: treasury bonus
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: TRADE_FAIR_BONUS,
                });
            }
            Season::Spring | Season::Winter => {
                // Spring: religious ceremony — no direct economic effect
                // Winter: war memorial — no direct economic effect
            }
        }
    }
}
