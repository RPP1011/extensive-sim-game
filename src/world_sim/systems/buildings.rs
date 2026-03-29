#![allow(unused)]
//! Guild buildings / auto-upgrade — fires every 3 ticks.
//!
//! Settlements can upgrade structures when they have enough treasury. In the
//! delta architecture, upgrades are expressed as treasury reductions. Building
//! tier bonuses are read by other systems.
//!
//! Original: `crates/headless_campaign/src/systems/buildings.rs`
//!
//! NEEDS STATE: `buildings: GuildBuildings` per settlement on WorldState
//! NEEDS DELTA: UpgradeBuilding { settlement_id, building_type, new_tier }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Building tick interval.
const BUILDING_TICK_INTERVAL: u64 = 3;

/// Minimum treasury to trigger an auto-upgrade.
const UPGRADE_TREASURY_THRESHOLD: f32 = 200.0;

/// Cost of an upgrade (deducted from treasury).
const UPGRADE_COST: f32 = 100.0;

pub fn compute_buildings(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BUILDING_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_buildings_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_buildings_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % BUILDING_TICK_INTERVAL != 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    if settlement.treasury >= UPGRADE_TREASURY_THRESHOLD && settlement.treasury > 0.0 {
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement_id,
            delta: -UPGRADE_COST,
        });
    }
}
