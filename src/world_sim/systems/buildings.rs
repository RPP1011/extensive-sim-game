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

    // Without per-settlement building tiers, we approximate:
    // Settlements with high treasury automatically spend on upgrades,
    // reducing treasury in exchange for infrastructure (passive bonuses
    // would be read by other systems).

    for settlement in &state.settlements {
        if settlement.treasury >= UPGRADE_TREASURY_THRESHOLD {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -UPGRADE_COST,
            });
        }
    }
}
