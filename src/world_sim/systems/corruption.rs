#![allow(unused)]
//! Corruption system — fires every 7 ticks.
//!
//! Guild operations degrade over time. High corruption causes gold siphoning,
//! supply loss, and NPC desertion. In the delta architecture, corruption maps
//! to UpdateTreasury (gold drain), ConsumeCommodity (supply loss), and
//! Damage (desertion/morale damage).
//!
//! Original: `crates/headless_campaign/src/systems/corruption.rs`
//!
//! NEEDS STATE: `corruption: CorruptionState` on WorldState (level, overseer)
//! NEEDS DELTA: ModifyCorruption

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// Corruption tick interval.
const CORRUPTION_INTERVAL: u64 = 7;

/// Gold siphoned per settlement at moderate corruption.
const MODERATE_GOLD_DRAIN: f32 = 5.0;

/// Gold siphoned per settlement at severe corruption.
const SEVERE_GOLD_DRAIN: f32 = 15.0;

/// Supply loss at moderate corruption.
const SUPPLY_LOSS: f32 = 2.0;

/// Damage to NPCs at severe corruption (morale/desertion proxy).
const DESERTION_DAMAGE: f32 = 3.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_corruption(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CORRUPTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without corruption level tracking, we use settlement population and
    // treasury as proxies: large wealthy settlements attract corruption.

    for settlement in &state.settlements {
        let wealth_factor = settlement.treasury / 100.0;
        let pop_factor = settlement.population as f32 / 500.0;
        let corruption_proxy = (wealth_factor + pop_factor).min(2.0);

        if corruption_proxy < 0.5 {
            continue; // Low corruption risk
        }

        // Moderate corruption (0.5-1.0): gold siphoning
        if corruption_proxy >= 0.5 {
            let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xC0EAEB);
            if roll < 0.10 {
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: -MODERATE_GOLD_DRAIN,
                });
            }
        }

        // High corruption (1.0-1.5): supply loss
        if corruption_proxy >= 1.0 {
            let roll = tick_hash(state.tick, settlement.id as u64 ^ 0x5EFFA1);
            if roll < 0.08 {
                out.push(WorldDelta::ConsumeCommodity {
                    location_id: settlement.id,
                    commodity: 0, // food
                    amount: SUPPLY_LOSS,
                });
            }
        }

        // Severe corruption (1.5+): NPC damage (desertion)
        if corruption_proxy >= 1.5 {
            if let Some(grid_id) = settlement.grid_id {
                if let Some(grid) = state.grid(grid_id) {
                    for &entity_id in &grid.entity_ids {
                        if let Some(entity) = state.entity(entity_id) {
                            if entity.alive && entity.kind == EntityKind::Npc {
                                let roll =
                                    tick_hash(state.tick, entity_id as u64 ^ 0xDE5EBB);
                                if roll < 0.03 {
                                    out.push(WorldDelta::Damage {
                                        target_id: entity_id,
                                        amount: DESERTION_DAMAGE,
                                        source_id: 0,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
