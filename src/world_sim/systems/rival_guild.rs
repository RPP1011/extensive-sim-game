#![allow(unused)]
//! Rival guild AI — fires every 7 ticks (activates at tick 67).
//!
//! A competing faction that steals quests, recruits, and sabotages the
//! player. In the delta architecture, rival actions are modeled as gold
//! drains on settlements and damage to NPCs.
//!
//! Original: `crates/headless_campaign/src/systems/rival_guild.rs`
//!
//! NEEDS STATE: `rival_guild: RivalGuildState` on WorldState
//! NEEDS DELTA: RivalAction (quest stolen, sabotage, etc.)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// Grace period before rival activates.
const ACTIVATION_TICK: u64 = 67;

/// Rival action cadence.
const TICK_CADENCE: u64 = 7;

/// Gold stolen per sabotage event.
const SABOTAGE_GOLD_DRAIN: f32 = 10.0;

/// Damage from rival raiders.
const RAID_DAMAGE: f32 = 5.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_rival_guild(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick < ACTIVATION_TICK {
        return;
    }
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    // Without rival_guild state, we approximate rival actions as periodic
    // threats against settlements and NPCs.

    for settlement in &state.settlements {
        let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xE1FA1);

        // 10% chance of sabotage per settlement per cycle
        if roll < 0.10 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -SABOTAGE_GOLD_DRAIN,
            });
        }

        // 5% chance of raiding NPCs in the settlement
        if roll > 0.95 {
            if let Some(grid_id) = settlement.grid_id {
                if let Some(grid) = state.grid(grid_id) {
                    for &entity_id in &grid.entity_ids {
                        if let Some(entity) = state.entity(entity_id) {
                            if entity.alive && entity.kind == EntityKind::Npc {
                                out.push(WorldDelta::Damage {
                                    target_id: entity_id,
                                    amount: RAID_DAMAGE,
                                    source_id: 0,
                                });
                                break; // One raid target per settlement
                            }
                        }
                    }
                }
            }
        }
    }
}
