#![allow(unused)]
//! Divine favor economy — fires every 7 ticks.
//!
//! Accumulated devotion from temples/shrines can grant miracles (heals,
//! gold blessings). Displeasure causes damage. Maps to Heal, Damage,
//! and UpdateTreasury deltas.
//!
//! Original: `crates/headless_campaign/src/systems/divine_favor.rs`
//!
//! NEEDS STATE: `divine_favor: Vec<DivineFavorEntry>` on WorldState
//! NEEDS STATE: `temples: Vec<Temple>` on WorldState
//! NEEDS DELTA: ModifyFavor

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often the divine favor system ticks.
const DIVINE_FAVOR_INTERVAL: u64 = 7;

/// Miracle heal amount.
const MIRACLE_HEAL: f32 = 15.0;

/// Miracle gold blessing amount.
const MIRACLE_GOLD: f32 = 20.0;

/// Displeasure damage amount.
const DISPLEASURE_DAMAGE: f32 = 5.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_divine_favor(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DIVINE_FAVOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without temples/divine_favor on WorldState, we approximate:
    // Settlements with high treasury (proxy for active temples) occasionally
    // grant healing miracles to nearby NPCs. Settlements with very low treasury
    // suffer divine displeasure (damage).

    for settlement in &state.settlements {
        let grid_id = match settlement.grid_id {
            Some(id) => id,
            None => continue,
        };
        let grid = match state.grid(grid_id) {
            Some(g) => g,
            None => continue,
        };

        if settlement.treasury > 100.0 {
            // Miracle: heal a random NPC in the settlement
            let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xFA17_4001);
            if roll < 0.15 {
                for &entity_id in &grid.entity_ids {
                    if let Some(entity) = state.entity(entity_id) {
                        if entity.alive && entity.kind == EntityKind::Npc && entity.hp < entity.max_hp {
                            out.push(WorldDelta::Heal {
                                target_id: entity_id,
                                amount: MIRACLE_HEAL,
                                source_id: 0,
                            });
                            break; // One miracle per settlement per tick
                        }
                    }
                }
            }

            // Gold blessing to settlement treasury
            let roll2 = tick_hash(state.tick, settlement.id as u64 ^ 0xB1E55);
            if roll2 < 0.10 {
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: MIRACLE_GOLD,
                });
            }
        } else if settlement.treasury < 10.0 {
            // Displeasure: damage NPCs
            let roll = tick_hash(state.tick, settlement.id as u64 ^ 0xD15FEEA5E);
            if roll < 0.10 {
                for &entity_id in &grid.entity_ids {
                    if let Some(entity) = state.entity(entity_id) {
                        if entity.alive && entity.kind == EntityKind::Npc {
                            out.push(WorldDelta::Damage {
                                target_id: entity_id,
                                amount: DISPLEASURE_DAMAGE,
                                source_id: 0,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }
}
