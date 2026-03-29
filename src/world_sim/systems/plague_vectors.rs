#![allow(unused)]
//! Plague vector propagation — fires every 13 ticks.
//!
//! Disease spreads between settlements via trade routes (entities traveling
//! between settlements). Active caravans accelerate spread. Settlements with
//! high stockpiles and treasury have better sanitation (lower spread chance).
//!
//! Original: `crates/headless_campaign/src/systems/plague_vectors.rs`
//!
//! NEEDS STATE: `plague_vectors: Vec<PlagueVector>` on WorldState
//! NEEDS STATE: `neighbors` on RegionState or settlement adjacency
//! NEEDS STATE: `trade_routes`, `caravans`, `migrations` on WorldState
//! NEEDS DELTA: SpawnPlague, RemovePlague, ModifyPopulation

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often plague vector propagation fires (in ticks).
const PLAGUE_VECTOR_INTERVAL: u64 = 13;

/// Damage to NPCs in plague-affected settlements.
const PLAGUE_DAMAGE: f32 = 3.0;

/// Treasury drain from plague in a settlement.
const PLAGUE_TREASURY_DRAIN: f32 = 8.0;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_plague_vectors(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PLAGUE_VECTOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without plague vector state, we model mortality effects on settlements
    // with very low treasury (proxy for overwhelmed medical capacity).

    for settlement in &state.settlements {
        // Settlements with near-zero treasury and high population suffer plague effects
        let plague_active = settlement.treasury < 5.0 && settlement.population > 50;
        if !plague_active {
            continue;
        }

        // Treasury drain from plague response
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement.id,
            delta: -PLAGUE_TREASURY_DRAIN,
        });

        // Damage NPCs in the settlement
        if let Some(grid_id) = settlement.grid_id {
            if let Some(grid) = state.grid(grid_id) {
                for &entity_id in &grid.entity_ids {
                    if let Some(entity) = state.entity(entity_id) {
                        if !entity.alive || entity.kind != EntityKind::Npc {
                            continue;
                        }
                        let roll = tick_hash(state.tick, entity_id as u64 ^ 0xBEEF_DEAD);
                        if roll < 0.20 {
                            out.push(WorldDelta::Damage {
                                target_id: entity_id,
                                amount: PLAGUE_DAMAGE,
                                source_id: 0,
                            });
                        }
                    }
                }
            }
        }
    }
}
