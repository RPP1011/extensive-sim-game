#![allow(unused)]
//! Disease outbreak system — fires every 7 ticks.
//!
//! Diseases can outbreak in regions with high population, spread to adjacent
//! regions, cause population/treasury loss, and damage entities in diseased
//! areas. Healer-class NPCs slow spread.
//!
//! Original: `crates/headless_campaign/src/systems/disease.rs`
//!
//! NEEDS STATE: `diseases: Vec<DiseaseState>` on WorldState (outbreak tracking)
//! NEEDS STATE: `population`, `civilian_morale`, `neighbors` on RegionState
//! NEEDS STATE: `archetype` / `class_tags` on NpcData for healer detection
//! NEEDS DELTA: SpawnDisease, RemoveDisease, ModifyPopulation, ModifyMorale

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// How often to tick the disease system (in ticks).
const DISEASE_INTERVAL: u64 = 7;

/// Damage dealt per tick to NPCs in diseased settlements.
const DISEASE_DAMAGE_PER_TICK: f32 = 2.0;

/// Treasury drain per tick for settlements affected by disease.
const DISEASE_TREASURY_DRAIN: f32 = 5.0;

/// Supply consumption increase in diseased settlements.
const DISEASE_SUPPLY_DRAIN: f32 = 1.0;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_disease(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DISEASE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without full disease tracking on WorldState, we approximate:
    // Settlements with low treasury and high population are "at risk".
    // NPCs in those settlements take periodic damage (infection proxy).

    for settlement in &state.settlements {
        // Risk heuristic: low treasury + high population
        let is_at_risk = settlement.treasury < 20.0 && settlement.population > 100;
        if !is_at_risk {
            continue;
        }

        // Treasury drain from disease management costs
        out.push(WorldDelta::UpdateTreasury {
            location_id: settlement.id,
            delta: -DISEASE_TREASURY_DRAIN,
        });

        // Consume extra supplies (commodity 0 = food) for medical needs
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement.id,
            commodity: 0,
            amount: DISEASE_SUPPLY_DRAIN,
        });

        // Damage NPCs in this settlement's grid
        if let Some(grid_id) = settlement.grid_id {
            if let Some(grid) = state.grid(grid_id) {
                for &entity_id in &grid.entity_ids {
                    if let Some(entity) = state.entity(entity_id) {
                        if !entity.alive || entity.kind != EntityKind::Npc {
                            continue;
                        }
                        // Healer-class NPCs are partially resistant
                        let is_healer = entity.npc.as_ref().map_or(false, |npc| {
                            npc.class_tags.iter().any(|t| {
                                t.contains("healer")
                                    || t.contains("cleric")
                                    || t.contains("priest")
                            })
                        });
                        let roll = tick_hash(state.tick, entity_id as u64 ^ 0xD15EA5E);
                        let infection_chance = if is_healer { 0.05 } else { 0.15 };
                        if roll < infection_chance {
                            out.push(WorldDelta::Damage {
                                target_id: entity_id,
                                amount: DISEASE_DAMAGE_PER_TICK,
                                source_id: 0, // environmental
                            });
                        }
                    }
                }
            }
        }
    }
}
