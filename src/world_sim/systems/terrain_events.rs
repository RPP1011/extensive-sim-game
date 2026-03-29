#![allow(unused)]
//! Terrain events system — natural disasters and geographic changes.
//!
//! Fires every 17 ticks. Rolls for terrain events (earthquakes, floods,
//! wildfires, etc.) and applies immediate effects as deltas. Season
//! influences which events can occur.
//!
//! Original: `crates/headless_campaign/src/systems/terrain_events.rs`
//!
//! NEEDS STATE: `active_terrain_events: Vec<TerrainEvent>` on WorldState
//! NEEDS STATE: `season` on WorldState (or derived from tick)
//! NEEDS STATE: `neighbors: Vec<u32>` on RegionState
//! NEEDS DELTA: SpawnTerrainEvent { ... }
//! NEEDS DELTA: ExpireTerrainEvent { event_id }
//! NEEDS DELTA: ModifyRegionThreat { region_id, delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

use super::seasons::{current_season, Season};

/// How often to roll for a terrain event (in ticks).
const TERRAIN_EVENT_INTERVAL: u64 = 17;

/// Base probability of a terrain event firing each roll.
const TERRAIN_EVENT_CHANCE: f32 = 0.03;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerrainEventType {
    Earthquake,
    Flood,
    Wildfire,
    Landslide,
    VolcanicEruption,
    Sinkhole,
}

pub fn compute_terrain_events(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TERRAIN_EVENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.regions.is_empty() {
        return;
    }

    // Roll for event.
    let roll = tick_hash(state.tick, 0xEA87_CAFE);
    if roll > TERRAIN_EVENT_CHANCE {
        return;
    }

    let season = current_season(state.tick);

    // Pick event type based on season weighting.
    let event_type = {
        let r = tick_hash(state.tick, 0x7E88_A19E);
        match season {
            Season::Spring => {
                if r < 0.3 {
                    TerrainEventType::Flood
                } else if r < 0.55 {
                    TerrainEventType::Landslide
                } else if r < 0.75 {
                    TerrainEventType::Earthquake
                } else {
                    TerrainEventType::Sinkhole
                }
            }
            Season::Summer => {
                if r < 0.35 {
                    TerrainEventType::Wildfire
                } else if r < 0.55 {
                    TerrainEventType::Earthquake
                } else if r < 0.7 {
                    TerrainEventType::VolcanicEruption
                } else {
                    TerrainEventType::Sinkhole
                }
            }
            Season::Autumn => {
                if r < 0.3 {
                    TerrainEventType::Wildfire
                } else if r < 0.55 {
                    TerrainEventType::Earthquake
                } else if r < 0.75 {
                    TerrainEventType::Landslide
                } else {
                    TerrainEventType::Sinkhole
                }
            }
            Season::Winter => {
                if r < 0.3 {
                    TerrainEventType::Flood
                } else if r < 0.55 {
                    TerrainEventType::Landslide
                } else if r < 0.75 {
                    TerrainEventType::Earthquake
                } else {
                    TerrainEventType::Sinkhole
                }
            }
        }
    };

    // Pick severity (0.3 - 1.0).
    let severity = 0.3 + tick_hash(state.tick, 0xBEEF_DEAD) * 0.7;

    // Pick affected region.
    let region_idx = (tick_hash(state.tick, 0xFEED_F00D) * state.regions.len() as f32) as usize;
    let region_idx = region_idx.min(state.regions.len().saturating_sub(1));
    let region = &state.regions[region_idx];

    // --- Apply effects via deltas ---
    match event_type {
        TerrainEventType::Earthquake | TerrainEventType::VolcanicEruption => {
            // Damage buildings in affected settlement(s).
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (region_idx as f32 * 20.0);
                let dy = settlement.pos.1 - (region_idx as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    // Drain treasury (building repair costs).
                    let cost = 10.0 * severity;
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -cost,
                    });
                    // Damage stockpiles.
                    for c in 0..crate::world_sim::NUM_COMMODITIES {
                        let loss = settlement.stockpile[c] * 0.1 * severity;
                        if loss > 0.001 {
                            out.push(WorldDelta::ConsumeCommodity {
                                location_id: settlement.id,
                                commodity: c,
                                amount: loss,
                            });
                        }
                    }
                }
            }
            // Damage entities near affected region.
            for entity in &state.entities {
                if !entity.alive {
                    continue;
                }
                let ex = entity.pos.0 - (region_idx as f32 * 20.0);
                let ey = entity.pos.1 - (region_idx as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: 5.0 * severity,
                        source_id: 0,
                    });
                }
            }
        }

        TerrainEventType::Flood => {
            // Damage stockpiles and treasury in nearby settlements.
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (region_idx as f32 * 20.0);
                let dy = settlement.pos.1 - (region_idx as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    for c in 0..crate::world_sim::NUM_COMMODITIES {
                        let loss = settlement.stockpile[c] * 0.15 * severity;
                        if loss > 0.001 {
                            out.push(WorldDelta::ConsumeCommodity {
                                location_id: settlement.id,
                                commodity: c,
                                amount: loss,
                            });
                        }
                    }
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -5.0 * severity,
                    });
                }
            }
        }

        TerrainEventType::Wildfire => {
            // Destroy resources (stockpile drain) but reduce monster threat.
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (region_idx as f32 * 20.0);
                let dy = settlement.pos.1 - (region_idx as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    for c in 0..crate::world_sim::NUM_COMMODITIES {
                        let loss = settlement.stockpile[c] * 0.25 * severity;
                        if loss > 0.001 {
                            out.push(WorldDelta::ConsumeCommodity {
                                location_id: settlement.id,
                                commodity: c,
                                amount: loss,
                            });
                        }
                    }
                }
            }
            // Kill monsters in the region (represented as Damage to hostile entities).
            for entity in &state.entities {
                if !entity.alive || entity.kind != EntityKind::Monster {
                    continue;
                }
                let ex = entity.pos.0 - (region_idx as f32 * 20.0);
                let ey = entity.pos.1 - (region_idx as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: 15.0 * severity,
                        source_id: 0,
                    });
                }
            }
        }

        TerrainEventType::Landslide => {
            // Damage entities and minor stockpile loss.
            for entity in &state.entities {
                if !entity.alive {
                    continue;
                }
                let ex = entity.pos.0 - (region_idx as f32 * 20.0);
                let ey = entity.pos.1 - (region_idx as f32 * 15.0);
                if ex * ex + ey * ey < 225.0 {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: 3.0 * severity,
                        source_id: 0,
                    });
                }
            }
        }

        TerrainEventType::Sinkhole => {
            // Minor damage, possible resource discovery (produce commodity).
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (region_idx as f32 * 20.0);
                let dy = settlement.pos.1 - (region_idx as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    // Discovery: produce a rare commodity.
                    let commodity =
                        (state.tick as usize + region_idx) % crate::world_sim::NUM_COMMODITIES;
                    out.push(WorldDelta::ProduceCommodity {
                        location_id: settlement.id,
                        commodity,
                        amount: 5.0 * severity,
                    });
                }
            }
        }
    }
}
