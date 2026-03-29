#![allow(unused)]
//! Dynamic world geography system — fires every 33 ticks.
//!
//! The world map slowly changes over time: forests grow, deserts expand,
//! rivers flood, settlements grow, roads degrade, and wilderness reclaims
//! abandoned regions. Changes affect settlement stockpiles, treasury, and
//! entity health.
//!
//! Original: `crates/headless_campaign/src/systems/geography.rs`
//!
//! NEEDS STATE: `geography_changes: Vec<GeographyChange>` on WorldState
//! NEEDS STATE: `season` on WorldState
//! NEEDS DELTA: StartGeographyChange { change_type, region_id }
//! NEEDS DELTA: CompleteGeographyChange { change_id }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

use super::seasons::{current_season, Season};

/// How often to evaluate geography changes (in ticks).
const GEOGRAPHY_INTERVAL: u64 = 33;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GeoChangeType {
    ForestGrowth,
    DesertExpansion,
    RiverFlood,
    SettlementGrowth,
    RoadDegradation,
    WildernessReclaim,
}

pub fn compute_geography(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GEOGRAPHY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.regions.is_empty() {
        return;
    }

    let season = current_season(state.tick);

    // Without geography_changes on WorldState, we apply one-shot effects
    // each tick that approximate the original system's gradual changes.
    // Each region is evaluated for conditions that would trigger a geography
    // change, and appropriate deltas are emitted.

    for (ri, region) in state.regions.iter().enumerate() {
        let roll = tick_hash(state.tick, 0x6E06_8A9Eu64.wrapping_add(ri as u64));

        // ForestGrowth: low threat, low population settlement nearby.
        // Effect: small commodity production boost (wood/herbs).
        if region.threat_level < 30.0 && roll < 0.05 {
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 && settlement.population < 300 {
                    // Produce wood (commodity 2) and herbs (commodity 3).
                    out.push(WorldDelta::ProduceCommodity {
                        location_id: settlement.id,
                        commodity: crate::world_sim::commodity::WOOD,
                        amount: 1.0,
                    });
                    out.push(WorldDelta::ProduceCommodity {
                        location_id: settlement.id,
                        commodity: crate::world_sim::commodity::HERBS,
                        amount: 0.5,
                    });
                }
            }
            continue;
        }

        // DesertExpansion: summer/autumn + high threat region.
        // Effect: drain food stockpile, treasury loss.
        if (season == Season::Summer || season == Season::Autumn)
            && region.threat_level > 50.0
            && roll < 0.04
        {
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 {
                    // Drain food (commodity 0).
                    let loss = settlement.stockpile[0] * 0.05;
                    if loss > 0.001 {
                        out.push(WorldDelta::ConsumeCommodity {
                            location_id: settlement.id,
                            commodity: crate::world_sim::commodity::FOOD,
                            amount: loss,
                        });
                    }
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -2.0,
                    });
                }
            }
            continue;
        }

        // RiverFlood: spring + high population.
        // Effect: rich soil (produce food) but infrastructure damage (treasury drain).
        if season == Season::Spring && roll < 0.04 {
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 && settlement.population > 400 {
                    out.push(WorldDelta::ProduceCommodity {
                        location_id: settlement.id,
                        commodity: crate::world_sim::commodity::FOOD, // food
                        amount: 2.0,
                    });
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: -5.0,
                    });
                }
            }
            continue;
        }

        // SettlementGrowth: high population + low threat.
        // Effect: treasury boost.
        if region.threat_level < 20.0 && roll < 0.03 {
            for settlement in &state.settlements {
                let dx = settlement.pos.0 - (ri as f32 * 20.0);
                let dy = settlement.pos.1 - (ri as f32 * 15.0);
                if dx * dx + dy * dy < 400.0 && settlement.population > 500 {
                    out.push(WorldDelta::UpdateTreasury {
                        location_id: settlement.id,
                        delta: 10.0,
                    });
                }
            }
            continue;
        }

        // RoadDegradation: high threat region.
        // Effect: slow entities (counter-force) — approximate as minor damage.
        if region.threat_level > 60.0 && roll < 0.04 {
            for entity in &state.entities {
                if !entity.alive || entity.kind != crate::world_sim::state::EntityKind::Npc {
                    continue;
                }
                let ex = entity.pos.0 - (ri as f32 * 20.0);
                let ey = entity.pos.1 - (ri as f32 * 15.0);
                if ex * ex + ey * ey < 400.0 {
                    // Minor fatigue-like damage from rough travel.
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: 1.0,
                        source_id: 0,
                    });
                }
            }
        }
    }
}
