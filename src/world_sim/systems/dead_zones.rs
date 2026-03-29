#![allow(unused)]
//! Dead zone system — every 17 ticks.
//!
//! Over-exploited regions accumulate extraction pressure from questing and
//! harvesting. When pressure exceeds a threshold, dead zones form and spread
//! corruption to neighbors, increasing monster threat and damaging entities.
//!
//! Original: `crates/headless_campaign/src/systems/dead_zones.rs`
//!
//! NEEDS STATE: `extraction_pressure: Vec<f32>` on WorldState (per region)
//! NEEDS STATE: `dead_zone_level: Vec<f32>` on WorldState (per region)
//! NEEDS STATE: `neighbors: Vec<Vec<u32>>` on RegionState
//! NEEDS DELTA: ModifyRegionThreat { region_id, delta }
//! NEEDS DELTA: ModifyRegionMorale { region_id, delta }

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

/// Cadence: runs every 17 ticks.
const DEAD_ZONE_INTERVAL: u64 = 17;

/// Pressure threshold above which dead zone level starts rising.
const PRESSURE_THRESHOLD: f32 = 0.5;
/// Rate at which dead zone level rises when pressure exceeds threshold.
const DEAD_ZONE_RISE_RATE: f32 = 0.01;
/// Dead zone level above which corruption spreads to neighbors.
const SPREAD_THRESHOLD: f32 = 0.6;
/// Rate at which dead zones spread to adjacent regions.
const SPREAD_RATE: f32 = 0.005;
/// Pressure threshold below which dead zones start recovering.
const RECOVERY_PRESSURE_THRESHOLD: f32 = 0.2;
/// Natural decay of extraction pressure per tick.
const PRESSURE_DECAY: f32 = 0.01;

/// Damage per tick to NPCs in high dead-zone regions (dz_level * this factor).
const NPC_DEAD_ZONE_DAMAGE: f32 = 2.0;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_dead_zones(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DEAD_ZONE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.regions.is_empty() {
        return;
    }

    // Without extraction_pressure/dead_zone_level on WorldState, we approximate
    // dead-zone behaviour using the existing threat_level as a proxy.
    //
    // Regions with very high threat_level (> 70) are treated as proto-dead-zones.
    // Entities inside them take environmental damage, and we damage nearby
    // settlements' stockpiles to represent reduced yields.

    for (ri, region) in state.regions.iter().enumerate() {
        if region.threat_level <= 70.0 {
            continue;
        }

        let severity = (region.threat_level - 70.0) / 30.0; // 0..1

        // --- Damage NPCs near settlements in high-threat regions ---
        for entity in &state.entities {
            if !entity.alive || entity.kind != crate::world_sim::state::EntityKind::Npc {
                continue;
            }
            // Proximity heuristic: use settlements as region anchors.
            let near = state.settlements.iter().any(|s| {
                let dx = entity.pos.0 - s.pos.0;
                let dy = entity.pos.1 - s.pos.1;
                dx * dx + dy * dy < 225.0 // within 15 units
            });
            if !near {
                continue;
            }
            let roll = tick_hash(state.tick, entity.id as u64 ^ 0xDEAD_20E5);
            if roll < severity * 0.3 {
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: NPC_DEAD_ZONE_DAMAGE * severity,
                    source_id: 0, // environmental
                });
            }
        }

        // --- Reduce stockpiles in affected settlements (resource yield penalty) ---
        for settlement in &state.settlements {
            let dx = settlement.pos.0 - (ri as f32 * 20.0);
            let dy = settlement.pos.1 - (ri as f32 * 15.0);
            if dx * dx + dy * dy > 400.0 {
                continue;
            }
            // Drain all commodities proportional to severity (-50% at max).
            for c in 0..crate::world_sim::NUM_COMMODITIES {
                let drain = settlement.stockpile[c] * 0.02 * severity;
                if drain > 0.001 {
                    out.push(WorldDelta::ConsumeCommodity {
                        location_id: settlement.id,
                        commodity: c,
                        amount: drain,
                    });
                }
            }
        }
    }
}
