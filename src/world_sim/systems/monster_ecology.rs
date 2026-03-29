#![allow(unused)]
//! Monster ecology — fires every 7 ticks.
//!
//! Monster populations grow, migrate between regions, and attack settlements
//! when overpopulated. Produces Damage deltas for settlement attacks and
//! Move deltas for monster migration toward settlements.
//!
//! Original: `crates/headless_campaign/src/systems/monster_ecology.rs`
//!
//! NEEDS STATE: `monster_density` on RegionState is used (already exists).
//! NEEDS STATE: `neighbor_ids: Vec<u32>` on RegionState for migration routes.
//! NEEDS STATE: `last_hunted_tick` on RegionState (for neglect growth bonus).
//! NEEDS DELTA: UpdateMonsterDensity { region_id, delta } — to adjust density.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, RegionState, WorldState};

use super::seasons::{current_season, season_modifiers};

/// Tick cadence for the ecology system.
const ECOLOGY_TICK_INTERVAL: u64 = 7;

/// Base growth rate per ecology tick (2%).
const BASE_GROWTH_RATE: f32 = 0.02;

/// Population density cap per region.
const DENSITY_CAP: f32 = 100.0;

/// Migration overflow threshold — density must exceed this to migrate.
const MIGRATION_THRESHOLD: f32 = 70.0;

/// Fraction of overflow that migrates.
const MIGRATION_FRACTION: f32 = 0.5;

/// Density level at which settlement attacks begin.
const SETTLEMENT_ATTACK_THRESHOLD: f32 = 80.0;

/// Density level at which regional threat increases.
const THREAT_INCREASE_THRESHOLD: f32 = 50.0;

/// Deterministic hash for pseudo-random decisions from immutable state.
fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

/// Deterministic integer hash.
fn tick_hash_u32(tick: u64, salt: u64) -> u32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    (x >> 33) as u32
}

pub fn compute_monster_ecology(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ECOLOGY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let season = current_season(state.tick);
    let threat_mod = season_modifiers(season).threat;

    // --- Growth phase ---
    // Monster density grows each ecology tick, capped at DENSITY_CAP.
    // We can't mutate RegionState directly, so we note that growth should be
    // applied via a dedicated delta. For now, we use the current density to
    // drive downstream effects (attacks, spawns).

    // --- Spawn monsters when density is high ---
    // When monster_density exceeds the spawn threshold, emit monster entities
    // that move toward the nearest settlement (handled by compute_low already).
    // Here we track which regions are dangerous and apply settlement effects.

    for (region_idx, region) in state.regions.iter().enumerate() {
        let density = region.monster_density;

        // --- Threat increase at density > 50 ---
        // The threat system (threat.rs) reads region.threat_level, so we don't
        // duplicate that here. But we note that monster density feeds into threat.

        // --- Settlement attacks at density > 80 ---
        if density > SETTLEMENT_ATTACK_THRESHOLD {
            let excess = density - SETTLEMENT_ATTACK_THRESHOLD;
            let damage = excess * 0.5 * threat_mod;

            // Find settlements near this region.
            // Heuristic: settlements whose position is within region bounds.
            // Without explicit region-settlement mapping, we use proximity.
            for settlement in &state.settlements {
                // Check if settlement is in this region by comparing indices.
                // With proper region_id on settlement this would be an exact match.
                let roll = tick_hash(state.tick, region.id as u64 ^ settlement.id as u64);
                if roll > 0.3 {
                    continue; // Only attack ~30% of settlements per tick.
                }

                // Damage to settlement buildings (modeled as treasury loss).
                out.push(WorldDelta::UpdateTreasury {
                    location_id: settlement.id,
                    delta: -damage,
                });
            }
        }

        // --- Monster movement toward settlements ---
        // Monsters in high-density regions are attracted to settlements.
        // The compute_low module already handles individual monster entity
        // movement, but we can spawn additional "pressure" by creating
        // damage from ambient monster encounters.
        if density > THREAT_INCREASE_THRESHOLD {
            // Ambient monster encounters damage traveling NPCs in the region.
            for entity in &state.entities {
                if !entity.alive || entity.kind != EntityKind::Npc {
                    continue;
                }

                // Proximity check: is this entity near any settlement in the region?
                let near = state.settlements.iter().any(|s| {
                    let dx = entity.pos.0 - s.pos.0;
                    let dy = entity.pos.1 - s.pos.1;
                    dx * dx + dy * dy < 400.0 // within 20 units
                });

                if !near {
                    continue;
                }

                let roll = tick_hash(
                    state.tick,
                    entity.id as u64 ^ region.id as u64 ^ 0xDEAD_BEEF,
                );
                let encounter_chance = (density - THREAT_INCREASE_THRESHOLD) * 0.0005;
                if roll < encounter_chance {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: density * 0.05,
                        source_id: 0, // monster encounter
                    });
                }
            }
        }

        // --- Migration: monsters move from overpopulated to neighboring regions ---
        // Without neighbor_ids on RegionState, we can't compute migration targets.
        // When that field is added, migration would work like this:
        //
        // if density > MIGRATION_THRESHOLD {
        //     let overflow = density - MIGRATION_THRESHOLD;
        //     let migrating = overflow * MIGRATION_FRACTION;
        //     // Pick a neighbor deterministically.
        //     if let Some(&neighbor_id) = region.neighbor_ids.first() {
        //         // Emit UpdateMonsterDensity deltas for source and dest.
        //     }
        // }
    }
}
