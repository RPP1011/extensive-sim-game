#![allow(unused)]
//! Espionage system — every 7 ticks.
//!
//! Manages spies planted by the guild in factions. Spies gather intel,
//! risk discovery, and can perform sabotage actions.
//!
//! Ported from `crates/headless_campaign/src/systems/espionage.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: spies: Vec<SpyState> on WorldState
//   SpyState { id: u32, adventurer_id: u32, target_faction_id: u32,
//              cover: f32, intel_gathered: f32 }
// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState.diplomatic_stance: DiplomaticStance
// NEEDS STATE: entities or adventurers with level, archetype, stealth stat

// NEEDS DELTA: AdjustSpyCover { spy_id: u32, delta: f32 }
// NEEDS DELTA: GatherIntel { spy_id: u32, target_faction_id: u32, amount: f32 }
// NEEDS DELTA: SpyCaught { spy_id: u32, adventurer_id: u32, faction_id: u32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: KillEntity { entity_id: u32 }

/// Cadence: runs every 7 ticks.
const ESPIONAGE_INTERVAL: u64 = 7;

/// Base intel gathered per espionage tick.
const BASE_INTEL_PER_TICK: f32 = 5.0;

/// Base cover degradation per espionage tick.
const BASE_COVER_DECAY: f32 = 2.0;

/// Extra cover decay when the target faction is hostile.
const HOSTILE_COVER_DECAY: f32 = 5.0;

/// Cover threshold below which discovery checks occur.
const DISCOVERY_THRESHOLD: f32 = 20.0;

/// Chance (0-1) of being caught per tick when cover < threshold.
const DISCOVERY_CHANCE: f32 = 0.1;

/// Faction relation penalty when a spy is caught.
const CAUGHT_RELATION_PENALTY: f32 = 20.0;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, spy_id: u32, salt: u32) -> f32 {
    let mut h = tick.wrapping_mul(6364136223846793005)
        .wrapping_add(spy_id as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(salt as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_espionage(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ESPIONAGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.spies exists, iterate each spy and compute intel/cover/discovery deltas.

    /*
    if state.spies.is_empty() {
        return;
    }

    for spy in &state.spies {
        // --- Look up adventurer stats ---
        let (adv_level, is_rogue, stealth_bonus) = state.entities.iter()
            .filter_map(|e| e.npc.as_ref().map(|n| (e, n)))
            .find(|(_, n)| n.adventurer_id == spy.adventurer_id)
            .map(|(e, n)| {
                let is_rogue = n.class_tags.iter()
                    .any(|t| t.to_lowercase().contains("rogue"));
                // Stealth bonus from class tags (simplified)
                let stealth = if is_rogue { 2.0 } else { 0.0 };
                (e.level, is_rogue, stealth)
            })
            .unwrap_or((1, false, 0.0));

        // --- Intel gathering ---
        let level_modifier = 1.0 + (adv_level as f32 - 1.0) * 0.1;
        let rogue_modifier = if is_rogue { 1.5 } else { 1.0 };
        let intel_delta = BASE_INTEL_PER_TICK * level_modifier * rogue_modifier;

        // --- Cover degradation ---
        let faction_hostile = state.factions.iter()
            .find(|f| f.id == spy.target_faction_id)
            .map(|f| matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            ))
            .unwrap_or(false);

        let cover_decay = if faction_hostile {
            HOSTILE_COVER_DECAY
        } else {
            BASE_COVER_DECAY
        };

        let new_cover = (spy.cover - cover_decay).max(0.0);

        // --- Discovery check ---
        let caught = if new_cover < DISCOVERY_THRESHOLD {
            let roll = deterministic_roll(state.tick, spy.id, 0);
            let adjusted_chance = (DISCOVERY_CHANCE - stealth_bonus * 0.02).max(0.01);
            roll < adjusted_chance
        } else {
            false
        };

        if caught {
            // Spy is caught: remove spy, mark adventurer dead, damage relations
            out.push(WorldDelta::SpyCaught {
                spy_id: spy.id,
                adventurer_id: spy.adventurer_id,
                faction_id: spy.target_faction_id,
            });
            out.push(WorldDelta::Die { entity_id: spy.adventurer_id });
            out.push(WorldDelta::AdjustRelationship {
                faction_id: spy.target_faction_id,
                delta: -CAUGHT_RELATION_PENALTY,
            });
        } else {
            // Normal tick: degrade cover and gather intel
            out.push(WorldDelta::AdjustSpyCover {
                spy_id: spy.id,
                delta: -cover_decay,
            });
            out.push(WorldDelta::GatherIntel {
                spy_id: spy.id,
                target_faction_id: spy.target_faction_id,
                amount: intel_delta,
            });
        }
    }
    */
}
