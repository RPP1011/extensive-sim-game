#![allow(unused)]
//! Adventurer mood/emotion system — delta architecture port.
//!
//! Moods affect combat effectiveness, morale drift, bond formation, and
//! quest preferences. Moods decay naturally, apply ongoing effects, and
//! spread via contagion within shared grids.
//!
//! Original: `crates/headless_campaign/src/systems/moods.rs`
//! Cadence: every 7 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityField, EntityKind, WorldState};
use crate::world_sim::state::entity_hash_f32;


/// Mood tick cadence (every 7 ticks).
const MOOD_TICK_INTERVAL: u64 = 7;

/// Duration range for mood decay (ticks).
const MOOD_MIN_DURATION: u64 = 17;
const MOOD_MAX_DURATION: u64 = 33;

/// Grieving morale drain duration (ticks).
const GRIEF_DURATION: u64 = 17;

/// Mood kinds — mirrors the original enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    Neutral,
    Excited,
    Inspired,
    Angry,
    Fearful,
    Grieving,
    Melancholic,
    Determined,
}

/// What caused a mood transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoodCause {
    NaturalDecay,
    Contagion,
    BattleVictory,
    BattleDefeat,
    AllyDeath,
    LevelUp,
    LongIdle,
    RivalEncounter,
    QuestSuccess,
    LowMoraleHighStress,
}

/// Snapshot of an entity's current mood.
#[derive(Debug, Clone, Copy)]
pub struct MoodSnapshot {
    pub mood: Mood,
    pub started_at: u64,
    pub expires_at: u64,
}

impl Default for MoodSnapshot {
    fn default() -> Self {
        Self {
            mood: Mood::Neutral,
            started_at: 0,
            expires_at: 0,
        }
    }
}

/// Compute mood deltas: natural decay, ongoing effects, and contagion.
///
/// Since WorldState lacks mood storage, this is a structural placeholder.
/// Once `mood_state` and `morale` fields are added to NpcData and the
/// appropriate delta variants exist, the body will emit real deltas.
pub fn compute_moods(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MOOD_TICK_INTERVAL != 0 {
        return;
    }

    // --- Phase 1: Natural decay — moods expire after their duration ---
    //   if tick >= entity.mood_state.expires_at:
    //     out.push(WorldDelta::SetMood { entity_id, mood: Mood::Neutral, ... });

    // --- Phase 2: Ongoing mood effects ---
    // Until mood_state exists on NpcData, derive mood proxy from entity state
    // and emit morale deltas accordingly.
    for entity in &state.entities {
        if !entity.alive { continue; }
        let hp_ratio = if entity.max_hp > 0.0 { entity.hp / entity.max_hp } else { 1.0 };

        // Proxy: entity on a high-fidelity grid with low HP → grieving/fearful
        let on_combat_grid = entity.grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
            .unwrap_or(false);

        if on_combat_grid && hp_ratio < 0.3 {
            // Fearful/grieving: morale drain
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: -2.0,
            });
        } else if on_combat_grid && hp_ratio > 0.8 {
            // Determined/excited in combat with good health: morale boost
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: 1.0,
            });
        } else if !on_combat_grid && hp_ratio > 0.9 {
            // Idle and healthy: small contentment boost
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: 0.5,
            });
        }
    }

    // --- Phase 3: Contagion — moods spread within shared grids ---
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_moods_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_moods_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % MOOD_TICK_INTERVAL != 0 {
        return;
    }

    // Contagion — moods spread within shared grids.
    // For each grid, check if any NPC has Excited (20% spread) or Fearful (15% spread).
    // Neutral NPCs on the same grid may catch the mood.
    // Note: contagion probability must use deterministic hashing (not mutable rng)
    // since compute phase is read-only. Use (tick ^ entity_id) based deterministic roll.

    let npc_entities: Vec<(u32, bool)> = entities
        .iter()
        .filter(|e| e.alive && e.npc.is_some())
        .map(|e| (e.id, true)) // second field would be mood == Neutral check
        .collect();

    // Contagion uses deterministic roll: hash(tick, entity_id) < threshold
}

// ---------------------------------------------------------------------------
// Mood query helpers (pure functions, no state mutation)
// ---------------------------------------------------------------------------

/// Combat power multiplier from mood.
pub fn mood_combat_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Inspired => 1.15,
        Mood::Angry => 1.10,
        Mood::Fearful => 0.90,
        _ => 1.0,
    }
}

/// Defense multiplier from mood.
pub fn mood_defense_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Angry => 0.90,
        _ => 1.0,
    }
}

/// XP gain multiplier from mood.
pub fn mood_xp_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Inspired => 1.10,
        _ => 1.0,
    }
}

/// Loyalty modifier from mood (additive per tick).
pub fn mood_loyalty_modifier(mood: Mood) -> f32 {
    match mood {
        Mood::Determined => 0.10,
        _ => 0.0,
    }
}

/// Rivalry formation chance bonus from mood.
pub fn mood_rivalry_bonus(mood: Mood) -> f32 {
    match mood {
        Mood::Angry => 0.20,
        _ => 0.0,
    }
}

/// Bond growth bonus from mood (shared grief).
pub fn mood_bond_growth_bonus(mood: Mood) -> f32 {
    match mood {
        Mood::Grieving => 0.50,
        _ => 0.0,
    }
}

/// Whether mood makes the entity reckless.
/// Uses deterministic hash instead of mutable rng.
pub fn is_reckless(mood: Mood, entity_id: u32, tick: u64) -> bool {
    match mood {
        Mood::Excited => {
            // Deterministic 10% chance using entity_hash.
            entity_hash_f32(entity_id, tick, 0xE3C1) < 0.10
        }
        _ => false,
    }
}

/// Whether this entity refuses high-threat quests due to fear.
pub fn refuses_high_threat(mood: Mood) -> bool {
    matches!(mood, Mood::Fearful)
}

/// Quest completion speed multiplier from mood.
pub fn mood_quest_speed_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Determined => 1.20,
        _ => 1.0,
    }
}
