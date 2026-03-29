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
use crate::world_sim::state::WorldState;

// NEEDS STATE: mood_state: MoodState on Entity or NpcData (mood, started_at, expires_at)
// NEEDS STATE: morale: f32 on Entity or NpcData
// NEEDS STATE: stress: f32 on Entity or NpcData
// NEEDS STATE: party_id: Option<u32> on Entity or NpcData
// NEEDS DELTA: SetMood { entity_id: u32, mood: Mood, started_at: u64, expires_at: u64 }
// NEEDS DELTA: AdjustMorale { entity_id: u32, delta: f32 }

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

    let tick = state.tick;

    // --- Phase 1: Natural decay — moods expire after their duration ---
    // NEEDS STATE: for each alive NPC entity with a non-Neutral mood:
    //   if tick >= entity.mood_state.expires_at:
    //     out.push(WorldDelta::SetMood { entity_id, mood: Mood::Neutral, ... });

    // --- Phase 2: Ongoing mood effects ---
    // NEEDS STATE + DELTA: for each alive NPC entity:
    //   match entity.mood_state.mood:
    //     Mood::Grieving if elapsed <= GRIEF_DURATION =>
    //       out.push(WorldDelta::AdjustMorale { entity_id, delta: -2.0 });
    //     Mood::Excited =>
    //       out.push(WorldDelta::AdjustMorale { entity_id, delta: 5.0 });
    //     Mood::Melancholic =>
    //       out.push(WorldDelta::AdjustMorale { entity_id, delta: -1.0 });

    // --- Phase 3: Contagion — moods spread within shared grids ---
    // For each grid, check if any NPC has Excited (20% spread) or Fearful (15% spread).
    // Neutral NPCs on the same grid may catch the mood.
    // NEEDS STATE: mood_state on NpcData, rng on WorldState (read-only — use tick+id as seed)
    // Note: contagion probability must use deterministic hashing (not mutable rng)
    // since compute phase is read-only. Use (tick ^ entity_id) based deterministic roll.

    for grid in &state.grids {
        let npc_entities: Vec<(u32, bool)> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| {
                let e = state.entity(eid)?;
                if e.alive && e.npc.is_some() {
                    Some((eid, true)) // second field would be mood == Neutral check
                } else {
                    None
                }
            })
            .collect();

        // NEEDS STATE: check if any NPC on grid has Excited/Fearful mood
        // NEEDS DELTA: SetMood for contagion targets
        // Contagion uses deterministic roll: hash(tick, entity_id) < threshold
    }
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
            // Deterministic 10% chance using hash of tick + entity_id.
            let hash =
                (tick.wrapping_mul(2654435761) ^ (entity_id as u64).wrapping_mul(40503)) & 0xFFFF;
            (hash as f32 / 65536.0) < 0.10
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
