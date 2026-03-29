#![allow(unused)]
//! Adventurer journal system — delta architecture port.
//!
//! Each NPC keeps a personal journal that accumulates entries from their
//! experiences, affecting personality drift (morale, loyalty) over time.
//! Journal entries are auto-generated from context and pruned to a max
//! of 20 entries per entity.
//!
//! Original: `crates/headless_campaign/src/systems/journals.rs`
//! Cadence: every 17 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: journal: Vec<JournalEntry> on Entity/NpcData
//   JournalEntry { tick, entry_type: JournalType, text, sentiment }
//   JournalType: BattleMemory, Triumph, QuestReflection, Regret, GriefEntry,
//                BondMoment, Ambition, Discovery
// NEEDS STATE: adventurer_bonds (for grief entries)
// NEEDS STATE: adventurer morale, stress, loyalty
// NEEDS DELTA: WriteJournalEntry { entity_id, entry_type, text, sentiment }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: AdjustLoyalty { entity_id, delta }
// NEEDS DELTA: AdjustStress { entity_id, delta }

/// Cadence gate.
const JOURNAL_TICK_INTERVAL: u64 = 17;

/// Maximum journal entries per NPC.
const MAX_JOURNAL_ENTRIES: usize = 20;

/// Number of recent entries for sentiment averaging.
const SENTIMENT_WINDOW: usize = 10;

/// Sentiment thresholds.
const POSITIVE_THRESHOLD: f32 = 0.3;
const NEGATIVE_THRESHOLD: f32 = -0.3;

/// Compute journal deltas: generate entries, apply personality drift.
///
/// Since WorldState lacks journal storage on entities, this is a structural
/// placeholder. Personality drift from journal sentiment can be expressed
/// as morale/stress adjustments once journal entries are tracked.
pub fn compute_journals(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % JOURNAL_TICK_INTERVAL != 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        // --- Battle memories ---
        // If entity is on a grid with hostiles (recently fought):
        //   Victory: sentiment +0.5 to +0.8
        //   Defeat (low hp): sentiment -0.6 to -0.9
        //   out.push(WorldDelta::WriteJournalEntry { entity_id, BattleMemory, text, sentiment })

        // Proxy: check if entity is on a combat grid
        if let Some(grid_id) = entity.grid_id {
            if let Some(grid) = state.grid(grid_id) {
                if grid.fidelity == crate::world_sim::fidelity::Fidelity::High {
                    // Entity is in combat — would generate battle memory entry
                    // NEEDS DELTA: WriteJournalEntry
                }
            }
        }

        // --- Grief entries for dead allies ---
        // NEEDS STATE: adventurer_bonds — for each dead entity with bond > 30:
        //   out.push(WorldDelta::WriteJournalEntry { entity_id, GriefEntry, ..., sentiment: -0.7 to -0.9 })

        // --- Bond moments (bond > 60 with co-located NPC) ---
        // out.push(WorldDelta::WriteJournalEntry { entity_id, BondMoment, ..., sentiment: +0.5 to +0.7 })

        // --- Ambition entries (level milestones) ---
        // out.push(WorldDelta::WriteJournalEntry { entity_id, Ambition, ..., sentiment: +0.4 to +0.7 })

        // --- Discovery entries (5% random chance) ---
        let roll = deterministic_roll(state.tick, entity.id);
        if roll < 0.05 {
            // out.push(WorldDelta::WriteJournalEntry { entity_id, Discovery, ..., sentiment: +0.3 to +0.5 })
        }

        // --- Personality drift from journal sentiment ---
        // NEEDS STATE: read journal entries, compute average sentiment
        // avg > POSITIVE_THRESHOLD: +1 morale, +0.5 loyalty
        // avg < NEGATIVE_THRESHOLD: -1 morale, +0.5 stress
    }
}

fn deterministic_roll(tick: u64, id: u32) -> f32 {
    let h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(id as u64);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xff51afd7ed558ccd);
    let h = h ^ (h >> 33);
    (h & 0xFFFF) as f32 / 65536.0
}
