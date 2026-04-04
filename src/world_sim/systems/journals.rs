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
use crate::world_sim::state::{Entity, WorldState};
use crate::world_sim::state::entity_hash_f32;

//   JournalEntry { tick, entry_type: JournalType, text, sentiment }
//   JournalType: BattleMemory, Triumph, QuestReflection, Regret, GriefEntry,
//                BondMoment, Ambition, Discovery

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

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_journals_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_journals_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % JOURNAL_TICK_INTERVAL != 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        // --- Battle memories ---
        if let Some(grid_id) = entity.grid_id {
            if let Some(grid) = state.fidelity_zone(grid_id) {
                if grid.fidelity == crate::world_sim::fidelity::Fidelity::High {
                }
            }
        }

        // --- Grief entries for dead allies ---

        // --- Bond moments ---

        // --- Ambition entries ---

        // --- Discovery entries (5% random chance) ---
        let roll = entity_hash_f32(entity.id, state.tick, 0);
        if roll < 0.05 {
        }

        // --- Personality drift from journal sentiment ---
    }
}

