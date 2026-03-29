#![allow(unused)]
//! Chronicle / narrative log system — delta architecture port.
//!
//! Records significant campaign events as narrative entries that feed back
//! into quest generation (revenge quests, memorial quests, faction-themed
//! quests, personal adventurer legacies).
//!
//! Original: `crates/headless_campaign/src/systems/chronicle.rs`
//! Cadence: every 3 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: chronicle: Vec<ChronicleEntry> on WorldState
//   where ChronicleEntry { tick: u64, entry_type: ChronicleType, text: String,
//                          participants: Vec<u32>, location_id: Option<u32>,
//                          faction_id: Option<u32>, significance: f32 }
// NEEDS STATE: pending_events: Vec<WorldEvent> on WorldState (events from current step)
// NEEDS DELTA: RecordChronicle { entry: ChronicleEntry }
// NEEDS DELTA: PruneChronicle { keep_count: usize }

/// Maximum chronicle entries before pruning.
const MAX_CHRONICLE_ENTRIES: usize = 100;

/// Minimum significance to record an entry.
const MIN_SIGNIFICANCE: f32 = 3.0;

/// Cadence: chronicle scans every 3 ticks.
const CHRONICLE_INTERVAL: u64 = 3;

/// Chronicle entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChronicleType {
    Death,
    QuestCompletion,
    BattleRecord,
    DiplomaticEvent,
    CrisisEvent,
    Construction,
    HeroicDeed,
    Tragedy,
    Discovery,
}

/// A chronicle entry — a narrative record of a significant event.
#[derive(Debug, Clone)]
pub struct ChronicleEntry {
    pub tick: u64,
    pub entry_type: ChronicleType,
    pub text: String,
    pub participants: Vec<u32>,
    pub location_id: Option<u32>,
    pub faction_id: Option<u32>,
    pub significance: f32,
}

/// Compute chronicle deltas by scanning world state for recordable conditions.
///
/// The original system scanned WorldEvents from the current step. In the
/// delta architecture, the compute phase reads only the frozen state snapshot.
/// Instead of event-scanning, we detect recordable conditions directly from
/// state (e.g., entities that just died, regions that changed faction).
///
/// Since WorldState lacks chronicle storage and event history, this is a
/// structural placeholder.
pub fn compute_chronicle(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CHRONICLE_INTERVAL != 0 {
        return;
    }

    // --- Detect deaths (entities that just died this tick window) ---
    // NEEDS STATE: death_tick or last_alive_tick on Entity to detect recent deaths
    // For each entity where entity.alive == false && entity.death_tick >= state.tick - CHRONICLE_INTERVAL:
    //   let significance = 8.0;
    //   let entry = ChronicleEntry {
    //     tick: state.tick,
    //     entry_type: ChronicleType::Death,
    //     text: format!("{} fell in battle", entity_name(state, entity.id)),
    //     participants: vec![entity.id],
    //     location_id: region_for_entity(state, entity),
    //     faction_id: None,
    //     significance,
    //   };
    //   out.push(WorldDelta::RecordChronicle { entry });

    // --- Detect region ownership changes ---
    // NEEDS STATE: previous_faction_id on RegionState to detect transitions
    // For each region where region.faction_id != region.previous_faction_id:
    //   significance = 6.0
    //   Record a DiplomaticEvent entry

    // --- Detect significant level-ups (level % 5 == 0, level >= 5) ---
    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        // Only record milestone levels.
        if entity.level >= 5 && entity.level % 5 == 0 {
            // NEEDS STATE: a way to know the level-up just happened (e.g. previous_level)
            // let entry = ChronicleEntry {
            //   tick: state.tick,
            //   entry_type: ChronicleType::HeroicDeed,
            //   text: format!("{} achieved level {}", entity_name(state, entity.id), entity.level),
            //   participants: vec![entity.id],
            //   location_id: None,
            //   faction_id: None,
            //   significance: 4.0,
            // };
            // out.push(WorldDelta::RecordChronicle { entry });
        }
    }

    // --- Prune if over capacity ---
    // NEEDS STATE: state.chronicle.len() > MAX_CHRONICLE_ENTRIES
    // out.push(WorldDelta::PruneChronicle { keep_count: MAX_CHRONICLE_ENTRIES });
}

// ---------------------------------------------------------------------------
// Chronicle query helpers (pure functions for quest generation)
// ---------------------------------------------------------------------------

/// Filter chronicle for recent tragedies (deaths, defeats).
pub fn recent_tragedies(chronicle: &[ChronicleEntry]) -> Vec<&ChronicleEntry> {
    chronicle
        .iter()
        .filter(|e| matches!(e.entry_type, ChronicleType::Death | ChronicleType::Tragedy))
        .collect()
}

/// Filter chronicle for entries involving a specific faction.
pub fn faction_history(chronicle: &[ChronicleEntry], faction_id: u32) -> Vec<&ChronicleEntry> {
    chronicle
        .iter()
        .filter(|e| e.faction_id == Some(faction_id))
        .collect()
}

/// Filter chronicle for entries involving a specific entity (adventurer legacy).
pub fn entity_legacy(chronicle: &[ChronicleEntry], entity_id: u32) -> Vec<&ChronicleEntry> {
    chronicle
        .iter()
        .filter(|e| e.participants.contains(&entity_id))
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Get a display name for an entity.
fn entity_name(state: &WorldState, id: u32) -> String {
    // NPC entities use adventurer_id as a proxy; real names would come from
    // a name table on WorldState.
    state
        .entity(id)
        .and_then(|e| e.npc.as_ref())
        .map(|n| format!("Adventurer #{}", n.adventurer_id))
        .unwrap_or_else(|| format!("Entity #{}", id))
}

/// Find the region containing an entity (by grid membership).
fn region_for_entity(state: &WorldState, entity: &crate::world_sim::state::Entity) -> Option<u32> {
    // Map grid → settlement → region. This is approximate; a proper
    // spatial lookup would be better.
    let grid_id = entity.grid_id?;
    let settlement = state
        .settlements
        .iter()
        .find(|s| s.grid_id == Some(grid_id))?;
    // Settlement doesn't directly have region_id; use nearest region.
    state
        .regions
        .iter()
        .min_by(|a, b| {
            let da = dist_sq(settlement.pos, (0.0, 0.0)); // placeholder
            let db = dist_sq(settlement.pos, (0.0, 0.0));
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.id)
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}
