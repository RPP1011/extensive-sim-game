#![allow(unused)]
//! Rumor spreading system — delta architecture port.
//!
//! Returning NPC entities bring information fragments (rumors) that
//! reveal hidden opportunities and predict threats. Rumors expire if
//! not acted on.
//!
//! Original: `crates/headless_campaign/src/systems/rumors.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: rumors: Vec<Rumor> on WorldState
//   where Rumor { id: u32, text: String, rumor_type: RumorType, accuracy: f32,
//                 source_tick: u64, revealed: bool, target_region_id: Option<u32>,
//                 target_faction_id: Option<u32> }
// NEEDS STATE: next_event_id: u32 on WorldState (for rumor ID generation)
// NEEDS DELTA: AddRumor { id: u32, rumor_type: RumorType, accuracy: f32, source_tick: u64,
//                         target_region_id: Option<u32>, target_faction_id: Option<u32> }
// NEEDS DELTA: ExpireRumor { id: u32 }

/// How often to check for new rumors (in ticks).
const RUMOR_INTERVAL: u64 = 10;

/// Rumors expire after this many ticks if not acted on.
const RUMOR_EXPIRY_TICKS: u64 = 67;

/// Maximum active (unrevealed) rumors at once.
const MAX_ACTIVE_RUMORS: usize = 5;

/// Chance per idle NPC to bring a rumor (30%).
const RUMOR_CHANCE: f32 = 0.30;

/// Types of rumors that can be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RumorType {
    HiddenQuest,
    CrisisWarning,
    FactionPlan,
    TreasureLocation,
    AmbushThreat,
    TradeOpportunity,
}

/// Compute rumor expiry and generation deltas.
///
/// 1. Expire old rumors that have not been investigated.
/// 2. For NPC entities at their home settlement (proxy for "returned"),
///    roll for a new rumor. Accuracy scales with entity level and
///    region visibility.
///
/// Since WorldState lacks rumor storage, this is a structural
/// placeholder. The grid/settlement structure can identify which
/// NPCs are "home" to source rumors.
pub fn compute_rumors(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RUMOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Expire old rumors ---
    // NEEDS STATE: iterate state.rumors
    // For each rumor in state.rumors where !rumor.revealed:
    //   if state.tick - rumor.source_tick >= RUMOR_EXPIRY_TICKS:
    //     out.push(WorldDelta::ExpireRumor { id: rumor.id });

    // --- Phase 2: Generate rumors from NPCs at home settlements ---
    // Count active (unrevealed) rumors.
    // NEEDS STATE: let active_count = state.rumors.iter().filter(|r| !r.revealed).count();
    // if active_count >= MAX_ACTIVE_RUMORS { return; }

    // Find NPCs that are at their home settlement (idle, returned from travel).
    for entity in &state.entities {
        if !entity.alive {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Check if NPC is at home settlement.
        let home_id = match npc.home_settlement_id {
            Some(id) => id,
            None => continue,
        };

        // Verify NPC is on the settlement's grid.
        let at_home = entity.grid_id.map_or(false, |gid| {
            state
                .settlement(home_id)
                .and_then(|s| s.grid_id)
                .map_or(false, |sgid| sgid == gid)
        });
        if !at_home {
            continue;
        }

        // NEEDS STATE: check active_count < MAX_ACTIVE_RUMORS

        // Deterministic roll for rumor generation (read-only state).
        let roll = deterministic_roll(state.tick, entity.id, 0);
        if roll > RUMOR_CHANCE {
            continue;
        }

        // Pick rumor type deterministically.
        let type_idx = deterministic_u32(state.tick, entity.id, 1) % 6;
        let rumor_type = match type_idx {
            0 => RumorType::HiddenQuest,
            1 => RumorType::CrisisWarning,
            2 => RumorType::FactionPlan,
            3 => RumorType::TreasureLocation,
            4 => RumorType::AmbushThreat,
            _ => RumorType::TradeOpportunity,
        };

        // Accuracy scales with entity level.
        let level_factor = (entity.level as f32 / 10.0).clamp(0.2, 1.0);

        // Pick target region.
        let target_region_id = if !state.regions.is_empty() {
            let idx = deterministic_u32(state.tick, entity.id, 2) as usize % state.regions.len();
            Some(state.regions[idx].id)
        } else {
            None
        };

        // Region visibility proxy: inverse of threat level (low threat = well-known).
        let vis_factor = target_region_id
            .and_then(|rid| state.regions.iter().find(|r| r.id == rid))
            .map(|r| (1.0 - r.threat_level / 100.0).clamp(0.1, 1.0))
            .unwrap_or(0.3);

        let accuracy = (level_factor * 0.6 + vis_factor * 0.4).clamp(0.0, 1.0);

        // NEEDS DELTA: AddRumor { id: next_id, rumor_type, accuracy, source_tick: state.tick,
        //                         target_region_id, target_faction_id: None }
        let _ = (rumor_type, accuracy); // suppress unused until delta exists
    }
}

// ---------------------------------------------------------------------------
// Deterministic hashing (no mutable rng in compute phase)
// ---------------------------------------------------------------------------

/// Deterministic float roll in [0, 1) from tick, entity_id, and salt.
fn deterministic_roll(tick: u64, entity_id: u32, salt: u32) -> f32 {
    let h = deterministic_u32(tick, entity_id, salt);
    (h & 0xFFFF) as f32 / 65536.0
}

/// Deterministic u32 hash from tick, entity_id, and salt.
fn deterministic_u32(tick: u64, entity_id: u32, salt: u32) -> u32 {
    // Simple mixing: multiply by large primes and XOR.
    let mut h = tick.wrapping_mul(2654435761) ^ (entity_id as u64).wrapping_mul(40503);
    h ^= (salt as u64).wrapping_mul(2246822519);
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h as u32
}
