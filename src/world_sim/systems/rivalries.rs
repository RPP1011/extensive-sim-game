//! Adventurer rivalry system — delta architecture port.
//!
//! NPCs with low bonds develop rivalries that affect party composition,
//! morale, and can escalate to duel challenges. Rivalries can be resolved
//! through mediation, forced cooperation, or natural attrition.
//!
//! Original: `crates/headless_campaign/src/systems/rivalries.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};

//   Rivalry { adventurer_a, adventurer_b, intensity, cause: RivalryCause, started_tick }

/// Cadence gate.
const RIVALRY_TICK_INTERVAL: u64 = 10;

/// Bond threshold below which rivalries can form.
const BOND_RIVALRY_THRESHOLD: f32 = -10.0;

/// Rivalry formation chance (5%).
const RIVALRY_FORMATION_CHANCE: f32 = 0.05;

/// Intensity threshold above which adventurers refuse to share a party.
const REFUSES_PARTY_THRESHOLD: f32 = 30.0;

/// Intensity threshold for duel challenge.
const DUEL_THRESHOLD: f32 = 70.0;

/// Morale penalty per tick cycle when intensity > 50 and both at guild.
const MORALE_PENALTY: f32 = 5.0;

/// Compute rivalry deltas: formation, intensity drift, morale penalties, duels.
///
/// Since WorldState lacks rivalry and bond storage, this is a structural
/// placeholder. The logic maps to the following delta emissions:
pub fn compute_rivalries(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RIVALRY_TICK_INTERVAL != 0 {
        return;
    }

    // --- 1. Rivalry formation from low bonds ---
    // For each alive NPC pair with bond < -10 and no existing rivalry:
    //   Deterministic roll < 0.05:
    //   out.push(WorldDelta::CreateRivalry { ... })

    // --- 2. Intensity drift ---
    // Both NPCs on same grid (co-located): +1.0
    // Separated: -0.5
    // out.push(WorldDelta::UpdateRivalry { entity_a, entity_b, intensity_delta })

    // --- 3. Morale penalty (intensity > 50, both co-located) ---
    // out.push(WorldDelta::AdjustMorale { entity_id: a, delta: -MORALE_PENALTY })
    // out.push(WorldDelta::AdjustMorale { entity_id: b, delta: -MORALE_PENALTY })

    // --- 4. Duel challenges (intensity > 70, both co-located, 10% chance) ---
    // Duels are player choices — need a ChoiceEvent delta or similar

    // --- 5. Cleanup: remove rivalries for dead NPCs or intensity <= 0 ---
    // out.push(WorldDelta::RemoveRivalry { entity_a, entity_b })

    // Identify co-located NPC pairs per settlement (structural skeleton)
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_rivalries_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_rivalries_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % RIVALRY_TICK_INTERVAL != 0 {
        return;
    }

    let npc_ids: Vec<u32> = entities
        .iter()
        .filter(|e| e.alive && e.npc.is_some())
        .map(|e| e.id)
        .collect();

    let _ = npc_ids;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Check if two NPCs have an active rivalry.
/// Requires rivalry state on WorldState.
pub fn has_rivalry(_a: u32, _b: u32) -> bool {
    false
}

/// Returns true if entity `a` refuses to party with entity `b` due to rivalry.
pub fn refuses_party(_a: u32, _b: u32) -> bool {
    false
}

/// Rivalry intensity between two entities (0 if none).
pub fn rivalry_intensity(_a: u32, _b: u32) -> f32 {
    0.0
}
