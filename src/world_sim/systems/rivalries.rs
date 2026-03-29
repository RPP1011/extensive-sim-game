#![allow(unused)]
//! Adventurer rivalry system — delta architecture port.
//!
//! NPCs with low bonds develop rivalries that affect party composition,
//! morale, and can escalate to duel challenges. Rivalries can be resolved
//! through mediation, forced cooperation, or natural attrition.
//!
//! Original: `crates/headless_campaign/src/systems/rivalries.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: rivalries: Vec<Rivalry> on WorldState
//   Rivalry { adventurer_a, adventurer_b, intensity, cause: RivalryCause, started_tick }
// NEEDS STATE: adventurer_bonds: HashMap<(u32,u32), f32> on WorldState
// NEEDS STATE: adventurer morale, status on Entity/NpcData
// NEEDS DELTA: CreateRivalry { entity_a, entity_b, intensity, cause }
// NEEDS DELTA: UpdateRivalry { entity_a, entity_b, intensity_delta }
// NEEDS DELTA: RemoveRivalry { entity_a, entity_b }
// NEEDS DELTA: AdjustMorale { entity_id, delta: f32 }

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

    // Identify co-located NPC pairs per grid (structural skeleton)
    for grid in &state.grids {
        let npc_ids: Vec<u32> = grid
            .entity_ids
            .iter()
            .copied()
            .filter(|&eid| {
                state
                    .entity(eid)
                    .map(|e| e.alive && e.npc.is_some())
                    .unwrap_or(false)
            })
            .collect();

        // NEEDS STATE: check bonds and existing rivalries for each pair
        let _ = npc_ids;
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Check if two NPCs have an active rivalry.
/// Requires rivalry state on WorldState.
pub fn has_rivalry(_a: u32, _b: u32) -> bool {
    // NEEDS STATE: rivalries
    false
}

/// Returns true if entity `a` refuses to party with entity `b` due to rivalry.
pub fn refuses_party(_a: u32, _b: u32) -> bool {
    // NEEDS STATE: rivalries, threshold > 30
    false
}

/// Rivalry intensity between two entities (0 if none).
pub fn rivalry_intensity(_a: u32, _b: u32) -> f32 {
    // NEEDS STATE: rivalries
    0.0
}
