//! Adventurer bond system — delta architecture port.
//!
//! Bonds grow between NPC entities that share a grid (party co-location proxy)
//! and decay slowly otherwise. Bonds affect morale and combat effectiveness.
//!
//! Original: `crates/headless_campaign/src/systems/bonds.rs`
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};


/// Tick cadence — bonds update every 50 ticks.
const BOND_TICK_INTERVAL: u64 = 50;

/// Per-tick decay applied to all bonds.
const BOND_DECAY_RATE: f32 = 0.1;

/// Per-tick growth for entities sharing the same grid.
const BOND_GROWTH_RATE: f32 = 0.5;

/// Maximum bond strength.
const BOND_MAX: f32 = 100.0;

/// Canonical bond key: always (min, max) for symmetric lookup.
fn bond_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Compute bond decay and growth deltas.
///
/// 1. All existing bonds decay by BOND_DECAY_RATE toward 0.
/// 2. NPC entities sharing the same local grid grow bonds by BOND_GROWTH_RATE.
///
/// Since WorldState lacks bond storage, this function is a structural
/// placeholder. Once `adventurer_bonds` is added to WorldState and
/// `UpdateBond` is added to WorldDelta, the logic will emit real deltas.
pub fn compute_bonds(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BOND_TICK_INTERVAL != 0 {
        return;
    }

    // --- Phase 1: Decay existing bonds ---
    // For each (key, value) in state.adventurer_bonds:
    //   out.push(WorldDelta::UpdateBond { entity_a: key.0, entity_b: key.1, delta: -BOND_DECAY_RATE });

    // --- Phase 2: Grow bonds for NPC entities sharing the same grid ---
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_bonds_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_bonds_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % BOND_TICK_INTERVAL != 0 {
        return;
    }

    // Collect NPC entity IDs from the settlement's entities.
    let npc_ids: Vec<u32> = entities
        .iter()
        .filter(|e| e.alive && e.npc.is_some())
        .map(|e| e.id)
        .collect();

    // Emit growth deltas for each pair.
    for i in 0..npc_ids.len() {
        for j in (i + 1)..npc_ids.len() {
            let (a, b) = bond_key(npc_ids[i], npc_ids[j]);
            let _ = (a, b); // suppress unused warning until delta exists
        }
    }
}

// ---------------------------------------------------------------------------
// Query helpers (pure functions, no state mutation)
// ---------------------------------------------------------------------------

/// Look up bond strength between two entities.
/// Requires `adventurer_bonds` on WorldState.
pub fn bond_strength(bonds: &std::collections::HashMap<(u32, u32), f32>, a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    *bonds.get(&bond_key(a, b)).unwrap_or(&0.0)
}

/// Morale bonus from bonds. Bond > 30 with a co-located entity: +5% morale.
pub fn morale_bonus(
    bonds: &std::collections::HashMap<(u32, u32), f32>,
    entity_id: u32,
    neighbors: &[u32],
) -> f32 {
    let max_bond = neighbors
        .iter()
        .filter(|&&id| id != entity_id)
        .map(|&id| bond_strength(bonds, entity_id, id))
        .fold(0.0f32, f32::max);
    if max_bond > 30.0 {
        5.0
    } else {
        0.0
    }
}

/// Combat power multiplier from bonds.
/// Bond > 60: 1.10, bond > 80: 1.15.
pub fn combat_power_multiplier(
    bonds: &std::collections::HashMap<(u32, u32), f32>,
    entity_id: u32,
    neighbors: &[u32],
) -> f32 {
    let max_bond = neighbors
        .iter()
        .filter(|&&id| id != entity_id)
        .map(|&id| bond_strength(bonds, entity_id, id))
        .fold(0.0f32, f32::max);
    if max_bond > 80.0 {
        1.15
    } else if max_bond > 60.0 {
        1.10
    } else {
        1.0
    }
}

/// Whether this entity has "Battle Brothers" status (bond > 80 with any neighbor).
pub fn has_battle_brothers(
    bonds: &std::collections::HashMap<(u32, u32), f32>,
    entity_id: u32,
    neighbors: &[u32],
) -> bool {
    neighbors
        .iter()
        .filter(|&&id| id != entity_id)
        .any(|&id| bond_strength(bonds, entity_id, id) > 80.0)
}

/// Average bond strength among a set of entity IDs.
pub fn average_group_bond(
    bonds: &std::collections::HashMap<(u32, u32), f32>,
    member_ids: &[u32],
) -> f32 {
    if member_ids.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u32;
    for (i, &a) in member_ids.iter().enumerate() {
        for &b in &member_ids[i + 1..] {
            total += bond_strength(bonds, a, b);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}
