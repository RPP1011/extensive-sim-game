#![allow(unused)]
//! Animal companion system — delta architecture port.
//!
//! NPCs can have animal companions that provide combat, travel, or utility
//! bonuses. Bond level grows when the owner is active and decays when idle.
//! High bond (>70) doubles the species bonus.
//!
//! Original: `crates/headless_campaign/src/systems/companions.rs`
//! Cadence: every 10 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};

// NEEDS STATE: companions: Vec<Companion> on WorldState or Entity
//   Companion { id, name, species, owner_id, bond_level, acquired_tick }
// NEEDS STATE: companion species enum with combat/travel/scouting/morale bonuses
// NEEDS DELTA: UpdateCompanionBond { companion_id, delta: f32 }
// NEEDS DELTA: RemoveCompanion { companion_id }
// NEEDS DELTA: SpawnCompanion { owner_id, species, name }

/// Cadence gate.
const COMPANION_TICK_INTERVAL: u64 = 10;

/// Bond growth per tick when owner is active (on grid with hostiles, traveling).
const BOND_GROWTH_RATE: f32 = 1.0;

/// Bond decay per tick when owner is idle.
const BOND_DECAY_RATE: f32 = 0.5;

/// Bond threshold for doubled species bonus.
const HIGH_BOND_THRESHOLD: f32 = 70.0;

/// Compute companion bond deltas: growth when active, decay when idle.
///
/// Since WorldState lacks companion storage, this is a structural placeholder.
/// Once companions are on Entity or a separate collection, the logic will
/// emit UpdateCompanionBond deltas.
pub fn compute_companions(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COMPANION_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_companions_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_companions_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % COMPANION_TICK_INTERVAL != 0 {
        return;
    }

    // For each entity with a companion (NEEDS STATE: companion data on Entity):
    //
    // Determine owner activity from grid context:
    //   - Entity on a grid with hostiles → active (fighting)
    //   - Entity moving between grids → active (traveling)
    //   - Entity idle at settlement → idle
    //
    // Active: out.push(WorldDelta::UpdateCompanionBond { companion_id, delta: BOND_GROWTH_RATE })
    // Idle:   out.push(WorldDelta::UpdateCompanionBond { companion_id, delta: -BOND_DECAY_RATE })
    //
    // Bond milestone events at 25, 50, 70 would be emitted as separate deltas
    // or logged via an event system layered on top of deltas.

    for entity in entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        // NEEDS STATE: check if entity has a companion
        // NEEDS STATE: determine activity from grid membership
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Returns the effective bonus multiplier for a companion.
/// High bond (>70) doubles the base bonus.
pub fn effective_multiplier(bond_level: f32) -> f32 {
    if bond_level > HIGH_BOND_THRESHOLD {
        2.0
    } else {
        1.0
    }
}
