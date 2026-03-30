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
use crate::world_sim::state::{Entity, EntityField, WorldState};

//   Companion { id, name, species, owner_id, bond_level, acquired_tick }

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

    // For each entity with a companion (companion data not yet on Entity):
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

        // Companion presence morale boost.
        // Until companion data exists on Entity, NPCs on active grids with
        // allies nearby receive a small "companionship" morale boost,
        // simulating the comfort of having companions in the field.
        let on_grid = entity.grid_id.is_some();
        if on_grid {
            // Count friendly allies on the same grid as a proxy for companion effect.
            let ally_count = entity.grid_id
                .and_then(|gid| state.grid(gid))
                .map(|g| {
                    g.entity_ids.iter().filter(|&&eid| {
                        eid != entity.id && state.entity(eid)
                            .map(|e| e.alive && e.team == crate::world_sim::state::WorldTeam::Friendly)
                            .unwrap_or(false)
                    }).count()
                })
                .unwrap_or(0);

            if ally_count > 0 {
                // Each companion adds a small morale boost (diminishing after 3).
                let bonus = (ally_count.min(3) as f32) * 0.5;
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: EntityField::Morale,
                    value: bonus,
                });
            }
        }
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
