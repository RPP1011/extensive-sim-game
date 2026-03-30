#![allow(unused)]
//! Class progression -- grants classes from behavior profiles and levels them up.
//!
//! Cadence: every 50 ticks.
//!
//! This system handles XP accumulation for existing classes based on behavior
//! alignment. Class acquisition (matching behavior profiles to class templates)
//! is handled by the `ClassGenerator` trait on `WorldSim` and run in the
//! runtime's post-system phase, since we only have `&WorldState` here.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, EntityField, WorldState};

const CLASS_INTERVAL: u64 = 50;
const MIN_BEHAVIOR_SUM: f32 = 10.0;

pub fn compute_class_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CLASS_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_class_progression_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_class_progression_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % CLASS_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Skip NPCs with insufficient behavior accumulation.
        let behavior_sum: f32 = npc.behavior_profile.iter().map(|&(_, v)| v).sum();
        if behavior_sum < MIN_BEHAVIOR_SUM {
            continue;
        }

        // XP removed — entity level derived from class levels.
    }
}
