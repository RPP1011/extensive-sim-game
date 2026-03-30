#![allow(unused)]
//! Bloodline legacy system — fires every 33 ticks.
//!
//! When high-level NPCs die, their bloodline can produce descendants with
//! inherited stat bonuses. In the delta architecture, this is modeled as
//! Heal (prestige recovery) and Buff (inherited power) effects.
//!
//! Original: `crates/headless_campaign/src/systems/bloodlines.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// Bloodline check interval.
const BLOODLINE_INTERVAL: u64 = 33;

/// Buff factor for bloodline descendants.
const BLOODLINE_BUFF_FACTOR: f32 = 1.10;

/// Buff duration for bloodline power.
const BLOODLINE_BUFF_MS: u32 = 30_000;


pub fn compute_bloodlines(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BLOODLINE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_bloodlines_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_bloodlines_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % BLOODLINE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without bloodline tracking, we apply a proxy: high-level NPCs (level >= 8)
    // occasionally receive a "bloodline power" buff representing inherited strength.

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.level < 8 {
            continue;
        }

        let has_bloodline = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "bloodline")
        });
        if has_bloodline {
            continue;
        }

        let roll = entity_hash_f32(entity.id, state.tick, 0xB100D);
        if roll < 0.05 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: StatusEffect {
                    kind: StatusEffectKind::Buff {
                        stat: "bloodline".to_string(),
                        factor: BLOODLINE_BUFF_FACTOR,
                    },
                    source_id: entity.id,
                    remaining_ms: BLOODLINE_BUFF_MS,
                },
            });
        }
    }
}
