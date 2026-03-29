#![allow(unused)]
//! Prophetic visions — fires every 17 ticks.
//!
//! High-level NPCs receive visions that foreshadow world events. Fulfilled
//! visions grant a morale buff. This system is primarily narrative; the only
//! mechanical delta is a Buff on fulfillment.
//!
//! Original: `crates/headless_campaign/src/systems/visions.rs`
//!
//! NEEDS STATE: `visions: Vec<Vision>` on WorldState
//! NEEDS STATE: `level`, `archetype` criteria on NpcData
//! NEEDS DELTA: AddVision, FulfillVision

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Vision generation cadence.
const VISION_INTERVAL: u64 = 17;

/// Morale buff duration on vision fulfillment.
const VISION_BUFF_MS: u32 = 10_000;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_visions(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % VISION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_visions_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_visions_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % VISION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without vision tracking state, we apply a simple heuristic:
    // High-level NPCs occasionally receive a "prophetic" buff that slightly
    // boosts their effectiveness (proxy for morale from fulfilled visions).

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.level < 7 {
            continue;
        }

        // Already has a vision buff? Skip.
        let has_vision = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "prophetic")
        });
        if has_vision {
            continue;
        }

        let roll = tick_hash(state.tick, entity.id as u64 ^ 0x0EAC1E);
        if roll < 0.05 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: StatusEffect {
                    kind: StatusEffectKind::Buff {
                        stat: "prophetic".to_string(),
                        factor: 1.05,
                    },
                    source_id: entity.id,
                    remaining_ms: VISION_BUFF_MS,
                },
            });
        }
    }
}
