#![allow(unused)]
//! Religious orders and temples — fires every 17 ticks.
//!
//! Temples provide blessings that buff nearby NPCs. Devotion drifts toward
//! zero over time. Active blessings grant Heal or Buff status effects.
//!
//! Original: `crates/headless_campaign/src/systems/religion.rs`
//!
//! NEEDS STATE: `temples: Vec<Temple>` on WorldState with `devotion`, `order`, `blessing_active`
//! NEEDS DELTA: ModifyDevotion

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// How often the religion system ticks.
const RELIGION_INTERVAL: u64 = 17;

/// Blessing heal amount per tick.
const BLESSING_HEAL: f32 = 2.0;

/// Blessing buff duration.
const BLESSING_BUFF_MS: u32 = 17_000;

pub fn compute_religion(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RELIGION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        // Only prosperous settlements have active temples
        if settlement.treasury < 50.0 {
            continue;
        }

        let range = state.group_index.settlement_entities(settlement.id);
        compute_religion_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_religion_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % RELIGION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without temples on WorldState, we use settlements with high treasury
    // as proxy for temple-blessed locations. NPCs at these settlements
    // receive periodic healing (blessing proxy).

    // Blessing: heal all NPCs in the settlement slightly
    for entity in entities {
        if entity.alive && entity.kind == EntityKind::Npc && entity.hp < entity.max_hp {
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: BLESSING_HEAL,
                source_id: 0,
            });
        }
    }
}
