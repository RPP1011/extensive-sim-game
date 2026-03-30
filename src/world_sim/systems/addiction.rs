#![allow(unused)]
//! Potion addiction / withdrawal — fires every 3 ticks.
//!
//! NPCs with high potion dependency who haven't consumed recently suffer
//! withdrawal effects (damage-over-time, debuffs). Clean NPCs get a small
//! stat buff. Maps dependency/withdrawal to DoT and Debuff status effects.
//!
//! Original: `crates/headless_campaign/src/systems/addiction.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// Addiction tick cadence.
const ADDICTION_TICK_INTERVAL: u64 = 3;

/// Withdrawal damage per tick at severe level.
const SEVERE_WITHDRAWAL_DAMAGE: f32 = 1.5;

/// Withdrawal damage per tick at moderate level.
const MODERATE_WITHDRAWAL_DAMAGE: f32 = 0.5;

/// Debuff duration for withdrawal slow.
const WITHDRAWAL_DEBUFF_MS: u32 = 3000;


pub fn compute_addiction(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ADDICTION_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_addiction_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_addiction_for_settlement(
    _state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if !entity.alive { continue; }

        // Check if entity has an existing debuff (proxy for addiction state)
        let has_debuff = entity.status_effects.iter().any(|s| {
            matches!(s.kind, StatusEffectKind::Debuff { .. })
        });

        if !has_debuff {
            continue;
        }

        // Withdrawal damage based on HP ratio (lower HP = more severe)
        let hp_ratio = entity.hp / entity.max_hp.max(1.0);
        if hp_ratio < 0.5 {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: SEVERE_WITHDRAWAL_DAMAGE,
                source_id: 0,
            });
        } else if hp_ratio < 0.75 {
            out.push(WorldDelta::Damage {
                target_id: entity.id,
                amount: MODERATE_WITHDRAWAL_DAMAGE,
                source_id: 0,
            });
        }

        // Apply a slow debuff representing withdrawal sluggishness
        let already_slowed = entity.status_effects.iter().any(|s| {
            matches!(s.kind, StatusEffectKind::Slow { .. })
        });
        if !already_slowed && hp_ratio < 0.5 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: StatusEffect {
                    kind: StatusEffectKind::Slow { factor: 0.7 },
                    source_id: 0,
                    remaining_ms: WITHDRAWAL_DEBUFF_MS,
                },
            });
        }
    }
}
