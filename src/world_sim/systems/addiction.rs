#![allow(unused)]
//! Potion addiction / withdrawal — fires every 3 ticks.
//!
//! NPCs with high potion dependency who haven't consumed recently suffer
//! withdrawal effects (damage-over-time, debuffs). Clean NPCs get a small
//! stat buff. Maps dependency/withdrawal to DoT and Debuff status effects.
//!
//! Original: `crates/headless_campaign/src/systems/addiction.rs`
//!
//! NEEDS STATE: `potion_dependency`, `withdrawal_severity`, `ticks_since_last_potion`
//!              on NpcData
//! NEEDS DELTA: ModifyDependency, ModifyWithdrawal

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Addiction tick cadence.
const ADDICTION_TICK_INTERVAL: u64 = 3;

/// Withdrawal damage per tick at severe level.
const SEVERE_WITHDRAWAL_DAMAGE: f32 = 1.5;

/// Withdrawal damage per tick at moderate level.
const MODERATE_WITHDRAWAL_DAMAGE: f32 = 0.5;

/// Debuff duration for withdrawal slow.
const WITHDRAWAL_DEBUFF_MS: u32 = 3000;

fn tick_hash(tick: u64, salt: u64) -> f32 {
    let x = tick.wrapping_mul(6364136223846793005).wrapping_add(salt);
    let x = x.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 33) as u32) as f32 / u32::MAX as f32
}

pub fn compute_addiction(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ADDICTION_TICK_INTERVAL != 0 {
        return;
    }

    // Without potion_dependency on NpcData, we use a proxy:
    // NPCs with existing DoT status effects and low HP are "withdrawal candidates".
    // This is a placeholder until NpcData gains addiction fields.

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }

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
                // Severe withdrawal
                out.push(WorldDelta::Damage {
                    target_id: entity.id,
                    amount: SEVERE_WITHDRAWAL_DAMAGE,
                    source_id: 0, // environmental
                });
            } else if hp_ratio < 0.75 {
                // Moderate withdrawal
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
}
