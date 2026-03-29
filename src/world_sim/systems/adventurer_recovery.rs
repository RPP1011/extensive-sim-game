#![allow(unused)]
//! Adventurer recovery — ported from headless_campaign.
//!
//! Every `RECOVERY_INTERVAL` ticks, NPC entities passively recover from
//! injuries (HP), stress, and fatigue. Recovery rate depends on current
//! activity: idle NPCs recover fastest, fighting NPCs don't recover at all.
//!
//! Original: `crates/headless_campaign/src/systems/adventurer_recovery.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

// NEEDS STATE: stress: f32 on NpcData
// NEEDS STATE: fatigue: f32 on NpcData
// NEEDS STATE: injury: f32 on NpcData
// NEEDS STATE: loyalty: f32 on NpcData
// NEEDS STATE: activity: ActivityStatus on NpcData
// NEEDS DELTA: UpdateCondition { entity_id: u32, stress_delta: f32, fatigue_delta: f32, injury_delta: f32, loyalty_delta: f32 }

/// How often recovery runs (in ticks).
const RECOVERY_INTERVAL: u64 = 100;

/// Per-tick recovery amounts (at 1.0x rate).
const STRESS_RECOVERY: f32 = 2.0;
const FATIGUE_RECOVERY: f32 = 3.0;
const INJURY_RECOVERY: f32 = 1.5;
const LOYALTY_RECOVERY: f32 = 0.5;

/// Injury below this threshold → adventurer transitions back to Idle.
const INJURY_THRESHOLD: f32 = 20.0;
/// Fatigue below this threshold → adventurer transitions back to Idle.
const FATIGUE_THRESHOLD: f32 = 30.0;

/// Compute recovery deltas for all NPC entities.
///
/// Maps recovery to the closest available WorldDelta variants:
///   - Injury recovery → `Heal` (HP recovery)
///   - Stress/fatigue recovery → `RemoveStatus` (clearing debuffs)
///   - Reactivation of recovered NPCs → (no direct delta, handled by
///     condition check on subsequent ticks)
pub fn compute_adventurer_recovery(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RECOVERY_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }

            // Determine recovery rate multiplier based on activity.
            // Infer activity from entity context (same as condition system):
            let on_hostile_grid = entity
                .grid_id
                .and_then(|gid| state.grid(gid))
                .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
                .unwrap_or(false);

            let injured = entity.hp < entity.max_hp * 0.5;
            let on_grid = entity.grid_id.is_some();

            let recovery_rate = if on_hostile_grid {
                0.0 // no recovery in combat
            } else if !on_grid && !injured {
                1.0 // idle — full recovery
            } else if injured && !on_grid {
                0.8 // injured but not in combat
            } else if on_grid {
                0.2 // on mission — slow recovery
            } else {
                0.3
            };

            if recovery_rate <= 0.0 {
                continue;
            }

            // --- Injury recovery as HP heal ---
            // Only recover HP when not actively fighting.
            if entity.hp < entity.max_hp && !on_hostile_grid {
                let heal_amount = INJURY_RECOVERY * recovery_rate;
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount: heal_amount,
                    source_id: entity.id,
                });
            }

            // --- Stress/fatigue recovery as debuff removal ---
            // Remove stress debuff (speed, discriminant 7 for Debuff).
            // Remove fatigue debuff (attack, discriminant 7 for Debuff).
            //
            // When real condition fields exist, this would decrement stress/fatigue
            // directly via UpdateCondition deltas. For now, we clear the debuff
            // status effects that adventurer_condition applies.
            //
            // Only clear debuffs when recovery rate is high enough (idle/injured).
            if recovery_rate >= 0.5 {
                // Clear all Debuff status effects (discriminant 7).
                let has_debuffs = entity.status_effects.iter().any(|se| {
                    matches!(
                        se.kind,
                        crate::world_sim::state::StatusEffectKind::Debuff { .. }
                    )
                });
                if has_debuffs {
                    out.push(WorldDelta::RemoveStatus {
                        target_id: entity.id,
                        status_discriminant: 7, // Debuff discriminant
                    });
                }
            }

            // --- Reactivation check ---
            // In the original system, injured adventurers transition back to Idle
            // when injury <= threshold and fatigue <= threshold.
            // In the delta architecture, we express this as:
            //   if HP is above 80% of max and entity has no debuffs → recovered.
            // No specific delta needed; the Heal above will bring HP up, and
            // RemoveStatus clears debuffs. The entity naturally returns to
            // normal operating state.
        }
    }
}
