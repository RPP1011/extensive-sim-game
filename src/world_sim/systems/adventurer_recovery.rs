//! Adventurer recovery — ported from headless_campaign.
//!
//! Every `RECOVERY_INTERVAL` ticks, NPC entities passively recover from
//! injuries (HP), stress, and fatigue. Recovery rate depends on current
//! activity: idle NPCs recover fastest, fighting NPCs don't recover at all.
//!
//! Original: `crates/headless_campaign/src/systems/adventurer_recovery.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, Entity, EntityKind, WorldState, tags};


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

/// How often medicine-accelerated healing runs (in ticks).
const MEDICINE_INTERVAL: u64 = 10;
/// HP threshold (fraction of max) below which medicine is used.
const MEDICINE_HP_THRESHOLD: f32 = 0.8;
/// Amount of MEDICINE commodity consumed per treatment.
const MEDICINE_CONSUME: f32 = 0.1;
/// HP healed per medicine treatment.
const MEDICINE_HEAL: f32 = 5.0;

/// Compute recovery deltas for all NPC entities.
///
/// Maps recovery to the closest available WorldDelta variants:
///   - Injury recovery → `Heal` (HP recovery)
///   - Stress/fatigue recovery → `RemoveStatus` (clearing debuffs)
///   - Reactivation of recovered NPCs → (no direct delta, handled by
///     condition check on subsequent ticks)
pub fn compute_adventurer_recovery(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Run if either recovery (every 100 ticks) or medicine (every 10 ticks) is due.
    let recovery_due = state.tick % RECOVERY_INTERVAL == 0;
    let medicine_due = state.tick % MEDICINE_INTERVAL == 0;
    if !recovery_due && !medicine_due {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_adventurer_recovery_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_adventurer_recovery_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    // --- Natural recovery (every RECOVERY_INTERVAL ticks) ---
    if state.tick % RECOVERY_INTERVAL == 0 {
        for entity in entities {
            if !entity.alive || entity.kind != EntityKind::Npc {
                continue;
            }

            // Determine recovery rate multiplier based on activity.
            let on_hostile_grid = entity
                .grid_id
                .and_then(|gid| state.grid(gid))
                .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
                .unwrap_or(false);

            let injured = entity.hp < entity.max_hp * 0.5;
            let on_grid = entity.grid_id.is_some();

            let recovery_rate = if on_hostile_grid {
                0.0
            } else if !on_grid && !injured {
                1.0
            } else if injured && !on_grid {
                0.8
            } else if on_grid {
                0.2
            } else {
                0.3
            };

            if recovery_rate <= 0.0 {
                continue;
            }

            // --- Injury recovery as HP heal ---
            if entity.hp < entity.max_hp && !on_hostile_grid {
                let heal_amount = INJURY_RECOVERY * recovery_rate;
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount: heal_amount,
                    source_id: entity.id,
                });
            }

            // --- Stress/fatigue recovery as debuff removal ---
            if recovery_rate >= 0.5 {
                let has_debuffs = entity.status_effects.iter().any(|se| {
                    matches!(
                        se.kind,
                        crate::world_sim::state::StatusEffectKind::Debuff { .. }
                    )
                });
                if has_debuffs {
                    out.push(WorldDelta::RemoveStatus {
                        target_id: entity.id,
                        status_discriminant: 7,
                    });
                }
            }
        }
    }

    // --- Medicine-accelerated healing ---
    // Every MEDICINE_INTERVAL ticks, injured NPCs consume medicine from the
    // settlement stockpile for faster healing.
    if state.tick % MEDICINE_INTERVAL != 0 {
        return;
    }

    use crate::world_sim::commodity;

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    if settlement.stockpile[commodity::MEDICINE] <= 0.0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }

        // Only treat injured NPCs (HP < 80% of max).
        if entity.hp >= entity.max_hp * MEDICINE_HP_THRESHOLD {
            continue;
        }

        // Check that settlement still has medicine (approximate: we emit deltas,
        // actual deduction happens in apply phase, but we avoid emitting unbounded).
        // Since this is a delta-based system, over-consumption is clamped on apply.

        // Consume medicine from settlement stockpile.
        out.push(WorldDelta::ConsumeCommodity {
            settlement_id: settlement_id,
            commodity: commodity::MEDICINE,
            amount: MEDICINE_CONSUME,
        });

        // Heal the NPC.
        out.push(WorldDelta::Heal {
            target_id: entity.id,
            amount: MEDICINE_HEAL,
            source_id: 0,
        });

        // Emit behavior tag: medicine:0.5
        let mut action = ActionTags::empty();
        action.add(tags::MEDICINE, 0.5);
        let action = crate::world_sim::action_context::with_context(&action, entity, state);
        out.push(WorldDelta::AddBehaviorTags {
            entity_id: entity.id,
            tags: action.tags,
            count: action.count,
        });
    }
}
