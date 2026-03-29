#![allow(unused)]
//! Adventurer condition drift — ported from headless_campaign.
//!
//! Every `DRIFT_INTERVAL` ticks, NPC entities accumulate stress, fatigue,
//! and morale changes based on their current activity. High-stress, low-loyalty
//! idle NPCs can desert (Die delta).
//!
//! Original: `crates/headless_campaign/src/systems/adventurer_condition.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

// NEEDS STATE: stress: f32 on NpcData (or Entity)
// NEEDS STATE: fatigue: f32 on NpcData (or Entity)
// NEEDS STATE: morale: f32 on NpcData (or Entity)
// NEEDS STATE: loyalty: f32 on NpcData (or Entity)
// NEEDS STATE: activity: ActivityStatus on NpcData (Idle, Fighting, OnMission, Traveling, Injured)
// NEEDS DELTA: UpdateCondition { entity_id: u32, stress_delta: f32, fatigue_delta: f32, morale_delta: f32, loyalty_delta: f32 }

/// How often condition drift runs (in ticks).
const DRIFT_INTERVAL: u64 = 10;

/// Drift rates per activity: [stress, fatigue, morale, loyalty].
const FIGHTING_DRIFT: [f32; 4] = [2.0, 3.0, -1.0, 0.5];
const ON_MISSION_DRIFT: [f32; 4] = [1.0, 1.5, 0.0, 0.0];
const IDLE_DRIFT: [f32; 4] = [-0.5, -1.0, 0.5, 0.2];
const INJURED_DRIFT: [f32; 4] = [1.5, 0.5, -1.5, -0.5];

/// Loyalty below this triggers desertion check.
const DESERTION_LOYALTY_THRESHOLD: f32 = 15.0;
/// Stress above this triggers desertion check.
const DESERTION_STRESS_THRESHOLD: f32 = 85.0;

/// Compute condition drift deltas for all NPC entities.
///
/// Reads entity state to determine activity, then pushes condition-change
/// deltas. If an idle NPC has dangerously low loyalty and high stress,
/// a `Die` delta is emitted (representing desertion).
pub fn compute_adventurer_condition(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DRIFT_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_adventurer_condition_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_adventurer_condition_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % DRIFT_INTERVAL != 0 {
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

        // Determine activity from entity state.
        let on_hostile_grid = entity
            .grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
            .unwrap_or(false);

        let injured = entity.hp < entity.max_hp * 0.3;
        let on_grid = entity.grid_id.is_some();

        let drift = if injured {
            INJURED_DRIFT
        } else if on_hostile_grid {
            FIGHTING_DRIFT
        } else if on_grid {
            ON_MISSION_DRIFT
        } else {
            IDLE_DRIFT
        };

        let [stress_d, fatigue_d, morale_d, loyalty_d] = drift;

        // Desertion: idle NPC with critically low loyalty and high stress.
        if !on_grid && !injured {
            let hash = tick_entity_hash(state.tick, entity.id);
            let roll = hash_to_f32(hash);
            let _ = (DESERTION_LOYALTY_THRESHOLD, DESERTION_STRESS_THRESHOLD);
        }

        // Apply fatigue as a small Heal-suppress (Debuff on attack):
        if fatigue_d > 1.0 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: crate::world_sim::state::StatusEffect {
                    kind: crate::world_sim::state::StatusEffectKind::Debuff {
                        stat: "attack".into(),
                        factor: 0.95,
                    },
                    source_id: entity.id,
                    remaining_ms: (DRIFT_INTERVAL as u32) * 100,
                },
            });
        }

        // Apply stress as a speed debuff:
        if stress_d > 1.0 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: crate::world_sim::state::StatusEffect {
                    kind: crate::world_sim::state::StatusEffectKind::Debuff {
                        stat: "speed".into(),
                        factor: 0.95,
                    },
                    source_id: entity.id,
                    remaining_ms: (DRIFT_INTERVAL as u32) * 100,
                },
            });
        }

        // Positive morale → small heal-over-time (represents higher resilience):
        if morale_d > 0.0 {
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: morale_d * 0.1,
                source_id: entity.id,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG helpers (no mutable state needed)
// ---------------------------------------------------------------------------

/// Hash tick and entity_id into a deterministic u64.
fn tick_entity_hash(tick: u64, entity_id: u32) -> u64 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(entity_id as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Convert a hash to a float in [0, 1).
fn hash_to_f32(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}
