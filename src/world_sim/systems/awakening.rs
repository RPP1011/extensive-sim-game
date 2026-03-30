#![allow(unused)]
//! Adventurer awakening / transformation — fires every 33 ticks.
//!
//! Rare event: high-level NPCs undergo a permanent power transformation.
//! Maps to Heal (full heal on awakening) and Buff status effects.
//!
//! Original: `crates/headless_campaign/src/systems/awakening.rs`
//!
//! NEEDS STATE: `awakenings: Vec<Awakening>` on WorldState (tracking who awakened)
//! NEEDS STATE: `history_tags` on NpcData for condition checks
//! NEEDS DELTA: PermanentStatBoost (awakening is permanent, not a timed buff)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// How often the awakening system ticks.
const AWAKENING_INTERVAL: u64 = 33;

/// Base chance of awakening when conditions are met.
const AWAKENING_CHANCE: f32 = 0.01;

/// Minimum level for awakening eligibility.
const MIN_AWAKENING_LEVEL: u32 = 8;

/// Heal-to-full on awakening.
const AWAKENING_HEAL_FRACTION: f32 = 1.0;

/// Buff duration representing awakened power (long-lasting proxy).
const AWAKENING_BUFF_MS: u32 = 60_000;


pub fn compute_awakening(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % AWAKENING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_awakening_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_awakening_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % AWAKENING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.level < MIN_AWAKENING_LEVEL {
            continue;
        }

        // Already has awakening buff? Skip.
        let already_awakened = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "awakened")
        });
        if already_awakened {
            continue;
        }

        // Roll for awakening
        let roll = entity_hash_f32(entity.id, state.tick, 0xA4AC);
        if roll >= AWAKENING_CHANCE {
            continue;
        }

        // Heal to full
        let missing = entity.max_hp - entity.hp;
        if missing > 0.0 {
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: missing,
                source_id: entity.id,
            });
        }

        // Apply a long-duration buff representing the awakened state
        out.push(WorldDelta::ApplyStatus {
            target_id: entity.id,
            status: StatusEffect {
                kind: StatusEffectKind::Buff {
                    stat: "awakened".to_string(),
                    factor: 1.25,
                },
                source_id: entity.id,
                remaining_ms: AWAKENING_BUFF_MS,
            },
        });
    }
}
