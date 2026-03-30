#![allow(unused)]
//! Legacy weapons — fires every 17 ticks.
//!
//! Weapons that grow with their wielder. In the delta architecture, weapon
//! leveling is represented as Buff status effects (attack bonuses). Weapon
//! creation and XP tracking require additional state.
//!
//! Original: `crates/headless_campaign/src/systems/legacy_weapons.rs`
//!
//! NEEDS STATE: `legacy_weapons: Vec<LegacyWeapon>` on WorldState
//! NEEDS DELTA: CreateLegacyWeapon, LegacyWeaponLevelUp

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// Legacy weapon check interval.
const LEGACY_WEAPON_INTERVAL: u64 = 17;

/// Attack buff from legacy weapon (stacking).
const WEAPON_BUFF_FACTOR: f32 = 1.10;

/// Buff duration (re-applied each tick cycle).
const WEAPON_BUFF_MS: u32 = 17_000;


pub fn compute_legacy_weapons(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % LEGACY_WEAPON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_legacy_weapons_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_legacy_weapons_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % LEGACY_WEAPON_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without legacy weapon tracking, we apply a proxy: high-level NPCs
    // (level >= 5) with high attack have a chance to manifest a "legacy weapon"
    // buff representing their bonded weapon's power.

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        if entity.level < 5 {
            continue;
        }

        let has_weapon_buff = entity.status_effects.iter().any(|s| {
            matches!(&s.kind, StatusEffectKind::Buff { stat, .. } if stat == "legacy_weapon")
        });
        if has_weapon_buff {
            continue;
        }

        let roll = entity_hash_f32(entity.id, state.tick, 0x540ED);
        if roll < 0.08 {
            out.push(WorldDelta::ApplyStatus {
                target_id: entity.id,
                status: StatusEffect {
                    kind: StatusEffectKind::Buff {
                        stat: "legacy_weapon".to_string(),
                        factor: WEAPON_BUFF_FACTOR,
                    },
                    source_id: entity.id,
                    remaining_ms: WEAPON_BUFF_MS,
                },
            });
        }
    }
}
