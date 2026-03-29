#![allow(unused)]
//! Equipment durability — fires every 50 ticks.
//!
//! Gear degrades through use:
//! - NPCs in combat (on High-fidelity grids) lose attack damage and armor.
//! - NPCs traveling/trading/adventuring lose move speed.
//! - Degradation emits negative `UpdateEntityField` deltas on the stats that
//!   equipment originally boosted. The equipping system will re-boost stats
//!   when more EQUIPMENT commodity becomes available.
//!
//! Gated on settlement treasury > -100 (skip degradation if economy collapsed).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EconomicIntent, Entity, EntityField, EntityKind, WorldState};

/// Durability check interval (ticks).
const DURABILITY_INTERVAL: u64 = 50;

/// Combat degradation: attack damage lost per degradation tick.
const COMBAT_ATTACK_DEGRADATION: f32 = -0.05;
/// Combat degradation: armor lost per degradation tick.
const COMBAT_ARMOR_DEGRADATION: f32 = -0.02;

/// Travel degradation: move speed lost per degradation tick.
const TRAVEL_SPEED_DEGRADATION: f32 = -0.001;

/// Don't degrade if the settlement treasury is below this threshold.
const ECONOMY_COLLAPSE_THRESHOLD: f32 = -100.0;

pub fn compute_equipment_durability(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DURABILITY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_equipment_durability_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_equipment_durability_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    // Gate: skip degradation if settlement economy has collapsed.
    let treasury = state.settlement(settlement_id)
        .map(|s| s.treasury)
        .unwrap_or(0.0);
    if treasury <= ECONOMY_COLLAPSE_THRESHOLD {
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

        // Check if the entity is on a High-fidelity grid (combat).
        let in_combat = entity.grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.fidelity == Fidelity::High)
            .unwrap_or(false);

        if in_combat {
            // Combat degradation: weapon and armor wear out.
            // Only degrade if the stat is still positive (quality > 0 means stat > base).
            if entity.attack_damage > 0.0 {
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: EntityField::AttackDamage,
                    value: COMBAT_ATTACK_DEGRADATION,
                });
            }
            if entity.armor > 0.0 {
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: EntityField::Armor,
                    value: COMBAT_ARMOR_DEGRADATION,
                });
            }
        }

        match &npc.economic_intent {
            EconomicIntent::Trade { .. }
            | EconomicIntent::Travel { .. }
            | EconomicIntent::Adventuring { .. } => {
                // Travel degradation: boots/gear wear out.
                if entity.move_speed > 0.0 {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::MoveSpeed,
                        value: TRAVEL_SPEED_DEGRADATION,
                    });
                }
            }
            _ => {}
        }
    }
}
