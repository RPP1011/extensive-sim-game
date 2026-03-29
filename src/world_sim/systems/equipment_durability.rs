#![allow(unused)]
//! Equipment durability — fires every 7 ticks.
//!
//! Gear degrades through use: fighting entities lose attack effectiveness,
//! traveling entities lose move speed. Mapped to Debuff status effects and
//! gold costs for repairs at settlements.
//!
//! Original: `crates/headless_campaign/src/systems/equipment_durability.rs`
//!
//! NEEDS STATE: `equipped_items` with `durability` on NpcData
//! NEEDS DELTA: ModifyDurability, EquipmentBroken

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EconomicIntent, Entity, EntityKind, StatusEffect, StatusEffectKind, WorldState};

/// Durability check interval.
const DURABILITY_INTERVAL: u64 = 7;

/// Gold cost for auto-repair at a settlement.
const REPAIR_COST: f32 = 5.0;

/// Debuff duration when equipment is severely degraded.
const DEGRADED_DEBUFF_MS: u32 = 7000;

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
    _state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        match &npc.economic_intent {
            EconomicIntent::Idle | EconomicIntent::Produce | EconomicIntent::Buy { .. } | EconomicIntent::Sell { .. } => {
                // At settlement: auto-repair costs gold
                if npc.gold >= REPAIR_COST {
                    out.push(WorldDelta::TransferGold {
                        from_id: entity.id,
                        to_id: settlement_id,
                        amount: REPAIR_COST,
                    });
                }
            }
            EconomicIntent::Travel { .. } | EconomicIntent::Trade { .. } => {
                let already_debuffed = entity.status_effects.iter().any(|s| {
                    matches!(&s.kind, StatusEffectKind::Debuff { stat, .. } if stat == "attack")
                });
                if !already_debuffed {
                    out.push(WorldDelta::ApplyStatus {
                        target_id: entity.id,
                        status: StatusEffect {
                            kind: StatusEffectKind::Debuff {
                                stat: "attack".to_string(),
                                factor: 0.95,
                            },
                            source_id: 0,
                            remaining_ms: DEGRADED_DEBUFF_MS,
                        },
                    });
                }
            }
        }
    }
}
