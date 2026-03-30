#![allow(unused)]
//! Divine favor economy — fires every 7 ticks.
//!
//! Accumulated devotion from temples/shrines can grant miracles (heals,
//! gold blessings). Displeasure causes damage. Maps to Heal, Damage,
//! and UpdateTreasury deltas.
//!
//! Original: `crates/headless_campaign/src/systems/divine_favor.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};
use crate::world_sim::state::{entity_hash_f32};

/// How often the divine favor system ticks.
const DIVINE_FAVOR_INTERVAL: u64 = 7;

/// Miracle heal amount.
const MIRACLE_HEAL: f32 = 15.0;

/// Miracle gold blessing amount.
const MIRACLE_GOLD: f32 = 20.0;

/// Displeasure damage amount.
const DISPLEASURE_DAMAGE: f32 = 5.0;


pub fn compute_divine_favor(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DIVINE_FAVOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_divine_favor_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_divine_favor_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % DIVINE_FAVOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without temples/divine_favor on WorldState, we approximate:
    // Settlements with high treasury (proxy for active temples) occasionally
    // grant healing miracles to nearby NPCs. Settlements with very low treasury
    // suffer divine displeasure (damage).

    let settlement = match state.settlements.iter().find(|s| s.id == settlement_id) {
        Some(s) => s,
        None => return,
    };

    if settlement.treasury > 100.0 {
        // Miracle: heal a random NPC in the settlement
        let roll = entity_hash_f32(settlement_id, state.tick, 0xFA17_4001);
        if roll < 0.15 {
            for entity in entities {
                if entity.alive && entity.kind == EntityKind::Npc && entity.hp < entity.max_hp {
                    out.push(WorldDelta::Heal {
                        target_id: entity.id,
                        amount: MIRACLE_HEAL,
                        source_id: 0,
                    });
                    break; // One miracle per settlement per tick
                }
            }
        }

        // Gold blessing to settlement treasury
        let roll2 = entity_hash_f32(settlement_id, state.tick, 0xB1E55);
        if roll2 < 0.10 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement_id,
                delta: MIRACLE_GOLD,
            });
        }
    } else if settlement.treasury < 10.0 {
        // Displeasure: damage NPCs
        let roll = entity_hash_f32(settlement_id, state.tick, 0xD15FEEA5E);
        if roll < 0.10 {
            for entity in entities {
                if entity.alive && entity.kind == EntityKind::Npc {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: DISPLEASURE_DAMAGE,
                        source_id: 0,
                    });
                    break;
                }
            }
        }
    }
}
