#![allow(unused)]
//! Named NPC reputation — fires every 10 ticks.
//!
//! Each NPC has a reputation that drifts toward neutral. Completed quests
//! in an NPC's region boost reputation. High reputation unlocks services
//! (discounts, healing). In the delta architecture, reputation effects map
//! to TransferGold (discounts) and Heal (healer services).
//!
//! Original: `crates/headless_campaign/src/systems/npc_reputation.rs`
//!
//! NEEDS STATE: `named_npcs: Vec<NamedNpc>` on WorldState
//! NEEDS STATE: `npc.reputation`, `npc.role` on NpcData
//! NEEDS DELTA: ModifyReputation, UnlockService

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState};

/// Reputation tick interval.
const REPUTATION_TICK_INTERVAL: u64 = 10;

/// Heal amount from healer NPCs with high reputation.
const HEALER_SERVICE_HEAL: f32 = 5.0;

pub fn compute_npc_reputation(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % REPUTATION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without named NPC reputation tracking, we approximate:
    // NPCs with healer class tags at settlements provide periodic healing
    // to other NPCs in the same grid (high-reputation healer service).

    for grid in &state.grids {
        // Check if any healer is present
        let has_healer = grid.entity_ids.iter().any(|&id| {
            state.entity(id).map_or(false, |e| {
                e.alive
                    && e.kind == EntityKind::Npc
                    && e.npc.as_ref().map_or(false, |npc| {
                        npc.class_tags.iter().any(|t| {
                            t.contains("healer") || t.contains("cleric") || t.contains("priest")
                        })
                    })
            })
        });

        if !has_healer {
            continue;
        }

        // Healer provides healing to injured NPCs in the same grid
        for &entity_id in &grid.entity_ids {
            if let Some(entity) = state.entity(entity_id) {
                if entity.alive
                    && entity.kind == EntityKind::Npc
                    && entity.hp < entity.max_hp
                {
                    out.push(WorldDelta::Heal {
                        target_id: entity_id,
                        amount: HEALER_SERVICE_HEAL,
                        source_id: 0,
                    });
                }
            }
        }
    }
}
