//! Named NPC reputation — fires every 10 ticks.
//!
//! Each NPC has a reputation that drifts toward neutral. Completed quests
//! in an NPC's region boost reputation. High reputation unlocks services
//! (discounts, healing). In the delta architecture, reputation effects map
//! to TransferGold (discounts) and Heal (healer services).
//!
//! Original: `crates/headless_campaign/src/systems/npc_reputation.rs`
//!

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

/// Reputation tick interval.
const REPUTATION_TICK_INTERVAL: u64 = 10;

/// Heal amount from healer NPCs with high reputation.
const HEALER_SERVICE_HEAL: f32 = 5.0;

pub fn compute_npc_reputation(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % REPUTATION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_npc_reputation_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_npc_reputation_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % REPUTATION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without named NPC reputation tracking, we approximate:
    // NPCs with healer class tags at settlements provide periodic healing
    // to other NPCs in the same grid (high-reputation healer service).

    // Check if any healer is present
    let has_healer = entities.iter().any(|e| {
        e.alive
            && e.kind == EntityKind::Npc
            && e.npc.as_ref().map_or(false, |npc| {
                npc.class_tags.iter().any(|t| {
                    t.contains("healer") || t.contains("cleric") || t.contains("priest")
                })
            })
    });

    if !has_healer {
        return;
    }

    // Healer provides healing to injured NPCs in the same settlement
    for entity in entities {
        if entity.alive
            && entity.kind == EntityKind::Npc
            && entity.hp < entity.max_hp
        {
            out.push(WorldDelta::Heal {
                target_id: entity.id,
                amount: HEALER_SERVICE_HEAL,
                source_id: 0,
            });
        }
    }
}
