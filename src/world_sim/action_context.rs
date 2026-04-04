//! Action context modifiers — add settlement/NPC/environment tags to actions.

use crate::world_sim::state::{ActionTags, Entity, WorldState, tags};

/// Merge base action tags with all applicable context modifiers.
pub fn with_context(
    base: &ActionTags,
    entity: &Entity,
    state: &WorldState,
) -> ActionTags {
    let mut result = *base;

    // Settlement context
    if let Some(npc) = &entity.npc {
        if let Some(sid) = npc.home_settlement_id {
            if let Some(settlement) = state.settlement(sid) {
                for &(tag_hash, weight) in &settlement.context_tags {
                    result.add(tag_hash, weight);
                }
            }
        }
    }

    // NPC state context
    add_npc_state_tags(entity, &mut result);

    // Environment context
    add_environment_tags(entity, state, &mut result);

    result
}

fn add_npc_state_tags(entity: &Entity, tags_out: &mut ActionTags) {
    // Injured: resilience (stacking)
    if entity.hp < entity.max_hp * 0.5 && entity.hp > 0.0 {
        tags_out.add(tags::RESILIENCE, 0.4);
    }
    if entity.hp < entity.max_hp * 0.25 && entity.hp > 0.0 {
        tags_out.add(tags::RESILIENCE, 0.3); // stacks
    }

    if let Some(npc) = &entity.npc {
        // High morale: leadership bonus
        if npc.morale > 70.0 {
            tags_out.add(tags::LEADERSHIP, 0.1);
        }
        // High stress: survival instinct
        if npc.stress > 60.0 {
            tags_out.add(tags::SURVIVAL, 0.2);
        }
        // High injury: endurance
        if npc.injury > 50.0 {
            tags_out.add(tags::ENDURANCE, 0.3);
        }
    }
}

fn add_environment_tags(entity: &Entity, state: &WorldState, tags_out: &mut ActionTags) {
    // Frontier: high-threat settlement
    if let Some(npc) = &entity.npc {
        if let Some(sid) = npc.home_settlement_id {
            if let Some(s) = state.settlement(sid) {
                if s.threat_level > 30.0 {
                    tags_out.add(tags::SURVIVAL, 0.3);
                    tags_out.add(tags::AWARENESS, 0.2);
                }
            }
        }
    }

    // In combat (High fidelity grid)
    if let Some(gid) = entity.grid_id {
        if let Some(g) = state.fidelity_zone(gid) {
            if g.fidelity == crate::world_sim::fidelity::Fidelity::High {
                tags_out.add(tags::COMBAT, 0.3);
            }
        }
    }
}
