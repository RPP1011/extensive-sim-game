#![allow(unused)]
//! Adventurer hobbies and downtime system — delta architecture port.
//!
//! Idle NPCs with low stress develop side activities that add variety to
//! their behavior tags. Hobby selection is deterministic (entity.id + tick)
//! and weighted by existing behavior profile.
//!
//! Original: `crates/headless_campaign/src/systems/hobbies.rs`
//! Cadence: every 30 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    ActionTags, Entity, EntityField, EconomicIntent, WorldState, tags,
    entity_hash, entity_hash_f32,
};

/// Cadence gate — hobbies tick every 30 ticks.
const HOBBY_TICK_INTERVAL: u64 = 30;

/// Morale threshold: NPCs must have morale > 30 to pursue hobbies.
const MORALE_THRESHOLD: f32 = 30.0;

/// Behavior value threshold for considering an NPC "high" in a tag domain.
const HIGH_BEHAVIOR_THRESHOLD: f32 = 3.0;

/// Compute hobby deltas for all settlements.
pub fn compute_hobbies(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick == 0 || state.tick % HOBBY_TICK_INTERVAL != 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_hobbies_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_hobbies_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick == 0 || state.tick % HOBBY_TICK_INTERVAL != 0 {
        return;
    }

    for entity in entities {
        if !entity.alive {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Only consider NPCs with Produce or Idle intent.
        match &npc.economic_intent {
            EconomicIntent::Produce | EconomicIntent::Idle => {}
            _ => continue,
        }

        // Morale gate.
        if npc.morale <= MORALE_THRESHOLD {
            continue;
        }

        // Build hobby tags based on NPC's existing behavior profile.
        let mut action = ActionTags::empty();
        select_hobby(npc, entity.id, state.tick, &mut action);

        // Apply settlement/NPC context modifiers.
        let action = crate::world_sim::action_context::with_context(&action, entity, state);

        // Emit behavior tag delta.
        out.push(WorldDelta::AddBehaviorTags {
            entity_id: entity.id,
            tags: action.tags,
            count: action.count,
        });

        // Small morale boost from engaging in a hobby (0.5 to 1.0,
        // deterministic based on entity id + tick).
        let morale_boost = 0.5 + deterministic_frac(entity.id, state.tick, 7) * 0.5;
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Morale,
            value: morale_boost,
        });
    }
}

/// Select hobby tags based on the NPC's strongest behavior domain.
/// Deterministic: when multiple domains tie, uses entity.id + tick to break ties.
fn select_hobby(
    npc: &crate::world_sim::state::NpcData,
    entity_id: u32,
    tick: u64,
    action: &mut ActionTags,
) {
    // Compute domain scores from accumulated behavior tags.
    let mining_labor = npc.behavior_value(tags::MINING) + npc.behavior_value(tags::LABOR);
    let trade_negotiation = npc.behavior_value(tags::TRADE) + npc.behavior_value(tags::NEGOTIATION);
    let combat_melee = npc.behavior_value(tags::COMBAT) + npc.behavior_value(tags::MELEE);
    let faith_ritual = npc.behavior_value(tags::FAITH) + npc.behavior_value(tags::RITUAL);

    // Find the strongest domain above threshold.
    // Use a deterministic tiebreaker via hashed ordering.
    let domains: [(f32, u8); 4] = [
        (mining_labor, 0),
        (trade_negotiation, 1),
        (combat_melee, 2),
        (faith_ritual, 3),
    ];

    // Find max score.
    let max_score = domains.iter().map(|d| d.0).fold(0.0f32, f32::max);

    if max_score >= HIGH_BEHAVIOR_THRESHOLD {
        // Among domains that are at or near the max, pick deterministically.
        // "Near" = within 1.0 of max, to allow some variety.
        let near_threshold = max_score - 1.0;
        let mut candidates: [(f32, u8); 4] = [(0.0, 0); 4];
        let mut n_candidates = 0u8;
        for &(score, idx) in &domains {
            if score >= near_threshold && score >= HIGH_BEHAVIOR_THRESHOLD {
                candidates[n_candidates as usize] = (score, idx);
                n_candidates += 1;
            }
        }

        // Deterministic pick among candidates.
        let pick_idx = deterministic_pick(entity_id, tick, n_candidates as u32);
        let chosen = candidates[pick_idx as usize].1;

        match chosen {
            0 => {
                // Mining/labor → studying geology: research + lore
                action.add(tags::RESEARCH, 0.3);
                action.add(tags::LORE, 0.2);
            }
            1 => {
                // Trade/negotiation → socializing: diplomacy + negotiation (persuasion)
                action.add(tags::DIPLOMACY, 0.3);
                action.add(tags::NEGOTIATION, 0.2);
            }
            2 => {
                // Combat/melee → training: tactics + discipline
                action.add(tags::TACTICS, 0.3);
                action.add(tags::DISCIPLINE, 0.2);
            }
            3 => {
                // Faith/ritual → scripture study: lore + research
                action.add(tags::LORE, 0.3);
                action.add(tags::RESEARCH, 0.2);
            }
            _ => unreachable!(),
        }
    } else {
        // Default: walking/foraging: exploration + survival
        action.add(tags::EXPLORATION, 0.2);
        action.add(tags::SURVIVAL, 0.1);
    }
}

// ---------------------------------------------------------------------------
// Deterministic helpers (no mutable RNG — compute phase is read-only)
// ---------------------------------------------------------------------------

/// Deterministic fractional value in [0.0, 1.0) from entity id, tick, and salt.
fn deterministic_frac(entity_id: u32, tick: u64, salt: u32) -> f32 {
    entity_hash_f32(entity_id, tick, salt as u64)
}

/// Deterministic pick in [0, n) from entity id and tick.
fn deterministic_pick(entity_id: u32, tick: u64, n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    entity_hash(entity_id, tick, 13) % n
}

/// Compute the passive bonus scale factor for a hobby.
/// Returns 0.0 at skill 0, ramps linearly to 1.0 at skill 50+.
pub fn hobby_bonus_scale(skill_level: f32) -> f32 {
    (skill_level / 50.0).min(1.0)
}
