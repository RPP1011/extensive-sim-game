#![allow(unused)]
//! Personal goals system — delta architecture port.
//!
//! Each NPC can pursue a personal ambition derived from their behavior profile.
//! Goals are not stored on entities — instead, thresholds are checked against
//! accumulated behavior tags and gold. A narrow window after crossing the
//! threshold ensures the reward fires only once.
//!
//! Goal types (derived from dominant behavior tags):
//! 1. **Mastery**: top behavior tag > 500 → aspire to 1000.
//! 2. **Wealth**: gold > 100 → aspire to 500.
//! 3. **Combat**: combat+melee > 100 → aspire to 300.
//! 4. **Scholarly**: research+lore > 100 → aspire to 300.
//! 5. **Social**: trade+diplomacy > 100 → aspire to 200.
//!
//! Cadence: every 100 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::{
    ChronicleCategory, ChronicleEntry, Entity, EntityField, EntityKind, WorldState,
    tags,
};

/// Cadence gate.
const GOAL_TICK_INTERVAL: u64 = 100;

/// Window size after crossing a threshold during which the reward fires.
/// Prevents repeated rewards: only fires when `value >= threshold && value < threshold + WINDOW`.
const REWARD_WINDOW: f32 = 50.0;

// --- Mastery goal ---
/// Prerequisite: top behavior tag must exceed this to have a mastery aspiration.
const MASTERY_PREREQ: f32 = 500.0;
/// Target threshold for mastery goal completion.
const MASTERY_TARGET: f32 = 1000.0;
const MASTERY_MORALE: f32 = 5.0;
const MASTERY_XP: u32 = 10;

// --- Wealth goal ---
const WEALTH_PREREQ: f32 = 100.0;
const WEALTH_TARGET: f32 = 500.0;
const WEALTH_MORALE: f32 = 3.0;
const WEALTH_XP: u32 = 5;

// --- Combat goal ---
const COMBAT_PREREQ: f32 = 100.0;
const COMBAT_TARGET: f32 = 300.0;
const COMBAT_MORALE: f32 = 5.0;
const COMBAT_XP: u32 = 15;

// --- Scholarly goal ---
const SCHOLARLY_PREREQ: f32 = 100.0;
const SCHOLARLY_TARGET: f32 = 300.0;
const SCHOLARLY_MORALE: f32 = 4.0;
const SCHOLARLY_XP: u32 = 10;

// --- Social goal ---
const SOCIAL_PREREQ: f32 = 100.0;
const SOCIAL_TARGET: f32 = 200.0;
const SOCIAL_MORALE: f32 = 3.0;
const SOCIAL_XP: u32 = 5;

/// Check if a value is in the one-shot reward window: [threshold, threshold + WINDOW).
#[inline]
fn in_reward_window(value: f32, threshold: f32) -> bool {
    value >= threshold && value < threshold + REWARD_WINDOW
}

/// Compute personal goal deltas for all settlements.
pub fn compute_personal_goals(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GOAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_personal_goals_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_personal_goals_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % GOAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // --- 1. Mastery: top behavior tag crosses 1000 ---
        // Find the maximum accumulated behavior tag value.
        let top_tag_value = npc
            .behavior_profile
            .iter()
            .map(|&(_, v)| v)
            .fold(0.0f32, f32::max);

        if top_tag_value > MASTERY_PREREQ && in_reward_window(top_tag_value, MASTERY_TARGET) {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: MASTERY_MORALE,
            });
            out.push(WorldDelta::AddXp {
                entity_id: entity.id,
                amount: MASTERY_XP,
            });
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Achievement,
                    text: format!(
                        "{} achieved mastery in their dominant skill",
                        entity_display_name(entity)
                    ),
                    entity_ids: vec![entity.id],
                },
            });
        }

        // --- 2. Wealth: gold crosses 500 ---
        if npc.gold > WEALTH_PREREQ && in_reward_window(npc.gold, WEALTH_TARGET) {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: WEALTH_MORALE,
            });
            out.push(WorldDelta::AddXp {
                entity_id: entity.id,
                amount: WEALTH_XP,
            });
        }

        // --- 3. Combat: combat + melee crosses 300 ---
        let combat_score =
            npc.behavior_value(tags::COMBAT) + npc.behavior_value(tags::MELEE);

        if combat_score > COMBAT_PREREQ && in_reward_window(combat_score, COMBAT_TARGET) {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: COMBAT_MORALE,
            });
            out.push(WorldDelta::AddXp {
                entity_id: entity.id,
                amount: COMBAT_XP,
            });
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Achievement,
                    text: format!(
                        "{} became a seasoned warrior",
                        entity_display_name(entity)
                    ),
                    entity_ids: vec![entity.id],
                },
            });
        }

        // --- 4. Scholarly: research + lore crosses 300 ---
        let scholarly_score =
            npc.behavior_value(tags::RESEARCH) + npc.behavior_value(tags::LORE);

        if scholarly_score > SCHOLARLY_PREREQ
            && in_reward_window(scholarly_score, SCHOLARLY_TARGET)
        {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: SCHOLARLY_MORALE,
            });
            out.push(WorldDelta::AddXp {
                entity_id: entity.id,
                amount: SCHOLARLY_XP,
            });
        }

        // --- 5. Social: trade + diplomacy crosses 200 ---
        let social_score =
            npc.behavior_value(tags::TRADE) + npc.behavior_value(tags::DIPLOMACY);

        if social_score > SOCIAL_PREREQ && in_reward_window(social_score, SOCIAL_TARGET) {
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: SOCIAL_MORALE,
            });
            out.push(WorldDelta::AddXp {
                entity_id: entity.id,
                amount: SOCIAL_XP,
            });
        }
    }
}
