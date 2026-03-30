#![allow(unused)]
//! Adventurer nicknames and earned titles — delta architecture port.
//!
//! NPCs earn descriptive nicknames from their deeds. Positive nicknames
//! grant faction relation bonuses and recruitment attraction; infamous
//! nicknames add intimidation at the cost of faction standing. Max 3
//! nicknames per entity.
//!
//! Original: `crates/headless_campaign/src/systems/nicknames.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};
use crate::world_sim::state::entity_hash;

// NEEDS STATE: nicknames: Vec<Nickname> on Entity/NpcData
//   Nickname { title, earned_tick, source: NicknameSource, reputation_modifier }
//   NicknameSource: CombatDeed, DiplomaticAchievement, ExplorationFeat, Sacrifice, etc.
// NEEDS STATE: history_tags: HashMap<String, u32> on Entity/NpcData
// NEEDS STATE: guild reputation
// NEEDS DELTA: GrantNickname { entity_id, title, source, reputation_modifier }
// NEEDS DELTA: AdjustReputation { delta }

/// Cadence gate.
const NICKNAME_TICK_INTERVAL: u64 = 17;

/// Maximum nicknames per entity.
const MAX_NICKNAMES: usize = 3;

/// Tag thresholds for nickname earning.
const COMBAT_KILL_THRESHOLD: u32 = 10;
const DIPLOMATIC_THRESHOLD: u32 = 5;
const EXPLORATION_THRESHOLD: u32 = 8;
const NEAR_DEATH_THRESHOLD: u32 = 3;

/// Nickname templates by category.
const COMBAT_NICKNAMES: &[&str] = &["the Blade", "Ironheart", "Bloodied"];
const DIPLOMATIC_NICKNAMES: &[&str] = &["the Peacemaker", "Silver Tongue", "the Ambassador"];
const EXPLORATION_NICKNAMES: &[&str] = &["Wanderer", "Pathfinder", "the Cartographer"];
const SURVIVAL_NICKNAMES: &[&str] = &["the Undying", "Phoenix", "Lucky"];

/// Compute nickname deltas: evaluate NPCs for nickname thresholds.
///
/// Since WorldState lacks history_tags and nickname storage on entities,
/// this is a structural placeholder. The reputation effect of nicknames
/// could be expressed via a guild-level UpdateTreasury or similar once
/// guild reputation is tracked.
pub fn compute_nicknames(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % NICKNAME_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_nicknames_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_nicknames_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % NICKNAME_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        // NEEDS STATE: read entity.npc.history_tags
        // NEEDS STATE: read entity.npc.nicknames
        //
        // Check each threshold:
        //   kill_count >= 10 && !has_combat_nickname:
        //     Pick deterministic nickname from COMBAT_NICKNAMES
        //     out.push(WorldDelta::GrantNickname { entity_id, title, source: CombatDeed, modifier: 0.05 })
        //
        //   diplo_count >= 5 && !has_diplomatic_nickname:
        //     out.push(WorldDelta::GrantNickname { ..., source: DiplomaticAchievement, modifier: 0.08 })
        //
        //   explore_count >= 8 && !has_exploration_nickname:
        //     out.push(WorldDelta::GrantNickname { ..., source: ExplorationFeat, modifier: 0.04 })
        //
        //   near_death >= 3 && !has_survival_nickname:
        //     out.push(WorldDelta::GrantNickname { ..., source: Sacrifice, modifier: 0.06 })
        //
        // After granting: enforce MAX_NICKNAMES (keep highest reputation_modifier)
        // Apply reputation effect: out.push(WorldDelta::AdjustReputation { delta: modifier * 2.0 })
    }
}

/// Pick a deterministic nickname from a template list.
fn pick_nickname<'a>(templates: &'a [&str], tick: u64, entity_id: u32) -> &'a str {
    let idx = entity_hash(entity_id, tick, 0xA1C4) as usize % templates.len();
    templates[idx]
}
