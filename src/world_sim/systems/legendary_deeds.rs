#![allow(unused)]
//! Legendary deeds and reputation titles — delta architecture port.
//!
//! NPCs earn titles from gameplay achievements (kill counts, diplomatic
//! actions, exploration, near-death survivals, etc.). Each deed type can
//! only be earned once per entity.
//!
//! Original: `crates/headless_campaign/src/systems/legendary_deeds.rs`
//! Cadence: every 7 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: deeds: Vec<LegendaryDeed> on Entity/NpcData
//   LegendaryDeed { title, earned_at_tick, deed_type: DeedType, bonus: DeedBonus }
//   DeedType: Slayer, Peacemaker, Explorer, Survivor, Wealthy, Undefeated, Defender, Savior
//   DeedBonus: CombatPowerBoost, FactionRelationBoost, QuestRewardBoost, MoraleAura, RecruitmentBoost
// NEEDS STATE: history_tags: HashMap<String, u32> on Entity/NpcData
// NEEDS DELTA: GrantDeed { entity_id, deed_type, title, bonus }

/// Cadence gate.
const DEED_TICK_INTERVAL: u64 = 7;

/// History tag thresholds for each deed type.
const SLAYER_THRESHOLD: u32 = 10;
const PEACEMAKER_THRESHOLD: u32 = 5;
const EXPLORER_THRESHOLD: u32 = 8;
const SURVIVOR_THRESHOLD: u32 = 3;
const LEGENDARY_THRESHOLD: u32 = 15;
const LONE_WOLF_THRESHOLD: u32 = 5;
const DEFENDER_THRESHOLD: u32 = 5;
const SAVIOR_THRESHOLD: u32 = 5;

/// Compute legendary deed deltas.
///
/// Since WorldState lacks history_tags and deed storage on entities,
/// this is a structural placeholder. The passive bonuses from deeds
/// (combat power, faction relations, morale) would be applied by
/// consuming systems that read the deed list.
pub fn compute_legendary_deeds(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DEED_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.npc.is_none() {
                continue;
            }

            // NEEDS STATE: read entity history_tags and existing deeds
            //
            // kill_count > 10 && !has_deed(Slayer):
            //   out.push(WorldDelta::GrantDeed { entity_id, Slayer, "Slayer", CombatPowerBoost(0.05) })
            //
            // diplo_count > 5 && !has_deed(Peacemaker):
            //   out.push(WorldDelta::GrantDeed { ..., Peacemaker, FactionRelationBoost(10.0) })
            //
            // explore_count > 8 && !has_deed(Explorer):
            //   out.push(WorldDelta::GrantDeed { ..., Explorer, QuestRewardBoost(0.15) })
            //
            // near_death > 3 && !has_deed(Survivor):
            //   out.push(WorldDelta::GrantDeed { ..., Survivor, MoraleAura(0.10) })
            //
            // quest_count > 15 && !has_deed(Wealthy):
            //   out.push(WorldDelta::GrantDeed { ..., "Legendary", QuestRewardBoost(0.20) })
            //
            // solo_count > 5 && !has_deed(Undefeated):
            //   out.push(WorldDelta::GrantDeed { ..., "Lone Wolf", CombatPowerBoost(0.10) })
            //
            // defense_count > 5 && !has_deed(Defender):
            //   out.push(WorldDelta::GrantDeed { ..., "Shield of the Realm", RecruitmentBoost(0.10) })
            //
            // rescue_count > 5 && !has_deed(Savior):
            //   out.push(WorldDelta::GrantDeed { ..., "Savior", FactionRelationBoost(5.0) })
        }
    }
}
