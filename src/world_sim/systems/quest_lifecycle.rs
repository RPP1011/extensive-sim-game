#![allow(unused)]
//! Quest lifecycle — every tick.
//!
//! Progresses quest status through: Dispatched -> InProgress -> InCombat -> Returning -> Complete.
//! On completion, emits Damage/Heal/Die/TransferGold deltas for consequences.
//!
//! Ported from `crates/headless_campaign/src/systems/quest_lifecycle.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: active_quests: Vec<ActiveQuest> on WorldState
// NEEDS STATE: ActiveQuest { id, status: QuestStatus, dispatched_party_id: Option<u32>,
//              quest_type: QuestType, threat_level: f32, elapsed_ticks: u64,
//              member_ids: Vec<u32>, reward_gold: f32 }
// NEEDS STATE: QuestStatus enum { Dispatched, InProgress, InCombat, Returning, Complete }
// NEEDS STATE: QuestType enum { Combat, Rescue, Exploration, Diplomatic, Escort, Gather }
// NEEDS STATE: active_battles on WorldState (for InCombat transition check)
// NEEDS STATE: parties: Vec<Party> on WorldState
// NEEDS STATE: Party { id, status: PartyStatus, member_ids: Vec<u32> }
// NEEDS STATE: PartyStatus enum { Idle, OnMission, Fighting, Returning }
// NEEDS DELTA: UpdateQuestStatus { quest_id: u32, new_status: u8 }
// NEEDS DELTA: StartBattle { quest_id: u32, party_id: u32, enemy_strength: f32, predicted_outcome: f32 }
// NEEDS DELTA: CompleteQuest { quest_id: u32, result: u8, party_id: u32, casualties: u32 }
// NEEDS DELTA: GainXp { entity_id: u32, amount: u32 }

/// Duration multiplier for non-combat quests (threat * this = ticks to complete).
const NON_COMBAT_DURATION_MULT: f32 = 2.0;

/// Base XP for victory.
const VICTORY_BASE_XP: u32 = 50;

/// XP per unit of threat on victory.
const VICTORY_THREAT_XP_RATE: f32 = 2.0;

/// Injury scaling on victory (fraction of damage taken).
const VICTORY_INJURY_SCALING: f32 = 0.3;

/// Base injury on defeat.
const DEFEAT_BASE_INJURY: f32 = 30.0;

/// Death chance when injury > 90%.
const DEATH_CHANCE: f32 = 0.15;

pub fn compute_quest_lifecycle(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Without the quest/party state fields on WorldState, we cannot iterate
    // active quests. This implementation sketches the delta-producing logic
    // that will activate once those fields are added.
    //
    // The pattern: for each active quest, read its status and the associated
    // entity states, then emit deltas for transitions and consequences.

    // -- Phase 1: Dispatched -> InProgress --
    // When the dispatched party's entities have arrived at the quest grid,
    // transition to InProgress. No deltas needed (status change only).
    // NEEDS DELTA: UpdateQuestStatus

    // -- Phase 2: InProgress -> InCombat (combat/rescue quests) --
    // After a delay at location, start a battle.
    // NEEDS DELTA: StartBattle

    // -- Phase 2b: InProgress -> Complete (non-combat quests) --
    // After threat * NON_COMBAT_DURATION_MULT ticks, complete the quest.

    // -- Phase 3: InCombat -> Complete --
    // When the associated battle resolves (victory/defeat), apply consequences.

    // -- Phase 4: Returning -> Complete --
    // When party returns to base (entity grid == home grid).

    // -- Consequences on completion --
    // Since we cannot read quest state yet, demonstrate the delta pattern
    // with a commented-out example:
    //
    // Victory:
    //   - Heal party members for surviving HP
    //   - TransferGold to party members (quest reward)
    //   - (XP/level-up would need a new delta variant)
    //
    // Defeat:
    //   - Damage party members based on threat severity
    //   - Die for members with injury > 90% who fail the death roll
    //
    // Example delta emission for a victorious quest:
    //
    //   for &member_id in &quest.member_ids {
    //       // Heal surviving members
    //       let heal_amount = battle_hp_ratio * 50.0;
    //       out.push(WorldDelta::Heal {
    //           target_id: member_id,
    //           amount: heal_amount,
    //           source_id: 0, // system-generated
    //       });
    //   }
    //
    //   // Distribute gold reward
    //   let share = quest.reward_gold / quest.member_ids.len() as f32;
    //   for &member_id in &quest.member_ids {
    //       out.push(WorldDelta::TransferGold {
    //           from_id: 0, // quest reward pool
    //           to_id: member_id,
    //           amount: share,
    //       });
    //   }
    //
    //   // Defeat: damage + possible death
    //   for &member_id in &quest.member_ids {
    //       let severity = quest.threat_level / 50.0;
    //       let injury = DEFEAT_BASE_INJURY + severity * 20.0;
    //       out.push(WorldDelta::Damage {
    //           target_id: member_id,
    //           amount: injury,
    //           source_id: 0,
    //       });
    //       // Death check would need RNG — deferred to state that has rng_state
    //   }
}
