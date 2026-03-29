#![allow(unused)]
//! Personal goals system — delta architecture port.
//!
//! Each NPC can pursue a personal ambition beyond guild missions. Goals
//! are assigned based on context and current situation. Fulfilling goals
//! boosts loyalty and morale; neglecting them causes decay.
//!
//! Original: `crates/headless_campaign/src/systems/personal_goals.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: personal_goal on Entity/NpcData
//   AdventurerGoal { goal_type: GoalType, progress, deadline_tick, fulfilled, abandoned }
//   GoalType: ReachLevel, AccumulateGold, DefeatNemesis, VisitHometown,
//             MasterSkill, FormBond, EarnTitle, RetireWealthy, AvengeAlly, ExploreAllRegions
// NEEDS STATE: adventurer loyalty, morale, level, gold, history_tags
// NEEDS DELTA: AssignGoal { entity_id, goal_type, deadline_tick }
// NEEDS DELTA: UpdateGoalProgress { entity_id, progress }
// NEEDS DELTA: FulfillGoal { entity_id }
// NEEDS DELTA: AbandonGoal { entity_id }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: AdjustLoyalty { entity_id, delta }

/// Cadence gate.
const GOAL_TICK_INTERVAL: u64 = 10;

/// Loyalty bonus on goal fulfillment.
const FULFILLMENT_LOYALTY: f32 = 20.0;
/// Morale bonus on goal fulfillment.
const FULFILLMENT_MORALE: f32 = 15.0;
/// Loyalty penalty when deadline passes with < 50% progress.
const NEGLECT_LOYALTY: f32 = 10.0;
/// Morale penalty when deadline passes with < 50% progress.
const NEGLECT_MORALE: f32 = 10.0;

/// Compute personal goal deltas: assign, update progress, fulfill/abandon.
///
/// Since WorldState lacks goal storage on entities, this is a structural
/// placeholder. Gold accumulation goals could reference entity NPC gold.
pub fn compute_personal_goals(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GOAL_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }
        let npc = entity.npc.as_ref().unwrap();

        // --- Phase 1: Assign goals to NPCs without one ---
        // NEEDS STATE: entity.personal_goal
        // Pick goal based on context:
        //   Low level → ReachLevel
        //   Low gold → AccumulateGold { target: level * 50 }
        //   High bond with someone → FormBond
        //   Deterministic weighted selection
        // out.push(WorldDelta::AssignGoal { entity_id, goal_type, deadline_tick })

        // --- Phase 2: Update progress on active goals ---
        // Compute progress based on goal type:
        //   ReachLevel: level >= target → 100%
        //   AccumulateGold: npc.gold / target * 100
        //   FormBond: bond_strength / 80 * 100
        //   EarnTitle: has leadership role or deed → 100%
        // out.push(WorldDelta::UpdateGoalProgress { entity_id, progress })

        // --- Phase 3: Check fulfillment (progress >= 100) ---
        // out.push(WorldDelta::FulfillGoal { entity_id })
        // out.push(WorldDelta::AdjustLoyalty { entity_id, delta: FULFILLMENT_LOYALTY })
        // out.push(WorldDelta::AdjustMorale { entity_id, delta: FULFILLMENT_MORALE })

        // --- Phase 4: Check deadline neglect (progress < 50% at deadline) ---
        // out.push(WorldDelta::AbandonGoal { entity_id })
        // out.push(WorldDelta::AdjustLoyalty { entity_id, delta: -NEGLECT_LOYALTY })
        // out.push(WorldDelta::AdjustMorale { entity_id, delta: -NEGLECT_MORALE })

        // Gold-based progress can use existing NPC gold field
        let _gold = npc.gold;
        let _level = entity.level;
    }
}
