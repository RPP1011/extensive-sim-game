#![allow(unused)]
//! World GOAP — goal-oriented action planning for NPC behavior.
//!
//! Lightweight GOAP for the world sim, separate from the tactical combat GOAP.
//! Uses NPC needs, inventory, and world state as preconditions. Plans decompose
//! high-level goals (Eat, Work, Trade) into sub-goal chains:
//!
//! "I want to eat" →
//!   precondition: near food building? NO →
//!     sub-goal: walk to inn →
//!   precondition: have gold? YES →
//!   action: eat (5 ticks)
//!
//! Plans are cached on the goal and re-evaluated when preconditions change.
//!
//! Cadence: every 20 ticks (only for NPCs with active goals missing plans).

use serde::{Serialize, Deserialize};
use crate::world_sim::state::*;
use crate::world_sim::commodity;

const PLAN_EVAL_INTERVAL: u64 = 20;

/// A planned step within a goal. Sub-goals that the NPC must complete in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedStep {
    pub action: PlannedAction,
    pub status: StepStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlannedAction {
    /// Walk to a specific position.
    MoveTo { target: (f32, f32) },
    /// Perform a timed action at current location.
    Perform { activity: String, ticks: u16 },
    /// Pick up / withdraw commodity from nearby building.
    Acquire { commodity: u8, amount: f32 },
    /// Deposit commodity at nearby building.
    Deposit { commodity: u8, amount: f32 },
    /// Pay gold.
    PayGold { amount: f32 },
    /// Wait idle for N ticks.
    Wait { ticks: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    InProgress,
    Complete,
    Failed,
}

/// Generate a plan (sequence of steps) for a goal based on current NPC state.
/// Returns None if the goal can't be planned (missing info).
fn plan_goal(
    goal: &Goal,
    npc: &NpcData,
    entity_pos: (f32, f32),
    settlement_food: f32,
) -> Option<Vec<PlannedStep>> {
    match &goal.kind {
        GoalKind::Eat => {
            let mut steps = Vec::new();
            // Step 1: Walk to food building (if target_pos set).
            if let Some(target) = goal.target_pos {
                let dx = target.0 - entity_pos.0;
                let dy = target.1 - entity_pos.1;
                if dx * dx + dy * dy > 9.0 { // more than 3 units away
                    steps.push(PlannedStep {
                        action: PlannedAction::MoveTo { target },
                        status: StepStatus::Pending,
                    });
                }
            }
            // Step 2: Pay for meal.
            if npc.gold >= 0.5 {
                steps.push(PlannedStep {
                    action: PlannedAction::PayGold { amount: 0.5 },
                    status: StepStatus::Pending,
                });
            }
            // Step 3: Eat (5-tick action).
            steps.push(PlannedStep {
                action: PlannedAction::Perform { activity: "eating".into(), ticks: 5 },
                status: StepStatus::Pending,
            });
            Some(steps)
        }
        GoalKind::Work => {
            let mut steps = Vec::new();
            // Step 1: Walk to work building.
            if let Some(target) = goal.target_pos {
                let dx = target.0 - entity_pos.0;
                let dy = target.1 - entity_pos.1;
                if dx * dx + dy * dy > 9.0 {
                    steps.push(PlannedStep {
                        action: PlannedAction::MoveTo { target },
                        status: StepStatus::Pending,
                    });
                }
            }
            // Step 2: Work (handled by WorkState machine, just mark as perform).
            steps.push(PlannedStep {
                action: PlannedAction::Perform { activity: "working".into(), ticks: 20 },
                status: StepStatus::Pending,
            });
            Some(steps)
        }
        GoalKind::Rest => {
            let mut steps = Vec::new();
            if let Some(target) = goal.target_pos {
                let dx = target.0 - entity_pos.0;
                let dy = target.1 - entity_pos.1;
                if dx * dx + dy * dy > 9.0 {
                    steps.push(PlannedStep {
                        action: PlannedAction::MoveTo { target },
                        status: StepStatus::Pending,
                    });
                }
            }
            steps.push(PlannedStep {
                action: PlannedAction::Perform { activity: "resting".into(), ticks: 30 },
                status: StepStatus::Pending,
            });
            Some(steps)
        }
        GoalKind::Socialize => {
            let mut steps = Vec::new();
            // Walk to a social building if target set.
            if let Some(target) = goal.target_pos {
                steps.push(PlannedStep {
                    action: PlannedAction::MoveTo { target },
                    status: StepStatus::Pending,
                });
            }
            steps.push(PlannedStep {
                action: PlannedAction::Perform { activity: "socializing".into(), ticks: 15 },
                status: StepStatus::Pending,
            });
            Some(steps)
        }
        GoalKind::Trade { destination_settlement_id } => {
            let mut steps = Vec::new();
            // Step 1: Acquire goods from local market.
            steps.push(PlannedStep {
                action: PlannedAction::Acquire { commodity: commodity::FOOD as u8, amount: 5.0 },
                status: StepStatus::Pending,
            });
            // Step 2: Travel to destination.
            if let Some(target) = goal.target_pos {
                steps.push(PlannedStep {
                    action: PlannedAction::MoveTo { target },
                    status: StepStatus::Pending,
                });
            }
            // Step 3: Deposit goods.
            steps.push(PlannedStep {
                action: PlannedAction::Deposit { commodity: commodity::FOOD as u8, amount: 5.0 },
                status: StepStatus::Pending,
            });
            Some(steps)
        }
        GoalKind::Flee { from } => {
            // Simple: move away from danger.
            let away = (entity_pos.0 + (entity_pos.0 - from.0).signum() * 20.0,
                        entity_pos.1 + (entity_pos.1 - from.1).signum() * 20.0);
            Some(vec![PlannedStep {
                action: PlannedAction::MoveTo { target: away },
                status: StepStatus::Pending,
            }])
        }
        _ => None, // other goals don't need multi-step plans yet
    }
}

/// Evaluate goals and generate plans for NPCs that need them.
/// Called post-apply from runtime.rs.
pub fn evaluate_world_goap(state: &mut WorldState) {
    if state.tick % PLAN_EVAL_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement_food: Vec<(u32, f32)> = state.settlements.iter()
        .map(|s| (s.id, s.stockpile[commodity::FOOD]))
        .collect();

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let entity_pos = entity.pos;
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Only plan for the active (top) goal.
        let goal = match npc.goal_stack.current() {
            Some(g) => g.clone(),
            None => continue,
        };

        // Skip if goal already has progress (plan in execution via other systems).
        if goal.progress > 0.0 { continue; }

        let food = npc.home_settlement_id
            .and_then(|sid| settlement_food.iter().find(|(id, _)| *id == sid))
            .map(|(_, f)| *f)
            .unwrap_or(0.0);

        // Generate plan for this goal.
        if let Some(steps) = plan_goal(&goal, npc, entity_pos, food) {
            // Mark goal as having a plan (progress > 0 = planned).
            if let Some(active) = npc.goal_stack.current_mut() {
                active.progress = 0.01; // "planned" sentinel
            }
            // The plan steps inform the goal's target_pos for pathfinding.
            // The first MoveTo step's target becomes the goal's target_pos.
            for step in &steps {
                if let PlannedAction::MoveTo { target } = &step.action {
                    if let Some(active) = npc.goal_stack.current_mut() {
                        active.target_pos = Some(*target);
                    }
                    break;
                }
            }
        }
    }
}
