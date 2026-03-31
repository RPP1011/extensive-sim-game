#![allow(unused)]
//! Plan execution system — advances NPC goal plans step by step.
//!
//! Reads the active goal's plan and executes the current step:
//! - Gather: walk to nearest resource node, harvest into inventory
//! - MoveTo: walk toward target position
//! - Perform: wait for N ticks (construction, eating, etc.)
//! - PlaceBuilding: spawn building shell from NPC inventory
//! - Acquire/Deposit/PayGold: inventory transfers
//!
//! When a step completes, advances plan_index. When all steps done, pops the goal.

use crate::world_sim::state::*;
use crate::world_sim::commodity;

const PLAN_EXEC_INTERVAL: u64 = 5;
const HARVEST_DIST_SQ: f32 = 25.0; // 5 units
const HARVEST_PER_CYCLE: f32 = 1.0;
const ARRIVAL_DIST_SQ: f32 = 9.0; // 3 units

/// Advance plans for all NPCs with active goals that have plans.
pub fn advance_plans(state: &mut WorldState) {
    if state.tick % PLAN_EXEC_INTERVAL != 0 { return; }

    let entity_count = state.entities.len();

    // Collect actions to execute: (npc_idx, action_to_take)
    let mut gather_actions: Vec<(usize, u8, f32)> = Vec::new();
    let mut move_actions: Vec<(usize, (f32, f32))> = Vec::new();
    let mut perform_advance: Vec<usize> = Vec::new();
    let mut step_complete: Vec<usize> = Vec::new();
    let mut goal_complete: Vec<usize> = Vec::new();
    let mut build_actions: Vec<(usize, u8)> = Vec::new(); // (npc_idx, building_type_u8)

    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        let goal = match npc.goal_stack.goals.first() {
            Some(g) if !g.plan.is_empty() => g,
            _ => continue,
        };

        let step_idx = goal.plan_index as usize;
        if step_idx >= goal.plan.len() {
            goal_complete.push(i);
            continue;
        }

        let step = &goal.plan[step_idx];
        if step.status == StepStatus::Complete {
            // Advance to next step
            step_complete.push(i);
            continue;
        }

        match &step.action {
            PlannedAction::Gather { commodity, amount } => {
                // Check if we already have enough
                let have = entity.inventory.as_ref()
                    .map(|inv| {
                        let ci = *commodity as usize;
                        if ci < inv.commodities.len() { inv.commodities[ci] } else { 0.0 }
                    })
                    .unwrap_or(0.0);
                if have >= *amount {
                    step_complete.push(i);
                } else {
                    gather_actions.push((i, *commodity, *amount));
                }
            }
            PlannedAction::MoveTo { target } => {
                let dx = target.0 - entity.pos.0;
                let dy = target.1 - entity.pos.1;
                if dx * dx + dy * dy <= ARRIVAL_DIST_SQ {
                    step_complete.push(i);
                } else {
                    move_actions.push((i, *target));
                }
            }
            PlannedAction::Perform { ticks, .. } => {
                if *ticks == 0 {
                    step_complete.push(i);
                } else {
                    perform_advance.push(i);
                }
            }
            PlannedAction::PlaceBuilding { building_type } => {
                // Collect for building placement after the loop
                build_actions.push((i, *building_type));
                step_complete.push(i);
            }
            PlannedAction::PayGold { amount } => {
                let npc = state.entities[i].npc.as_ref().unwrap();
                if npc.gold >= *amount {
                    step_complete.push(i);
                    // Gold deducted elsewhere
                }
            }
            PlannedAction::Wait { ticks } => {
                if *ticks == 0 {
                    step_complete.push(i);
                } else {
                    perform_advance.push(i);
                }
            }
            _ => {}
        }
    }

    // Execute gather actions: find resource, move toward it, harvest
    for (npc_idx, commodity_idx, amount_needed) in gather_actions {
        let npc_pos = state.entities[npc_idx].pos;

        // Find nearest matching resource
        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = f32::MAX;
        for j in 0..entity_count {
            let res = &state.entities[j];
            if !res.alive || res.kind != EntityKind::Resource { continue; }
            let rd = match &res.resource { Some(r) => r, None => continue };
            if rd.remaining <= 0.0 { continue; }
            if rd.resource_type.commodity() != commodity_idx as usize { continue; }
            let dx = res.pos.0 - npc_pos.0;
            let dy = res.pos.1 - npc_pos.1;
            let d = dx * dx + dy * dy;
            if d < best_dist_sq { best_dist_sq = d; best_idx = Some(j); }
        }

        let ri = match best_idx { Some(r) => r, None => continue };
        let res_pos = state.entities[ri].pos;
        let dx = res_pos.0 - npc_pos.0;
        let dy = res_pos.1 - npc_pos.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq > HARVEST_DIST_SQ {
            // Set move_target — movement system will handle actual position updates.
            state.entities[npc_idx].move_target = Some(res_pos);
        } else {
            // Close enough — harvest directly.
            let available = state.entities[ri].resource.as_ref()
                .map(|r| r.remaining).unwrap_or(0.0);
            let harvest = HARVEST_PER_CYCLE.min(available);
            if harvest > 0.0 {
                if let Some(rd) = &mut state.entities[ri].resource {
                    rd.remaining -= harvest;
                    if rd.remaining <= 0.0 && rd.regrow_rate <= 0.0 {
                        state.entities[ri].alive = false;
                    }
                }
                if let Some(inv) = &mut state.entities[npc_idx].inventory {
                    let ci = commodity_idx as usize;
                    if ci < inv.commodities.len() {
                        inv.commodities[ci] += harvest;
                    }
                }
            }
        }
    }

    // Execute move actions — set move_target for the movement system.
    for (npc_idx, target) in move_actions {
        state.entities[npc_idx].move_target = Some(target);
    }

    // Execute build actions — place buildings BEFORE advancing steps
    // so the Build goal is still on the stack when process_npc_builds checks.
    if !build_actions.is_empty() {
        super::buildings::process_npc_builds(state);
    }


    // Advance perform timers
    for npc_idx in perform_advance {
        if let Some(npc) = &mut state.entities[npc_idx].npc {
            if let Some(goal) = npc.goal_stack.goals.first_mut() {
                let si = goal.plan_index as usize;
                if si < goal.plan.len() {
                    match &mut goal.plan[si].action {
                        PlannedAction::Perform { ticks, .. } => {
                            *ticks = ticks.saturating_sub(PLAN_EXEC_INTERVAL as u16);
                        }
                        PlannedAction::Wait { ticks } => {
                            *ticks = ticks.saturating_sub(PLAN_EXEC_INTERVAL as u16);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Advance completed steps
    for npc_idx in step_complete {
        if let Some(npc) = &mut state.entities[npc_idx].npc {
            if let Some(goal) = npc.goal_stack.goals.first_mut() {
                let si = goal.plan_index as usize;
                if si < goal.plan.len() {
                    goal.plan[si].status = StepStatus::Complete;
                }
                goal.plan_index += 1;
                // Update progress
                if !goal.plan.is_empty() {
                    goal.progress = goal.plan_index as f32 / goal.plan.len() as f32;
                }
                // If all steps complete, mark goal done
                if goal.plan_index as usize >= goal.plan.len() {
                    goal.progress = 1.0;
                }
            }
        }
    }

    // Remove completed goals
    for npc_idx in goal_complete {
        if let Some(npc) = &mut state.entities[npc_idx].npc {
            if let Some(goal) = npc.goal_stack.goals.first() {
                if goal.progress >= 1.0 || goal.plan_index as usize >= goal.plan.len() {
                    npc.goal_stack.goals.remove(0);
                }
            }
        }
    }
}
