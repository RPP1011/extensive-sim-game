//! A* regressive GOAP planner. Starts from the goal's desired state,
//! works backward through action effects to find a sequence of actions
//! whose preconditions are satisfied by the current world state.

use super::action::GoapAction;
use super::goal::{Goal, Precondition};
use super::world_state::WorldState;

/// A completed plan — sequence of action indices into a GoapDef's action list.
#[derive(Debug, Clone)]
pub struct Plan {
    pub actions: Vec<usize>,
    pub goal_name: String,
    pub created_tick: u64,
}

const MAX_ITERATIONS: u32 = 64;

/// A planner node: set of unsatisfied conditions + path so far.
#[derive(Debug, Clone)]
struct Node {
    /// Conditions still unsatisfied (prop_idx, precondition).
    unsatisfied: Vec<(usize, Precondition)>,
    /// Action indices taken (in reverse order — goal to start).
    path: Vec<usize>,
    /// Accumulated cost.
    cost: f32,
}

impl Node {
    fn heuristic(&self) -> f32 {
        self.unsatisfied.len() as f32
    }

    fn priority(&self) -> f32 {
        self.cost + self.heuristic()
    }
}

/// Run the A* regressive planner.
///
/// Returns `None` if no valid plan found within budget.
/// For trivial cases (few actions, shallow plans), uses a fast linear scan
/// instead of full A*.
pub fn plan(
    current_ws: &WorldState,
    goal: &Goal,
    actions: &[GoapAction],
    tick: u64,
    action_cost_mods: Option<&std::collections::HashMap<String, f32>>,
) -> Option<Plan> {
    // Fast path: if any single action directly resolves all goal conditions
    // from the current world state, skip A* entirely. This covers desugared
    // behavior trees (single-goal, 1-step plans) and most native GOAP defs
    // where the best action is immediately available.
    if let Some(plan) = try_single_action_plan(current_ws, goal, actions, tick, action_cost_mods) {
        return Some(plan);
    }

    // Full A* for multi-step plans
    // Start node: all goal conditions are "unsatisfied"
    let initial_unsatisfied: Vec<(usize, Precondition)> = goal
        .desired
        .iter()
        .filter(|(prop, pre)| !pre.is_satisfied(current_ws.get(*prop)))
        .cloned()
        .collect();

    if initial_unsatisfied.is_empty() {
        // Goal already satisfied
        return None;
    }

    let mut open = Vec::with_capacity(32);
    open.push(Node {
        unsatisfied: initial_unsatisfied,
        path: Vec::new(),
        cost: 0.0,
    });

    let mut iterations = 0u32;

    while let Some(current) = pop_best(&mut open) {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            break;
        }

        // Check if current world state satisfies all remaining conditions
        if current.unsatisfied.iter().all(|(prop, pre)| pre.is_satisfied(current_ws.get(*prop))) {
            // Plan found — reverse the path (it was built goal→start)
            let mut action_seq = current.path;
            action_seq.reverse();
            return Some(Plan {
                actions: action_seq,
                goal_name: goal.name.clone(),
                created_tick: tick,
            });
        }

        // Try each action that could satisfy at least one unsatisfied condition
        for (action_idx, action) in actions.iter().enumerate() {
            // Pre-filter: does this action affect any unsatisfied property?
            let props_needed: Vec<usize> = current.unsatisfied.iter().map(|(p, _)| *p).collect();
            if !action.affects_any(&props_needed) {
                continue;
            }

            // Avoid duplicate actions in plan (simple cycle prevention)
            if current.path.contains(&action_idx) {
                continue;
            }

            // Apply action effects: remove satisfied conditions, add new preconditions
            let mut new_unsatisfied = Vec::new();

            // Keep conditions that this action doesn't resolve
            for (prop, pre) in &current.unsatisfied {
                let resolved = action.effects.iter().any(|(ep, ev)| {
                    *ep == *prop && pre.is_satisfied(*ev)
                });
                if !resolved {
                    new_unsatisfied.push((*prop, pre.clone()));
                }
            }

            // Add action's preconditions that aren't already satisfied by current world state
            // and aren't already in our unsatisfied list
            for (prop, pre) in &action.preconditions {
                if !pre.is_satisfied(current_ws.get(*prop))
                    && !new_unsatisfied.iter().any(|(p, _)| *p == *prop)
                {
                    new_unsatisfied.push((*prop, pre.clone()));
                }
            }

            let action_cost = action.cost
                * action_cost_mods
                    .and_then(|m| m.get(&action.name))
                    .copied()
                    .unwrap_or(1.0);

            let mut new_path = current.path.clone();
            new_path.push(action_idx);

            open.push(Node {
                unsatisfied: new_unsatisfied,
                path: new_path,
                cost: current.cost + action_cost,
            });
        }
    }

    None // No plan found within budget
}

/// Fast path: find the cheapest single action that directly resolves all goal
/// conditions from the current world state. O(actions × conditions) with no
/// heap allocation. Returns None if no single action suffices (fall through to A*).
fn try_single_action_plan(
    current_ws: &WorldState,
    goal: &Goal,
    actions: &[GoapAction],
    tick: u64,
    action_cost_mods: Option<&std::collections::HashMap<String, f32>>,
) -> Option<Plan> {
    // Collect unsatisfied goal conditions
    let unsatisfied: Vec<&(usize, Precondition)> = goal
        .desired
        .iter()
        .filter(|(prop, pre)| !pre.is_satisfied(current_ws.get(*prop)))
        .collect();

    if unsatisfied.is_empty() {
        return None; // goal already satisfied
    }

    let mut best_idx: Option<usize> = None;
    let mut best_cost = f32::MAX;

    for (i, action) in actions.iter().enumerate() {
        // All preconditions must be met right now
        if !action.preconditions_met(current_ws) {
            continue;
        }

        // All unsatisfied goal conditions must be resolved by this action's effects
        let resolves_all = unsatisfied.iter().all(|&(prop, ref pre)| {
            // Either already satisfied by world state, or this action's effects satisfy it
            pre.is_satisfied(current_ws.get(*prop))
                || action.effects.iter().any(|(ep, ev)| *ep == *prop && pre.is_satisfied(*ev))
        });

        if !resolves_all {
            continue;
        }

        let cost = action.cost
            * action_cost_mods
                .and_then(|m| m.get(&action.name))
                .copied()
                .unwrap_or(1.0);

        if cost < best_cost {
            best_cost = cost;
            best_idx = Some(i);
        }
    }

    best_idx.map(|idx| Plan {
        actions: vec![idx],
        goal_name: goal.name.clone(),
        created_tick: tick,
    })
}

/// Pop the node with the lowest priority (cost + heuristic).
fn pop_best(open: &mut Vec<Node>) -> Option<Node> {
    if open.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_priority = open[0].priority();
    for (i, node) in open.iter().enumerate().skip(1) {
        let p = node.priority();
        if p < best_priority {
            best_priority = p;
            best_idx = i;
        }
    }
    Some(open.swap_remove(best_idx))
}
