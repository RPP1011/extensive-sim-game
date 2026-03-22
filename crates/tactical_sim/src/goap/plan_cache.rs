//! Per-unit plan persistence and replan trigger logic.

use super::action::GoapAction;
use super::goal::{select_goal, Goal};
use super::planner::{self, Plan};
use super::world_state::WorldState;
use std::collections::HashMap;

/// Per-unit plan execution state.
#[derive(Debug, Clone)]
pub struct UnitPlanState {
    pub current_plan: Option<Plan>,
    pub current_step: usize,
    pub step_ticks: u32,
    pub replan_count: u32,
    /// Tick of last replan (for oscillation detection).
    pub last_replan_tick: u64,
    /// Hold timer for oscillation guard.
    pub forced_hold_until: u64,
    /// Recent goal names for oscillation detection (ring buffer of last 3).
    pub goal_history: [Option<String>; 3],
    /// Ticks at which recent goal switches occurred.
    pub goal_switch_ticks: [u64; 3],
    /// Index into goal_history ring buffer.
    pub goal_history_idx: usize,
    /// Consecutive ticks this unit has been idle (Hold with no plan).
    pub idle_ticks: u32,
}

impl Default for UnitPlanState {
    fn default() -> Self {
        Self {
            current_plan: None,
            current_step: 0,
            step_ticks: 0,
            replan_count: 0,
            last_replan_tick: 0,
            forced_hold_until: 0,
            goal_history: [None, None, None],
            goal_switch_ticks: [0; 3],
            goal_history_idx: 0,
            idle_ticks: 0,
        }
    }
}

/// Replan trigger result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplanReason {
    NoPlan,
    GoalChange,
    PreconditionViolation,
    PlanComplete,
    Timeout,
    Periodic,
}

impl UnitPlanState {
    /// Check if we should replan. Returns the reason if so.
    pub fn check_replan(
        &self,
        ws: &WorldState,
        goals: &[Goal],
        actions: &[GoapAction],
        tick: u64,
        hysteresis: f32,
        is_casting: bool,
    ) -> Option<ReplanReason> {
        // Never replan while casting/channeling
        if is_casting {
            return None;
        }

        // Forced hold (oscillation guard)
        if tick < self.forced_hold_until {
            return None;
        }

        let plan = match &self.current_plan {
            None => return Some(ReplanReason::NoPlan),
            Some(p) => p,
        };

        // Plan complete
        if self.current_step >= plan.actions.len() {
            return Some(ReplanReason::PlanComplete);
        }

        // Goal change — higher insistence goal appeared
        let current_goal_name = &plan.goal_name;
        if let Some(best_goal) = select_goal(goals, ws, Some(current_goal_name), hysteresis) {
            if best_goal.name != *current_goal_name {
                return Some(ReplanReason::GoalChange);
            }
        }

        // Current action's preconditions violated
        let action_idx = plan.actions[self.current_step];
        if action_idx < actions.len() {
            let action = &actions[action_idx];
            if !action.preconditions_met(ws) {
                return Some(ReplanReason::PreconditionViolation);
            }
        }

        // Timeout — step exceeded duration * 2
        if action_idx < actions.len() {
            let action = &actions[action_idx];
            if self.step_ticks > action.duration_ticks * 2 {
                return Some(ReplanReason::Timeout);
            }
        }

        // Periodic — every 20 ticks validate
        if tick > 0 && tick % 20 == 0 {
            return Some(ReplanReason::Periodic);
        }

        None
    }

    /// Trigger a replan. Updates oscillation tracking.
    pub fn replan(
        &mut self,
        ws: &WorldState,
        goals: &[Goal],
        actions: &[GoapAction],
        tick: u64,
        hysteresis: f32,
        action_cost_mods: Option<&HashMap<String, f32>>,
    ) {
        // Oscillation guard: if >3 replans in 10 ticks, hold for 5
        if tick - self.last_replan_tick <= 10 {
            self.replan_count += 1;
        } else {
            self.replan_count = 1;
        }
        self.last_replan_tick = tick;

        if self.replan_count > 3 {
            self.forced_hold_until = tick + 5;
            self.current_plan = None;
            self.current_step = 0;
            self.step_ticks = 0;
            return;
        }

        let current_goal_name = self.current_plan.as_ref().map(|p| p.goal_name.as_str());
        let goal = match select_goal(goals, ws, current_goal_name, hysteresis) {
            Some(g) => g,
            None => {
                self.current_plan = None;
                self.current_step = 0;
                self.step_ticks = 0;
                return;
            }
        };

        let new_plan = planner::plan(ws, goal, actions, tick, action_cost_mods);

        // Track goal switches for oscillation detection
        let new_goal = new_plan.as_ref().map(|p| p.goal_name.as_str());
        let prev_goal = self.current_plan.as_ref().map(|p| p.goal_name.as_str());
        if new_goal != prev_goal {
            if let Some(name) = new_goal {
                self.goal_history[self.goal_history_idx] = Some(name.to_string());
                self.goal_switch_ticks[self.goal_history_idx] = tick;
                self.goal_history_idx = (self.goal_history_idx + 1) % 3;
            }
        }

        self.current_plan = new_plan;
        self.current_step = 0;
        self.step_ticks = 0;
    }

    /// Advance to next step in plan. Returns true if plan still has steps.
    pub fn advance_step(&mut self) -> bool {
        self.current_step += 1;
        self.step_ticks = 0;
        self.current_plan
            .as_ref()
            .map_or(false, |p| self.current_step < p.actions.len())
    }

    /// Get the current action index, if plan is active.
    pub fn current_action_idx(&self) -> Option<usize> {
        let plan = self.current_plan.as_ref()?;
        if self.current_step < plan.actions.len() {
            Some(plan.actions[self.current_step])
        } else {
            None
        }
    }

    /// Tick the step timer.
    pub fn tick_step(&mut self) {
        self.step_ticks += 1;
    }

    /// Detect goal oscillation: A→B→A within 10 ticks.
    /// Returns the oscillating goal name if detected.
    pub fn detect_goal_oscillation(&self, tick: u64) -> Option<&str> {
        // Need at least 3 entries to detect A→B→A
        let h = &self.goal_history;
        let t = &self.goal_switch_ticks;

        // Check all 3-element windows in the ring buffer
        for offset in 0..3 {
            let i0 = (self.goal_history_idx + 3 - 3 + offset) % 3;
            let i1 = (i0 + 1) % 3;
            let i2 = (i1 + 1) % 3;

            if let (Some(g0), Some(_g1), Some(g2)) = (&h[i0], &h[i1], &h[i2]) {
                // A→B→A pattern
                if g0 == g2 && t[i2] > 0 && tick.saturating_sub(t[i0]) <= 10 {
                    return Some(g2.as_str());
                }
            }
        }
        None
    }

    /// Track idle ticks. Call with `true` when the unit emitted Hold with no plan.
    pub fn track_idle(&mut self, is_idle: bool) {
        if is_idle {
            self.idle_ticks += 1;
        } else {
            self.idle_ticks = 0;
        }
    }
}
