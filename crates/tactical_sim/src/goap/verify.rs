//! Runtime verification for GOAP invariants. Extends the verify_tick() pattern.

use super::dsl::RoleHint;
use super::world_state::{self, WorldState};
use super::GoapAiState;
use crate::sim::{is_alive, SimState};

/// GOAP-specific invariant violation.
#[derive(Debug, Clone, PartialEq)]
pub enum GoapViolation {
    /// Plan references a dead target.
    PlanTargetDead { unit_id: u32, target_id: u32 },
    /// current_step >= plan.actions.len() but plan not cleared.
    PlanStepOverflow { unit_id: u32, step: usize, plan_len: usize },
    /// Plan's goal is already satisfied but plan not cleared.
    PlanGoalSatisfied { unit_id: u32, goal_name: String },
    /// Current action's preconditions unmet for sustained period.
    PlanActionPreconditionViolated { unit_id: u32, action_name: String },
    /// Cached world state disagrees with SimState on key field.
    WorldStateStaleProp { unit_id: u32, prop: &'static str },
    /// TARGET_ID points to non-existent unit.
    WorldStateInvalidTarget { unit_id: u32, target_id: u32 },
    /// More than N replans in a short window.
    ExcessiveReplanning { unit_id: u32, count: u32 },
    /// Goal switched A→B→A within 10 ticks.
    GoalOscillation { unit_id: u32, goal_name: String },
    /// Unit holding for too long while enemies exist.
    UnitIdleWithEnemies { unit_id: u32, idle_ticks: u32 },
    /// Healer ignoring critical ally.
    HealerIgnoringCritical { unit_id: u32, ally_id: u32 },
}

/// Run GOAP verification checks. Returns list of violations found.
pub fn verify_goap(goap: &GoapAiState, state: &SimState) -> Vec<GoapViolation> {
    let mut violations = Vec::new();

    for (&unit_id, plan_state) in &goap.plans {
        let unit = match state.units.iter().find(|u| u.id == unit_id && is_alive(u)) {
            Some(u) => u,
            None => continue,
        };

        let def = match goap.defs.get(&unit_id) {
            Some(d) => d,
            None => continue,
        };

        // ── Plan integrity ──

        if let Some(plan) = &plan_state.current_plan {
            // PlanStepOverflow
            if plan_state.current_step >= plan.actions.len() {
                violations.push(GoapViolation::PlanStepOverflow {
                    unit_id,
                    step: plan_state.current_step,
                    plan_len: plan.actions.len(),
                });
            }

            // PlanGoalSatisfied
            if let Some(ws) = goap.world_cache.get(&unit_id) {
                if let Some(goal) = def.goals.iter().find(|g| g.name == plan.goal_name) {
                    if goal.is_satisfied(ws) {
                        violations.push(GoapViolation::PlanGoalSatisfied {
                            unit_id,
                            goal_name: plan.goal_name.clone(),
                        });
                    }
                }
            }

            // PlanActionPreconditionViolated (>2 ticks without replan)
            if let Some(action_idx) = plan_state.current_action_idx() {
                if action_idx < def.actions.len() {
                    let action = &def.actions[action_idx];
                    if let Some(ws) = goap.world_cache.get(&unit_id) {
                        if !action.preconditions_met(ws) && plan_state.step_ticks > 2 {
                            violations.push(GoapViolation::PlanActionPreconditionViolated {
                                unit_id,
                                action_name: action.name.clone(),
                            });
                        }
                    }
                }
            }

            // PlanTargetDead — check if any action in the plan references a target
            // that has since died. We check the current action's intent template for
            // target references by resolving the cached world state TARGET_ID.
            if let Some(ws) = goap.world_cache.get(&unit_id) {
                if let world_state::PropValue::Id(Some(target_id)) =
                    ws.get(world_state::TARGET_ID)
                {
                    if !state.units.iter().any(|u| u.id == target_id && is_alive(u)) {
                        violations.push(GoapViolation::PlanTargetDead {
                            unit_id,
                            target_id,
                        });
                    }
                }
            }
        }

        // ── World state consistency ──

        if let Some(ws) = goap.world_cache.get(&unit_id) {
            // WorldStateInvalidTarget
            if let world_state::PropValue::Id(Some(target_id)) =
                ws.get(world_state::TARGET_ID)
            {
                if !state.units.iter().any(|u| u.id == target_id && is_alive(u)) {
                    violations.push(GoapViolation::WorldStateInvalidTarget {
                        unit_id,
                        target_id,
                    });
                }
            }

            // WorldStateStaleProp — spot-check key properties against SimState
            let unit_idx = state.units.iter().position(|u| u.id == unit_id);
            if let Some(idx) = unit_idx {
                let fresh = WorldState::extract(state, idx);

                // Check SELF_HP_PCT
                let cached_hp = ws.get(world_state::SELF_HP_PCT).as_float();
                let fresh_hp = fresh.get(world_state::SELF_HP_PCT).as_float();
                if (cached_hp - fresh_hp).abs() > 0.01 {
                    violations.push(GoapViolation::WorldStateStaleProp {
                        unit_id,
                        prop: "SELF_HP_PCT",
                    });
                }

                // Check SELF_IS_CCD
                if ws.get(world_state::SELF_IS_CCD).as_bool()
                    != fresh.get(world_state::SELF_IS_CCD).as_bool()
                {
                    violations.push(GoapViolation::WorldStateStaleProp {
                        unit_id,
                        prop: "SELF_IS_CCD",
                    });
                }

                // Check ENEMY_COUNT
                let cached_ec = ws.get(world_state::ENEMY_COUNT).as_float();
                let fresh_ec = fresh.get(world_state::ENEMY_COUNT).as_float();
                if (cached_ec - fresh_ec).abs() > 0.5 {
                    violations.push(GoapViolation::WorldStateStaleProp {
                        unit_id,
                        prop: "ENEMY_COUNT",
                    });
                }
            }
        }

        // ── Oscillation detection ──

        // ExcessiveReplanning — >5 replans in 15 ticks
        if plan_state.replan_count > 5
            && state.tick.saturating_sub(plan_state.last_replan_tick) <= 15
        {
            violations.push(GoapViolation::ExcessiveReplanning {
                unit_id,
                count: plan_state.replan_count,
            });
        }

        // GoalOscillation — A→B→A within 10 ticks
        if let Some(goal_name) = plan_state.detect_goal_oscillation(state.tick) {
            violations.push(GoapViolation::GoalOscillation {
                unit_id,
                goal_name: goal_name.to_string(),
            });
        }

        // ── Behavioral sanity ──

        // UnitIdleWithEnemies — holding >10 ticks while enemies exist and HP > 30%
        if plan_state.idle_ticks > 10 {
            let hp_pct = if unit.max_hp > 0 {
                unit.hp as f32 / unit.max_hp as f32
            } else {
                0.0
            };
            let enemies_exist = state.units.iter().any(|u| is_alive(u) && u.team != unit.team);
            if enemies_exist && hp_pct > 0.3 {
                violations.push(GoapViolation::UnitIdleWithEnemies {
                    unit_id,
                    idle_ticks: plan_state.idle_ticks,
                });
            }
        }

        // HealerIgnoringCritical — support-role unit not healing ally <20% HP for >5 ticks
        if def.role_hint == Some(RoleHint::Support) && plan_state.idle_ticks > 5 {
            for ally in &state.units {
                if !is_alive(ally) || ally.team != unit.team || ally.id == unit.id {
                    continue;
                }
                let ally_pct = if ally.max_hp > 0 {
                    ally.hp as f32 / ally.max_hp as f32
                } else {
                    1.0
                };
                if ally_pct < 0.2 {
                    violations.push(GoapViolation::HealerIgnoringCritical {
                        unit_id,
                        ally_id: ally.id,
                    });
                    break; // one violation per healer per tick is enough
                }
            }
        }
    }

    violations
}
