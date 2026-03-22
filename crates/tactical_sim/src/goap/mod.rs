//! GOAP (Goal-Oriented Action Planning) AI system.
//!
//! Replaces stateless force-based steering with persistent, goal-driven plans.
//! Each unit picks a goal, plans an action sequence, and executes across ticks.
//! Replanning occurs only when the world changes meaningfully.

pub mod action;
pub mod dsl;
pub mod goal;
pub mod party;
pub mod plan_cache;
pub mod planner;
pub mod spatial;
pub mod target;
pub mod verify;
pub mod world_state;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use self::target::Target;
use crate::sim::{
    distance, is_alive, move_away, move_towards, position_at_range, IntentAction, SimState,
    UnitIntent,
};

use self::action::IntentTemplate;
use self::dsl::GoapDef;
use self::party::PartyCulture;
use self::plan_cache::UnitPlanState;
use self::world_state::WorldState;

/// Main GOAP AI state, holding per-unit definitions and plans.
#[derive(Debug, Clone)]
pub struct GoapAiState {
    /// unit_id → loaded GOAP definition.
    pub defs: HashMap<u32, GoapDef>,
    /// unit_id → current plan execution state.
    pub plans: HashMap<u32, UnitPlanState>,
    /// unit_id → cached world state (refreshed each tick).
    pub world_cache: HashMap<u32, WorldState>,
    /// Optional party culture for modifying GOAP parameters.
    pub culture: Option<PartyCulture>,
}

impl GoapAiState {
    pub fn new(defs: HashMap<u32, GoapDef>, culture: Option<PartyCulture>) -> Self {
        let plans = defs.keys().map(|&id| (id, UnitPlanState::default())).collect();
        Self {
            defs,
            plans,
            world_cache: HashMap::new(),
            culture,
        }
    }

    /// Generate intents for all GOAP-enabled units. Returns intents only for
    /// units that have a GoapDef; other units should be handled by the caller.
    ///
    /// `focus_targets` maps team → focus target ID from the squad blackboard.
    /// When set, GOAP units on that team override their TARGET_ID to the focus target.
    pub fn generate_intents(
        &mut self,
        state: &SimState,
        _dt_ms: u32,
        focus_targets: &std::collections::HashMap<crate::sim::Team, Option<u32>>,
    ) -> Vec<UnitIntent> {
        let tick = state.tick;
        let hysteresis = self.culture.as_ref().map_or(0.15, |c| c.replan_hysteresis);
        let action_cost_mods = self.culture.as_ref().map(|c| &c.action_cost_modifiers);
        let _goal_insistence_mods = self.culture.as_ref().map(|c| &c.goal_insistence_modifiers);

        let mut intents = Vec::new();

        for (unit_idx, unit) in state.units.iter().enumerate() {
            if !is_alive(unit) {
                continue;
            }
            let unit_id = unit.id;

            let def = match self.defs.get(&unit_id) {
                Some(d) => d,
                None => continue,
            };

            // Extract world state, incorporating team focus target if set
            let focus = focus_targets
                .get(&unit.team)
                .copied()
                .flatten();
            let ws = if focus.is_some() {
                WorldState::extract_with_focus(state, unit_idx, focus)
            } else {
                WorldState::extract(state, unit_idx)
            };
            self.world_cache.insert(unit_id, ws.clone());

            // Get or create plan state
            let plan_state = self.plans.entry(unit_id).or_default();

            // Check for casting/channeling (don't interrupt)
            let is_casting = unit.casting.is_some() || unit.channeling.is_some();

            // Check replan triggers
            if let Some(_reason) = plan_state.check_replan(
                &ws,
                &def.goals,
                &def.actions,
                tick,
                hysteresis,
                is_casting,
            ) {
                plan_state.replan(&ws, &def.goals, &def.actions, tick, hysteresis, action_cost_mods);
            }

            // Execute current action or hold
            let action = if is_casting {
                IntentAction::Hold
            } else if tick < plan_state.forced_hold_until {
                IntentAction::Hold
            } else if let Some(action_idx) = plan_state.current_action_idx() {
                if action_idx < def.actions.len() {
                    let goap_action = &def.actions[action_idx];

                    // Check if step completed (preconditions for this action met its effects)
                    if plan_state.step_ticks >= goap_action.duration_ticks {
                        let has_more = plan_state.advance_step();
                        if !has_more {
                            // Plan complete — clear it so we replan next tick
                            plan_state.current_plan = None;
                            plan_state.current_step = 0;
                        }
                        // Try next action
                        if let Some(next_idx) = plan_state.current_action_idx() {
                            if next_idx < def.actions.len() {
                                resolve_intent(&def.actions[next_idx].intent, state, unit_idx)
                            } else {
                                IntentAction::Hold
                            }
                        } else {
                            IntentAction::Hold
                        }
                    } else {
                        plan_state.tick_step();
                        resolve_intent(&goap_action.intent, state, unit_idx)
                    }
                } else {
                    IntentAction::Hold
                }
            } else {
                IntentAction::Hold
            };

            // Track idle state for verification
            let is_idle = matches!(action, IntentAction::Hold)
                && plan_state.current_plan.is_none()
                && !is_casting;
            plan_state.track_idle(is_idle);

            intents.push(UnitIntent {
                unit_id,
                action,
            });
        }

        // Deconflict: prevent multiple units from healing/CC'ing the same target.
        // Second unit to target the same heal/CC switches to attack nearest enemy.
        deconflict_intents(&mut intents, state);

        // Debug verification
        #[cfg(debug_assertions)]
        {
            let violations = verify::verify_goap(self, state);
            for v in &violations {
                eprintln!("[GOAP verify] {:?}", v);
            }
        }

        intents
    }

    /// Check if a unit has a GOAP definition.
    pub fn has_def(&self, unit_id: u32) -> bool {
        self.defs.contains_key(&unit_id)
    }

    /// Get the set of unit IDs managed by GOAP.
    pub fn managed_units(&self) -> impl Iterator<Item = u32> + '_ {
        self.defs.keys().copied()
    }
}

/// Resolve an IntentTemplate to a concrete IntentAction using current SimState.
fn resolve_intent(template: &IntentTemplate, state: &SimState, unit_idx: usize) -> IntentAction {
    let unit = &state.units[unit_idx];

    match template {
        IntentTemplate::AttackTarget(target) => {
            match resolve_target(target, state, unit_idx) {
                Some(target_id) => {
                    let target_unit = state.units.iter().find(|u| u.id == target_id);
                    if let Some(t) = target_unit {
                        if distance(unit.position, t.position) <= unit.attack_range {
                            IntentAction::Attack { target_id }
                        } else {
                            // Move toward target to get in range
                            let pos = position_at_range(
                                unit.position,
                                t.position,
                                unit.attack_range * 0.9,
                            );
                            IntentAction::MoveTo { position: pos }
                        }
                    } else {
                        IntentAction::Hold
                    }
                }
                None => IntentAction::Hold,
            }
        }
        IntentTemplate::ChaseTarget(target) => {
            match resolve_target(target, state, unit_idx) {
                Some(target_id) => {
                    let target_unit = state.units.iter().find(|u| u.id == target_id);
                    if let Some(t) = target_unit {
                        let pos = move_towards(unit.position, t.position, unit.move_speed_per_sec * 0.1);
                        IntentAction::MoveTo { position: pos }
                    } else {
                        IntentAction::Hold
                    }
                }
                None => IntentAction::Hold,
            }
        }
        IntentTemplate::FleeTarget(target) => {
            match resolve_target(target, state, unit_idx) {
                Some(target_id) => {
                    let target_unit = state.units.iter().find(|u| u.id == target_id);
                    if let Some(t) = target_unit {
                        let pos = move_away(unit.position, t.position, unit.move_speed_per_sec * 0.1);
                        IntentAction::MoveTo { position: pos }
                    } else {
                        IntentAction::Hold
                    }
                }
                None => IntentAction::Hold,
            }
        }
        IntentTemplate::MaintainDistance(target, desired_range) => {
            match resolve_target(target, state, unit_idx) {
                Some(target_id) => {
                    let target_unit = state.units.iter().find(|u| u.id == target_id);
                    if let Some(t) = target_unit {
                        let pos = position_at_range(unit.position, t.position, *desired_range);
                        IntentAction::MoveTo { position: pos }
                    } else {
                        IntentAction::Hold
                    }
                }
                None => IntentAction::Hold,
            }
        }
        IntentTemplate::CastIfReady(slot, target) => {
            // Check if ability is off cooldown
            let ready = if *slot < unit.abilities.len() {
                unit.abilities[*slot].cooldown_remaining_ms == 0
            } else {
                false
            };
            if !ready {
                return IntentAction::Hold;
            }
            match resolve_target(target, state, unit_idx) {
                Some(target_id) => {
                    use crate::effects::AbilityTarget;
                    IntentAction::UseAbility {
                        ability_index: *slot,
                        target: AbilityTarget::Unit(target_id),
                    }
                }
                None => IntentAction::Hold,
            }
        }
        IntentTemplate::Hold => IntentAction::Hold,
    }
}

/// Resolve a behavior Target to a unit ID. Reuses logic from behavior interpreter.
fn resolve_target(target: &Target, state: &SimState, unit_idx: usize) -> Option<u32> {
    let unit = &state.units[unit_idx];
    match target {
        Target::Self_ => Some(unit.id),
        Target::NearestEnemy => nearest_enemy(state, unit),
        Target::NearestAlly => nearest_ally(state, unit),
        Target::LowestHpEnemy => lowest_hp_by(state, unit, false),
        Target::LowestHpAlly => lowest_hp_by(state, unit, true),
        Target::HighestDpsEnemy => {
            state.units.iter()
                .filter(|u| is_alive(u) && u.team != unit.team)
                .max_by_key(|u| u.total_damage_done)
                .map(|u| u.id)
        }
        Target::HighestThreatEnemy => {
            let pos = unit.position;
            state.units.iter()
                .filter(|u| is_alive(u) && u.team != unit.team)
                .max_by(|a, b| {
                    let ta = a.total_damage_done as f32 / distance(pos, a.position).max(0.1);
                    let tb = b.total_damage_done as f32 / distance(pos, b.position).max(0.1);
                    ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|u| u.id)
        }
        Target::CastingEnemy => {
            state.units.iter()
                .find(|u| is_alive(u) && u.team != unit.team && u.casting.is_some())
                .map(|u| u.id)
        }
        Target::EnemyAttacking(inner) => {
            let inner_id = resolve_target(inner, state, unit_idx)?;
            state.units.iter()
                .find(|u| {
                    is_alive(u) && u.team != unit.team
                        && u.casting.map_or(false, |c| c.target_id == inner_id)
                })
                .map(|u| u.id)
        }
        Target::Tagged(_) | Target::UnitId(_) => None,
    }
}

/// Post-pass: if multiple units are healing the same ally or CC'ing the same
/// enemy, reassign duplicates to attack their nearest enemy instead.
fn deconflict_intents(intents: &mut [UnitIntent], state: &SimState) {
    use std::collections::HashSet;

    // Track claimed heal targets (CastHeal) and CC targets (CastControl / UseAbility with heal hint)
    let mut claimed_heals: HashSet<u32> = HashSet::new();
    let mut claimed_cc: HashSet<u32> = HashSet::new();

    // First pass: identify first claimer for each target (by intent order = unit order)
    for intent in intents.iter() {
        match intent.action {
            IntentAction::CastHeal { target_id } => { claimed_heals.insert(target_id); }
            IntentAction::CastControl { target_id } => { claimed_cc.insert(target_id); }
            _ => {}
        }
    }

    // Second pass: if a target is already claimed, reassign duplicate to attack
    let mut active_heals: HashSet<u32> = HashSet::new();
    let mut active_cc: HashSet<u32> = HashSet::new();

    for intent in intents.iter_mut() {
        match intent.action {
            IntentAction::CastHeal { target_id } => {
                if !active_heals.insert(target_id) {
                    // Duplicate heal — switch to attack nearest enemy
                    if let Some(unit) = state.units.iter().find(|u| u.id == intent.unit_id) {
                        if let Some(enemy_id) = nearest_enemy(state, unit) {
                            intent.action = IntentAction::Attack { target_id: enemy_id };
                        }
                    }
                }
            }
            IntentAction::CastControl { target_id } => {
                if !active_cc.insert(target_id) {
                    if let Some(unit) = state.units.iter().find(|u| u.id == intent.unit_id) {
                        if let Some(enemy_id) = nearest_enemy(state, unit) {
                            intent.action = IntentAction::Attack { target_id: enemy_id };
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn nearest_enemy(state: &SimState, unit: &crate::sim::UnitState) -> Option<u32> {
    let pos = unit.position;
    state.units.iter()
        .filter(|u| is_alive(u) && u.team != unit.team)
        .min_by(|a, b| {
            distance(pos, a.position)
                .partial_cmp(&distance(pos, b.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}

fn nearest_ally(state: &SimState, unit: &crate::sim::UnitState) -> Option<u32> {
    let pos = unit.position;
    state.units.iter()
        .filter(|u| is_alive(u) && u.team == unit.team && u.id != unit.id)
        .min_by(|a, b| {
            distance(pos, a.position)
                .partial_cmp(&distance(pos, b.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}

fn lowest_hp_by(state: &SimState, unit: &crate::sim::UnitState, ally: bool) -> Option<u32> {
    state.units.iter()
        .filter(|u| {
            is_alive(u)
                && if ally {
                    u.team == unit.team && u.id != unit.id
                } else {
                    u.team != unit.team
                }
        })
        .min_by_key(|u| u.hp)
        .map(|u| u.id)
}
