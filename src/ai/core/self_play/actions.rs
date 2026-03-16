//! Action validity mask and action-to-intent conversion (V1/V2 flat action space).

use super::{MAX_ABILITIES, NUM_ACTIONS};
use crate::ai::core::{
    distance, is_alive, move_towards, move_away, position_at_range,
    IntentAction, SimState, SimVec2,
};
use crate::ai::effects::AbilityTarget;

// V3 pointer action types (re-exported for use by actions_pointer)
pub const ACTION_TYPE_ATTACK: usize = 0;
pub const ACTION_TYPE_MOVE: usize = 1;
pub const ACTION_TYPE_HOLD: usize = 2;
// 3..10 = ability 0..7

// Re-export V3/V4 types and functions
pub use super::actions_pointer::{
    TokenInfo, pointer_action_to_intent, intent_to_v3_action, build_token_infos,
};
pub use super::actions_dual_head::{
    NUM_MOVE_DIRS, COMBAT_TYPE_ATTACK, COMBAT_TYPE_HOLD,
    move_dir_offset, move_dir_to_intent, combat_action_to_intent, intent_to_v4_action,
};

/// Returns a mask of which actions are valid for this unit right now.
pub fn action_mask(state: &SimState, unit_id: u32) -> [bool; NUM_ACTIONS] {
    let mut mask = [false; NUM_ACTIONS];
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return mask,
    };

    let has_enemies = state.units.iter().any(|u| u.team != unit.team && is_alive(u));

    // Attack actions (0-2): need enemies
    mask[0] = has_enemies; // attack nearest
    mask[1] = has_enemies; // attack weakest
    mask[2] = has_enemies; // attack focus

    // Ability actions (3-10): need ability to exist and be ready
    for i in 0..MAX_ABILITIES {
        if let Some(slot) = unit.abilities.get(i) {
            let ready = slot.cooldown_remaining_ms == 0
                && (slot.def.resource_cost <= 0 || unit.resource >= slot.def.resource_cost);
            if ready {
                let has_target = match slot.def.targeting {
                    crate::ai::effects::AbilityTargeting::TargetEnemy => has_enemies,
                    crate::ai::effects::AbilityTargeting::TargetAlly => {
                        state.units.iter().any(|u| u.team == unit.team && is_alive(u))
                    }
                    _ => true,
                };
                mask[3 + i] = has_target;
            }
        }
    }

    // Move actions (11-12): always valid if enemies exist
    mask[11] = has_enemies; // move toward
    mask[12] = has_enemies; // move away

    // Hold (13): always valid
    mask[13] = true;

    mask
}

/// Convert a discrete action index to a concrete IntentAction.
pub fn action_to_intent(
    action: usize,
    unit_id: u32,
    state: &SimState,
) -> IntentAction {
    action_to_intent_with_focus(action, unit_id, state, None)
}

/// Reverse-map an IntentAction back to a discrete action index.
pub fn intent_to_action(
    intent: &IntentAction,
    unit_id: u32,
    state: &SimState,
) -> usize {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return 13, // Hold
    };

    match intent {
        IntentAction::Attack { target_id } => {
            let enemies: Vec<&crate::ai::core::UnitState> = state.units.iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .collect();
            let nearest_id = enemies.iter()
                .min_by(|a, b| distance(unit.position, a.position)
                    .partial_cmp(&distance(unit.position, b.position))
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| e.id);
            let weakest_id = enemies.iter()
                .min_by(|a, b| {
                    let ha = a.hp as f32 / a.max_hp.max(1) as f32;
                    let hb = b.hp as f32 / b.max_hp.max(1) as f32;
                    ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.id);

            if Some(*target_id) == nearest_id {
                0
            } else if Some(*target_id) == weakest_id {
                1
            } else {
                2
            }
        }
        IntentAction::UseAbility { ability_index, .. } => {
            3 + ability_index.min(&7)
        }
        IntentAction::MoveTo { position } => {
            let nearest_enemy = state.units.iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .min_by(|a, b| distance(unit.position, a.position)
                    .partial_cmp(&distance(unit.position, b.position))
                    .unwrap_or(std::cmp::Ordering::Equal));
            if let Some(enemy) = nearest_enemy {
                let cur_dist = distance(unit.position, enemy.position);
                let new_dist = distance(*position, enemy.position);
                if new_dist < cur_dist { 11 } else { 12 }
            } else {
                13
            }
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => 13,
        IntentAction::CastAbility { target_id }
        | IntentAction::CastHeal { target_id }
        | IntentAction::CastControl { target_id } => {
            let enemies: Vec<&crate::ai::core::UnitState> = state.units.iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .collect();
            let nearest_id = enemies.iter()
                .min_by(|a, b| distance(unit.position, a.position)
                    .partial_cmp(&distance(unit.position, b.position))
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| e.id);
            if Some(*target_id) == nearest_id { 0 } else { 2 }
        }
    }
}

/// Convert a discrete action index to an IntentAction, with optional focus target.
pub fn action_to_intent_with_focus(
    action: usize,
    unit_id: u32,
    state: &SimState,
    focus_target: Option<u32>,
) -> IntentAction {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return IntentAction::Hold,
    };

    let enemies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    let allies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();

    let nearest_enemy = enemies.iter().min_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_enemy = enemies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_ally = allies.iter()
        .filter(|a| a.id != unit_id)
        .min_by(|a, b| {
            let ha = a.hp as f32 / a.max_hp.max(1) as f32;
            let hb = b.hp as f32 / b.max_hp.max(1) as f32;
            ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
        });

    match action {
        0 => nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id }).unwrap_or(IntentAction::Hold),
        1 => weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }).unwrap_or(IntentAction::Hold),
        2 => {
            if let Some(ft) = focus_target {
                if enemies.iter().any(|e| e.id == ft) {
                    IntentAction::Attack { target_id: ft }
                } else {
                    weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }).unwrap_or(IntentAction::Hold)
                }
            } else {
                weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }).unwrap_or(IntentAction::Hold)
            }
        }
        a @ 3..=10 => {
            let ability_index = a - 3;
            if let Some(slot) = unit.abilities.get(ability_index) {
                let target = match slot.def.targeting {
                    crate::ai::effects::AbilityTargeting::TargetEnemy => {
                        nearest_enemy.map(|e| AbilityTarget::Unit(e.id)).unwrap_or(AbilityTarget::None)
                    }
                    crate::ai::effects::AbilityTargeting::TargetAlly => {
                        let heal_target = weakest_ally.map(|a| a.id).unwrap_or(unit_id);
                        AbilityTarget::Unit(heal_target)
                    }
                    crate::ai::effects::AbilityTargeting::GroundTarget => {
                        if !enemies.is_empty() {
                            let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
                            let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
                            AbilityTarget::Position(SimVec2 { x: cx, y: cy })
                        } else {
                            AbilityTarget::None
                        }
                    }
                    _ => AbilityTarget::Unit(unit_id),
                };
                IntentAction::UseAbility { ability_index, target }
            } else {
                IntentAction::Hold
            }
        }
        11 => {
            nearest_enemy.map(|e| {
                let desired = position_at_range(unit.position, e.position, unit.attack_range * 0.9);
                let next = move_towards(unit.position, desired, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: next }
            }).unwrap_or(IntentAction::Hold)
        }
        12 => {
            nearest_enemy.map(|e| {
                let away = move_away(unit.position, e.position, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: away }
            }).unwrap_or(IntentAction::Hold)
        }
        _ => IntentAction::Hold,
    }
}
