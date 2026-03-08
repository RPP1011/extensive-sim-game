//! Action validity mask and action-to-intent conversion.

use super::{MAX_ABILITIES, NUM_ACTIONS};
use crate::ai::core::{
    distance, is_alive, move_towards, move_away, position_at_range,
    IntentAction, SimState, SimVec2,
};
use crate::ai::effects::AbilityTarget;

// V3 pointer action types
pub const ACTION_TYPE_ATTACK: usize = 0;
pub const ACTION_TYPE_MOVE: usize = 1;
pub const ACTION_TYPE_HOLD: usize = 2;
// 3..10 = ability 0..7

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
                // Check if there's a valid target
                let has_target = match slot.def.targeting {
                    crate::ai::effects::AbilityTargeting::TargetEnemy => has_enemies,
                    crate::ai::effects::AbilityTargeting::TargetAlly => {
                        state.units.iter().any(|u| u.team == unit.team && is_alive(u))
                    }
                    _ => true, // self-cast, AoE, ground target, etc.
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
/// Used for recording what any AI system decided in a uniform format.
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
            // Determine if target is nearest or weakest enemy
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
                0 // attack nearest
            } else if Some(*target_id) == weakest_id {
                1 // attack weakest
            } else {
                2 // attack focus (some other target)
            }
        }
        IntentAction::UseAbility { ability_index, .. } => {
            3 + ability_index.min(&7) // clamp to max 8 abilities
        }
        IntentAction::MoveTo { position } => {
            // Determine if moving toward or away from nearest enemy
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
                13 // Hold if no enemies
            }
        }
        IntentAction::Hold => 13,
        // Legacy cast variants — map to attack (they target an enemy)
        IntentAction::CastAbility { target_id }
        | IntentAction::CastHeal { target_id }
        | IntentAction::CastControl { target_id } => {
            // These are legacy; treat as attack on the target
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
            // Use focus target from search if available, else weakest
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
                        // Target enemy cluster centroid
                        if !enemies.is_empty() {
                            let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
                            let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
                            AbilityTarget::Position(SimVec2 { x: cx, y: cy })
                        } else {
                            AbilityTarget::None
                        }
                    }
                    _ => AbilityTarget::Unit(unit_id), // self-cast, SelfAoe, etc.
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

// ---------------------------------------------------------------------------
// V3: Pointer-based action space
// ---------------------------------------------------------------------------

/// Metadata about a token in the entity sequence, used to interpret pointer targets.
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// Type: 0=self, 1=enemy, 2=ally, 3=threat, 4=position
    pub type_id: usize,
    /// Unit ID (for entity tokens; None for threats/positions)
    pub unit_id: Option<u32>,
    /// World position of this token's referent
    pub position: SimVec2,
}

/// Convert a pointer action (action_type + target_token_idx) to an IntentAction.
pub fn pointer_action_to_intent(
    action_type: usize,
    target_token_idx: usize,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> IntentAction {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return IntentAction::Hold,
    };

    let step = unit.move_speed_per_sec * 0.1;

    match action_type {
        ACTION_TYPE_ATTACK => {
            if target_token_idx < token_infos.len() {
                let target = &token_infos[target_token_idx];
                if let Some(tid) = target.unit_id {
                    IntentAction::Attack { target_id: tid }
                } else {
                    IntentAction::Hold
                }
            } else {
                IntentAction::Hold
            }
        }
        ACTION_TYPE_MOVE => {
            if target_token_idx >= token_infos.len() {
                return IntentAction::Hold;
            }
            let target = &token_infos[target_token_idx];
            match target.type_id {
                1 => {
                    // Move toward enemy — position at attack range
                    let desired = position_at_range(
                        unit.position, target.position, unit.attack_range * 0.9,
                    );
                    let next = move_towards(unit.position, desired, step);
                    IntentAction::MoveTo { position: next }
                }
                2 => {
                    // Move toward ally
                    let next = move_towards(unit.position, target.position, step);
                    IntentAction::MoveTo { position: next }
                }
                3 => {
                    // Move AWAY from threat (zone avoidance!)
                    let away = move_away(unit.position, target.position, step);
                    IntentAction::MoveTo { position: away }
                }
                4 => {
                    // Move toward position token (cover, elevation, etc.)
                    let next = move_towards(unit.position, target.position, step);
                    IntentAction::MoveTo { position: next }
                }
                _ => IntentAction::Hold,
            }
        }
        ACTION_TYPE_HOLD => IntentAction::Hold,
        t @ 3..=10 => {
            let ability_index = t - 3;
            if target_token_idx >= token_infos.len() {
                return IntentAction::Hold;
            }
            if ability_index >= unit.abilities.len() {
                return IntentAction::Hold;
            }
            let target_info = &token_infos[target_token_idx];
            let ability_target = match target_info.type_id {
                1 | 2 => {
                    // Entity target
                    if let Some(tid) = target_info.unit_id {
                        AbilityTarget::Unit(tid)
                    } else {
                        AbilityTarget::None
                    }
                }
                0 => {
                    // Self-target
                    AbilityTarget::Unit(unit_id)
                }
                4 => {
                    // Position target (ground-target abilities)
                    AbilityTarget::Position(target_info.position)
                }
                3 => {
                    // Threat target — use threat position for ground-target
                    AbilityTarget::Position(target_info.position)
                }
                _ => AbilityTarget::None,
            };
            IntentAction::UseAbility { ability_index, target: ability_target }
        }
        _ => IntentAction::Hold,
    }
}

/// Build token info list from game state V2 data for pointer action interpretation.
pub fn build_token_infos(
    state: &SimState,
    unit_id: u32,
    entity_types: &[u8],
    positions_data: &[Vec<f32>],
) -> Vec<TokenInfo> {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return Vec::new(),
    };

    let mut infos = Vec::new();

    // Entity tokens: match ordering from extract_game_state_v2
    // Self first
    infos.push(TokenInfo {
        type_id: 0,
        unit_id: Some(unit_id),
        position: unit.position,
    });

    // Enemies sorted by distance
    let mut enemies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    enemies.sort_by(|a, b| {
        distance(unit.position, a.position)
            .partial_cmp(&distance(unit.position, b.position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for e in &enemies {
        infos.push(TokenInfo { type_id: 1, unit_id: Some(e.id), position: e.position });
    }

    // Allies sorted by HP%
    let mut allies: Vec<&crate::ai::core::UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit_id)
        .collect();
    allies.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for a in &allies {
        infos.push(TokenInfo { type_id: 2, unit_id: Some(a.id), position: a.position });
    }

    // Threat tokens: use zone/projectile positions
    for zone in &state.zones {
        if zone.source_team != unit.team {
            let zone_radius = crate::ai::core::ability_eval::area_max_radius_pub(&zone.area)
                .unwrap_or(2.0);
            if distance(unit.position, zone.position) < zone_radius + 3.0 {
                infos.push(TokenInfo { type_id: 3, unit_id: None, position: zone.position });
            }
        }
    }

    // Position tokens: reconstruct world positions from relative dx/dy
    for pos_feats in positions_data {
        if pos_feats.len() >= 2 {
            let world_pos = crate::ai::core::sim_vec2(
                unit.position.x + pos_feats[0] * 20.0,
                unit.position.y + pos_feats[1] * 20.0,
            );
            infos.push(TokenInfo { type_id: 4, unit_id: None, position: world_pos });
        }
    }

    infos
}
