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

// V4 dual-head action types
// Move directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=stay
pub const NUM_MOVE_DIRS: usize = 9;
// Combat types: 0=attack, 1=hold, 2..9=ability 0..7
pub const COMBAT_TYPE_ATTACK: usize = 0;
pub const COMBAT_TYPE_HOLD: usize = 1;
// 2..9 = ability 0..7

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

/// Convert an IntentAction to V3 pointer format (action_type, target_idx) given token infos.
/// Returns (action_type, target_idx) or None if the intent can't be mapped.
pub fn intent_to_v3_action(
    intent: &IntentAction,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> Option<(usize, usize)> {
    match intent {
        IntentAction::Attack { target_id } => {
            // Find the target in the token sequence
            let idx = token_infos.iter().position(|t| t.unit_id == Some(*target_id))?;
            Some((ACTION_TYPE_ATTACK, idx))
        }
        IntentAction::UseAbility { ability_index, target } => {
            if *ability_index > 7 { return Some((ACTION_TYPE_HOLD, 0)); }
            let action_type = 3 + ability_index;
            let target_idx = match target {
                AbilityTarget::Unit(tid) => {
                    token_infos.iter().position(|t| t.unit_id == Some(*tid))?
                }
                AbilityTarget::Position(pos) => {
                    // Find closest position/threat token to target position
                    token_infos.iter().enumerate()
                        .filter(|(_, t)| t.type_id >= 3) // positions and threats only
                        .min_by(|(_, a), (_, b)| {
                            distance(*pos, a.position)
                                .partial_cmp(&distance(*pos, b.position))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or_else(|| {
                            // Fall back to self token for self-targeted abilities
                            0
                        })
                }
                AbilityTarget::None => {
                    // Self-targeted ability → self token (index 0)
                    0
                }
            };
            Some((action_type, target_idx))
        }
        IntentAction::MoveTo { position } => {
            let unit = state.units.iter().find(|u| u.id == unit_id)?;
            let cur_dist_to_pos = distance(unit.position, *position);
            if cur_dist_to_pos < 0.5 {
                // Barely moving → hold
                return Some((ACTION_TYPE_HOLD, 0));
            }

            // Find the token whose position best matches the intent direction
            // For "move toward enemy", pick the closest enemy token to the target position
            // For "move away", pick the closest threat/enemy we're moving away from
            let move_dir = crate::ai::core::sim_vec2(
                position.x - unit.position.x,
                position.y - unit.position.y,
            );
            let move_len = (move_dir.x * move_dir.x + move_dir.y * move_dir.y).sqrt();
            if move_len < 0.01 {
                return Some((ACTION_TYPE_HOLD, 0));
            }

            // Check if we're moving toward or away from each token
            let mut best_toward: Option<(usize, f32)> = None;
            for (i, t) in token_infos.iter().enumerate() {
                if t.type_id == 0 { continue; } // skip self
                let to_token = crate::ai::core::sim_vec2(
                    t.position.x - unit.position.x,
                    t.position.y - unit.position.y,
                );
                let token_dist = (to_token.x * to_token.x + to_token.y * to_token.y).sqrt();
                if token_dist < 0.01 { continue; }

                // Cosine similarity between move direction and direction to token
                let cos_sim = (move_dir.x * to_token.x + move_dir.y * to_token.y)
                    / (move_len * token_dist);

                // For threats (type=3), pointer_action_to_intent moves AWAY,
                // so high cos_sim to a threat means we wouldn't pick it (we'd need negative cos_sim)
                let effective_cos = if t.type_id == 3 { -cos_sim } else { cos_sim };

                if effective_cos > 0.5 {
                    let score = effective_cos / (1.0 + token_dist * 0.01);
                    if best_toward.as_ref().map_or(true, |b| score > b.1) {
                        best_toward = Some((i, score));
                    }
                }
            }

            if let Some((idx, _)) = best_toward {
                Some((ACTION_TYPE_MOVE, idx))
            } else {
                // No good match — hold
                Some((ACTION_TYPE_HOLD, 0))
            }
        }
        IntentAction::Hold => Some((ACTION_TYPE_HOLD, 0)),
        // Legacy cast variants — map to attack on the target
        IntentAction::CastAbility { target_id }
        | IntentAction::CastHeal { target_id }
        | IntentAction::CastControl { target_id } => {
            let idx = token_infos.iter().position(|t| t.unit_id == Some(*target_id))?;
            Some((ACTION_TYPE_ATTACK, idx))
        }
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

// ---------------------------------------------------------------------------
// V4: Dual-head (directional movement + combat pointer)
// ---------------------------------------------------------------------------

/// Direction index to unit (dx, dy). 0=N (+y), CW.
pub fn move_dir_offset(dir: usize) -> (f32, f32) {
    match dir {
        0 => ( 0.0,  1.0), // N
        1 => ( 0.707, 0.707), // NE
        2 => ( 1.0,  0.0), // E
        3 => ( 0.707,-0.707), // SE
        4 => ( 0.0, -1.0), // S
        5 => (-0.707,-0.707), // SW
        6 => (-1.0,  0.0), // W
        7 => (-0.707, 0.707), // NW
        _ => ( 0.0,  0.0), // stay (8)
    }
}

/// Convert a movement direction + unit into a MoveTo intent (or Hold for stay).
pub fn move_dir_to_intent(dir: usize, unit_id: u32, state: &SimState) -> IntentAction {
    if dir >= 8 {
        return IntentAction::Hold;
    }
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return IntentAction::Hold,
    };
    let step = unit.move_speed_per_sec * 0.1;
    let (dx, dy) = move_dir_offset(dir);
    IntentAction::MoveTo {
        position: crate::ai::core::sim_vec2(
            unit.position.x + dx * step,
            unit.position.y + dy * step,
        ),
    }
}

/// Convert a V4 combat action (attack/hold/ability + target pointer) into an IntentAction.
pub fn combat_action_to_intent(
    combat_type: usize,
    target_idx: usize,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> IntentAction {
    match combat_type {
        COMBAT_TYPE_ATTACK => {
            if target_idx < token_infos.len() {
                if let Some(tid) = token_infos[target_idx].unit_id {
                    IntentAction::Attack { target_id: tid }
                } else {
                    IntentAction::Hold
                }
            } else {
                IntentAction::Hold
            }
        }
        COMBAT_TYPE_HOLD => IntentAction::Hold,
        t @ 2..=9 => {
            let ability_index = t - 2;
            let unit = match state.units.iter().find(|u| u.id == unit_id) {
                Some(u) => u,
                None => return IntentAction::Hold,
            };
            if ability_index >= unit.abilities.len() {
                return IntentAction::Hold;
            }
            if target_idx >= token_infos.len() {
                return IntentAction::Hold;
            }
            let target_info = &token_infos[target_idx];
            let ability_target = match target_info.type_id {
                1 | 2 => {
                    if let Some(tid) = target_info.unit_id {
                        AbilityTarget::Unit(tid)
                    } else {
                        AbilityTarget::None
                    }
                }
                0 => AbilityTarget::Unit(unit_id),
                3 | 4 => AbilityTarget::Position(target_info.position),
                _ => AbilityTarget::None,
            };
            IntentAction::UseAbility { ability_index, target: ability_target }
        }
        _ => IntentAction::Hold,
    }
}

/// Convert an oracle IntentAction to V4 format: (move_dir, combat_type, target_idx).
/// Returns None if the intent can't be mapped.
pub fn intent_to_v4_action(
    intent: &IntentAction,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> Option<(usize, usize, usize)> {
    match intent {
        IntentAction::Attack { target_id } => {
            let idx = token_infos.iter().position(|t| t.unit_id == Some(*target_id))?;
            // Attack implies moving toward the target
            let unit = state.units.iter().find(|u| u.id == unit_id)?;
            let target = state.units.iter().find(|u| u.id == *target_id)?;
            let move_dir = position_to_dir(unit.position, target.position);
            Some((move_dir, COMBAT_TYPE_ATTACK, idx))
        }
        IntentAction::UseAbility { ability_index, target } => {
            if *ability_index > 7 { return Some((8, COMBAT_TYPE_HOLD, 0)); }
            let combat_type = 2 + ability_index;
            let target_idx = match target {
                AbilityTarget::Unit(tid) => {
                    token_infos.iter().position(|t| t.unit_id == Some(*tid))?
                }
                AbilityTarget::Position(pos) => {
                    token_infos.iter().enumerate()
                        .filter(|(_, t)| t.type_id >= 3)
                        .min_by(|(_, a), (_, b)| {
                            distance(*pos, a.position)
                                .partial_cmp(&distance(*pos, b.position))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                }
                AbilityTarget::None => 0,
            };
            // Ability use: stay in place (let sim handle range check)
            Some((8, combat_type, target_idx))
        }
        IntentAction::MoveTo { position } => {
            let unit = state.units.iter().find(|u| u.id == unit_id)?;
            let move_dir = position_to_dir(unit.position, *position);
            // Moving without combat → hold
            Some((move_dir, COMBAT_TYPE_HOLD, 0))
        }
        IntentAction::Hold => Some((8, COMBAT_TYPE_HOLD, 0)),
        IntentAction::CastAbility { target_id }
        | IntentAction::CastHeal { target_id }
        | IntentAction::CastControl { target_id } => {
            let idx = token_infos.iter().position(|t| t.unit_id == Some(*target_id))?;
            Some((8, COMBAT_TYPE_ATTACK, idx))
        }
    }
}

/// Convert a position delta to one of 8 cardinal directions (or 8=stay).
fn position_to_dir(from: SimVec2, to: SimVec2) -> usize {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 0.5 {
        return 8; // stay
    }
    // atan2 → direction index. atan2(dy,dx): 0=E, pi/2=N
    let angle = dy.atan2(dx); // radians, -pi..pi
    // Map to 0..7: N=0 (pi/2), NE=1 (pi/4), E=2 (0), ...
    // Shift so N=0: subtract pi/2, negate to go CW
    // Actually simpler: use octant from angle
    let octant = ((angle + std::f32::consts::PI) / (std::f32::consts::PI / 4.0)) as usize;
    // angle=-pi→0, -3pi/4→1, ..., pi→8
    // Map: E=0→2, NE=pi/4→..., N=pi/2→...
    // Let's just use a direct mapping:
    // atan2 result and desired direction:
    //   N: angle ≈ pi/2  → dir 0
    //   NE: angle ≈ pi/4 → dir 1
    //   E: angle ≈ 0     → dir 2
    //   SE: angle ≈ -pi/4 → dir 3
    //   S: angle ≈ -pi/2  → dir 4
    //   SW: angle ≈ -3pi/4 → dir 5
    //   W: angle ≈ ±pi    → dir 6
    //   NW: angle ≈ 3pi/4  → dir 7
    let _ = octant; // discard the naive approach above
    // Normalized angle in [0, 2pi)
    let a = if angle < 0.0 { angle + 2.0 * std::f32::consts::PI } else { angle };
    // a=0→E, pi/2→N, pi→W, 3pi/2→S
    // We want: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    // Rotate: d = (pi/2 - a) mod 2pi, divide by pi/4
    let shifted = (std::f32::consts::FRAC_PI_2 - a + 2.0 * std::f32::consts::PI)
        % (2.0 * std::f32::consts::PI);
    let idx = ((shifted + std::f32::consts::PI / 8.0) / (std::f32::consts::PI / 4.0)) as usize;
    idx % 8
}
