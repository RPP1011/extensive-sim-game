//! V3: Pointer-based action space.
//!
//! Split from `actions.rs` to keep files under 500 lines.

use crate::sim::{
    distance, is_alive, move_towards, move_away, position_at_range,
    IntentAction, SimState, SimVec2,
};
use crate::effects::AbilityTarget;

use super::actions::{ACTION_TYPE_ATTACK, ACTION_TYPE_MOVE, ACTION_TYPE_HOLD};

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
                    let desired = position_at_range(
                        unit.position, target.position, unit.attack_range * 0.9,
                    );
                    let next = move_towards(unit.position, desired, step);
                    IntentAction::MoveTo { position: next }
                }
                2 => {
                    let next = move_towards(unit.position, target.position, step);
                    IntentAction::MoveTo { position: next }
                }
                3 => {
                    let away = move_away(unit.position, target.position, step);
                    IntentAction::MoveTo { position: away }
                }
                4 => {
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
                    if let Some(tid) = target_info.unit_id {
                        AbilityTarget::Unit(tid)
                    } else {
                        AbilityTarget::None
                    }
                }
                0 => AbilityTarget::Unit(unit_id),
                4 => AbilityTarget::Position(target_info.position),
                3 => AbilityTarget::Position(target_info.position),
                _ => AbilityTarget::None,
            };
            IntentAction::UseAbility { ability_index, target: ability_target }
        }
        _ => IntentAction::Hold,
    }
}

/// Convert an IntentAction to V3 pointer format (action_type, target_idx) given token infos.
pub fn intent_to_v3_action(
    intent: &IntentAction,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> Option<(usize, usize)> {
    match intent {
        IntentAction::Attack { target_id } => {
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
            Some((action_type, target_idx))
        }
        IntentAction::MoveTo { position } => {
            let unit = state.units.iter().find(|u| u.id == unit_id)?;
            let cur_dist_to_pos = distance(unit.position, *position);
            if cur_dist_to_pos < 0.5 {
                return Some((ACTION_TYPE_HOLD, 0));
            }

            let move_dir = crate::sim::sim_vec2(
                position.x - unit.position.x,
                position.y - unit.position.y,
            );
            let move_len = (move_dir.x * move_dir.x + move_dir.y * move_dir.y).sqrt();
            if move_len < 0.01 {
                return Some((ACTION_TYPE_HOLD, 0));
            }

            let mut best_toward: Option<(usize, f32)> = None;
            for (i, t) in token_infos.iter().enumerate() {
                if t.type_id == 0 { continue; }
                let to_token = crate::sim::sim_vec2(
                    t.position.x - unit.position.x,
                    t.position.y - unit.position.y,
                );
                let token_dist = (to_token.x * to_token.x + to_token.y * to_token.y).sqrt();
                if token_dist < 0.01 { continue; }

                let cos_sim = (move_dir.x * to_token.x + move_dir.y * to_token.y)
                    / (move_len * token_dist);

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
                Some((ACTION_TYPE_HOLD, 0))
            }
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => Some((ACTION_TYPE_HOLD, 0)),
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
    _entity_types: &[u8],
    positions_data: &[Vec<f32>],
) -> Vec<TokenInfo> {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return Vec::new(),
    };

    let mut infos = Vec::new();

    // Self first
    infos.push(TokenInfo {
        type_id: 0,
        unit_id: Some(unit_id),
        position: unit.position,
    });

    // Enemies sorted by distance
    let mut enemies: Vec<&crate::sim::UnitState> = state.units.iter()
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
    let mut allies: Vec<&crate::sim::UnitState> = state.units.iter()
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

    // Threat tokens
    for zone in &state.zones {
        if zone.source_team != unit.team {
            let zone_radius = crate::sim::ability_eval::area_max_radius_pub(&zone.area)
                .unwrap_or(2.0);
            if distance(unit.position, zone.position) < zone_radius + 3.0 {
                infos.push(TokenInfo { type_id: 3, unit_id: None, position: zone.position });
            }
        }
    }

    // Position tokens
    for pos_feats in positions_data {
        if pos_feats.len() >= 2 {
            let world_pos = crate::sim::sim_vec2(
                unit.position.x + pos_feats[0] * 20.0,
                unit.position.y + pos_feats[1] * 20.0,
            );
            infos.push(TokenInfo { type_id: 4, unit_id: None, position: world_pos });
        }
    }

    infos
}
