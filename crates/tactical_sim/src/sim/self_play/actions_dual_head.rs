//! V4: Dual-head action space (directional movement + combat pointer).
//!
//! Split from `actions.rs` to keep files under 500 lines.

use crate::sim::{
    distance,
    IntentAction, SimState, SimVec2,
};
use crate::effects::AbilityTarget;

use super::actions_pointer::TokenInfo;

// V4 dual-head action types
// Move directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=stay
pub const NUM_MOVE_DIRS: usize = 9;
// Combat types: 0=attack, 1=hold, 2..9=ability 0..7
pub const COMBAT_TYPE_ATTACK: usize = 0;
pub const COMBAT_TYPE_HOLD: usize = 1;
// 2..9 = ability 0..7

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
        position: crate::sim::sim_vec2(
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
pub fn intent_to_v4_action(
    intent: &IntentAction,
    unit_id: u32,
    state: &SimState,
    token_infos: &[TokenInfo],
) -> Option<(usize, usize, usize)> {
    match intent {
        IntentAction::Attack { target_id } => {
            let idx = token_infos.iter().position(|t| t.unit_id == Some(*target_id))?;
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
            Some((8, combat_type, target_idx))
        }
        IntentAction::MoveTo { position } => {
            let unit = state.units.iter().find(|u| u.id == unit_id)?;
            let move_dir = position_to_dir(unit.position, *position);
            Some((move_dir, COMBAT_TYPE_HOLD, 0))
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => Some((8, COMBAT_TYPE_HOLD, 0)),
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
    let angle = dy.atan2(dx); // radians, -pi..pi
    // Normalized angle in [0, 2pi)
    let a = if angle < 0.0 { angle + 2.0 * std::f32::consts::PI } else { angle };
    // We want: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    let shifted = (std::f32::consts::FRAC_PI_2 - a + 2.0 * std::f32::consts::PI)
        % (2.0 * std::f32::consts::PI);
    let idx = ((shifted + std::f32::consts::PI / 8.0) / (std::f32::consts::PI / 4.0)) as usize;
    idx % 8
}
