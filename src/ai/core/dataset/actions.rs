//! Action classification enums and helpers.

use serde::{Deserialize, Serialize};

use super::super::{distance, is_alive, IntentAction, SimState};

// ---------------------------------------------------------------------------
// Action classes — abstract oracle actions into learnable categories
// ---------------------------------------------------------------------------

/// 10 discrete action classes the student model predicts (legacy full model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionClass {
    AttackNearest = 0,
    AttackWeakest = 1,
    UseDamageAbility = 2,
    UseHealAbility = 3,
    UseCcAbility = 4,
    UseDefenseAbility = 5,
    UseUtilityAbility = 6,
    MoveToward = 7,
    MoveAway = 8,
    Hold = 9,
}

impl ActionClass {
    pub fn count() -> usize {
        10
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::AttackNearest,
            1 => Self::AttackWeakest,
            2 => Self::UseDamageAbility,
            3 => Self::UseHealAbility,
            4 => Self::UseCcAbility,
            5 => Self::UseDefenseAbility,
            6 => Self::UseUtilityAbility,
            7 => Self::MoveToward,
            8 => Self::MoveAway,
            _ => Self::Hold,
        }
    }
}

/// 5-class combat action for the simplified student model.
/// Ability decisions are handled by frozen ability evaluators, so the
/// student only needs to decide between attack/move/hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombatActionClass {
    AttackNearest = 0,
    AttackWeakest = 1,
    MoveToward = 2,
    MoveAway = 3,
    Hold = 4,
}

impl CombatActionClass {
    pub fn count() -> usize {
        5
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::AttackNearest,
            1 => Self::AttackWeakest,
            2 => Self::MoveToward,
            3 => Self::MoveAway,
            _ => Self::Hold,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::AttackNearest => "AttackNearest",
            Self::AttackWeakest => "AttackWeakest",
            Self::MoveToward => "MoveToward",
            Self::MoveAway => "MoveAway",
            Self::Hold => "Hold",
        }
    }
}

/// Classify an oracle IntentAction into a CombatActionClass (5-class).
/// Returns None if the action is an ability (handled by ability evaluators).
pub fn classify_combat_action(action: &IntentAction, unit_id: u32, state: &SimState) -> Option<CombatActionClass> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;

    match action {
        IntentAction::Attack { target_id } => {
            let nearest = nearest_enemy_id(state, unit_id);
            let weakest = weakest_enemy_id(state, unit_id);
            if nearest == Some(*target_id) {
                Some(CombatActionClass::AttackNearest)
            } else if weakest == Some(*target_id) {
                Some(CombatActionClass::AttackWeakest)
            } else {
                Some(CombatActionClass::AttackNearest)
            }
        }
        // All ability actions → None (handled by frozen ability evaluators)
        IntentAction::CastAbility { .. }
        | IntentAction::CastHeal { .. }
        | IntentAction::CastControl { .. }
        | IntentAction::UseAbility { .. } => None,
        IntentAction::MoveTo { position } => {
            let nearest_enemy_pos = state
                .units
                .iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .min_by(|a, b| {
                    distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|u| u.position);

            if let Some(ep) = nearest_enemy_pos {
                let current_dist = distance(unit.position, ep);
                let new_dist = distance(*position, ep);
                if new_dist < current_dist - 0.1 {
                    Some(CombatActionClass::MoveToward)
                } else {
                    Some(CombatActionClass::MoveAway)
                }
            } else {
                Some(CombatActionClass::Hold)
            }
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => Some(CombatActionClass::Hold),
    }
}

/// Classify an oracle IntentAction into an ActionClass given the game state.
pub fn classify_action(action: &IntentAction, unit_id: u32, state: &SimState) -> ActionClass {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return ActionClass::Hold,
    };

    match action {
        IntentAction::Attack { target_id } => {
            let nearest = nearest_enemy_id(state, unit_id);
            let weakest = weakest_enemy_id(state, unit_id);
            if nearest == Some(*target_id) {
                ActionClass::AttackNearest
            } else if weakest == Some(*target_id) {
                ActionClass::AttackWeakest
            } else {
                ActionClass::AttackNearest // fallback
            }
        }
        IntentAction::CastAbility { .. } => ActionClass::UseDamageAbility,
        IntentAction::CastHeal { .. } => ActionClass::UseHealAbility,
        IntentAction::CastControl { .. } => ActionClass::UseCcAbility,
        IntentAction::UseAbility { ability_index, .. } => {
            if let Some(slot) = unit.abilities.get(*ability_index) {
                match slot.def.ai_hint.as_str() {
                    "damage" => ActionClass::UseDamageAbility,
                    "heal" => ActionClass::UseHealAbility,
                    "crowd_control" => ActionClass::UseCcAbility,
                    "defense" => ActionClass::UseDefenseAbility,
                    "utility" => ActionClass::UseUtilityAbility,
                    _ => ActionClass::UseDamageAbility,
                }
            } else {
                ActionClass::UseDamageAbility
            }
        }
        IntentAction::MoveTo { position } => {
            let nearest_enemy_pos = state
                .units
                .iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .min_by(|a, b| {
                    distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|u| u.position);

            if let Some(ep) = nearest_enemy_pos {
                let current_dist = distance(unit.position, ep);
                let new_dist = distance(*position, ep);
                if new_dist < current_dist - 0.1 {
                    ActionClass::MoveToward
                } else {
                    ActionClass::MoveAway
                }
            } else {
                ActionClass::Hold
            }
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => ActionClass::Hold,
    }
}

/// Classify an oracle IntentAction into the 14-class self-play action index.
/// Returns None if the action can't be mapped.
pub fn classify_action_raw(action: &IntentAction, unit_id: u32, state: &SimState) -> Option<usize> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;

    Some(match action {
        IntentAction::Attack { target_id } => {
            let nearest = nearest_enemy_id(state, unit_id);
            let weakest = weakest_enemy_id(state, unit_id);
            if nearest == Some(*target_id) {
                0 // AttackNearest
            } else if weakest == Some(*target_id) {
                1 // AttackWeakest
            } else {
                2 // AttackFocus
            }
        }
        IntentAction::UseAbility { ability_index, .. } => {
            3 + ability_index.min(&7) // Abi0..Abi7
        }
        IntentAction::CastAbility { .. } => 3,  // map legacy to Abi0
        IntentAction::CastHeal { .. } => 4,     // map legacy to Abi1
        IntentAction::CastControl { .. } => 5,  // map legacy to Abi2
        IntentAction::MoveTo { position } => {
            let nearest_enemy_pos = state.units.iter()
                .filter(|u| u.team != unit.team && is_alive(u))
                .min_by(|a, b| {
                    distance(unit.position, a.position)
                        .partial_cmp(&distance(unit.position, b.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|u| u.position);

            if let Some(ep) = nearest_enemy_pos {
                let cur = distance(unit.position, ep);
                let new = distance(*position, ep);
                if new < cur - 0.1 { 11 } else { 12 } // MoveToward / MoveAway
            } else {
                13 // Hold
            }
        }
        IntentAction::Skulk { .. } | IntentAction::Hide | IntentAction::Hold => 13,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(super) fn nearest_enemy_id(state: &SimState, unit_id: u32) -> Option<u32> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    state
        .units
        .iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .min_by(|a, b| {
            distance(unit.position, a.position)
                .partial_cmp(&distance(unit.position, b.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}

pub(super) fn weakest_enemy_id(state: &SimState, unit_id: u32) -> Option<u32> {
    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    state
        .units
        .iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .min_by(|a, b| {
            let ha = a.hp as f32 / a.max_hp.max(1) as f32;
            let hb = b.hp as f32 / b.max_hp.max(1) as f32;
            ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}
