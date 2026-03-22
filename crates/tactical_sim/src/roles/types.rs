use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::sim::{
    distance, move_towards, position_at_range, IntentAction,
    SimEvent, SimState, SimVec2, Team, UnitIntent,
};
use crate::phase::AiPhase;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    Tank,
    Dps,
    Healer,
}

#[derive(Debug, Clone, Copy)]
pub struct RoleProfile {
    pub preferred_range_min: f32,
    pub preferred_range_max: f32,
    pub leash_distance: f32,
    pub threat_sensitivity: f32,
    pub focus_bonus: f32,
}

#[derive(Debug, Clone)]
struct UnitMemory {
    anchor_position: SimVec2,
    sticky_target: Option<u32>,
    lock_ticks: u32,
}

#[derive(Debug, Clone)]
pub struct RoleAiState {
    pub role_by_unit: HashMap<u32, Role>,
    threat: HashMap<(u32, u32), f32>,
    memory: HashMap<u32, UnitMemory>,
}

impl RoleAiState {
    pub fn new(initial: &SimState, role_by_unit: HashMap<u32, Role>) -> Self {
        let memory = initial
            .units
            .iter()
            .map(|u| {
                (
                    u.id,
                    UnitMemory {
                        anchor_position: u.position,
                        sticky_target: None,
                        lock_ticks: 0,
                    },
                )
            })
            .collect();
        Self {
            role_by_unit,
            threat: HashMap::new(),
            memory,
        }
    }

    fn role_for(&self, unit_id: u32) -> Role {
        *self.role_by_unit.get(&unit_id).unwrap_or(&Role::Dps)
    }

    pub fn update_from_events(&mut self, events: &[SimEvent]) {
        self.threat.retain(|_, value| {
            *value *= 0.97;
            *value > 0.05
        });

        for event in events {
            match *event {
                SimEvent::DamageApplied {
                    source_id,
                    target_id,
                    amount,
                    ..
                } => {
                    let multiplier = match self.role_for(source_id) {
                        Role::Tank => 1.65,
                        Role::Dps => 1.0,
                        Role::Healer => 0.8,
                    };
                    *self.threat.entry((target_id, source_id)).or_insert(0.0) +=
                        amount as f32 * multiplier;
                }
                SimEvent::HealApplied {
                    source_id,
                    target_id,
                    amount,
                    ..
                } => {
                    // Light healing threat against enemies currently fighting the healed target.
                    for ((enemy_id, ally_id), threat_val) in self.threat.clone() {
                        if ally_id == target_id && threat_val > 0.0 {
                            *self.threat.entry((enemy_id, source_id)).or_insert(0.0) +=
                                amount as f32 * 0.35;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

impl AiPhase for RoleAiState {
    fn generate_intents(&mut self, state: &SimState, dt_ms: u32) -> Vec<UnitIntent> {
        generate_intents(state, self, dt_ms)
    }

    fn update_from_events(&mut self, events: &[SimEvent]) {
        self.update_from_events(events);
    }
}

pub fn role_profile(role: Role) -> RoleProfile {
    match role {
        Role::Tank => RoleProfile {
            preferred_range_min: 0.8,
            preferred_range_max: 1.6,
            leash_distance: 14.0,
            threat_sensitivity: 0.2,
            focus_bonus: 4.0,
        },
        Role::Dps => RoleProfile {
            preferred_range_min: 1.2,
            preferred_range_max: 2.2,
            leash_distance: 16.0,
            threat_sensitivity: 1.3,
            focus_bonus: 3.0,
        },
        Role::Healer => RoleProfile {
            preferred_range_min: 1.5,
            preferred_range_max: 3.5,
            leash_distance: 18.0,
            threat_sensitivity: 1.6,
            focus_bonus: 2.0,
        },
    }
}

pub fn generate_intents(state: &SimState, ai: &mut RoleAiState, dt_ms: u32) -> Vec<UnitIntent> {
    let mut intents = Vec::new();
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();

    for unit_id in ids {
        let Some(unit_idx) = state.units.iter().position(|u| u.id == unit_id) else {
            continue;
        };
        let unit = &state.units[unit_idx];
        let role = ai.role_for(unit_id);
        let profile = role_profile(role);

        let (anchor_position, sticky_target, lock_ticks) = {
            let memory = ai.memory.get(&unit_id).expect("memory initialized");
            (
                memory.anchor_position,
                memory.sticky_target,
                memory.lock_ticks,
            )
        };

        if distance(unit.position, anchor_position) > profile.leash_distance {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::MoveTo {
                    position: anchor_position,
                },
            });
            continue;
        }

        if role == Role::Healer {
            if let Some(intent) = healer_intent(state, ai, unit_id, dt_ms) {
                intents.push(intent);
                continue;
            }
        }

        let enemies = opposing_units(state, unit.team);
        if enemies.is_empty() {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        }

        let selected_target = select_target_for_role(state, ai, unit_id, role, &enemies);
        if let Some(target_id) = selected_target {
            if let Some(memory) = ai.memory.get_mut(&unit_id) {
                if sticky_target == Some(target_id) {
                    memory.sticky_target = sticky_target;
                    memory.lock_ticks = lock_ticks.saturating_sub(1);
                } else {
                    memory.sticky_target = Some(target_id);
                    memory.lock_ticks = 4;
                }
            }

            let action = choose_offensive_action(state, unit_id, target_id, role, profile, dt_ms);
            intents.push(UnitIntent { unit_id, action });
        } else {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
        }
    }

    intents
}

fn opposing_units(state: &SimState, team: Team) -> Vec<u32> {
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != team)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn allied_units(state: &SimState, team: Team) -> Vec<u32> {
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn healer_intent(
    state: &SimState,
    ai: &RoleAiState,
    healer_id: u32,
    dt_ms: u32,
) -> Option<UnitIntent> {
    let healer = state.units.iter().find(|u| u.id == healer_id)?;
    if healer.heal_amount <= 0 {
        return None;
    }

    let allies = allied_units(state, healer.team);
    let triage = allies
        .iter()
        .filter_map(|ally_id| {
            let ally = state.units.iter().find(|u| u.id == *ally_id)?;
            let missing_hp = ally.max_hp - ally.hp;
            if missing_hp <= 0 {
                return None;
            }
            let incoming = estimate_incoming_dps(state, ai, *ally_id);
            let ttd = ally.hp as f32 / incoming.max(0.5);
            Some((*ally_id, missing_hp, ttd))
        })
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let Some((ally_id, _missing_hp, ttd)) = triage else {
        return None;
    };

    let ally = state.units.iter().find(|u| u.id == ally_id)?;
    let dist = distance(healer.position, ally.position);
    let high_priority = ttd <= 7.0;

    if healer.heal_cooldown_remaining_ms == 0 && dist <= healer.heal_range && high_priority {
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::CastHeal { target_id: ally_id },
        });
    }

    if high_priority && dist > healer.heal_range {
        let max_step = healer.move_speed_per_sec * (dt_ms as f32 / 1000.0);
        let desired_pos =
            position_at_range(healer.position, ally.position, healer.heal_range * 0.85);
        let next_pos = move_towards(healer.position, desired_pos, max_step);
        return Some(UnitIntent {
            unit_id: healer_id,
            action: IntentAction::MoveTo { position: next_pos },
        });
    }

    None
}

fn estimate_incoming_dps(state: &SimState, ai: &RoleAiState, ally_id: u32) -> f32 {
    let Some(ally) = state.units.iter().find(|u| u.id == ally_id) else {
        return 0.0;
    };
    let enemies = opposing_units(state, ally.team);

    enemies
        .iter()
        .filter_map(|enemy_id| {
            let enemy = state.units.iter().find(|u| u.id == *enemy_id)?;
            let own = *ai.threat.get(&(*enemy_id, ally_id)).unwrap_or(&0.0);
            let max_vs_enemy = allied_units(state, ally.team)
                .iter()
                .map(|id| *ai.threat.get(&(*enemy_id, *id)).unwrap_or(&0.0))
                .fold(0.0_f32, f32::max);
            let pressure = if max_vs_enemy <= 0.0 {
                0.2
            } else {
                (own / max_vs_enemy).clamp(0.0, 1.0)
            };
            Some(enemy.attack_damage as f32 * pressure)
        })
        .sum::<f32>()
}

fn select_target_for_role(
    state: &SimState,
    ai: &RoleAiState,
    unit_id: u32,
    role: Role,
    enemies: &[u32],
) -> Option<u32> {
    let profile = role_profile(role);
    let memory = ai.memory.get(&unit_id)?;

    enemies.iter().copied().max_by(|a, b| {
        let sa = target_score(state, ai, unit_id, *a, role, profile, memory);
        let sb = target_score(state, ai, unit_id, *b, role, profile, memory);
        sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
    })
}

fn target_score(
    state: &SimState,
    ai: &RoleAiState,
    unit_id: u32,
    target_id: u32,
    role: Role,
    profile: RoleProfile,
    memory: &UnitMemory,
) -> f32 {
    let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
        return f32::MIN;
    };
    let Some(target) = state.units.iter().find(|u| u.id == target_id) else {
        return f32::MIN;
    };

    let dist = crate::sim::distance(unit.position, target.position);
    let range_center = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;
    let range_penalty = (dist - range_center).abs() * 2.0;
    let sticky_bonus = if memory.sticky_target == Some(target_id) {
        profile.focus_bonus
    } else {
        0.0
    };

    let base = (100 - target.hp).max(0) as f32 * 0.15 - range_penalty + sticky_bonus;

    let tank_modifier = if role == Role::Tank {
        allied_units(state, unit.team)
            .iter()
            .map(|ally_id| *ai.threat.get(&(target_id, *ally_id)).unwrap_or(&0.0))
            .sum::<f32>()
            * 0.04
    } else {
        0.0
    };

    let dps_threat_penalty = if role == Role::Dps {
        let own = *ai.threat.get(&(target_id, unit_id)).unwrap_or(&0.0);
        let maybe_tank_id = allied_units(state, unit.team)
            .into_iter()
            .find(|id| ai.role_for(*id) == Role::Tank);
        let tank_threat = maybe_tank_id
            .map(|tank_id| *ai.threat.get(&(target_id, tank_id)).unwrap_or(&0.0))
            .unwrap_or(0.0);
        ((own - tank_threat * 0.9).max(0.0)) * profile.threat_sensitivity
    } else {
        0.0
    };

    base + tank_modifier - dps_threat_penalty
}

fn choose_offensive_action(
    state: &SimState,
    unit_id: u32,
    target_id: u32,
    _role: Role,
    profile: RoleProfile,
    dt_ms: u32,
) -> IntentAction {
    let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
        return IntentAction::Hold;
    };
    let Some(target) = state.units.iter().find(|u| u.id == target_id) else {
        return IntentAction::Hold;
    };

    let dist = distance(unit.position, target.position);
    if unit.control_duration_ms > 0
        && unit.control_cooldown_remaining_ms == 0
        && dist <= unit.control_range
        && target.control_remaining_ms == 0
    {
        return IntentAction::CastControl { target_id };
    }
    if unit.ability_cooldown_remaining_ms == 0
        && unit.ability_damage > 0
        && dist <= unit.ability_range
        && target.hp > unit.attack_damage
    {
        return IntentAction::CastAbility { target_id };
    }

    if dist <= unit.attack_range {
        return IntentAction::Attack { target_id };
    }

    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
    let desired_distance = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;
    let desired_pos = position_at_range(unit.position, target.position, desired_distance);
    let next_pos = move_towards(unit.position, desired_pos, max_step);
    IntentAction::MoveTo { position: next_pos }
}
