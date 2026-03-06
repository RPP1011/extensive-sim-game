use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ai::core::{
    distance, move_towards, position_at_range, run_replay, step, IntentAction, ReplayResult,
    SimEvent, SimState, SimVec2, Team, UnitIntent,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    let dist = crate::ai::core::distance(unit.position, target.position);
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

pub fn sample_phase2_party_state(seed: u64) -> SimState {
    let mut state = crate::ai::core::sample_duel_state(seed);
    state.units = vec![
        crate::ai::core::UnitState {
            id: 1,
            team: Team::Hero,
            hp: 145,
            max_hp: 145,
            position: crate::ai::core::sim_vec2(-2.0, -1.0),
            move_speed_per_sec: 3.8,
            attack_damage: 11,
            attack_range: 1.3,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 22,
            ability_range: 1.8,
            ability_cooldown_ms: 3_100,
            ability_cast_time_ms: 520,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
        crate::ai::core::UnitState {
            id: 2,
            team: Team::Hero,
            hp: 98,
            max_hp: 98,
            position: crate::ai::core::sim_vec2(-4.0, 0.2),
            move_speed_per_sec: 4.7,
            attack_damage: 14,
            attack_range: 1.5,
            attack_cooldown_ms: 580,
            attack_cast_time_ms: 240,
            cooldown_remaining_ms: 0,
            ability_damage: 28,
            ability_range: 2.1,
            ability_cooldown_ms: 2_600,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
        crate::ai::core::UnitState {
            id: 3,
            team: Team::Hero,
            hp: 92,
            max_hp: 92,
            position: crate::ai::core::sim_vec2(-5.5, 1.0),
            move_speed_per_sec: 4.2,
            attack_damage: 8,
            attack_range: 1.3,
            attack_cooldown_ms: 650,
            attack_cast_time_ms: 260,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 24,
            heal_range: 2.7,
            heal_cooldown_ms: 2_200,
            heal_cast_time_ms: 420,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
        crate::ai::core::UnitState {
            id: 4,
            team: Team::Enemy,
            hp: 142,
            max_hp: 142,
            position: crate::ai::core::sim_vec2(8.0, -1.0),
            move_speed_per_sec: 3.7,
            attack_damage: 10,
            attack_range: 1.2,
            attack_cooldown_ms: 720,
            attack_cast_time_ms: 320,
            cooldown_remaining_ms: 0,
            ability_damage: 21,
            ability_range: 1.7,
            ability_cooldown_ms: 3_000,
            ability_cast_time_ms: 500,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
        crate::ai::core::UnitState {
            id: 5,
            team: Team::Enemy,
            hp: 96,
            max_hp: 96,
            position: crate::ai::core::sim_vec2(10.0, 0.2),
            move_speed_per_sec: 4.8,
            attack_damage: 13,
            attack_range: 1.5,
            attack_cooldown_ms: 560,
            attack_cast_time_ms: 230,
            cooldown_remaining_ms: 0,
            ability_damage: 27,
            ability_range: 2.0,
            ability_cooldown_ms: 2_600,
            ability_cast_time_ms: 410,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
        crate::ai::core::UnitState {
            id: 6,
            team: Team::Enemy,
            hp: 94,
            max_hp: 94,
            position: crate::ai::core::sim_vec2(11.5, 1.0),
            move_speed_per_sec: 4.1,
            attack_damage: 8,
            attack_range: 1.3,
            attack_cooldown_ms: 660,
            attack_cast_time_ms: 270,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 23,
            heal_range: 2.6,
            heal_cooldown_ms: 2_200,
            heal_cast_time_ms: 430,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        },
    ];
    state.units.sort_by_key(|u| u.id);
    state
}

pub fn default_roles() -> HashMap<u32, Role> {
    HashMap::from([
        (1, Role::Tank),
        (2, Role::Dps),
        (3, Role::Healer),
        (4, Role::Tank),
        (5, Role::Dps),
        (6, Role::Healer),
    ])
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    roles: HashMap<u32, Role>,
) -> Vec<Vec<UnitIntent>> {
    let mut ai = RoleAiState::new(initial, roles);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let intents = generate_intents(&state, &mut ai, dt_ms);
        script.push(intents.clone());
        let (new_state, events) = step(state, &intents, dt_ms);
        ai.update_from_events(&events);
        state = new_state;
    }

    script
}

pub fn run_phase2_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let initial = sample_phase2_party_state(seed);
    let script = generate_scripted_intents(&initial, ticks, dt_ms, default_roles());
    run_replay(initial, &script, ticks, dt_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::FIXED_TICK_MS;

    fn run_phase2_with_seed(seed: u64) -> ReplayResult {
        run_phase2_sample(seed, 260, FIXED_TICK_MS)
    }

    #[test]
    fn phase2_is_deterministic() {
        let a = run_phase2_with_seed(17);
        let b = run_phase2_with_seed(17);
        assert_eq!(a.event_log_hash, b.event_log_hash);
        assert_eq!(a.final_state_hash, b.final_state_hash);
    }

    #[test]
    fn healers_perform_healing() {
        let result = run_phase2_with_seed(17);
        assert!(result.metrics.heals_started > 0);
        assert!(result.metrics.heals_completed > 0);
        assert!(!result.metrics.total_healing_by_unit.is_empty());
    }

    #[test]
    fn tanks_absorb_pressure_better_than_dps() {
        let result = run_phase2_with_seed(17);
        let taken = result
            .metrics
            .damage_taken_by_unit
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let hero_tank = *taken.get(&1).unwrap_or(&0);
        let hero_dps = *taken.get(&2).unwrap_or(&0);
        let enemy_tank = *taken.get(&4).unwrap_or(&0);
        let enemy_dps = *taken.get(&5).unwrap_or(&0);

        assert!(hero_tank >= hero_dps || enemy_tank >= enemy_dps);
    }

    #[test]
    fn phase2_competent_small_party_fight() {
        let result = run_phase2_with_seed(17);
        assert_eq!(result.metrics.invariant_violations, 0);
        assert!(result.metrics.casts_completed + result.metrics.heals_completed > 0);
    }

    #[test]
    fn phase2_regression_snapshot() {
        let result = run_phase2_with_seed(17);
        assert_eq!(result.event_log_hash, 0x0122_c661_c895_0a47);
        assert_eq!(result.final_state_hash, 0xde59_67a3_db01_1c82);
        assert_eq!(result.metrics.winner, None);
        assert_eq!(
            result.metrics.final_hp_by_unit,
            vec![(1, 36), (2, 84), (3, 51), (4, 0), (5, 0), (6, 47)]
        );
        assert_eq!(result.metrics.heals_completed, 11);
        assert_eq!(result.metrics.invariant_violations, 0);
    }

    #[test]
    fn phase2_multi_seed_metric_bands() {
        let seeds = [11_u64, 13, 17, 19, 23, 29];
        let mut hero_wins = 0_u32;
        let mut total_ttk_ticks = 0_u32;
        let mut fights_with_death = 0_u32;
        let mut total_heals = 0_u32;

        for seed in seeds {
            let result = run_phase2_with_seed(seed);
            if result.metrics.winner == Some(Team::Hero) {
                hero_wins += 1;
            }
            if let Some(ttk) = result.metrics.tick_to_first_death {
                total_ttk_ticks += ttk as u32;
                fights_with_death += 1;
            }
            total_heals += result.metrics.heals_completed;
            assert_eq!(result.metrics.invariant_violations, 0);
        }

        let hero_win_rate = hero_wins as f32 / 6.0;
        let avg_ttk = if fights_with_death == 0 {
            0.0
        } else {
            total_ttk_ticks as f32 / fights_with_death as f32
        };

        assert!(hero_win_rate >= 0.30 && hero_win_rate <= 1.00);
        assert!(avg_ttk > 8.0 && avg_ttk < 180.0);
        assert!(total_heals >= 10);
    }

    #[test]
    fn healer_targets_allies_only() {
        let initial = sample_phase2_party_state(31);
        let roles = default_roles();
        let mut ai = RoleAiState::new(&initial, roles);
        let mut state = initial;

        for _ in 0..140 {
            let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
            for intent in &intents {
                if let IntentAction::CastHeal { target_id } = intent.action {
                    let src = state.units.iter().find(|u| u.id == intent.unit_id).unwrap();
                    let dst = state.units.iter().find(|u| u.id == target_id).unwrap();
                    assert_eq!(src.team, dst.team);
                }
            }
            let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
            ai.update_from_events(&events);
            state = new_state;
        }
    }

    #[test]
    fn tank_gets_first_contact_often() {
        let seeds = [41_u64, 43, 47, 53, 59];
        let mut tank_first_hits = 0_u32;
        let mut checked = 0_u32;

        for seed in seeds {
            let initial = sample_phase2_party_state(seed);
            let roles = default_roles();
            let mut ai = RoleAiState::new(&initial, roles.clone());
            let mut state = initial;
            let mut first_source: Option<u32> = None;

            for _ in 0..90 {
                let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
                let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
                if first_source.is_none() {
                    first_source = events.iter().find_map(|event| match event {
                        SimEvent::DamageApplied { source_id, .. } => Some(*source_id),
                        _ => None,
                    });
                }
                ai.update_from_events(&events);
                state = new_state;
                if first_source.is_some() {
                    break;
                }
            }

            if let Some(source_id) = first_source {
                checked += 1;
                if roles.get(&source_id) == Some(&Role::Tank) {
                    tank_first_hits += 1;
                }
            }
        }

        assert!(checked > 0);
        let ratio = tank_first_hits as f32 / checked as f32;
        assert!((0.0..=1.0).contains(&ratio));
    }

    #[test]
    fn dps_does_not_retarget_too_frequently() {
        let initial = sample_phase2_party_state(73);
        let roles = default_roles();
        let mut ai = RoleAiState::new(&initial, roles.clone());
        let mut state = initial;
        let dps_ids = [2_u32, 5_u32];
        let mut last_target_by_dps: HashMap<u32, u32> = HashMap::new();
        let mut retargets = 0_u32;

        for _ in 0..160 {
            let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
            for intent in &intents {
                if !dps_ids.contains(&intent.unit_id) {
                    continue;
                }
                let maybe_target = match intent.action {
                    IntentAction::Attack { target_id } => Some(target_id),
                    IntentAction::CastAbility { target_id } => Some(target_id),
                    _ => None,
                };
                if let Some(target_id) = maybe_target {
                    if let Some(last_target) = last_target_by_dps.get(&intent.unit_id) {
                        if *last_target != target_id {
                            retargets += 1;
                        }
                    }
                    last_target_by_dps.insert(intent.unit_id, target_id);
                }
            }
            let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
            ai.update_from_events(&events);
            state = new_state;
        }

        assert!(retargets <= 24);
    }

    #[test]
    fn no_enemy_units_results_in_hold_actions() {
        let mut state = sample_phase2_party_state(61);
        state.units.retain(|u| u.team == Team::Hero);
        let mut ai = RoleAiState::new(&state, default_roles());
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        assert!(intents
            .iter()
            .all(|i| matches!(i.action, IntentAction::Hold)));
    }

    #[test]
    fn healer_with_no_alive_allies_does_not_cast_heal() {
        let mut state = sample_phase2_party_state(67);
        for unit in &mut state.units {
            if unit.id != 3 && unit.team == Team::Hero {
                unit.hp = 0;
            }
        }
        let mut ai = RoleAiState::new(&state, default_roles());
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        let healer_intent = intents.iter().find(|intent| intent.unit_id == 3).unwrap();
        assert!(!matches!(
            healer_intent.action,
            IntentAction::CastHeal { .. }
        ));
    }

    #[test]
    fn out_of_range_healer_prefers_move_to_heal_target() {
        let mut state = sample_phase2_party_state(71);
        {
            let healer = state.units.iter_mut().find(|u| u.id == 3).unwrap();
            healer.position = crate::ai::core::sim_vec2(-12.0, 0.0);
        }
        {
            let tank = state.units.iter_mut().find(|u| u.id == 1).unwrap();
            tank.hp = 25;
        }
        let mut ai = RoleAiState::new(&state, default_roles());
        let intents = generate_intents(&state, &mut ai, FIXED_TICK_MS);
        let healer_intent = intents.iter().find(|intent| intent.unit_id == 3).unwrap();
        assert!(matches!(healer_intent.action, IntentAction::MoveTo { .. }));
    }

    #[test]
    fn small_param_mutation_changes_hash() {
        let baseline = run_phase2_with_seed(17);
        let mut altered = sample_phase2_party_state(17);
        let enemy_dps = altered.units.iter_mut().find(|u| u.id == 5).unwrap();
        enemy_dps.attack_damage += 1;
        let script = generate_scripted_intents(&altered, 260, FIXED_TICK_MS, default_roles());
        let mutated = run_replay(altered, &script, 260, FIXED_TICK_MS);
        assert_ne!(baseline.event_log_hash, mutated.event_log_hash);
    }

    #[test]
    fn fuzz_invariants_hold_across_seed_sweep() {
        for seed in 80_u64..90_u64 {
            let result = run_phase2_with_seed(seed);
            assert_eq!(result.metrics.invariant_violations, 0);
            assert_eq!(result.metrics.dead_source_attack_intents, 0);
            assert_eq!(result.per_tick_state_hashes.len(), 260);
        }
    }
}
