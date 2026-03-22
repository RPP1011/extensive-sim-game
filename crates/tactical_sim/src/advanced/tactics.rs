use std::collections::HashMap;

use crate::sim::{
    distance, move_towards, position_at_range, run_replay, sim_vec2, step, IntentAction,
    ReplayResult, SimState, Team, UnitIntent, UnitState,
};
use crate::personality::{
    default_personalities, generate_scripted_intents, sample_phase5_party_state,
};
use crate::roles::{default_roles, Role};

use super::spatial::*;

fn archetypes_from_roles(roles: &HashMap<u32, Role>) -> HashMap<u32, Archetype> {
    roles
        .iter()
        .map(|(id, role)| {
            let a = match role {
                Role::Tank => Archetype::Bruiser,
                Role::Dps => Archetype::Caster,
                Role::Healer => Archetype::Healer,
            };
            (*id, a)
        })
        .collect()
}

fn alive_ids_by_team(state: &SimState, team: Team) -> Vec<u32> {
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn lowest_hp_enemy(state: &SimState, team: Team) -> Option<u32> {
    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != team)
        .min_by(|a, b| a.hp.cmp(&b.hp).then_with(|| a.id.cmp(&b.id)))
        .map(|u| u.id)
}

fn anti_stack_direction(unit_id: u32, ally_id: u32) -> (f32, f32) {
    let sx = if unit_id >= ally_id {
        1.0_f32
    } else {
        -1.0_f32
    };
    let sy = if (unit_id ^ ally_id) & 1 == 0 {
        0.45_f32
    } else {
        -0.45_f32
    };
    let len = (sx * sx + sy * sy).sqrt();
    (sx / len, sy / len)
}

fn crowd_repel_vector(state: &SimState, unit: &UnitState, radius: f32) -> (f32, f32, f32) {
    let mut repel_x = 0.0_f32;
    let mut repel_y = 0.0_f32;
    let mut crowd_score = 0.0_f32;
    for ally in state.units.iter().filter(|u| {
        u.hp > 0
            && u.team == unit.team
            && u.id != unit.id
            && distance(unit.position, u.position) < radius
    }) {
        let dx = unit.position.x - ally.position.x;
        let dy = unit.position.y - ally.position.y;
        let d = (dx * dx + dy * dy).sqrt();
        let pressure = ((radius - d.max(0.0)) / radius).clamp(0.0, 1.0).powf(1.8);
        let (nx, ny) = if d <= f32::EPSILON {
            anti_stack_direction(unit.id, ally.id)
        } else {
            (dx / d, dy / d)
        };
        repel_x += nx * pressure;
        repel_y += ny * pressure;
        crowd_score += pressure;
    }
    (repel_x, repel_y, crowd_score)
}

pub(super) fn apply_strong_anti_stack_step(
    state: &SimState,
    unit: &UnitState,
    desired_step_target: crate::sim::SimVec2,
    max_step: f32,
) -> crate::sim::SimVec2 {
    let (repel_x, repel_y, crowd_score) = crowd_repel_vector(state, unit, STRONG_ANTI_STACK_RADIUS);
    if crowd_score <= 0.01 {
        return desired_step_target;
    }
    let mag = (repel_x * repel_x + repel_y * repel_y).sqrt();
    if mag <= f32::EPSILON {
        return desired_step_target;
    }
    let push = max_step * STRONG_ANTI_STACK_GAIN * crowd_score.min(1.8);
    let pushed_target = sim_vec2(
        desired_step_target.x + (repel_x / mag) * push,
        desired_step_target.y + (repel_y / mag) * push,
    );
    move_towards(unit.position, pushed_target, max_step)
}

fn apply_spatial_overrides(
    state: &SimState,
    roles: &HashMap<u32, Role>,
    intents: &mut [UnitIntent],
    dt_ms: u32,
) {
    let healer_by_team: HashMap<Team, u32> = [Team::Hero, Team::Enemy]
        .iter()
        .filter_map(|team| {
            alive_ids_by_team(state, *team)
                .into_iter()
                .find(|id| roles.get(id) == Some(&Role::Healer))
                .map(|id| (*team, id))
        })
        .collect();

    for intent in intents.iter_mut() {
        let Some(unit) = state
            .units
            .iter()
            .find(|u| u.id == intent.unit_id && u.hp > 0)
        else {
            continue;
        };
        let role = *roles.get(&unit.id).unwrap_or(&Role::Dps);
        let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);

        // Formation lanes by role.
        let target_y = match role {
            Role::Tank => -1.0,
            Role::Dps => 0.2,
            Role::Healer => 1.0,
        };
        if (unit.position.y - target_y).abs() > 0.7 {
            let desired = crate::sim::sim_vec2(unit.position.x, target_y);
            let next = move_towards(unit.position, desired, max_step);
            intent.action = IntentAction::MoveTo { position: next };
            continue;
        }

        // Protect-healer bubble for tanks.
        if role == Role::Tank {
            if let Some(healer_id) = healer_by_team.get(&unit.team).copied() {
                if let Some(healer) = state.units.iter().find(|u| u.id == healer_id && u.hp > 0) {
                    let d = distance(unit.position, healer.position);
                    if d > 3.0 {
                        let desired = position_at_range(unit.position, healer.position, 2.2);
                        let next = move_towards(unit.position, desired, max_step);
                        intent.action = IntentAction::MoveTo { position: next };
                        continue;
                    }
                }
            }
        }

        // Strong anti-stack spacing from all nearby allies.
        let (repel_x, repel_y, crowd_score) =
            crowd_repel_vector(state, unit, STRONG_ANTI_STACK_RADIUS);
        if crowd_score > 0.2 {
            let mag = (repel_x * repel_x + repel_y * repel_y).sqrt();
            if mag > f32::EPSILON {
                let scatter_target = sim_vec2(
                    unit.position.x + (repel_x / mag) * max_step * STRONG_ANTI_STACK_GAIN,
                    unit.position.y + (repel_y / mag) * max_step * STRONG_ANTI_STACK_GAIN,
                );
                let next = move_towards(unit.position, scatter_target, max_step);
                intent.action = IntentAction::MoveTo { position: next };
                continue;
            }
        }
    }
}

fn shield_active(state_tick: u64, unit_id: u32, archetypes: &HashMap<u32, Archetype>) -> bool {
    archetypes.get(&unit_id) == Some(&Archetype::Bruiser)
        && (state_tick % 90 >= 35 && state_tick % 90 <= 55)
}

fn apply_tactical_rules(
    state: &SimState,
    archetypes: &HashMap<u32, Archetype>,
    intents: &mut [UnitIntent],
    dt_ms: u32,
) {
    for intent in intents.iter_mut() {
        let Some(unit) = state
            .units
            .iter()
            .find(|u| u.id == intent.unit_id && u.hp > 0)
        else {
            continue;
        };

        // Interrupt dangerous cast if possible.
        if unit.ability_damage > 0 && unit.ability_cooldown_remaining_ms == 0 {
            if let Some(casting_enemy) = state.units.iter().find(|u| {
                u.hp > 0
                    && u.team != unit.team
                    && u.casting.is_some()
                    && archetypes.get(&u.id) != Some(&Archetype::Healer)
            }) {
                if distance(unit.position, casting_enemy.position) <= unit.ability_range * 0.95 {
                    intent.action = IntentAction::CastAbility {
                        target_id: casting_enemy.id,
                    };
                    continue;
                }
            }
        }

        // Avoid hitting temporary reflect shield units.
        let current_target = match intent.action {
            IntentAction::Attack { target_id } => Some(target_id),
            IntentAction::CastAbility { target_id } => Some(target_id),
            _ => None,
        };
        if let Some(target_id) = current_target {
            if shield_active(state.tick, target_id, archetypes) {
                if let Some(replacement) = state
                    .units
                    .iter()
                    .filter(|u| {
                        u.hp > 0
                            && u.team != unit.team
                            && u.id != target_id
                            && !shield_active(state.tick, u.id, archetypes)
                    })
                    .min_by(|a, b| a.hp.cmp(&b.hp).then_with(|| a.id.cmp(&b.id)))
                    .map(|u| u.id)
                {
                    intent.action = match intent.action {
                        IntentAction::CastAbility { .. } => IntentAction::CastAbility {
                            target_id: replacement,
                        },
                        _ => IntentAction::Attack {
                            target_id: replacement,
                        },
                    };
                    continue;
                }
            }
        }

        // Switch to add/healer priority when exposed.
        if let Some(exposed_healer) = state
            .units
            .iter()
            .filter(|u| {
                u.hp > 0 && u.team != unit.team && archetypes.get(&u.id) == Some(&Archetype::Healer)
            })
            .min_by(|a, b| a.hp.cmp(&b.hp).then_with(|| a.id.cmp(&b.id)))
        {
            let hp_pct = exposed_healer.hp as f32 / exposed_healer.max_hp.max(1) as f32;
            if hp_pct <= 0.65 {
                let dist = distance(unit.position, exposed_healer.position);
                if unit.ability_damage > 0
                    && unit.ability_cooldown_remaining_ms == 0
                    && dist <= unit.ability_range * 0.95
                {
                    intent.action = IntentAction::CastAbility {
                        target_id: exposed_healer.id,
                    };
                } else if dist <= unit.attack_range {
                    intent.action = IntentAction::Attack {
                        target_id: exposed_healer.id,
                    };
                } else {
                    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let desired = position_at_range(
                        unit.position,
                        exposed_healer.position,
                        unit.attack_range * 0.9,
                    );
                    let next = move_towards(unit.position, desired, max_step);
                    intent.action = IntentAction::MoveTo { position: next };
                }
            }
        }
    }
}

fn apply_coordination(
    state: &SimState,
    roles: &HashMap<u32, Role>,
    coord: &mut Phase9CoordState,
    intents: &mut [UnitIntent],
    dt_ms: u32,
) {
    // Burst-window trigger from low-health enemy.
    for team in [Team::Hero, Team::Enemy] {
        if let Some(low_enemy) = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team != team)
            .find(|u| (u.hp as f32 / u.max_hp.max(1) as f32) <= 0.35)
        {
            let _ = low_enemy;
            coord.burst_until_by_team.insert(team, state.tick + 30);
        }
    }

    // Interrupt ownership: single owner per team.
    for team in [Team::Hero, Team::Enemy] {
        let interrupter = alive_ids_by_team(state, team)
            .into_iter()
            .filter(|id| roles.get(id) != Some(&Role::Healer))
            .min();
        let casting_enemy = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team != team && u.casting.is_some())
            .min_by(|a, b| a.id.cmp(&b.id))
            .map(|u| u.id);
        if let (Some(owner_id), Some(target_id)) = (interrupter, casting_enemy) {
            for intent in intents.iter_mut().filter(|i| i.unit_id == owner_id) {
                intent.action = IntentAction::CastAbility { target_id };
            }
        }
    }

    for intent in intents.iter_mut() {
        let Some(unit) = state
            .units
            .iter()
            .find(|u| u.id == intent.unit_id && u.hp > 0)
        else {
            continue;
        };

        // Emergency save protocol.
        if roles.get(&unit.id) == Some(&Role::Healer) && unit.heal_amount > 0 {
            if let Some(critical) = state
                .units
                .iter()
                .filter(|u| u.hp > 0 && u.team == unit.team)
                .min_by(|a, b| {
                    (a.hp as f32 / a.max_hp.max(1) as f32)
                        .partial_cmp(&(b.hp as f32 / b.max_hp.max(1) as f32))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let hp_pct = critical.hp as f32 / critical.max_hp.max(1) as f32;
                if hp_pct <= 0.28 {
                    let dist = distance(unit.position, critical.position);
                    if unit.heal_cooldown_remaining_ms == 0 && dist <= unit.heal_range {
                        intent.action = IntentAction::CastHeal {
                            target_id: critical.id,
                        };
                    } else {
                        let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                        let desired = position_at_range(
                            unit.position,
                            critical.position,
                            unit.heal_range * 0.9,
                        );
                        let next = move_towards(unit.position, desired, max_step);
                        intent.action = IntentAction::MoveTo { position: next };
                    }
                    continue;
                }
            }
        }

        // Burst protocol: non-healers commit to shared target while window is active.
        if roles.get(&unit.id) != Some(&Role::Healer) {
            let in_burst = coord
                .burst_until_by_team
                .get(&unit.team)
                .copied()
                .unwrap_or(0)
                >= state.tick;
            if in_burst {
                if let Some(target_id) = lowest_hp_enemy(state, unit.team) {
                    let Some(target) = state.units.iter().find(|u| u.id == target_id) else {
                        continue;
                    };
                    let dist = distance(unit.position, target.position);
                    if unit.ability_damage > 0
                        && unit.ability_cooldown_remaining_ms == 0
                        && dist <= unit.ability_range * 0.95
                    {
                        intent.action = IntentAction::CastAbility { target_id };
                    } else if dist <= unit.attack_range {
                        intent.action = IntentAction::Attack { target_id };
                    } else {
                        let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                        let desired = position_at_range(
                            unit.position,
                            target.position,
                            unit.attack_range * 0.9,
                        );
                        let next = move_towards(unit.position, desired, max_step);
                        intent.action = IntentAction::MoveTo { position: next };
                    }
                }
            }
        }
    }
}

fn build_script_with_overrides(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    enable_phase8: bool,
    enable_phase9: bool,
) -> Vec<Vec<UnitIntent>> {
    let roles = default_roles();
    let personalities = default_personalities();
    let (base_script, _) =
        generate_scripted_intents(initial, ticks, dt_ms, roles.clone(), personalities);
    let archetypes = archetypes_from_roles(&roles);
    let mut coord = Phase9CoordState::default();
    let mut pressure = EncounterPressureState::default();

    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    for tick in 0..ticks as usize {
        let mut intents = base_script.get(tick).cloned().unwrap_or_default();
        apply_spatial_overrides(&state, &roles, &mut intents, dt_ms);
        if enable_phase8 {
            apply_tactical_rules(&state, &archetypes, &mut intents, dt_ms);
        }
        if enable_phase9 {
            apply_coordination(&state, &roles, &mut coord, &mut intents, dt_ms);
        }
        apply_encounter_pressure_tactics(&state, None, &mut pressure, &mut intents, dt_ms);
        script.push(intents.clone());
        let (next, _) = step(state, &intents, dt_ms);
        state = next;
    }
    script
}

pub fn run_spatial_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let initial = sample_phase5_party_state(seed);
    let script = build_script_with_overrides(&initial, ticks, dt_ms, false, false);
    run_replay(initial, &script, ticks, dt_ms)
}

pub fn run_tactical_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let initial = sample_phase5_party_state(seed);
    let script = build_script_with_overrides(&initial, ticks, dt_ms, true, false);
    run_replay(initial, &script, ticks, dt_ms)
}

pub fn run_coordination_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let initial = sample_phase5_party_state(seed);
    let script = build_script_with_overrides(&initial, ticks, dt_ms, true, true);
    run_replay(initial, &script, ticks, dt_ms)
}
