use std::collections::{HashMap, VecDeque};

use crate::ai::core::{
    distance, move_towards, position_at_range, run_replay, sim_vec2, step, IntentAction,
    ReplayResult, SimState, Team, UnitIntent, UnitState,
};
use crate::ai::pathing::{clamp_step_to_walkable, has_line_of_sight, next_waypoint, GridNav};
use crate::ai::roles::Role;

use super::spatial::*;
use super::tactics::apply_strong_anti_stack_step;

pub(crate) fn horde_chokepoint_nav() -> GridNav {
    let mut nav = GridNav::new(-20.0, 20.0, -10.0, 10.0, 0.7);
    nav.add_block_rect(-0.8, 0.8, -9.5, 9.5);
    nav.carve_rect(-1.2, 1.2, -1.4, 1.4);
    nav.precompute_wall_proximity();
    nav
}

pub fn horde_chokepoint_state(seed: u64) -> SimState {
    let mut units = Vec::new();
    let hero_specs = vec![
        (1, -14.0, -1.2, 180, 16, 28, 32, Role::Tank),
        (2, -15.5, 0.1, 110, 19, 34, 26, Role::Dps),
        (3, -14.5, 1.3, 95, 10, 0, 30, Role::Healer),
    ];
    for (id, x, y, hp, atk, abil, heal, role) in hero_specs {
        let (ability_damage, heal_amount) = match role {
            Role::Healer => (0, heal),
            _ => (abil, 0),
        };
        units.push(UnitState {
            id,
            team: Team::Hero,
            hp,
            max_hp: hp,
            position: sim_vec2(x, y),
            move_speed_per_sec: 4.2,
            attack_damage: atk,
            attack_range: 1.4,
            attack_cooldown_ms: 650,
            attack_cast_time_ms: 250,
            cooldown_remaining_ms: 0,
            ability_damage,
            ability_range: 2.0,
            ability_cooldown_ms: 2_500,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount,
            heal_range: 2.8,
            heal_cooldown_ms: 2_100,
            heal_cast_time_ms: 380,
            heal_cooldown_remaining_ms: 0,
            control_range: if role == Role::Tank { 1.9 } else { 0.0 },
            control_duration_ms: if role == Role::Tank { 700 } else { 0 },
            control_cooldown_ms: if role == Role::Tank { 5_400 } else { 0 },
            control_cast_time_ms: if role == Role::Tank { 320 } else { 0 },
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        });
    }

    let mut next_id = 10_u32;
    for row in 0..3 {
        for col in 0..4 {
            let x = 11.5 + col as f32 * 1.0;
            let y = -2.0 + row as f32 * 2.0;
            units.push(UnitState {
                id: next_id,
                team: Team::Enemy,
                hp: 78,
                max_hp: 78,
                position: sim_vec2(x, y),
                move_speed_per_sec: 4.4,
                attack_damage: 12,
                attack_range: 1.2,
                attack_cooldown_ms: 700,
                attack_cast_time_ms: 260,
                cooldown_remaining_ms: 0,
                ability_damage: 14,
                ability_range: 1.8,
                ability_cooldown_ms: 2_800,
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
                abilities: Vec::new(),
                passives: Vec::new(),
                status_effects: Vec::new(),
                shield_hp: 0,
                resistance_tags: HashMap::new(),
                state_history: VecDeque::new(),
                channeling: None,
                resource: 0,
                max_resource: 0,
                resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
            });
            next_id += 1;
        }
    }
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

pub fn horde_chokepoint_hero_favored_state(seed: u64) -> SimState {
    let mut units = Vec::new();
    let hero_specs = vec![
        (1, -13.8, -1.2, 240, 20, 38, 36, Role::Tank),
        (2, -15.3, 0.0, 150, 26, 50, 0, Role::Dps),
        (3, -14.4, 1.3, 125, 12, 0, 44, Role::Healer),
        (4, -16.2, -0.2, 132, 22, 44, 0, Role::Dps),
    ];
    for (id, x, y, hp, atk, abil, heal, role) in hero_specs {
        let (ability_damage, heal_amount) = match role {
            Role::Healer => (0, heal),
            _ => (abil, 0),
        };
        units.push(UnitState {
            id,
            team: Team::Hero,
            hp,
            max_hp: hp,
            position: sim_vec2(x, y),
            move_speed_per_sec: 4.35,
            attack_damage: atk,
            attack_range: 1.5,
            attack_cooldown_ms: 620,
            attack_cast_time_ms: 230,
            cooldown_remaining_ms: 0,
            ability_damage,
            ability_range: 2.2,
            ability_cooldown_ms: 2_300,
            ability_cast_time_ms: 360,
            ability_cooldown_remaining_ms: 0,
            heal_amount,
            heal_range: 3.1,
            heal_cooldown_ms: 1_800,
            heal_cast_time_ms: 300,
            heal_cooldown_remaining_ms: 0,
            control_range: if role == Role::Tank || id == 4 {
                2.2
            } else {
                0.0
            },
            control_duration_ms: if role == Role::Tank || id == 4 {
                850
            } else {
                0
            },
            control_cooldown_ms: if role == Role::Tank || id == 4 {
                4_600
            } else {
                0
            },
            control_cast_time_ms: if role == Role::Tank || id == 4 {
                280
            } else {
                0
            },
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        });
    }

    let mut next_id = 20_u32;
    for row in 0..2 {
        for col in 0..4 {
            let x = 11.8 + col as f32 * 1.1;
            let y = -1.8 + row as f32 * 2.1;
            units.push(UnitState {
                id: next_id,
                team: Team::Enemy,
                hp: 68,
                max_hp: 68,
                position: sim_vec2(x, y),
                move_speed_per_sec: 4.3,
                attack_damage: 10,
                attack_range: 1.2,
                attack_cooldown_ms: 760,
                attack_cast_time_ms: 270,
                cooldown_remaining_ms: 0,
                ability_damage: 10,
                ability_range: 1.7,
                ability_cooldown_ms: 3_000,
                ability_cast_time_ms: 450,
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
                abilities: Vec::new(),
                passives: Vec::new(),
                status_effects: Vec::new(),
                shield_hp: 0,
                resistance_tags: HashMap::new(),
                state_history: VecDeque::new(),
                channeling: None,
                resource: 0,
                max_resource: 0,
                resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
            });
            next_id += 1;
        }
    }
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

pub fn build_environment_reactive_intents(
    state: &SimState,
    nav: &GridNav,
    dt_ms: u32,
) -> Vec<UnitIntent> {
    let mut intents = Vec::new();
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();

    for unit_id in ids {
        let Some(unit) = state.units.iter().find(|u| u.id == unit_id) else {
            continue;
        };

        if unit.heal_amount > 0 {
            if let Some(ally) = state
                .units
                .iter()
                .filter(|u| u.hp > 0 && u.team == unit.team)
                .min_by(|a, b| {
                    (a.hp as f32 / a.max_hp.max(1) as f32)
                        .partial_cmp(&(b.hp as f32 / b.max_hp.max(1) as f32))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let ally_hp = ally.hp as f32 / ally.max_hp.max(1) as f32;
                if ally_hp < 0.75 {
                    let d = distance(unit.position, ally.position);
                    let action = if unit.heal_cooldown_remaining_ms == 0 && d <= unit.heal_range {
                        IntentAction::CastHeal { target_id: ally.id }
                    } else {
                        let slope_cost = nav.slope_cost_at_pos(unit.position).max(0.1);
                        let max_step =
                            (unit.move_speed_per_sec * (dt_ms as f32 / 1000.0)) / slope_cost;
                        let next = move_towards(
                            unit.position,
                            next_waypoint(nav, unit.position, ally.position),
                            max_step,
                        );
                        let next = apply_strong_anti_stack_step(state, unit, next, max_step);
                        let next = clamp_step_to_walkable(nav, unit.position, next);
                        IntentAction::MoveTo { position: next }
                    };
                    intents.push(UnitIntent { unit_id, action });
                    continue;
                }
            }
        }

        let target = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team != unit.team)
            .min_by(|a, b| {
                let a_vis = has_line_of_sight(nav, unit.position, a.position);
                let b_vis = has_line_of_sight(nav, unit.position, b.position);
                let a_score = if a_vis { 0_i32 } else { 1_i32 };
                let b_score = if b_vis { 0_i32 } else { 1_i32 };
                a_score
                    .cmp(&b_score)
                    .then_with(|| {
                        distance(unit.position, a.position)
                            .partial_cmp(&distance(unit.position, b.position))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.id.cmp(&b.id))
            });
        let Some(target) = target else {
            intents.push(UnitIntent {
                unit_id,
                action: IntentAction::Hold,
            });
            continue;
        };

        let los_clear = has_line_of_sight(nav, unit.position, target.position);
        let d = distance(unit.position, target.position);
        let action = if unit.control_duration_ms > 0
            && unit.control_cooldown_remaining_ms == 0
            && d <= unit.control_range
            && target.control_remaining_ms == 0
            && los_clear
        {
            IntentAction::CastControl {
                target_id: target.id,
            }
        } else if unit.ability_damage > 0
            && unit.ability_cooldown_remaining_ms == 0
            && d <= unit.ability_range * 0.95
            && los_clear
        {
            IntentAction::CastAbility {
                target_id: target.id,
            }
        } else if d <= unit.attack_range && los_clear {
            IntentAction::Attack {
                target_id: target.id,
            }
        } else {
            let goal = position_at_range(unit.position, target.position, unit.attack_range * 0.9);
            let p = next_waypoint(nav, unit.position, goal);
            let slope_cost = nav.slope_cost_at_pos(unit.position).max(0.1);
            let max_step = (unit.move_speed_per_sec * (dt_ms as f32 / 1000.0)) / slope_cost;
            let next = choose_visibility_biased_step(state, nav, unit, p, max_step);
            let next = apply_strong_anti_stack_step(state, unit, next, max_step);
            let next = clamp_step_to_walkable(nav, unit.position, next);
            IntentAction::MoveTo { position: next }
        };
        intents.push(UnitIntent { unit_id, action });
    }
    intents
}

pub fn build_horde_chokepoint_script(
    seed: u64,
    ticks: u32,
    dt_ms: u32,
) -> (SimState, Vec<Vec<UnitIntent>>) {
    let nav = horde_chokepoint_nav();
    let initial = horde_chokepoint_state(seed);
    build_horde_script_from_initial(initial, nav, ticks, dt_ms)
}

pub fn build_horde_chokepoint_hero_favored_script(
    seed: u64,
    ticks: u32,
    dt_ms: u32,
) -> (SimState, Vec<Vec<UnitIntent>>) {
    let nav = horde_chokepoint_nav();
    let initial = horde_chokepoint_hero_favored_state(seed);
    build_horde_script_from_initial(initial, nav, ticks, dt_ms)
}

fn build_horde_script_from_initial(
    initial: SimState,
    nav: GridNav,
    ticks: u32,
    dt_ms: u32,
) -> (SimState, Vec<Vec<UnitIntent>>) {
    let mut pressure = EncounterPressureState::default();
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        let mut intents = build_environment_reactive_intents(&state, &nav, dt_ms);
        apply_encounter_pressure_tactics(&state, Some(&nav), &mut pressure, &mut intents, dt_ms);
        script.push(intents.clone());
        let (next, _) = step(state, &intents, dt_ms);
        state = next;
    }
    (initial, script)
}

pub fn run_horde_chokepoint_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let (initial, script) = build_horde_chokepoint_script(seed, ticks, dt_ms);
    run_replay(initial, &script, ticks, dt_ms)
}

pub fn run_horde_chokepoint_hero_favored_sample(seed: u64, ticks: u32, dt_ms: u32) -> ReplayResult {
    let (initial, script) = build_horde_chokepoint_hero_favored_script(seed, ticks, dt_ms);
    run_replay(initial, &script, ticks, dt_ms)
}

pub fn run_horde_chokepoint_hero_favored_hp_scaled_sample(
    seed: u64,
    ticks: u32,
    dt_ms: u32,
    hero_hp_scale: f32,
) -> ReplayResult {
    let nav = horde_chokepoint_nav();
    let mut initial = horde_chokepoint_hero_favored_state(seed);
    let hp_scale = hero_hp_scale.max(0.1);
    for unit in initial.units.iter_mut().filter(|u| u.team == Team::Hero) {
        let scaled_max = ((unit.max_hp as f32) * hp_scale).round() as i32;
        let clamped = scaled_max.max(1);
        unit.max_hp = clamped;
        unit.hp = clamped;
    }
    let (initial, script) = build_horde_script_from_initial(initial, nav, ticks, dt_ms);
    run_replay(initial, &script, ticks, dt_ms)
}
