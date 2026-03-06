use std::collections::HashMap;

use crate::ai::core::{
    distance, move_towards, position_at_range, run_replay, sim_vec2, step, IntentAction,
    ReplayResult, SimState, SimVec2, Team, UnitIntent, UnitState,
};
use crate::ai::pathing::{clamp_step_to_walkable, has_line_of_sight, next_waypoint, GridNav};
use crate::ai::personality::{
    default_personalities, generate_scripted_intents, sample_phase5_party_state,
};
use crate::ai::roles::{default_roles, Role};

const STRONG_ANTI_STACK_RADIUS: f32 = 1.75;
const STRONG_ANTI_STACK_GAIN: f32 = 1.6;
const SWARM_SCAN_RADIUS: f32 = 3.4;
const ARC_NET_RADIUS: f32 = 3.0;
const STEAM_VENT_RADIUS: f32 = 3.5;
const ARC_NET_DURATION_TICKS: u64 = 16;
const ARC_NET_COOLDOWN_TICKS: u64 = 95;
const STEAM_VENT_DURATION_TICKS: u64 = 11;
const STEAM_VENT_COOLDOWN_TICKS: u64 = 140;
const VISIBILITY_SCAN_RANGE: f32 = 6.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    Bruiser,
    Caster,
    Healer,
}

#[derive(Debug, Clone)]
pub struct Phase9CoordState {
    burst_until_by_team: HashMap<Team, u64>,
}

impl Default for Phase9CoordState {
    fn default() -> Self {
        Self {
            burst_until_by_team: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TeamPressureToolsState {
    arc_net_until_tick: u64,
    arc_net_ready_at_tick: u64,
    arc_net_center: SimVec2,
    steam_vent_until_tick: u64,
    steam_vent_ready_at_tick: u64,
    steam_vent_center: SimVec2,
}

#[derive(Debug, Clone, Default)]
struct EncounterPressureState {
    hero: TeamPressureToolsState,
    enemy: TeamPressureToolsState,
}

impl EncounterPressureState {
    fn team(&self, team: Team) -> &TeamPressureToolsState {
        match team {
            Team::Hero => &self.hero,
            Team::Enemy => &self.enemy,
        }
    }

    fn team_mut(&mut self, team: Team) -> &mut TeamPressureToolsState {
        match team {
            Team::Hero => &mut self.hero,
            Team::Enemy => &mut self.enemy,
        }
    }
}

fn opposite(team: Team) -> Team {
    match team {
        Team::Hero => Team::Enemy,
        Team::Enemy => Team::Hero,
    }
}

fn visible_enemy_count_from_position(
    state: &SimState,
    nav: &GridNav,
    team: Team,
    from: SimVec2,
    scan_range: f32,
) -> u32 {
    state
        .units
        .iter()
        .filter(|u| {
            u.hp > 0
                && u.team != team
                && distance(from, u.position) <= scan_range
                && has_line_of_sight(nav, from, u.position)
        })
        .count() as u32
}

fn choose_visibility_biased_step(
    state: &SimState,
    nav: &GridNav,
    unit: &UnitState,
    waypoint: SimVec2,
    max_step: f32,
) -> SimVec2 {
    let base = move_towards(unit.position, waypoint, max_step);
    let mut candidates = vec![base];
    let dx = waypoint.x - unit.position.x;
    let dy = waypoint.y - unit.position.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len > f32::EPSILON {
        let nx = dx / len;
        let ny = dy / len;
        let lateral = max_step * 0.55;
        let left = sim_vec2(base.x - ny * lateral, base.y + nx * lateral);
        let right = sim_vec2(base.x + ny * lateral, base.y - nx * lateral);
        candidates.push(left);
        candidates.push(right);
    }

    let mut best = unit.position;
    let mut best_score = i32::MIN;
    for cand in candidates {
        let stepped = clamp_step_to_walkable(nav, unit.position, cand);
        let vis = visible_enemy_count_from_position(
            state,
            nav,
            unit.team,
            stepped,
            VISIBILITY_SCAN_RANGE,
        ) as i32;
        let dist_term = (distance(stepped, waypoint) * 100.0).round() as i32;
        let score = vis * 1000 - dist_term;
        if score > best_score {
            best_score = score;
            best = stepped;
        }
    }
    best
}

fn nearest_enemy_id_for_unit(state: &SimState, unit: &UnitState) -> Option<u32> {
    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .min_by(|a, b| {
            distance(unit.position, a.position)
                .partial_cmp(&distance(unit.position, b.position))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        })
        .map(|u| u.id)
}

fn frontline_anchor(state: &SimState, team: Team) -> Option<SimVec2> {
    let enemies = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != team)
        .collect::<Vec<_>>();
    if enemies.is_empty() {
        return None;
    }

    state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .min_by(|a, b| {
            let da = enemies
                .iter()
                .map(|e| distance(a.position, e.position))
                .fold(f32::INFINITY, f32::min);
            let db = enemies
                .iter()
                .map(|e| distance(b.position, e.position))
                .fold(f32::INFINITY, f32::min);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        })
        .map(|u| u.position)
}

fn local_pressure_profile(state: &SimState, team: Team, anchor: SimVec2) -> (f32, usize, bool) {
    let local_enemies = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != team && distance(u.position, anchor) <= SWARM_SCAN_RADIUS)
        .collect::<Vec<_>>();
    let local_allies = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team && distance(u.position, anchor) <= SWARM_SCAN_RADIUS)
        .count();
    let y_span = local_enemies
        .iter()
        .map(|u| u.position.y)
        .fold(None, |acc: Option<(f32, f32)>, y| match acc {
            None => Some((y, y)),
            Some((mn, mx)) => Some((mn.min(y), mx.max(y))),
        })
        .map(|(mn, mx)| mx - mn)
        .unwrap_or(99.0);
    let tight_cluster = local_enemies.len() >= 3 && y_span <= 2.8;
    let score = (local_enemies.len() as f32 / local_allies.max(1) as f32)
        + if tight_cluster { 0.9 } else { 0.0 };
    (score, local_enemies.len(), tight_cluster)
}

fn maybe_trigger_pressure_tools_for_team(
    state: &SimState,
    intents: &mut [UnitIntent],
    pressure: &mut EncounterPressureState,
    team: Team,
) {
    let team_alive = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .count();
    if team_alive == 0 {
        return;
    }
    let Some(anchor) = frontline_anchor(state, team) else {
        return;
    };
    let (pressure_score, local_enemies, tight_cluster) =
        local_pressure_profile(state, team, anchor);
    let tools = pressure.team_mut(team);

    if state.tick >= tools.arc_net_ready_at_tick && pressure_score >= 2.0 && local_enemies >= 3 {
        tools.arc_net_until_tick = state.tick + ARC_NET_DURATION_TICKS;
        tools.arc_net_ready_at_tick = state.tick + ARC_NET_COOLDOWN_TICKS;
        tools.arc_net_center = anchor;

        if let Some(caster) = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team == team && u.ability_damage > 0)
            .min_by(|a, b| {
                distance(a.position, anchor)
                    .partial_cmp(&distance(b.position, anchor))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            })
        {
            if let Some(target_id) = nearest_enemy_id_for_unit(state, caster) {
                if let Some(intent) = intents.iter_mut().find(|i| i.unit_id == caster.id) {
                    intent.action = IntentAction::CastAbility { target_id };
                }
            }
        }
    }

    if state.tick >= tools.steam_vent_ready_at_tick
        && team_alive >= 2
        && pressure_score >= 2.35
        && tight_cluster
        && local_enemies >= 4
    {
        tools.steam_vent_until_tick = state.tick + STEAM_VENT_DURATION_TICKS;
        tools.steam_vent_ready_at_tick = state.tick + STEAM_VENT_COOLDOWN_TICKS;
        tools.steam_vent_center = anchor;
    }
}

fn apply_pressure_tools_for_team(
    state: &SimState,
    nav: Option<&GridNav>,
    intents: &mut [UnitIntent],
    pressure: &EncounterPressureState,
    team: Team,
    dt_ms: u32,
) {
    let tools = pressure.team(team);
    let arc_net_active = state.tick <= tools.arc_net_until_tick;
    let steam_vent_active = state.tick <= tools.steam_vent_until_tick;
    if !arc_net_active && !steam_vent_active {
        return;
    }

    let impacted_team = opposite(team);
    for intent in intents.iter_mut() {
        let Some(unit) = state
            .units
            .iter()
            .find(|u| u.id == intent.unit_id && u.hp > 0 && u.team == impacted_team)
        else {
            continue;
        };

        if steam_vent_active
            && distance(unit.position, tools.steam_vent_center) <= STEAM_VENT_RADIUS
        {
            let dx = unit.position.x - tools.steam_vent_center.x;
            let dy = unit.position.y - tools.steam_vent_center.y;
            let len = (dx * dx + dy * dy).sqrt();
            let (nx, ny) = if len <= f32::EPSILON {
                (if unit.team == Team::Enemy { 1.0 } else { -1.0 }, 0.0_f32)
            } else {
                (dx / len, dy / len)
            };
            let retreat_goal = sim_vec2(unit.position.x + nx * 3.2, unit.position.y + ny * 2.0);
            let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0) * 1.25;
            let next = if let Some(nav) = nav {
                let toward = next_waypoint(nav, unit.position, retreat_goal);
                let stepped = move_towards(unit.position, toward, max_step);
                clamp_step_to_walkable(nav, unit.position, stepped)
            } else {
                move_towards(unit.position, retreat_goal, max_step)
            };
            intent.action = IntentAction::MoveTo { position: next };
            continue;
        }

        if arc_net_active && distance(unit.position, tools.arc_net_center) <= ARC_NET_RADIUS {
            match intent.action {
                IntentAction::MoveTo { position } => {
                    let max_step = unit.move_speed_per_sec * (dt_ms as f32 / 1000.0);
                    let slowed = (max_step * 0.3).max(0.06);
                    let next = if let Some(nav) = nav {
                        let stepped = move_towards(unit.position, position, slowed);
                        clamp_step_to_walkable(nav, unit.position, stepped)
                    } else {
                        move_towards(unit.position, position, slowed)
                    };
                    intent.action = IntentAction::MoveTo { position: next };
                }
                IntentAction::Attack { .. } | IntentAction::CastAbility { .. } => {
                    if (state.tick + unit.id as u64).is_multiple_of(2) {
                        intent.action = IntentAction::Hold;
                    }
                }
                _ => {}
            }
        }
    }
}

fn apply_encounter_pressure_tactics(
    state: &SimState,
    nav: Option<&GridNav>,
    pressure: &mut EncounterPressureState,
    intents: &mut [UnitIntent],
    dt_ms: u32,
) {
    for team in [Team::Hero, Team::Enemy] {
        maybe_trigger_pressure_tools_for_team(state, intents, pressure, team);
    }
    for team in [Team::Hero, Team::Enemy] {
        apply_pressure_tools_for_team(state, nav, intents, pressure, team, dt_ms);
    }
}

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

fn apply_strong_anti_stack_step(
    state: &SimState,
    unit: &UnitState,
    desired_step_target: crate::ai::core::SimVec2,
    max_step: f32,
) -> crate::ai::core::SimVec2 {
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
            let desired = crate::ai::core::sim_vec2(unit.position.x, target_y);
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

pub(crate) fn horde_chokepoint_nav() -> GridNav {
    let mut nav = GridNav::new(-20.0, 20.0, -10.0, 10.0, 0.7);
    nav.add_block_rect(-0.8, 0.8, -9.5, 9.5);
    nav.carve_rect(-1.2, 1.2, -1.4, 1.4);
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
            });
            next_id += 1;
        }
    }
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
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
            });
            next_id += 1;
        }
    }
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::FIXED_TICK_MS;

    #[test]
    fn phase7_is_deterministic() {
        let a = run_spatial_sample(37, 320, FIXED_TICK_MS);
        let b = run_spatial_sample(37, 320, FIXED_TICK_MS);
        assert_eq!(a.event_log_hash, b.event_log_hash);
        assert_eq!(a.final_state_hash, b.final_state_hash);
    }

    #[test]
    fn phase8_is_deterministic() {
        let a = run_tactical_sample(37, 320, FIXED_TICK_MS);
        let b = run_tactical_sample(37, 320, FIXED_TICK_MS);
        assert_eq!(a.event_log_hash, b.event_log_hash);
        assert_eq!(a.final_state_hash, b.final_state_hash);
    }

    #[test]
    fn phase9_improves_resolution_over_phase7() {
        let p7 = run_spatial_sample(31, 320, FIXED_TICK_MS);
        let p9 = run_coordination_sample(31, 320, FIXED_TICK_MS);
        let p7_enemy_alive = p7
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();
        let p9_enemy_alive = p9
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();
        assert!(p9_enemy_alive <= p7_enemy_alive);
        assert_eq!(p9.metrics.invariant_violations, 0);
    }

    #[test]
    fn horde_chokepoint_pathing_is_deterministic() {
        let a = run_horde_chokepoint_sample(101, 420, FIXED_TICK_MS);
        let b = run_horde_chokepoint_sample(101, 420, FIXED_TICK_MS);
        assert_eq!(a.event_log_hash, b.event_log_hash);
        assert_eq!(a.final_state_hash, b.final_state_hash);
        assert_eq!(a.metrics.invariant_violations, 0);
    }

    #[test]
    fn horde_chokepoint_hero_favored_is_hero_win() {
        let result = run_horde_chokepoint_hero_favored_sample(202, 420, FIXED_TICK_MS);
        assert_eq!(result.metrics.winner, Some(Team::Hero));
        assert_eq!(result.metrics.invariant_violations, 0);
    }
}
