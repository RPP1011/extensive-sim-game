use std::collections::HashMap;

use crate::ai::core::{
    distance, move_towards, sim_vec2, IntentAction,
    SimState, SimVec2, Team, UnitIntent, UnitState,
};
use crate::ai::pathing::{clamp_step_to_walkable, has_line_of_sight, next_waypoint, GridNav};

pub(super) const STRONG_ANTI_STACK_RADIUS: f32 = 1.75;
pub(super) const STRONG_ANTI_STACK_GAIN: f32 = 1.6;
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
    pub burst_until_by_team: HashMap<Team, u64>,
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
pub(super) struct EncounterPressureState {
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

pub(super) fn choose_visibility_biased_step(
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

    // Find nearest enemy for cover evaluation
    let nearest_enemy_pos = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .map(|u| (distance(unit.position, u.position), u.position))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, pos)| pos);

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
        // Bonus for positions with cover from nearest enemy
        let cover_term = if let Some(enemy_pos) = nearest_enemy_pos {
            (crate::ai::pathing::cover_factor(nav, stepped, enemy_pos) * 500.0) as i32
        } else {
            0
        };
        // Bonus for elevation advantage
        let elev_term = (nav.elevation_at_pos(stepped) * 200.0) as i32;
        let score = vis * 1000 - dist_term + cover_term + elev_term;
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

pub(super) fn apply_encounter_pressure_tactics(
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
