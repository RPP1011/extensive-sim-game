use std::collections::HashMap;
use std::collections::VecDeque;

use contracts::*;
use crate::effects::StatusKind;

use super::types::*;
use super::unit_store::UnitStore;
use super::events::SimEvent;
use super::math::*;
use super::helpers::*;
use super::resolve::resolve_cast;
use super::intent::*;
use super::tick_systems::*;
use super::tick_world::*;
#[cfg(debug_assertions)]
use super::verify::verify_tick;
use crate::pathing::has_line_of_sight;

#[requires(state.units.iter().all(|u| u.position.x.is_finite() && u.position.y.is_finite()))]
#[requires(state.units.iter().all(|u| u.hp <= u.max_hp))]
#[ensures(ret.0.tick == old(state.tick) + 1)]
#[ensures(ret.0.units.len() >= old(state.units.len()))]
pub fn step(mut state: SimState, intents: &[UnitIntent], dt_ms: u32) -> (SimState, Vec<SimEvent>) {
    state.tick += 1;
    let tick = state.tick;
    let mut events = Vec::new();
    let intents_by_unit = collect_intents(intents);

    tick_hero_cooldowns(&mut state, dt_ms);
    tick_status_effects(&mut state, tick, dt_ms, &mut events);
    advance_projectiles(&mut state, tick, dt_ms, &mut events);
    tick_periodic_passives(&mut state, tick, dt_ms, &mut events);
    tick_zones(&mut state, tick, dt_ms, &mut events);
    tick_channels(&mut state, tick, dt_ms, &mut events);
    tick_tethers(&mut state, tick, dt_ms, &mut events);

    for unit in &mut state.units {
        if unit.hp > 0 {
            unit.state_history.push_back((state.tick as u32, unit.position, unit.hp));
            if unit.state_history.len() > 50 {
                unit.state_history.pop_front();
            }
        }
    }

    // Shuffle unit processing order to avoid first-mover advantage
    let mut unit_order: Vec<usize> = (0..state.units.len()).collect();
    for i in (1..unit_order.len()).rev() {
        let j = (next_rand_u32(&mut state) as usize) % (i + 1);
        unit_order.swap(i, j);
    }

    for &idx in &unit_order {
        if !is_alive(&state.units[idx]) {
            continue;
        }

        // Directed summons don't act independently — skip intent processing
        if state.units[idx].directed {
            continue;
        }

        if state.units[idx].cooldown_remaining_ms > 0 {
            state.units[idx].cooldown_remaining_ms =
                state.units[idx].cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].ability_cooldown_remaining_ms > 0 {
            state.units[idx].ability_cooldown_remaining_ms = state.units[idx]
                .ability_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].heal_cooldown_remaining_ms > 0 {
            state.units[idx].heal_cooldown_remaining_ms = state.units[idx]
                .heal_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].control_cooldown_remaining_ms > 0 {
            state.units[idx].control_cooldown_remaining_ms = state.units[idx]
                .control_cooldown_remaining_ms.saturating_sub(dt_ms);
        }
        if state.units[idx].control_remaining_ms > 0 {
            // Check if unit is casting an unstoppable ability — if so, ignore CC
            let is_unstoppable = state.units[idx].casting.as_ref().map_or(false, |cast| {
                if let CastKind::HeroAbility(ai) = cast.kind {
                    state.units[idx].abilities.get(ai).map_or(false, |s| s.def.unstoppable)
                } else {
                    false
                }
            });
            if is_unstoppable {
                // Clear the CC, cast continues
                state.units[idx].control_remaining_ms = 0;
            } else {
                state.units[idx].control_remaining_ms =
                    state.units[idx].control_remaining_ms.saturating_sub(dt_ms);
                state.units[idx].casting = None;
                events.push(SimEvent::UnitControlled {
                    tick, unit_id: state.units[idx].id,
                });
                continue;
            }
        }

        if state.units[idx].channeling.is_some() {
            continue;
        }

        if state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Banish)) {
            state.units[idx].casting = None;
            continue;
        }

        if let Some(fear_source) = state.units[idx].status_effects.iter().find_map(|s| {
            if let StatusKind::Fear { source_pos } = s.kind { Some(source_pos) } else { None }
        }) {
            state.units[idx].casting = None;
            let max_delta = state.units[idx].move_speed_per_sec * (dt_ms as f32 / 1000.0);
            let start = state.units[idx].position;
            let next = move_away(start, fear_source, max_delta);
            if distance(start, next) > f32::EPSILON {
                state.units[idx].position = next;
                // Refresh elevation after fear-forced movement
                if let Some(ref nav) = state.grid_nav {
                    state.units[idx].elevation = nav.elevation_at_pos(next);
                }
                events.push(SimEvent::Moved {
                    tick, unit_id: state.units[idx].id,
                    from_x100: to_x100(start.x), from_y100: to_x100(start.y),
                    to_x100: to_x100(next.x), to_y100: to_x100(next.y),
                });
            }
            continue;
        }

        let is_polymorphed = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Polymorph));
        if is_polymorphed {
            state.units[idx].casting = None;
        }

        if let Some(mut cast) = state.units[idx].casting {
            cast.remaining_ms = cast.remaining_ms.saturating_sub(dt_ms);
            if cast.remaining_ms == 0 {
                state.units[idx].casting = None;
                resolve_cast(idx, cast.target_id, cast.target_pos, cast.kind, tick, &mut state, &mut events);
            } else {
                state.units[idx].casting = Some(cast);
            }
            continue;
        }

        let mut intent = intents_by_unit
            .iter()
            .find(|(unit_id, _)| *unit_id == state.units[idx].id)
            .map(|(_, action)| *action)
            .unwrap_or(IntentAction::Hold);

        let is_rooted = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Root));
        let is_silenced = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Silence));
        let is_confused = state.units[idx].status_effects.iter().any(|s| matches!(s.kind, StatusKind::Confuse));

        if is_rooted {
            if let IntentAction::MoveTo { .. } = intent {
                intent = IntentAction::Hold;
            }
        }

        if is_silenced || is_polymorphed {
            match intent {
                IntentAction::CastAbility { .. } | IntentAction::CastControl { .. }
                | IntentAction::UseAbility { .. } => {
                    intent = IntentAction::Hold;
                }
                _ => {}
            }
        }

        if is_polymorphed {
            if let IntentAction::Attack { .. } = intent {
                intent = IntentAction::Hold;
            }
        }

        if let Some(taunter_id) = state.units[idx].status_effects.iter().find_map(|s| {
            if let StatusKind::Taunt { taunter_id } = s.kind { Some(taunter_id) } else { None }
        }) {
            match intent {
                IntentAction::Attack { .. } => {
                    intent = IntentAction::Attack { target_id: taunter_id };
                }
                _ => {}
            }
        }

        if is_confused {
            let alive: Vec<u32> = state.units.iter()
                .filter(|u| u.hp > 0 && u.id != state.units[idx].id)
                .map(|u| u.id)
                .collect();
            if !alive.is_empty() {
                let random_target = alive[(next_rand_u32(&mut state) as usize) % alive.len()];
                match intent {
                    IntentAction::Attack { .. } => {
                        intent = IntentAction::Attack { target_id: random_target };
                    }
                    IntentAction::CastAbility { .. } => {
                        intent = IntentAction::CastAbility { target_id: random_target };
                    }
                    _ => {}
                }
            }
        }

        match intent {
            IntentAction::Hold => {}
            IntentAction::MoveTo { position } => {
                move_towards_position(idx, position, tick, &mut state, dt_ms, &mut events);
                clamp_leash(idx, &mut state);
            }
            IntentAction::Attack { target_id } => {
                try_start_attack(idx, target_id, tick, dt_ms, &mut state, &mut events);
            }
            IntentAction::CastAbility { target_id } => {
                try_start_ability(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::CastHeal { target_id } => {
                try_start_heal(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::CastControl { target_id } => {
                try_start_control(idx, target_id, tick, &mut state, &mut events);
            }
            IntentAction::UseAbility { ability_index, target } => {
                try_start_hero_ability(idx, ability_index, target, tick, &mut state, &mut events);
            }
            IntentAction::Skulk { objective } => {
                // Move toward objective using concealment-aware pathfinding
                if let Some(ref nav) = state.grid_nav.clone() {
                    let unit_pos = state.units[idx].position;
                    let unit_team = state.units[idx].team;

                    // Compute enemy vision: cells visible to any enemy
                    let mut enemy_vision = std::collections::HashSet::new();
                    for enemy in state.units.iter().filter(|u| u.team != unit_team && u.hp > 0) {
                        let vis = nav.visible_cells_from(enemy.position, 8.0);
                        enemy_vision.extend(vis);
                    }

                    let current_cell = nav.cell_of(unit_pos);
                    let goal_cell = nav.cell_of(objective);
                    let next_cell = nav.skulk_next_cell(current_cell, goal_cell, &enemy_vision);
                    let next_pos = nav.cell_center(next_cell.0, next_cell.1);

                    move_towards_position(idx, next_pos, tick, &mut state, dt_ms, &mut events);
                } else {
                    // No nav grid — fall back to direct movement
                    move_towards_position(idx, objective, tick, &mut state, dt_ms, &mut events);
                }
            }
            IntentAction::Hide => {
                // Move to nearest cell that breaks LOS with all enemies
                if let Some(ref nav) = state.grid_nav.clone() {
                    let unit_pos = state.units[idx].position;
                    let unit_team = state.units[idx].team;
                    let unit_cell = nav.cell_of(unit_pos);

                    // Find enemy positions
                    let enemy_positions: Vec<SimVec2> = state.units.iter()
                        .filter(|u| u.team != unit_team && u.hp > 0)
                        .map(|u| u.position)
                        .collect();

                    if enemy_positions.is_empty() {
                        // No enemies — just hold
                    } else {
                        // Search nearby cells for one that breaks LOS with all enemies
                        let search_radius = 5i32;
                        let mut best_cell = unit_cell;
                        let mut best_score = f32::MIN;

                        for dx in -search_radius..=search_radius {
                            for dy in -search_radius..=search_radius {
                                let c = (unit_cell.0 + dx, unit_cell.1 + dy);
                                if nav.blocked.contains(&c) || !nav.in_bounds(c.0, c.1) {
                                    continue;
                                }
                                let cell_pos = nav.cell_center(c.0, c.1);

                                // Count how many enemies CAN'T see this cell
                                let hidden_from = enemy_positions.iter()
                                    .filter(|&&ep| !has_line_of_sight(nav, ep, cell_pos))
                                    .count();

                                let concealment = nav.concealment_at(c);
                                let dist = ((dx * dx + dy * dy) as f32).sqrt();

                                // Score: prefer hiding from more enemies, near walls, close by
                                let score = hidden_from as f32 * 3.0
                                    + concealment * 2.0
                                    - dist * 0.5;

                                if score > best_score {
                                    best_score = score;
                                    best_cell = c;
                                }
                            }
                        }

                        if best_cell != unit_cell {
                            let hide_pos = nav.cell_center(best_cell.0, best_cell.1);
                            move_towards_position(idx, hide_pos, tick, &mut state, dt_ms, &mut events);
                        }
                    }
                }
            }
        }
    }

    // Ensure dead units have no stale casting/channeling/control state.
    // Effects triggered during death processing (e.g. OnDeath passives) can
    // re-set these fields after the initial clear_dead_unit_state call.
    for unit in state.units.iter_mut() {
        if unit.hp <= 0 {
            unit.casting = None;
            unit.channeling = None;
            unit.control_remaining_ms = 0;
        }
    }

    // Clean up status effects on living units that reference dead units.
    // When a taunter/partner/protector/host dies, the status effect on the
    // victim must be removed to prevent stale references.
    let dead_ids: Vec<u32> = state.units.iter()
        .filter(|u| u.hp <= 0)
        .map(|u| u.id)
        .collect();
    if !dead_ids.is_empty() {
        for unit in state.units.iter_mut() {
            if unit.hp <= 0 { continue; }
            unit.status_effects.retain(|se| {
                match &se.kind {
                    StatusKind::Taunt { taunter_id } => !dead_ids.contains(taunter_id),
                    StatusKind::Duel { partner_id } => !dead_ids.contains(partner_id),
                    StatusKind::Link { partner_id, .. } => !dead_ids.contains(partner_id),
                    StatusKind::Redirect { protector_id, .. } => !dead_ids.contains(protector_id),
                    StatusKind::Attached { host_id } => !dead_ids.contains(host_id),
                    _ => true,
                }
            });
        }
    }

    // Runtime verification: check invariants after every tick in debug/test builds.
    #[cfg(debug_assertions)]
    {
        let report = verify_tick(&state);
        assert!(
            report.is_ok(),
            "Runtime verification failed at tick {}: {:?}",
            state.tick,
            report.violations,
        );
    }

    (state, events)
}

pub fn sample_duel_state(seed: u64) -> SimState {
    let mut units = vec![
        UnitState {
            id: 1,
            team: Team::Hero,
            hp: 100,
            max_hp: 100,
            position: sim_vec2(0.0, 0.0),
            move_speed_per_sec: 4.0,
            attack_damage: 14,
            attack_range: 1.4,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 20,
            ability_range: 1.6,
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
        },
        UnitState {
            id: 2,
            team: Team::Enemy,
            hp: 100,
            max_hp: 100,
            position: sim_vec2(8.0, 0.0),
            move_speed_per_sec: 3.5,
            attack_damage: 12,
            attack_range: 1.2,
            attack_cooldown_ms: 900,
            attack_cast_time_ms: 400,
            cooldown_remaining_ms: 0,
            ability_damage: 16,
            ability_range: 1.4,
            ability_cooldown_ms: 3_200,
            ability_cast_time_ms: 600,
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
        },
    ];
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: seed,
        units: UnitStore::new(units),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

pub fn sample_duel_script(ticks: u32) -> Vec<Vec<UnitIntent>> {
    let mut script = Vec::with_capacity(ticks as usize);
    for _ in 0..ticks {
        script.push(vec![
            UnitIntent {
                unit_id: 1,
                action: IntentAction::Attack { target_id: 2 },
            },
            UnitIntent {
                unit_id: 2,
                action: IntentAction::Attack { target_id: 1 },
            },
        ]);
    }
    script
}
