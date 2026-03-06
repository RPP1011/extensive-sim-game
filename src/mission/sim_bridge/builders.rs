use std::collections::{HashMap, VecDeque};

use crate::ai::core::{sim_vec2, SimState, SimVec2, Team, UnitState};
use crate::ai::effects::HeroToml;
use crate::mission::hero_templates::hero_toml_to_unit;

use super::types::scale_enemy_stats;

// ---------------------------------------------------------------------------
// Helper: build a default SimState
// ---------------------------------------------------------------------------

/// Creates a test `SimState` with `player_count` hero units and `enemy_count`
/// enemy units placed in a 20x20 room.
///
/// - Hero units start along x = 2..8, y = 5 (left side).
/// - Enemy units start along x = 12..18, y = 15 (right side).
pub fn build_default_sim(player_count: usize, enemy_count: usize, seed: u64) -> SimState {
    let mut units = Vec::new();
    let mut next_id: u32 = 1;

    // Hero units
    let hero_x_step = if player_count > 1 {
        6.0 / (player_count as f32 - 1.0)
    } else {
        0.0
    };
    for i in 0..player_count {
        let x = 2.0 + i as f32 * hero_x_step;
        units.push(UnitState {
            id: next_id,
            team: Team::Hero,
            hp: 100,
            max_hp: 100,
            position: SimVec2 { x, y: 5.0 },
            move_speed_per_sec: 3.0,
            attack_damage: 15,
            attack_range: 1.5,
            attack_cooldown_ms: 1000,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
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

    // Enemy units
    let enemy_x_step = if enemy_count > 1 {
        6.0 / (enemy_count as f32 - 1.0)
    } else {
        0.0
    };
    for i in 0..enemy_count {
        let x = 12.0 + i as f32 * enemy_x_step;
        units.push(UnitState {
            id: next_id,
            team: Team::Enemy,
            hp: 100,
            max_hp: 100,
            position: SimVec2 { x, y: 15.0 },
            move_speed_per_sec: 3.0,
            attack_damage: 15,
            attack_range: 1.5,
            attack_cooldown_ms: 1000,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
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

/// Creates a `SimState` with `hero_count` default hero units and a
/// caller-supplied enemy wave.
pub fn build_sim_with_templates(
    hero_count: usize,
    enemy_wave: Vec<UnitState>,
    seed: u64,
) -> SimState {
    let mut sim = build_default_sim(hero_count, 0, seed);
    sim.units.extend(enemy_wave);
    sim
}

/// Build a SimState from hero templates + an enemy wave.
/// Heroes are positioned along x=2..8, y=5.
pub fn build_sim_with_hero_templates(
    hero_templates: &[HeroToml],
    enemy_wave: Vec<UnitState>,
    seed: u64,
) -> SimState {
    let mut units: Vec<UnitState> = hero_templates
        .iter()
        .enumerate()
        .map(|(i, toml)| {
            let x = 2.0 + (i as f32) * (6.0 / hero_templates.len().max(1) as f32);
            hero_toml_to_unit(toml, (i + 1) as u32, Team::Hero, sim_vec2(x, 5.0))
        })
        .collect();

    let enemy_start_id = units.len() as u32 + 1;
    for (j, mut enemy) in enemy_wave.into_iter().enumerate() {
        enemy.id = enemy_start_id + j as u32;
        units.push(enemy);
    }

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

/// Like `build_default_sim` but applies difficulty scaling to all enemy units
/// based on `global_turn`.  Hero stats are not modified.
pub fn build_sim_scaled(
    player_count: usize,
    enemy_count: usize,
    seed: u64,
    global_turn: u32,
) -> SimState {
    let mut sim = build_default_sim(player_count, enemy_count, seed);
    for unit in sim.units.iter_mut().filter(|u| u.team == Team::Enemy) {
        scale_enemy_stats(unit, global_turn);
    }
    sim
}
