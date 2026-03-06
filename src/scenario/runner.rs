use crate::ai::core::{sim_vec2, SimState, SimVec2, Team};
use crate::ai::effects::HeroToml;
use crate::ai::pathing::GridNav;
use crate::game_core::RoomType;
use crate::mission::enemy_templates::default_enemy_wave;
use crate::mission::hero_templates::{load_embedded_templates, hero_toml_to_unit, HeroTemplate};
use crate::mission::room_gen::{generate_room, NavGrid};
use crate::mission::sim_bridge::{build_sim_with_hero_templates, build_sim_with_templates, scale_enemy_stats};

use super::types::ScenarioCfg;

// ---------------------------------------------------------------------------
// Unified AI
// ---------------------------------------------------------------------------

/// Build a unified SquadAiState that covers both hero and enemy units,
/// inferring personalities from unit stats.
pub(crate) fn build_unified_ai(
    sim: &SimState,
) -> crate::ai::squad::SquadAiState {
    crate::ai::squad::SquadAiState::new_inferred(sim)
}

// ---------------------------------------------------------------------------
// Hero template resolution
// ---------------------------------------------------------------------------

/// Resolve hero template names into HeroToml structs.
pub(crate) fn resolve_hero_templates(names: &[String]) -> Vec<HeroToml> {
    let embedded = load_embedded_templates();
    names
        .iter()
        .filter_map(|name| {
            let from_enum = HeroTemplate::ALL.iter().find(|t| {
                t.file_name().trim_end_matches(".toml").eq_ignore_ascii_case(name)
            });
            if let Some(t) = from_enum {
                return embedded.get(t).cloned();
            }
            // Try assets/hero_templates/ first, then assets/lol_heroes/
            let path = format!("assets/hero_templates/{}.toml", name.to_lowercase());
            if let Ok(content) = std::fs::read_to_string(&path) {
                return toml::from_str::<HeroToml>(&content).ok();
            }
            let lol_path = format!("assets/lol_heroes/{}.toml", name);
            let content = std::fs::read_to_string(&lol_path).ok()?;
            toml::from_str::<HeroToml>(&content).ok()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hero vs Hero builder
// ---------------------------------------------------------------------------

/// Build a SimState with hero templates on both teams.
fn build_hero_vs_hero(
    hero_tomls: &[HeroToml],
    enemy_tomls: &[HeroToml],
    seed: u64,
) -> SimState {
    let mut units: Vec<crate::ai::core::UnitState> = Vec::new();
    let mut next_id: u32 = 1;

    // Team A (Hero) — positioned along x=2..8, y=5
    for (i, toml) in hero_tomls.iter().enumerate() {
        let x = 2.0 + (i as f32) * (6.0 / hero_tomls.len().max(1) as f32);
        units.push(hero_toml_to_unit(toml, next_id, Team::Hero, sim_vec2(x, 5.0)));
        next_id += 1;
    }

    // Team B (Enemy) — positioned along x=12..18, y=15
    for (i, toml) in enemy_tomls.iter().enumerate() {
        let x = 12.0 + (i as f32) * (6.0 / enemy_tomls.len().max(1) as f32);
        units.push(hero_toml_to_unit(toml, next_id, Team::Enemy, sim_vec2(x, 15.0)));
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

// ---------------------------------------------------------------------------
// State builders
// ---------------------------------------------------------------------------

/// Build the initial SimState for a scenario config (open-field spawn, no room layout).
pub fn run_scenario_to_state(
    cfg: &ScenarioCfg,
) -> (
    SimState,
    crate::ai::squad::SquadAiState,
) {
    let mut sim = if !cfg.enemy_hero_templates.is_empty() {
        // Hero vs Hero mode
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates);
        let enemy_tomls = resolve_hero_templates(&cfg.enemy_hero_templates);
        build_hero_vs_hero(&hero_tomls, &enemy_tomls, cfg.seed)
    } else {
        let enemy_x_step = if cfg.enemy_count > 1 {
            6.0 / (cfg.enemy_count as f32 - 1.0)
        } else {
            0.0
        };
        let spawn_positions: Vec<SimVec2> = (0..cfg.enemy_count)
            .map(|i| SimVec2 {
                x: 12.0 + i as f32 * enemy_x_step,
                y: 15.0,
            })
            .collect();

        let enemy_wave = default_enemy_wave(cfg.enemy_count, cfg.seed, &spawn_positions);

        if !cfg.hero_templates.is_empty() {
            let hero_tomls = resolve_hero_templates(&cfg.hero_templates);
            build_sim_with_hero_templates(&hero_tomls, enemy_wave, cfg.seed)
        } else {
            build_sim_with_templates(cfg.hero_count, enemy_wave, cfg.seed)
        }
    };

    if cfg.enemy_hero_templates.is_empty() {
        let global_turn = cfg.difficulty.saturating_sub(1) * 5;
        for unit in sim.units.iter_mut().filter(|u| u.team == Team::Enemy) {
            scale_enemy_stats(unit, global_turn);
        }
    }

    if cfg.hp_multiplier != 1.0 {
        let m = cfg.hp_multiplier;
        for unit in sim.units.iter_mut() {
            unit.hp = (unit.hp as f32 * m) as i32;
            unit.max_hp = (unit.max_hp as f32 * m) as i32;
        }
    }

    let squad_state = build_unified_ai(&sim);
    (sim, squad_state)
}

/// Convert a [`NavGrid`] (room_gen, row-major, usize indices) into a
/// [`GridNav`] (pathing, i32 indices, HashSet-based blocked set).
pub fn navgrid_to_gridnav(nav: &NavGrid) -> GridNav {
    nav.to_gridnav()
}

/// Build the initial SimState, unified AI, **and** a [`GridNav`] derived from
/// the scenario's room layout.
pub fn run_scenario_to_state_with_room(
    cfg: &ScenarioCfg,
) -> (
    SimState,
    crate::ai::squad::SquadAiState,
    GridNav,
) {
    let room_type = RoomType::from_str(&cfg.room_type).unwrap_or(RoomType::Entry);
    let layout = generate_room(cfg.seed, room_type);

    let mut sim = if !cfg.enemy_hero_templates.is_empty() {
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates);
        let enemy_tomls = resolve_hero_templates(&cfg.enemy_hero_templates);
        build_hero_vs_hero(&hero_tomls, &enemy_tomls, cfg.seed)
    } else {
        let enemy_spawns = &layout.enemy_spawn.positions;
        let spawn_positions: Vec<SimVec2> = (0..cfg.enemy_count)
            .map(|i| {
                if enemy_spawns.is_empty() {
                    SimVec2 { x: 12.0 + i as f32, y: 15.0 }
                } else {
                    enemy_spawns[i % enemy_spawns.len()]
                }
            })
            .collect();

        let enemy_wave = default_enemy_wave(cfg.enemy_count, cfg.seed, &spawn_positions);

        if !cfg.hero_templates.is_empty() {
            let hero_tomls = resolve_hero_templates(&cfg.hero_templates);
            build_sim_with_hero_templates(&hero_tomls, enemy_wave, cfg.seed)
        } else {
            build_sim_with_templates(cfg.hero_count, enemy_wave, cfg.seed)
        }
    };

    if cfg.enemy_hero_templates.is_empty() {
        let global_turn = cfg.difficulty.saturating_sub(1) * 5;
        for unit in sim.units.iter_mut().filter(|u| u.team == Team::Enemy) {
            scale_enemy_stats(unit, global_turn);
        }
    }

    if cfg.hp_multiplier != 1.0 {
        let m = cfg.hp_multiplier;
        for unit in sim.units.iter_mut() {
            unit.hp = (unit.hp as f32 * m) as i32;
            unit.max_hp = (unit.max_hp as f32 * m) as i32;
        }
    }

    let hero_spawns = &layout.player_spawn.positions;
    if !hero_spawns.is_empty() {
        let mut hi = 0;
        for unit in sim.units.iter_mut().filter(|u| u.team == Team::Hero) {
            unit.position = hero_spawns[hi % hero_spawns.len()];
            hi += 1;
        }
    }

    let grid_nav = navgrid_to_gridnav(&layout.nav);
    // Only inject grid_nav into SimState if a hero has obstacle abilities,
    // as A* pathfinding on every tick is expensive for batch testing.
    let has_obstacle_ability = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .any(|u| u.abilities.iter().any(|s| {
            s.def.effects.iter().any(|ce| matches!(ce.effect, crate::ai::effects::Effect::Obstacle { .. }))
        }));
    if has_obstacle_ability {
        sim.grid_nav = Some(grid_nav.clone());
    }
    let squad_state = build_unified_ai(&sim);
    (sim, squad_state, grid_nav)
}
