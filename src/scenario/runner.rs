use crate::ai::core::{sim_vec2, SimState, SimVec2, Team};
use crate::ai::effects::HeroToml;
use crate::ai::effects::manifest::AbilityManifest;
use crate::ai::pathing::GridNav;
use crate::game_core::RoomType;
use crate::mission::enemy_templates::default_enemy_wave;
use crate::mission::hero_templates::{load_embedded_templates, hero_toml_to_unit, parse_hero_toml_with_dsl, HeroTemplate};
use crate::mission::room_gen::{generate_room, NavGrid};
use crate::mission::sim_bridge::{build_sim_with_hero_templates, build_sim_with_templates, scale_enemy_stats};

use super::types::ScenarioCfg;

// ---------------------------------------------------------------------------
// Unified AI
// ---------------------------------------------------------------------------

/// Build a unified SquadAiState that covers both hero and enemy units,
/// inferring personalities from unit stats.
pub fn build_unified_ai(
    sim: &SimState,
) -> crate::ai::squad::SquadAiState {
    crate::ai::squad::SquadAiState::new_inferred(sim)
}

// ---------------------------------------------------------------------------
// Hero template resolution
// ---------------------------------------------------------------------------

/// Resolve hero template names into HeroToml structs.
pub(crate) fn resolve_hero_templates(
    names: &[String],
    manifest: Option<&AbilityManifest>,
) -> Vec<HeroToml> {
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

            // If name is a path (contains / or \), try it directly
            let mut search_paths: Vec<String> = Vec::new();
            if name.contains('/') || name.contains('\\') {
                search_paths.push(name.clone());
                if !name.ends_with(".toml") {
                    search_paths.push(format!("{}.toml", name));
                }
            } else {
                search_paths.push(format!("assets/hero_templates/{}.toml", name.to_lowercase()));
                search_paths.push(format!("assets/lol_heroes/{}.toml", name));
                // Search dataset/heroes/ subdirectories
                if let Ok(entries) = std::fs::read_dir("dataset/heroes") {
                    let lower_name = name.to_lowercase();
                    for entry in entries.flatten() {
                        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            search_paths.push(format!(
                                "{}/{}.toml",
                                entry.path().display(),
                                lower_name,
                            ));
                        }
                    }
                }
            }

            let mut toml: Option<HeroToml> = None;
            for path in &search_paths {
                if let Ok(content) = std::fs::read_to_string(path) {
                    // Check for .ability DSL file
                    let dsl_content = std::path::Path::new(path)
                        .with_extension("ability")
                        .to_str()
                        .and_then(|p| std::fs::read_to_string(p).ok());
                    toml = parse_hero_toml_with_dsl(&content, dsl_content.as_deref()).ok();
                    if toml.is_some() {
                        break;
                    }
                }
            }

            // Resolve ability_refs and passive_refs from manifest
            if let (Some(ref mut t), Some(m)) = (&mut toml, manifest) {
                for ref_name in &t.ability_refs.clone() {
                    if let Some(def) = m.get_ability(ref_name) {
                        t.abilities.push(def.clone());
                    } else {
                        eprintln!("Warning: ability ref '{}' not found in manifest", ref_name);
                    }
                }
                for ref_name in &t.passive_refs.clone() {
                    if let Some(def) = m.get_passive(ref_name) {
                        t.passives.push(def.clone());
                    } else {
                        eprintln!("Warning: passive ref '{}' not found in manifest", ref_name);
                    }
                }
            }

            toml
        })
        .collect()
}

/// Try to load an AbilityManifest from the given path or default location.
pub(crate) fn load_manifest(manifest_path: Option<&str>) -> Option<AbilityManifest> {
    let path = manifest_path.unwrap_or("dataset/manifest.toml");
    let manifest_path = std::path::Path::new(path);
    if !manifest_path.exists() {
        return None;
    }
    let base_dir = manifest_path.parent().unwrap_or(std::path::Path::new("."));
    match AbilityManifest::load(manifest_path, base_dir) {
        Ok(m) => {
            eprintln!("Loaded ability manifest: {} entries from {}", m.len(), path);
            Some(m)
        }
        Err(e) => {
            eprintln!("Warning: failed to load manifest {}: {}", path, e);
            None
        }
    }
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
    build_hero_vs_hero_with_spawns(hero_tomls, enemy_tomls, seed, None, None)
}

pub fn build_hvh_with_spawns_and_tomls(
    hero_tomls: &[HeroToml],
    enemy_tomls: &[HeroToml],
    seed: u64,
    hero_spawns: &[SimVec2],
    enemy_spawns: &[SimVec2],
) -> SimState {
    build_hero_vs_hero_with_spawns(
        hero_tomls, enemy_tomls, seed,
        Some(hero_spawns), Some(enemy_spawns),
    )
}

fn build_hero_vs_hero_with_spawns(
    hero_tomls: &[HeroToml],
    enemy_tomls: &[HeroToml],
    seed: u64,
    hero_spawns: Option<&[SimVec2]>,
    enemy_spawns: Option<&[SimVec2]>,
) -> SimState {
    let mut units: Vec<crate::ai::core::UnitState> = Vec::new();
    let mut next_id: u32 = 1;

    // Team A (Hero)
    for (i, toml) in hero_tomls.iter().enumerate() {
        let pos = if let Some(spawns) = hero_spawns {
            if !spawns.is_empty() { spawns[i % spawns.len()] }
            else { sim_vec2(2.0 + (i as f32) * (6.0 / hero_tomls.len().max(1) as f32), 5.0) }
        } else {
            sim_vec2(2.0 + (i as f32) * (6.0 / hero_tomls.len().max(1) as f32), 5.0)
        };
        units.push(hero_toml_to_unit(toml, next_id, Team::Hero, pos));
        next_id += 1;
    }

    // Team B (Enemy)
    for (i, toml) in enemy_tomls.iter().enumerate() {
        let pos = if let Some(spawns) = enemy_spawns {
            if !spawns.is_empty() { spawns[i % spawns.len()] }
            else { sim_vec2(12.0 + (i as f32) * (6.0 / enemy_tomls.len().max(1) as f32), 15.0) }
        } else {
            sim_vec2(12.0 + (i as f32) * (6.0 / enemy_tomls.len().max(1) as f32), 15.0)
        };
        units.push(hero_toml_to_unit(toml, next_id, Team::Enemy, pos));
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
    let manifest = load_manifest(cfg.manifest_path.as_deref());
    let manifest_ref = manifest.as_ref();

    let mut sim = if !cfg.enemy_units.is_empty() {
        // Drill mode: spawn enemies from EnemyUnitDef definitions
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
        let mut enemy_templates: Vec<String> = Vec::new();
        for eu in &cfg.enemy_units {
            if let Some(ref tmpl) = eu.template {
                enemy_templates.push(tmpl.clone());
            }
        }
        let enemy_tomls = resolve_hero_templates(&enemy_templates, manifest_ref);
        let mut s = build_hero_vs_hero(&hero_tomls, &enemy_tomls, cfg.seed);

        // Apply per-unit overrides from EnemyUnitDef
        let enemy_ids: Vec<u32> = s.units.iter()
            .filter(|u| u.team == Team::Enemy)
            .map(|u| u.id)
            .collect();
        for (i, eu) in cfg.enemy_units.iter().enumerate() {
            if let Some(&uid) = enemy_ids.get(i) {
                if let Some(unit) = s.units.iter_mut().find(|u| u.id == uid) {
                    if let Some(hp) = eu.hp_override {
                        unit.hp = hp;
                        unit.max_hp = hp;
                    }
                    if let Some(dps) = eu.dps_override {
                        unit.attack_damage = (dps * 0.5) as i32;
                    }
                    if let Some(range) = eu.range_override {
                        unit.attack_range = range;
                    }
                    if let Some(speed) = eu.move_speed_override {
                        unit.move_speed_per_sec = speed;
                    }
                    if let Some(ref pos) = eu.position {
                        unit.position = SimVec2 { x: pos[0], y: pos[1] };
                    }
                }
            }
        }
        s
    } else if !cfg.enemy_hero_templates.is_empty() {
        // Hero vs Hero mode
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
        let enemy_tomls = resolve_hero_templates(&cfg.enemy_hero_templates, manifest_ref);
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
            let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
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

/// Score spawn zone quality for terrain balance checking.
/// Returns a composite score based on nearby cover and average elevation.
fn score_spawn_quality(nav: &NavGrid, positions: &[SimVec2]) -> f32 {
    if positions.is_empty() {
        return 0.0;
    }
    let grid_nav = nav.to_gridnav();
    let mut total_cover = 0.0f32;
    let mut total_elev = 0.0f32;
    for pos in positions {
        // Count blocked neighbors within radius 3 as cover
        let cell = grid_nav.cell_of(*pos);
        for dx in -3..=3i32 {
            for dy in -3..=3i32 {
                if grid_nav.blocked.contains(&(cell.0 + dx, cell.1 + dy)) {
                    total_cover += 1.0;
                }
            }
        }
        total_elev += grid_nav.elevation_at_pos(*pos);
    }
    let n = positions.len() as f32;
    (total_cover / n) + (total_elev / n)
}

/// Find the nearest walkable cell to the given world-space position.
fn find_nearest_walkable(nav: &NavGrid, x: f32, y: f32) -> SimVec2 {
    let center_c = (x as usize).min(nav.cols.saturating_sub(1));
    let center_r = (y as usize).min(nav.rows.saturating_sub(1));

    // Check center first
    if nav.walkable[center_r * nav.cols + center_c] {
        return nav.cell_centre(center_c, center_r);
    }

    // Spiral outward
    let mut best = nav.cell_centre(center_c, center_r);
    let mut best_dist = f32::MAX;
    let max_radius = nav.cols.max(nav.rows);
    for r in 1..max_radius {
        for dr in -(r as isize)..=(r as isize) {
            for dc in -(r as isize)..=(r as isize) {
                if dr.unsigned_abs() != r && dc.unsigned_abs() != r {
                    continue; // only check the ring perimeter
                }
                let c = center_c as isize + dc;
                let row = center_r as isize + dr;
                if c < 0 || row < 0 || c as usize >= nav.cols || row as usize >= nav.rows {
                    continue;
                }
                let (uc, ur) = (c as usize, row as usize);
                if nav.walkable[ur * nav.cols + uc] {
                    let pos = nav.cell_centre(uc, ur);
                    let d = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);
                    if d < best_dist {
                        best_dist = d;
                        best = pos;
                    }
                }
            }
        }
        if best_dist < f32::MAX {
            return best;
        }
    }
    best
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

    let manifest = load_manifest(cfg.manifest_path.as_deref());
    let manifest_ref = manifest.as_ref();

    let mut sim = if !cfg.enemy_units.is_empty() {
        // Drill mode with room-aware spawns
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
        let mut enemy_templates: Vec<String> = Vec::new();
        for eu in &cfg.enemy_units {
            if let Some(ref tmpl) = eu.template {
                enemy_templates.push(tmpl.clone());
            }
        }
        let enemy_tomls = resolve_hero_templates(&enemy_templates, manifest_ref);
        let mut s = build_hero_vs_hero_with_spawns(
            &hero_tomls, &enemy_tomls, cfg.seed,
            Some(&layout.player_spawn.positions),
            Some(&layout.enemy_spawn.positions),
        );
        // Apply overrides
        let enemy_ids: Vec<u32> = s.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();
        for (i, eu) in cfg.enemy_units.iter().enumerate() {
            if let Some(&uid) = enemy_ids.get(i) {
                if let Some(unit) = s.units.iter_mut().find(|u| u.id == uid) {
                    if let Some(hp) = eu.hp_override {
                        unit.hp = hp; unit.max_hp = hp;
                    }
                    if let Some(dps) = eu.dps_override {
                        unit.attack_damage = (dps * 0.5) as i32;
                    }
                    if let Some(range) = eu.range_override {
                        unit.attack_range = range;
                    }
                    if let Some(speed) = eu.move_speed_override {
                        unit.move_speed_per_sec = speed;
                    }
                    if let Some(ref pos) = eu.position {
                        unit.position = SimVec2 { x: pos[0], y: pos[1] };
                    }
                }
            }
        }
        s
    } else if !cfg.enemy_hero_templates.is_empty() {
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
        let enemy_tomls = resolve_hero_templates(&cfg.enemy_hero_templates, manifest_ref);
        build_hero_vs_hero_with_spawns(
            &hero_tomls, &enemy_tomls, cfg.seed,
            Some(&layout.player_spawn.positions),
            Some(&layout.enemy_spawn.positions),
        )
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
            let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
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

    // For drills, spawn heroes at room center for uniform position token coverage.
    // For non-drill scenarios, use the room's player spawn zone.
    if cfg.drill_type.is_some() {
        let cx = layout.width / 2.0;
        let cy = layout.depth / 2.0;
        // Find nearest walkable cell to center
        let center = find_nearest_walkable(&layout.nav, cx, cy);
        for unit in sim.units.iter_mut().filter(|u| u.team == Team::Hero) {
            unit.position = center;
        }
    } else {
        let hero_spawns = &layout.player_spawn.positions;
        if !hero_spawns.is_empty() {
            let mut hi = 0;
            for unit in sim.units.iter_mut().filter(|u| u.team == Team::Hero) {
                unit.position = hero_spawns[hi % hero_spawns.len()];
                hi += 1;
            }
        }
    }

    // Spawn quality balance check
    let player_quality = score_spawn_quality(&layout.nav, &layout.player_spawn.positions);
    let enemy_quality = score_spawn_quality(&layout.nav, &layout.enemy_spawn.positions);
    let quality_diff = (player_quality - enemy_quality).abs();
    if quality_diff > 5.0 {
        eprintln!(
            "Warning: spawn quality imbalance {:.1} (player={:.1}, enemy={:.1}) for seed={} room={:?}",
            quality_diff, player_quality, enemy_quality, cfg.seed, room_type,
        );
    }

    let grid_nav = navgrid_to_gridnav(&layout.nav);
    sim.grid_nav = Some(grid_nav.clone());
    let squad_state = build_unified_ai(&sim);
    (sim, squad_state, grid_nav)
}
