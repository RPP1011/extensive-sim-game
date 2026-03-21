//! Shared combat setup interface — the single entry point for creating a combat
//! encounter from a scenario config.  Both the game UI and headless tests use
//! this; no Bevy dependency.

use crate::ai::core::SimState;
use crate::ai::pathing::GridNav;
use crate::ai::squad::SquadAiState;
use crate::game_core::RoomType;
use crate::mission::room_gen::RoomLayout;
use crate::scenario::types::ScenarioCfg;

use super::runner::{
    build_unified_ai, load_manifest, navgrid_to_gridnav, resolve_hero_templates,
    find_nearest_walkable, score_spawn_quality, build_hvh_with_spawns_and_tomls,
};

/// Everything needed to run or render a combat encounter.
/// Pure data — no Bevy, no rendering, no ECS.
pub struct CombatSetup {
    pub sim: SimState,
    pub grid_nav: GridNav,
    pub layout: RoomLayout,
    pub squad_ai: SquadAiState,
}

/// The single entry point for creating a combat encounter from a scenario
/// config.  Both the game UI and headless tests call this.
pub fn build_combat(cfg: &ScenarioCfg) -> CombatSetup {
    use crate::ai::core::{SimVec2, Team};
    use crate::mission::enemy_templates::default_enemy_wave;
    use crate::mission::room_gen::generate_room;
    use crate::mission::sim_bridge::{
        build_sim_with_hero_templates, build_sim_with_templates, scale_enemy_stats,
    };

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
        let mut s = build_hvh_with_spawns_and_tomls(
            &hero_tomls,
            &enemy_tomls,
            cfg.seed,
            &layout.player_spawn.positions,
            &layout.enemy_spawn.positions,
        );
        // Apply per-unit overrides from EnemyUnitDef
        let enemy_ids: Vec<u32> = s
            .units
            .iter()
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
                        unit.position = SimVec2 {
                            x: pos[0],
                            y: pos[1],
                        };
                    }
                }
            }
        }
        s
    } else if !cfg.enemy_hero_templates.is_empty() {
        let hero_tomls = resolve_hero_templates(&cfg.hero_templates, manifest_ref);
        let enemy_tomls = resolve_hero_templates(&cfg.enemy_hero_templates, manifest_ref);
        build_hvh_with_spawns_and_tomls(
            &hero_tomls,
            &enemy_tomls,
            cfg.seed,
            &layout.player_spawn.positions,
            &layout.enemy_spawn.positions,
        )
    } else {
        let enemy_spawns = &layout.enemy_spawn.positions;
        let spawn_positions: Vec<SimVec2> = (0..cfg.enemy_count)
            .map(|i| {
                if enemy_spawns.is_empty() {
                    SimVec2 {
                        x: 12.0 + i as f32,
                        y: 15.0,
                    }
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

    // Difficulty scaling for non-HvH
    if cfg.enemy_hero_templates.is_empty() {
        let global_turn = cfg.difficulty.saturating_sub(1) * 5;
        for unit in sim.units.iter_mut().filter(|u| u.team == Team::Enemy) {
            scale_enemy_stats(unit, global_turn);
        }
    }

    // HP multiplier
    if cfg.hp_multiplier != 1.0 {
        let m = cfg.hp_multiplier;
        for unit in sim.units.iter_mut() {
            unit.hp = (unit.hp as f32 * m) as i32;
            unit.max_hp = (unit.max_hp as f32 * m) as i32;
        }
    }

    // Position heroes: drills go to room center, normal scenarios use spawn zones
    if cfg.drill_type.is_some() {
        let cx = layout.width / 2.0;
        let cy = layout.depth / 2.0;
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
    let squad_ai = build_unified_ai(&sim);

    CombatSetup {
        sim,
        grid_nav,
        layout,
        squad_ai,
    }
}

// ---------------------------------------------------------------------------
// Campaign → ScenarioCfg builder
// ---------------------------------------------------------------------------

/// Default enemy templates per archetype/faction feel.
const DEFAULT_ENEMY_POOL: &[&str] = &[
    "knight", "berserker", "ranger", "mage", "assassin", "shaman",
];

/// Build a `ScenarioCfg` from campaign parameters.  Pure function, no Bevy types.
///
/// Maps campaign parameters into a scenario config:
/// - `difficulty` + `global_turn` → `enemy_count`, `hp_multiplier`
/// - `hero_templates` from the roster's assigned party
/// - Enemy templates selected from standard pool
/// - Room type and seed passed through
pub fn scenario_from_campaign(
    hero_templates: &[String],
    difficulty: u32,
    global_turn: u32,
    room_type: RoomType,
    seed: u64,
    _region_faction: Option<&str>,
) -> ScenarioCfg {
    // Scale enemy count: base 2 + difficulty, capped at 6
    let enemy_count = (2 + difficulty as usize).min(6);

    // HP multiplier scales gently with global turn
    let hp_multiplier = 1.0 + (global_turn as f32 / 30.0).min(1.0);

    // Pick enemy templates deterministically from pool based on seed
    let enemy_hero_templates: Vec<String> = (0..enemy_count)
        .map(|i| {
            let idx = ((seed as usize).wrapping_add(i * 7)) % DEFAULT_ENEMY_POOL.len();
            DEFAULT_ENEMY_POOL[idx].to_string()
        })
        .collect();

    let room_type_str = match room_type {
        RoomType::Entry => "Entry",
        RoomType::Pressure => "Pressure",
        RoomType::Pivot => "Pivot",
        RoomType::Setpiece => "Setpiece",
        RoomType::Recovery => "Recovery",
        RoomType::Climax => "Climax",
        RoomType::Open => "Open",
    };

    ScenarioCfg {
        name: format!("campaign_t{}d{}", global_turn, difficulty),
        seed,
        hero_count: hero_templates.len(),
        enemy_count,
        difficulty,
        hero_templates: hero_templates.to_vec(),
        enemy_hero_templates,
        hp_multiplier,
        room_type: room_type_str.to_string(),
        max_ticks: 3000,
        ..ScenarioCfg::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::Team;

    #[test]
    fn build_combat_produces_correct_unit_counts() {
        let cfg = ScenarioCfg {
            name: "test_basic".into(),
            seed: 42,
            hero_count: 2,
            enemy_count: 3,
            hero_templates: vec![],
            ..ScenarioCfg::default()
        };
        let setup = build_combat(&cfg);
        let heroes = setup.sim.units.iter().filter(|u| u.team == Team::Hero).count();
        let enemies = setup.sim.units.iter().filter(|u| u.team == Team::Enemy).count();
        assert_eq!(heroes, 2);
        assert_eq!(enemies, 3);
    }

    #[test]
    fn build_combat_with_hero_templates() {
        let cfg = ScenarioCfg {
            name: "test_templates".into(),
            seed: 100,
            hero_templates: vec!["knight".into(), "mage".into()],
            enemy_count: 2,
            ..ScenarioCfg::default()
        };
        let setup = build_combat(&cfg);
        let heroes = setup.sim.units.iter().filter(|u| u.team == Team::Hero).count();
        assert_eq!(heroes, 2);
        // Grid nav should be populated
        assert!(setup.sim.grid_nav.is_some());
    }

    #[test]
    fn campaign_scenario_scales_with_difficulty() {
        let cfg_low = scenario_from_campaign(
            &["knight".into()],
            1,
            0,
            RoomType::Entry,
            42,
            None,
        );
        let cfg_high = scenario_from_campaign(
            &["knight".into(), "mage".into()],
            5,
            15,
            RoomType::Climax,
            42,
            None,
        );
        // Higher difficulty → more enemies
        assert!(cfg_high.enemy_count > cfg_low.enemy_count);
        // Higher global_turn → higher HP multiplier
        assert!(cfg_high.hp_multiplier > cfg_low.hp_multiplier);
        // Hero templates preserved
        assert_eq!(cfg_high.hero_templates.len(), 2);
    }

    #[test]
    fn campaign_scenario_generates_enemy_templates() {
        let cfg = scenario_from_campaign(
            &["knight".into()],
            3,
            10,
            RoomType::Pressure,
            99,
            None,
        );
        // Should have enemy hero templates (HvH mode)
        assert!(!cfg.enemy_hero_templates.is_empty());
        assert_eq!(cfg.enemy_hero_templates.len(), cfg.enemy_count);
    }
}
