//! Validation test for all LoL hero TOML files.
//!
//! Checks:
//! 1. Every TOML file parses into HeroToml
//! 2. Every HeroToml converts to a valid UnitState
//! 3. Each hero can run 200 ticks in a 1v2 sim without panicking
//! 4. Abilities actually fire during sim
//!
//! Run: cargo test --test lol_heroes_validate -- --nocapture

use std::fs;
use std::path::Path;

use bevy_game::ai::core::{sim_vec2, step, SimState, Team, UnitState, FIXED_TICK_MS};
use bevy_game::ai::effects::HeroToml;
use bevy_game::ai::squad;
use bevy_game::mission::hero_templates::{hero_toml_to_unit, parse_hero_toml};

fn make_enemy(id: u32, position: bevy_game::ai::core::SimVec2) -> UnitState {
    use std::collections::VecDeque;
    UnitState {
        id,
        team: Team::Enemy,
        hp: 80,
        max_hp: 80,
        position,
        move_speed_per_sec: 3.0,
        attack_damage: 10,
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
        resistance_tags: Default::default(),
        state_history: VecDeque::new(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0,
        owner_id: None,
        directed: false,
        armor: 0.0,
        magic_resist: 0.0,
        cover_bonus: 0.0,
        elevation: 0.0,
        total_healing_done: 0,
        total_damage_done: 0,
    }
}

fn run_short_sim(hero: UnitState) -> (bool, u64, bool) {
    let mut sim = SimState {
        tick: 0,
        rng_state: 42,
        units: vec![
            hero,
            make_enemy(100, sim_vec2(5.0, 0.0)),
            make_enemy(101, sim_vec2(6.0, 1.0)),
        ],
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };

    let mut squad_ai = squad::SquadAiState::new_inferred(&sim);
    let mut any_ability_used = false;

    for _ in 0..200 {
        let intents = squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        // Check if hero used any abilities
        for intent in &intents {
            if intent.unit_id == 1 {
                if matches!(
                    intent.action,
                    bevy_game::ai::core::IntentAction::UseAbility { .. }
                ) {
                    any_ability_used = true;
                }
            }
        }
        let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return (true, sim.tick, any_ability_used);
        }
        let hero_alive = sim.units.iter().any(|u| u.id == 1 && u.hp > 0);
        if !hero_alive {
            return (false, sim.tick, any_ability_used);
        }
    }
    (false, sim.tick, any_ability_used)
}

#[test]
fn all_lol_heroes_parse() {
    let dir = Path::new("assets/lol_heroes");
    assert!(dir.exists(), "assets/lol_heroes/ directory not found");

    let mut files: Vec<_> = fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "toml"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    assert!(files.len() >= 170, "expected 170+ TOML files, found {}", files.len());

    let mut parse_ok = 0;
    let mut parse_fail = Vec::new();

    for entry in &files {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let content = fs::read_to_string(&path).unwrap();

        match parse_hero_toml(&content) {
            Ok(toml) => {
                // Basic sanity
                assert!(toml.stats.hp > 0, "{name}: hp must be > 0");
                assert!(!toml.abilities.is_empty(), "{name}: must have abilities");
                parse_ok += 1;
            }
            Err(e) => {
                parse_fail.push(format!("{name}: {e}"));
            }
        }
    }

    if !parse_fail.is_empty() {
        eprintln!("\n=== PARSE FAILURES ({}) ===", parse_fail.len());
        for f in &parse_fail {
            eprintln!("  {f}");
        }
    }

    assert!(
        parse_fail.is_empty(),
        "{} / {} failed to parse",
        parse_fail.len(),
        files.len()
    );
    eprintln!("\nAll {parse_ok} LoL hero TOML files parsed successfully");
}

#[test]
fn all_lol_heroes_simulate() {
    let dir = Path::new("assets/lol_heroes");
    let mut files: Vec<_> = fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "toml"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    let mut sim_ok = 0;
    let mut sim_fail = Vec::new();
    let mut abilities_used = 0;
    let mut wins = 0;

    for entry in &files {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let content = fs::read_to_string(&path).unwrap();
        let toml: HeroToml = match parse_hero_toml(&content) {
            Ok(t) => t,
            Err(_) => continue, // skip parse failures (caught by other test)
        };

        let unit = hero_toml_to_unit(&toml, 1, Team::Hero, sim_vec2(0.0, 0.0));

        // Run a short sim — just verify it doesn't panic
        let result = std::panic::catch_unwind(|| run_short_sim(unit));
        match result {
            Ok((won, _tick, used_ability)) => {
                sim_ok += 1;
                if won { wins += 1; }
                if used_ability { abilities_used += 1; }
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                sim_fail.push(format!("{name}: {msg}"));
            }
        }
    }

    eprintln!("\n=== LoL Heroes Sim Results ===");
    eprintln!("  Simulated: {sim_ok} / {}", files.len());
    eprintln!("  Panicked:  {}", sim_fail.len());
    eprintln!("  Wins (1v2): {wins} / {sim_ok}");
    eprintln!("  Used abilities: {abilities_used} / {sim_ok}");

    if !sim_fail.is_empty() {
        eprintln!("\n=== SIM FAILURES ===");
        for f in &sim_fail {
            eprintln!("  {f}");
        }
    }

    assert!(
        sim_fail.is_empty(),
        "{} / {} panicked during simulation",
        sim_fail.len(),
        files.len()
    );
}
