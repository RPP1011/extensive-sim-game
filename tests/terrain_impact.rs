//! Benchmark: measure how terrain affects hero win rates.
//!
//! Runs 1000 scenarios per (hero, enemy_count, room_type) combination,
//! comparing terrain-aware combat (cover + elevation) against open-field.
//!
//! Run with: cargo test --test terrain_impact -- --nocapture

use bevy_game::ai::core::{distance, step, SimState, Team, FIXED_TICK_MS};
use bevy_game::ai::pathing::{cover_factor, GridNav};
use bevy_game::ai::squad;
use bevy_game::scenario::{run_scenario_to_state_with_room, ScenarioCfg};

const MAX_TICKS: u64 = 3000;
const SEEDS: u64 = 30;

// ---------------------------------------------------------------------------
// Terrain-aware sim loop (mirrors what mission execution does)
// ---------------------------------------------------------------------------

fn run_with_terrain(cfg: &ScenarioCfg) -> (String, u64) {
    let (mut sim, mut squad_ai, grid_nav) = run_scenario_to_state_with_room(cfg);

    // Initial terrain modifier pass
    update_terrain(&mut sim, &grid_nav);

    for _ in 0..MAX_TICKS {
        let intents = squad::generate_intents_with_terrain(&sim, &mut squad_ai, FIXED_TICK_MS, Some(&grid_nav));
        let (mut new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
        update_terrain(&mut new_sim, &grid_nav);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return ("Victory".into(), sim.tick);
        }
        if heroes_alive == 0 {
            return ("Defeat".into(), sim.tick);
        }
    }
    ("Timeout".into(), sim.tick)
}

fn run_without_terrain(cfg: &ScenarioCfg) -> (String, u64) {
    let (mut sim, mut squad_ai, _grid_nav) = run_scenario_to_state_with_room(cfg);
    // Don't call update_terrain — cover_bonus and elevation stay at 0.0
    // Also disable pathfinding so movement uses straight-line math
    sim.grid_nav = None;

    for _ in 0..MAX_TICKS {
        let intents = squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return ("Victory".into(), sim.tick);
        }
        if heroes_alive == 0 {
            return ("Defeat".into(), sim.tick);
        }
    }
    ("Timeout".into(), sim.tick)
}

fn update_terrain(sim: &mut SimState, nav: &GridNav) {
    let unit_count = sim.units.len();
    for i in 0..unit_count {
        if sim.units[i].hp <= 0 {
            sim.units[i].cover_bonus = 0.0;
            sim.units[i].elevation = 0.0;
            continue;
        }
        sim.units[i].elevation = nav.elevation_at_pos(sim.units[i].position);

        let pos = sim.units[i].position;
        let team = sim.units[i].team;
        let mut nearest_enemy_pos = None;
        let mut nearest_dist = f32::INFINITY;
        for j in 0..unit_count {
            if sim.units[j].hp <= 0 || sim.units[j].team == team {
                continue;
            }
            let d = distance(pos, sim.units[j].position);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_enemy_pos = Some(sim.units[j].position);
            }
        }
        sim.units[i].cover_bonus = match nearest_enemy_pos {
            Some(ep) => cover_factor(nav, pos, ep),
            None => 0.0,
        };
    }
}

// ---------------------------------------------------------------------------
// Batch runner
// ---------------------------------------------------------------------------

struct BatchResult {
    hero: String,
    enemies: usize,
    room: String,
    wins_terrain: u32,
    wins_no_terrain: u32,
    avg_ticks_terrain: f64,
    avg_ticks_no_terrain: f64,
}

fn run_batch(hero: &str, enemy_count: usize, room_type: &str) -> BatchResult {
    let mut wins_t = 0u32;
    let mut wins_nt = 0u32;
    let mut ticks_t = 0u64;
    let mut ticks_nt = 0u64;

    for seed in 0..SEEDS {
        let cfg = ScenarioCfg {
            name: format!("{hero}_vs_{enemy_count}_{room_type}"),
            seed,
            hero_count: 1,
            enemy_count,
            difficulty: 1,
            max_ticks: MAX_TICKS,
            room_type: room_type.to_string(),
            hero_templates: vec![hero.to_string()],
            enemy_hero_templates: Vec::new(),
            hp_multiplier: 1.0,
        };

        let (outcome_t, tick_t) = run_with_terrain(&cfg);
        let (outcome_nt, tick_nt) = run_without_terrain(&cfg);

        if outcome_t == "Victory" {
            wins_t += 1;
        }
        if outcome_nt == "Victory" {
            wins_nt += 1;
        }
        ticks_t += tick_t;
        ticks_nt += tick_nt;
    }

    BatchResult {
        hero: hero.to_string(),
        enemies: enemy_count,
        room: room_type.to_string(),
        wins_terrain: wins_t,
        wins_no_terrain: wins_nt,
        avg_ticks_terrain: ticks_t as f64 / SEEDS as f64,
        avg_ticks_no_terrain: ticks_nt as f64 / SEEDS as f64,
    }
}

// ---------------------------------------------------------------------------
// Test entry point
// ---------------------------------------------------------------------------

/// Party-based test: multiple party compositions vs hordes
/// Run explicitly: cargo test --test terrain_impact terrain_impact_party -- --ignored --nocapture
#[test]
#[ignore]
fn terrain_impact_party() {
    let parties: Vec<(&str, Vec<String>)> = vec![
        ("WRCR", vec!["warrior".into(), "ranger".into(), "cleric".into(), "rogue".into()]),
        ("EWRC", vec!["engineer".into(), "warrior".into(), "ranger".into(), "cleric".into()]),
    ];
    let enemy_counts = [6, 10];
    let room_types = ["Entry", "Pressure", "Setpiece"];

    println!();
    println!(
        "{:<16} {:>3} {:<10} {:>6} {:>6} {:>+7} {:>8} {:>8}",
        "Party", "vs", "Room", "W%_T", "W%_NT", "Delta", "Ticks_T", "Ticks_NT"
    );
    println!("{}", "-".repeat(76));

    for (label, party) in &parties {
        for &enemies in &enemy_counts {
            for &room in &room_types {
                let mut wins_t = 0u32;
                let mut wins_nt = 0u32;
                let mut ticks_t = 0u64;
                let mut ticks_nt = 0u64;

                let party_seeds = 5u64;
                for seed in 0..party_seeds {
                    let cfg = ScenarioCfg {
                        name: format!("{label}_vs_{enemies}_{room}"),
                        seed,
                        hero_count: 4,
                        enemy_count: enemies,
                        difficulty: 1,
                        max_ticks: MAX_TICKS,
                        room_type: room.to_string(),
                        hero_templates: party.clone(),
                        enemy_hero_templates: Vec::new(),
                        hp_multiplier: 1.0,
                    };
                    let (ot, tt) = run_with_terrain(&cfg);
                    let (ont, tnt) = run_without_terrain(&cfg);
                    if ot == "Victory" { wins_t += 1; }
                    if ont == "Victory" { wins_nt += 1; }
                    ticks_t += tt;
                    ticks_nt += tnt;
                }
                let pct_t = wins_t as f64 / party_seeds as f64 * 100.0;
                let pct_nt = wins_nt as f64 / party_seeds as f64 * 100.0;
                let delta = pct_t - pct_nt;
                println!(
                    "{:<16} {:>3} {:<10} {:>5.1}% {:>5.1}% {:>+6.1}% {:>8.0} {:>8.0}",
                    label, enemies, room, pct_t, pct_nt, delta,
                    ticks_t as f64 / party_seeds as f64,
                    ticks_nt as f64 / party_seeds as f64,
                );
            }
        }
        println!();
    }
}

/// Run explicitly: cargo test --test terrain_impact terrain_impact_benchmark -- --ignored --nocapture
#[test]
#[ignore]
fn terrain_impact_benchmark() {
    let heroes = ["warrior", "ranger", "mage", "cleric", "rogue", "paladin", "engineer"];
    let enemy_counts = [2];
    let room_types = ["Entry", "Pressure", "Setpiece"];

    println!();
    println!(
        "{:<10} {:>3} {:<10} {:>6} {:>6} {:>+7} {:>8} {:>8}",
        "Hero", "vs", "Room", "W%_T", "W%_NT", "Delta", "Ticks_T", "Ticks_NT"
    );
    println!("{}", "-".repeat(72));

    for hero in &heroes {
        for &enemies in &enemy_counts {
            for &room in &room_types {
                let r = run_batch(hero, enemies, room);
                let pct_t = r.wins_terrain as f64 / SEEDS as f64 * 100.0;
                let pct_nt = r.wins_no_terrain as f64 / SEEDS as f64 * 100.0;
                let delta = pct_t - pct_nt;
                println!(
                    "{:<10} {:>3} {:<10} {:>5.1}% {:>5.1}% {:>+6.1}% {:>8.0} {:>8.0}",
                    r.hero, r.enemies, r.room, pct_t, pct_nt, delta,
                    r.avg_ticks_terrain, r.avg_ticks_no_terrain,
                );
            }
        }
        println!();
    }
}
