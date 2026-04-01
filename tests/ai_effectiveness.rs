//! AI Effectiveness Simulations
//!
//! Runs a battery of scenarios to measure how well the hero AI handles
//! different enemy group sizes, compositions, difficulties, and room types.
//! Results are printed as a summary table.

use game::scenario::{run_scenario, ScenarioCfg};

fn cfg(
    name: &str,
    seed: u64,
    heroes: usize,
    enemies: usize,
    difficulty: u32,
    room_type: &str,
    hero_templates: Vec<String>,
) -> ScenarioCfg {
    ScenarioCfg {
        name: name.to_string(),
        seed,
        hero_count: heroes,
        enemy_count: enemies,
        difficulty,
        max_ticks: 5000,
        room_type: room_type.to_string(),
        hero_templates,
        enemy_hero_templates: Vec::new(),
        hp_multiplier: 1.0,
        ..Default::default()
    }
}

struct SimResult {
    name: String,
    outcome: String,
    ticks: u64,
    heroes_alive: usize,
    hero_deaths: usize,
    total_hero_dmg: i64,
    total_hero_heal: i64,
    total_enemy_dmg: i64,
    hero_cc_applied: u32,
    enemy_cc_applied: u32,
    abilities_used: u32,
}

fn run_and_summarize(c: ScenarioCfg) -> SimResult {
    let r = run_scenario(&c);
    let total_hero_dmg: i64 = r.unit_stats.iter().filter(|u| u.team == "Hero").map(|u| u.damage_dealt).sum();
    let total_hero_heal: i64 = r.unit_stats.iter().filter(|u| u.team == "Hero").map(|u| u.healing_done).sum();
    let total_enemy_dmg: i64 = r.unit_stats.iter().filter(|u| u.team == "Enemy").map(|u| u.damage_dealt).sum();
    let hero_cc: u32 = r.unit_stats.iter().filter(|u| u.team == "Hero").map(|u| u.cc_applied_count).sum();
    let enemy_cc: u32 = r.unit_stats.iter().filter(|u| u.team == "Enemy").map(|u| u.cc_applied_count).sum();
    let abilities: u32 = r.unit_stats.iter().filter(|u| u.team == "Hero").map(|u| u.abilities_used).sum();
    SimResult {
        name: r.scenario_name,
        outcome: r.outcome,
        ticks: r.tick,
        heroes_alive: r.final_hero_count,
        hero_deaths: r.hero_deaths,
        total_hero_dmg,
        total_hero_heal,
        total_enemy_dmg,
        hero_cc_applied: hero_cc,
        enemy_cc_applied: enemy_cc,
        abilities_used: abilities,
    }
}

fn print_table(label: &str, results: &[SimResult]) {
    println!("\n{}", "=".repeat(120));
    println!("  {}", label);
    println!("{}", "=".repeat(120));
    println!(
        "{:<35} {:>7} {:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>4} {:>4} {:>5}",
        "Scenario", "Outcome", "Ticks", "Alive", "Deaths", "HeroDmg", "HeroHeal", "EnemDmg", "HCC", "ECC", "Abils"
    );
    println!("{:-<120}", "");
    for r in results {
        println!(
            "{:<35} {:>7} {:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>4} {:>4} {:>5}",
            r.name, r.outcome, r.ticks, r.heroes_alive, r.hero_deaths,
            r.total_hero_dmg, r.total_hero_heal, r.total_enemy_dmg,
            r.hero_cc_applied, r.enemy_cc_applied, r.abilities_used,
        );
    }

    // Summary stats
    let wins = results.iter().filter(|r| r.outcome == "Victory").count();
    let losses = results.iter().filter(|r| r.outcome == "Defeat").count();
    let timeouts = results.iter().filter(|r| r.outcome == "Timeout").count();
    println!("{:-<120}", "");
    println!(
        "Win rate: {}/{} ({:.0}%)  |  Losses: {}  |  Timeouts: {}",
        wins,
        results.len(),
        wins as f64 / results.len() as f64 * 100.0,
        losses,
        timeouts,
    );
    if !results.is_empty() {
        let avg_ticks: f64 = results.iter().map(|r| r.ticks as f64).sum::<f64>() / results.len() as f64;
        let avg_alive: f64 = results.iter().map(|r| r.heroes_alive as f64).sum::<f64>() / results.len() as f64;
        println!(
            "Avg ticks: {:.0}  |  Avg heroes alive: {:.1}",
            avg_ticks, avg_alive,
        );
    }
}

// ─── Test 1: Scaling enemy count (4 heroes vs 2..10 enemies) ───

#[test]
fn ai_vs_scaling_enemy_count() {
    let results: Vec<SimResult> = (2..=10)
        .map(|e| {
            run_and_summarize(cfg(
                &format!("4v{} Entry d2", e),
                100 + e as u64,
                4, e as usize, 2, "Entry", vec![],
            ))
        })
        .collect();
    print_table("SCALING ENEMY COUNT (4 heroes, difficulty 2, Entry room)", &results);
}

// ─── Test 2: Scaling difficulty (4v4, difficulty 1..5) ───

#[test]
fn ai_vs_scaling_difficulty() {
    let results: Vec<SimResult> = (1..=5)
        .map(|d| {
            run_and_summarize(cfg(
                &format!("4v4 d{}", d),
                42,
                4, 4, d, "Entry", vec![],
            ))
        })
        .collect();
    print_table("SCALING DIFFICULTY (4v4, Entry room)", &results);
}

// ─── Test 3: Different room types (4v4, difficulty 2) ───

#[test]
fn ai_across_room_types() {
    let room_types = ["Entry", "Pressure", "Pivot", "Setpiece", "Recovery", "Climax"];
    let results: Vec<SimResult> = room_types
        .iter()
        .map(|rt| {
            run_and_summarize(cfg(
                &format!("4v4 {}", rt),
                42,
                4, 4, 2, rt, vec![],
            ))
        })
        .collect();
    print_table("ROOM TYPE COMPARISON (4v4, difficulty 2)", &results);
}

// ─── Test 4: Named hero templates vs enemies ───

#[test]
fn named_heroes_vs_enemies() {
    let comps: Vec<(&str, Vec<String>)> = vec![
        ("Knight+Pyro+Druid+Assassin", vec!["knight", "pyromancer", "druid", "assassin"].into_iter().map(String::from).collect()),
        ("Templar+Necro+Bard+Monk", vec!["templar", "necromancer", "bard", "monk"].into_iter().map(String::from).collect()),
        ("Berserker+Shaman+Engineer+Shadow", vec!["berserker", "shaman", "engineer", "shadow_dancer"].into_iter().map(String::from).collect()),
        ("Warden+Warlock+Cryo+Samurai", vec!["warden", "warlock", "cryomancer", "samurai"].into_iter().map(String::from).collect()),
        ("BloodMage+Alchemist+Elem+WitchDoc", vec!["blood_mage", "alchemist", "elementalist", "witch_doctor"].into_iter().map(String::from).collect()),
    ];
    let results: Vec<SimResult> = comps
        .into_iter()
        .map(|(label, templates)| {
            run_and_summarize(cfg(
                &format!("{} 4v6", label),
                200,
                4, 6, 2, "Entry", templates,
            ))
        })
        .collect();
    print_table("NAMED HERO COMPS vs 6 ENEMIES (difficulty 2)", &results);
}

// ─── Test 5: Seed variance (same config, 10 different seeds) ───

#[test]
fn seed_variance_4v4() {
    let results: Vec<SimResult> = (0..10)
        .map(|i| {
            run_and_summarize(cfg(
                &format!("4v4 seed={}", 1000 + i * 37),
                1000 + i * 37,
                4, 4, 2, "Entry", vec![],
            ))
        })
        .collect();
    print_table("SEED VARIANCE (4v4, difficulty 2, Entry room, 10 seeds)", &results);
}

// ─── Test 6: Outnumbered stress tests ───

#[test]
fn outnumbered_stress() {
    let configs = vec![
        ("2v4 d1", 2, 4, 1),
        ("2v6 d1", 2, 6, 1),
        ("3v6 d2", 3, 6, 2),
        ("3v8 d1", 3, 8, 1),
        ("4v8 d2", 4, 8, 2),
        ("4v10 d1", 4, 10, 1),
        ("4v10 d3", 4, 10, 3),
        ("6v10 d2", 6, 10, 2),
        ("6v12 d3", 6, 12, 3),
        ("8v12 d3", 8, 12, 3),
    ];
    let results: Vec<SimResult> = configs
        .into_iter()
        .map(|(label, h, e, d)| {
            run_and_summarize(cfg(label, 500, h, e, d, "Entry", vec![]))
        })
        .collect();
    print_table("OUTNUMBERED STRESS TESTS", &results);
}

// ─── Test 7: Large battles ───

#[test]
fn large_battles() {
    let configs = vec![
        ("6v6 d2", 6, 6, 2),
        ("8v8 d2", 8, 8, 2),
        ("8v8 d4", 8, 8, 4),
        ("6v6 d4", 6, 6, 4),
        ("6v6 Climax d3", 6, 6, 3),
    ];
    let results: Vec<SimResult> = configs
        .into_iter()
        .enumerate()
        .map(|(i, (label, h, e, d))| {
            let rt = if label.contains("Climax") { "Climax" } else { "Entry" };
            run_and_summarize(cfg(label, 700 + i as u64, h, e, d, rt, vec![]))
        })
        .collect();
    print_table("LARGE BATTLES", &results);
}
