//! LoL Hero Ability Efficacy Scoring
//!
//! Runs every LoL hero TOML in a standardized 1v3 sim (500 ticks, 3 seeds),
//! collects per-ability stats, and reports efficacy scores.
//!
//! Run: cargo test --test lol_ability_efficacy -- --nocapture

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;

use bevy_game::ai::core::{sim_vec2, step, UnitIntent, IntentAction, SimEvent, SimState, SimVec2, Team, UnitState, FIXED_TICK_MS};
use bevy_game::ai::effects::{AbilityTarget, AbilityTargeting, HeroToml};
use bevy_game::ai::squad;
use bevy_game::mission::hero_templates::{hero_toml_to_unit, parse_hero_toml};

const MAX_TICKS: u64 = 2000;
const SEEDS: u64 = 3;
const ENEMY_COUNT: usize = 4;

fn make_enemy(id: u32, position: SimVec2) -> UnitState {
    UnitState {
        id,
        team: Team::Enemy,
        hp: 100,
        max_hp: 100,
        position,
        move_speed_per_sec: 3.0,
        attack_damage: 12,
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
        total_damage_done: 0,
        total_healing_done: 0,
        cover_bonus: 0.0,
        elevation: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Per-ability metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct AbilityMetrics {
    name: String,
    times_used: u32,
    damage_dealt: i64,
    healing_done: i64,
    shield_granted: i64,
    cc_applied: u32,
    cc_duration_ms: u64,
}

#[derive(Debug, Clone)]
struct HeroReport {
    champion: String,
    wins: u32,
    total_damage: i64,
    total_healing: i64,
    total_shielding: i64,
    total_cc_count: u32,
    total_cc_duration_ms: u64,
    kills: u32,
    deaths: u32,
    abilities_used: u32,
    passives_triggered: u32,
    abilities: Vec<AbilityMetrics>,
    /// How many distinct abilities were used at least once (out of total defined)
    ability_coverage: (u32, u32),
    /// Efficacy score: weighted composite
    efficacy_score: f64,
}

fn compute_efficacy(r: &HeroReport) -> f64 {
    // Weighted composite:
    //   damage: 1 point per 10 damage dealt
    //   healing: 1 point per 10 healing
    //   shielding: 0.8 points per 10 shielding
    //   CC: 3 points per CC application + 0.5 per 1000ms CC duration
    //   kills: 15 points per kill
    //   wins: 30 points per win
    //   ability coverage: 10 points per unique ability used
    //   penalty: -5 per death
    let seeds = SEEDS as f64;
    let dmg = r.total_damage as f64 / seeds / 10.0;
    let heal = r.total_healing as f64 / seeds / 10.0;
    let shield = r.total_shielding as f64 / seeds * 0.8 / 10.0;
    let cc = r.total_cc_count as f64 / seeds * 3.0
        + r.total_cc_duration_ms as f64 / seeds / 1000.0 * 0.5;
    let kills = r.kills as f64 / seeds * 15.0;
    let wins = r.wins as f64 * 30.0;
    let coverage = r.ability_coverage.0 as f64 * 10.0;
    let death_penalty = r.deaths as f64 / seeds * 5.0;

    dmg + heal + shield + cc + kills + wins + coverage - death_penalty
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn lol_ability_efficacy_report() {
    let dir = Path::new("assets/lol_heroes");
    let mut files: Vec<_> = fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "toml"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    let mut reports: Vec<HeroReport> = Vec::new();
    let mut dead_abilities: Vec<(String, String)> = Vec::new(); // (champion, ability_name)

    for entry in &files {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let content = fs::read_to_string(&path).unwrap();
        let toml: HeroToml = match parse_hero_toml(&content) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let defined_abilities: Vec<String> = toml.abilities.iter().map(|a| a.name.clone()).collect();
        let num_defined = defined_abilities.len() as u32;

        let mut agg_abilities: HashMap<String, AbilityMetrics> = HashMap::new();
        let mut total_damage: i64 = 0;
        let mut total_healing: i64 = 0;
        let mut total_shielding: i64 = 0;
        let mut total_cc_count: u32 = 0;
        let mut total_cc_duration_ms: u64 = 0;
        let mut kills: u32 = 0;
        let mut deaths: u32 = 0;
        let mut abilities_used: u32 = 0;
        let mut passives_triggered: u32 = 0;
        let mut wins: u32 = 0;

        for seed in 0..SEEDS {
            let mut hero = hero_toml_to_unit(&toml, 1, Team::Hero, sim_vec2(0.0, 0.0));
            // 3x HP so the fight lasts long enough to cycle all ability cooldowns
            hero.hp *= 3;
            hero.max_hp *= 3;
            // Add an ally so target_ally abilities have a valid target
            let mut ally = make_enemy(2, sim_vec2(-1.0, 1.0));
            ally.team = Team::Hero;
            ally.hp = 200;
            ally.max_hp = 200;
            let enemies: Vec<UnitState> = (0..ENEMY_COUNT).map(|i| {
                make_enemy(100 + i as u32, sim_vec2(5.0 + i as f32 * 1.5, 1.0))
            }).collect();

            let mut units = vec![hero, ally];
            units.extend(enemies);

            let mut sim = SimState {
                tick: 0,
                rng_state: seed,
                units,
                projectiles: Vec::new(),
                passive_trigger_depth: 0,
                zones: Vec::new(),
                tethers: Vec::new(),
                grid_nav: None,
            };

            let mut squad_ai = squad::SquadAiState::new_inferred(&sim);

            // Track per-ability stats via events
            let mut active_ability: HashMap<u32, String> = HashMap::new();
            let mut seed_damage: i64 = 0;
            let mut seed_healing: i64 = 0;
            let mut seed_shielding: i64 = 0;
            let mut seed_cc_count: u32 = 0;
            let mut seed_cc_duration_ms: u64 = 0;
            let mut seed_kills: u32 = 0;
            let mut seed_deaths: u32 = 0;
            let mut seed_abilities_used: u32 = 0;
            let mut seed_passives_triggered: u32 = 0;
            let mut outcome = "Timeout";

            for _ in 0..MAX_TICKS {
                let intents = squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
                let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
                sim = new_sim;

                for ev in &events {
                    match ev {
                        SimEvent::AbilityUsed { unit_id, ability_name, .. } if *unit_id == 1 => {
                            seed_abilities_used += 1;
                            active_ability.insert(1, ability_name.clone());
                            let entry = agg_abilities.entry(ability_name.clone()).or_insert_with(|| {
                                AbilityMetrics { name: ability_name.clone(), ..Default::default() }
                            });
                            entry.times_used += 1;
                        }
                        SimEvent::PassiveTriggered { unit_id, passive_name, .. } if *unit_id == 1 => {
                            seed_passives_triggered += 1;
                            active_ability.insert(1, passive_name.clone());
                        }
                        SimEvent::DamageApplied { source_id, amount, target_hp_after, .. } if *source_id == 1 => {
                            let amt = *amount as i64;
                            seed_damage += amt;
                            if let Some(ab_name) = active_ability.get(&1u32) {
                                let entry = agg_abilities.entry(ab_name.clone()).or_insert_with(|| {
                                    AbilityMetrics { name: ab_name.clone(), ..Default::default() }
                                });
                                entry.damage_dealt += amt;
                            }
                        }
                        SimEvent::HealApplied { source_id, amount, .. } if *source_id == 1 => {
                            let amt = *amount as i64;
                            seed_healing += amt;
                            if let Some(ab_name) = active_ability.get(&1u32) {
                                let entry = agg_abilities.entry(ab_name.clone()).or_insert_with(|| {
                                    AbilityMetrics { name: ab_name.clone(), ..Default::default() }
                                });
                                entry.healing_done += amt;
                            }
                        }
                        SimEvent::ShieldApplied { unit_id, amount, .. } if *unit_id == 1 => {
                            seed_shielding += *amount as i64;
                            if let Some(ab_name) = active_ability.get(&1u32) {
                                let entry = agg_abilities.entry(ab_name.clone()).or_insert_with(|| {
                                    AbilityMetrics { name: ab_name.clone(), ..Default::default() }
                                });
                                entry.shield_granted += *amount as i64;
                            }
                        }
                        SimEvent::ControlApplied { source_id, duration_ms, .. } if *source_id == 1 => {
                            seed_cc_count += 1;
                            seed_cc_duration_ms += *duration_ms as u64;
                            if let Some(ab_name) = active_ability.get(&1u32) {
                                let entry = agg_abilities.entry(ab_name.clone()).or_insert_with(|| {
                                    AbilityMetrics { name: ab_name.clone(), ..Default::default() }
                                });
                                entry.cc_applied += 1;
                                entry.cc_duration_ms += *duration_ms as u64;
                            }
                        }
                        SimEvent::UnitDied { unit_id, .. } => {
                            if *unit_id == 1 {
                                seed_deaths += 1;
                            } else {
                                seed_kills += 1;
                            }
                        }
                        _ => {}
                    }
                }

                let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
                if enemies_alive == 0 { outcome = "Victory"; break; }
                let hero_alive = sim.units.iter().any(|u| u.id == 1 && u.hp > 0);
                if !hero_alive { outcome = "Defeat"; break; }
            }

            if outcome == "Victory" { wins += 1; }
            total_damage += seed_damage;
            total_healing += seed_healing;
            total_shielding += seed_shielding;
            total_cc_count += seed_cc_count;
            total_cc_duration_ms += seed_cc_duration_ms;
            kills += seed_kills;
            deaths += seed_deaths;
            abilities_used += seed_abilities_used;
            passives_triggered += seed_passives_triggered;
        }

        // Find unused abilities
        let used_names: std::collections::HashSet<&str> =
            agg_abilities.keys().map(|s| s.as_str()).collect();
        let mut unused_indices: Vec<(usize, String)> = Vec::new();
        let mut num_used = 0u32;
        for (idx, ab_name) in defined_abilities.iter().enumerate() {
            if used_names.contains(ab_name.as_str()) {
                num_used += 1;
            } else {
                unused_indices.push((idx, ab_name.clone()));
            }
        }

        // Forced-fire pass: for each unused ability, set up a fresh sim state,
        // reset cooldowns, and manually fire the ability to verify it works.
        for (ab_idx, ab_name) in &unused_indices {
            let mut hero = hero_toml_to_unit(&toml, 1, Team::Hero, sim_vec2(0.0, 0.0));
            // Reset all ability cooldowns and ensure enough resource
            let ab_range = hero.abilities.get(*ab_idx).map_or(1.5, |s| s.def.range);
            for slot in hero.abilities.iter_mut() {
                slot.cooldown_remaining_ms = 0;
            }
            let needed = hero.abilities.get(*ab_idx).map_or(0, |s| s.def.resource_cost);
            hero.resource = hero.max_resource.max(needed as i32);
            hero.max_resource = hero.max_resource.max(needed as i32);
            let mut ally = make_enemy(2, sim_vec2(-1.0, 1.0));
            ally.team = Team::Hero;
            ally.hp = 50; // Low HP so heal/shield abilities have valid context
            ally.max_hp = 200;
            // Spawn enemies within ability range
            let spawn_dist = if ab_range > 0.0 { ab_range * 0.8 } else { 1.0 };
            let enemies: Vec<UnitState> = (0..2).map(|i| {
                make_enemy(100 + i as u32, sim_vec2(spawn_dist + i as f32 * 0.5, 0.0))
            }).collect();
            let mut units = vec![hero, ally];
            units.extend(enemies);
            let mut forced_sim = SimState {
                tick: 0, rng_state: 99, units,
                projectiles: Vec::new(), passive_trigger_depth: 0,
                zones: Vec::new(), tethers: Vec::new(), grid_nav: None,
            };

            // Determine appropriate target for the ability's targeting type
            let targeting = &forced_sim.units[0].abilities[*ab_idx].def.targeting;
            let target = match targeting {
                AbilityTargeting::TargetEnemy => AbilityTarget::Unit(100),
                AbilityTargeting::TargetAlly => AbilityTarget::Unit(2),
                AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
                    AbilityTarget::Position(sim_vec2(3.0, 0.0))
                }
                _ => AbilityTarget::None,
            };

            let intent = UnitIntent {
                unit_id: 1,
                action: IntentAction::UseAbility {
                    ability_index: *ab_idx,
                    target,
                },
            };

            // Step a few ticks to let projectiles/effects resolve
            let mut forced_used = false;
            for _ in 0..20 {
                let intents = if forced_sim.tick == 0 {
                    vec![intent]
                } else {
                    vec![UnitIntent { unit_id: 1, action: IntentAction::Hold }]
                };
                let (new_sim, events) = step(forced_sim, &intents, FIXED_TICK_MS);
                forced_sim = new_sim;
                for ev in &events {
                    if matches!(ev, SimEvent::AbilityUsed { unit_id: 1, .. }) {
                        forced_used = true;
                    }
                }
            }

            if forced_used {
                num_used += 1;
                let entry = agg_abilities.entry(ab_name.clone()).or_insert_with(|| {
                    AbilityMetrics { name: ab_name.clone(), ..Default::default() }
                });
                entry.times_used += 1; // Mark as used via forced-fire
            } else {
                dead_abilities.push((name.clone(), ab_name.clone()));
            }
        }

        let mut abilities: Vec<AbilityMetrics> = agg_abilities.into_values().collect();
        abilities.sort_by(|a, b| b.damage_dealt.cmp(&a.damage_dealt));

        let mut report = HeroReport {
            champion: name,
            wins,
            total_damage,
            total_healing,
            total_shielding,
            total_cc_count,
            total_cc_duration_ms,
            kills,
            deaths,
            abilities_used,
            passives_triggered,
            abilities,
            ability_coverage: (num_used, num_defined),
            efficacy_score: 0.0,
        };
        report.efficacy_score = compute_efficacy(&report);
        reports.push(report);
    }

    // Sort by efficacy score descending
    reports.sort_by(|a, b| b.efficacy_score.partial_cmp(&a.efficacy_score).unwrap());

    // --- Print Champion Rankings ---
    println!();
    println!(
        "{:<18} {:>5} {:>6} {:>6} {:>6} {:>4} {:>6} {:>3} {:>3} {:>5} {:>5} {:>7}",
        "Champion", "Score", "Dmg", "Heal", "Shld", "CC#", "CC_ms", "K", "D", "Ab#", "Cov", "W/L"
    );
    println!("{}", "-".repeat(100));

    for r in &reports {
        let cov = format!("{}/{}", r.ability_coverage.0, r.ability_coverage.1);
        let wl = format!("{}/{}", r.wins, SEEDS);
        println!(
            "{:<18} {:>5.0} {:>6} {:>6} {:>6} {:>4} {:>6} {:>3} {:>3} {:>5} {:>5} {:>7}",
            r.champion,
            r.efficacy_score,
            r.total_damage / SEEDS as i64,
            r.total_healing / SEEDS as i64,
            r.total_shielding / SEEDS as i64,
            r.total_cc_count / SEEDS as u32,
            r.total_cc_duration_ms / SEEDS as u64,
            r.kills / SEEDS as u32,
            r.deaths / SEEDS as u32,
            r.abilities_used / SEEDS as u32,
            cov,
            wl,
        );
    }

    // --- Per-ability breakdown for top 10 ---
    println!();
    println!("=== Top 10 Champions — Ability Breakdown ===");
    for r in reports.iter().take(10) {
        println!();
        println!("  {} (score={:.0}, {}/{})", r.champion, r.efficacy_score,
                 r.wins, SEEDS);
        println!(
            "    {:<30} {:>5} {:>6} {:>6} {:>6} {:>4} {:>6}",
            "Ability", "Used", "Dmg", "Heal", "Shld", "CC#", "CC_ms"
        );
        for ab in &r.abilities {
            println!(
                "    {:<30} {:>5} {:>6} {:>6} {:>6} {:>4} {:>6}",
                ab.name,
                ab.times_used / SEEDS as u32,
                ab.damage_dealt / SEEDS as i64,
                ab.healing_done / SEEDS as i64,
                ab.shield_granted / SEEDS as i64,
                ab.cc_applied / SEEDS as u32,
                ab.cc_duration_ms / SEEDS as u64,
            );
        }
    }

    // --- Dead abilities (never used) ---
    println!();
    println!("=== Unused Abilities ({}) ===", dead_abilities.len());
    for (champ, ab) in &dead_abilities {
        println!("  {champ}: {ab}");
    }

    // --- Bottom 10 ---
    println!();
    println!("=== Bottom 10 Champions ===");
    println!(
        "{:<18} {:>5} {:>6} {:>6} {:>4} {:>5} {:>7}",
        "Champion", "Score", "Dmg", "Heal", "CC#", "Cov", "W/L"
    );
    for r in reports.iter().rev().take(10) {
        let cov = format!("{}/{}", r.ability_coverage.0, r.ability_coverage.1);
        let wl = format!("{}/{}", r.wins, SEEDS);
        println!(
            "{:<18} {:>5.0} {:>6} {:>6} {:>4} {:>5} {:>7}",
            r.champion,
            r.efficacy_score,
            r.total_damage / SEEDS as i64,
            r.total_healing / SEEDS as i64,
            r.total_cc_count / SEEDS as u32,
            cov,
            wl,
        );
    }

    // --- Summary stats ---
    let avg_score: f64 = reports.iter().map(|r| r.efficacy_score).sum::<f64>() / reports.len() as f64;
    let avg_coverage: f64 = reports.iter().map(|r| r.ability_coverage.0 as f64 / r.ability_coverage.1 as f64).sum::<f64>() / reports.len() as f64;
    let total_wins: u32 = reports.iter().map(|r| r.wins).sum();
    let zero_dmg = reports.iter().filter(|r| r.total_damage == 0).count();

    println!();
    println!("=== Summary ===");
    println!("  Champions tested:    {}", reports.len());
    println!("  Avg efficacy score:  {avg_score:.1}");
    println!("  Avg ability coverage: {:.0}%", avg_coverage * 100.0);
    println!("  Total wins (1v{}):  {total_wins}/{}", ENEMY_COUNT, reports.len() as u64 * SEEDS);
    println!("  Zero-damage heroes:  {zero_dmg}");
    println!("  Unused abilities:    {}", dead_abilities.len());
}
