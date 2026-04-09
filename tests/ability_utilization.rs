//! Ability utilization eval: measures how well heroes use their ability kits.
//!
//! Tracks per-hero metrics across many scenarios:
//! - Ability uses vs basic attacks
//! - Damage from abilities vs basic attacks
//! - Ability category breakdown (damage/heal/cc/defense/utility)
//! - Win rate correlation with ability usage
//!
//! Run with: cargo test --test ability_utilization -- --nocapture

#![allow(dead_code, unused_variables, unused_imports)]

use std::collections::{HashMap, VecDeque};

use game::ai::core::{step, sim_vec2, SimEvent, SimState, Team, UnitState, FIXED_TICK_MS};
use game::ai::squad;
use game::scenario::{run_scenario_to_state, ScenarioCfg};

const MAX_TICKS: u64 = 3000;
const SEEDS: u64 = 50;

// ---------------------------------------------------------------------------
// Per-hero metrics
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct HeroMetrics {
    scenarios: u32,
    wins: u32,
    total_ticks: u64,
    // Action counts
    basic_attacks: u32,
    ability_uses: u32,
    // Ability category counts
    abilities_by_hint: HashMap<String, u32>,
    abilities_by_name: HashMap<String, u32>,
    // Damage tracking
    damage_from_attacks: i64,
    damage_from_abilities: i64,
    total_damage_dealt: i64,
    total_healing_done: i64,
    // CC tracking
    controls_applied: u32,
    shields_applied: u32,
    // Conditional/empowered effect tracking
    conditional_effects_fired: u32,
    conditional_effects_by_type: HashMap<String, u32>,
}

// ---------------------------------------------------------------------------
// Sim runner with event tracking
// ---------------------------------------------------------------------------

fn run_scenario_tracked(cfg: &ScenarioCfg) -> (String, u64, Vec<SimEvent>) {
    let (mut sim, mut squad_ai) = run_scenario_to_state(cfg);
    let mut all_events = Vec::new();

    for _ in 0..MAX_TICKS {
        let intents = squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        all_events.extend(events);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return ("Victory".into(), sim.tick, all_events);
        }
        if heroes_alive == 0 {
            return ("Defeat".into(), sim.tick, all_events);
        }
    }
    ("Timeout".into(), sim.tick, all_events)
}

/// Determine which unit IDs are heroes and collect their names.
fn hero_info(cfg: &ScenarioCfg) -> Vec<(u32, String)> {
    let (sim, _) = run_scenario_to_state(cfg);
    sim.units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .enumerate()
        .map(|(i, u)| {
            let name = if i < cfg.hero_templates.len() {
                cfg.hero_templates[i].clone()
            } else {
                format!("hero_{}", u.id)
            };
            (u.id, name)
        })
        .collect()
}

/// Build a map from unit_id → ability_index → (name, ai_hint).
fn ability_catalog(cfg: &ScenarioCfg) -> HashMap<u32, Vec<(String, String)>> {
    let (sim, _) = run_scenario_to_state(cfg);
    let mut catalog = HashMap::new();
    for u in &sim.units {
        if u.team == Team::Hero {
            let abilities: Vec<(String, String)> = u.abilities
                .iter()
                .map(|slot| (slot.def.name.clone(), slot.def.ai_hint.clone()))
                .collect();
            catalog.insert(u.id, abilities);
        }
    }
    catalog
}

// ---------------------------------------------------------------------------
// Event analysis
// ---------------------------------------------------------------------------

fn analyze_events(
    events: &[SimEvent],
    hero_ids: &HashMap<u32, String>,
    ability_catalog: &HashMap<u32, Vec<(String, String)>>,
    metrics: &mut HashMap<String, HeroMetrics>,
) {
    // Track which ability was last used by each hero (for damage attribution)
    let mut last_ability_tick: HashMap<u32, u64> = HashMap::new();

    for event in events {
        match event {
            SimEvent::AbilityUsed { unit_id, ability_index, ability_name, tick, .. } => {
                if let Some(hero_name) = hero_ids.get(unit_id) {
                    let m = metrics.entry(hero_name.clone()).or_default();
                    m.ability_uses += 1;
                    *m.abilities_by_name.entry(ability_name.clone()).or_insert(0) += 1;

                    if let Some(catalog) = ability_catalog.get(unit_id) {
                        if let Some((_name, hint)) = catalog.get(*ability_index) {
                            *m.abilities_by_hint.entry(hint.clone()).or_insert(0) += 1;
                        }
                    }

                    last_ability_tick.insert(*unit_id, *tick);
                }
            }
            SimEvent::DamageApplied { source_id, amount, tick, .. } => {
                if let Some(hero_name) = hero_ids.get(source_id) {
                    let m = metrics.entry(hero_name.clone()).or_default();
                    m.total_damage_dealt += *amount as i64;

                    // Attribute: if ability was used within last 2 ticks, it's ability damage
                    let is_ability_dmg = last_ability_tick
                        .get(source_id)
                        .map_or(false, |t| tick.saturating_sub(*t) <= 2);

                    if is_ability_dmg {
                        m.damage_from_abilities += *amount as i64;
                    } else {
                        m.damage_from_attacks += *amount as i64;
                        m.basic_attacks += 1;
                    }
                }
            }
            SimEvent::HealApplied { source_id, amount, .. } => {
                if let Some(hero_name) = hero_ids.get(source_id) {
                    let m = metrics.entry(hero_name.clone()).or_default();
                    m.total_healing_done += *amount as i64;
                }
            }
            SimEvent::ControlApplied { source_id, .. } => {
                if let Some(hero_name) = hero_ids.get(source_id) {
                    let m = metrics.entry(hero_name.clone()).or_default();
                    m.controls_applied += 1;
                }
            }
            SimEvent::ShieldApplied { unit_id, .. } => {
                if hero_ids.contains_key(unit_id) {
                }
            }
            SimEvent::ConditionalEffectApplied { unit_id, condition, .. } => {
                if let Some(hero_name) = hero_ids.get(unit_id) {
                    let m = metrics.entry(hero_name.clone()).or_default();
                    m.conditional_effects_fired += 1;
                    *m.conditional_effects_by_type.entry(condition.clone()).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Test: solo hero ability utilization
// ---------------------------------------------------------------------------

#[test]
fn ability_utilization_solo() {
    let heroes = ["warrior", "ranger", "mage", "cleric", "rogue"];
    let enemy_counts = [2, 4];

    println!();
    println!(
        "{:<10} {:>3}  {:>5} {:>5}  {:>6} {:>6} {:>5}  {:>4} {:>4} {:>4} {:>4} {:>4}  {:>5}",
        "Hero", "vs", "W%", "Ticks", "Atks", "Abils", "Ratio",
        "dmg", "heal", "cc", "def", "util",
        "AbDmg%"
    );
    println!("{}", "-".repeat(100));

    for hero in &heroes {
        for &enemies in &enemy_counts {
            let mut metrics: HashMap<String, HeroMetrics> = HashMap::new();
            let mut total_wins = 0u32;
            let mut total_ticks = 0u64;

            let cfg_template = ScenarioCfg {
                name: format!("{hero}_solo"),
                seed: 0,
                hero_count: 1,
                enemy_count: enemies,
                difficulty: 1,
                max_ticks: MAX_TICKS,
                room_type: "Entry".to_string(),
                hero_templates: vec![hero.to_string()],
                enemy_hero_templates: Vec::new(),
                hp_multiplier: 1.0,
                ..Default::default()
            };

            let hero_names = hero_info(&cfg_template);
            let hero_id_map: HashMap<u32, String> = hero_names.into_iter().collect();
            let catalog = ability_catalog(&cfg_template);

            for seed in 0..SEEDS {
                let cfg = ScenarioCfg {
                    seed,
                    ..cfg_template.clone()
                };
                let (outcome, ticks, events) = run_scenario_tracked(&cfg);
                if outcome == "Victory" {
                    total_wins += 1;
                }
                total_ticks += ticks;

                analyze_events(&events, &hero_id_map, &catalog, &mut metrics);
            }

            for (name, m) in &mut metrics {
                m.scenarios = SEEDS as u32;
                m.wins = total_wins;
                m.total_ticks = total_ticks;
            }

            // Print
            if let Some(m) = metrics.get(*hero) {
                let win_pct = m.wins as f64 / m.scenarios.max(1) as f64 * 100.0;
                let avg_ticks = m.total_ticks as f64 / m.scenarios.max(1) as f64;
                let avg_atks = m.basic_attacks as f64 / m.scenarios.max(1) as f64;
                let avg_abils = m.ability_uses as f64 / m.scenarios.max(1) as f64;
                let ratio = if avg_atks > 0.0 { avg_abils / avg_atks } else { f64::INFINITY };

                let hint_dmg = *m.abilities_by_hint.get("damage").unwrap_or(&0);
                let hint_heal = *m.abilities_by_hint.get("heal").unwrap_or(&0);
                let hint_cc = *m.abilities_by_hint.get("crowd_control").unwrap_or(&0)
                    + *m.abilities_by_hint.get("control").unwrap_or(&0);
                let hint_def = *m.abilities_by_hint.get("defense").unwrap_or(&0);
                let hint_util = *m.abilities_by_hint.get("utility").unwrap_or(&0);

                let total_dmg = m.damage_from_attacks + m.damage_from_abilities;
                let ab_dmg_pct = if total_dmg > 0 {
                    m.damage_from_abilities as f64 / total_dmg as f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "{:<10} {:>3}  {:>4.0}% {:>5.0}  {:>5.1} {:>5.1} {:>5.2}  {:>4} {:>4} {:>4} {:>4} {:>4}  {:>4.0}%",
                    hero, enemies, win_pct, avg_ticks,
                    avg_atks, avg_abils, ratio,
                    hint_dmg, hint_heal, hint_cc, hint_def, hint_util,
                    ab_dmg_pct,
                );
            }
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Test: party ability utilization
// ---------------------------------------------------------------------------

#[test]
fn ability_utilization_party() {
    let party = vec![
        "warrior".to_string(),
        "ranger".to_string(),
        "cleric".to_string(),
        "rogue".to_string(),
    ];
    let enemy_counts = [4, 8, 12];

    println!();
    println!(
        "{:<10} {:>3}  {:>5} {:>5}  {:>5} {:>5} {:>5}  {:>4} {:>4} {:>4} {:>4} {:>4}  {:>5} {:>6}",
        "Hero", "vs", "W%", "Ticks", "Atks", "Abils", "Ratio",
        "dmg", "heal", "cc", "def", "util",
        "AbDmg%", "HealTot"
    );
    println!("{}", "-".repeat(110));

    for &enemies in &enemy_counts {
        let mut metrics: HashMap<String, HeroMetrics> = HashMap::new();
        let mut total_wins = 0u32;
        let mut total_ticks = 0u64;

        let cfg_template = ScenarioCfg {
            name: format!("party_vs_{enemies}"),
            seed: 0,
            hero_count: 4,
            enemy_count: enemies,
            difficulty: 1,
            max_ticks: MAX_TICKS,
            room_type: "Entry".to_string(),
            hero_templates: party.clone(),
            enemy_hero_templates: Vec::new(),
            hp_multiplier: 1.0,
            ..Default::default()
        };

        let hero_names = hero_info(&cfg_template);
        let hero_id_map: HashMap<u32, String> = hero_names.into_iter().collect();
        let catalog = ability_catalog(&cfg_template);

        for seed in 0..SEEDS {
            let cfg = ScenarioCfg {
                seed,
                ..cfg_template.clone()
            };
            let (outcome, ticks, events) = run_scenario_tracked(&cfg);
            if outcome == "Victory" {
                total_wins += 1;
            }
            total_ticks += ticks;

            analyze_events(&events, &hero_id_map, &catalog, &mut metrics);

            // Mark wins on each hero
            for m in metrics.values_mut() {
                m.scenarios = (seed + 1) as u32;
            }
        }

        for m in metrics.values_mut() {
            m.wins = total_wins;
            m.total_ticks = total_ticks;
        }

        let win_pct = total_wins as f64 / SEEDS as f64 * 100.0;

        for hero in &party {
            if let Some(m) = metrics.get(hero.as_str()) {
                let avg_ticks = m.total_ticks as f64 / SEEDS as f64;
                let avg_atks = m.basic_attacks as f64 / SEEDS as f64;
                let avg_abils = m.ability_uses as f64 / SEEDS as f64;
                let ratio = if avg_atks > 0.0 { avg_abils / avg_atks } else { f64::INFINITY };

                let hint_dmg = *m.abilities_by_hint.get("damage").unwrap_or(&0);
                let hint_heal = *m.abilities_by_hint.get("heal").unwrap_or(&0);
                let hint_cc = *m.abilities_by_hint.get("crowd_control").unwrap_or(&0)
                    + *m.abilities_by_hint.get("control").unwrap_or(&0);
                let hint_def = *m.abilities_by_hint.get("defense").unwrap_or(&0);
                let hint_util = *m.abilities_by_hint.get("utility").unwrap_or(&0);

                let total_dmg = m.damage_from_attacks + m.damage_from_abilities;
                let ab_dmg_pct = if total_dmg > 0 {
                    m.damage_from_abilities as f64 / total_dmg as f64 * 100.0
                } else {
                    0.0
                };

                println!(
                    "{:<10} {:>3}  {:>4.0}% {:>5.0}  {:>5.1} {:>5.1} {:>5.2}  {:>4} {:>4} {:>4} {:>4} {:>4}  {:>4.0}% {:>6}",
                    hero, enemies, win_pct, avg_ticks,
                    avg_atks, avg_abils, ratio,
                    hint_dmg, hint_heal, hint_cc, hint_def, hint_util,
                    ab_dmg_pct, m.total_healing_done,
                );
            }
        }
        println!();
    }

    // --- Per-ability breakdown for party vs 8 ---
    println!("\n=== Per-Ability Breakdown (Party vs 8, {} seeds) ===", SEEDS);
    println!(
        "{:<10} {:<20} {:>6}  {:>5}",
        "Hero", "Ability", "Uses", "Avg/Scn"
    );
    println!("{}", "-".repeat(50));

    let cfg_final = ScenarioCfg {
        name: "party_vs_8_detail".into(),
        seed: 0,
        hero_count: 4,
        enemy_count: 8,
        difficulty: 1,
        max_ticks: MAX_TICKS,
        room_type: "Entry".to_string(),
        hero_templates: party.clone(),
        enemy_hero_templates: Vec::new(),
        hp_multiplier: 1.0,
        ..Default::default()
    };

    let mut detail_metrics: HashMap<String, HeroMetrics> = HashMap::new();
    let hero_names = hero_info(&cfg_final);
    let hero_id_map: HashMap<u32, String> = hero_names.into_iter().collect();
    let catalog = ability_catalog(&cfg_final);

    for seed in 0..SEEDS {
        let cfg = ScenarioCfg {
            seed,
            ..cfg_final.clone()
        };
        let (_outcome, _ticks, events) = run_scenario_tracked(&cfg);
        analyze_events(&events, &hero_id_map, &catalog, &mut detail_metrics);
    }

    for hero in &party {
        if let Some(m) = detail_metrics.get(hero.as_str()) {
            let mut abilities: Vec<_> = m.abilities_by_name.iter().collect();
            abilities.sort_by(|a, b| b.1.cmp(a.1));
            for (name, count) in abilities {
                println!(
                    "{:<10} {:<20} {:>6}  {:>5.1}",
                    hero, name, count, *count as f64 / SEEDS as f64,
                );
            }
            if m.abilities_by_name.is_empty() {
                println!("{:<10} {:<20} {:>6}  {:>5.1}", hero, "(none)", 0, 0.0);
            }
        }
    }

    // --- Conditional/Empowered effect breakdown ---
    println!("\n=== Conditional Effect Utilization (Party vs 8, {} seeds) ===", SEEDS);
    println!(
        "{:<10} {:>6}  {:<40} {:>5}",
        "Hero", "Total", "Condition", "Count"
    );
    println!("{}", "-".repeat(70));

    for hero in &party {
        if let Some(m) = detail_metrics.get(hero.as_str()) {
            let total = m.conditional_effects_fired;
            let avg = total as f64 / SEEDS as f64;
            if total == 0 {
                println!("{:<10} {:>6}  {:<40} {:>5}", hero, 0, "(none)", "-");
                continue;
            }
            let mut conditions: Vec<_> = m.conditional_effects_by_type.iter().collect();
            conditions.sort_by(|a, b| b.1.cmp(a.1));
            for (i, (cond, count)) in conditions.iter().enumerate() {
                let cond_short = if cond.len() > 38 { &cond[..38] } else { cond.as_str() };
                if i == 0 {
                    println!(
                        "{:<10} {:>5} ({:.1}/s)  {:<38} {:>5}",
                        hero, total, avg, cond_short, count,
                    );
                } else {
                    println!(
                        "{:<10} {:>12}  {:<38} {:>5}",
                        "", "", cond_short, count,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test: rogue vs dummy — isolated ability rotation analysis
// ---------------------------------------------------------------------------

fn make_dummy(id: u32, hp: i32, position: game::ai::core::SimVec2) -> UnitState {
    UnitState {
        id,
        team: Team::Enemy,
        hp,
        max_hp: hp,
        position,
        move_speed_per_sec: 0.0,
        attack_damage: 0,
        attack_range: 0.0,
        attack_cooldown_ms: 99999,
        attack_cast_time_ms: 0,
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

#[test]
fn rogue_vs_dummy() {
    // Build a rogue from template, then swap enemies for a single high-HP dummy
    let cfg = ScenarioCfg {
        name: "rogue_dummy".into(),
        seed: 42,
        hero_count: 1,
        enemy_count: 1,
        difficulty: 1,
        max_ticks: 2000,
        room_type: "Entry".to_string(),
        hero_templates: vec!["rogue".to_string()],
        enemy_hero_templates: Vec::new(),
        hp_multiplier: 1.0,
        ..Default::default()
    };

    let (mut sim, _) = run_scenario_to_state(&cfg);

    // Replace the enemy with a high-HP dummy at close range
    // Dummy deals damage (20 dps) so threat reduction from CC is meaningful
    sim.units.retain(|u| u.team == Team::Hero);
    let rogue_pos = sim.units[0].position;
    let dummy_pos = sim_vec2(rogue_pos.x + 3.0, rogue_pos.y);
    let mut dummy = make_dummy(100, 5000, dummy_pos);
    dummy.attack_damage = 10;
    dummy.attack_range = 1.5;
    dummy.attack_cooldown_ms = 1200;
    dummy.attack_cast_time_ms = 300;
    sim.units.push(dummy);

    // Give the rogue extra HP to survive longer
    sim.units[0].hp = 400;
    sim.units[0].max_hp = 400;

    // Rebuild AI after modifying units
    let mut squad_ai = squad::SquadAiState::new_inferred(&sim);

    let mut all_events = Vec::new();
    let max_ticks = 2000u64;

    for _ in 0..max_ticks {
        let intents = squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        all_events.extend(events);
        sim = new_sim;

        // Stop if dummy dies (shouldn't with 5000 HP)
        let dummy_alive = sim.units.iter().any(|u| u.team == Team::Enemy && u.hp > 0);
        if !dummy_alive {
            break;
        }
    }

    // Analyze: print tick-by-tick ability timeline
    println!();
    println!("=== Rogue vs Dummy (5000 HP, 0 dmg, 0 move) — {} ticks ===", sim.tick);
    println!();

    let mut ability_uses: HashMap<String, u32> = HashMap::new();
    let mut conditional_fires: HashMap<String, u32> = HashMap::new();
    let mut total_damage = 0i64;
    let mut ability_damage = 0i64;
    let mut basic_attacks = 0u32;
    let mut last_ability_tick = 0u64;
    let mut timeline: Vec<String> = Vec::new();

    for event in &all_events {
        match event {
            SimEvent::AbilityUsed { tick, ability_name, unit_id, .. } => {
                if sim.units.iter().any(|u| u.id == *unit_id && u.team == Team::Hero) || *unit_id == 1 {
                    *ability_uses.entry(ability_name.clone()).or_insert(0) += 1;
                    last_ability_tick = *tick;
                    timeline.push(format!("t={:>4}  USE  {}", tick, ability_name));
                }
            }
            SimEvent::ConditionalEffectApplied { tick, unit_id, condition, .. } => {
                if *unit_id == 1 {
                    *conditional_fires.entry(condition.clone()).or_insert(0) += 1;
                    timeline.push(format!("t={:>4}  COND {}", tick, condition));
                }
            }
            SimEvent::ControlApplied { tick, source_id, target_id, duration_ms, .. } => {
                if *source_id == 1 {
                    timeline.push(format!("t={:>4}  STUN target={} dur={}ms", tick, target_id, duration_ms));
                }
            }
            SimEvent::DamageApplied { tick, source_id, amount, .. } => {
                if *source_id == 1 {
                    total_damage += *amount as i64;
                    if tick.saturating_sub(last_ability_tick) <= 2 {
                        ability_damage += *amount as i64;
                    } else {
                        basic_attacks += 1;
                    }
                }
            }
            _ => {}
        }
    }

    // Print first 80 timeline entries
    println!("--- Timeline (first 80 events) ---");
    for line in timeline.iter().take(80) {
        println!("  {}", line);
    }
    if timeline.len() > 80 {
        println!("  ... ({} more events)", timeline.len() - 80);
    }

    println!();
    println!("--- Ability Uses ---");
    let mut sorted_abilities: Vec<_> = ability_uses.iter().collect();
    sorted_abilities.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in &sorted_abilities {
        println!("  {:<20} {:>4}", name, count);
    }

    println!();
    println!("--- Conditional Effects Fired ---");
    if conditional_fires.is_empty() {
        println!("  (none)");
    } else {
        let mut sorted_conds: Vec<_> = conditional_fires.iter().collect();
        sorted_conds.sort_by(|a, b| b.1.cmp(a.1));
        for (cond, count) in &sorted_conds {
            println!("  {:<40} {:>4}", cond, count);
        }
    }

    println!();
    println!("--- Damage Summary ---");
    println!("  Total damage:     {:>6}", total_damage);
    println!("  Ability damage:   {:>6} ({:.0}%)", ability_damage, if total_damage > 0 { ability_damage as f64 / total_damage as f64 * 100.0 } else { 0.0 });
    println!("  Basic attacks:    {:>6}", basic_attacks);
    println!("  Dummy HP remain:  {:>6}", sim.units.iter().find(|u| u.id == 100).map_or(0, |u| u.hp));

    // Check: Garrote should fire, then Backstab conditional should trigger
    let garrote_uses = ability_uses.get("Garrote").copied().unwrap_or(0);
    let backstab_uses = ability_uses.get("Backstab").copied().unwrap_or(0);
    let stun_conditionals = conditional_fires.get("TargetIsStunned").copied().unwrap_or(0);

    println!();
    println!("--- Combo Analysis ---");
    println!("  Garrote uses:               {:>4}", garrote_uses);
    println!("  Backstab uses:              {:>4}", backstab_uses);
    println!("  TargetIsStunned conditionals: {:>4}", stun_conditionals);
    let combo_rate = if backstab_uses > 0 {
        stun_conditionals as f64 / backstab_uses as f64 * 100.0
    } else {
        0.0
    };
    println!("  Backstab combo rate:        {:>4.0}%", combo_rate);
}
