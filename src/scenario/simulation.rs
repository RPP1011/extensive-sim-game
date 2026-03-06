use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::ai::core::{step, SimEvent, Team, FIXED_TICK_MS};

use super::types::*;
use super::runner::run_scenario_to_state;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find or create an AbilityStats entry for the given ability name.
fn get_or_create_ability_stats<'a>(
    ability_stats: &'a mut Vec<AbilityStats>,
    ability_name: &str,
) -> &'a mut AbilityStats {
    let idx = ability_stats
        .iter()
        .position(|a| a.ability_name == ability_name);
    match idx {
        Some(i) => &mut ability_stats[i],
        None => {
            ability_stats.push(AbilityStats {
                ability_name: ability_name.to_string(),
                times_used: 0,
                damage_dealt: 0,
                healing_done: 0,
                shield_granted: 0,
                cc_applied_count: 0,
                cc_duration_ms: 0,
            });
            ability_stats.last_mut().unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// Core runner
// ---------------------------------------------------------------------------

pub fn run_scenario(cfg: &ScenarioCfg) -> ScenarioResult {
    run_scenario_impl(cfg, None)
}

/// Run a scenario with optional ability evaluator weights for interrupt-driven ability usage.
pub fn run_scenario_with_ability_eval(cfg: &ScenarioCfg, weights_path: &Path) -> ScenarioResult {
    run_scenario_impl(cfg, Some(weights_path))
}

fn run_scenario_impl(cfg: &ScenarioCfg, ability_eval_path: Option<&Path>) -> ScenarioResult {
    let (mut sim, mut squad_state) = run_scenario_to_state(cfg);

    if let Some(path) = ability_eval_path {
        if let Err(e) = squad_state.load_ability_eval_weights(path) {
            eprintln!("Warning: failed to load ability eval weights: {e}");
        }
    }

    let hero_ids: HashSet<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    let hero_template_names: Vec<String> = if !cfg.hero_templates.is_empty() {
        cfg.hero_templates.clone()
    } else {
        sim.units
            .iter()
            .filter(|u| u.team == Team::Hero)
            .map(|u| format!("Hero_{}", u.id))
            .collect()
    };

    let mut stats_map: HashMap<u32, UnitStats> = HashMap::new();
    let mut hero_idx = 0usize;
    for u in &sim.units {
        let (team_str, template) = if u.team == Team::Hero {
            let t = hero_template_names
                .get(hero_idx)
                .cloned()
                .unwrap_or_else(|| format!("Hero_{}", u.id));
            hero_idx += 1;
            ("Hero".to_string(), t)
        } else {
            ("Enemy".to_string(), "Enemy".to_string())
        };
        stats_map.insert(
            u.id,
            UnitStats {
                unit_id: u.id,
                team: team_str,
                template,
                max_hp: u.max_hp,
                final_hp: u.hp,
                damage_dealt: 0,
                damage_taken: 0,
                overkill_dealt: 0,
                healing_done: 0,
                healing_received: 0,
                overhealing: 0,
                lifesteal_healing: 0,
                shield_received: 0,
                shield_absorbed: 0,
                cc_applied_count: 0,
                cc_received_count: 0,
                cc_duration_applied_ms: 0,
                abilities_used: 0,
                passives_triggered: 0,
                attacks_missed: 0,
                kills: 0,
                deaths: 0,
                reflect_damage: 0,
                ability_stats: Vec::new(),
            },
        );
    }

    let mut last_damage_source: HashMap<u32, u32> = HashMap::new();
    let mut active_ability: HashMap<u32, String> = HashMap::new();

    let mut event_log: Vec<String> = Vec::new();
    let mut hero_deaths: usize = 0;
    let mut outcome = "Timeout".to_string();

    for _ in 0..cfg.max_ticks {
        let all_intents =
            crate::ai::squad::generate_intents(&sim, &mut squad_state, FIXED_TICK_MS);

        let (new_sim, events) = step(sim, &all_intents, FIXED_TICK_MS);
        sim = new_sim;

        for ev in &events {
            match ev {
                SimEvent::DamageApplied {
                    source_id, target_id, amount, target_hp_after, ..
                } => {
                    let amt = *amount as i64;
                    if let Some(s) = stats_map.get_mut(source_id) {
                        s.damage_dealt += amt;
                        if *target_hp_after < 0 {
                            s.overkill_dealt += (-*target_hp_after) as i64;
                        }
                        if let Some(ab_name) = active_ability.get(source_id) {
                            let ab = get_or_create_ability_stats(&mut s.ability_stats, ab_name);
                            ab.damage_dealt += amt;
                        }
                    }
                    if let Some(t) = stats_map.get_mut(target_id) {
                        t.damage_taken += amt;
                    }
                    last_damage_source.insert(*target_id, *source_id);
                }
                SimEvent::HealApplied {
                    source_id, target_id, amount, target_hp_before, target_hp_after, ..
                } => {
                    let amt = *amount as i64;
                    let actual_heal = (*target_hp_after - *target_hp_before) as i64;
                    let overheal = amt - actual_heal.max(0);
                    if let Some(s) = stats_map.get_mut(source_id) {
                        s.healing_done += amt;
                        if let Some(ab_name) = active_ability.get(source_id) {
                            let ab = get_or_create_ability_stats(&mut s.ability_stats, ab_name);
                            ab.healing_done += amt;
                        }
                    }
                    if let Some(t) = stats_map.get_mut(target_id) {
                        t.healing_received += amt;
                        if overheal > 0 {
                            t.overhealing += overheal;
                        }
                    }
                }
                SimEvent::ShieldApplied { unit_id, amount, .. } => {
                    let amt = *amount as i64;
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.shield_received += amt;
                    }
                    if let Some(ab_name) = active_ability.get(unit_id).cloned() {
                        if let Some(u) = stats_map.get_mut(unit_id) {
                            let ab = get_or_create_ability_stats(&mut u.ability_stats, &ab_name);
                            ab.shield_granted += amt;
                        }
                    }
                }
                SimEvent::ShieldAbsorbed { unit_id, absorbed, .. } => {
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.shield_absorbed += *absorbed as i64;
                    }
                }
                SimEvent::ControlApplied {
                    source_id, target_id, duration_ms, ..
                } => {
                    let dur = *duration_ms as u64;
                    if let Some(s) = stats_map.get_mut(source_id) {
                        s.cc_applied_count += 1;
                        s.cc_duration_applied_ms += dur;
                        if let Some(ab_name) = active_ability.get(source_id) {
                            let ab = get_or_create_ability_stats(&mut s.ability_stats, ab_name);
                            ab.cc_applied_count += 1;
                            ab.cc_duration_ms += dur;
                        }
                    }
                    if let Some(t) = stats_map.get_mut(target_id) {
                        t.cc_received_count += 1;
                    }
                }
                SimEvent::AbilityUsed { unit_id, ability_name, .. } => {
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.abilities_used += 1;
                        let ab = get_or_create_ability_stats(&mut u.ability_stats, ability_name);
                        ab.times_used += 1;
                    }
                    active_ability.insert(*unit_id, ability_name.clone());
                }
                SimEvent::PassiveTriggered { unit_id, passive_name, .. } => {
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.passives_triggered += 1;
                    }
                    active_ability.insert(*unit_id, passive_name.clone());
                }
                SimEvent::AttackMissed { source_id, .. } => {
                    if let Some(s) = stats_map.get_mut(source_id) {
                        s.attacks_missed += 1;
                    }
                }
                SimEvent::UnitDied { tick, unit_id } => {
                    event_log.push(format!("tick={}: unit {} died", tick, unit_id));
                    if hero_ids.contains(unit_id) {
                        hero_deaths += 1;
                    }
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.deaths += 1;
                    }
                    if let Some(&killer_id) = last_damage_source.get(unit_id) {
                        if let Some(k) = stats_map.get_mut(&killer_id) {
                            k.kills += 1;
                        }
                    }
                }
                SimEvent::LifestealHeal { unit_id, amount, .. } => {
                    if let Some(u) = stats_map.get_mut(unit_id) {
                        u.lifesteal_healing += *amount as i64;
                    }
                }
                SimEvent::ReflectDamage { source_id, amount, .. } => {
                    if let Some(s) = stats_map.get_mut(source_id) {
                        s.reflect_damage += *amount as i64;
                    }
                }
                _ => {}
            }
        }

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

        if enemies_alive == 0 {
            outcome = "Victory".to_string();
            break;
        }
        if heroes_alive == 0 {
            outcome = "Defeat".to_string();
            break;
        }
    }

    for u in &sim.units {
        if let Some(s) = stats_map.get_mut(&u.id) {
            s.final_hp = u.hp;
        }
    }

    let mut unit_stats: Vec<UnitStats> = stats_map.into_values().collect();
    unit_stats.sort_by(|a, b| {
        let team_ord = if a.team == "Hero" { 0u8 } else { 1u8 };
        let team_ord_b = if b.team == "Hero" { 0u8 } else { 1u8 };
        team_ord.cmp(&team_ord_b).then(a.unit_id.cmp(&b.unit_id))
    });

    let final_hero_count = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
    let final_enemy_count = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

    ScenarioResult {
        scenario_name: cfg.name.clone(),
        outcome,
        tick: sim.tick,
        passed: true,
        assertions: Vec::new(),
        final_hero_count,
        final_enemy_count,
        events: event_log,
        hero_deaths,
        unit_stats,
    }
}

// ---------------------------------------------------------------------------
// Assertion checking
// ---------------------------------------------------------------------------

pub fn check_assertions(
    result: &ScenarioResult,
    asserts: &ScenarioAssert,
) -> Vec<AssertionResult> {
    let mut out = Vec::new();

    if let Some(expected) = &asserts.outcome {
        let passes = if expected == "Any" {
            true
        } else if expected == "Either" {
            result.outcome == "Victory" || result.outcome == "Defeat"
        } else {
            &result.outcome == expected
        };
        out.push(AssertionResult {
            name: "outcome".to_string(),
            passed: passes,
            value: result.outcome.clone(),
            expected: expected.clone(),
        });
    }

    if let Some(max_t) = asserts.max_ticks_to_win {
        let passes = result.outcome == "Victory" && result.tick <= max_t;
        out.push(AssertionResult {
            name: "max_ticks_to_win".to_string(),
            passed: passes,
            value: format!("{} (outcome={})", result.tick, result.outcome),
            expected: format!("<= {} ticks, outcome=Victory", max_t),
        });
    }

    if let Some(min_alive) = asserts.min_heroes_alive {
        let passes = result.final_hero_count >= min_alive;
        out.push(AssertionResult {
            name: "min_heroes_alive".to_string(),
            passed: passes,
            value: result.final_hero_count.to_string(),
            expected: format!(">= {}", min_alive),
        });
    }

    if let Some(max_dead) = asserts.max_heroes_dead {
        let passes = result.hero_deaths <= max_dead;
        out.push(AssertionResult {
            name: "max_heroes_dead".to_string(),
            passed: passes,
            value: result.hero_deaths.to_string(),
            expected: format!("<= {}", max_dead),
        });
    }

    out
}

// ---------------------------------------------------------------------------
// File loading
// ---------------------------------------------------------------------------

/// Read a `.toml` scenario file and return the parsed `ScenarioFile`.
pub fn load_scenario_file(path: &Path) -> Result<ScenarioFile, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    toml::from_str::<ScenarioFile>(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))
}
