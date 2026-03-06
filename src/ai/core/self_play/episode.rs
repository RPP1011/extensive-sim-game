//! Episode runners and I/O.

use std::io::Write;
use std::path::Path;

use super::features::{extract_features};
use super::actions::{action_mask, action_to_intent};
use super::policy::{PolicyWeights, Episode, Step};
use crate::ai::core::{is_alive, step, IntentAction, SimState, Team, UnitIntent, FIXED_TICK_MS};
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Self-play episode runner
// ---------------------------------------------------------------------------

/// Run one episode with the policy controlling hero units.
/// Enemy units use default AI.
/// `step_interval`: record steps every N ticks to reduce data size (1 = every tick).
pub fn run_episode(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    policy: &PolicyWeights,
    temperature: f32,
    rng_seed: u64,
    step_interval: u64,
) -> Episode {
    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut rng = rng_seed;
    let mut steps = Vec::new();

    let hero_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    // Track accumulated HP delta between recorded steps
    let mut interval_hero_hp: i32 = sim.units.iter().filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
    let mut interval_enemy_hp: i32 = sim.units.iter().filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
    let total_hp_start = (interval_hero_hp + interval_enemy_hp).max(1) as f32;

    for tick in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let record = tick % step_interval == 0;

        if record {
            // Compute accumulated reward since last record
            let cur_hero_hp: i32 = sim.units.iter().filter(|u| u.team == Team::Hero).map(|u| u.hp.max(0)).sum();
            let cur_enemy_hp: i32 = sim.units.iter().filter(|u| u.team == Team::Enemy).map(|u| u.hp.max(0)).sum();
            let enemy_dmg = (interval_enemy_hp - cur_enemy_hp).max(0) as f32;
            let hero_dmg = (interval_hero_hp - cur_hero_hp).max(0) as f32;
            let step_r = (enemy_dmg - hero_dmg) / total_hp_start;
            interval_hero_hp = cur_hero_hp;
            interval_enemy_hp = cur_enemy_hp;

            for &uid in &hero_ids {
                if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                    continue;
                }
                if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                    if u.casting.is_some() || u.control_remaining_ms > 0 {
                        continue;
                    }
                }

                let features = extract_features(&sim, uid);
                let mask = action_mask(&sim, uid);

                let (action, prob) = policy.sample_action(&features, &mask, temperature, &mut rng);
                let intent_action = action_to_intent(action, uid, &sim);

                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent { unit_id: uid, action: intent_action });

                steps.push(Step {
                    unit_id: uid,
                    features: features.to_vec(),
                    action,
                    log_prob: prob.ln(),
                    mask: mask.to_vec(),
                    step_reward: step_r,
                });
            }
        } else {
            // Non-recorded tick: still apply policy actions
            for &uid in &hero_ids {
                if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                    continue;
                }
                if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                    if u.casting.is_some() || u.control_remaining_ms > 0 {
                        continue;
                    }
                }

                let features = extract_features(&sim, uid);
                let mask = action_mask(&sim, uid);
                let (action, _) = policy.sample_action(&features, &mask, temperature, &mut rng);
                let intent_action = action_to_intent(action, uid, &sim);
                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent { unit_id: uid, action: intent_action });
            }
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return Episode {
                scenario: scenario_name.to_string(),
                outcome: "Victory".to_string(),
                reward: 1.0,
                ticks: sim.tick,
                steps,
            };
        }
        if heroes_alive == 0 {
            return Episode {
                scenario: scenario_name.to_string(),
                outcome: "Defeat".to_string(),
                reward: -1.0,
                ticks: sim.tick,
                steps,
            };
        }
    }

    // Shaped reward on timeout: HP differential
    let hero_hp_frac = hp_fraction(&sim, Team::Hero);
    let enemy_hp_frac = hp_fraction(&sim, Team::Enemy);
    let shaped = (enemy_hp_frac - hero_hp_frac).clamp(-1.0, 1.0) * 0.5; // [-0.5, 0.5]

    Episode {
        scenario: scenario_name.to_string(),
        outcome: "Timeout".to_string(),
        reward: shaped,
        ticks: sim.tick,
        steps,
    }
}

/// Fraction of HP lost for a team (0.0 = full hp, 1.0 = all dead).
fn hp_fraction(sim: &SimState, team: Team) -> f32 {
    let mut lost = 0i32;
    let mut total = 0i32;
    for u in &sim.units {
        if u.team == team {
            total += u.max_hp;
            lost += u.max_hp - u.hp.max(0);
        }
    }
    if total == 0 { 0.0 } else { lost as f32 / total as f32 }
}

/// Run one episode with greedy policy (no exploration) for evaluation.
pub fn run_episode_greedy(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    policy: &PolicyWeights,
) -> Episode {
    run_episode_greedy_with_focus(initial_sim, initial_squad_ai, scenario_name, max_ticks, policy, 0)
}

/// Like `run_episode_greedy` but periodically runs joint focus-target search.
/// `focus_interval`: how often (in ticks) to re-evaluate focus target. 0 = disabled.
pub fn run_episode_greedy_with_focus(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    policy: &PolicyWeights,
    focus_interval: u64,
) -> Episode {
    use crate::ai::core::oracle::squad_oracle;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;

    let hero_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    // Cache squad plan: recompute every focus_interval ticks
    let mut squad_plan: Option<Vec<(u32, IntentAction)>> = None;
    let mut plan_tick: u64 = u64::MAX;

    for tick in 0..max_ticks {
        // Run squad oracle at intervals (or use policy if disabled)
        let use_squad = focus_interval > 0 && tick >= plan_tick + focus_interval ||
                        focus_interval > 0 && tick == 0;

        if use_squad {
            let plan = squad_oracle(&sim, &squad_ai, Team::Hero, 10);
            squad_plan = Some(plan.actions);
            plan_tick = tick;
        }

        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        // Apply squad plan or policy for each hero
        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let intent_action = if let Some(ref plan) = squad_plan {
                // Use squad oracle's committed action if available
                plan.iter()
                    .find(|(id, _)| *id == uid)
                    .map(|(_, a)| *a)
                    .unwrap_or_else(|| {
                        let features = extract_features(&sim, uid);
                        let mask = action_mask(&sim, uid);
                        let action = policy.greedy_action(&features, &mask);
                        action_to_intent(action, uid, &sim)
                    })
            } else {
                let features = extract_features(&sim, uid);
                let mask = action_mask(&sim, uid);
                let action = policy.greedy_action(&features, &mask);
                action_to_intent(action, uid, &sim)
            };

            intents.retain(|i| i.unit_id != uid);
            intents.push(UnitIntent { unit_id: uid, action: intent_action });
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 {
            return Episode {
                scenario: scenario_name.to_string(),
                outcome: "Victory".to_string(),
                reward: 1.0,
                ticks: sim.tick,
                steps: Vec::new(),
            };
        }
        if heroes_alive == 0 {
            return Episode {
                scenario: scenario_name.to_string(),
                outcome: "Defeat".to_string(),
                reward: -1.0,
                ticks: sim.tick,
                steps: Vec::new(),
            };
        }
    }

    let hero_hp_frac = hp_fraction(&sim, Team::Hero);
    let enemy_hp_frac = hp_fraction(&sim, Team::Enemy);
    let shaped = (enemy_hp_frac - hero_hp_frac).clamp(-1.0, 1.0) * 0.5;

    Episode {
        scenario: scenario_name.to_string(),
        outcome: "Timeout".to_string(),
        reward: shaped,
        ticks: sim.tick,
        steps: Vec::new(),
    }
}

/// Write episodes as JSONL.
pub fn write_episodes(episodes: &[Episode], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for ep in episodes {
        let line = serde_json::to_string(ep).unwrap();
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}

/// Load policy weights from JSON.
pub fn load_policy(path: &Path) -> Result<PolicyWeights, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse policy: {e}"))
}
