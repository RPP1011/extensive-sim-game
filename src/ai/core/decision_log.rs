use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::oracle::{score_actions_with_depth, ScoredAction, DEFAULT_ROLLOUT_TICKS};
use super::{is_alive, IntentAction, SimState, Team, FIXED_TICK_MS};
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// One decision record: what the AI chose vs what the oracle recommends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub tick: u64,
    pub unit_id: u32,
    pub ai_action: IntentAction,
    pub oracle_top3: Vec<ScoredAction>,
    pub ai_action_score: f64,
    pub oracle_best_score: f64,
    pub score_delta: f64,
    pub matched_top1: bool,
}

/// Summary statistics across all decisions in a scenario run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSummary {
    pub scenario_name: String,
    pub outcome: String,
    pub total_ticks: u64,
    pub total_decisions: u32,
    pub matched_top1_count: u32,
    pub match_rate: f64,
    pub avg_score_delta: f64,
    pub max_score_delta: f64,
    pub wrong_target_count: u32,
    pub wrong_ability_count: u32,
    pub wrong_position_count: u32,
}

// ---------------------------------------------------------------------------
// Decision logging
// ---------------------------------------------------------------------------

/// Classify why the AI diverged from oracle recommendation.
fn classify_divergence(ai: &IntentAction, oracle_best: &IntentAction) -> &'static str {
    match (ai, oracle_best) {
        // Same action type but different target
        (IntentAction::Attack { target_id: a }, IntentAction::Attack { target_id: b }) if a != b => {
            "wrong_target"
        }
        (
            IntentAction::CastAbility { target_id: a },
            IntentAction::CastAbility { target_id: b },
        ) if a != b => "wrong_target",
        (
            IntentAction::CastControl { target_id: a },
            IntentAction::CastControl { target_id: b },
        ) if a != b => "wrong_target",
        (IntentAction::CastHeal { target_id: a }, IntentAction::CastHeal { target_id: b })
            if a != b =>
        {
            "wrong_target"
        }
        // Different action type entirely
        (IntentAction::Attack { .. }, _) | (_, IntentAction::Attack { .. }) => "wrong_ability",
        (IntentAction::MoveTo { .. }, _) | (_, IntentAction::MoveTo { .. }) => "wrong_position",
        _ => "wrong_ability",
    }
}

/// Run a scenario with oracle evaluation at every tick for hero units.
/// Writes JSONL decision records to `output_path` and returns a summary.
pub fn run_with_oracle(
    sim: SimState,
    squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    output_path: Option<&Path>,
) -> OracleSummary {
    run_with_oracle_depth(sim, squad_ai, scenario_name, max_ticks, output_path, DEFAULT_ROLLOUT_TICKS)
}

/// Run a scenario with oracle evaluation using a configurable rollout depth.
pub fn run_with_oracle_depth(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    output_path: Option<&Path>,
    rollout_ticks: u64,
) -> OracleSummary {
    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;

    let hero_ids: Vec<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    let mut writer: Option<std::io::BufWriter<std::fs::File>> = output_path.map(|p| {
        let file = std::fs::File::create(p).expect("Failed to create decision log file");
        std::io::BufWriter::new(file)
    });

    let mut total_decisions = 0u32;
    let mut matched_top1 = 0u32;
    let mut total_delta = 0.0f64;
    let mut max_delta = 0.0f64;
    let mut wrong_target = 0u32;
    let mut wrong_ability = 0u32;
    let mut wrong_position = 0u32;
    let mut outcome = "Timeout".to_string();

    for _ in 0..max_ticks {
        // Generate AI intents
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        // Build a lookup of what the AI chose per unit
        let ai_choices: std::collections::HashMap<u32, IntentAction> = intents
            .iter()
            .map(|i| (i.unit_id, i.action))
            .collect();

        // Get focus target for hero team
        let focus_target = squad_ai.blackboard_for_team(Team::Hero).map(|b| b.focus_target).flatten();

        // Oracle evaluation for each alive hero unit
        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            // Skip units that are casting or controlled
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let oracle = score_actions_with_depth(&sim, &squad_ai, uid, focus_target, rollout_ticks);

            if oracle.scored_actions.is_empty() {
                continue;
            }

            let ai_action = ai_choices.get(&uid).copied().unwrap_or(IntentAction::Hold);

            // Find AI's action score in oracle results
            let ai_score = oracle
                .scored_actions
                .iter()
                .find(|s| actions_equivalent(&s.action, &ai_action))
                .map(|s| s.score)
                .unwrap_or(oracle.scored_actions.last().map(|s| s.score).unwrap_or(0.0));

            let best_score = oracle.scored_actions[0].score;
            let delta = best_score - ai_score;
            let is_match = actions_equivalent(&oracle.scored_actions[0].action, &ai_action);

            let record = DecisionRecord {
                tick: sim.tick,
                unit_id: uid,
                ai_action,
                oracle_top3: oracle.scored_actions.into_iter().take(3).collect(),
                ai_action_score: ai_score,
                oracle_best_score: best_score,
                score_delta: delta,
                matched_top1: is_match,
            };

            // Write JSONL
            if let Some(ref mut w) = writer {
                let line = serde_json::to_string(&record).unwrap();
                writeln!(w, "{}", line).ok();
            }

            total_decisions += 1;
            if is_match {
                matched_top1 += 1;
            } else {
                let divergence = classify_divergence(&ai_action, &record.oracle_top3[0].action);
                match divergence {
                    "wrong_target" => wrong_target += 1,
                    "wrong_ability" => wrong_ability += 1,
                    "wrong_position" => wrong_position += 1,
                    _ => {}
                }
            }
            total_delta += delta;
            if delta > max_delta {
                max_delta = delta;
            }
        }

        // Step simulation forward
        let (new_sim, _events) = super::step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();

        if enemies_alive == 0 {
            outcome = "Victory".to_string();
            break;
        }
        if heroes_alive == 0 {
            outcome = "Defeat".to_string();
            break;
        }
    }

    if let Some(ref mut w) = writer {
        w.flush().ok();
    }

    let match_rate = if total_decisions > 0 {
        matched_top1 as f64 / total_decisions as f64
    } else {
        0.0
    };
    let avg_delta = if total_decisions > 0 {
        total_delta / total_decisions as f64
    } else {
        0.0
    };

    OracleSummary {
        scenario_name: scenario_name.to_string(),
        outcome,
        total_ticks: sim.tick,
        total_decisions,
        matched_top1_count: matched_top1,
        match_rate,
        avg_score_delta: avg_delta,
        max_score_delta: max_delta,
        wrong_target_count: wrong_target,
        wrong_ability_count: wrong_ability,
        wrong_position_count: wrong_position,
    }
}

// ---------------------------------------------------------------------------
// Oracle-played mode — heroes use oracle top pick every tick
// ---------------------------------------------------------------------------

/// Result of an oracle-played scenario run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePlayedResult {
    pub scenario_name: String,
    pub outcome: String,
    pub total_ticks: u64,
    pub heroes_alive: usize,
    pub enemies_alive: usize,
}

/// Run a scenario where hero units use the oracle's top-scoring action each tick
/// instead of the default AI. Enemy units still use the default AI.
pub fn run_oracle_played(
    sim: SimState,
    squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
) -> OraclePlayedResult {
    run_oracle_played_depth(sim, squad_ai, scenario_name, max_ticks, DEFAULT_ROLLOUT_TICKS)
}

/// Oracle-played mode with configurable rollout depth.
pub fn run_oracle_played_depth(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    rollout_ticks: u64,
) -> OraclePlayedResult {
    use super::UnitIntent;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;

    let hero_ids: Vec<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    let mut outcome = "Timeout".to_string();

    for _ in 0..max_ticks {
        // Generate default AI intents (used for enemies, fallback for heroes)
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        // Get focus target for hero team
        let focus_target = squad_ai
            .blackboard_for_team(Team::Hero)
            .and_then(|b| b.focus_target);

        // Replace hero intents with oracle top picks
        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            // Skip units that are casting or controlled
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let oracle = score_actions_with_depth(&sim, &squad_ai, uid, focus_target, rollout_ticks);
            if let Some(best) = oracle.scored_actions.first() {
                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent {
                    unit_id: uid,
                    action: best.action,
                });
            }
        }

        let (new_sim, _events) = super::step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();

        if enemies_alive == 0 {
            outcome = "Victory".to_string();
            break;
        }
        if heroes_alive == 0 {
            outcome = "Defeat".to_string();
            break;
        }
    }

    let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
    let enemies_alive = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .count();

    OraclePlayedResult {
        scenario_name: scenario_name.to_string(),
        outcome,
        total_ticks: sim.tick,
        heroes_alive,
        enemies_alive,
    }
}

/// Check if two actions are "equivalent" (same type + same target).
fn actions_equivalent(a: &IntentAction, b: &IntentAction) -> bool {
    use crate::ai::effects::AbilityTarget;

    match (a, b) {
        (IntentAction::Attack { target_id: a }, IntentAction::Attack { target_id: b }) => a == b,
        (
            IntentAction::CastAbility { target_id: a },
            IntentAction::CastAbility { target_id: b },
        ) => a == b,
        (IntentAction::CastHeal { target_id: a }, IntentAction::CastHeal { target_id: b }) => {
            a == b
        }
        (
            IntentAction::CastControl { target_id: a },
            IntentAction::CastControl { target_id: b },
        ) => a == b,
        (
            IntentAction::UseAbility {
                ability_index: ai,
                target: at,
            },
            IntentAction::UseAbility {
                ability_index: bi,
                target: bt,
            },
        ) => {
            ai == bi
                && match (at, bt) {
                    (AbilityTarget::Unit(a), AbilityTarget::Unit(b)) => a == b,
                    (AbilityTarget::Position(a), AbilityTarget::Position(b)) => {
                        a.x == b.x && a.y == b.y
                    }
                    (AbilityTarget::None, AbilityTarget::None) => true,
                    _ => false,
                }
        }
        (IntentAction::MoveTo { position: a }, IntentAction::MoveTo { position: b }) => {
            a.x == b.x && a.y == b.y
        }
        (IntentAction::Hold, IntentAction::Hold) => true,
        _ => false,
    }
}
