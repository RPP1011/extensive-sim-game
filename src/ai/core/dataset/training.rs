//! Training sample types, dataset generation, and I/O.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::super::oracle::score_actions_with_depth;
use super::super::self_play::{extract_features, action_mask};
use super::super::{is_alive, step, SimState, Team, FIXED_TICK_MS};
use super::actions::{classify_action, classify_action_raw, classify_combat_action};
use super::features::extract_unit_features;
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Training sample
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    pub label: u8,
    pub score: f64,
    pub tick: u64,
    pub unit_id: u32,
    pub scenario: String,
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

/// Run oracle-played scenario and emit training samples.
pub fn generate_dataset(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    rollout_ticks: u64,
) -> Vec<TrainingSample> {
    use super::super::UnitIntent;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut samples = Vec::new();

    let hero_ids: Vec<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    for _ in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        let focus_target = squad_ai
            .blackboard_for_team(Team::Hero)
            .and_then(|b| b.focus_target);

        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let oracle = score_actions_with_depth(&sim, &squad_ai, uid, focus_target, rollout_ticks);
            if oracle.scored_actions.is_empty() {
                continue;
            }

            let best = &oracle.scored_actions[0];
            let label = classify_action(&best.action, uid, &sim);
            let features = extract_unit_features(&sim, &squad_ai, uid);

            samples.push(TrainingSample {
                features: features.to_vec(),
                label: label as u8,
                score: best.score,
                tick: sim.tick,
                unit_id: uid,
                scenario: scenario_name.to_string(),
            });

            // Override hero intent with oracle pick
            intents.retain(|i| i.unit_id != uid);
            intents.push(UnitIntent {
                unit_id: uid,
                action: best.action,
            });
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    samples
}

/// Training sample for the 5-class combat-only student model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatTrainingSample {
    pub features: Vec<f32>,
    pub label: u8,
    pub score: f64,
    pub tick: u64,
    pub unit_id: u32,
    pub scenario: String,
}

/// Run oracle-played scenario and emit 5-class combat-only training samples.
/// Ability actions from the oracle are skipped (they'll be handled by frozen evaluators).
pub fn generate_combat_dataset(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    rollout_ticks: u64,
) -> Vec<CombatTrainingSample> {
    use super::super::UnitIntent;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut samples = Vec::new();

    let hero_ids: Vec<u32> = sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    for _ in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        let focus_target = squad_ai
            .blackboard_for_team(Team::Hero)
            .and_then(|b| b.focus_target);

        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let oracle = score_actions_with_depth(&sim, &squad_ai, uid, focus_target, rollout_ticks);
            if oracle.scored_actions.is_empty() {
                continue;
            }

            let best = &oracle.scored_actions[0];
            // Only emit samples for non-ability actions
            if let Some(label) = classify_combat_action(&best.action, uid, &sim) {
                let features = extract_unit_features(&sim, &squad_ai, uid);
                samples.push(CombatTrainingSample {
                    features: features.to_vec(),
                    label: label as u8,
                    score: best.score,
                    tick: sim.tick,
                    unit_id: uid,
                    scenario: scenario_name.to_string(),
                });
            }

            // Override hero intent with oracle pick
            intents.retain(|i| i.unit_id != uid);
            intents.push(UnitIntent {
                unit_id: uid,
                action: best.action,
            });
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    samples
}

/// Write combat samples as JSONL to a file.
pub fn write_combat_dataset(samples: &[CombatTrainingSample], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for s in samples {
        let line = serde_json::to_string(s).unwrap();
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}

/// Write samples as JSONL to a file.
pub fn write_dataset(samples: &[TrainingSample], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for s in samples {
        let line = serde_json::to_string(s).unwrap();
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw-feature dataset (311 features, 14 action classes)
// ---------------------------------------------------------------------------

/// Training sample using raw 311 features and 14-class action labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTrainingSample {
    pub features: Vec<f32>,
    pub label: u8,
    pub mask: Vec<bool>,
    pub score: f64,
    pub tick: u64,
    pub unit_id: u32,
    pub scenario: String,
}

/// Generate oracle-labeled dataset with raw 311 features and 14-class actions.
pub fn generate_raw_dataset(
    initial_sim: SimState,
    initial_squad_ai: SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    rollout_ticks: u64,
) -> Vec<RawTrainingSample> {
    use super::super::UnitIntent;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut samples = Vec::new();

    let hero_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    for _ in 0..max_ticks {
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        let focus_target = squad_ai
            .blackboard_for_team(Team::Hero)
            .and_then(|b| b.focus_target);

        for &uid in &hero_ids {
            if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) {
                continue;
            }
            if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                if u.casting.is_some() || u.control_remaining_ms > 0 {
                    continue;
                }
            }

            let oracle = score_actions_with_depth(&sim, &squad_ai, uid, focus_target, rollout_ticks);
            if oracle.scored_actions.is_empty() {
                continue;
            }

            let best = &oracle.scored_actions[0];
            let label = match classify_action_raw(&best.action, uid, &sim) {
                Some(l) => l,
                None => continue,
            };

            let features = extract_features(&sim, uid);
            let mask = action_mask(&sim, uid);

            samples.push(RawTrainingSample {
                features: features.to_vec(),
                label: label as u8,
                mask: mask.to_vec(),
                score: best.score,
                tick: sim.tick,
                unit_id: uid,
                scenario: scenario_name.to_string(),
            });

            // Override hero intent with oracle pick
            intents.retain(|i| i.unit_id != uid);
            intents.push(UnitIntent { unit_id: uid, action: best.action });
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    samples
}

/// Write raw training samples as JSONL.
pub fn write_raw_dataset(samples: &[RawTrainingSample], path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for s in samples {
        let line = serde_json::to_string(s).unwrap();
        writeln!(writer, "{}", line)?;
    }
    writer.flush()?;
    Ok(())
}
