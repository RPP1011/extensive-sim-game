//! MCTS bootstrap data export for behavioral cloning.
//!
//! Runs MCTS campaigns and captures entity tokens at each decision point,
//! producing JSONL training data with visit-count distributions.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

use super::{mcts_search, MctsConfig, MctsDecision};
use crate::headless_campaign::config::CampaignConfig;
use crate::headless_campaign::state::{CampaignOutcome, CampaignPhase, CampaignState};
use crate::headless_campaign::step::step_campaign;
use crate::headless_campaign::tokens::EntityToken;
use crate::headless_campaign::world_templates::WorldTemplate;

// ---------------------------------------------------------------------------
// Export sample
// ---------------------------------------------------------------------------

/// A single training sample from an MCTS-guided campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MctsExportSample {
    /// Campaign seed for reproducibility.
    pub seed: u64,
    /// Name of the starting choice selected (or empty if tick > 0).
    pub starting_choice_name: String,
    /// Name of the world template used for this campaign.
    pub world_template_name: String,
    /// Campaign tick when this decision was made.
    pub tick: u64,
    /// Entity tokens representing the full campaign state at decision time.
    pub tokens: Vec<EntityToken>,
    /// MCTS visit-count distribution over valid actions.
    pub action_distribution: Vec<(String, u32)>,
    /// Best action (most visited).
    pub best_action: String,
    /// Value estimate at this state.
    pub value_estimate: f64,
    /// Campaign outcome (filled after campaign completes).
    pub campaign_outcome: Option<CampaignOutcome>,
}

// ---------------------------------------------------------------------------
// World template name derivation
// ---------------------------------------------------------------------------

/// Derive the world template name for a given seed, mirroring
/// `CampaignState::load_world_template` logic.
fn world_template_name_for_seed(seed: u64) -> String {
    let dir = std::path::Path::new("dataset/campaign/world_templates");
    if dir.exists() {
        if let Ok(templates) = WorldTemplate::load_from_dir(dir) {
            if !templates.is_empty() {
                let idx = (seed as usize) % templates.len();
                return templates[idx].name.clone();
            }
        }
    }
    "Frontier".to_string()
}

// ---------------------------------------------------------------------------
// Export campaign runner
// ---------------------------------------------------------------------------

/// Run a full campaign with MCTS, capturing entity tokens at each decision point.
///
/// Returns the campaign outcome and all export samples with tokens.
pub fn export_mcts_campaign(
    seed: u64,
    campaign_config: &CampaignConfig,
    mcts_config: &MctsConfig,
) -> (CampaignOutcome, Vec<MctsExportSample>) {
    let mut state = CampaignState::with_config(seed, campaign_config.clone());
    let mut samples: Vec<MctsExportSample> = Vec::new();
    let mut next_decision_tick = mcts_config.decision_interval_ticks;

    let world_template_name = world_template_name_for_seed(seed);

    // Handle starting choice via MCTS before the main loop.
    // The starting choice is part of the MCTS tree (not pre-selected).
    if state.phase != CampaignPhase::Playing {
        let tokens = state.to_tokens();
        let (action, decision) = mcts_search(&state, mcts_config);

        // Extract the starting choice name from the chosen action.
        let starting_choice_name = format_starting_choice_name(&decision);

        samples.push(MctsExportSample {
            seed,
            starting_choice_name,
            world_template_name: world_template_name.clone(),
            tick: 0,
            tokens,
            action_distribution: decision.action_visits.clone(),
            best_action: decision.best_action.clone(),
            value_estimate: decision.value_estimate,
            campaign_outcome: None,
        });
        step_campaign(&mut state, Some(action));
    }

    loop {
        if state.tick >= mcts_config.max_campaign_ticks {
            for s in &mut samples {
                s.campaign_outcome = Some(CampaignOutcome::Timeout);
            }
            return (CampaignOutcome::Timeout, samples);
        }

        // Decision point
        if state.tick >= next_decision_tick {
            let tokens = state.to_tokens();
            let (action, decision) = mcts_search(&state, mcts_config);

            samples.push(MctsExportSample {
                seed,
                starting_choice_name: String::new(),
                world_template_name: world_template_name.clone(),
                tick: state.tick,
                tokens,
                action_distribution: decision.action_visits.clone(),
                best_action: decision.best_action.clone(),
                value_estimate: decision.value_estimate,
                campaign_outcome: None,
            });

            let result = step_campaign(&mut state, Some(action));
            if let Some(outcome) = result.outcome {
                for s in &mut samples {
                    s.campaign_outcome = Some(outcome);
                }
                return (outcome, samples);
            }

            next_decision_tick = state.tick + mcts_config.decision_interval_ticks;
        } else {
            let result = step_campaign(&mut state, None);
            if let Some(outcome) = result.outcome {
                for s in &mut samples {
                    s.campaign_outcome = Some(outcome);
                }
                return (outcome, samples);
            }
        }
    }
}

/// Extract the starting choice name from an MCTS decision's best_action string.
fn format_starting_choice_name(decision: &MctsDecision) -> String {
    // The best_action for starting choices is Debug-formatted as:
    // Some(ChooseStartingPackage { choice: StartingChoice { name: "...", ... } })
    // We extract the name from the best_action string.
    let s = &decision.best_action;
    if let Some(start) = s.find("name: \"") {
        let rest = &s[start + 7..];
        if let Some(end) = rest.find('"') {
            return rest[..end].to_string();
        }
    }
    // Fallback: return the full debug string
    s.clone()
}

// ---------------------------------------------------------------------------
// JSONL writer
// ---------------------------------------------------------------------------

/// Append export samples to a JSONL file. Creates the file if it doesn't exist.
pub fn write_samples_jsonl(
    samples: &[MctsExportSample],
    path: &Path,
) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut writer = std::io::BufWriter::new(file);

    for sample in samples {
        serde_json::to_writer(&mut writer, sample)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
