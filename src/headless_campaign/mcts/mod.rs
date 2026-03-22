//! MCTS (Monte Carlo Tree Search) for campaign decision bootstrap.
//!
//! Runs UCT tree search over `CampaignState` to discover good guild
//! management policies without any learned model. The search uses the
//! combat oracle for fast battle resolution during rollouts.
//!
//! Output: visit-count distributions over actions at each decision point,
//! used as behavioral cloning targets for the playtester model.

mod rollout;

use serde::{Deserialize, Serialize};
use std::f64;

use super::actions::CampaignAction;
use super::state::{CampaignOutcome, CampaignState, CAMPAIGN_TICK_MS};
use super::step::step_campaign;

pub use rollout::heuristic_rollout_policy;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Number of simulations (tree walks) per decision.
    pub simulations_per_move: u32,
    /// UCT exploration constant (C in UCB1). Higher = more exploration.
    pub exploration_constant: f64,
    /// How many ticks to simulate during each rollout.
    pub rollout_horizon_ticks: u64,
    /// Discount factor for future rewards.
    pub discount_factor: f64,
    /// How many ticks to advance between agent decisions.
    pub decision_interval_ticks: u64,
    /// Maximum campaign ticks before stopping MCTS.
    pub max_campaign_ticks: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations_per_move: 200,
            exploration_constant: 1.414, // sqrt(2)
            rollout_horizon_ticks: 5000, // ~8 minutes game time
            discount_factor: 0.99,
            decision_interval_ticks: 50, // decide every 5s game time
            max_campaign_ticks: 50_000,
        }
    }
}

// ---------------------------------------------------------------------------
// MCTS Node
// ---------------------------------------------------------------------------

struct MctsNode {
    /// Action that led to this node (None for root).
    action: Option<CampaignAction>,
    /// Visit count.
    visits: u32,
    /// Total accumulated value.
    total_value: f64,
    /// Children (expanded actions).
    children: Vec<MctsNode>,
    /// Actions not yet expanded.
    untried_actions: Vec<CampaignAction>,
}

impl MctsNode {
    fn new(action: Option<CampaignAction>, valid_actions: Vec<CampaignAction>) -> Self {
        Self {
            action,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            untried_actions: valid_actions,
        }
    }

    /// UCB1 score for child selection.
    fn ucb1(&self, parent_visits: u32, c: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_value / self.visits as f64;
        let exploration = c * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }

    /// Select the child with highest UCB1 score.
    fn select_child(&self, c: f64) -> usize {
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ua = a.ucb1(self.visits, c);
                let ub = b.ucb1(self.visits, c);
                ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Is this node fully expanded (no untried actions)?
    fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    /// Is this a leaf node?
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MCTS Search
// ---------------------------------------------------------------------------

/// Result of MCTS search at a single decision point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MctsDecision {
    /// Visit-count distribution over valid actions.
    pub action_visits: Vec<(String, u32)>,
    /// Best action (most visited).
    pub best_action: String,
    /// Value estimate at this state.
    pub value_estimate: f64,
    /// Total simulations run.
    pub total_simulations: u32,
    /// Campaign tick when this decision was made.
    pub tick: u64,
}

/// Run MCTS search at the current state and return the best action + statistics.
pub fn mcts_search(
    state: &CampaignState,
    config: &MctsConfig,
) -> (CampaignAction, MctsDecision) {
    let valid_actions = state.valid_actions();
    if valid_actions.len() <= 1 {
        // Only Wait is valid — no search needed
        let action = valid_actions.into_iter().next().unwrap_or(CampaignAction::Wait);
        let decision = MctsDecision {
            action_visits: vec![(format!("{:?}", action), 1)],
            best_action: format!("{:?}", action),
            value_estimate: 0.0,
            total_simulations: 0,
            tick: state.tick,
        };
        return (action, decision);
    }

    let mut root = MctsNode::new(None, valid_actions);

    for _ in 0..config.simulations_per_move {
        let mut sim_state = state.clone();

        // Selection: walk down the tree using UCB1
        let mut path = vec![0usize]; // dummy root index
        let mut current = &mut root;

        while !current.is_leaf() && current.is_fully_expanded() {
            let child_idx = current.select_child(config.exploration_constant);
            // Apply the child's action and advance
            let action = current.children[child_idx].action.clone();
            if let Some(a) = action {
                step_campaign(&mut sim_state, Some(a));
            }
            // Advance decision_interval ticks
            for _ in 0..config.decision_interval_ticks.saturating_sub(1) {
                step_campaign(&mut sim_state, None);
            }
            path.push(child_idx);
            current = &mut current.children[child_idx];
        }

        // Expansion: if not fully expanded, expand one untried action
        if !current.untried_actions.is_empty() {
            let action_idx = (sim_state.rng as usize) % current.untried_actions.len();
            let action = current.untried_actions.remove(action_idx);

            // Apply action
            step_campaign(&mut sim_state, Some(action.clone()));
            for _ in 0..config.decision_interval_ticks.saturating_sub(1) {
                step_campaign(&mut sim_state, None);
            }

            let child_valid = sim_state.valid_actions();
            let child = MctsNode::new(Some(action), child_valid);
            current.children.push(child);
            let new_idx = current.children.len() - 1;
            path.push(new_idx);
            current = &mut current.children[new_idx];
        }

        // Rollout: simulate from current state using heuristic policy
        let value = rollout::heuristic_rollout(
            &mut sim_state,
            config.rollout_horizon_ticks,
            config.discount_factor,
        );

        // Backpropagation: update all nodes on the path
        // We need to walk the tree again with the path indices
        root.visits += 1;
        root.total_value += value;
        let mut node = &mut root;
        for &idx in path.iter().skip(1) {
            node = &mut node.children[idx];
            node.visits += 1;
            node.total_value += value;
        }
    }

    // Select most-visited child
    let best_idx = root
        .children
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| c.visits)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let action_visits: Vec<(String, u32)> = root
        .children
        .iter()
        .map(|c| (format!("{:?}", c.action), c.visits))
        .collect();

    let best_action = root.children[best_idx]
        .action
        .clone()
        .unwrap_or(CampaignAction::Wait);

    let value_estimate = if root.visits > 0 {
        root.total_value / root.visits as f64
    } else {
        0.0
    };

    let decision = MctsDecision {
        action_visits,
        best_action: format!("{:?}", best_action),
        value_estimate,
        total_simulations: config.simulations_per_move,
        tick: state.tick,
    };

    (best_action, decision)
}

// ---------------------------------------------------------------------------
// MCTS Campaign Runner
// ---------------------------------------------------------------------------

/// Training sample from MCTS: state observation + visit-count policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MctsTrainingSample {
    pub tick: u64,
    pub decision: MctsDecision,
    /// Campaign seed for reproducibility.
    pub seed: u64,
    /// Eventual campaign outcome (filled after campaign ends).
    pub campaign_outcome: Option<CampaignOutcome>,
}

/// Run a full campaign with MCTS making decisions at each decision point.
/// Returns the final outcome and all training samples.
pub fn run_mcts_campaign(
    seed: u64,
    config: &MctsConfig,
) -> (CampaignOutcome, Vec<MctsTrainingSample>) {
    let mut state = CampaignState::default_test_campaign(seed);
    let mut samples: Vec<MctsTrainingSample> = Vec::new();
    let mut next_decision_tick = config.decision_interval_ticks;

    loop {
        if state.tick >= config.max_campaign_ticks {
            // Fill in outcome for all samples
            for s in &mut samples {
                s.campaign_outcome = Some(CampaignOutcome::Timeout);
            }
            return (CampaignOutcome::Timeout, samples);
        }

        // Decision point
        if state.tick >= next_decision_tick {
            let (action, decision) = mcts_search(&state, config);

            samples.push(MctsTrainingSample {
                tick: state.tick,
                decision,
                seed,
                campaign_outcome: None,
            });

            let result = step_campaign(&mut state, Some(action));
            if let Some(outcome) = result.outcome {
                for s in &mut samples {
                    s.campaign_outcome = Some(outcome);
                }
                return (outcome, samples);
            }

            next_decision_tick = state.tick + config.decision_interval_ticks;
        } else {
            // Tick forward without action
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_search_produces_decision() {
        let state = CampaignState::default_test_campaign(42);
        let config = MctsConfig {
            simulations_per_move: 50,
            rollout_horizon_ticks: 500,
            decision_interval_ticks: 20,
            max_campaign_ticks: 5000,
            ..Default::default()
        };

        // Advance state until quests appear
        let mut s = state;
        for _ in 0..2000 {
            step_campaign(&mut s, None);
            if !s.request_board.is_empty() {
                break;
            }
        }

        let (_action, decision) = mcts_search(&s, &config);
        assert!(decision.total_simulations > 0);
        assert!(!decision.action_visits.is_empty());
    }

    #[test]
    fn test_mcts_campaign_completes() {
        let config = MctsConfig {
            simulations_per_move: 20,
            rollout_horizon_ticks: 200,
            decision_interval_ticks: 100,
            max_campaign_ticks: 5000,
            ..Default::default()
        };

        let (outcome, samples) = run_mcts_campaign(42, &config);

        // Should produce some samples
        assert!(
            !samples.is_empty() || outcome == CampaignOutcome::Timeout,
            "Expected samples or timeout"
        );
        // All samples should have outcome filled
        for s in &samples {
            assert!(s.campaign_outcome.is_some());
        }
    }
}
