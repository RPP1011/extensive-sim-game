//! BFS state-space exploration with cluster-and-prune.
//!
//! 1. Start from a set of root states
//! 2. At each root, try every ACTION TYPE (grouped — not every target variant)
//! 3. Step forward N ticks per branch to let the action play out
//! 4. Record (root_tokens, action_type, leaf_tokens, leaf_value)
//! 5. Cluster the leaf states by feature similarity, preserving extremes
//! 6. Pick diverse representatives per cluster as next-wave roots
//! 7. Repeat for K waves
//!
//! Search improvements:
//! - Strategic rollout modes (aggressive/economic/diplomatic/consistent/mixed)
//! - Action encoding for RL, temporal discounting, advantage estimation
//! - UCB-style action selection for balanced exploration
//! - Adaptive branch count based on action space size and state tension
//! - Progressive widening for large action spaces
//! - Counterfactual "do nothing" baselines per root
//! - State novelty bonus to encourage diverse state exploration
//! - Importance sampling weights for RL training bias correction
//! - Sample validation, deduplication, balanced sampling
//! - Compact serialization and reproducibility metadata
//!
//! This gives full action coverage without exponential blowup.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::action_meta::{
    action_meta_features, action_context, predict_outcome, action_synergy,
};
use super::actions::*;
use super::batch::heuristic_policy;
use super::config::{CampaignConfig, Difficulty};
use super::heuristic_bc::action_type_name;
use super::state::*;
use super::step::step_campaign;
use super::tokens::EntityToken;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single BFS exploration sample: "from this state, taking this action type
/// leads to this outcome after N ticks."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BfsSample {
    /// Tokens at the root state (before action).
    pub root_tokens: Vec<EntityToken>,
    /// Action type taken.
    pub action_type: String,
    /// Specific action (for reconstruction).
    pub action_detail: String,
    /// Fixed-size numeric encoding of the action for RL consumption. (pass 4)
    #[serde(default)]
    pub action_encoding: Vec<f32>,
    /// Tokens at the leaf state (after action + N ticks).
    pub leaf_tokens: Vec<EntityToken>,
    /// Heuristic value estimate at the leaf.
    pub leaf_value: f32,
    /// Temporally discounted leaf value (gamma^ticks_elapsed * leaf_value). (pass 4)
    #[serde(default)]
    pub discounted_value: f32,
    /// Campaign outcome if terminal, else None.
    pub leaf_outcome: Option<String>,
    /// Root tick.
    pub root_tick: u64,
    /// Leaf tick.
    pub leaf_tick: u64,
    /// Seed of the originating campaign.
    pub seed: u64,
    /// Wave number in the BFS.
    pub wave: u32,
    /// Which cluster this leaf was assigned to.
    pub cluster_id: u32,
    /// Difficulty level of the originating campaign.
    pub difficulty: Difficulty,
    /// Heuristic value at the root (for computing spread).
    pub root_value: f32,
    /// Game phase tag (early/mid/late).
    pub phase_tag: String,
    /// All valid action types at the root state (so RL agent knows alternatives). (pass 2)
    #[serde(default)]
    pub valid_action_types: Vec<String>,
    /// State delta: what changed between root and leaf. (pass 2)
    #[serde(default)]
    pub state_delta: Option<StateDelta>,
    /// Summary of all actions taken during the branch rollout. (pass 2)
    #[serde(default)]
    pub action_sequence: Vec<String>,
    /// Normalized value: how this state compares to average at this game phase. (pass 2)
    #[serde(default)]
    pub relative_value: f32,
    /// Strategic value score of the initial action (how interesting it is). (pass 2)
    #[serde(default)]
    pub action_strategic_value: f32,
    /// Rollout mode used for this branch. (pass 2)
    #[serde(default)]
    pub rollout_mode: String,
    /// Reward signal: leaf_value - root_value (for TD learning). (pass 4)
    #[serde(default)]
    pub value_delta: f32,
    /// Whether this was the best action from this root (for advantage estimation). (pass 4)
    #[serde(default)]
    pub is_best_action: bool,
    /// Advantage: branch_value - mean_value across all branches from same root. (pass 4)
    #[serde(default)]
    pub advantage: f32,
    /// Priority score for prioritized experience replay. (pass 4)
    #[serde(default)]
    pub replay_priority: f32,
    /// Intermediate value estimates at 25%, 50%, 75% of branch length. (pass 4)
    #[serde(default)]
    pub intermediate_values: Vec<f32>,
    /// Value of the counterfactual "do nothing" baseline branch. (pass 5)
    #[serde(default)]
    pub baseline_value: f32,
    /// Advantage of this action's leaf value vs the baseline. (pass 5)
    #[serde(default)]
    pub advantage_vs_baseline: f32,
    /// State novelty score (0-1): how rare this leaf state's feature bucket is. (pass 5)
    #[serde(default)]
    pub state_novelty: f32,
    /// Importance sampling weight = 1/P(action_selected). (pass 5)
    #[serde(default)]
    pub importance_weight: f32,
    /// Action metadata features (category, cost, risk, etc.) — 8 floats. (pass 8)
    #[serde(default)]
    pub action_meta: Vec<f32>,
    /// Action prerequisite context (why this action is valid) — 5 floats. (pass 8)
    #[serde(default)]
    pub action_prereqs: Vec<f32>,
    /// Predicted outcome features (gold change, reputation, risk, duration) — 4 floats. (pass 8)
    #[serde(default)]
    pub action_outcome: Vec<f32>,
    /// Synergy score with recent actions (0-1). (pass 8)
    #[serde(default)]
    pub action_synergy: f32,
    /// Curriculum difficulty tier (1=beginner, 2=intermediate, 3=advanced, 4=expert). (pass 9)
    #[serde(default)]
    pub curriculum_tier: u8,
    /// Skill tags this sample teaches (e.g. "economy", "diplomacy", "combat_timing"). (pass 9)
    #[serde(default)]
    pub teaches: Vec<String>,
    /// Whether this sample was part of a winning trajectory (set post-hoc). (pass 9)
    #[serde(default)]
    pub trajectory_won: Option<bool>,
    /// Final campaign progress at trajectory end (set post-hoc). (pass 9)
    #[serde(default)]
    pub campaign_outcome: Option<f32>,
    /// Cascade score: how much value improved over subsequent steps (set post-hoc). (pass 9)
    #[serde(default)]
    pub cascade_score: f32,
    /// Links positive/negative contrastive pair samples from the same root. (pass 9)
    #[serde(default)]
    pub contrastive_pair_id: Option<u32>,
    /// Whether this is the positive (best) or negative (worst) example in a pair. (pass 9)
    #[serde(default)]
    pub is_positive_example: bool,
    /// Scalar measuring how interesting/complex this state is for learning. (pass 9)
    #[serde(default)]
    pub state_complexity: f32,
    /// Estimated impact of this action on the state (0-1). (pass 10)
    #[serde(default)]
    pub estimated_impact: f32,
}

/// State delta between root and leaf — captures what changed during a branch. (pass 2)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StateDelta {
    /// Gold difference (leaf - root).
    pub gold_diff: f32,
    /// Reputation difference.
    pub reputation_diff: f32,
    /// Adventurer count difference.
    pub adventurer_count_diff: i32,
    /// Quest completions during the branch.
    pub quests_completed: u32,
    /// Quest failures during the branch.
    pub quests_failed: u32,
    /// Total level-ups during the branch.
    pub level_ups: u32,
    /// Deaths during the branch.
    pub deaths: u32,
    /// Progress difference (campaign_progress).
    pub progress_diff: f32,
    /// Threat level difference.
    pub threat_diff: f32,
}

/// A state snapshot used as a BFS root.
#[derive(Clone)]
struct RootState {
    state: CampaignState,
    seed: u64,
    wave: u32,
    difficulty: Difficulty,
}

/// Dataset format version. Increment when the BfsSample schema changes. (pass 6)
pub const BFS_DATASET_VERSION: u32 = 3;

/// Shared mutable state for UCB action selection and state novelty tracking
/// across all roots in a BFS wave. (pass 5)
struct BfsSharedState {
    /// How many times each action type has been selected across all roots.
    action_type_counts: HashMap<String, u64>,
    /// Total actions selected across all roots.
    total_action_selections: u64,
    /// State feature hash bucket visit counts for novelty computation.
    state_hash_counts: HashMap<u64, u64>,
    /// Total states seen (for novelty normalization).
    total_states_seen: u64,
}

impl BfsSharedState {
    fn new() -> Self {
        Self {
            action_type_counts: HashMap::new(),
            total_action_selections: 0,
            state_hash_counts: HashMap::new(),
            total_states_seen: 0,
        }
    }

    /// Record that an action type was selected.
    fn record_action(&mut self, action_type: &str) {
        *self.action_type_counts.entry(action_type.to_string()).or_insert(0) += 1;
        self.total_action_selections += 1;
    }

    /// UCB score for an action type: strategic_value + C * sqrt(ln(total) / count).
    fn ucb_score(&self, action_type: &str, strategic_value: f32) -> f32 {
        const C: f32 = 1.0;
        let count = *self.action_type_counts.get(action_type).unwrap_or(&0);
        let total = self.total_action_selections.max(1);
        if count == 0 {
            strategic_value + C * 5.0
        } else {
            let exploration = ((total as f32).ln() / count as f32).sqrt();
            strategic_value + C * exploration
        }
    }

    /// Compute a simple hash of key state features for novelty detection.
    fn state_hash(state: &CampaignState) -> u64 {
        let alive = state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .count() as u64;
        let gold_bucket = (state.guild.gold / 50.0).floor() as u64;
        let quest_count = state.active_quests.len() as u64;
        let crisis_count = state.overworld.active_crises.len() as u64;
        let war_count = state.factions.iter()
            .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
            .count() as u64;
        let progress_bucket = (state.overworld.campaign_progress * 10.0).floor() as u64;
        let rep_bucket = (state.guild.reputation / 20.0).floor() as u64;

        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for val in [alive, gold_bucket, quest_count,
                    crisis_count, war_count, progress_bucket, rep_bucket] {
            h ^= val;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h
    }

    /// Record a state visit and return novelty score (0-1).
    fn record_state_and_novelty(&mut self, state: &CampaignState) -> f32 {
        let hash = Self::state_hash(state);
        let count = self.state_hash_counts.entry(hash).or_insert(0);
        *count += 1;
        self.total_states_seen += 1;
        1.0 / (*count as f32).sqrt()
    }
}

/// Serializable snapshot of BFS hyperparameters for reproducibility. (pass 6)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BfsConfigSnapshot {
    pub dataset_version: u32,
    pub max_waves: u32,
    pub ticks_per_branch: u64,
    pub clusters_per_wave: usize,
    pub initial_roots: usize,
    pub trajectory_max_ticks: u64,
    pub root_sample_interval: u64,
    pub base_seed: u64,
    pub threads: usize,
    pub timestamp: String,
}

/// Configuration for BFS exploration.
#[derive(Clone, Debug)]
pub struct BfsConfig {
    /// Maximum BFS waves (0 = unlimited, run until all leaves are terminal).
    pub max_waves: u32,
    /// Ticks to advance after each action (let it play out).
    pub ticks_per_branch: u64,
    /// Number of clusters per wave (controls width).
    pub clusters_per_wave: usize,
    /// Number of initial root states to generate.
    pub initial_roots: usize,
    /// Max ticks for initial heuristic trajectory.
    pub trajectory_max_ticks: u64,
    /// Interval for sampling roots from heuristic trajectory.
    pub root_sample_interval: u64,
    /// Campaign config.
    pub campaign_config: CampaignConfig,
    /// Base seed.
    pub base_seed: u64,
    /// Number of threads.
    pub threads: usize,
    /// Output JSONL path.
    pub output_path: String,
    /// Optional LLM config for content generation.
    pub llm_config: Option<super::llm::LlmConfig>,
    /// Optional VAE model for instant content generation.
    pub vae_model: Option<std::sync::Arc<super::vae_inference::ContentVaeWeights>>,
}

impl BfsConfig {
    /// Create a serializable snapshot of the hyperparameters. (pass 6)
    pub fn snapshot(&self) -> BfsConfigSnapshot {
        BfsConfigSnapshot {
            dataset_version: BFS_DATASET_VERSION,
            max_waves: self.max_waves,
            ticks_per_branch: self.ticks_per_branch,
            clusters_per_wave: self.clusters_per_wave,
            initial_roots: self.initial_roots,
            trajectory_max_ticks: self.trajectory_max_ticks,
            root_sample_interval: self.root_sample_interval,
            base_seed: self.base_seed,
            threads: self.threads,
            timestamp: chrono_timestamp(),
        }
    }
}

/// ISO-8601 timestamp without pulling in chrono. (pass 6)
fn chrono_timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    let days = secs / 86400;
    let mut y: u64 = 1970;
    let mut rem = days;
    loop {
        let ylen: u64 = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < ylen { break; }
        rem -= ylen;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let mdays: [u64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0u64;
    for &ml in &mdays {
        if rem < ml { break; }
        rem -= ml;
        mo += 1;
    }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, s)
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_waves: 0, // 0 = run until completion
            ticks_per_branch: 200, // ~20s game time per branch
            clusters_per_wave: 20,
            initial_roots: 50,
            trajectory_max_ticks: 15000,
            root_sample_interval: 300,
            campaign_config: CampaignConfig::default(),
            base_seed: 2026,
            threads: 0,
            output_path: "generated/bfs_explore.jsonl".into(),
            llm_config: None,
            vae_model: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Action grouping — with strategic diversity
// ---------------------------------------------------------------------------

/// Categorize actions into strategic buckets for ensuring diversity.
fn strategic_bucket(action_type: &str) -> &'static str {
    // Check prefix matches for parameterized action types (pass 3)
    if action_type.starts_with("DiplomaticAction_") { return "diplomacy"; }
    if action_type.starts_with("SetSpendPriority_") { return "economy"; }
    match action_type {
        "AcceptQuest" | "DeclineQuest" | "DispatchQuest" | "AssignToPool" | "UnassignFromPool" => "quest_mgmt",
        "TrainAdventurer" | "EquipGear" => "development",
        "DiplomaticAction" | "ProposeCoalition" | "RequestCoalitionAid" => "diplomacy",
        "HireScout" => "scouting",
        "PurchaseSupplies" | "SendRunner" => "logistics",
        "HireMercenary" | "CallRescue" | "InterceptChampion" => "military",
        "SetSpendPriority" => "economy",
        "UseAbility" => "ability",
        "Rest" => "rest",
        "Wait" => "wait",
        "RespondToChoice" => "choice",
        "StartingChoice" => "setup",
        _ => "other",
    }
}

/// Estimate tension level of a state (0.0 = calm, 1.0+ = high tension). (pass 5)
fn estimate_tension(state: &CampaignState) -> f32 {
    let mut tension = 0.0f32;
    tension += state.overworld.active_crises.len() as f32 * 0.3;
    let wars = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .count() as f32;
    tension += wars * 0.4;
    if state.guild.gold < 50.0 { tension += 0.3; }
    if state.guild.gold < 20.0 { tension += 0.3; }
    tension += (state.overworld.global_threat_level / 100.0).clamp(0.0, 0.5);
    tension += state.active_battles.len() as f32 * 0.2;
    let injured = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.injury > 40.0)
        .count() as f32;
    tension += injured * 0.15;
    let low_loyalty = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0)
        .count() as f32;
    tension += low_loyalty * 0.2;
    tension
}

/// Rate how strategically interesting an action is for RL training data. (pass 2)
fn strategic_value(action_type: &str, state: &CampaignState) -> f32 {
    let base: f32 = match action_type {
        "Wait" => 0.05,
        "Rest" => {
            let has_injured = state.adventurers.iter()
                .any(|a| a.status != AdventurerStatus::Dead && a.injury > 30.0);
            let has_progression = !state.pending_progression.is_empty();
            if has_injured || has_progression { 0.5 } else { 0.1 }
        }
        "AcceptQuest" => 0.9,
        "DeclineQuest" => 0.6,
        "DispatchQuest" => 1.0,
        "AssignToPool" | "UnassignFromPool" => 0.5,
        "TrainAdventurer" => 0.7,
        "EquipGear" => 0.6,
        "DiplomaticAction_ImproveRelations" | "DiplomaticAction_TradeAgreement" => 0.8,
        "DiplomaticAction_Threaten" | "DiplomaticAction_ProposeCeasefire" => 0.9,
        "ProposeCoalition" => 0.95,
        "RequestCoalitionAid" => 0.85,
        "HireMercenary" => 0.8,
        "CallRescue" => 0.9,
        "InterceptChampion" => 1.0,
        "PurchaseSupplies" | "SendRunner" => 0.5,
        "HireScout" => 0.7,
        s if s.starts_with("SetSpendPriority") => 0.3,
        "UseAbility" => 0.85,
        "RespondToChoice" => 0.6,
        "StartingChoice" => 0.4,
        _ => 0.5,
    };
    let valid = state.valid_actions();
    let bucket = strategic_bucket(action_type);
    let bucket_count = valid.iter()
        .filter(|a| strategic_bucket(&action_type_name(a)) == bucket)
        .count();
    let rarity_bonus = if bucket_count <= 1 { 0.2 } else { 0.0 };
    (base + rarity_bonus).min(1.0_f32)
}

/// Compute strategic value for UCB scoring (simpler version for pass 5). (pass 5)
fn strategic_value_of(action_type: &str) -> f32 {
    match strategic_bucket(action_type) {
        "quest_mgmt" => 0.8,
        "development" => 0.6,
        "diplomacy" => 0.7,
        "military" => 0.9,
        "economy" => 0.5,
        "ability" => 0.7,
        "scouting" => 0.5,
        "logistics" => 0.4,
        "rest" => 0.3,
        "wait" => 0.1,
        _ => 0.5,
    }
}

// ---------------------------------------------------------------------------
// Action pruning by relevance (pass 10)
// ---------------------------------------------------------------------------

/// Remove actions that are clearly suboptimal given the current state.
/// This focuses BFS compute on decisions that actually matter while ensuring
/// at least 1 action per strategic bucket survives.
fn prune_irrelevant_actions(actions: &[CampaignAction], state: &CampaignState) -> Vec<CampaignAction> {
    if actions.len() <= 10 {
        return actions.to_vec();
    }

    let no_injured = !state.adventurers.iter()
        .any(|a| a.status != AdventurerStatus::Dead && (a.injury > 10.0 || a.stress > 30.0));
    let no_progression = state.pending_progression.is_empty();
    let _gold_high = state.guild.gold > 200.0;
    let supplies_high = state.guild.supplies > 80.0;
    let no_diseases = state.diseases.is_empty();
    let no_corruption_crisis = !state.overworld.active_crises.iter().any(|c| {
        matches!(c, ActiveCrisis::Corruption { corrupted_regions, .. } if !corrupted_regions.is_empty())
    });
    let no_wars = !state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::AtWar);
    let no_battles = state.active_battles.is_empty();
    let no_field_parties = !state.parties.iter()
        .any(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting));
    let all_scouted = state.overworld.locations.iter().all(|l| l.scouted);
    let no_active_quests = state.active_quests.is_empty();
    let no_idle = !state.adventurers.iter()
        .any(|a| a.status == AdventurerStatus::Idle);
    let low_gold = state.guild.gold < 30.0;
    let no_inventory = state.guild.inventory.is_empty();

    let mut kept = Vec::with_capacity(actions.len());

    for action in actions {
        let dominated = match action {
            CampaignAction::Rest if no_injured && no_progression => true,
            CampaignAction::PurchaseSupplies { .. } if supplies_high || no_field_parties => true,
            CampaignAction::SendRunner { .. } if no_field_parties => true,
            CampaignAction::CallRescue { .. } if no_battles => true,
            CampaignAction::HireMercenary { .. } if no_active_quests || low_gold => true,
            CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::ProposeCeasefire, .. }
                if no_wars => true,
            CampaignAction::RequestCoalitionAid { .. } if no_wars && no_battles => true,
            CampaignAction::HireScout { .. } if all_scouted => true,
            CampaignAction::TrainAdventurer { .. } if no_idle || low_gold => true,
            CampaignAction::EquipGear { .. } if no_inventory || no_idle => true,
            CampaignAction::InterceptChampion { .. } if no_battles
                && state.overworld.active_crises.is_empty() => true,
            CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::ImproveRelations, faction_id, .. }
                if no_corruption_crisis && no_wars && state.factions.iter()
                    .find(|f| f.id == *faction_id)
                    .map(|f| f.diplomatic_stance == DiplomaticStance::Friendly || f.diplomatic_stance == DiplomaticStance::Coalition)
                    .unwrap_or(false) => true,
            CampaignAction::SendRunner { party_id, payload: RunnerPayload::Supplies(_) }
                if no_diseases && state.parties.iter()
                    .find(|p| p.id == *party_id)
                    .map(|p| p.supply_level > 50.0)
                    .unwrap_or(true) => true,
            _ => false,
        };

        if !dominated {
            kept.push(action.clone());
        }
    }

    // Ensure at least 1 action per strategic bucket survives
    for action in actions {
        let atype = action_type_name(action);
        let bucket = strategic_bucket(&atype);
        let bucket_in_kept = kept.iter().any(|a| {
            strategic_bucket(&action_type_name(a)) == bucket
        });
        if !bucket_in_kept {
            kept.push(action.clone());
            break;
        }
    }

    if kept.len() < 3 && !actions.is_empty() {
        return actions.to_vec();
    }

    kept
}

// ---------------------------------------------------------------------------
// Action clustering (pass 10)
// ---------------------------------------------------------------------------

/// Group similar actions and sample representatives to reduce effective
/// action space from 100+ to ~30 while preserving diversity.
fn cluster_similar_actions(actions: &[CampaignAction], state: &CampaignState) -> Vec<CampaignAction> {
    if actions.len() <= 30 {
        return actions.to_vec();
    }

    let mut result: Vec<CampaignAction> = Vec::with_capacity(30);
    let mut dispatch_quests: Vec<&CampaignAction> = Vec::new();
    let mut spend_priorities: Vec<&CampaignAction> = Vec::new();
    let mut assign_pool: Vec<&CampaignAction> = Vec::new();
    let mut unassign_pool: Vec<&CampaignAction> = Vec::new();
    let mut diplomacy: Vec<&CampaignAction> = Vec::new();
    let mut train: Vec<&CampaignAction> = Vec::new();
    let mut equip: Vec<&CampaignAction> = Vec::new();
    let mut hire_scout: Vec<&CampaignAction> = Vec::new();
    let mut abilities: Vec<&CampaignAction> = Vec::new();
    let mut purchase: Vec<&CampaignAction> = Vec::new();
    let mut other: Vec<&CampaignAction> = Vec::new();

    for action in actions {
        match action {
            CampaignAction::DispatchQuest { .. } => dispatch_quests.push(action),
            CampaignAction::SetSpendPriority { .. } => spend_priorities.push(action),
            CampaignAction::AssignToPool { .. } => assign_pool.push(action),
            CampaignAction::UnassignFromPool { .. } => unassign_pool.push(action),
            CampaignAction::DiplomaticAction { .. }
            | CampaignAction::ProposeCoalition { .. }
            | CampaignAction::RequestCoalitionAid { .. } => diplomacy.push(action),
            CampaignAction::TrainAdventurer { .. } => train.push(action),
            CampaignAction::EquipGear { .. } => equip.push(action),
            CampaignAction::HireScout { .. } => hire_scout.push(action),
            CampaignAction::UseAbility { .. } => abilities.push(action),
            CampaignAction::PurchaseSupplies { .. } => purchase.push(action),
            _ => other.push(action),
        }
    }

    fn sample_k<'a>(items: &[&'a CampaignAction], k: usize) -> Vec<&'a CampaignAction> {
        if items.len() <= k { return items.to_vec(); }
        let mut sampled = Vec::with_capacity(k);
        for i in 0..k {
            let idx = i * (items.len() - 1) / (k - 1).max(1);
            sampled.push(items[idx]);
        }
        sampled
    }

    for &a in &sample_k(&dispatch_quests, 3) { result.push(a.clone()); }
    for &a in &spend_priorities { result.push(a.clone()); }
    for &a in &sample_k(&assign_pool, 3) { result.push(a.clone()); }
    for &a in &sample_k(&unassign_pool, 2) { result.push(a.clone()); }
    for &a in &sample_k(&diplomacy, 4) { result.push(a.clone()); }
    for &a in &sample_k(&train, 2) { result.push(a.clone()); }
    for &a in &sample_k(&equip, 2) { result.push(a.clone()); }
    for &a in &sample_k(&hire_scout, 2) { result.push(a.clone()); }

    let mut sorted_abilities = abilities.clone();
    sorted_abilities.sort_by(|a, b| {
        let ca = action_gold_cost(a, state);
        let cb = action_gold_cost(b, state);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &a in &sample_k(&sorted_abilities, 3) { result.push(a.clone()); }

    for &a in &sample_k(&purchase, 2) { result.push(a.clone()); }
    for &a in &other { result.push(a.clone()); }

    result
}

// ---------------------------------------------------------------------------
// Hierarchical action selection (pass 10)
// ---------------------------------------------------------------------------

/// Two-level action selection: first pick a strategic bucket via UCB,
/// then pick an action within the selected bucket uniformly.
fn hierarchical_select_actions(
    grouped: &[(String, CampaignAction)],
    target_count: usize,
    shared: &Mutex<BfsSharedState>,
    rng_seed: u64,
) -> Vec<usize> {
    if grouped.len() <= target_count {
        return (0..grouped.len()).collect();
    }

    let mut bucket_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, (action_type, _)) in grouped.iter().enumerate() {
        let bucket = strategic_bucket(action_type);
        bucket_indices.entry(bucket).or_default().push(idx);
    }

    let buckets: Vec<&str> = bucket_indices.keys().copied().collect();
    if buckets.is_empty() {
        return vec![];
    }

    let mut selected: Vec<usize> = Vec::with_capacity(target_count);
    let mut selected_set: HashSet<usize> = HashSet::new();
    let mut rng = rng_seed;

    let mut bucket_scores: Vec<(&str, f32)> = {
        let shared_guard = shared.lock().unwrap();
        buckets.iter().map(|&bucket| {
            let sv = strategic_value_of(
                bucket_indices[bucket].first()
                    .map(|&i| grouped[i].0.as_str())
                    .unwrap_or("Wait")
            );
            let ucb = shared_guard.ucb_score(bucket, sv);
            (bucket, ucb)
        }).collect()
    };
    bucket_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut round = 0;
    while selected.len() < target_count {
        let mut any_added = false;
        for &(bucket, _) in &bucket_scores {
            if selected.len() >= target_count { break; }
            let indices = &bucket_indices[bucket];
            if round < indices.len() {
                xorshift(&mut rng);
                let available: Vec<usize> = indices.iter()
                    .filter(|i| !selected_set.contains(i))
                    .copied()
                    .collect();
                if !available.is_empty() {
                    let pick = available[rng as usize % available.len()];
                    selected_set.insert(pick);
                    selected.push(pick);
                    any_added = true;
                }
            }
        }
        round += 1;
        if !any_added { break; }
    }

    selected
}

// ---------------------------------------------------------------------------
// Action impact estimation (pass 10)
// ---------------------------------------------------------------------------

/// Quick heuristic to estimate how much an action will change the state.
fn estimate_impact(action: &CampaignAction, state: &CampaignState) -> f32 {
    let base = match action {
        CampaignAction::InterceptChampion { .. } => 0.9,
        CampaignAction::CallRescue { .. } => 0.85,
        CampaignAction::HireMercenary { .. } => 0.75,
        CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::Threaten, .. } => 0.8,
        CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::ProposeCeasefire, .. } => 0.85,
        CampaignAction::ProposeCoalition { .. } => 0.8,
        CampaignAction::RequestCoalitionAid { .. } => 0.7,
        CampaignAction::DispatchQuest { .. } => 0.7,
        CampaignAction::AcceptQuest { .. } => 0.65,
        CampaignAction::TrainAdventurer { .. } => 0.55,
        CampaignAction::UseAbility { .. } => 0.65,
        CampaignAction::EquipGear { .. } => 0.5,
        CampaignAction::HireScout { .. } => 0.5,
        CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::ImproveRelations, .. } => 0.45,
        CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::TradeAgreement, .. } => 0.5,
        CampaignAction::DiplomaticAction { action_type: DiplomacyActionType::RequestAid, .. } => 0.55,
        CampaignAction::PurchaseSupplies { amount, .. } => (amount / 50.0).clamp(0.2, 0.6),
        CampaignAction::SendRunner { .. } => 0.4,
        CampaignAction::AssignToPool { .. } => 0.35,
        CampaignAction::UnassignFromPool { .. } => 0.3,
        CampaignAction::DeclineQuest { .. } => 0.3,
        CampaignAction::RespondToChoice { .. } => 0.5,
        CampaignAction::SetSpendPriority { .. } => 0.2,
        CampaignAction::ChooseStartingPackage { .. } => 0.4,
        CampaignAction::Rest => {
            let has_injured = state.adventurers.iter()
                .any(|a| a.status != AdventurerStatus::Dead && a.injury > 30.0);
            if has_injured { 0.5 } else { 0.15 }
        }
        CampaignAction::Wait => 0.05,
    };

    let mut impact = base;
    if !state.overworld.active_crises.is_empty() {
        impact *= 1.15;
    }
    let cost = action_gold_cost(action, state);
    if cost > 0.0 && state.guild.gold < 50.0 {
        impact *= 1.2;
    }
    if matches!(action, CampaignAction::DispatchQuest { .. }) && state.active_quests.len() <= 1 {
        impact *= 1.2;
    }
    impact.clamp(0.0, 1.0)
}

/// Strategic rollout modes for branch simulation. (pass 2)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RolloutMode {
    Mixed,
    Aggressive,
    Economic,
    Diplomatic,
    Consistent,
}

impl RolloutMode {
    fn label(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Aggressive => "aggressive",
            Self::Economic => "economic",
            Self::Diplomatic => "diplomatic",
            Self::Consistent => "consistent",
        }
    }

    fn from_action(_action_type: &str, rng: &mut u64) -> Self {
        xorshift(rng);
        let roll = *rng % 100;
        match roll {
            0..=29 => Self::Consistent,
            30..=49 => Self::Aggressive,
            50..=64 => Self::Economic,
            65..=79 => Self::Diplomatic,
            _ => Self::Mixed,
        }
    }

    fn prefers_bucket(self, bucket: &str) -> bool {
        match self {
            Self::Aggressive => matches!(bucket, "quest_mgmt" | "military" | "ability"),
            Self::Economic => matches!(bucket, "economy" | "logistics" | "development" | "scouting"),
            Self::Diplomatic => matches!(bucket, "diplomacy"),
            Self::Consistent | Self::Mixed => false,
        }
    }
}

/// Get one representative action per action type from valid actions.
/// For diplomacy actions, split by stance-specific subtypes to preserve diversity.
fn group_actions(valid_actions: &[CampaignAction]) -> Vec<(String, CampaignAction)> {
    let mut seen: HashMap<String, CampaignAction> = HashMap::new();
    for action in valid_actions {
        let type_name = match action {
            CampaignAction::DiplomaticAction { action_type, .. } => {
                format!("DiplomaticAction_{:?}", action_type)
            }
            CampaignAction::SetSpendPriority { priority } => {
                format!("SetSpendPriority_{:?}", priority)
            }
            _ => action_type_name(action),
        };
        seen.entry(type_name).or_insert_with(|| action.clone());
    }
    seen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Action encoding for RL (pass 4)
// ---------------------------------------------------------------------------

/// Number of action types for one-hot encoding.
const NUM_ACTION_TYPES: usize = 21;
/// Number of strategic buckets for one-hot encoding.
const NUM_STRATEGIC_BUCKETS: usize = 11;
/// Total action encoding dimension.
pub const ACTION_ENCODING_DIM: usize = NUM_ACTION_TYPES + NUM_STRATEGIC_BUCKETS + 3; // 35

/// Temporal discount factor per tick elapsed.
const GAMMA: f64 = 0.99;

/// Map action type name to a stable index for one-hot encoding.
fn action_type_index(action_type: &str) -> usize {
    match action_type {
        "Wait" => 0,
        "Rest" => 1,
        "AcceptQuest" => 2,
        "DeclineQuest" => 3,
        "AssignToPool" => 4,
        "UnassignFromPool" => 5,
        "DispatchQuest" => 6,
        "PurchaseSupplies" => 7,
        "TrainAdventurer" => 8,
        "EquipGear" => 9,
        "SendRunner" => 10,
        "HireMercenary" => 11,
        "CallRescue" => 12,
        "HireScout" => 13,
        "DiplomaticAction" => 14,
        "UseAbility" => 15,
        "SetSpendPriority" => 16,
        "StartingChoice" => 17,
        "RespondToChoice" => 18,
        "ProposeCoalition" => 19,
        "RequestCoalitionAid" => 20,
        s if s.starts_with("DiplomaticAction_") => 14,
        s if s.starts_with("SetSpendPriority_") => 16,
        _ => 0,
    }
}

/// Map strategic bucket name to a stable index for one-hot encoding.
fn strategic_bucket_index(bucket: &str) -> usize {
    match bucket {
        "quest_mgmt" => 0,
        "development" => 1,
        "diplomacy" => 2,
        "scouting" => 3,
        "logistics" => 4,
        "military" => 5,
        "economy" => 6,
        "ability" => 7,
        "rest" => 8,
        "wait" => 9,
        _ => 10,
    }
}

/// Produce a fixed-size numeric encoding of a campaign action.
pub fn encode_action(action: &CampaignAction, state: &CampaignState) -> Vec<f32> {
    let mut enc = vec![0.0f32; ACTION_ENCODING_DIM];
    let type_name = action_type_name(action);
    let type_idx = action_type_index(&type_name);
    enc[type_idx] = 1.0;
    let bucket = strategic_bucket(&type_name);
    let bucket_idx = strategic_bucket_index(bucket);
    enc[NUM_ACTION_TYPES + bucket_idx] = 1.0;

    let target_id: f32 = match action {
        CampaignAction::AcceptQuest { request_id } | CampaignAction::DeclineQuest { request_id } => *request_id as f32,
        CampaignAction::AssignToPool { adventurer_id, .. } | CampaignAction::UnassignFromPool { adventurer_id, .. } => *adventurer_id as f32,
        CampaignAction::DispatchQuest { quest_id } => *quest_id as f32,
        CampaignAction::PurchaseSupplies { party_id, .. } | CampaignAction::SendRunner { party_id, .. } => *party_id as f32,
        CampaignAction::TrainAdventurer { adventurer_id, .. } | CampaignAction::EquipGear { adventurer_id, .. } => *adventurer_id as f32,
        CampaignAction::HireMercenary { quest_id } => *quest_id as f32,
        CampaignAction::CallRescue { battle_id } => *battle_id as f32,
        CampaignAction::HireScout { location_id } => *location_id as f32,
        CampaignAction::DiplomaticAction { faction_id, .. } | CampaignAction::ProposeCoalition { faction_id } | CampaignAction::RequestCoalitionAid { faction_id } => *faction_id as f32,
        CampaignAction::UseAbility { unlock_id, .. } => *unlock_id as f32,
        CampaignAction::InterceptChampion { party_id, .. } => *party_id as f32,
        _ => 0.0,
    };
    enc[NUM_ACTION_TYPES + NUM_STRATEGIC_BUCKETS] = target_id / 100.0;

    let cost = action_gold_cost(action, state);
    enc[NUM_ACTION_TYPES + NUM_STRATEGIC_BUCKETS + 1] = (1.0 + cost).ln() / 5.0;

    let duration = action_expected_duration(action);
    enc[NUM_ACTION_TYPES + NUM_STRATEGIC_BUCKETS + 2] = duration as f32 / 1000.0;

    enc
}

fn action_gold_cost(action: &CampaignAction, state: &CampaignState) -> f32 {
    match action {
        CampaignAction::PurchaseSupplies { amount, .. } => amount * state.config.economy.supply_cost_per_unit,
        CampaignAction::TrainAdventurer { .. } => state.config.economy.training_cost,
        CampaignAction::SendRunner { .. } => state.config.economy.runner_cost,
        CampaignAction::HireMercenary { .. } => state.config.economy.mercenary_cost,
        CampaignAction::CallRescue { .. } => state.config.economy.rescue_bribe_cost,
        CampaignAction::HireScout { .. } => state.config.economy.scout_cost,
        CampaignAction::UseAbility { unlock_id, .. } => {
            state.unlocks.iter().find(|u| u.id == *unlock_id)
                .map(|u| u.properties.resource_cost).unwrap_or(0.0)
        }
        CampaignAction::DiplomaticAction { action_type, .. } => match action_type {
            DiplomacyActionType::ImproveRelations => 10.0,
            DiplomacyActionType::TradeAgreement => 15.0,
            DiplomacyActionType::Threaten => 5.0,
            DiplomacyActionType::ProposeCeasefire => 25.0,
            DiplomacyActionType::RequestAid => 0.0,
        },
        _ => 0.0,
    }
}

fn action_expected_duration(action: &CampaignAction) -> u64 {
    match action {
        CampaignAction::Wait => 1,
        CampaignAction::Rest => 50,
        CampaignAction::AcceptQuest { .. } | CampaignAction::DeclineQuest { .. } => 1,
        CampaignAction::AssignToPool { .. } | CampaignAction::UnassignFromPool { .. } => 1,
        CampaignAction::DispatchQuest { .. } => 200,
        CampaignAction::PurchaseSupplies { .. } => 5,
        CampaignAction::TrainAdventurer { .. } => 100,
        CampaignAction::EquipGear { .. } => 5,
        CampaignAction::SendRunner { .. } => 50,
        CampaignAction::HireMercenary { .. } => 30,
        CampaignAction::CallRescue { .. } => 80,
        CampaignAction::HireScout { .. } => 100,
        CampaignAction::DiplomaticAction { .. } => 20,
        CampaignAction::ProposeCoalition { .. } => 30,
        CampaignAction::RequestCoalitionAid { .. } => 40,
        CampaignAction::UseAbility { .. } => 10,
        CampaignAction::SetSpendPriority { .. } => 1,
        CampaignAction::ChooseStartingPackage { .. } => 1,
        CampaignAction::RespondToChoice { .. } => 1,
        CampaignAction::InterceptChampion { .. } => 150,
    }
}

/// Compute temporal discount factor: gamma^(ticks_elapsed/100).
fn temporal_discount(ticks_elapsed: u64) -> f32 {
    let steps = ticks_elapsed as f64 / 100.0;
    GAMMA.powf(steps) as f32
}

/// Compute replay priority for prioritized experience replay. (pass 4)
fn compute_replay_priority(
    advantage: f32,
    action_type: &str,
    action_type_counts: &HashMap<String, u64>,
    state: &CampaignState,
) -> f32 {
    let mut priority = 1.0f32;
    priority += advantage.abs() * 2.0;
    let count = action_type_counts.get(action_type).copied().unwrap_or(0);
    if count < 5 { priority += 3.0; } else if count < 20 { priority += 1.0; }
    if state.guild.gold < 30.0 { priority += 2.0; }
    if !state.overworld.active_crises.is_empty() { priority += 1.5; }
    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead).count();
    if alive <= 2 { priority += 2.5; }
    if !state.active_battles.is_empty() { priority += 1.0; }
    if state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::AtWar) { priority += 1.0; }
    if (action_type == "Wait" || action_type == "Rest") && state.guild.gold > 100.0 && alive >= 4 {
        priority *= 0.3;
    }
    priority.max(0.1)
}

// ---------------------------------------------------------------------------
// Leaf value estimation
// ---------------------------------------------------------------------------

/// Quick heuristic value for a campaign state.
/// Higher = better. Range roughly 0-5 with multiplicative factors for spread.
fn estimate_value(state: &CampaignState) -> f32 {
    let adventurers: Vec<_> = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .collect();
    let alive = adventurers.len() as f32;

    if alive == 0.0 {
        return 0.0;
    }

    let mean_health = adventurers.iter()
        .map(|a| {
            let injury_penalty = a.injury / 100.0;
            let stress_penalty = a.stress / 200.0;
            let morale_bonus = a.morale / 200.0;
            (1.0 - injury_penalty - stress_penalty + morale_bonus).clamp(0.0, 1.0)
        })
        .sum::<f32>() / alive;

    let gold_score = (state.guild.gold / 200.0).clamp(0.0, 1.0);
    let supply_score = (state.guild.supplies / 100.0).clamp(0.0, 1.0);
    let rep_score = state.guild.reputation / 100.0;

    let progress = state.overworld.campaign_progress;
    let quests_won = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Victory).count() as f32;
    let quests_lost = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Defeat).count() as f32;
    let win_rate = if quests_won + quests_lost > 0.0 {
        quests_won / (quests_won + quests_lost)
    } else {
        0.5
    };

    let threat = (state.overworld.global_threat_level / 100.0).clamp(0.0, 1.0);
    let crisis_pressure = (state.overworld.active_crises.len() as f32 * 0.25).min(1.0);

    let idle = adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .count() as f32;
    let _capacity = (idle / alive.max(1.0)).clamp(0.0, 1.0);
    let quest_load = (state.active_quests.len() as f32 / state.guild.active_quest_capacity.max(1) as f32)
        .clamp(0.0, 1.0);

    let mean_level = adventurers.iter().map(|a| a.level as f32).sum::<f32>() / alive;
    let level_score = (mean_level / 20.0).min(1.0);

    let injured_count = adventurers.iter()
        .filter(|a| a.injury > 40.0).count() as f32;
    let deserted = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Dead).count() as f32;

    let roster_factor = (alive / (alive + deserted).max(1.0)).powf(0.5);
    let health_factor = mean_health.powf(1.5);
    let economy_factor = (gold_score * 0.6 + supply_score * 0.2 + rep_score * 0.2).powf(0.8);

    let progress_score = progress * 2.0 + (quests_won / 25.0).min(1.0) + win_rate;

    let war_penalty = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .map(|f| {
            let strength_ratio = f.military_strength / 50.0;
            strength_ratio.min(3.0) * 0.3
        })
        .sum::<f32>();

    let guild_faction_id = state.diplomacy.guild_faction_id;
    let guild_regions = state.overworld.regions.iter()
        .filter(|r| r.owner_faction_id == guild_faction_id)
        .count() as f32;
    let total_regions = state.overworld.regions.len().max(1) as f32;
    let territory_score = guild_regions / total_regions;
    let territory_penalty = if territory_score < 0.5 { (1.0 - territory_score) * 0.8 } else { 0.0 };

    let mean_visibility = if state.overworld.regions.is_empty() {
        0.5
    } else {
        state.overworld.regions.iter().map(|r| r.visibility).sum::<f32>()
            / state.overworld.regions.len() as f32
    };
    let scouting_bonus = (mean_visibility - 0.3).max(0.0) * 0.4;

    let rival_penalty = if state.rival_guild.active {
        let rep_gap = (state.rival_guild.reputation - state.guild.reputation) / 100.0;
        let quest_gap = (state.rival_guild.quests_completed as f32
            - state.completed_quests.len() as f32) / 20.0;
        (rep_gap.max(0.0) * 0.4 + quest_gap.max(0.0) * 0.3).min(1.0)
    } else {
        0.0
    };

    let buildings = &state.guild_buildings;
    let total_building_tiers = (buildings.training_grounds
        + buildings.watchtower + buildings.trade_post
        + buildings.barracks + buildings.infirmary
        + buildings.war_room) as f32;
    let building_bonus = (total_building_tiers / 18.0).min(1.0) * 0.3;

    let bond_count = state.adventurer_bonds.len() as f32;
    let mean_bond = if bond_count > 0.0 {
        state.adventurer_bonds.values().sum::<f32>() / bond_count / 100.0
    } else {
        0.0
    };
    let bond_bonus = mean_bond * 0.15;

    let coalition_allies = state.factions.iter()
        .filter(|f| f.coalition_member)
        .count() as f32;
    let coalition_bonus = (coalition_allies * 0.15).min(0.5);

    let trade_bonus = (state.guild.total_trade_income / 100.0).min(0.3);

    let season_modifier = match state.overworld.season {
        Season::Winter => -0.1,
        Season::Summer => 0.05,
        _ => 0.0,
    };

    let desertion_risk = adventurers.iter()
        .filter(|a| a.loyalty < 30.0)
        .count() as f32 * 0.1;

    // --- Extended system tracker value contributions (pass 3) ---
    let t = &state.system_trackers;
    let espionage_bonus = (t.total_intel_gathered / 100.0).min(0.3);
    let merc_bonus = (t.total_mercenary_strength / 100.0).min(0.25);
    let debt_penalty = (t.total_debt / 200.0).min(0.4);
    let monster_penalty = (t.highest_monster_aggression / 100.0 * 0.3).min(0.3)
        + (t.total_monster_population / 200.0 * 0.2).min(0.2);
    let prisoner_bonus = (t.prisoner_count as f32 * 0.05).min(0.2);
    let captured_penalty = t.captured_adventurer_count as f32 * 0.1;
    let caravan_bonus = (t.active_caravan_routes as f32 * 0.06
        + t.caravan_trade_income / 100.0).min(0.3);
    let civil_war_value = if t.guild_civil_war_involvement {
        -(t.active_civil_war_count as f32 * 0.15).min(0.3)
    } else {
        (t.active_civil_war_count as f32 * 0.05).min(0.15)
    };
    let bm_risk = (t.black_market_heat / 100.0 * 0.2).min(0.2);
    let agreement_bonus = ((t.trade_agreement_count + t.non_aggression_pact_count
        + t.mutual_defense_count) as f32 * 0.04).min(0.25);
    let legacy_bonus = (t.total_legacy_bonuses / 100.0).min(0.2);
    let war_exhaust_penalty = (t.max_war_exhaustion / 100.0 * 0.25).min(0.25);
    let narrative_bonus = ((t.total_deeds_earned as f32) / 30.0).min(0.1);
    let pop_bonus = (t.total_population / 1000.0 * t.mean_population_morale / 100.0).min(0.2);

    // --- Stuckness penalty (pass 2) ---
    let quests_total = quests_won + quests_lost;
    let _ticks_per_quest = if quests_total > 0.0 {
        state.tick as f32 / quests_total
    } else {
        state.tick as f32
    };
    let stuck_penalty = if state.tick > 5000 && quests_total < 1.0 {
        0.5
    } else if _ticks_per_quest > 3000.0 && state.tick > 3000 {
        0.2
    } else {
        0.0
    };

    let injured_penalty = injured_count * 0.15;
    let threat_penalty = threat * 0.5 + crisis_pressure * 0.4;
    let bankruptcy_penalty = if state.guild.gold < 20.0 { 0.5 } else { 0.0 };

    let base = progress_score + level_score * 0.5 + quest_load * 0.3
        + territory_score * 0.5
        + scouting_bonus + building_bonus + bond_bonus
        + coalition_bonus + trade_bonus + season_modifier
        // Pass 3 additions:
        + espionage_bonus + merc_bonus + prisoner_bonus + caravan_bonus
        + civil_war_value + agreement_bonus + legacy_bonus
        + narrative_bonus + pop_bonus;
    let v = base * roster_factor * health_factor * economy_factor
        - injured_penalty - threat_penalty - bankruptcy_penalty
        - war_penalty - territory_penalty - rival_penalty - desertion_risk
        - stuck_penalty
        // Pass 3 penalties:
        - debt_penalty - monster_penalty - captured_penalty - bm_risk
        - war_exhaust_penalty;

    v.clamp(0.0, 5.0)
}

/// Average value by game phase, used for relative value computation. (pass 2)
fn phase_average_value(phase: &str) -> f32 {
    match phase {
        "early" => 1.2,
        "mid" => 2.0,
        "late" => 2.5,
        _ => 1.5,
    }
}

/// Compute relative value: how this state compares to average at this game phase. (pass 2)
fn relative_value(value: f32, phase: &str) -> f32 {
    let avg = phase_average_value(phase);
    if avg > 0.0 { (value - avg) / avg } else { value }
}

/// Feature vector for clustering leaf states.
/// Expanded to capture scouting, buildings, bonds, rival, season, and system trackers.
fn state_features(state: &CampaignState) -> Vec<f32> {
    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count() as f32;
    let idle = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .count() as f32;
    let gold_bucket = (state.guild.gold / 50.0).floor().min(10.0);
    let rep = state.guild.reputation / 20.0;
    let active_quests = state.active_quests.len() as f32;
    let active_battles = state.active_battles.len() as f32;
    let parties = state.parties.len() as f32;
    let progress = state.overworld.campaign_progress * 10.0;
    let threat = state.overworld.global_threat_level / 20.0;
    let pending_choices = state.pending_choices.len() as f32;
    let unlocks = state.unlocks.len() as f32;
    let (mean_stress, mean_injury, mean_morale) = if alive > 0.0 {
        let advs: Vec<_> = state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .collect();
        (
            advs.iter().map(|a| a.stress).sum::<f32>() / alive / 20.0,
            advs.iter().map(|a| a.injury).sum::<f32>() / alive / 20.0,
            advs.iter().map(|a| a.morale).sum::<f32>() / alive / 20.0,
        )
    } else { (0.0, 0.0, 0.0) };

    let mean_level = if alive > 0.0 {
        state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.level as f32).sum::<f32>() / alive / 10.0
    } else { 0.0 };

    let crises = state.overworld.active_crises.len() as f32;
    let supplies = (state.guild.supplies / 20.0).min(5.0);
    let quests_done = (state.completed_quests.len() as f32 / 5.0).min(10.0);

    let mean_visibility = if state.overworld.regions.is_empty() {
        0.5
    } else {
        state.overworld.regions.iter().map(|r| r.visibility).sum::<f32>()
            / state.overworld.regions.len() as f32
    };

    let b = &state.guild_buildings;
    let building_total = (b.training_grounds + b.watchtower + b.trade_post
        + b.barracks + b.infirmary + b.war_room) as f32 / 6.0;

    let rival_rep = if state.rival_guild.active {
        state.rival_guild.reputation / 20.0
    } else {
        0.0
    };

    let bond_density = state.adventurer_bonds.len() as f32 / (alive.max(1.0) * (alive.max(1.0) - 1.0) / 2.0).max(1.0);

    let coalition_count = state.factions.iter()
        .filter(|f| f.coalition_member).count() as f32;
    let war_count = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar).count() as f32;

    let season_idx = match state.overworld.season {
        Season::Spring => 0.0,
        Season::Summer => 1.0,
        Season::Autumn => 2.0,
        Season::Winter => 3.0,
    };

    let guild_faction_id = state.diplomacy.guild_faction_id;
    let guild_regions = state.overworld.regions.iter()
        .filter(|r| r.owner_faction_id == guild_faction_id)
        .count() as f32;
    let territory_ratio = guild_regions / state.overworld.regions.len().max(1) as f32;

    // --- Extended system tracker dimensions for clustering (pass 3) ---
    let t = &state.system_trackers;
    let spy_intel = (t.total_intel_gathered / 50.0).min(5.0);
    let merc_strength = (t.total_mercenary_strength / 50.0).min(3.0);
    let bm_heat = t.black_market_heat / 20.0;
    let prisoners = t.prisoner_count as f32;
    let debt_bucket = (t.total_debt / 50.0).floor().min(5.0);
    let credit = t.credit_rating / 20.0;
    let rumors = t.active_rumor_count as f32;
    let civil_wars = t.active_civil_war_count as f32;
    let trade_agreements = t.trade_agreement_count as f32;
    let rival_rep_gap = t.rival_reputation_gap / 20.0;
    let caravans = t.active_caravan_routes as f32;
    let retired = t.retired_count as f32;
    let monster_pop = (t.total_monster_population / 50.0).min(5.0);
    let monster_aggro = t.highest_monster_aggression / 20.0;
    let festivals = t.active_festival_count as f32;
    let mentorships = t.active_mentorship_count as f32;
    let rivalries = t.active_rivalry_count as f32;
    let war_exhaust = t.max_war_exhaustion / 20.0;
    let chronicle = (t.chronicle_entry_count as f32 / 5.0).min(5.0);
    let deeds = (t.total_deeds_earned as f32 / 5.0).min(5.0);
    let population = (t.total_population / 200.0).min(5.0);
    let site_prep = (t.total_site_preparation / 5.0).min(3.0);

    vec![
        alive, idle, gold_bucket, rep, active_quests, active_battles,
        parties, progress, threat, pending_choices, unlocks, mean_stress,
        mean_injury, mean_morale, mean_level, crises, supplies, quests_done,
        // Extended dimensions (pass 2):
        mean_visibility, building_total, rival_rep, bond_density,
        coalition_count, war_count, season_idx, territory_ratio,
        // Extended dimensions (pass 3 — system trackers):
        spy_intel, merc_strength, bm_heat, prisoners, debt_bucket, credit,
        rumors, civil_wars, trade_agreements, rival_rep_gap, caravans,
        retired, monster_pop, monster_aggro, festivals, mentorships,
        rivalries, war_exhaust, chronicle, deeds, population, site_prep,
    ]
}

// ---------------------------------------------------------------------------
// Clustering (k-medoids with extreme preservation)
// ---------------------------------------------------------------------------

/// Cluster leaf states into k groups, return representative indices.
/// Always preserves the best and worst leaves to keep extreme outcomes.
fn cluster_and_select_diverse(
    leaves: &[(CampaignState, Vec<f32>)],
    k: usize,
    leaf_values: &[f32],
) -> Vec<usize> {
    if leaves.len() <= k {
        return (0..leaves.len()).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(k);

    if let Some(best_idx) = leaf_values.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        selected.push(best_idx);
    }
    if let Some(worst_idx) = leaf_values.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        if !selected.contains(&worst_idx) {
            selected.push(worst_idx);
        }
    }

    let remaining_k = k.saturating_sub(selected.len());
    if remaining_k > 0 {
        let medoids = cluster_and_select_medians(leaves, remaining_k + selected.len());
        for idx in medoids {
            if !selected.contains(&idx) && selected.len() < k {
                selected.push(idx);
            }
        }
    }

    selected
}

/// Cluster leaf states into k groups, return the median state index per cluster.
fn cluster_and_select_medians(
    leaves: &[(CampaignState, Vec<f32>)],
    k: usize,
) -> Vec<usize> {
    if leaves.len() <= k {
        return (0..leaves.len()).collect();
    }

    let step = leaves.len() / k;
    let mut medoid_indices: Vec<usize> = (0..k).map(|i| i * step).collect();

    for _ in 0..5 {
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, (_, features)) in leaves.iter().enumerate() {
            let nearest = medoid_indices
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    let da = distance(&leaves[a].1, features);
                    let db = distance(&leaves[b].1, features);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            clusters[nearest].push(i);
        }

        for (ci, members) in clusters.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            let dim = leaves[0].1.len();
            let centroid: Vec<f32> = (0..dim)
                .map(|d| {
                    members.iter().map(|&m| leaves[m].1[d]).sum::<f32>() / members.len() as f32
                })
                .collect();
            let best = members
                .iter()
                .min_by(|&&a, &&b| {
                    let da = distance(&leaves[a].1, &centroid);
                    let db = distance(&leaves[b].1, &centroid);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(members[0]);
            medoid_indices[ci] = best;
        }
    }

    medoid_indices
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Game phase tagging
// ---------------------------------------------------------------------------

/// Classify tick into game phase for coverage tracking.
fn phase_tag(tick: u64, progress: f32) -> String {
    if progress < 0.2 || tick < 2000 {
        "early".to_string()
    } else if progress < 0.6 || tick < 8000 {
        "mid".to_string()
    } else {
        "late".to_string()
    }
}

// ---------------------------------------------------------------------------
// Curriculum design (pass 9)
// ---------------------------------------------------------------------------

/// Compute curriculum difficulty tier (1-4) from state complexity metrics.
fn compute_curriculum_tier(state: &CampaignState, valid_action_count: usize) -> u8 {
    let war_count = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .count();
    let crisis_count = state.overworld.active_crises.len();
    let threat = state.overworld.global_threat_level;
    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();
    let total_adventurers = state.adventurers.len();
    let progress = state.overworld.campaign_progress;

    let t = &state.system_trackers;
    let active_systems = [
        war_count > 0,
        crisis_count > 0,
        t.total_intel_gathered > 10.0,
        t.total_mercenary_strength > 0.0,
        t.active_civil_war_count > 0,
        t.prisoner_count > 0,
        t.active_caravan_routes > 0,
        t.black_market_heat > 10.0,
        state.rival_guild.active,
        !state.active_battles.is_empty(),
        state.guild_buildings.training_grounds + state.guild_buildings.watchtower
            + state.guild_buildings.trade_post + state.guild_buildings.barracks
            + state.guild_buildings.infirmary + state.guild_buildings.war_room > 3,
        state.factions.iter().any(|f| f.coalition_member),
    ].iter().filter(|&&x| x).count();

    let near_defeat = alive <= 2 && total_adventurers >= 4;
    let simultaneous_crises = crisis_count >= 2;
    let civil_war = t.active_civil_war_count > 0 && t.guild_civil_war_involvement;

    if (simultaneous_crises && war_count > 0)
        || civil_war
        || (near_defeat && (crisis_count > 0 || war_count > 0))
        || (active_systems >= 8 && threat > 60.0)
        || (progress > 0.8 && active_systems >= 6)
    {
        return 4;
    }

    if war_count > 0
        || crisis_count > 0
        || t.total_intel_gathered > 20.0
        || t.total_mercenary_strength > 30.0
        || active_systems >= 5
        || (threat > 50.0 && valid_action_count >= 10)
    {
        return 3;
    }

    if valid_action_count >= 6
        || state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::Hostile
            || f.diplomatic_stance == DiplomaticStance::Friendly)
        || state.guild.gold > 100.0
        || active_systems >= 3
        || progress > 0.3
    {
        return 2;
    }

    1
}

/// Derive skill tags from what systems are active and what actions are available.
fn compute_teaches(
    state: &CampaignState,
    action_type: &str,
    delta: &Option<StateDelta>,
    valid_action_types: &[String],
) -> Vec<String> {
    let mut skills = Vec::new();

    match strategic_bucket(action_type) {
        "economy" | "logistics" => skills.push("economy".to_string()),
        "diplomacy" => skills.push("diplomacy".to_string()),
        "quest_mgmt" => skills.push("quest_management".to_string()),
        "military" => skills.push("combat_timing".to_string()),
        "development" => skills.push("development".to_string()),
        "scouting" => skills.push("exploration".to_string()),
        "rest" => skills.push("resource_recovery".to_string()),
        _ => {}
    }

    if state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::AtWar) {
        skills.push("war_management".to_string());
    }
    if !state.overworld.active_crises.is_empty() {
        skills.push("crisis_response".to_string());
    }
    if state.guild.gold < 30.0 {
        skills.push("resource_scarcity".to_string());
    }
    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead).count();
    if alive <= 2 {
        skills.push("survival".to_string());
    }

    if let Some(d) = delta {
        if d.deaths > 0 { skills.push("retreat_decision".to_string()); }
        if d.gold_diff < -50.0 { skills.push("spending_decision".to_string()); }
        if d.reputation_diff.abs() > 10.0 { skills.push("reputation_management".to_string()); }
        if d.quests_completed > 0 && d.quests_failed > 0 { skills.push("risk_assessment".to_string()); }
    }

    let has_diplo = valid_action_types.iter().any(|t| strategic_bucket(t) == "diplomacy");
    let has_military = valid_action_types.iter().any(|t| strategic_bucket(t) == "military");
    if has_diplo && has_military {
        skills.push("strategic_choice".to_string());
    }

    skills.sort();
    skills.dedup();
    skills
}

/// Compute state complexity score (0.0-1.0): how "interesting" a state is for learning.
fn compute_state_complexity(state: &CampaignState, valid_action_count: usize) -> f32 {
    let mut score = 0.0f32;

    let t = &state.system_trackers;
    let war_count = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .count();
    let crisis_count = state.overworld.active_crises.len();

    score += (war_count as f32 * 0.08).min(0.16);
    score += (crisis_count as f32 * 0.07).min(0.14);
    score += if state.rival_guild.active { 0.05 } else { 0.0 };
    score += (state.active_battles.len() as f32 * 0.06).min(0.12);
    score += if t.active_civil_war_count > 0 { 0.08 } else { 0.0 };
    score += if t.total_intel_gathered > 10.0 { 0.04 } else { 0.0 };
    score += if t.active_caravan_routes > 0 { 0.03 } else { 0.0 };
    score += if t.prisoner_count > 0 { 0.03 } else { 0.0 };

    score += (state.pending_choices.len() as f32 * 0.05).min(0.15);
    score += ((valid_action_count as f32 - 3.0).max(0.0) * 0.01).min(0.15);

    let hostile_factions = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::Hostile)
        .count();
    score += (hostile_factions as f32 * 0.04).min(0.08);

    if state.guild.gold < 30.0 { score += 0.06; }
    if state.guild.supplies < 10.0 { score += 0.04; }

    let injured = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.injury > 30.0)
        .count();
    score += (injured as f32 * 0.03).min(0.09);

    let low_loyalty = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0)
        .count();
    score += (low_loyalty as f32 * 0.04).min(0.08);

    score += (state.overworld.global_threat_level / 100.0 * 0.1).min(0.1);

    score.clamp(0.0, 1.0)
}

/// Apply hindsight labels to a trajectory of samples after the campaign is complete.
pub fn apply_hindsight_labels(
    samples: &mut [BfsSample],
    final_outcome: Option<&str>,
    final_progress: f32,
) {
    let won = match final_outcome {
        Some("Victory") => Some(true),
        Some("Defeat") => Some(false),
        _ => None,
    };

    let n = samples.len();
    for i in 0..n {
        samples[i].trajectory_won = won;
        samples[i].campaign_outcome = Some(final_progress);

        let mut cascade = 0.0f32;
        let mut count = 0;
        let current_value = samples[i].leaf_value;
        for j in (i + 1)..n.min(i + 4) {
            cascade += samples[j].leaf_value - current_value;
            count += 1;
        }
        samples[i].cascade_score = if count > 0 { cascade / count as f32 } else { 0.0 };
    }
}

/// Assign contrastive pair IDs to samples: for each root, pair the best and worst branches.
pub fn assign_contrastive_pairs(samples: &mut [BfsSample], start_pair_id: u32) -> u32 {
    let mut root_groups: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    for (i, s) in samples.iter().enumerate() {
        root_groups.entry((s.seed, s.root_tick)).or_default().push(i);
    }

    let mut pair_id = start_pair_id;
    for (_key, indices) in &root_groups {
        if indices.len() < 2 { continue; }

        let best_idx = indices.iter().copied()
            .max_by(|&a, &b| samples[a].leaf_value.partial_cmp(&samples[b].leaf_value)
                .unwrap_or(std::cmp::Ordering::Equal));
        let worst_idx = indices.iter().copied()
            .min_by(|&a, &b| samples[a].leaf_value.partial_cmp(&samples[b].leaf_value)
                .unwrap_or(std::cmp::Ordering::Equal));

        if let (Some(best), Some(worst)) = (best_idx, worst_idx) {
            if best != worst {
                let spread = samples[best].leaf_value - samples[worst].leaf_value;
                if spread > 0.1 {
                    samples[best].contrastive_pair_id = Some(pair_id);
                    samples[best].is_positive_example = true;
                    samples[worst].contrastive_pair_id = Some(pair_id);
                    samples[worst].is_positive_example = false;
                    pair_id += 1;
                }
            }
        }
    }
    pair_id
}

// ---------------------------------------------------------------------------
// Sample validation (pass 6)
// ---------------------------------------------------------------------------

/// Validate a BFS sample for data quality. Returns `true` if the sample is clean.
pub fn validate_sample(sample: &BfsSample) -> bool {
    for tok in &sample.root_tokens {
        for &v in &tok.features {
            if !v.is_finite() { return false; }
        }
    }
    for tok in &sample.leaf_tokens {
        for &v in &tok.features {
            if !v.is_finite() { return false; }
        }
    }
    if sample.root_tokens.is_empty() || sample.leaf_tokens.is_empty() { return false; }
    if !sample.leaf_value.is_finite() || sample.leaf_value < -10.0 || sample.leaf_value > 100.0 { return false; }
    if !sample.root_value.is_finite() || sample.root_value < -10.0 || sample.root_value > 100.0 { return false; }
    if sample.action_type.is_empty() { return false; }
    if sample.root_tokens.len() == sample.leaf_tokens.len() {
        let all_same = sample.root_tokens.iter().zip(sample.leaf_tokens.iter()).all(|(r, l)| {
            r.type_id == l.type_id
                && r.features.len() == l.features.len()
                && r.features.iter().zip(l.features.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
        });
        if all_same { return false; }
    }
    if sample.leaf_tick < sample.root_tick { return false; }
    true
}

// ---------------------------------------------------------------------------
// Compact serialization (pass 6)
// ---------------------------------------------------------------------------

/// Serialize a BfsSample to compact JSON with fixed 3-decimal precision for floats.
pub fn serialize_sample(sample: &BfsSample) -> String {
    if let Ok(json) = serde_json::to_string(sample) {
        compact_floats(&json)
    } else {
        String::new()
    }
}

/// Reduce float precision in JSON string to 3 decimal places.
fn compact_floats(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if (bytes[i].is_ascii_digit() || (bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()))
            && i > 0
            && (bytes[i - 1] == b'[' || bytes[i - 1] == b',' || bytes[i - 1] == b':')
        {
            let start = i;
            if bytes[i] == b'-' { i += 1; }
            while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                let frac_start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
                let frac_len = i - frac_start;
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'e' || bytes[i] == b'E' || bytes[i] == b'+' || bytes[i] == b'-') { i += 1; }
                    out.push_str(&json[start..i]);
                } else if frac_len > 3 {
                    if let Ok(val) = json[start..i].parse::<f64>() {
                        use std::fmt::Write;
                        let _ = write!(out, "{:.3}", val);
                    } else {
                        out.push_str(&json[start..i]);
                    }
                } else {
                    out.push_str(&json[start..i]);
                }
            } else {
                out.push_str(&json[start..i]);
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dataset statistics (pass 6)
// ---------------------------------------------------------------------------

/// Comprehensive statistics tracked across all BFS samples in a run.
#[derive(Debug, Default)]
pub struct BfsDatasetStats {
    pub total_generated: usize,
    pub valid_samples: usize,
    pub rejected_samples: usize,
    pub deduped_samples: usize,
    pub balanced_removed: usize,
    pub terminal_boosted: usize,
    value_sum: f64,
    value_sq_sum: f64,
    value_count: usize,
    value_min: f32,
    value_max: f32,
    spread_sum: f64,
    spread_count: usize,
    dominant_roots: usize,
    pub action_histogram: HashMap<String, usize>,
    pub phase_histogram: HashMap<String, usize>,
    branch_len_sum: u64,
    branch_len_count: usize,
    pub tier_histogram: [usize; 5], // index 1-4 used (pass 9)
}

impl BfsDatasetStats {
    pub fn new() -> Self {
        Self { value_min: f32::MAX, value_max: f32::MIN, ..Default::default() }
    }

    pub fn record(&mut self, sample: &BfsSample) {
        self.valid_samples += 1;
        let v = sample.leaf_value as f64;
        self.value_sum += v;
        self.value_sq_sum += v * v;
        self.value_count += 1;
        if sample.leaf_value < self.value_min { self.value_min = sample.leaf_value; }
        if sample.leaf_value > self.value_max { self.value_max = sample.leaf_value; }
        *self.action_histogram.entry(sample.action_type.clone()).or_insert(0) += 1;
        *self.phase_histogram.entry(sample.phase_tag.clone()).or_insert(0) += 1;
        let branch_len = sample.leaf_tick.saturating_sub(sample.root_tick);
        self.branch_len_sum += branch_len;
        self.branch_len_count += 1;
        let tier = (sample.curriculum_tier as usize).min(4);
        if tier >= 1 { self.tier_histogram[tier] += 1; }
    }

    pub fn record_spread(&mut self, spread: f32, is_dominant: bool) {
        self.spread_sum += spread as f64;
        self.spread_count += 1;
        if is_dominant { self.dominant_roots += 1; }
    }

    fn value_mean(&self) -> f64 {
        if self.value_count == 0 { 0.0 } else { self.value_sum / self.value_count as f64 }
    }

    fn value_std(&self) -> f64 {
        if self.value_count < 2 { return 0.0; }
        let mean = self.value_mean();
        (self.value_sq_sum / self.value_count as f64 - mean * mean).max(0.0).sqrt()
    }

    fn mean_spread(&self) -> f64 {
        if self.spread_count == 0 { 0.0 } else { self.spread_sum / self.spread_count as f64 }
    }

    fn mean_branch_len(&self) -> f64 {
        if self.branch_len_count == 0 { 0.0 } else { self.branch_len_sum as f64 / self.branch_len_count as f64 }
    }

    pub fn print_report(&self) {
        eprintln!("\n=== BFS Dataset Quality Report ===");
        eprintln!("Samples: {} generated, {} valid, {} rejected ({:.1}% rejection rate)",
            self.total_generated, self.valid_samples, self.rejected_samples,
            if self.total_generated > 0 { self.rejected_samples as f64 / self.total_generated as f64 * 100.0 } else { 0.0 });
        eprintln!("Deduplication: {} removed", self.deduped_samples);
        eprintln!("Balancing: {} removed, {} terminal boosted", self.balanced_removed, self.terminal_boosted);
        eprintln!("\nValue distribution: mean={:.3}, std={:.3}, min={:.3}, max={:.3}",
            self.value_mean(), self.value_std(),
            if self.value_min == f32::MAX { 0.0 } else { self.value_min as f64 },
            if self.value_max == f32::MIN { 0.0 } else { self.value_max as f64 });
        eprintln!("Mean spread per root: {:.3}", self.mean_spread());
        eprintln!("Dominant roots (best > 1 std): {}/{}", self.dominant_roots, self.spread_count);
        eprintln!("Mean branch length: {:.0} ticks", self.mean_branch_len());
        let total = self.valid_samples.max(1) as f64;
        let mut actions: Vec<_> = self.action_histogram.iter().collect();
        actions.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!("\nAction type distribution:");
        for (name, count) in &actions {
            let pct = **count as f64 / total * 100.0;
            let bar_len = (pct / 2.0).round() as usize;
            let bar: String = "#".repeat(bar_len.min(30));
            eprintln!("  {:<30} {:>5} ({:>5.1}%) {}", name, count, pct, bar);
        }
        eprintln!("\nPhase coverage:");
        for phase in &["early", "mid", "late"] {
            let count = self.phase_histogram.get(*phase).copied().unwrap_or(0);
            let pct = count as f64 / total * 100.0;
            eprintln!("  {:<8} {:>5} ({:>5.1}%)", phase, count, pct);
            if pct < 10.0 && self.valid_samples > 100 {
                eprintln!("  WARNING: phase '{}' has <10% coverage", phase);
            }
        }
        let tier_labels = ["", "beginner", "intermediate", "advanced", "expert"];
        eprintln!("\nCurriculum tier distribution:");
        for tier in 1..=4 {
            let count = self.tier_histogram[tier];
            let pct = count as f64 / total * 100.0;
            eprintln!("  Tier {} ({:<12}) {:>5} ({:>5.1}%)", tier, tier_labels[tier], count, pct);
        }
    }
}

// ---------------------------------------------------------------------------
// Deduplication (pass 6)
// ---------------------------------------------------------------------------

fn dedup_key(sample: &BfsSample) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for tok in &sample.root_tokens {
        hash ^= tok.type_id as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        for &f in &tok.features {
            let rounded = (f * 10.0).round() as i32;
            hash ^= rounded as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    for b in sample.action_type.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn deduplicate_samples(samples: &mut Vec<BfsSample>) -> usize {
    let mut best: HashMap<u64, usize> = HashMap::new();
    let mut keep = vec![false; samples.len()];
    let mut removed = 0usize;
    for (i, sample) in samples.iter().enumerate() {
        let key = dedup_key(sample);
        let spread = (sample.leaf_value - sample.root_value).abs();
        if let Some(&prev_idx) = best.get(&key) {
            let prev_spread = (samples[prev_idx].leaf_value - samples[prev_idx].root_value).abs();
            if spread > prev_spread {
                keep[prev_idx] = false;
                keep[i] = true;
                best.insert(key, i);
            }
            removed += 1;
        } else {
            best.insert(key, i);
            keep[i] = true;
        }
    }
    let mut write = 0;
    for read in 0..samples.len() {
        if keep[read] { samples.swap(write, read); write += 1; }
    }
    samples.truncate(write);
    removed
}

// ---------------------------------------------------------------------------
// Balanced sampling (pass 6)
// ---------------------------------------------------------------------------

fn balance_dataset(samples: &mut Vec<BfsSample>) -> (usize, usize) {
    if samples.is_empty() { return (0, 0); }
    let total = samples.len();
    let max_per_action = total / 5;
    let mut action_counts: HashMap<String, usize> = HashMap::new();
    for s in samples.iter() { *action_counts.entry(s.action_type.clone()).or_insert(0) += 1; }
    let over_represented: HashSet<String> = action_counts.iter()
        .filter(|(_, &count)| count > max_per_action)
        .map(|(name, _)| name.clone()).collect();
    let mut removed = 0usize;
    if !over_represented.is_empty() {
        let mut kept_counts: HashMap<String, usize> = HashMap::new();
        let mut keep = vec![true; samples.len()];
        for (i, s) in samples.iter().enumerate() {
            if over_represented.contains(&s.action_type) {
                let count = kept_counts.entry(s.action_type.clone()).or_insert(0);
                if *count >= max_per_action { keep[i] = false; removed += 1; } else { *count += 1; }
            }
        }
        let mut write = 0;
        for read in 0..samples.len() {
            if keep[read] { samples.swap(write, read); write += 1; }
        }
        samples.truncate(write);
    }
    let terminal_count = samples.iter().filter(|s| s.leaf_outcome.is_some()).count();
    let threshold = samples.len() / 20;
    let boosted = if terminal_count < threshold && terminal_count > 0 { terminal_count } else { 0 };
    (removed, boosted)
}

/// Print a dataset balance report. (pass 6)
pub fn dataset_balance_report(samples: &[BfsSample]) {
    if samples.is_empty() { eprintln!("Dataset is empty, no balance report."); return; }
    let total = samples.len();
    eprintln!("\n=== Dataset Balance Report ===");
    eprintln!("Total samples: {}", total);
    let mut action_counts: HashMap<&str, usize> = HashMap::new();
    for s in samples { *action_counts.entry(&s.action_type).or_insert(0) += 1; }
    let mut actions: Vec<_> = action_counts.iter().collect();
    actions.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("\nAction types:");
    for (name, count) in &actions {
        let pct = **count as f64 / total as f64 * 100.0;
        eprintln!("  {:<30} {:>5} ({:>5.1}%){}", name, count, pct,
            if pct > 20.0 { " [OVER-REPRESENTED]" } else { "" });
    }
    let mut phase_counts: HashMap<&str, usize> = HashMap::new();
    for s in samples { *phase_counts.entry(&s.phase_tag).or_insert(0) += 1; }
    eprintln!("\nPhase distribution:");
    for phase in &["early", "mid", "late"] {
        let count = phase_counts.get(phase).copied().unwrap_or(0);
        let pct = count as f64 / total as f64 * 100.0;
        eprintln!("  {:<8} {:>5} ({:>5.1}%){}", phase, count, pct,
            if pct < 10.0 { " [UNDER-REPRESENTED]" } else { "" });
    }
    let victories = samples.iter().filter(|s| s.leaf_outcome.as_deref() == Some("Victory")).count();
    let defeats = samples.iter().filter(|s| s.leaf_outcome.as_deref() == Some("Defeat")).count();
    let terminal = victories + defeats;
    eprintln!("\nTerminal states: {} ({:.1}%) — {} victories, {} defeats",
        terminal, terminal as f64 / total as f64 * 100.0, victories, defeats);
}

// ---------------------------------------------------------------------------
// BFS exploration
// ---------------------------------------------------------------------------

/// Run the full BFS exploration pipeline.
pub fn run_bfs_exploration(config: &BfsConfig) -> BfsStats {
    let threads = if config.threads == 0 {
        rayon::current_num_threads()
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build rayon pool");

    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let writer = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&config.output_path).expect("Failed to create output file"),
    ));

    // Write header line with config snapshot for reproducibility (pass 6)
    {
        use std::io::Write;
        let mut w = writer.lock().unwrap();
        if let Ok(header) = serde_json::to_string(&config.snapshot()) {
            writeln!(w, "{}", header).ok();
        }
    }

    let llm_store = config.llm_config.as_ref().map(|cfg| {
        std::sync::Arc::new(super::llm::create_store(cfg))
    });

    let mut stats = BfsStats::default();
    let mut dataset_stats = BfsDatasetStats::new();
    let t0 = std::time::Instant::now();

    let max_waves_str = if config.max_waves == 0 {
        "unlimited (until completion)".to_string()
    } else {
        config.max_waves.to_string()
    };
    eprintln!("=== BFS State-Space Exploration (dataset v{}) ===", BFS_DATASET_VERSION);
    eprintln!("Max waves: {}, Clusters/wave: {}, Ticks/branch: {}",
        max_waves_str, config.clusters_per_wave, config.ticks_per_branch);
    eprintln!("Initial roots: {}, Threads: {}", config.initial_roots, threads);

    let mut roots = generate_initial_roots(config, &llm_store);
    eprintln!("Generated {} initial root states", roots.len());
    stats.initial_roots = roots.len();

    let mut wave: u32 = 0;
    let mut total_terminal = 0u64;
    let mut total_victories = 0u64;
    let mut total_defeats = 0u64;

    let mut global_action_types: HashSet<String> = HashSet::new();
    let mut phase_counts: HashMap<String, usize> = HashMap::new();
    let mut all_accumulated_samples: Vec<BfsSample> = Vec::new();

    // Shared state for UCB and novelty across waves (pass 5)
    let shared = Mutex::new(BfsSharedState::new());

    loop {
        if config.max_waves > 0 && wave >= config.max_waves {
            eprintln!("  Reached max waves ({}), stopping", config.max_waves);
            break;
        }
        if roots.is_empty() {
            eprintln!("  No more roots to expand (all terminal), stopping");
            break;
        }

        let wave_t0 = std::time::Instant::now();
        let num_roots = roots.len();

        let (samples, leaves): (Vec<Vec<BfsSample>>, Vec<Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)>>) = pool.install(|| {
            use rayon::prelude::*;
            roots
                .par_iter()
                .map(|root| expand_root(root, config, wave, &shared))
                .unzip()
        });

        let raw_samples: Vec<BfsSample> = samples.into_iter().flatten().collect();
        let all_leaves: Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)> =
            leaves.into_iter().flatten().collect();

        // Validate samples (pass 6)
        let generated_count = raw_samples.len();
        dataset_stats.total_generated += generated_count;
        let valid_samples: Vec<BfsSample> = raw_samples.into_iter()
            .filter(|s| validate_sample(s))
            .collect();
        let rejected = generated_count - valid_samples.len();
        dataset_stats.rejected_samples += rejected;
        if rejected > 0 {
            eprintln!("    Rejected {} invalid samples in wave {}", rejected, wave);
        }

        let wave_terminal: usize = valid_samples.iter().filter(|s| s.leaf_outcome.is_some()).count();
        let wave_victories: usize = valid_samples.iter()
            .filter(|s| s.leaf_outcome.as_deref() == Some("Victory")).count();
        let wave_defeats: usize = valid_samples.iter()
            .filter(|s| s.leaf_outcome.as_deref() == Some("Defeat")).count();
        total_terminal += wave_terminal as u64;
        total_victories += wave_victories as u64;
        total_defeats += wave_defeats as u64;

        for s in &valid_samples {
            global_action_types.insert(s.action_type.clone());
            *phase_counts.entry(s.phase_tag.clone()).or_insert(0) += 1;
            dataset_stats.record(s);
        }

        // Compute spread stats (pass 6)
        let mut per_root: HashMap<u64, (f32, f32)> = HashMap::new();
        for s in &valid_samples {
            let key = s.root_tick ^ s.seed;
            let entry = per_root.entry(key).or_insert((f32::MAX, f32::MIN));
            entry.0 = entry.0.min(s.leaf_value);
            entry.1 = entry.1.max(s.leaf_value);
        }
        let value_std = dataset_stats.value_std() as f32;
        let mut root_spreads: Vec<f32> = Vec::new();
        for (_, (lo, hi)) in &per_root {
            let spread = hi - lo;
            root_spreads.push(spread);
            let is_dominant = value_std > 0.0 && spread > value_std;
            dataset_stats.record_spread(spread, is_dominant);
        }
        let mean_spread = if root_spreads.is_empty() { 0.0 }
            else { root_spreads.iter().sum::<f32>() / root_spreads.len() as f32 };

        all_accumulated_samples.extend(valid_samples.iter().cloned());

        stats.total_samples += valid_samples.len();
        stats.total_leaves += all_leaves.len();

        if all_leaves.is_empty() {
            let unique_action_types: HashSet<&str> = valid_samples
                .iter()
                .map(|s| s.action_type.as_str())
                .collect();
            eprintln!(
                "  Wave {}: {} roots -> {} samples (all terminal: {}V/{}D), {} action types, {:.1}s",
                wave, num_roots, valid_samples.len(),
                wave_victories, wave_defeats,
                unique_action_types.len(),
                wave_t0.elapsed().as_secs_f64(),
            );
            eprintln!("  All branches reached terminal states, stopping");
            break;
        }

        let leaf_states_features: Vec<(CampaignState, Vec<f32>)> = all_leaves
            .iter()
            .map(|(s, f, _, _, _)| (s.clone(), f.clone()))
            .collect();
        let leaf_values: Vec<f32> = all_leaves.iter().map(|(_, _, v, _, _)| *v).collect();

        let selected_indices =
            cluster_and_select_diverse(&leaf_states_features, config.clusters_per_wave, &leaf_values);

        roots = selected_indices
            .iter()
            .filter_map(|&idx| {
                all_leaves.get(idx).map(|(state, _, _, seed, diff)| RootState {
                    state: state.clone(),
                    seed: *seed,
                    wave: wave + 1,
                    difficulty: *diff,
                })
            })
            .collect();

        let unique_action_types: HashSet<&str> = valid_samples
            .iter()
            .map(|s| s.action_type.as_str())
            .collect();

        eprintln!(
            "  Wave {}: {} roots -> {} samples ({} terminal: {}V/{}D), {} live -> {} next ({} types, spread={:.3}, {:.1}s)",
            wave, num_roots,
            valid_samples.len(), wave_terminal, wave_victories, wave_defeats,
            all_leaves.len(),
            roots.len(),
            unique_action_types.len(),
            mean_spread,
            wave_t0.elapsed().as_secs_f64(),
        );

        wave += 1;
    }

    // Post-processing: deduplication (pass 6)
    let pre_dedup = all_accumulated_samples.len();
    let dedup_removed = deduplicate_samples(&mut all_accumulated_samples);
    dataset_stats.deduped_samples = dedup_removed;
    if dedup_removed > 0 {
        eprintln!("Deduplication: removed {} near-duplicate samples ({} -> {})",
            dedup_removed, pre_dedup, all_accumulated_samples.len());
    }

    // Post-processing: balanced sampling (pass 6)
    let (balance_removed, terminal_boosted) = balance_dataset(&mut all_accumulated_samples);
    dataset_stats.balanced_removed = balance_removed;
    dataset_stats.terminal_boosted = terminal_boosted;
    if balance_removed > 0 {
        eprintln!("Balancing: removed {} overrepresented samples", balance_removed);
    }

    // Post-processing: assign contrastive pairs (pass 9)
    let pair_count = assign_contrastive_pairs(&mut all_accumulated_samples, 0);
    if pair_count > 0 {
        eprintln!("Contrastive pairs: {} pairs assigned", pair_count);
    }

    // Post-processing: hindsight labels by trajectory (pass 9)
    {
        let mut by_seed: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, s) in all_accumulated_samples.iter().enumerate() {
            by_seed.entry(s.seed).or_default().push(i);
        }
        let mut hindsight_count = 0usize;
        let trajectory_count = by_seed.len();
        for (_seed, mut indices) in by_seed {
            indices.sort_by_key(|&i| all_accumulated_samples[i].root_tick);
            let last_idx = *indices.last().unwrap();
            let final_outcome = all_accumulated_samples[last_idx].leaf_outcome.clone();
            let final_progress = all_accumulated_samples[last_idx].leaf_value / 5.0;

            let won = match final_outcome.as_deref() {
                Some("Victory") => Some(true),
                Some("Defeat") => Some(false),
                _ => None,
            };

            let n = indices.len();
            for (pos, &idx) in indices.iter().enumerate() {
                all_accumulated_samples[idx].trajectory_won = won;
                all_accumulated_samples[idx].campaign_outcome = Some(final_progress);

                let mut cascade = 0.0f32;
                let mut count = 0;
                let current_value = all_accumulated_samples[idx].leaf_value;
                for j in (pos + 1)..n.min(pos + 4) {
                    cascade += all_accumulated_samples[indices[j]].leaf_value - current_value;
                    count += 1;
                }
                all_accumulated_samples[idx].cascade_score =
                    if count > 0 { cascade / count as f32 } else { 0.0 };
                hindsight_count += 1;
            }
        }
        eprintln!("Hindsight labels: applied to {} samples across {} trajectories",
            hindsight_count, trajectory_count);
    }

    // Write all validated, deduped, balanced samples with compact serialization (pass 6)
    {
        use std::io::Write;
        let mut w = writer.lock().unwrap();
        for sample in &all_accumulated_samples {
            let json = serialize_sample(sample);
            if !json.is_empty() {
                writeln!(w, "{}", json).ok();
            }
        }
    }

    stats.total_samples = all_accumulated_samples.len();

    let elapsed = t0.elapsed().as_secs_f64();
    stats.elapsed_secs = elapsed;
    stats.total_terminal = total_terminal as usize;
    stats.total_victories = total_victories as usize;
    stats.total_defeats = total_defeats as usize;
    stats.waves_completed = wave;
    stats.unique_action_types = global_action_types.len();
    stats.phase_coverage = phase_counts;

    eprintln!("\n=== BFS Exploration Complete ===");
    eprintln!("Waves: {}", wave);
    eprintln!("Total samples (after QA): {}", stats.total_samples);
    eprintln!("Terminal outcomes: {} ({} victories, {} defeats)",
        total_terminal, total_victories, total_defeats);
    eprintln!("Total leaves explored: {}", stats.total_leaves);
    eprintln!("Unique action types: {}", stats.unique_action_types);
    eprintln!("Phase coverage: {:?}", stats.phase_coverage);
    eprintln!("Rate: {:.0} samples/s", stats.total_samples as f64 / elapsed);
    eprintln!("Output: {} ({:.1} MB)",
        config.output_path,
        std::fs::metadata(&config.output_path).map(|m| m.len() as f64 / 1_000_000.0).unwrap_or(0.0));

    if let Some(ref store) = llm_store {
        let (total, hits, valid) = store.stats();
        eprintln!("LLM: {} requests, {} cache hits, {} valid generations", total, hits, valid);
    }

    dataset_stats.print_report();
    dataset_balance_report(&all_accumulated_samples);

    stats
}

#[derive(Debug, Default)]
pub struct BfsStats {
    pub initial_roots: usize,
    pub total_samples: usize,
    pub total_leaves: usize,
    pub total_terminal: usize,
    pub total_victories: usize,
    pub total_defeats: usize,
    pub waves_completed: u32,
    pub elapsed_secs: f64,
    pub unique_action_types: usize,
    pub phase_coverage: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Diverse policy for human-like root generation
// ---------------------------------------------------------------------------

/// Policy that simulates varied human playstyles for root generation.
fn diverse_root_policy(state: &CampaignState, rng: &mut u64) -> Option<CampaignAction> {
    if state.phase != CampaignPhase::Playing {
        if let Some(choice) = state.pending_choices.first() {
            xorshift(rng);
            let idx = (*rng as usize) % choice.options.len().max(1);
            return Some(CampaignAction::RespondToChoice {
                choice_id: choice.id,
                option_index: idx,
            });
        }
        if state.phase == CampaignPhase::ChoosingStartingPackage {
            if !state.available_starting_choices.is_empty() {
                xorshift(rng);
                let idx = (*rng as usize) % state.available_starting_choices.len();
                return Some(CampaignAction::ChooseStartingPackage {
                    choice: state.available_starting_choices[idx].clone(),
                });
            }
        }
        return None;
    }

    if let Some(choice) = state.pending_choices.first() {
        xorshift(rng);
        let idx = (*rng as usize) % choice.options.len().max(1);
        return Some(CampaignAction::RespondToChoice {
            choice_id: choice.id,
            option_index: idx,
        });
    }

    xorshift(rng);
    let roll = *rng % 100;

    if roll < 8 {
        let has_stressed = state.adventurers.iter()
            .any(|a| a.status != AdventurerStatus::Dead && (a.stress > 50.0 || a.injury > 30.0));
        if has_stressed || !state.pending_progression.is_empty() {
            return Some(CampaignAction::Rest);
        }
    }

    if roll < 13 && roll >= 8 && state.guild.gold > state.config.economy.training_cost {
        let idle: Vec<u32> = state.adventurers.iter()
            .filter(|a| a.status == AdventurerStatus::Idle)
            .map(|a| a.id).collect();
        if !idle.is_empty() {
            xorshift(rng);
            let idx = (*rng as usize) % idle.len();
            let training_types = [TrainingType::Combat, TrainingType::Exploration,
                                  TrainingType::Leadership, TrainingType::Survival];
            xorshift(rng);
            let ti = (*rng as usize) % training_types.len();
            return Some(CampaignAction::TrainAdventurer {
                adventurer_id: idle[idx],
                training_type: training_types[ti],
            });
        }
    }

    if roll < 20 && roll >= 13 && !state.factions.is_empty() {
        xorshift(rng);
        let fi = (*rng as usize) % state.factions.len();
        let faction = &state.factions[fi];
        let action_type = match faction.diplomatic_stance {
            DiplomaticStance::AtWar => DiplomacyActionType::ProposeCeasefire,
            DiplomaticStance::Hostile => {
                xorshift(rng);
                if *rng % 2 == 0 { DiplomacyActionType::ImproveRelations }
                else { DiplomacyActionType::Threaten }
            }
            DiplomaticStance::Neutral => {
                xorshift(rng);
                if *rng % 2 == 0 { DiplomacyActionType::ImproveRelations }
                else { DiplomacyActionType::TradeAgreement }
            }
            DiplomaticStance::Friendly => {
                xorshift(rng);
                if *rng % 3 == 0 && faction.relationship_to_guild > 60.0 && !faction.coalition_member {
                    return Some(CampaignAction::ProposeCoalition { faction_id: faction.id });
                }
                DiplomacyActionType::TradeAgreement
            }
            DiplomaticStance::Coalition => {
                if faction.military_strength > 20.0 {
                    return Some(CampaignAction::RequestCoalitionAid { faction_id: faction.id });
                }
                DiplomacyActionType::TradeAgreement
            }
        };
        return Some(CampaignAction::DiplomaticAction {
            faction_id: faction.id,
            action_type,
        });
    }

    if roll < 24 && roll >= 20 && state.guild.gold >= state.config.economy.scout_cost {
        let unscouted: Vec<usize> = state.overworld.locations.iter()
            .filter(|l| !l.scouted).map(|l| l.id).collect();
        if !unscouted.is_empty() {
            xorshift(rng);
            let idx = (*rng as usize) % unscouted.len();
            return Some(CampaignAction::HireScout { location_id: unscouted[idx] });
        }
    }

    if roll < 27 && roll >= 24 {
        xorshift(rng);
        let priorities = [SpendPriority::Balanced, SpendPriority::SaveForEmergencies,
                          SpendPriority::InvestInGrowth, SpendPriority::MilitaryFocus];
        let pi = (*rng as usize) % priorities.len();
        return Some(CampaignAction::SetSpendPriority { priority: priorities[pi] });
    }

    if roll < 30 && roll >= 27 && !state.guild.inventory.is_empty() {
        let idle: Vec<u32> = state.adventurers.iter()
            .filter(|a| a.status == AdventurerStatus::Idle).map(|a| a.id).collect();
        if !idle.is_empty() {
            xorshift(rng);
            let ai = (*rng as usize) % idle.len();
            xorshift(rng);
            let ii = (*rng as usize) % state.guild.inventory.len();
            return Some(CampaignAction::EquipGear {
                adventurer_id: idle[ai],
                item_id: state.guild.inventory[ii].id,
            });
        }
    }

    if roll < 33 && roll >= 30 && state.guild.gold >= state.config.economy.supply_cost_per_unit * 10.0 {
        let field_parties: Vec<u32> = state.parties.iter()
            .filter(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting))
            .map(|p| p.id).collect();
        if !field_parties.is_empty() {
            xorshift(rng);
            let pi = (*rng as usize) % field_parties.len();
            return Some(CampaignAction::PurchaseSupplies { party_id: field_parties[pi], amount: 10.0 });
        }
    }

    if roll < 35 && roll >= 33 && state.guild.gold >= state.config.economy.runner_cost {
        let field_parties: Vec<u32> = state.parties.iter()
            .filter(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting))
            .map(|p| p.id).collect();
        if !field_parties.is_empty() {
            xorshift(rng);
            let pi = (*rng as usize) % field_parties.len();
            xorshift(rng);
            let payload = if *rng % 2 == 0 { RunnerPayload::Supplies(20.0) } else { RunnerPayload::Message };
            return Some(CampaignAction::SendRunner { party_id: field_parties[pi], payload });
        }
    }

    if roll < 37 && roll >= 35 {
        let ready_abilities: Vec<&UnlockInstance> = state.unlocks.iter()
            .filter(|u| u.active && !u.properties.is_passive
                && u.cooldown_remaining_ms == 0
                && state.guild.gold >= u.properties.resource_cost)
            .collect();
        if !ready_abilities.is_empty() {
            xorshift(rng);
            let ai = (*rng as usize) % ready_abilities.len();
            let unlock = ready_abilities[ai];
            let target = match unlock.properties.target_type {
                TargetType::GuildSelf => Some(AbilityTarget::Party(0)),
                TargetType::Adventurer => {
                    let live: Vec<u32> = state.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead).map(|a| a.id).collect();
                    if live.is_empty() { None } else { xorshift(rng); Some(AbilityTarget::Adventurer(live[(*rng as usize) % live.len()])) }
                }
                TargetType::Party => {
                    if state.parties.is_empty() { None } else { xorshift(rng); Some(AbilityTarget::Party(state.parties[(*rng as usize) % state.parties.len()].id)) }
                }
                TargetType::Quest => {
                    if state.active_quests.is_empty() { None } else { xorshift(rng); Some(AbilityTarget::Quest(state.active_quests[(*rng as usize) % state.active_quests.len()].id)) }
                }
                TargetType::Area => {
                    if state.overworld.locations.is_empty() { None } else { xorshift(rng); Some(AbilityTarget::Location(state.overworld.locations[(*rng as usize) % state.overworld.locations.len()].id)) }
                }
                TargetType::Faction => {
                    if state.factions.is_empty() { None } else { xorshift(rng); Some(AbilityTarget::Faction(state.factions[(*rng as usize) % state.factions.len()].id)) }
                }
            };
            if let Some(t) = target {
                return Some(CampaignAction::UseAbility { unlock_id: unlock.id, target: t });
            }
        }
    }

    if roll < 40 && roll >= 37 {
        if let Some(quest) = state.request_board.first() {
            return Some(CampaignAction::DeclineQuest { request_id: quest.id });
        }
    }

    if roll < 42 && roll >= 40 && state.guild.gold >= state.config.economy.mercenary_cost {
        let eligible: Vec<u32> = state.active_quests.iter()
            .filter(|q| matches!(q.status, ActiveQuestStatus::InProgress | ActiveQuestStatus::InCombat))
            .map(|q| q.id).collect();
        if !eligible.is_empty() {
            xorshift(rng);
            let qi = (*rng as usize) % eligible.len();
            return Some(CampaignAction::HireMercenary { quest_id: eligible[qi] });
        }
    }

    if roll < 43 && roll >= 42 {
        for battle in &state.active_battles {
            if battle.status == BattleStatus::Active && battle.party_health_ratio < 0.4 {
                let can_afford = state.guild.gold >= state.config.economy.rescue_bribe_cost;
                let has_free = state.npc_relationships.iter()
                    .any(|r| r.rescue_available && r.relationship_score > 50.0);
                if can_afford || has_free {
                    return Some(CampaignAction::CallRescue { battle_id: battle.id });
                }
            }
        }
    }

    heuristic_policy(state)
}

/// Inline xorshift64 step.
fn xorshift(rng: &mut u64) {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
}

/// Pick an action for strategic rollout based on the rollout mode. (pass 2)
fn strategic_rollout_action(
    state: &CampaignState,
    rng: &mut u64,
    mode: RolloutMode,
    initial_bucket: &str,
) -> Option<CampaignAction> {
    if mode == RolloutMode::Mixed { return diverse_root_policy(state, rng); }
    if state.phase != CampaignPhase::Playing { return diverse_root_policy(state, rng); }
    if !state.pending_choices.is_empty() { return diverse_root_policy(state, rng); }

    xorshift(rng);
    if *rng % 100 < 60 {
        let valid = state.valid_actions();
        let preferred: Vec<&CampaignAction> = valid.iter()
            .filter(|a| {
                let atype = action_type_name(a);
                let bucket = strategic_bucket(&atype);
                match mode {
                    RolloutMode::Consistent => bucket == initial_bucket,
                    _ => mode.prefers_bucket(bucket),
                }
            })
            .collect();
        if !preferred.is_empty() {
            xorshift(rng);
            let idx = (*rng as usize) % preferred.len();
            return Some(preferred[idx].clone());
        }
    }
    diverse_root_policy(state, rng)
}

/// Compute the state delta between root and leaf states. (pass 2)
fn compute_state_delta(root: &CampaignState, leaf: &CampaignState) -> StateDelta {
    let root_alive = root.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead).count() as i32;
    let leaf_alive = leaf.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead).count() as i32;
    let root_completed = root.completed_quests.len();
    let leaf_completed = leaf.completed_quests.len();
    let new_completions = leaf_completed.saturating_sub(root_completed);
    let quests_completed = leaf.completed_quests.iter()
        .skip(root_completed)
        .filter(|q| q.result == QuestResult::Victory)
        .count() as u32;
    let quests_failed = new_completions as u32 - quests_completed;
    let root_max_level: u32 = root.adventurers.iter().map(|a| a.level).sum();
    let leaf_max_level: u32 = leaf.adventurers.iter().map(|a| a.level).sum();
    let level_ups = leaf_max_level.saturating_sub(root_max_level);
    let root_dead = root.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Dead).count() as u32;
    let leaf_dead = leaf.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Dead).count() as u32;
    StateDelta {
        gold_diff: leaf.guild.gold - root.guild.gold,
        reputation_diff: leaf.guild.reputation - root.guild.reputation,
        adventurer_count_diff: leaf_alive - root_alive,
        quests_completed,
        quests_failed,
        level_ups,
        deaths: leaf_dead.saturating_sub(root_dead),
        progress_diff: leaf.overworld.campaign_progress - root.overworld.campaign_progress,
        threat_diff: leaf.overworld.global_threat_level - root.overworld.global_threat_level,
    }
}

// ---------------------------------------------------------------------------
// Root generation
// ---------------------------------------------------------------------------

fn generate_initial_roots(
    config: &BfsConfig,
    llm_store: &Option<std::sync::Arc<super::llm::ContentStore>>,
) -> Vec<RootState> {
    let mut roots = Vec::new();
    let campaigns_needed = (config.initial_roots / 3).max(10);
    let mut early_count = 0usize;
    let mut mid_count = 0usize;
    let mut late_count = 0usize;
    let target_per_phase = config.initial_roots / 3;

    for i in 0..campaigns_needed as u64 {
        let seed = config.base_seed.wrapping_add(i * 7919);
        let difficulty = match i as usize % 20 {
            0..=2 => Difficulty::Easy,
            3..=7 => Difficulty::Normal,
            8..=14 => Difficulty::Hard,
            _ => Difficulty::Brutal,
        };
        let campaign_config = CampaignConfig::with_difficulty(difficulty);
        let mut state = CampaignState::with_config(seed, campaign_config);
        state.llm_config = config.llm_config.clone();
        state.llm_store = llm_store.clone();
        state.vae_model = config.vae_model.clone();
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        let extended_max = config.trajectory_max_ticks;
        for _tick in 0..extended_max {
            let action = diverse_root_policy(&state, &mut rng);
            let result = step_campaign(&mut state, action);
            let alive = state.adventurers.iter()
                .filter(|a| a.status != AdventurerStatus::Dead).count();
            let is_viable = alive >= 2 && state.guild.gold > 5.0;
            let phase = phase_tag(state.tick, state.overworld.campaign_progress);
            let phase_has_room = match phase.as_str() {
                "early" => early_count < target_per_phase * 2,
                "mid" => mid_count < target_per_phase * 2,
                "late" => late_count < target_per_phase * 2,
                _ => true,
            };
            let has_quest_board = !state.request_board.is_empty();
            let has_active_quest = !state.active_quests.is_empty();
            let has_battle = !state.active_battles.is_empty();
            let has_injured = state.adventurers.iter().any(|a| a.injury > 30.0);
            let has_crisis = !state.overworld.active_crises.is_empty();
            let has_pending_choice = !state.pending_choices.is_empty();
            let has_rival = state.rival_guild.active;
            let has_war = state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::AtWar);
            let has_low_loyalty = state.adventurers.iter()
                .any(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0);
            let has_tension = is_viable && (
                has_quest_board || has_active_quest || has_battle
                || has_injured || has_crisis || has_pending_choice
                || has_rival || has_war || has_low_loyalty
                || state.guild.gold < 150.0
                || state.overworld.global_threat_level > 30.0
                || state.completed_quests.len() >= 3
            );
            let is_crisis_state = has_crisis || has_war || has_battle
                || (state.overworld.global_threat_level > 60.0);
            let on_cadence = state.tick % config.root_sample_interval == 0;
            let crisis_sample = is_crisis_state && state.tick % (config.root_sample_interval / 3).max(1) == 0;
            let min_tick = config.root_sample_interval;
            if state.tick >= min_tick && has_tension && phase_has_room
                && (on_cadence || crisis_sample)
            {
                roots.push(RootState { state: state.clone(), seed, wave: 0, difficulty });
                match phase.as_str() {
                    "early" => early_count += 1,
                    "mid" => mid_count += 1,
                    "late" => late_count += 1,
                    _ => {}
                }
            }
            if result.outcome.is_some() { break; }
        }
    }

    eprintln!("  Root phase distribution: early={}, mid={}, late={}", early_count, mid_count, late_count);

    if roots.len() > config.initial_roots * 2 {
        let features: Vec<(CampaignState, Vec<f32>)> = roots
            .iter().map(|r| (r.state.clone(), state_features(&r.state))).collect();
        let medians = cluster_and_select_medians(&features, config.initial_roots);
        roots = medians.into_iter().filter_map(|idx| roots.get(idx).cloned()).collect();
    }

    roots
}

// ---------------------------------------------------------------------------
// Root expansion
// ---------------------------------------------------------------------------

/// Find the most passive/safe action for use as the counterfactual baseline. (pass 5)
fn find_baseline_action(grouped: &[(String, CampaignAction)]) -> Option<&(String, CampaignAction)> {
    if let Some(a) = grouped.iter().find(|(t, _)| t == "Wait") { return Some(a); }
    if let Some(a) = grouped.iter().find(|(t, _)| t == "Rest") { return Some(a); }
    grouped.iter().min_by(|(a, _), (b, _)| {
        strategic_value_of(a).partial_cmp(&strategic_value_of(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Run a single branch: apply action, advance ticks_per_branch.
/// Returns (terminal, outcome_str, branch_state, intermediate_values, action_sequence).
fn run_branch(
    root: &RootState,
    action: &CampaignAction,
    action_type: &str,
    config: &BfsConfig,
    rollout_mode: RolloutMode,
    initial_bucket: &str,
) -> (bool, Option<String>, CampaignState, Vec<f32>, Vec<String>) {
    let mut branch_state = root.state.clone();
    step_campaign(&mut branch_state, Some(action.clone()));

    let mut terminal = false;
    let mut outcome_str = None;
    let action_hash = action_type.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut branch_rng = root.seed.wrapping_mul(6364136223846793005)
        .wrapping_add(root.state.tick)
        .wrapping_add(action_hash);

    let tpb = config.ticks_per_branch.max(200);
    let checkpoints = [tpb / 4, tpb / 2, tpb * 3 / 4];
    let mut intermediate_values = Vec::with_capacity(3);
    let mut tick_count = 0u64;
    let mut action_sequence = vec![action_type.to_string()];

    for _ in 0..tpb {
        let h_action = strategic_rollout_action(&branch_state, &mut branch_rng, rollout_mode, initial_bucket);
        if action_sequence.len() < 20 {
            if let Some(ref ha) = h_action {
                action_sequence.push(action_type_name(ha));
            }
        }
        let result = step_campaign(&mut branch_state, h_action);
        tick_count += 1;

        if !terminal && checkpoints.contains(&tick_count) {
            intermediate_values.push(estimate_value(&branch_state));
        }

        if let Some(outcome) = result.outcome {
            outcome_str = Some(format!("{:?}", outcome));
            terminal = true;
            let terminal_val = match outcome_str.as_deref() {
                Some("Victory") => 5.0,
                Some("Defeat") => 0.0,
                _ => 2.5,
            };
            while intermediate_values.len() < 3 { intermediate_values.push(terminal_val); }
            break;
        }
    }
    while intermediate_values.len() < 3 { intermediate_values.push(estimate_value(&branch_state)); }

    (terminal, outcome_str, branch_state, intermediate_values, action_sequence)
}

/// Compute leaf value from branch outcome.
fn leaf_value_from_outcome(terminal: bool, outcome_str: &Option<String>, branch_state: &CampaignState) -> f32 {
    if terminal {
        match outcome_str.as_deref() {
            Some("Victory") => 5.0,
            Some("Defeat") => 0.0,
            _ => 2.5,
        }
    } else {
        estimate_value(branch_state)
    }
}

/// Expand a single root: try action types selected via UCB + progressive widening,
/// run a counterfactual baseline, compute novelty and importance weights.
fn expand_root(
    root: &RootState,
    config: &BfsConfig,
    wave: u32,
    shared: &Mutex<BfsSharedState>,
) -> (Vec<BfsSample>, Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)>) {
    let root_value = estimate_value(&root.state);
    if root_value < 0.05 {
        return (vec![], vec![]);
    }

    let raw_valid_actions = root.state.valid_actions();
    // Pass 10: prune irrelevant actions, then cluster for diversity
    let pruned_actions = prune_irrelevant_actions(&raw_valid_actions, &root.state);
    let clustered_actions = cluster_similar_actions(&pruned_actions, &root.state);
    let grouped = group_actions(&clustered_actions);
    let root_tokens = root.state.to_tokens();
    let root_tick = root.state.tick;
    let root_phase = phase_tag(root_tick, root.state.overworld.campaign_progress);
    // valid_action_types still reflects ALL valid actions for RL context
    let valid_action_types: Vec<String> = {
        let all_grouped = group_actions(&raw_valid_actions);
        all_grouped.iter().map(|(t, _)| t.clone()).collect()
    };

    let num_action_types = grouped.len();
    if num_action_types == 0 {
        return (vec![], vec![]);
    }

    // Counterfactual baseline (pass 5)
    let baseline_value = if let Some(baseline_entry) = find_baseline_action(&grouped) {
        let (terminal, outcome_str, branch_state, _, _) = run_branch(
            root, &baseline_entry.1, &baseline_entry.0, config, RolloutMode::Mixed, "wait",
        );
        leaf_value_from_outcome(terminal, &outcome_str, &branch_state)
    } else {
        root_value
    };

    // Adaptive branching (pass 5)
    let tension = estimate_tension(&root.state);
    let tension_multiplier = (1.0 + tension * 0.3).clamp(1.0, 1.5);
    let action_space_factor = (num_action_types as f32 / 10.0).clamp(0.5, 2.0);
    let target_branches = ((num_action_types as f32) * action_space_factor * tension_multiplier)
        .round() as usize;
    let target_branches = target_branches.min(num_action_types);

    // Compute impact estimates for all grouped actions (pass 10)
    let impact_scores: Vec<f32> = grouped.iter()
        .map(|(_, action)| estimate_impact(action, &root.state))
        .collect();

    // UCB-style action selection with impact weighting (pass 5 + pass 10)
    let mut scored_actions: Vec<(f32, usize)> = {
        let shared_guard = shared.lock().unwrap();
        grouped.iter().enumerate().map(|(idx, (action_type, _))| {
            let sv = strategic_value_of(action_type);
            let ucb = shared_guard.ucb_score(action_type, sv);
            // Blend UCB with impact estimate (pass 10)
            let impact_bonus = impact_scores[idx] * 0.5;
            (ucb + impact_bonus, idx)
        }).collect()
    };
    scored_actions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Hierarchical action selection (pass 10): ensure bucket diversity
    let hierarchical_rng = root.seed.wrapping_mul(6364136223846793005)
        .wrapping_add(root.state.tick).wrapping_add(wave as u64);
    let hierarchical_indices = hierarchical_select_actions(
        &grouped, target_branches, shared, hierarchical_rng,
    );

    // Progressive widening (pass 5)
    let first_wave_k = if num_action_types > 20 {
        (num_action_types as f32).sqrt().ceil() as usize
    } else {
        target_branches
    };

    // Merge UCB-scored top-k with hierarchical diversity picks (pass 10)
    let mut selected_set: HashSet<usize> = HashSet::new();
    let mut selected_indices: Vec<usize> = Vec::with_capacity(target_branches);

    // First half from UCB scoring (exploitation)
    let ucb_budget = first_wave_k.min(target_branches) / 2;
    for &(_, idx) in scored_actions.iter().take(ucb_budget) {
        if selected_set.insert(idx) {
            selected_indices.push(idx);
        }
    }
    // Second half from hierarchical bucket-diverse selection (exploration)
    for idx in hierarchical_indices {
        if selected_indices.len() >= first_wave_k.min(target_branches) { break; }
        if selected_set.insert(idx) {
            selected_indices.push(idx);
        }
    }
    // Fill remainder from UCB if needed
    for &(_, idx) in &scored_actions {
        if selected_indices.len() >= first_wave_k.min(target_branches) { break; }
        if selected_set.insert(idx) {
            selected_indices.push(idx);
        }
    }

    let total_ucb: f32 = scored_actions.iter().map(|(s, _)| *s).sum();

    // Track local action counts for replay priority (pass 4)
    let mut local_action_counts: HashMap<String, u64> = HashMap::new();
    for (action_type, _) in &grouped {
        *local_action_counts.entry(action_type.clone()).or_default() += 1;
    }

    // First pass: collect branch results
    struct BranchResult {
        action_type: String,
        action_detail: String,
        action_encoding: Vec<f32>,
        leaf_tokens: Vec<EntityToken>,
        leaf_value: f32,
        leaf_outcome: Option<String>,
        leaf_tick: u64,
        leaf_features: Vec<f32>,
        terminal: bool,
        branch_state: CampaignState,
        intermediate_values: Vec<f32>,
        action_sequence: Vec<String>,
        rollout_mode: RolloutMode,
        state_novelty: f32,
        importance_weight: f32,
        action_strategic_value: f32,
        ucb_idx: usize,
        // pass 8 fields
        action_meta: Vec<f32>,
        action_prereqs: Vec<f32>,
        action_outcome: Vec<f32>,
        action_synergy_score: f32,
        // pass 10 field
        estimated_impact: f32,
    }

    // Compute curriculum fields once per root (pass 9)
    let root_curriculum_tier = compute_curriculum_tier(&root.state, num_action_types);
    let root_state_complexity = compute_state_complexity(&root.state, num_action_types);

    // Track recent actions for synergy (pass 8)
    let recent_action_types: Vec<String> = Vec::new();

    let mut branch_results: Vec<BranchResult> = Vec::new();

    for &idx in &selected_indices {
        let (action_type, action) = &grouped[idx];
        let action_detail = format!("{:?}", action);
        let action_encoding = encode_action(action, &root.state);
        let sv = strategic_value(action_type, &root.state);

        let action_hash = action_type.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut mode_rng = root.seed.wrapping_mul(6364136223846793005)
            .wrapping_add(root.state.tick).wrapping_add(action_hash);
        let rollout_mode = RolloutMode::from_action(action_type, &mut mode_rng);
        let initial_bucket = strategic_bucket(action_type);

        let (terminal, outcome_str, branch_state, intermediate_values, action_sequence) =
            run_branch(root, action, action_type, config, rollout_mode, initial_bucket);
        let leaf_value = leaf_value_from_outcome(terminal, &outcome_str, &branch_state);

        let leaf_tokens = branch_state.to_tokens();
        let leaf_features = state_features(&branch_state);

        let state_novelty = {
            let mut shared_guard = shared.lock().unwrap();
            shared_guard.record_action(action_type);
            shared_guard.record_state_and_novelty(&branch_state)
        };

        let action_ucb = scored_actions.iter()
            .find(|(_, i)| *i == idx).map(|(s, _)| *s).unwrap_or(1.0);
        let selection_prob = if total_ucb > 0.0 { action_ucb / total_ucb } else { 1.0 / num_action_types as f32 };
        let importance_weight = (1.0 / selection_prob.max(0.001)).min(100.0);

        // Pass 8: compute action metadata
        let meta_feats = action_meta_features(action_type);
        let ctx = action_context(action, &root.state);
        let outcome_pred = predict_outcome(action, &root.state);
        let synergy = action_synergy(&recent_action_types, action_type);

        // Pass 10: impact estimate
        let action_impact = impact_scores.get(idx).copied().unwrap_or(0.5);

        branch_results.push(BranchResult {
            action_type: action_type.clone(),
            action_detail,
            action_encoding,
            leaf_tokens,
            leaf_value,
            leaf_outcome: outcome_str,
            leaf_tick: branch_state.tick,
            leaf_features,
            terminal,
            branch_state,
            intermediate_values,
            action_sequence,
            rollout_mode,
            state_novelty,
            importance_weight,
            action_strategic_value: sv,
            ucb_idx: idx,
            action_meta: meta_feats,
            action_prereqs: ctx.to_features(),
            action_outcome: outcome_pred.to_features(),
            action_synergy_score: synergy,
            estimated_impact: action_impact,
        });
    }

    // Progressive widening expansion (pass 5)
    if num_action_types > 20 && selected_indices.len() < target_branches {
        let first_wave_values: Vec<f32> = branch_results.iter().map(|b| b.leaf_value).collect();
        let best = first_wave_values.iter().cloned().fold(f32::MIN, f32::max);
        let worst = first_wave_values.iter().cloned().fold(f32::MAX, f32::min);
        let spread = best - worst;

        if spread < 0.2 {
            let extra_k = first_wave_k.min(target_branches - selected_indices.len());
            let already_selected: HashSet<usize> = selected_indices.iter().copied().collect();
            let extra_indices: Vec<usize> = scored_actions.iter()
                .map(|(_, idx)| *idx)
                .filter(|idx| !already_selected.contains(idx))
                .take(extra_k)
                .collect();

            for &idx in &extra_indices {
                let (action_type, action) = &grouped[idx];
                let action_detail = format!("{:?}", action);
                let action_encoding = encode_action(action, &root.state);
                let sv = strategic_value(action_type, &root.state);

                let action_hash = action_type.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                let mut mode_rng = root.seed.wrapping_mul(6364136223846793005)
                    .wrapping_add(root.state.tick).wrapping_add(action_hash);
                let rollout_mode = RolloutMode::from_action(action_type, &mut mode_rng);
                let initial_bucket = strategic_bucket(action_type);

                let (terminal, outcome_str, branch_state, intermediate_values, action_sequence) =
                    run_branch(root, action, action_type, config, rollout_mode, initial_bucket);
                let leaf_value = leaf_value_from_outcome(terminal, &outcome_str, &branch_state);
                let leaf_tokens = branch_state.to_tokens();
                let leaf_features = state_features(&branch_state);

                let state_novelty = {
                    let mut shared_guard = shared.lock().unwrap();
                    shared_guard.record_action(action_type);
                    shared_guard.record_state_and_novelty(&branch_state)
                };

                let action_ucb = scored_actions.iter()
                    .find(|(_, i)| *i == idx).map(|(s, _)| *s).unwrap_or(1.0);
                let selection_prob = if total_ucb > 0.0 { action_ucb / total_ucb } else { 1.0 / num_action_types as f32 };
                let importance_weight = (1.0 / selection_prob.max(0.001)).min(100.0);

                let meta_feats = action_meta_features(action_type);
                let ctx = action_context(action, &root.state);
                let outcome_pred = predict_outcome(action, &root.state);
                let synergy = action_synergy(&recent_action_types, action_type);
                let action_impact = impact_scores.get(idx).copied().unwrap_or(0.5);

                branch_results.push(BranchResult {
                    action_type: action_type.clone(),
                    action_detail,
                    action_encoding,
                    leaf_tokens,
                    leaf_value,
                    leaf_outcome: outcome_str,
                    leaf_tick: branch_state.tick,
                    leaf_features,
                    terminal,
                    branch_state,
                    intermediate_values,
                    action_sequence,
                    rollout_mode,
                    state_novelty,
                    importance_weight,
                    action_strategic_value: sv,
                    ucb_idx: idx,
                    action_meta: meta_feats,
                    action_prereqs: ctx.to_features(),
                    action_outcome: outcome_pred.to_features(),
                    action_synergy_score: synergy,
                    estimated_impact: action_impact,
                });
            }
            selected_indices.extend(extra_indices);
        }
    }

    // Compute advantages (pass 4)
    let branch_values: Vec<f32> = branch_results.iter().map(|b| b.leaf_value).collect();
    let n = branch_values.len() as f32;
    let mean_value = if n > 0.0 { branch_values.iter().sum::<f32>() / n } else { 0.0 };
    let raw_advantages: Vec<f32> = branch_values.iter().map(|v| v - mean_value).collect();
    let adv_std = if raw_advantages.len() > 1 {
        let var = raw_advantages.iter().map(|a| a * a).sum::<f32>() / (raw_advantages.len() as f32 - 1.0);
        var.sqrt().max(1e-8)
    } else { 1.0 };
    let normalized_advantages: Vec<f32> = raw_advantages.iter().map(|a| a / adv_std).collect();
    let best_idx = branch_values.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    // Build samples and leaves
    let mut samples = Vec::with_capacity(branch_results.len());
    let mut leaves = Vec::new();

    for (i, br) in branch_results.into_iter().enumerate() {
        let ticks_elapsed = br.leaf_tick.saturating_sub(root_tick);
        let discount = temporal_discount(ticks_elapsed);
        let discounted_value = br.leaf_value * discount;
        let value_delta = br.leaf_value - root_value;
        let is_best = best_idx == Some(i);
        let advantage = normalized_advantages[i];
        // Weight replay priority by impact (pass 10): high-impact actions get higher priority
        let impact_weighted_priority = compute_replay_priority(advantage, &br.action_type, &local_action_counts, &root.state)
            * (0.5 + br.estimated_impact);
        let delta = compute_state_delta(&root.state, &br.branch_state);
        let rel_value = relative_value(br.leaf_value, &root_phase);
        let teaches = compute_teaches(&root.state, &br.action_type, &Some(delta.clone()), &valid_action_types);

        samples.push(BfsSample {
            root_tokens: root_tokens.clone(),
            action_type: br.action_type,
            action_detail: br.action_detail,
            action_encoding: br.action_encoding,
            leaf_tokens: br.leaf_tokens,
            leaf_value: br.leaf_value,
            discounted_value,
            leaf_outcome: br.leaf_outcome,
            root_tick,
            leaf_tick: br.leaf_tick,
            seed: root.seed,
            wave,
            cluster_id: 0,
            difficulty: root.difficulty,
            root_value,
            phase_tag: root_phase.clone(),
            valid_action_types: valid_action_types.clone(),
            state_delta: Some(delta),
            action_sequence: br.action_sequence,
            relative_value: rel_value,
            action_strategic_value: br.action_strategic_value,
            rollout_mode: br.rollout_mode.label().to_string(),
            value_delta,
            is_best_action: is_best,
            advantage,
            replay_priority: impact_weighted_priority,
            intermediate_values: br.intermediate_values,
            baseline_value,
            advantage_vs_baseline: br.leaf_value - baseline_value,
            state_novelty: br.state_novelty,
            importance_weight: br.importance_weight,
            // Pass 8 fields
            action_meta: br.action_meta,
            action_prereqs: br.action_prereqs,
            action_outcome: br.action_outcome,
            action_synergy: br.action_synergy_score,
            // Pass 9 fields
            curriculum_tier: root_curriculum_tier,
            teaches,
            trajectory_won: None,     // set post-hoc via apply_hindsight_labels
            campaign_outcome: None,   // set post-hoc via apply_hindsight_labels
            cascade_score: 0.0,       // set post-hoc via apply_hindsight_labels
            contrastive_pair_id: None, // set post-hoc via assign_contrastive_pairs
            is_positive_example: false,
            state_complexity: root_state_complexity,
            // Pass 10 field
            estimated_impact: br.estimated_impact,
        });

        if !br.terminal {
            leaves.push((br.branch_state, br.leaf_features, br.leaf_value, root.seed, root.difficulty));
        }
    }

    (samples, leaves)
}
