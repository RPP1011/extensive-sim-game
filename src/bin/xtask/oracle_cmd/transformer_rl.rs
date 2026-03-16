//! Actor-critic RL episode generation using the ability transformer.
//!
//! Runs scenarios with the transformer making ALL hero decisions (not just abilities).
//! Records episodes as JSONL for PPO training in Python.

use std::process::ExitCode;

use serde::{Deserialize, Serialize};

use super::collect_toml_paths;

// ---------------------------------------------------------------------------
// Episode types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlEpisode {
    pub scenario: String,
    pub outcome: String,
    pub reward: f32,
    pub ticks: u64,
    /// Per-unit ability token IDs (unit_id -> list of token ID lists).
    pub unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    /// Per-unit ability names (unit_id -> list of ability names).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
    pub steps: Vec<RlStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlStep {
    pub tick: u64,
    pub unit_id: u32,
    pub game_state: Vec<f32>,
    pub action: usize,
    pub log_prob: f32,
    pub mask: Vec<bool>,
    pub step_reward: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entities: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_types: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threats: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zones: Option<Vec<Vec<f32>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_type: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_idx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_dir: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combat_type: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_move: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_combat: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lp_pointer: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_features: Option<Vec<f32>>,
    /// Target movement position (world-space [x, y]) for continuous movement training
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_move_pos: Option<[f32; 2]>,
    /// DAgger: teacher (GOAP) action at this state (what the expert would have done)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_move_dir: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_combat_type: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_target_idx: Option<usize>,
}

impl RlStep {
    /// Create with no teacher labels (non-DAgger paths)
    pub fn without_teacher(
        tick: u64, unit_id: u32, game_state: Vec<f32>, action: usize,
        log_prob: f32, mask: Vec<bool>, step_reward: f32,
    ) -> Self {
        Self {
            tick, unit_id, game_state, action, log_prob, mask, step_reward,
            entities: None, entity_types: None, threats: None, positions: None, zones: None,
            action_type: None, target_idx: None, move_dir: None, combat_type: None,
            lp_move: None, lp_combat: None, lp_pointer: None,
            aggregate_features: None,
            target_move_pos: None,
            teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LCG + softmax
// ---------------------------------------------------------------------------

pub(crate) fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

pub(crate) fn masked_softmax_sample(
    logits: &[f32],
    mask: &[bool],
    temperature: f32,
    rng: &mut u64,
) -> (usize, f32) {
    let n = logits.len();
    let temp = temperature.max(0.01);

    let mut max_scaled = f32::NEG_INFINITY;
    for i in 0..n {
        if mask[i] {
            let scaled = logits[i] / temp;
            if scaled > max_scaled { max_scaled = scaled; }
        }
    }

    let mut probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for i in 0..n {
        if mask[i] {
            let e = ((logits[i] / temp) - max_scaled).exp();
            probs[i] = e;
            sum += e;
        }
    }

    if sum > 0.0 {
        for p in &mut probs { *p /= sum; }
    } else {
        let valid = mask.iter().filter(|&&m| m).count() as f32;
        for (i, p) in probs.iter_mut().enumerate() {
            *p = if mask[i] { 1.0 / valid } else { 0.0 };
        }
    }

    let r = lcg_f32(rng);
    let mut cum = 0.0;
    let mut chosen = n - 1;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum { chosen = i; break; }
    }

    // Return UNTEMPERED log probability
    let mut max_raw = f32::NEG_INFINITY;
    for i in 0..n {
        if mask[i] && logits[i] > max_raw { max_raw = logits[i]; }
    }
    let mut log_sum_exp = 0.0f32;
    for i in 0..n {
        if mask[i] { log_sum_exp += (logits[i] - max_raw).exp(); }
    }
    let log_prob = logits[chosen] - max_raw - log_sum_exp.ln();

    (chosen, log_prob)
}

// ---------------------------------------------------------------------------
// Policy abstraction
// ---------------------------------------------------------------------------

pub(crate) const NUM_ACTIONS: usize = 14;
pub(crate) const MAX_ABILITIES: usize = 8;

/// Either actor-critic weights (full policy), legacy transformer weights (bootstrap),
/// or the combined ability-eval + squad AI system.
pub(crate) enum Policy {
    ActorCritic(bevy_game::ai::core::ability_transformer::ActorCriticWeights),
    ActorCriticV2(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV2),
    ActorCriticV3(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV3),
    ActorCriticV4(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV4),
    ActorCriticV5(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV5),
    /// GPU inference via SHM (legacy).
    GpuServer(std::sync::Arc<bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient>),
    /// GPU inference via Burn/LibTorch (in-process, no SHM) — V5 model.
    #[cfg(feature = "burn-gpu")]
    BurnServer(std::sync::Arc<bevy_game::ai::core::burn_model::inference::BurnInferenceClient>),
    /// GPU inference via Burn/LibTorch — V6 model (spatial cross-attn + latent interface).
    #[cfg(feature = "burn-gpu")]
    BurnServerV6(std::sync::Arc<bevy_game::ai::core::burn_model::inference_v6::BurnInferenceClientV6>),
    Legacy(bevy_game::ai::core::ability_transformer::AbilityTransformerWeights),
    /// Uses existing squad AI -- no transformer.
    Combined,
    /// Uniformly random actions -- no model inference.
    Random,
}

impl Policy {
    pub(crate) fn load(path: &std::path::Path) -> Result<Self, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        if json_str.contains("\"actor_critic_v5\"") {
            return Ok(Policy::ActorCriticV5(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV5::from_json(&json_str)?));
        }
        if json_str.contains("\"actor_critic_v4\"") {
            return Ok(Policy::ActorCriticV4(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV4::from_json(&json_str)?));
        }
        if json_str.contains("\"actor_critic_v3\"") {
            return Ok(Policy::ActorCriticV3(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV3::from_json(&json_str)?));
        }
        if json_str.contains("\"actor_critic_v2\"") {
            return Ok(Policy::ActorCriticV2(bevy_game::ai::core::ability_transformer::ActorCriticWeightsV2::from_json(&json_str)?));
        }
        if json_str.contains("\"actor_critic\"") {
            return Ok(Policy::ActorCritic(bevy_game::ai::core::ability_transformer::ActorCriticWeights::from_json(&json_str)?));
        }
        let tw = bevy_game::ai::core::ability_transformer::AbilityTransformerWeights::from_json(&json_str)?;
        Ok(Policy::Legacy(tw))
    }

    pub(crate) fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        match self {
            Policy::ActorCritic(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV2(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV3(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV4(ac) => ac.encode_cls(token_ids),
            Policy::ActorCriticV5(ac) => ac.encode_cls(token_ids),
            Policy::Legacy(tw) => tw.encode_cls(token_ids),
            Policy::GpuServer(_) | Policy::Combined | Policy::Random => Vec::new(),
            #[cfg(feature = "burn-gpu")]
            Policy::BurnServer(_) | Policy::BurnServerV6(_) => Vec::new(),
        }
    }

    pub(crate) fn needs_transformer(&self) -> bool {
        #[cfg(feature = "burn-gpu")]
        if matches!(self, Policy::BurnServer(_) | Policy::BurnServerV6(_)) { return false; }
        !matches!(self, Policy::Combined | Policy::GpuServer(_) | Policy::Random)
    }

    pub(crate) fn is_v5(&self) -> bool {
        matches!(self, Policy::ActorCriticV5(_))
    }

    pub(crate) fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        match self {
            Policy::ActorCritic(ac) => ac.project_external_cls(cls),
            Policy::ActorCriticV3(ac) => ac.project_external_cls(cls),
            Policy::ActorCriticV4(ac) => ac.project_external_cls(cls),
            Policy::ActorCriticV5(ac) => ac.project_external_cls(cls),
            _ => cls.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Behavior DSL loading and intent overrides
// ---------------------------------------------------------------------------

/// Load behavior trees for enemy units that have a `behavior` field set.
pub(crate) fn load_behavior_trees(
    sim: &bevy_game::ai::core::SimState,
    cfg: &bevy_game::scenario::ScenarioCfg,
) -> std::collections::HashMap<u32, bevy_game::ai::behavior::BehaviorTree> {
    use bevy_game::ai::behavior::parse_behavior;
    use bevy_game::ai::core::Team;

    let mut trees = std::collections::HashMap::new();
    if cfg.enemy_units.is_empty() { return trees; }

    let enemy_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();

    for (i, eu) in cfg.enemy_units.iter().enumerate() {
        if let Some(ref behavior_name) = eu.behavior {
            if let Some(&uid) = enemy_ids.get(i) {
                let path = format!("assets/behaviors/{}.behavior", behavior_name);
                match std::fs::read_to_string(&path) {
                    Ok(content) => match parse_behavior(&content) {
                        Ok(tree) => { trees.insert(uid, tree); }
                        Err(e) => eprintln!("Warning: failed to parse {}: {}", path, e),
                    },
                    Err(e) => eprintln!("Warning: failed to read {}: {}", path, e),
                }
            }
        }
    }
    trees
}

/// Override intents for units that have behavior trees.
pub(crate) fn apply_behavior_overrides(
    intents: &mut [bevy_game::ai::core::UnitIntent],
    behaviors: &std::collections::HashMap<u32, bevy_game::ai::behavior::BehaviorTree>,
    sim: &bevy_game::ai::core::SimState,
    tick: u64,
) {
    use bevy_game::ai::behavior::evaluate_behavior;
    if behaviors.is_empty() { return; }
    for intent in intents.iter_mut() {
        if let Some(tree) = behaviors.get(&intent.unit_id) {
            intent.action = evaluate_behavior(tree, sim, intent.unit_id, tick);
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario-level action mask enforcement
// ---------------------------------------------------------------------------

pub(crate) fn apply_action_mask(combat_mask: &mut [bool], action_mask: Option<&str>) {
    match action_mask {
        Some("move_only") => {
            combat_mask[0] = false;
            for slot in combat_mask.iter_mut().skip(2) { *slot = false; }
        }
        Some("move_attack") => {
            for slot in combat_mask.iter_mut().skip(2) { *slot = false; }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

pub fn run_transformer_rl(args: crate::cli::TransformerRlArgs) -> ExitCode {
    match args.sub {
        crate::cli::TransformerRlSubcommand::Generate(gen_args) => super::rl_generate::run_generate(gen_args),
        crate::cli::TransformerRlSubcommand::Eval(eval_args) => super::rl_eval::run_eval(eval_args),
        crate::cli::TransformerRlSubcommand::ImpalaTrain(train_args) => super::impala_train::run_impala_train(train_args),
    }
}

// Re-export run_rl_episode for use by rl_eval
pub(crate) use super::rl_episode::run_rl_episode;
