//! Actor-critic RL episode generation using the ability transformer.
//!
//! Runs scenarios with the transformer making ALL hero decisions (not just abilities).
//! Records episodes as JSONL for PPO training in Python.

use std::process::ExitCode;

use serde::{Deserialize, Serialize};


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
    #[allow(dead_code)]
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

pub(crate) const MAX_ABILITIES: usize = 8;

/// V5 actor-critic policy, combined squad AI, or random baseline.
pub(crate) enum Policy {
    ActorCriticV5(game::ai::core::ability_transformer::ActorCriticWeightsV5),
    /// GPU inference via Burn/LibTorch (in-process, no SHM) — V5 model.
    #[cfg(feature = "burn-gpu")]
    BurnServer(std::sync::Arc<game::ai::core::burn_model::inference::BurnInferenceClient>),
    /// GPU inference via Burn/LibTorch — V6 model (spatial cross-attn + latent interface).
    #[cfg(feature = "burn-gpu")]
    BurnServerV6(std::sync::Arc<game::ai::core::burn_model::inference_v6::BurnInferenceClientV6>),
    /// Uses existing squad AI -- no transformer.
    Combined,
    /// Uniformly random actions -- no model inference.
    Random,
}

impl Policy {
    pub(crate) fn load(path: &std::path::Path) -> Result<Self, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        Ok(Policy::ActorCriticV5(game::ai::core::ability_transformer::ActorCriticWeightsV5::from_json(&json_str)?))
    }

    pub(crate) fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        match self {
            Policy::ActorCriticV5(ac) => ac.encode_cls(token_ids),
            Policy::Combined | Policy::Random => Vec::new(),
            #[cfg(feature = "burn-gpu")]
            Policy::BurnServer(_) | Policy::BurnServerV6(_) => Vec::new(),
        }
    }

    pub(crate) fn needs_transformer(&self) -> bool {
        #[cfg(feature = "burn-gpu")]
        if matches!(self, Policy::BurnServer(_) | Policy::BurnServerV6(_)) { return false; }
        !matches!(self, Policy::Combined | Policy::Random)
    }

    #[allow(dead_code)]
    pub(crate) fn is_v5(&self) -> bool {
        matches!(self, Policy::ActorCriticV5(_))
    }

    pub(crate) fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        match self {
            Policy::ActorCriticV5(ac) => ac.project_external_cls(cls),
            _ => cls.to_vec(),
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
// Precomputed scenarios (extracted from former rl_gpu_sim module)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(crate) struct PrecomputedScenario {
    pub(crate) sim: game::ai::core::SimState,
    pub(crate) squad_ai: game::ai::squad::SquadAiState,
    pub(crate) scenario_name: String,
    pub(crate) max_ticks: u64,
    pub(crate) unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    pub(crate) unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
    pub(crate) cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>>,
    pub(crate) hero_ids: Vec<u32>,
    pub(crate) enemy_ids: Vec<u32>,
    pub(crate) initial_hero_hp: i32,
    pub(crate) initial_enemy_hp: i32,
    pub(crate) avg_unit_hp: f32,
    pub(crate) initial_hero_count: f32,
    pub(crate) initial_enemy_count: f32,
    pub(crate) drill_objective_type: Option<String>,
    pub(crate) drill_target_position: Option<[f32; 2]>,
    pub(crate) drill_target_radius: Option<f32>,
    pub(crate) action_mask: Option<String>,
    pub(crate) objective: Option<game::scenario::ObjectiveDef>,
}

/// Build all scenario templates once. Returns one `PrecomputedScenario` per scenario.
pub(crate) fn precompute_scenarios(
    scenario_files: &[game::scenario::ScenarioFile],
    max_ticks_override: Option<u64>,
    tokenizer: &game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    registry: Option<&game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> Vec<PrecomputedScenario> {
    use game::ai::core::Team;
    use game::ai::effects::dsl::emit::emit_ability_dsl;
    use game::scenario::run_scenario_to_state_with_room;

    scenario_files.iter().map(|sf| {
        let cfg = &sf.scenario;
        let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);
        let (mut sim, squad_ai, nav) = run_scenario_to_state_with_room(cfg);
        sim.grid_nav = Some(nav);

        let hero_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let enemy_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();

        let mut unit_abilities = std::collections::HashMap::new();
        let mut unit_ability_names = std::collections::HashMap::new();
        let mut cls_cache = std::collections::HashMap::new();

        for &uid in hero_ids.iter().chain(enemy_ids.iter()) {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                let mut tokens_list = Vec::new();
                let mut names_list = Vec::new();
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    tokens_list.push(tokenizer.encode_with_cls(&dsl));
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            cls_cache.insert((uid, idx), reg_cls.to_vec());
                        }
                    }
                    names_list.push(slot.def.name.clone());
                }
                unit_abilities.insert(uid, tokens_list);
                unit_ability_names.insert(uid, names_list);
            }
        }

        let initial_hero_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
        let initial_enemy_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
        let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
        let avg_unit_hp = (initial_hero_hp + initial_enemy_hp) as f32 / n_units;
        let initial_hero_count = sim.units.iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;
        let initial_enemy_count = sim.units.iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;

        let (drill_objective_type, drill_target_position, drill_target_radius) =
            if let Some(ref obj) = cfg.objective {
                (Some(obj.objective_type.clone()), obj.position, obj.radius)
            } else if cfg.drill_type.is_some() {
                (cfg.drill_type.clone(), cfg.target_position, Some(1.0))
            } else {
                (None, None, None)
            };

        PrecomputedScenario {
            sim, squad_ai, scenario_name: cfg.name.clone(), max_ticks,
            unit_abilities, unit_ability_names, cls_cache,
            hero_ids, enemy_ids,
            initial_hero_hp, initial_enemy_hp, avg_unit_hp,
            initial_hero_count, initial_enemy_count,
            drill_objective_type, drill_target_position, drill_target_radius,
            action_mask: cfg.action_mask.clone(),
            objective: cfg.objective.clone(),
        }
    }).collect()
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
