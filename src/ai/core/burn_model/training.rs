//! IMPALA V-trace training loop for V6 actor-critic using Burn autodiff.
//!
//! Structure:
//! - vtrace_targets(): pure f32 V-trace computation
//! - ImpalaConfig / TrainingSample: configuration and data types
//! - pack_batch(): samples → padded Burn tensors (shared by all forward paths)
//! - compute_loss(): forward pass + loss computation
//! - train_step(): loss → backward → optimizer step
//! - predict_values(): no-grad value head prediction
//! - predict_values_and_logprobs(): no-grad values + action log probs for IS ratios
//! - rescore_replay_buffer(): recompute V-trace over replay buffer with current policy

use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::optim::{GradientsParams, Optimizer};

use super::actor_critic_v6::{ActorCriticV6, ActorCriticV6Config};
use super::config::*;

// V-trace: re-export from rl4burn (identical signature, drop-in replacement)
pub use rl4burn::vtrace_targets;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ImpalaConfig {
    pub lr: f64,
    pub weight_decay: f32,
    pub beta_1: f32,
    pub beta_2: f32,
    pub batch_size: usize,
    pub gamma: f32,
    pub step_interval: u64,
    pub entropy_coef: f32,
    pub value_coef: f32,
    pub grad_clip: f32,
    pub clip_rho: f32,
    pub clip_c: f32,
    pub ppo_clip: f32,
    pub grokfast_alpha: f32,
    pub grokfast_lamb: f32,
    pub use_grokfast: bool,
    pub tail_drop_min: usize,
    pub tail_drop_max: usize,
}

impl Default for ImpalaConfig {
    fn default() -> Self {
        Self {
            lr: 5e-4, weight_decay: 0.01, beta_1: 0.9, beta_2: 0.98,
            batch_size: 512, gamma: 0.99, step_interval: 3,
            entropy_coef: 0.01, value_coef: 0.5, grad_clip: 5.0,
            clip_rho: 1.0, clip_c: 1.0, ppo_clip: 0.2,
            grokfast_alpha: 0.98, grokfast_lamb: 2.0, use_grokfast: false,
            tail_drop_min: 4, tail_drop_max: 12,
        }
    }
}

// ---------------------------------------------------------------------------
// Training sample
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TrainingSample {
    pub entities: Vec<Vec<f32>>,
    pub entity_types: Vec<u8>,
    pub zones: Vec<Vec<f32>>,
    pub aggregate_features: Vec<f32>,
    pub corner_tokens: Vec<Vec<f32>>,
    /// Model's position output [x, y] in world space (deterministic, no sampling)
    pub target_move_pos: [f32; 2],
    /// Behavior policy log prob for movement (0.0 for deterministic)
    pub behavior_lp_move: f32,
    pub step_reward: f32,
    pub combat_type: usize,
    pub target_idx: usize,
    pub combat_mask: Vec<bool>,
    /// Behavior log prob for combat+pointer
    pub behavior_log_prob: f32,
    pub value_target: f32,
    pub advantage: f32,
    /// Trajectory grouping for V-trace rescore
    pub traj_id: u64,
    pub traj_pos: usize,
    pub traj_terminal: bool,
}

// ---------------------------------------------------------------------------
// Batch packing: samples → Burn tensors (shared by all forward paths)
// ---------------------------------------------------------------------------

/// Packed batch of training samples as Burn tensors, ready for model forward pass.
pub struct PackedBatch<B: Backend> {
    pub ent_feat: Tensor<B, 3>,
    pub ent_types: Tensor<B, 2, Int>,
    pub ent_mask: Tensor<B, 2, Bool>,
    pub zone_feat: Tensor<B, 3>,
    pub zone_mask: Tensor<B, 2, Bool>,
    pub agg_feat: Tensor<B, 2>,
    pub corner_tokens: Option<Tensor<B, 3>>,
    pub corner_mask: Option<Tensor<B, 2, Bool>>,
    pub combat_masks: Tensor<B, 2>,   // [B, NUM_COMBAT_TYPES] float for masking
    pub move_targets: Vec<f32>,        // [B*2] world-space
    pub combat_targets: Vec<i64>,      // [B]
    pub pointer_targets: Vec<i64>,     // [B]
    pub advantages: Vec<f32>,          // [B]
    pub value_targets: Vec<f32>,       // [B]
    pub old_lp_combat: Vec<f32>,       // [B]
    pub bs: usize,
    pub max_ent: usize,
}

pub fn pack_batch<B: Backend>(
    samples: &[TrainingSample],
    device: &B::Device,
) -> PackedBatch<B> {
    let bs = samples.len();
    let max_ent = samples.iter().map(|s| s.entities.len()).max().unwrap_or(1).max(1);
    let max_zones = samples.iter().map(|s| s.zones.len()).max().unwrap_or(1).max(1);
    let max_corners = samples.iter().map(|s| s.corner_tokens.len()).max().unwrap_or(0);
    let has_corners = max_corners > 0;
    let max_corners = max_corners.max(1);

    let mut ent_data = vec![0.0f32; bs * max_ent * ENTITY_DIM];
    let mut etype_data = vec![0.0f32; bs * max_ent];
    let mut emask_data = vec![0.0f32; bs * max_ent];
    let mut zone_data = vec![0.0f32; bs * max_zones * ZONE_DIM];
    let mut zmask_data = vec![0.0f32; bs * max_zones];
    let mut agg_data = vec![0.0f32; bs * AGG_DIM];
    let mut corner_data = vec![0.0f32; bs * max_corners * 11];
    let mut cmask_data = vec![0.0f32; bs * max_corners];
    let mut combat_mask_data = vec![0.0f32; bs * NUM_COMBAT_TYPES];

    let mut move_targets = vec![0.0f32; bs * 2];
    let mut combat_targets = vec![0i64; bs];
    let mut pointer_targets = vec![0i64; bs];
    let mut advantages = vec![0.0f32; bs];
    let mut value_targets = vec![0.0f32; bs];
    let mut old_lp_combat = vec![0.0f32; bs];

    for (i, s) in samples.iter().enumerate() {
        for (j, ent) in s.entities.iter().enumerate().take(max_ent) {
            let base = (i * max_ent + j) * ENTITY_DIM;
            let len = ent.len().min(ENTITY_DIM);
            ent_data[base..base + len].copy_from_slice(&ent[..len]);
            etype_data[i * max_ent + j] = s.entity_types.get(j).copied().unwrap_or(0) as f32;
            emask_data[i * max_ent + j] = 1.0;
        }
        for (j, zone) in s.zones.iter().enumerate().take(max_zones) {
            let base = (i * max_zones + j) * ZONE_DIM;
            let len = zone.len().min(ZONE_DIM);
            zone_data[base..base + len].copy_from_slice(&zone[..len]);
            zmask_data[i * max_zones + j] = 1.0;
        }
        let alen = s.aggregate_features.len().min(AGG_DIM);
        if alen > 0 {
            agg_data[i * AGG_DIM..i * AGG_DIM + alen]
                .copy_from_slice(&s.aggregate_features[..alen]);
        }
        for (j, corner) in s.corner_tokens.iter().enumerate().take(max_corners) {
            let base = (i * max_corners + j) * 11;
            let len = corner.len().min(11);
            corner_data[base..base + len].copy_from_slice(&corner[..len]);
            cmask_data[i * max_corners + j] = 1.0;
        }
        for (j, &m) in s.combat_mask.iter().enumerate().take(NUM_COMBAT_TYPES) {
            if m { combat_mask_data[i * NUM_COMBAT_TYPES + j] = 1.0; }
        }

        move_targets[i * 2] = s.target_move_pos[0];
        move_targets[i * 2 + 1] = s.target_move_pos[1];
        combat_targets[i] = s.combat_type as i64;
        pointer_targets[i] = s.target_idx.min(max_ent - 1) as i64;
        advantages[i] = s.advantage;
        value_targets[i] = s.value_target;
        old_lp_combat[i] = s.behavior_log_prob;
    }

    let ent_feat = Tensor::<B, 1>::from_floats(ent_data.as_slice(), device).reshape([bs, max_ent, ENTITY_DIM]);
    let ent_types = Tensor::<B, 1>::from_floats(etype_data.as_slice(), device).reshape([bs, max_ent]).int();
    let ent_mask = Tensor::<B, 1>::from_floats(emask_data.as_slice(), device).reshape([bs, max_ent]).greater_elem(0.5);
    let zone_feat = Tensor::<B, 1>::from_floats(zone_data.as_slice(), device).reshape([bs, max_zones, ZONE_DIM]);
    let zone_mask = Tensor::<B, 1>::from_floats(zmask_data.as_slice(), device).reshape([bs, max_zones]).greater_elem(0.5);
    let agg_feat = Tensor::<B, 1>::from_floats(agg_data.as_slice(), device).reshape([bs, AGG_DIM]);
    let combat_masks = Tensor::<B, 1>::from_floats(combat_mask_data.as_slice(), device).reshape([bs, NUM_COMBAT_TYPES]);

    let (corner_tokens, corner_mask) = if has_corners {
        let ct = Tensor::<B, 1>::from_floats(corner_data.as_slice(), device).reshape([bs, max_corners, 11]);
        let cm = Tensor::<B, 1>::from_floats(cmask_data.as_slice(), device).reshape([bs, max_corners]).greater_elem(0.5);
        (Some(ct), Some(cm))
    } else {
        (None, None)
    };

    PackedBatch {
        ent_feat, ent_types, ent_mask, zone_feat, zone_mask, agg_feat,
        corner_tokens, corner_mask, combat_masks,
        move_targets, combat_targets, pointer_targets,
        advantages, value_targets, old_lp_combat,
        bs, max_ent,
    }
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

/// Behavioral cloning loss: maximize log prob of expert actions + MSE on position.
/// No advantage weighting — pure supervised imitation learning.
pub fn compute_bc_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &ActorCriticV6<B>,
    samples: &[TrainingSample],
    n_latents: usize,
    device: &B::Device,
) -> (Tensor<B, 1>, TrainMetrics)
where
    B::InnerBackend: burn::tensor::backend::Backend,
{
    let batch = pack_batch(samples, device);
    let bs = batch.bs;
    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    let (output, _h_new, value_out) = model.forward(
        batch.ent_feat, batch.ent_types, batch.zone_feat,
        batch.ent_mask, batch.zone_mask,
        &ability_cls, Some(batch.agg_feat), None,
        batch.corner_tokens, batch.corner_mask, Some(n_latents),
    );

    // Movement: MSE to expert target position
    let target_pos = Tensor::<B, 1>::from_floats(batch.move_targets.as_slice(), device)
        .reshape([bs, 2]) / POSITION_NORM;
    let move_loss: Tensor<B, 1> = (output.target_pos - target_pos).powf_scalar(2.0)
        .sum_dim(1).squeeze_dim::<1>(1).mean().unsqueeze();

    // Combat type: cross-entropy
    let masked_logits = output.combat.combat_logits
        + (batch.combat_masks - 1.0) * 1e9;
    let combat_lp = log_softmax(masked_logits, 1)
        .gather(1, Tensor::<B, 1, Int>::from_ints(batch.combat_targets.as_slice(), device).reshape([bs, 1]))
        .squeeze_dim::<1>(1);
    let combat_ce = combat_lp.mean().neg().unsqueeze(); // -mean(log_prob)

    // Pointer target: cross-entropy, only for attack actions (combat_type 0 = attack)
    // For hold/move actions, the pointer target is meaningless (masked to -1e9 by combat head)
    let ptr_lp = log_softmax(output.combat.attack_ptr, 1)
        .gather(1, Tensor::<B, 1, Int>::from_ints(batch.pointer_targets.as_slice(), device).reshape([bs, 1]))
        .squeeze_dim::<1>(1);
    // Mask: only train pointer on attack-type actions (combat_type 0)
    let is_attack: Vec<f32> = batch.combat_targets.iter()
        .map(|&ct| if ct == 0 { 1.0 } else { 0.0 }).collect();
    let atk_mask = Tensor::<B, 1>::from_floats(is_attack.as_slice(), device);
    let n_atk = atk_mask.clone().sum().into_scalar().elem::<f32>().max(1.0);
    let ptr_ce = ((ptr_lp * atk_mask).sum() / n_atk).neg().unsqueeze();

    // Value loss: Huber
    let value_pred: Tensor<B, 1> = value_out.attrition.squeeze_dim::<1>(1);
    let value_target = Tensor::<B, 1>::from_floats(batch.value_targets.as_slice(), device);
    let value_loss = rl4burn::value_loss(value_pred, value_target);

    // Total: move + combat + pointer + value
    let policy_loss = move_loss.clone() + combat_ce.clone() + ptr_ce.clone();
    let total = policy_loss.clone() + value_loss.clone() * 0.5;

    let metrics = TrainMetrics {
        total_loss: total.clone().into_scalar().elem::<f32>(),
        policy_loss: policy_loss.into_scalar().elem::<f32>(),
        value_loss: value_loss.into_scalar().elem::<f32>(),
        mean_advantage: 0.0,
    };

    (total, metrics)
}

#[derive(Clone, Debug, Default)]
pub struct TrainMetrics {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub mean_advantage: f32,
}

/// Compute unified policy + value loss.
///
/// Policy loss: advantage * (move_mse + neg_combat_log_prob)
///   - Both action heads receive gradient from the same advantage signal
///   - Move: MSE between model position and recorded position, positive-advantage only
///   - Combat: negative log prob of recorded action (standard policy gradient)
///
/// Value loss: Huber(V(s) - vtrace_target, δ=1.0)
pub fn compute_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &ActorCriticV6<B>,
    samples: &[TrainingSample],
    config: &ImpalaConfig,
    n_latents: usize,
    device: &B::Device,
) -> (Tensor<B, 1>, TrainMetrics)
where
    B::InnerBackend: burn::tensor::backend::Backend,
{
    let batch = pack_batch(samples, device);
    let bs = batch.bs;
    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    let (output, _h_new, value_out) = model.forward(
        batch.ent_feat, batch.ent_types, batch.zone_feat,
        batch.ent_mask, batch.zone_mask,
        &ability_cls, Some(batch.agg_feat), None,
        batch.corner_tokens, batch.corner_mask, Some(n_latents),
    );

    // --- Normalize advantages per-minibatch, clamp ---
    let adv_vec = rl4burn::normalize(&batch.advantages, config.grad_clip);
    let adv = Tensor::<B, 1>::from_floats(adv_vec.as_slice(), device);

    // --- Policy loss: unified advantage-weighted action cost ---

    // Movement: MSE to recorded position, positive-advantage only
    // (negative advantage + MSE is degenerate — pushes away from all positions)
    let target_pos = Tensor::<B, 1>::from_floats(batch.move_targets.as_slice(), device)
        .reshape([bs, 2]) / POSITION_NORM;
    let move_mse: Tensor<B, 1> = (output.target_pos - target_pos).powf_scalar(2.0)
        .sum_dim(1).squeeze_dim::<1>(1);
    let move_cost = move_mse * adv.clone().clamp_min(0.0);

    // Combat: negative log prob of recorded action (standard REINFORCE, both signs)
    let masked_logits = output.combat.combat_logits
        + (batch.combat_masks - 1.0) * 1e9;
    let combat_lp = log_softmax(masked_logits, 1)
        .gather(1, Tensor::<B, 1, Int>::from_ints(batch.combat_targets.as_slice(), device).reshape([bs, 1]))
        .squeeze_dim::<1>(1);
    let ptr_lp = log_softmax(output.combat.attack_ptr, 1)
        .gather(1, Tensor::<B, 1, Int>::from_ints(batch.pointer_targets.as_slice(), device).reshape([bs, 1]))
        .squeeze_dim::<1>(1);
    let combat_cost = (combat_lp + ptr_lp) * adv.clone();

    // Combined: both heads, same advantage, one mean
    let policy_loss = (move_cost - combat_cost).mean();

    // --- Value loss: Huber (δ=1.0) ---
    let value_pred: Tensor<B, 1> = value_out.attrition.squeeze_dim::<1>(1);
    let value_target = Tensor::<B, 1>::from_floats(batch.value_targets.as_slice(), device);
    let value_loss = rl4burn::value_loss(value_pred, value_target);

    // --- Total ---
    let total = policy_loss.clone() + value_loss.clone() * config.value_coef;

    let metrics = TrainMetrics {
        total_loss: total.clone().into_scalar().elem::<f32>(),
        policy_loss: policy_loss.into_scalar().elem::<f32>(),
        value_loss: value_loss.into_scalar().elem::<f32>(),
        mean_advantage: adv.mean().into_scalar().elem::<f32>(),
    };

    (total, metrics)
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

pub fn train_step<B, O>(
    model: ActorCriticV6<B>,
    optimizer: &mut O,
    samples: &[TrainingSample],
    config: &ImpalaConfig,
    device: &B::Device,
    rng: &mut u64,
) -> (ActorCriticV6<B>, TrainMetrics)
where
    B: burn::tensor::backend::AutodiffBackend,
    B::InnerBackend: burn::tensor::backend::Backend,
    O: Optimizer<ActorCriticV6<B>, B>,
{
    let j_range = config.tail_drop_max - config.tail_drop_min + 1;
    let j_idx = (lcg(rng) as usize) % j_range;
    let n_latents = config.tail_drop_min + j_idx;

    let (loss, metrics) = compute_loss(&model, samples, config, n_latents, device);

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &model);
    let grads_params = rl4burn::clip_grad_norm(&model, grads_params, config.grad_clip);
    let model = optimizer.step(config.lr, model, grads_params);

    (model, metrics)
}

pub fn train_step_bc<B, O>(
    model: ActorCriticV6<B>,
    optimizer: &mut O,
    samples: &[TrainingSample],
    config: &ImpalaConfig,
    device: &B::Device,
    rng: &mut u64,
) -> (ActorCriticV6<B>, TrainMetrics)
where
    B: burn::tensor::backend::AutodiffBackend,
    B::InnerBackend: burn::tensor::backend::Backend,
    O: Optimizer<ActorCriticV6<B>, B>,
{
    let j_range = config.tail_drop_max - config.tail_drop_min + 1;
    let j_idx = (lcg(rng) as usize) % j_range;
    let n_latents = config.tail_drop_min + j_idx;

    let (loss, metrics) = compute_bc_loss(&model, samples, n_latents, device);

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &model);
    let grads_params = rl4burn::clip_grad_norm(&model, grads_params, config.grad_clip);
    let model = optimizer.step(config.lr, model, grads_params);

    (model, metrics)
}

fn lcg(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

// ---------------------------------------------------------------------------
// No-grad value prediction
// ---------------------------------------------------------------------------

/// Predict V(s) for a batch using the value head (no-grad).
pub fn predict_values<B: Backend>(
    model: &ActorCriticV6<B>,
    samples: &[TrainingSample],
    device: &B::Device,
) -> Vec<f32> {
    if samples.is_empty() { return Vec::new(); }
    let batch = pack_batch(samples, device);
    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    let (_output, _h, value_out) = model.forward(
        batch.ent_feat, batch.ent_types, batch.zone_feat,
        batch.ent_mask, batch.zone_mask,
        &ability_cls, Some(batch.agg_feat), None,
        None, None, Some(N_LATENT_TOKENS),
    );

    value_out.attrition.squeeze_dim::<1>(1).to_data()
        .as_slice::<f32>().unwrap().to_vec()
}

/// Per-sample evaluation: value + current-policy log probs.
pub struct SampleEval {
    pub value: f32,
    pub lp_move: f32,
    pub lp_combat: f32,
}

/// Values + action log probs under current policy (no-grad). For IS ratios.
pub fn predict_values_and_logprobs<B: Backend>(
    model: &ActorCriticV6<B>,
    samples: &[TrainingSample],
    device: &B::Device,
) -> Vec<SampleEval> {
    if samples.is_empty() { return Vec::new(); }
    let batch = pack_batch(samples, device);
    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    let (output, _h, value_out) = model.forward(
        batch.ent_feat, batch.ent_types, batch.zone_feat,
        batch.ent_mask, batch.zone_mask,
        &ability_cls, Some(batch.agg_feat), None,
        None, None, Some(N_LATENT_TOKENS),
    );

    let values: Vec<f32> = value_out.attrition.squeeze_dim::<1>(1).to_data()
        .as_slice::<f32>().unwrap().to_vec();

    // Move log prob (deterministic: lp_move = 0 for IS purposes)
    // Combat log prob
    let masked = output.combat.combat_logits + (batch.combat_masks - 1.0) * 1e9;
    let combat_lp_all = log_softmax(masked, 1).to_data();
    let combat_lp_slice: &[f32] = combat_lp_all.as_slice().unwrap();

    let ptr_lp_all = log_softmax(output.combat.attack_ptr, 1).to_data();
    let ptr_lp_slice: &[f32] = ptr_lp_all.as_slice().unwrap();
    let n_tokens = ptr_lp_all.shape[1];

    let bs = samples.len();
    (0..bs).map(|i| {
        let ct = samples[i].combat_type.min(NUM_COMBAT_TYPES - 1);
        let ti = samples[i].target_idx.min(n_tokens - 1);
        SampleEval {
            value: values[i],
            lp_move: 0.0, // deterministic movement
            lp_combat: combat_lp_slice[i * NUM_COMBAT_TYPES + ct]
                + ptr_lp_slice[i * n_tokens + ti],
        }
    }).collect()
}

// ---------------------------------------------------------------------------
// Replay buffer rescore
// ---------------------------------------------------------------------------

/// Recompute V-trace targets for replay buffer using current value head + IS ratios.
/// Groups by traj_id, handles incomplete trajectories from eviction.
pub fn rescore_replay_buffer<B: Backend>(
    samples: &mut Vec<TrainingSample>,
    model: &ActorCriticV6<B>,
    device: &B::Device,
    gamma_eff: f32,
    clip_rho: f32,
    clip_c: f32,
) {
    if samples.is_empty() { return; }

    let batch_sz = 256;
    let mut all_evals: Vec<SampleEval> = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(batch_sz) {
        all_evals.extend(predict_values_and_logprobs(model, &chunk.to_vec(), device));
    }

    // Group by trajectory
    let mut traj_map: std::collections::HashMap<u64, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for (i, s) in samples.iter().enumerate() {
        traj_map.entry(s.traj_id).or_default().push((i, s.traj_pos));
    }

    for (_traj_id, mut indices) in traj_map {
        indices.sort_by_key(|&(_, pos)| pos);
        let len = indices.len();
        let is_terminal = samples[indices[0].0].traj_terminal;

        // Check trajectory completeness: if positions have gaps, skip V-trace
        // and fall back to single-step TD for incomplete trajectories.
        let positions: Vec<usize> = indices.iter().map(|&(_, p)| p).collect();
        let is_contiguous = positions.windows(2).all(|w| w[1] == w[0] + 1);

        let rewards: Vec<f32> = indices.iter().map(|&(i, _)| samples[i].step_reward).collect();
        let values: Vec<f32> = indices.iter().map(|&(i, _)| all_evals[i].value).collect();

        if is_contiguous && len > 1 {
            // Full V-trace with IS ratios
            let log_rhos: Vec<f32> = indices.iter().map(|&(i, _)| {
                let ev = &all_evals[i];
                let old_lp = samples[i].behavior_log_prob;
                (ev.lp_combat - old_lp).clamp(-10.0, 10.0)
            }).collect();
            let discounts = vec![gamma_eff; len];
            let bootstrap = if is_terminal { 0.0 } else { *values.last().unwrap_or(&0.0) };

            let (vs_targets, advantages) = vtrace_targets(
                &log_rhos, &discounts, &rewards, &values, bootstrap, clip_rho, clip_c,
            );
            for (j, &(buf_idx, _)) in indices.iter().enumerate() {
                samples[buf_idx].value_target = vs_targets[j];
                samples[buf_idx].advantage = advantages[j];
            }
        } else {
            // Incomplete trajectory: use single-step TD(0)
            for (j, &(buf_idx, _)) in indices.iter().enumerate() {
                let next_v = if j + 1 < len { values[j + 1] } else if is_terminal { 0.0 } else { values[j] };
                samples[buf_idx].value_target = rewards[j] + gamma_eff * next_v;
                samples[buf_idx].advantage = rewards[j] + gamma_eff * next_v - values[j];
            }
        }
    }
}

// See src/bin/xtask/oracle_cmd/impala_train.rs for the training loop CLI.
