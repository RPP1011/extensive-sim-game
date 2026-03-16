//! IMPALA V-trace training loop for V6 actor-critic using Burn autodiff.
//!
//! Eliminates the Python training loop and SHM protocol entirely.
//! Episode generation and gradient updates both happen in-process.
//!
//! Training pattern:
//! 1. Generate episodes (Rust sim + Burn V6 inference)
//! 2. Extract trajectories, compute V-trace targets per trajectory
//! 3. Train on minibatches: forward → loss → backward → optimizer step
//! 4. Save checkpoint
//!
//! Uses `Autodiff<LibTorch>` backend for automatic differentiation.

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

use super::actor_critic_v6::{ActorCriticV6, ActorCriticV6Config};
use super::config::*;

// ---------------------------------------------------------------------------
// V-trace (pure f32, no tensors needed)
// ---------------------------------------------------------------------------

/// V-trace off-policy correction for a single trajectory.
///
/// Returns (value_targets, advantages) as parallel Vec<f32>.
pub fn vtrace_targets(
    log_rhos: &[f32],
    discounts: &[f32],
    rewards: &[f32],
    values: &[f32],
    bootstrap_value: f32,
    clip_rho: f32,
    clip_c: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t_len = log_rhos.len();
    assert_eq!(t_len, discounts.len());
    assert_eq!(t_len, rewards.len());
    assert_eq!(t_len, values.len());

    // Clipped importance weights
    let rhos: Vec<f32> = log_rhos
        .iter()
        .map(|&lr| lr.clamp(-20.0, 20.0).exp())
        .collect();
    let rho_bar: Vec<f32> = rhos.iter().map(|&r| r.min(clip_rho)).collect();
    let c_bar: Vec<f32> = rhos.iter().map(|&r| r.min(clip_c)).collect();

    // values_{t+1}: shift values left, append bootstrap
    let values_tp1: Vec<f32> = values[1..]
        .iter()
        .copied()
        .chain(std::iter::once(bootstrap_value))
        .collect();

    // TD errors
    let deltas: Vec<f32> = (0..t_len)
        .map(|t| rho_bar[t] * (rewards[t] + discounts[t] * values_tp1[t] - values[t]))
        .collect();

    // Backward temporal accumulation
    let mut vs_minus_v = vec![0.0f32; t_len];
    let mut acc = 0.0f32;
    for t in (0..t_len).rev() {
        acc = deltas[t] + discounts[t] * c_bar[t] * acc;
        vs_minus_v[t] = acc;
    }

    // V-trace targets
    let vs: Vec<f32> = (0..t_len).map(|t| vs_minus_v[t] + values[t]).collect();

    // Policy gradient advantages
    let vs_tp1: Vec<f32> = vs[1..]
        .iter()
        .copied()
        .chain(std::iter::once(bootstrap_value))
        .collect();
    let advantages: Vec<f32> = (0..t_len)
        .map(|t| rho_bar[t] * (rewards[t] + discounts[t] * vs_tp1[t] - values[t]))
        .collect();

    (vs, advantages)
}

// ---------------------------------------------------------------------------
// Grokfast EMA gradient filter
// ---------------------------------------------------------------------------

/// Grokfast EMA state: maintains per-parameter gradient EMA.
///
/// Applied between backward() and optimizer.step():
///   grad_filtered = grad + lamb * ema
///   ema = alpha * ema + (1 - alpha) * grad
pub struct GrokfastEma {
    /// EMA buffers keyed by parameter ID.
    ema: std::collections::HashMap<burn::module::ParamId, Vec<f32>>,
    pub alpha: f32,
    pub lamb: f32,
}

impl GrokfastEma {
    pub fn new(alpha: f32, lamb: f32) -> Self {
        Self {
            ema: std::collections::HashMap::new(),
            alpha,
            lamb,
        }
    }

    /// Filter gradients in-place: grad = grad + lamb * ema, then update ema.
    ///
    /// Operates on the raw GradientsParams by extracting, modifying, and
    /// reinserting each gradient tensor.
    pub fn filter<B: burn::tensor::backend::AutodiffBackend>(
        &mut self,
        grads: &mut B::Gradients,
        model: &ActorCriticV6<B>,
    ) {
        // Grokfast EMA operates on individual parameter gradients.
        // In Burn, we access gradients through the model's parameters.
        // For now, we'll apply the EMA filter after converting to GradientsParams.
        // This is a simplified version — full implementation would iterate named params.
        let _ = (grads, model);
        // TODO: Iterate model parameters, extract grad from grads, apply EMA filter,
        // put filtered grad back. Requires Burn's param visitor pattern.
        // For initial training runs, skip Grokfast and add it once basic RL works.
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

/// IMPALA training configuration.
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
    /// PPO clip epsilon for policy ratio clipping.
    pub ppo_clip: f32,
    pub grokfast_alpha: f32,
    pub grokfast_lamb: f32,
    pub use_grokfast: bool,
    /// Latent tail drop range: sample J from [min, max] during training.
    pub tail_drop_min: usize,
    pub tail_drop_max: usize,
}

impl Default for ImpalaConfig {
    fn default() -> Self {
        Self {
            lr: 5e-4,
            weight_decay: 1.0,
            beta_1: 0.9,
            beta_2: 0.98,
            batch_size: 512,
            gamma: 0.99,
            step_interval: 3,
            entropy_coef: 0.01,
            value_coef: 0.5,
            grad_clip: 1.0,
            clip_rho: 1.0,
            clip_c: 1.0,
            ppo_clip: 0.2,
            grokfast_alpha: 0.98,
            grokfast_lamb: 2.0,
            use_grokfast: false,
            tail_drop_min: 4,
            tail_drop_max: 12,
        }
    }
}

// ---------------------------------------------------------------------------
// Training step data
// ---------------------------------------------------------------------------

/// A single training sample extracted from an RlStep.
#[derive(Clone)]
pub struct TrainingSample {
    /// Entity features [n_entities, ENTITY_DIM]
    pub entities: Vec<Vec<f32>>,
    /// Entity type IDs
    pub entity_types: Vec<u8>,
    /// Zone features [n_zones, ZONE_DIM]
    pub zones: Vec<Vec<f32>>,
    /// Aggregate features [AGG_DIM]
    pub aggregate_features: Vec<f32>,
    /// Corner tokens [n_corners, 11] (may be empty)
    pub corner_tokens: Vec<Vec<f32>>,
    /// Sampled target position [x, y] in world space (from Gaussian policy during generation)
    pub target_move_pos: [f32; 2],
    /// Behavior policy log prob for movement (Gaussian log prob at generation time)
    pub behavior_lp_move: f32,
    /// Per-step reward (dense HP differential + event bonuses)
    pub step_reward: f32,
    /// Action taken: combat type (0-9)
    pub combat_type: usize,
    /// Action taken: target entity index
    pub target_idx: usize,
    /// Action mask (which combat types are valid)
    pub combat_mask: Vec<bool>,
    /// Behavior log prob for combat+pointer (from data-generating policy)
    pub behavior_log_prob: f32,
    /// V-trace value target
    pub value_target: f32,
    /// V-trace advantage
    pub advantage: f32,
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

/// Compute IMPALA loss on a batch of training samples.
///
/// Returns (total_loss, metrics) where loss is a scalar Burn tensor with grad tracking.
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
    let bs = samples.len();

    // Find max dims for padding
    let max_ent = samples.iter().map(|s| s.entities.len()).max().unwrap_or(1).max(1);
    let max_zones = samples.iter().map(|s| s.zones.len()).max().unwrap_or(1).max(1);
    let max_corners = samples.iter().map(|s| s.corner_tokens.len()).max().unwrap_or(0);
    let has_corners = max_corners > 0;
    let max_corners = max_corners.max(1);

    // Pack into flat f32 buffers
    let mut ent_data = vec![0.0f32; bs * max_ent * ENTITY_DIM];
    let mut etype_data = vec![0.0f32; bs * max_ent];
    let mut emask_data = vec![0.0f32; bs * max_ent];
    let mut zone_data = vec![0.0f32; bs * max_zones * ZONE_DIM];
    let mut zmask_data = vec![0.0f32; bs * max_zones];
    let mut agg_data = vec![0.0f32; bs * AGG_DIM];
    let mut corner_data = vec![0.0f32; bs * max_corners * 11];
    let mut cmask_data = vec![0.0f32; bs * max_corners];

    let mut move_pos_targets = vec![0.0f32; bs * 2];
    let mut old_lp_move = vec![0.0f32; bs];
    let mut old_lp_combat = vec![0.0f32; bs];
    let mut combat_targets = vec![0i64; bs];
    let mut pointer_targets = vec![0i64; bs];
    let mut combat_masks = vec![0.0f32; bs * NUM_COMBAT_TYPES];
    let mut advantages = vec![0.0f32; bs];
    let mut value_targets = vec![0.0f32; bs];

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

        move_pos_targets[i * 2] = s.target_move_pos[0];
        move_pos_targets[i * 2 + 1] = s.target_move_pos[1];
        old_lp_move[i] = s.behavior_lp_move;
        old_lp_combat[i] = s.behavior_log_prob;
        combat_targets[i] = s.combat_type as i64;
        pointer_targets[i] = s.target_idx.min(max_ent - 1) as i64;
        for (j, &m) in s.combat_mask.iter().enumerate().take(NUM_COMBAT_TYPES) {
            if m { combat_masks[i * NUM_COMBAT_TYPES + j] = 1.0; }
        }
        advantages[i] = s.advantage;
        value_targets[i] = s.value_target;
    }

    // Build Burn tensors
    let ent_feat = Tensor::<B, 1>::from_floats(ent_data.as_slice(), &device)
        .reshape([bs, max_ent, ENTITY_DIM]);
    let ent_types = Tensor::<B, 1>::from_floats(etype_data.as_slice(), &device)
        .reshape([bs, max_ent]).int();
    let ent_mask = Tensor::<B, 1>::from_floats(emask_data.as_slice(), &device)
        .reshape([bs, max_ent]).greater_elem(0.5);
    let zone_feat = Tensor::<B, 1>::from_floats(zone_data.as_slice(), &device)
        .reshape([bs, max_zones, ZONE_DIM]);
    let zone_mask = Tensor::<B, 1>::from_floats(zmask_data.as_slice(), &device)
        .reshape([bs, max_zones]).greater_elem(0.5);
    let agg_feat = Tensor::<B, 1>::from_floats(agg_data.as_slice(), &device)
        .reshape([bs, AGG_DIM]);

    let (corner_tokens, corner_mask) = if has_corners {
        let ct = Tensor::<B, 1>::from_floats(corner_data.as_slice(), &device)
            .reshape([bs, max_corners, 11]);
        let cm = Tensor::<B, 1>::from_floats(cmask_data.as_slice(), &device)
            .reshape([bs, max_corners]).greater_elem(0.5);
        (Some(ct), Some(cm))
    } else {
        (None, None)
    };

    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    // Forward pass (with value head — needed for training)
    let (output, _h_new, value_out) = model.forward(
        ent_feat, ent_types, zone_feat, ent_mask.clone(), zone_mask,
        &ability_cls, Some(agg_feat), None,
        corner_tokens, corner_mask, Some(n_latents),
    );

    // Advantages tensor (detached — no gradient through advantages)
    let adv_tensor = Tensor::<B, 1>::from_floats(advantages.as_slice(), device);

    // --- Movement loss (Gaussian policy gradient) ---
    // Model outputs μ [B, 2] and log_std [2]. Sampled action was target_move_pos (world space).
    let mu = output.target_pos; // [B, 2] normalized
    let sampled = Tensor::<B, 1>::from_floats(move_pos_targets.as_slice(), device)
        .reshape([bs, 2]) / POSITION_NORM; // normalize to match μ

    // log N(x|μ,σ²) = -0.5 * ((x-μ)/σ)² - log(σ) - 0.5*log(2π)
    let log_std = output.move_log_std.unwrap_or_else(|| Tensor::zeros([2], device));
    let log_std_expanded: Tensor<B, 2> = log_std.clone().unsqueeze_dim::<2>(0).expand([bs, 2]);
    let std_expanded = log_std_expanded.clone().exp();
    let diff = (sampled - mu) / std_expanded.clone();
    let move_log_prob: Tensor<B, 1> = (diff.powf_scalar(2.0).neg() * 0.5
        - log_std_expanded
        - 0.9189) // 0.5 * log(2π) ≈ 0.9189
        .sum_dim(1).squeeze_dim::<1>(1); // sum over [x, y] → [B]

    // Movement entropy: H = 0.5 * log(2πeσ²) = log(σ) + 0.5*log(2πe) ≈ log(σ) + 1.4189
    let move_entropy: Tensor<B, 1> = (log_std.clone() + 1.4189).mean().unsqueeze();

    // PPO-clipped movement policy loss
    let old_lp_move_tensor = Tensor::<B, 1>::from_floats(old_lp_move.as_slice(), device);
    let move_ratio = (move_log_prob - old_lp_move_tensor).exp(); // π_new / π_old
    let clipped_ratio = move_ratio.clone().clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip);
    let surr1 = move_ratio * adv_tensor.clone();
    let surr2 = clipped_ratio * adv_tensor.clone();
    // PPO loss: -E[min(surr1, surr2)] — take the pessimistic bound
    let move_loss = surr1.min_pair(surr2).mean().neg();

    // --- Policy loss (combat + pointer) ---

    // Combat type log probs (masked)
    let combat_logits = output.combat.combat_logits; // [B, 10]
    let cmask_tensor = Tensor::<B, 1>::from_floats(combat_masks.as_slice(), &device)
        .reshape([bs, NUM_COMBAT_TYPES]);
    let masked_logits = combat_logits.clone()
        + (cmask_tensor.clone() - 1.0) * 1e9; // -1e9 where mask is 0
    let combat_log_probs = log_softmax(masked_logits, 1); // [B, 10]

    // Gather log prob of taken combat action
    let combat_targets_tensor = Tensor::<B, 1, Int>::from_ints(
        combat_targets.as_slice(), &device,
    ).reshape([bs, 1]);
    let combat_lp: Tensor<B, 1> = combat_log_probs.gather(1, combat_targets_tensor).squeeze_dim::<1>(1); // [B]

    // Pointer log probs (attack pointer)
    let attack_ptr = output.combat.attack_ptr; // [B, n_tokens]
    // Mask to enemy tokens only (type_id == 1)
    let enemy_mask = ent_mask.clone(); // Simplified: use full entity mask
    let ptr_log_probs = log_softmax(attack_ptr, 1); // [B, n_tokens]
    let pointer_targets_tensor = Tensor::<B, 1, Int>::from_ints(
        pointer_targets.as_slice(), &device,
    ).reshape([bs, 1]);
    // Clamp pointer targets to valid range
    let ptr_lp: Tensor<B, 1> = ptr_log_probs.gather(1, pointer_targets_tensor).squeeze_dim::<1>(1); // [B]

    // Combined combat log prob (combat type + pointer target)
    let log_prob = combat_lp.clone() + ptr_lp.clone();

    // PPO-clipped combat policy loss
    let old_lp_combat_tensor = Tensor::<B, 1>::from_floats(old_lp_combat.as_slice(), device);
    let combat_ratio = (log_prob.clone() - old_lp_combat_tensor).exp();
    let combat_clipped = combat_ratio.clone().clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip);
    let csurr1 = combat_ratio * adv_tensor.clone();
    let csurr2 = combat_clipped * adv_tensor.clone();
    let policy_loss = csurr1.min_pair(csurr2).mean().neg();

    // --- Value loss ---
    let value_pred: Tensor<B, 1> = value_out.attrition.squeeze_dim::<1>(1); // [B]
    let vt_tensor = Tensor::<B, 1>::from_floats(value_targets.as_slice(), &device);
    let value_loss = (value_pred.clone() - vt_tensor.clone()).powf_scalar(2.0).mean() * 0.5;

    // --- Entropy bonus ---
    let combat_probs = softmax(combat_logits * cmask_tensor, 1);
    let combat_entropy = (combat_probs.clone() * combat_probs.clone().log().neg())
        .sum_dim(1).mean();

    // --- Total loss ---
    let total_loss = policy_loss.clone()
        + move_loss.clone()
        + value_loss.clone() * config.value_coef
        - combat_entropy.clone() * config.entropy_coef
        - move_entropy.clone().mean() * config.entropy_coef;

    // Metrics (detached scalars)
    let metrics = TrainMetrics {
        total_loss: total_loss.clone().into_scalar().elem::<f32>(),
        policy_loss: policy_loss.into_scalar().elem::<f32>(),
        move_loss: move_loss.into_scalar().elem::<f32>(),
        value_loss: value_loss.into_scalar().elem::<f32>(),
        entropy: combat_entropy.into_scalar().elem::<f32>(),
        move_std: std_expanded.mean().into_scalar().elem::<f32>(),
        mean_advantage: adv_tensor.mean().into_scalar().elem::<f32>(),
    };

    // Reshape to 1D for backward
    let loss_1d: Tensor<B, 1> = total_loss.unsqueeze();

    (loss_1d, metrics)
}

/// Training metrics from one step.
#[derive(Clone, Debug, Default)]
pub struct TrainMetrics {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub move_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    /// Current σ of movement Gaussian (learned)
    pub move_std: f32,
    pub mean_advantage: f32,
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

/// Execute one training step: forward → loss → backward → optimizer update.
///
/// Returns the updated model and metrics.
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
    // Sample tail drop count for latent interface
    let j_range = config.tail_drop_max - config.tail_drop_min + 1;
    let j_idx = (lcg(rng) as usize) % j_range;
    let n_latents = config.tail_drop_min + j_idx;

    // Forward + loss
    let (loss, metrics) = compute_loss(&model, samples, config, n_latents, device);

    // Backward
    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &model);

    // Optimizer step
    let model = optimizer.step(config.lr, model, grads_params);

    (model, metrics)
}

/// Simple LCG for tail drop sampling (no external dependency).
fn lcg(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

// ---------------------------------------------------------------------------
// Value prediction (no-grad) for V-trace
// ---------------------------------------------------------------------------

/// Predict V(s) for a batch of training samples using the value head.
///
/// Runs the model in no-grad mode (inner backend) to get value predictions
/// without building a computation graph. Used for V-trace target computation.
pub fn predict_values<B: Backend>(
    model: &ActorCriticV6<B>,
    samples: &[TrainingSample],
    device: &B::Device,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let bs = samples.len();
    let max_ent = samples.iter().map(|s| s.entities.len()).max().unwrap_or(1).max(1);
    let max_zones = samples.iter().map(|s| s.zones.len()).max().unwrap_or(1).max(1);

    let mut ent_data = vec![0.0f32; bs * max_ent * ENTITY_DIM];
    let mut etype_data = vec![0.0f32; bs * max_ent];
    let mut emask_data = vec![0.0f32; bs * max_ent];
    let mut zone_data = vec![0.0f32; bs * max_zones * ZONE_DIM];
    let mut zmask_data = vec![0.0f32; bs * max_zones];
    let mut agg_data = vec![0.0f32; bs * AGG_DIM];

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
    }

    let ent_feat = Tensor::<B, 1>::from_floats(ent_data.as_slice(), device)
        .reshape([bs, max_ent, ENTITY_DIM]);
    let ent_types = Tensor::<B, 1>::from_floats(etype_data.as_slice(), device)
        .reshape([bs, max_ent]).int();
    let ent_mask = Tensor::<B, 1>::from_floats(emask_data.as_slice(), device)
        .reshape([bs, max_ent]).greater_elem(0.5);
    let zone_feat = Tensor::<B, 1>::from_floats(zone_data.as_slice(), device)
        .reshape([bs, max_zones, ZONE_DIM]);
    let zone_mask = Tensor::<B, 1>::from_floats(zmask_data.as_slice(), device)
        .reshape([bs, max_zones]).greater_elem(0.5);
    let agg_feat = Tensor::<B, 1>::from_floats(agg_data.as_slice(), device)
        .reshape([bs, AGG_DIM]);

    let ability_cls: Vec<Option<Tensor<B, 2>>> = vec![None; MAX_ABILITIES];

    let (_output, _h_new, value_out) = model.forward(
        ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
        &ability_cls, Some(agg_feat), None,
        None, None, None,
    );

    let val_data = value_out.attrition.squeeze_dim::<1>(1).to_data();
    val_data.as_slice::<f32>().unwrap().to_vec()
}

// Training data preparation is done in the CLI command (xtask impala-train-v6)
// which has access to the RlEpisode/RlStep types.
// See src/bin/xtask/oracle_cmd/transformer_rl.rs for RlStep/RlEpisode types.
