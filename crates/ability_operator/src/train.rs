//! Training loop for the ability latent operator.

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::prelude::*;

use crate::data::OperatorDataset;
use crate::grokfast::GrokfastEma;
use crate::loss::{self, LossMask, TargetDeltas};
use crate::model::*;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub max_steps: usize,
    pub eval_every: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub seed: u64,
    pub grokfast_alpha: f32,
    pub grokfast_lamb: f32,
    pub max_grad_norm: f32,
    pub weight_decay: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            max_steps: 200_000,
            eval_every: 5_000,
            batch_size: 1024,
            lr: 1e-3,
            seed: 42,
            grokfast_alpha: 0.98,
            grokfast_lamb: 2.0,
            max_grad_norm: 1.0,
            weight_decay: 1.0,
        }
    }
}

/// Evaluation metrics.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    pub loss: f32,
    pub hp_mae: f32,
    pub hp_mae_baseline: f32,
    pub hp_mae_improvement_pct: f32,
    pub exists_bce: f32,
    pub exists_bce_baseline: f32,
    pub exists_bce_improvement_pct: f32,
    pub pos_mae: f32,
    pub pos_mae_baseline: f32,
    pub pos_mae_improvement_pct: f32,
}

/// Select a batch of indices from the dataset tensors.
fn select_batch<B: Backend>(
    dataset: &OperatorDataset<B>,
    indices: &[usize],
) -> BatchData<B> {
    let device = dataset.entity_features.device();
    let idx: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let idx_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::from(idx.as_slice()),
        &device,
    );

    let select_3d = |t: &Tensor<B, 3>| -> Tensor<B, 3> {
        t.clone().select(0, idx_tensor.clone())
    };
    let select_2d_int = |t: &Tensor<B, 2, Int>| -> Tensor<B, 2, Int> {
        t.clone().select(0, idx_tensor.clone())
    };
    let select_2d_bool = |t: &Tensor<B, 2, Bool>| -> Tensor<B, 2, Bool> {
        t.clone().select(0, idx_tensor.clone())
    };
    let select_2d = |t: &Tensor<B, 2>| -> Tensor<B, 2> {
        t.clone().select(0, idx_tensor.clone())
    };
    let select_1d = |t: &Tensor<B, 1>| -> Tensor<B, 1> {
        t.clone().select(0, idx_tensor.clone())
    };
    let select_1d_int = |t: &Tensor<B, 1, Int>| -> Tensor<B, 1, Int> {
        t.clone().select(0, idx_tensor.clone())
    };

    let masks: Vec<LossMask> = indices.iter().map(|&i| dataset.loss_masks[i].clone()).collect();

    BatchData {
        entity_features: select_3d(&dataset.entity_features),
        entity_types: select_2d_int(&dataset.entity_types),
        entity_mask: select_2d_bool(&dataset.entity_mask),
        threat_features: select_3d(&dataset.threat_features),
        threat_mask: select_2d_bool(&dataset.threat_mask),
        position_features: select_3d(&dataset.position_features),
        position_mask: select_2d_bool(&dataset.position_mask),
        ability_features: select_3d(&dataset.ability_features),
        ability_types: select_2d_int(&dataset.ability_types),
        ability_mask: select_2d_bool(&dataset.ability_mask),
        ability_cls: select_2d(&dataset.ability_cls),
        caster_slot: select_1d_int(&dataset.caster_slot),
        duration_norm: select_1d(&dataset.duration_norm),
        target_hp: select_3d(&dataset.target_hp),
        target_cc: select_3d(&dataset.target_cc),
        target_cc_stun: select_3d(&dataset.target_cc_stun),
        target_pos: select_3d(&dataset.target_pos),
        target_exists: select_3d(&dataset.target_exists),
        masks,
    }
}

struct BatchData<B: Backend> {
    entity_features: Tensor<B, 3>,
    entity_types: Tensor<B, 2, Int>,
    entity_mask: Tensor<B, 2, Bool>,
    threat_features: Tensor<B, 3>,
    threat_mask: Tensor<B, 2, Bool>,
    position_features: Tensor<B, 3>,
    position_mask: Tensor<B, 2, Bool>,
    ability_features: Tensor<B, 3>,
    ability_types: Tensor<B, 2, Int>,
    ability_mask: Tensor<B, 2, Bool>,
    ability_cls: Tensor<B, 2>,
    caster_slot: Tensor<B, 1, Int>,
    duration_norm: Tensor<B, 1>,
    target_hp: Tensor<B, 3>,
    target_cc: Tensor<B, 3>,
    target_cc_stun: Tensor<B, 3>,
    target_pos: Tensor<B, 3>,
    target_exists: Tensor<B, 3>,
    masks: Vec<LossMask>,
}

/// Evaluate model on a set of indices — all computation on GPU, single sync at end.
fn evaluate<B: Backend>(
    model: &AbilityLatentOperator<B>,
    dataset: &OperatorDataset<B>,
    indices: &[usize],
    batch_size: usize,
) -> EvalMetrics {
    let device = dataset.entity_features.device();

    // Accumulators as 1-element GPU tensors to avoid per-batch CPU syncs
    let mut total_loss_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut hp_mae_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut hp_baseline_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut exists_bce_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut exists_baseline_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut pos_mae_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut pos_baseline_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut hp_count = 0usize;
    let mut pos_count = 0usize;
    let mut n_batches = 0usize;

    for start in (0..indices.len()).step_by(batch_size) {
        let end = (start + batch_size).min(indices.len());
        let batch_indices = &indices[start..end];
        let batch = select_batch(dataset, batch_indices);
        let bs = batch_indices.len();

        let output = model.forward(
            batch.entity_features,
            batch.entity_types,
            batch.entity_mask,
            batch.threat_features,
            batch.threat_mask,
            batch.position_features,
            batch.position_mask,
            batch.ability_features,
            batch.ability_types,
            batch.ability_mask,
            batch.ability_cls,
            batch.caster_slot,
            batch.duration_norm,
        );

        let targets = TargetDeltas {
            hp: batch.target_hp.clone(),
            cc: batch.target_cc,
            cc_stun: batch.target_cc_stun,
            pos: batch.target_pos.clone(),
            exists: batch.target_exists.clone(),
        };

        let loss_val = loss::compute_loss(&output, &targets, &batch.masks);
        total_loss_acc = total_loss_acc + loss_val;

        // Build per-sample mask tensors for vectorized MAE computation
        let hp_mask_vec: Vec<f32> = batch.masks.iter().map(|m| if m.hp { 1.0 } else { 0.0 }).collect();
        let pos_mask_vec: Vec<f32> = batch.masks.iter().map(|m| if m.pos { 1.0 } else { 0.0 }).collect();

        let hp_mask_count: usize = batch.masks.iter().filter(|m| m.hp).count();
        let pos_mask_count: usize = batch.masks.iter().filter(|m| m.pos).count();

        // HP MAE: compute per-sample MAE, multiply by mask, sum
        if hp_mask_count > 0 {
            // (B, E, 3) -> per-sample mean -> (B,)
            let hp_diff = (output.hp_mean - batch.target_hp.clone()).abs();
            let hp_per_sample = hp_diff.mean_dim(2).mean_dim(1).squeeze::<1>(); // (B,)
            let baseline_per_sample = batch.target_hp.abs().mean_dim(2).mean_dim(1).squeeze::<1>();

            let hp_mask_t = Tensor::<B, 1>::from_data(
                TensorData::from(hp_mask_vec.as_slice()), &device,
            );
            hp_mae_acc = hp_mae_acc + (hp_per_sample * hp_mask_t.clone()).sum();
            hp_baseline_acc = hp_baseline_acc + (baseline_per_sample * hp_mask_t).sum();
            hp_count += hp_mask_count;
        }

        // Position MAE
        if pos_mask_count > 0 {
            let pos_diff = (output.pos_mean - batch.target_pos.clone()).abs();
            let pos_per_sample = pos_diff.mean_dim(2).mean_dim(1).squeeze::<1>();
            let baseline_per_sample = batch.target_pos.abs().mean_dim(2).mean_dim(1).squeeze::<1>();

            let pos_mask_t = Tensor::<B, 1>::from_data(
                TensorData::from(pos_mask_vec.as_slice()), &device,
            );
            pos_mae_acc = pos_mae_acc + (pos_per_sample * pos_mask_t.clone()).sum();
            pos_baseline_acc = pos_baseline_acc + (baseline_per_sample * pos_mask_t).sum();
            pos_count += pos_mask_count;
        }

        // Exists BCE — all samples
        let exists_bce = loss::bce_with_logits(
            output.exists_logits,
            batch.target_exists.clone(),
        );
        exists_bce_acc = exists_bce_acc + exists_bce;

        let baseline_logits = Tensor::<B, 3>::full(batch.target_exists.dims(), 10.0, &device);
        let baseline_bce = loss::bce_with_logits(baseline_logits, batch.target_exists);
        exists_baseline_acc = exists_baseline_acc + baseline_bce;

        n_batches += 1;
    }

    // Single GPU→CPU sync: read all accumulators at once
    let results: Vec<f32> = Tensor::cat(
        vec![
            total_loss_acc,
            hp_mae_acc,
            hp_baseline_acc,
            exists_bce_acc,
            exists_baseline_acc,
            pos_mae_acc,
            pos_baseline_acc,
        ],
        0,
    ).into_data().to_vec::<f32>().unwrap();

    let n = n_batches.max(1) as f32;
    let total_loss = results[0];
    let hp_mae = if hp_count > 0 { results[1] / hp_count as f32 } else { 0.0 };
    let hp_baseline = if hp_count > 0 { results[2] / hp_count as f32 } else { 1.0 };
    let exists_bce = results[3] / n;
    let exists_baseline = results[4] / n;
    let pos_mae = if pos_count > 0 { results[5] / pos_count as f32 } else { 0.0 };
    let pos_baseline = if pos_count > 0 { results[6] / pos_count as f32 } else { 1.0 };

    let hp_imp = if hp_baseline > 1e-8 { (hp_baseline - hp_mae) / hp_baseline * 100.0 } else { 0.0 };
    let exists_imp = if exists_baseline > 1e-8 { (exists_baseline - exists_bce) / exists_baseline * 100.0 } else { 0.0 };
    let pos_imp = if pos_baseline > 1e-8 { (pos_baseline - pos_mae) / pos_baseline * 100.0 } else { 0.0 };

    EvalMetrics {
        loss: total_loss / n,
        hp_mae,
        hp_mae_baseline: hp_baseline,
        hp_mae_improvement_pct: hp_imp,
        exists_bce,
        exists_bce_baseline: exists_baseline,
        exists_bce_improvement_pct: exists_imp,
        pos_mae,
        pos_mae_baseline: pos_baseline,
        pos_mae_improvement_pct: pos_imp,
    }
}

/// Run the training loop.
pub fn train<B: burn::tensor::backend::AutodiffBackend>(
    dataset: &OperatorDataset<B>,
    config: &TrainConfig,
    output_path: &std::path::Path,
    device: &B::Device,
) -> AbilityLatentOperator<B> {
    let (train_idx, val_idx) = dataset.train_val_split();
    eprintln!(
        "Dataset: {} samples ({} train, {} val)",
        dataset.n_samples,
        train_idx.len(),
        val_idx.len(),
    );

    let mut model = AbilityLatentOperator::<B>::new(device);

    let optim_config = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_beta_1(0.9)
        .with_beta_2(0.98)
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)));
    let mut optim = optim_config.init();

    let mut _grokfast = GrokfastEma::new(config.grokfast_alpha, config.grokfast_lamb);

    // Simple LCG for shuffling
    let mut rng = config.seed;
    let mut shuffled_train: Vec<usize> = train_idx.clone();

    let mut best_val_loss = f32::INFINITY;
    let mut step = 0usize;
    // Accumulate loss on GPU, only sync at eval time
    let inner_device = &dataset.entity_features.device();
    let mut train_loss_acc = Tensor::<<B as burn::tensor::backend::AutodiffBackend>::InnerBackend, 1>::zeros([1], inner_device);
    let mut train_loss_count = 0usize;

    while step < config.max_steps {
        // Shuffle training indices
        for i in (1..shuffled_train.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            shuffled_train.swap(i, j);
        }

        for start in (0..shuffled_train.len()).step_by(config.batch_size) {
            if step >= config.max_steps {
                break;
            }

            let end = (start + config.batch_size).min(shuffled_train.len());
            let batch_indices = &shuffled_train[start..end];
            let batch = select_batch(dataset, batch_indices);

            let output = model.forward(
                batch.entity_features,
                batch.entity_types,
                batch.entity_mask,
                batch.threat_features,
                batch.threat_mask,
                batch.position_features,
                batch.position_mask,
                batch.ability_features,
                batch.ability_types,
                batch.ability_mask,
                batch.ability_cls,
                batch.caster_slot,
                batch.duration_norm,
            );

            let targets = TargetDeltas {
                hp: batch.target_hp,
                cc: batch.target_cc,
                cc_stun: batch.target_cc_stun,
                pos: batch.target_pos,
                exists: batch.target_exists,
            };

            let loss = loss::compute_loss(&output, &targets, &batch.masks);

            // Track loss on GPU (no CPU sync)
            train_loss_acc = train_loss_acc + loss.clone().inner();
            train_loss_count += 1;

            // Backward + optimize
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);

            step += 1;

            // Eval
            if step % config.eval_every == 0 {
                // Single CPU sync for training loss
                let avg_train_loss: f32 = train_loss_acc.into_data().to_vec::<f32>().unwrap()[0]
                    / train_loss_count.max(1) as f32;
                train_loss_acc = Tensor::<<B as burn::tensor::backend::AutodiffBackend>::InnerBackend, 1>::zeros([1], inner_device);
                train_loss_count = 0;

                // Eval on inner (non-autodiff) model
                let inner_model = model.valid();
                let metrics = evaluate(&inner_model, &dataset.to_inner(), &val_idx, config.batch_size);

                eprintln!(
                    "step {step:>6} | train_loss {avg_train_loss:.4} | val_loss {:.4} | \
                     hp_mae_imp {:.1}% | exists_bce_imp {:.1}% | pos_mae_imp {:.1}%",
                    metrics.loss,
                    metrics.hp_mae_improvement_pct,
                    metrics.exists_bce_improvement_pct,
                    metrics.pos_mae_improvement_pct,
                );

                if metrics.loss < best_val_loss {
                    best_val_loss = metrics.loss;
                    eprintln!("  -> new best val_loss {best_val_loss:.4} at step {step}");
                    model
                        .clone()
                        .save_file(output_path, &burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new())
                        .unwrap_or_else(|e| eprintln!("  warning: checkpoint save failed: {e}"));
                }
            }
        }
    }

    model
}
