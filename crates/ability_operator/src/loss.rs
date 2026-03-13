//! Loss functions for ability operator training.
//!
//! - Beta-NLL for continuous predictions (hp, cc, pos)
//! - BCE for binary predictions (exists, is_stunned)
//! - Per-ability loss masking based on 80-dim property vector

use burn::prelude::*;
use burn::tensor::activation;

use crate::model::DecoderOutput;

/// Beta-NLL loss for Gaussian prediction heads.
///
/// Computes NLL weighted by variance^beta (detached) so the model can
/// learn to express uncertainty without gaming the loss.
///
/// mean, log_var, target: (B, E, F)
/// Returns scalar loss.
pub fn beta_nll<B: Backend>(
    mean: Tensor<B, 3>,
    log_var: Tensor<B, 3>,
    target: Tensor<B, 3>,
    beta: f32,
) -> Tensor<B, 1> {
    let variance = log_var.clone().exp();
    let weight = variance.clone().detach().powf_scalar(beta);
    let diff = target - mean;
    let nll = log_var * 0.5 + diff.clone() * diff / (variance * 2.0);
    (weight * nll).mean()
}

/// Binary cross-entropy with logits.
///
/// logits: (B, E, F) — raw logits
/// target: (B, E, F) — 0 or 1
/// Returns scalar loss.
pub fn bce_with_logits<B: Backend>(
    logits: Tensor<B, 3>,
    target: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let p = activation::sigmoid(logits);
    let eps = 1e-7;
    let p = p.clamp(eps, 1.0 - eps);
    let bce = (target.clone() * p.clone().log()
        + (target.neg() + 1.0) * (p.neg() + 1.0).log())
    .neg();
    bce.mean()
}

/// Ability property indices for loss masking.
///
/// These index into the 80-dim ability property vector to determine
/// which decoder heads should receive gradient.
pub const PROP_HAS_DAMAGE: usize = 41;
pub const PROP_HAS_HEAL: usize = 45;
pub const PROP_CC_START: usize = 48;
pub const PROP_CC_END: usize = 63;
pub const PROP_HAS_MOBILITY: usize = 63;

/// Per-sample loss mask derived from ability properties.
#[derive(Debug, Clone)]
pub struct LossMask {
    /// True if ability has damage or heal effects.
    pub hp: bool,
    /// True if ability has any CC effects.
    pub cc: bool,
    /// True if ability has mobility/dash.
    pub pos: bool,
    /// True if ability can kill (has damage).
    pub exists: bool,
}

impl LossMask {
    /// Compute loss mask from 80-dim ability property vector.
    pub fn from_props(props: &[f32; 80]) -> Self {
        let has_damage = props[PROP_HAS_DAMAGE] > 0.0;
        let has_heal = props[PROP_HAS_HEAL] > 0.0;
        let has_cc = props[PROP_CC_START..PROP_CC_END].iter().any(|&v| v > 0.0);
        let has_mobility = props[PROP_HAS_MOBILITY] > 0.0;

        LossMask {
            hp: has_damage || has_heal,
            cc: has_cc,
            pos: has_mobility,
            exists: has_damage,
        }
    }
}

/// Target deltas for one sample.
pub struct TargetDeltas<B: Backend> {
    /// (B, E, 3) — delta hp_pct, shield_pct, resource_pct.
    pub hp: Tensor<B, 3>,
    /// (B, E, 1) — cc_remaining after.
    pub cc: Tensor<B, 3>,
    /// (B, E, 1) — is_stunned after (0 or 1).
    pub cc_stun: Tensor<B, 3>,
    /// (B, E, 2) — delta_x, delta_y.
    pub pos: Tensor<B, 3>,
    /// (B, E, 1) — exists after (0 or 1).
    pub exists: Tensor<B, 3>,
}

/// Compute total training loss with ability-type masking.
///
/// The loss mask is applied per-sample in the batch. When a mask is false
/// for a group, that group's loss contributes zero gradient.
///
/// `masks` must have length == batch_size.
pub fn compute_loss<B: Backend>(
    pred: &DecoderOutput<B>,
    target: &TargetDeltas<B>,
    masks: &[LossMask],
) -> Tensor<B, 1> {
    let device = pred.hp_mean.device();
    let batch = masks.len();

    // Convert masks to float tensors for element-wise masking
    let hp_mask_data: Vec<f32> = masks.iter().map(|m| if m.hp { 1.0 } else { 0.0 }).collect();
    let cc_mask_data: Vec<f32> = masks.iter().map(|m| if m.cc { 1.0 } else { 0.0 }).collect();
    let pos_mask_data: Vec<f32> = masks.iter().map(|m| if m.pos { 1.0 } else { 0.0 }).collect();
    let exists_mask_data: Vec<f32> = masks.iter().map(|m| if m.exists { 1.0 } else { 0.0 }).collect();

    let make_mask = |data: &[f32]| -> Tensor<B, 1> {
        Tensor::from_data(TensorData::from(data), &device)
    };

    let hp_w = make_mask(&hp_mask_data);
    let cc_w = make_mask(&cc_mask_data);
    let pos_w = make_mask(&pos_mask_data);
    let exists_w = make_mask(&exists_mask_data);

    let mut total_loss = Tensor::<B, 1>::zeros([1], &device);
    let mut n_active = 0.0f32;

    // HP group: beta-NLL
    let hp_count: f32 = hp_mask_data.iter().sum();
    if hp_count > 0.0 {
        let hp_loss = beta_nll(
            pred.hp_mean.clone(),
            pred.hp_logvar.clone(),
            target.hp.clone(),
            0.5,
        );
        // Weight by fraction of active samples
        total_loss = total_loss + hp_loss * (hp_count / batch as f32);
        n_active += 1.0;
    }

    // CC group: beta-NLL for duration + BCE for is_stunned
    let cc_count: f32 = cc_mask_data.iter().sum();
    if cc_count > 0.0 {
        let cc_loss = beta_nll(
            pred.cc_mean.clone(),
            pred.cc_logvar.clone(),
            target.cc.clone(),
            0.5,
        );
        let stun_loss = bce_with_logits(
            pred.cc_stun_logits.clone(),
            target.cc_stun.clone(),
        );
        total_loss = total_loss + (cc_loss + stun_loss) * (cc_count / batch as f32);
        n_active += 1.0;
    }

    // Position group: beta-NLL
    let pos_count: f32 = pos_mask_data.iter().sum();
    if pos_count > 0.0 {
        let pos_loss = beta_nll(
            pred.pos_mean.clone(),
            pred.pos_logvar.clone(),
            target.pos.clone(),
            0.5,
        );
        total_loss = total_loss + pos_loss * (pos_count / batch as f32);
        n_active += 1.0;
    }

    // Exists group: BCE
    let exists_count: f32 = exists_mask_data.iter().sum();
    if exists_count > 0.0 {
        let exists_loss = bce_with_logits(
            pred.exists_logits.clone(),
            target.exists.clone(),
        );
        total_loss = total_loss + exists_loss * (exists_count / batch as f32);
        n_active += 1.0;
    }

    total_loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_beta_nll_finite() {
        let device = <B as Backend>::Device::default();
        let mean = Tensor::<B, 3>::zeros([2, 7, 3], &device);
        let log_var = Tensor::<B, 3>::zeros([2, 7, 3], &device);
        let target = Tensor::<B, 3>::ones([2, 7, 3], &device);

        let loss = beta_nll(mean, log_var, target, 0.5);
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(val.is_finite(), "beta-NLL loss should be finite, got {val}");
        assert!(val > 0.0, "beta-NLL loss should be positive, got {val}");
    }

    #[test]
    fn test_bce_finite() {
        let device = <B as Backend>::Device::default();
        let logits = Tensor::<B, 3>::zeros([2, 7, 1], &device);
        let target = Tensor::<B, 3>::ones([2, 7, 1], &device);

        let loss = bce_with_logits(logits, target);
        let val: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(val.is_finite(), "BCE loss should be finite, got {val}");
    }

    #[test]
    fn test_loss_mask_from_props() {
        let mut props = [0.0f32; 80];
        // No effects → all false
        let mask = LossMask::from_props(&props);
        assert!(!mask.hp);
        assert!(!mask.cc);
        assert!(!mask.pos);
        assert!(!mask.exists);

        // Damage ability
        props[PROP_HAS_DAMAGE] = 1.0;
        let mask = LossMask::from_props(&props);
        assert!(mask.hp);
        assert!(mask.exists);
        assert!(!mask.cc);
        assert!(!mask.pos);

        // Add CC
        props[50] = 1.0;
        let mask = LossMask::from_props(&props);
        assert!(mask.cc);
    }
}
