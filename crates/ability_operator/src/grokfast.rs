//! GrokfastEMA gradient filter.
//!
//! Implements the exponential moving average gradient filter from Lee et al.
//! that accelerates grokking by amplifying slow-varying gradient components.
//!
//! After backward pass: grad = grad + lamb * ema
//! where ema = alpha * ema + (1 - alpha) * grad

use burn::module::Module;
use burn::prelude::*;

/// GrokfastEMA gradient filter state.
///
/// Maintains an EMA of gradients per parameter and applies the filter
/// to amplify generalizing gradient components.
pub struct GrokfastEma {
    /// EMA decay factor (0.98 typical).
    alpha: f32,
    /// Amplification factor (2.0 typical).
    lamb: f32,
    /// EMA state per parameter, stored as flat f32 vectors keyed by param index.
    ema_state: Vec<Vec<f32>>,
    /// Whether the EMA has been initialized.
    initialized: bool,
}

impl GrokfastEma {
    pub fn new(alpha: f32, lamb: f32) -> Self {
        GrokfastEma {
            alpha,
            lamb,
            ema_state: Vec::new(),
            initialized: false,
        }
    }

    /// Apply the Grokfast EMA filter to model gradients.
    ///
    /// This modifies the gradients in-place by:
    /// 1. Updating the EMA: ema = alpha * ema + (1 - alpha) * grad
    /// 2. Replacing grad with: grad + lamb * ema
    ///
    /// Must be called after backward() and before optimizer.step().
    ///
    /// Returns the modified GradientsParams.
    pub fn apply<B: burn::tensor::backend::AutodiffBackend, M: Module<B>>(
        &mut self,
        grads: burn::optim::GradientsParams,
        model: &M,
    ) -> burn::optim::GradientsParams {
        // GrokfastEMA operates on the raw gradient tensors.
        // Since Burn's GradientsParams doesn't expose per-parameter iteration
        // in a way that allows in-place modification easily, we work with
        // the parameter-level API.
        //
        // For now, return grads unmodified — the actual filtering happens
        // at the tensor level in the training loop where we have access to
        // individual parameter gradients.
        //
        // TODO: Implement proper per-parameter EMA when Burn's grad API allows it.
        // The training loop will call apply_to_tensor() per parameter instead.
        grads
    }

    /// Apply GrokfastEMA to a single gradient tensor.
    ///
    /// `param_idx`: unique index for this parameter (for EMA state lookup).
    /// `grad`: the current gradient tensor.
    ///
    /// Returns the filtered gradient: grad + lamb * ema.
    pub fn apply_to_tensor<B: Backend>(
        &mut self,
        param_idx: usize,
        grad: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let grad_data: Vec<f32> = grad.clone().into_data().to_vec::<f32>().unwrap();
        let device = grad.device();

        // Ensure EMA state exists for this parameter
        while self.ema_state.len() <= param_idx {
            self.ema_state.push(Vec::new());
        }

        if self.ema_state[param_idx].is_empty() {
            // Initialize EMA to current gradient
            self.ema_state[param_idx] = grad_data.clone();
        } else {
            // Update EMA: ema = alpha * ema + (1 - alpha) * grad
            let ema = &mut self.ema_state[param_idx];
            for (e, &g) in ema.iter_mut().zip(grad_data.iter()) {
                *e = self.alpha * *e + (1.0 - self.alpha) * g;
            }
        }

        // Filtered gradient: grad + lamb * ema
        let ema_tensor = Tensor::<B, 1>::from_data(
            TensorData::from(self.ema_state[param_idx].as_slice()),
            &device,
        );
        grad + ema_tensor * self.lamb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_grokfast_ema_basic() {
        let device = <B as Backend>::Device::default();
        let mut grokfast = GrokfastEma::new(0.98, 2.0);

        let grad = Tensor::<B, 1>::from_data(
            TensorData::from([1.0f32, 2.0, 3.0].as_slice()),
            &device,
        );

        // First call: EMA initialized to grad, output = grad + lamb * grad = 3 * grad
        let filtered = grokfast.apply_to_tensor(0, grad.clone());
        let vals: Vec<f32> = filtered.into_data().to_vec::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5); // 1 + 2*1 = 3
        assert!((vals[1] - 6.0).abs() < 1e-5); // 2 + 2*2 = 6

        // Second call with same gradient: EMA decays toward grad
        let filtered = grokfast.apply_to_tensor(0, grad);
        let vals: Vec<f32> = filtered.into_data().to_vec::<f32>().unwrap();
        // ema = 0.98*1.0 + 0.02*1.0 = 1.0, filtered = 1.0 + 2*1.0 = 3.0
        assert!((vals[0] - 3.0).abs() < 1e-5);
    }
}
