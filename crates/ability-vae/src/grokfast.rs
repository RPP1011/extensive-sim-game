//! GrokfastEMA gradient filter for burn.
//!
//! Implements the exponential moving average gradient filter from Lee et al.
//! that accelerates grokking by amplifying slow-varying gradient components.
//!
//! After backward pass: filtered_grad = grad + lamb * ema
//! where ema = alpha * ema + (1 - alpha) * grad

use burn::module::{AutodiffModule, ParamId, list_param_ids};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use std::collections::HashMap;

/// GrokfastEMA gradient filter.
pub struct GrokfastEma<B: Backend> {
    alpha: f32,
    lamb: f32,
    // EMA state per parameter, stored as tensors on device
    ema_state: HashMap<ParamId, Tensor<B, 1>>,
}

impl<B: Backend> GrokfastEma<B> {
    pub fn new(alpha: f32, lamb: f32) -> Self {
        Self {
            alpha,
            lamb,
            ema_state: HashMap::new(),
        }
    }

    /// Apply Grokfast EMA filter to all gradients.
    ///
    /// For each parameter:
    ///   ema = alpha * ema + (1 - alpha) * grad
    ///   filtered_grad = grad + lamb * ema
    pub fn apply<AB: AutodiffBackend<InnerBackend = B>, M: AutodiffModule<AB>>(
        &mut self,
        mut grads: GradientsParams,
        model: &M,
    ) -> GradientsParams {
        let param_ids = list_param_ids(model);

        for id in param_ids {
            // Try each tensor dimension, flatten to 1D for EMA, reshape back.
            // Order: try 1D first (biases), then 2D (weights), then higher.
            if let Some(grad) = grads.remove::<B, 1>(id.clone()) {
                let filtered = self.filter_tensor(id.clone(), grad);
                grads.register::<B, 1>(id, filtered);
            } else if let Some(grad) = grads.remove::<B, 2>(id.clone()) {
                let [d0, d1] = grad.dims();
                let flat = grad.reshape([d0 * d1]);
                let filtered = self.filter_tensor(id.clone(), flat);
                grads.register::<B, 2>(id, filtered.reshape([d0, d1]));
            } else if let Some(grad) = grads.remove::<B, 3>(id.clone()) {
                let [d0, d1, d2] = grad.dims();
                let flat = grad.reshape([d0 * d1 * d2]);
                let filtered = self.filter_tensor(id.clone(), flat);
                grads.register::<B, 3>(id, filtered.reshape([d0, d1, d2]));
            }
        }

        grads
    }

    /// Filter a single flattened gradient tensor.
    fn filter_tensor(&mut self, id: ParamId, grad: Tensor<B, 1>) -> Tensor<B, 1> {
        if let Some(ema) = self.ema_state.get(&id) {
            // Update EMA: ema = alpha * ema + (1 - alpha) * grad
            let new_ema = ema.clone() * self.alpha + grad.clone() * (1.0 - self.alpha);
            // Filtered: grad + lamb * ema
            let filtered = grad + new_ema.clone() * self.lamb;
            self.ema_state.insert(id, new_ema);
            filtered
        } else {
            // Initialize EMA to grad * (1 - alpha)
            let ema = grad.clone() * (1.0 - self.alpha);
            let filtered = grad + ema.clone() * self.lamb;
            self.ema_state.insert(id, ema);
            filtered
        }
    }
}
