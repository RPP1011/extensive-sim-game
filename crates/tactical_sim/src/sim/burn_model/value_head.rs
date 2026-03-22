//! Value head for V6 curriculum pretraining.
//!
//! Two-headed value prediction:
//! - Attrition ratio: team-level fight outcome quality [0, 1]
//! - Survival ticks: per-unit survival prediction [0, 1]
//!
//! Used during pretraining stages 0a-0d. Not exported for inference
//! (RL uses the value head internally; the inference client only needs
//! action heads).

use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

use super::config::*;

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    attrition_l1: Linear<B>,
    attrition_l2: Linear<B>,
    survival_l1: Linear<B>,
    survival_l2: Linear<B>,
    gelu: Gelu,
}

#[derive(Config, Debug)]
pub struct ValueHeadConfig {
    #[config(default = "D_MODEL")]
    pub d_model: usize,
}

impl ValueHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ValueHead<B> {
        let d = self.d_model;
        ValueHead {
            attrition_l1: LinearConfig::new(d, d).init(device),
            attrition_l2: LinearConfig::new(d, 1).init(device),
            survival_l1: LinearConfig::new(d, d).init(device),
            survival_l2: LinearConfig::new(d, 1).init(device),
            gelu: Gelu::new(),
        }
    }
}

/// Value head output.
pub struct ValueOutput<B: Backend> {
    /// Attrition ratio prediction: [B, 1]
    pub attrition: Tensor<B, 2>,
    /// Survival ticks prediction: [B, 1]
    pub survival: Tensor<B, 2>,
}

impl<B: Backend> ValueHead<B> {
    /// Predict fight value from pooled representation.
    ///
    /// pooled: [B, d] — mean-pooled latent tokens or entity tokens
    pub fn forward(&self, pooled: Tensor<B, 2>) -> ValueOutput<B> {
        let attrition = self.attrition_l2.forward(
            self.gelu.forward(self.attrition_l1.forward(pooled.clone())),
        );
        let survival = self.survival_l2.forward(
            self.gelu.forward(self.survival_l1.forward(pooled)),
        );
        ValueOutput { attrition, survival }
    }
}
