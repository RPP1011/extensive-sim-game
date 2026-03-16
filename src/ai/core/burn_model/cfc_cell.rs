//! Closed-form Continuous-time (CfC) temporal cell.
//!
//! Replaces GRU for temporal context. Input-dependent time constants
//! allow the cell to learn different integration speeds for different
//! game states.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

#[derive(Module, Debug)]
pub struct CfCCell<B: Backend> {
    f_gate: Linear<B>,  // (d_model + h_dim) -> h_dim
    h_gate: Linear<B>,  // (d_model + h_dim) -> h_dim
    t_a: Linear<B>,     // (d_model + h_dim) -> h_dim
    t_b: Linear<B>,     // (d_model + h_dim) -> h_dim
    proj: Linear<B>,    // h_dim -> d_model
    h_dim: usize,
}

#[derive(Config, Debug)]
pub struct CfCCellConfig {
    pub d_model: usize,
    #[config(default = "64")]
    pub h_dim: usize,
}

impl CfCCellConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CfCCell<B> {
        let input_dim = self.d_model + self.h_dim;
        CfCCell {
            f_gate: LinearConfig::new(input_dim, self.h_dim).init(device),
            h_gate: LinearConfig::new(input_dim, self.h_dim).init(device),
            t_a: LinearConfig::new(input_dim, self.h_dim).init(device),
            t_b: LinearConfig::new(input_dim, self.h_dim).init(device),
            proj: LinearConfig::new(self.h_dim, self.d_model).init(device),
            h_dim: self.h_dim,
        }
    }
}

impl<B: Backend> CfCCell<B> {
    pub fn h_dim(&self) -> usize {
        self.h_dim
    }

    /// CfC forward: (x, h_prev, delta_t) -> (output, h_new)
    ///
    /// x: [B, d_model], h_prev: [B, h_dim], delta_t: scalar
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        h_prev: Tensor<B, 2>,
        delta_t: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // combined = cat([x, h_prev])
        let combined = Tensor::cat(vec![x, h_prev.clone()], 1);

        // f = sigmoid(f_gate(combined))
        let f = sigmoid(self.f_gate.forward(combined.clone()));

        // candidate = tanh(h_gate(combined))
        let candidate = self.h_gate.forward(combined.clone()).tanh();

        // t = sigmoid(t_a(combined)) * delta_t + t_b(combined)
        let t_a_out = sigmoid(self.t_a.forward(combined.clone()));
        let t_b_out = self.t_b.forward(combined);
        let t = t_a_out * delta_t + t_b_out;

        // h_new = tanh(f * h_prev + (1 - f) * candidate * t)
        let ones = f.clone().ones_like();
        let h_new = (f.clone() * h_prev + (ones - f) * candidate * t).tanh();

        // output = proj(h_new)
        let output = self.proj.forward(h_new.clone());

        (output, h_new)
    }
}
