//! Conditional Flow Matching on the grammar-defined ability space.
//!
//! Instead of diffusion's noise prediction, flow matching learns a velocity
//! field v(x_t, t) that transports samples from noise (t=0) to data (t=1)
//! along optimal transport paths.
//!
//! Training:
//!   - Sample x_0 ~ data, x_1 ~ N(0,1)
//!   - t ~ U(0,1)
//!   - x_t = (1-t) * x_0 + t * x_1   (linear interpolation)
//!   - Target velocity: u_t = x_1 - x_0  (straight path)
//!   - Loss: ||v_θ(x_t, t) - u_t||²
//!
//! Sampling:
//!   - Start from x_0 ~ N(0,1)
//!   - Integrate: x_{i+1} = x_i + v_θ(x_i, t_i) * dt
//!   - Clamp to [0,1] at the end
//!
//! This is simpler, faster, and more stable than DDPM.

use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::prelude::*;
use burn::tensor::activation;

use super::grammar_space::GRAMMAR_DIM;

const HIDDEN_DIM: usize = 512;
const TIME_EMB_DIM: usize = 128;
const NUM_LABELS: usize = 10;

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// ---------------------------------------------------------------------------

fn sinusoidal_embedding<B: Backend>(
    t: Tensor<B, 1>,  // [B] floats in [0, 1]
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [batch] = t.dims();
    let half = dim / 2;

    let mut freq_data = vec![0.0f32; half];
    for i in 0..half {
        freq_data[i] = (-(10000.0f32.ln()) * i as f32 / half as f32).exp();
    }
    let freqs = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(freq_data, [half]),
        device,
    );

    // t * 1000 to spread the embedding (t is in [0,1], needs scaling)
    let t_scaled = t * 1000.0;
    let t_expanded = t_scaled.unsqueeze_dim::<2>(1); // [B, 1]
    let f_expanded = freqs.unsqueeze::<2>(); // [1, half]
    let angles = t_expanded * f_expanded; // [B, half]

    let sin = angles.clone().sin();
    let cos = angles.cos();
    Tensor::cat(vec![sin, cos], 1)
}

// ---------------------------------------------------------------------------
// Velocity network: predicts v(x_t, t, label)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct FlowModel<B: Backend> {
    time_proj: Linear<B>,
    label_emb: Linear<B>,

    fc1: Linear<B>,
    norm1: LayerNorm<B>,
    fc2: Linear<B>,
    norm2: LayerNorm<B>,
    fc3: Linear<B>,
    norm3: LayerNorm<B>,
    fc_out: Linear<B>,
}

impl<B: Backend> FlowModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_dim = GRAMMAR_DIM + TIME_EMB_DIM + NUM_LABELS;
        Self {
            time_proj: LinearConfig::new(TIME_EMB_DIM, TIME_EMB_DIM).init(device),
            label_emb: LinearConfig::new(NUM_LABELS, NUM_LABELS).init(device),
            fc1: LinearConfig::new(input_dim, HIDDEN_DIM).init(device),
            norm1: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc2: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm2: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc3: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm3: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc_out: LinearConfig::new(HIDDEN_DIM, GRAMMAR_DIM).init(device),
        }
    }

    /// Predict velocity v(x_t, t, label).
    pub fn forward(
        &self,
        x_t: Tensor<B, 2>,     // [B, GRAMMAR_DIM]
        t: Tensor<B, 1>,       // [B] in [0, 1]
        labels: Tensor<B, 2>,  // [B, NUM_LABELS]
    ) -> Tensor<B, 2> {
        let device = x_t.device();
        let time_emb = sinusoidal_embedding(t, TIME_EMB_DIM, &device);
        let time_emb = activation::gelu(self.time_proj.forward(time_emb));
        let label_emb = activation::gelu(self.label_emb.forward(labels));

        let h = Tensor::cat(vec![x_t, time_emb, label_emb], 1);
        let h = activation::gelu(self.norm1.forward(self.fc1.forward(h)));
        let h = activation::gelu(self.norm2.forward(self.fc2.forward(h)));
        let h = activation::gelu(self.norm3.forward(self.fc3.forward(h)));
        self.fc_out.forward(h)
    }

    pub fn num_params(&self) -> usize {
        let input_dim = GRAMMAR_DIM + TIME_EMB_DIM + NUM_LABELS;
        TIME_EMB_DIM * TIME_EMB_DIM
        + NUM_LABELS * NUM_LABELS
        + input_dim * HIDDEN_DIM
        + HIDDEN_DIM * HIDDEN_DIM * 2
        + HIDDEN_DIM * GRAMMAR_DIM
        + HIDDEN_DIM * 4
    }

    /// Create a FlowModel that accepts arbitrary conditioning dim (not just NUM_LABELS).
    pub fn new_with_cond_dim(cond_dim: usize, device: &B::Device) -> Self {
        let input_dim = GRAMMAR_DIM + TIME_EMB_DIM + cond_dim;
        Self {
            time_proj: LinearConfig::new(TIME_EMB_DIM, TIME_EMB_DIM).init(device),
            label_emb: LinearConfig::new(cond_dim, cond_dim).init(device),
            fc1: LinearConfig::new(input_dim, HIDDEN_DIM).init(device),
            norm1: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc2: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm2: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc3: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm3: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc_out: LinearConfig::new(HIDDEN_DIM, GRAMMAR_DIM).init(device),
        }
    }

    /// Forward with arbitrary conditioning (text embedding instead of one-hot labels).
    pub fn forward_cond(
        &self,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 2>,  // [B, cond_dim]
    ) -> Tensor<B, 2> {
        let device = x_t.device();
        let time_emb = sinusoidal_embedding(t, TIME_EMB_DIM, &device);
        let time_emb = activation::gelu(self.time_proj.forward(time_emb));
        let cond_emb = activation::gelu(self.label_emb.forward(cond));

        let h = Tensor::cat(vec![x_t, time_emb, cond_emb], 1);
        let h = activation::gelu(self.norm1.forward(self.fc1.forward(h)));
        let h = activation::gelu(self.norm2.forward(self.fc2.forward(h)));
        let h = activation::gelu(self.norm3.forward(self.fc3.forward(h)));
        self.fc_out.forward(h)
    }
}

// ---------------------------------------------------------------------------
// FiLM-conditioned flow model — conditioning modulates each layer
// ---------------------------------------------------------------------------

/// FiLM (Feature-wise Linear Modulation) conditioned flow model.
/// Instead of concatenating the conditioning, it produces per-layer
/// scale (gamma) and shift (beta) that modulate the MLP activations:
///   h = gamma * h + beta
/// This is much more expressive than concatenation.
#[derive(Module, Debug)]
pub struct FiLMFlowModel<B: Backend> {
    time_proj: Linear<B>,

    // Main denoiser MLP (only takes x_t + time as input)
    fc1: Linear<B>,
    norm1: LayerNorm<B>,
    fc2: Linear<B>,
    norm2: LayerNorm<B>,
    fc3: Linear<B>,
    norm3: LayerNorm<B>,
    fc_out: Linear<B>,

    // FiLM generators: conditioning → (gamma, beta) per layer
    film1: Linear<B>,  // cond → 2*HIDDEN (gamma1, beta1)
    film2: Linear<B>,  // cond → 2*HIDDEN (gamma2, beta2)
    film3: Linear<B>,  // cond → 2*HIDDEN (gamma3, beta3)
}

impl<B: Backend> FiLMFlowModel<B> {
    pub fn new(cond_dim: usize, device: &B::Device) -> Self {
        let input_dim = GRAMMAR_DIM + TIME_EMB_DIM;
        Self {
            time_proj: LinearConfig::new(TIME_EMB_DIM, TIME_EMB_DIM).init(device),
            fc1: LinearConfig::new(input_dim, HIDDEN_DIM).init(device),
            norm1: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc2: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm2: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc3: LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device),
            norm3: LayerNormConfig::new(HIDDEN_DIM).init(device),
            fc_out: LinearConfig::new(HIDDEN_DIM, GRAMMAR_DIM).init(device),
            film1: LinearConfig::new(cond_dim, HIDDEN_DIM * 2).init(device),
            film2: LinearConfig::new(cond_dim, HIDDEN_DIM * 2).init(device),
            film3: LinearConfig::new(cond_dim, HIDDEN_DIM * 2).init(device),
        }
    }

    pub fn forward(
        &self,
        x_t: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, _] = x_t.dims();
        let device = x_t.device();

        let time_emb = sinusoidal_embedding(t, TIME_EMB_DIM, &device);
        let time_emb = activation::gelu(self.time_proj.forward(time_emb));

        // Input: just x_t + time (no conditioning concatenated)
        let h = Tensor::cat(vec![x_t, time_emb], 1);

        // Layer 1 + FiLM modulation
        let h = self.norm1.forward(self.fc1.forward(h));
        let film1 = self.film1.forward(cond.clone());
        let gamma1 = film1.clone().slice([0..batch, 0..HIDDEN_DIM]);
        let beta1 = film1.slice([0..batch, HIDDEN_DIM..HIDDEN_DIM * 2]);
        let h = activation::gelu(h * (gamma1 + 1.0) + beta1);

        // Layer 2 + FiLM
        let h = self.norm2.forward(self.fc2.forward(h));
        let film2 = self.film2.forward(cond.clone());
        let gamma2 = film2.clone().slice([0..batch, 0..HIDDEN_DIM]);
        let beta2 = film2.slice([0..batch, HIDDEN_DIM..HIDDEN_DIM * 2]);
        let h = activation::gelu(h * (gamma2 + 1.0) + beta2);

        // Layer 3 + FiLM
        let h = self.norm3.forward(self.fc3.forward(h));
        let film3 = self.film3.forward(cond);
        let gamma3 = film3.clone().slice([0..batch, 0..HIDDEN_DIM]);
        let beta3 = film3.slice([0..batch, HIDDEN_DIM..HIDDEN_DIM * 2]);
        let h = activation::gelu(h * (gamma3 + 1.0) + beta3);

        self.fc_out.forward(h)
    }
}

/// Flow matching loss for FiLM model.
pub fn film_flow_loss<B: Backend>(
    model: &FiLMFlowModel<B>,
    x_data: Tensor<B, 2>,
    cond: Tensor<B, 2>,
    device: &B::Device,
) -> FlowOutput<B> {
    let [batch, dim] = x_data.dims();

    let t = Tensor::<B, 1>::random([batch], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    let x_noise = Tensor::<B, 2>::random([batch, dim], burn::tensor::Distribution::Normal(0.0, 1.0), device);

    let t_expanded = t.clone().unsqueeze_dim::<2>(1);
    let x_t = x_data.clone() * (t_expanded.clone().neg() + 1.0) + x_noise.clone() * t_expanded;
    let target_v = x_noise - x_data;

    let pred_v = model.forward(x_t, t, cond);

    let diff = pred_v - target_v;
    let loss = (diff.clone() * diff).mean();
    let loss_val: f64 = loss.clone().into_scalar().elem();

    FlowOutput { loss: loss.unsqueeze(), loss_val }
}

/// Sample from FiLM flow model.
pub fn film_sample<B: Backend>(
    model: &FiLMFlowModel<B>,
    n_samples: usize,
    cond: Tensor<B, 2>,  // [1, cond_dim] or [n_samples, cond_dim]
    n_steps: usize,
    device: &B::Device,
) -> Vec<[f32; GRAMMAR_DIM]> {
    let mut x = Tensor::<B, 2>::random(
        [n_samples, GRAMMAR_DIM],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let cond = if cond.dims()[0] == 1 {
        cond.expand([n_samples as i64, -1])
    } else {
        cond
    };

    let dt = 1.0 / n_steps as f32;

    for step in 0..n_steps {
        let t_val = 1.0 - step as f32 / n_steps as f32;
        let t = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![t_val; n_samples], [n_samples]),
            device,
        );
        let v = model.forward(x.clone(), t, cond.clone());
        x = x - v * dt;
    }

    x = x.clamp(0.0, 1.0);
    let data: Vec<f32> = x.to_data().to_vec().unwrap();
    let mut results = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in 0..GRAMMAR_DIM { v[d] = data[i * GRAMMAR_DIM + d]; }
        results.push(v);
    }
    results
}

// ---------------------------------------------------------------------------
// Training loss
// ---------------------------------------------------------------------------

pub struct FlowOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub loss_val: f64,
}

/// Conditional flow matching loss.
pub fn flow_loss<B: Backend>(
    model: &FlowModel<B>,
    x_data: Tensor<B, 2>,    // [B, GRAMMAR_DIM] clean data in [0,1]
    labels: Tensor<B, 2>,    // [B, NUM_LABELS]
    cfg_drop_prob: f32,
    device: &B::Device,
) -> FlowOutput<B> {
    let [batch, dim] = x_data.dims();

    // Sample t ~ U(0, 1)
    let t = Tensor::<B, 1>::random(
        [batch],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        device,
    );

    // Sample noise x_1 ~ N(0, 1)
    let x_noise = Tensor::<B, 2>::random(
        [batch, dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    // Interpolate: x_t = (1 - t) * x_data + t * x_noise
    let t_expanded = t.clone().unsqueeze_dim::<2>(1); // [B, 1]
    let x_t = x_data.clone() * (t_expanded.clone().neg() + 1.0) + x_noise.clone() * t_expanded;

    // Target velocity: u = x_noise - x_data (direction from data to noise)
    let target_v = x_noise - x_data;

    // CFG: randomly drop labels
    let labels = if cfg_drop_prob > 0.0 {
        let drop_mask = Tensor::<B, 2>::random(
            [batch, 1],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        ).lower_elem(cfg_drop_prob).float();
        labels * (drop_mask.neg() + 1.0)
    } else {
        labels
    };

    // Predict velocity
    let pred_v = model.forward(x_t, t, labels);

    // MSE loss
    let diff = pred_v - target_v;
    let loss = (diff.clone() * diff).mean();
    let loss_val: f64 = loss.clone().into_scalar().elem();

    FlowOutput {
        loss: loss.unsqueeze(),
        loss_val,
    }
}

// ---------------------------------------------------------------------------
// Sampling: Euler integration from noise to data
// ---------------------------------------------------------------------------

/// Sample abilities by integrating the learned flow from noise to data.
/// `n_steps`: number of Euler steps (20-50 is usually fine).
pub fn sample<B: Backend>(
    model: &FlowModel<B>,
    n_samples: usize,
    labels: Option<Tensor<B, 2>>,
    cfg_scale: f32,
    n_steps: usize,
    device: &B::Device,
) -> Vec<[f32; GRAMMAR_DIM]> {
    // Start from noise (t=1)
    let mut x = Tensor::<B, 2>::random(
        [n_samples, GRAMMAR_DIM],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let uncond_labels = Tensor::<B, 2>::zeros([n_samples, NUM_LABELS], device);
    let cond_labels = labels.unwrap_or_else(|| uncond_labels.clone());

    let dt = 1.0 / n_steps as f32;

    // Integrate from t=1 (noise) to t=0 (data)
    for step in 0..n_steps {
        let t_val = 1.0 - step as f32 / n_steps as f32;
        let t = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![t_val; n_samples], [n_samples]),
            device,
        );

        // Predict velocity (with CFG)
        let v = if cfg_scale > 1.0 {
            let v_cond = model.forward(x.clone(), t.clone(), cond_labels.clone());
            let v_uncond = model.forward(x.clone(), t, uncond_labels.clone());
            v_uncond.clone() + (v_cond - v_uncond) * cfg_scale
        } else {
            model.forward(x.clone(), t, cond_labels.clone())
        };

        // Euler step: x -= v * dt (going from noise toward data)
        x = x - v * dt;
    }

    // Clamp to [0, 1]
    x = x.clamp(0.0, 1.0);

    let data: Vec<f32> = x.to_data().to_vec().unwrap();
    let mut results = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in 0..GRAMMAR_DIM {
            v[d] = data[i * GRAMMAR_DIM + d];
        }
        results.push(v);
    }
    results
}

// Label constants
pub const LABEL_DAMAGE: usize = 0;
pub const LABEL_HEAL: usize = 1;
pub const LABEL_CC: usize = 2;
pub const LABEL_DEFENSE: usize = 3;
pub const LABEL_UTILITY: usize = 4;
pub const LABEL_ECONOMY: usize = 5;
pub const LABEL_DIPLOMACY: usize = 6;
pub const LABEL_STEALTH: usize = 7;
pub const LABEL_LEADERSHIP: usize = 8;

pub fn make_label<B: Backend>(
    label_idx: usize,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data = vec![0.0f32; batch * NUM_LABELS];
    for i in 0..batch {
        if label_idx < NUM_LABELS {
            data[i * NUM_LABELS + label_idx] = 1.0;
        }
    }
    Tensor::from_data(
        burn::tensor::TensorData::new(data, [batch, NUM_LABELS]),
        device,
    )
}
