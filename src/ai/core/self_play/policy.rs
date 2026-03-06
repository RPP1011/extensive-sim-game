//! Policy weights (MLP), softmax, trajectory types.

use serde::{Deserialize, Serialize};

use super::NUM_ACTIONS;

// ---------------------------------------------------------------------------
// Policy weights (simple MLP, variable-depth)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyWeights {
    pub layers: Vec<LayerWeights>,
    /// Per-feature scale for input normalization. If set, features are divided by these values.
    #[serde(default)]
    pub input_scale: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub w: Vec<Vec<f32>>, // [in_dim][out_dim]
    pub b: Vec<f32>,       // [out_dim]
}

impl PolicyWeights {
    /// Forward pass → raw logits (no softmax).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut activations = input.to_vec();
        // Apply input scaling if set (divide by scale to normalize)
        if !self.input_scale.is_empty() {
            for (i, a) in activations.iter_mut().enumerate() {
                if let Some(&s) = self.input_scale.get(i) {
                    if s > 1e-6 { *a /= s; }
                }
            }
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let is_last = i == self.layers.len() - 1;
            let out_dim = layer.b.len();
            let mut next = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let mut sum = layer.b[j];
                for (k, &x) in activations.iter().enumerate() {
                    sum += x * layer.w[k][j];
                }
                next[j] = if is_last { sum } else { sum.max(0.0) };
            }
            activations = next;
        }
        activations
    }

    /// Sample an action using masked softmax with temperature.
    pub fn sample_action(
        &self,
        features: &[f32],
        mask: &[bool; NUM_ACTIONS],
        temperature: f32,
        rng: &mut u64,
    ) -> (usize, f32) {
        let logits = self.forward(features);
        let probs = masked_softmax(&logits, mask, temperature);

        // Sample from distribution
        let r = lcg_f32(rng);
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return (i, probs[i]);
            }
        }
        // Fallback: last valid action
        for i in (0..NUM_ACTIONS).rev() {
            if mask[i] {
                return (i, probs[i]);
            }
        }
        (NUM_ACTIONS - 1, 1.0) // Hold
    }

    /// Greedy action (highest probability after masking).
    pub fn greedy_action(&self, features: &[f32], mask: &[bool; NUM_ACTIONS]) -> usize {
        let logits = self.forward(features);
        let mut best_idx = NUM_ACTIONS - 1; // Hold
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if mask[i] && v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }
}

pub fn masked_softmax(logits: &[f32], mask: &[bool; NUM_ACTIONS], temperature: f32) -> Vec<f32> {
    let mut probs = vec![0.0f32; NUM_ACTIONS];
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if mask[i] {
            let scaled = v / temperature.max(0.01);
            if scaled > max_val { max_val = scaled; }
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if mask[i] {
            let e = ((v / temperature.max(0.01)) - max_val).exp();
            probs[i] = e;
            sum += e;
        }
    }
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        // Uniform over valid actions
        let valid = mask.iter().filter(|&&m| m).count() as f32;
        for (i, p) in probs.iter_mut().enumerate() {
            *p = if mask[i] { 1.0 / valid } else { 0.0 };
        }
    }
    probs
}

/// Simple LCG for fast random floats in [0, 1).
pub fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

// ---------------------------------------------------------------------------
// Trajectory recording
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub unit_id: u32,
    pub features: Vec<f32>,
    pub action: usize,
    pub log_prob: f32,
    pub mask: Vec<bool>,
    pub step_reward: f32, // per-step reward (HP delta)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub scenario: String,
    pub outcome: String,   // "Victory" | "Defeat" | "Timeout"
    pub reward: f32,       // final: +1 win, -1 loss, shaped on timeout
    pub ticks: u64,
    pub steps: Vec<Step>,
}
