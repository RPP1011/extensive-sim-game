//! Student MLP — distilled tactical decision model.
//!
//! Tiny 2-layer MLP: 60 inputs → 128 → 64 → 9 outputs (~17K params).
//! No ML framework — just flat arrays and matrix multiplies.
//! AVX2+FMA SIMD acceleration when available (~4x faster).
//!
//! Outputs: 6 personality weights (sigmoid → [0,1]) + 3 formation logits (softmax).
//!
//! Input features (60):
//!   [0..40]  Aggregate stats: HP, cooldowns, roles, personality, formation, game phase
//!   [40..60] Spatial features: distances, engagement, clustering, role compliance, threats

use serde::{Deserialize, Serialize};

use crate::personality::PersonalityProfile;
use crate::squad::FormationMode;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentWeights {
    pub w1: Vec<Vec<f32>>, // [input_dim][hidden1]
    pub b1: Vec<f32>,      // [hidden1]
    pub w2: Vec<Vec<f32>>, // [hidden1][hidden2]
    pub b2: Vec<f32>,      // [hidden2]
    pub w3: Vec<Vec<f32>>, // [hidden2][9]
    pub b3: Vec<f32>,      // [9]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StudentArchitecture {
    input_dim: usize,
    hidden1: usize,
    hidden2: usize,
    output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StudentFile {
    architecture: StudentArchitecture,
    weights: StudentWeights,
}

/// Flat, cache-friendly representation for fast inference.
#[derive(Debug, Clone)]
pub struct StudentMLP {
    pub(super) input_dim: usize,
    pub(super) hidden1: usize,
    pub(super) hidden2: usize,
    // Row-major flat arrays
    pub(super) w1: Vec<f32>,
    pub(super) b1: Vec<f32>,
    pub(super) w2: Vec<f32>,
    pub(super) b2: Vec<f32>,
    pub(super) w3: Vec<f32>,
    pub(super) b3: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StudentOutput {
    pub personality: PersonalityProfile,
    pub formation: FormationMode,
    pub formation_probs: [f32; 3],
}

impl StudentMLP {
    /// Load from the JSON format exported by the Python `StudentMLP.to_dict()`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: StudentFile =
            serde_json::from_str(json_str).map_err(|e| format!("parse error: {e}"))?;
        let a = &file.architecture;
        let w = &file.weights;

        // Flatten 2D weight matrices to row-major 1D
        let w1 = flatten(&w.w1, a.input_dim, a.hidden1)?;
        let w2 = flatten(&w.w2, a.hidden1, a.hidden2)?;
        let w3 = flatten(&w.w3, a.hidden2, a.output_dim)?;

        Ok(Self {
            input_dim: a.input_dim,
            hidden1: a.hidden1,
            hidden2: a.hidden2,
            w1,
            b1: w.b1.clone(),
            w2,
            b2: w.b2.clone(),
            w3,
            b3: w.b3.clone(),
        })
    }

    pub fn param_count(&self) -> usize {
        self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.w3.len()
            + self.b3.len()
    }

    /// Run inference on a pre-computed feature vector. Returns raw output[9].
    /// Dispatches to AVX2+FMA SIMD when available, scalar fallback otherwise.
    #[inline]
    pub fn forward_raw(&self, input: &[f32]) -> [f32; 9] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: feature detection guards the intrinsics.
                return unsafe { self.forward_raw_avx2(input) };
            }
        }
        self.forward_raw_scalar(input)
    }

    /// Scalar fallback for non-AVX2 platforms.
    fn forward_raw_scalar(&self, input: &[f32]) -> [f32; 9] {
        debug_assert_eq!(input.len(), self.input_dim);

        // Layer 1: input × W1 + b1 → ReLU
        let mut h1 = [0.0_f32; 128];
        for j in 0..self.hidden1 {
            let mut sum = self.b1[j];
            let w_offset = j;
            for i in 0..self.input_dim {
                sum += input[i] * self.w1[i * self.hidden1 + w_offset];
            }
            h1[j] = sum.max(0.0);
        }

        // Layer 2: h1 × W2 + b2 → ReLU
        let mut h2 = [0.0_f32; 64];
        for j in 0..self.hidden2 {
            let mut sum = self.b2[j];
            for i in 0..self.hidden1 {
                sum += h1[i] * self.w2[i * self.hidden2 + j];
            }
            h2[j] = sum.max(0.0);
        }

        // Layer 3: h2 × W3 + b3
        let mut out = [0.0_f32; 9];
        for j in 0..9 {
            let mut sum = self.b3[j];
            for i in 0..self.hidden2 {
                sum += h2[i] * self.w3[i * 9 + j];
            }
            out[j] = sum;
        }

        out
    }

    /// AVX2+FMA SIMD forward pass using broadcast-and-accumulate pattern.
    ///
    /// Instead of iterating output neurons and dot-producting each,
    /// we broadcast each input element and FMA across 8 output neurons at once.
    /// Layer 1 (128 outputs) = 16 AVX2 registers, Layer 2 (64) = 8, Layer 3 (9) = 1 + scalar.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_raw_avx2(&self, input: &[f32]) -> [f32; 9] {
        debug_assert_eq!(input.len(), self.input_dim);
        debug_assert_eq!(self.hidden1, 128);
        debug_assert_eq!(self.hidden2, 64);

        // ---- Layer 1: input[60] × W1[60×128] + b1[128] → ReLU ----
        // 16 accumulators × 8 floats = 128 outputs
        let b1_ptr = self.b1.as_ptr();
        let mut acc0  = _mm256_loadu_ps(b1_ptr);
        let mut acc1  = _mm256_loadu_ps(b1_ptr.add(8));
        let mut acc2  = _mm256_loadu_ps(b1_ptr.add(16));
        let mut acc3  = _mm256_loadu_ps(b1_ptr.add(24));
        let mut acc4  = _mm256_loadu_ps(b1_ptr.add(32));
        let mut acc5  = _mm256_loadu_ps(b1_ptr.add(40));
        let mut acc6  = _mm256_loadu_ps(b1_ptr.add(48));
        let mut acc7  = _mm256_loadu_ps(b1_ptr.add(56));
        let mut acc8  = _mm256_loadu_ps(b1_ptr.add(64));
        let mut acc9  = _mm256_loadu_ps(b1_ptr.add(72));
        let mut acc10 = _mm256_loadu_ps(b1_ptr.add(80));
        let mut acc11 = _mm256_loadu_ps(b1_ptr.add(88));
        let mut acc12 = _mm256_loadu_ps(b1_ptr.add(96));
        let mut acc13 = _mm256_loadu_ps(b1_ptr.add(104));
        let mut acc14 = _mm256_loadu_ps(b1_ptr.add(112));
        let mut acc15 = _mm256_loadu_ps(b1_ptr.add(120));

        let w1_ptr = self.w1.as_ptr();
        let in_ptr = input.as_ptr();
        for i in 0..self.input_dim {
            let x = _mm256_set1_ps(*in_ptr.add(i));
            let base = w1_ptr.add(i * 128);
            acc0  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base), acc0);
            acc1  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(8)), acc1);
            acc2  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(16)), acc2);
            acc3  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(24)), acc3);
            acc4  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(32)), acc4);
            acc5  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(40)), acc5);
            acc6  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(48)), acc6);
            acc7  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(56)), acc7);
            acc8  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(64)), acc8);
            acc9  = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(72)), acc9);
            acc10 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(80)), acc10);
            acc11 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(88)), acc11);
            acc12 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(96)), acc12);
            acc13 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(104)), acc13);
            acc14 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(112)), acc14);
            acc15 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(120)), acc15);
        }

        // ReLU and store h1
        let zero = _mm256_setzero_ps();
        let mut h1 = [0.0_f32; 128];
        let h1_ptr = h1.as_mut_ptr();
        _mm256_storeu_ps(h1_ptr,          _mm256_max_ps(zero, acc0));
        _mm256_storeu_ps(h1_ptr.add(8),   _mm256_max_ps(zero, acc1));
        _mm256_storeu_ps(h1_ptr.add(16),  _mm256_max_ps(zero, acc2));
        _mm256_storeu_ps(h1_ptr.add(24),  _mm256_max_ps(zero, acc3));
        _mm256_storeu_ps(h1_ptr.add(32),  _mm256_max_ps(zero, acc4));
        _mm256_storeu_ps(h1_ptr.add(40),  _mm256_max_ps(zero, acc5));
        _mm256_storeu_ps(h1_ptr.add(48),  _mm256_max_ps(zero, acc6));
        _mm256_storeu_ps(h1_ptr.add(56),  _mm256_max_ps(zero, acc7));
        _mm256_storeu_ps(h1_ptr.add(64),  _mm256_max_ps(zero, acc8));
        _mm256_storeu_ps(h1_ptr.add(72),  _mm256_max_ps(zero, acc9));
        _mm256_storeu_ps(h1_ptr.add(80),  _mm256_max_ps(zero, acc10));
        _mm256_storeu_ps(h1_ptr.add(88),  _mm256_max_ps(zero, acc11));
        _mm256_storeu_ps(h1_ptr.add(96),  _mm256_max_ps(zero, acc12));
        _mm256_storeu_ps(h1_ptr.add(104), _mm256_max_ps(zero, acc13));
        _mm256_storeu_ps(h1_ptr.add(112), _mm256_max_ps(zero, acc14));
        _mm256_storeu_ps(h1_ptr.add(120), _mm256_max_ps(zero, acc15));

        // ---- Layer 2: h1[128] × W2[128×64] + b2[64] → ReLU ----
        // 8 accumulators × 8 floats = 64 outputs
        let b2_ptr = self.b2.as_ptr();
        let mut a2_0 = _mm256_loadu_ps(b2_ptr);
        let mut a2_1 = _mm256_loadu_ps(b2_ptr.add(8));
        let mut a2_2 = _mm256_loadu_ps(b2_ptr.add(16));
        let mut a2_3 = _mm256_loadu_ps(b2_ptr.add(24));
        let mut a2_4 = _mm256_loadu_ps(b2_ptr.add(32));
        let mut a2_5 = _mm256_loadu_ps(b2_ptr.add(40));
        let mut a2_6 = _mm256_loadu_ps(b2_ptr.add(48));
        let mut a2_7 = _mm256_loadu_ps(b2_ptr.add(56));

        let w2_ptr = self.w2.as_ptr();
        for i in 0..128 {
            let x = _mm256_set1_ps(*h1_ptr.add(i));
            let base = w2_ptr.add(i * 64);
            a2_0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base), a2_0);
            a2_1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(8)), a2_1);
            a2_2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(16)), a2_2);
            a2_3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(24)), a2_3);
            a2_4 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(32)), a2_4);
            a2_5 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(40)), a2_5);
            a2_6 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(48)), a2_6);
            a2_7 = _mm256_fmadd_ps(x, _mm256_loadu_ps(base.add(56)), a2_7);
        }

        // ReLU and store h2
        let mut h2 = [0.0_f32; 64];
        let h2_ptr = h2.as_mut_ptr();
        _mm256_storeu_ps(h2_ptr,        _mm256_max_ps(zero, a2_0));
        _mm256_storeu_ps(h2_ptr.add(8),  _mm256_max_ps(zero, a2_1));
        _mm256_storeu_ps(h2_ptr.add(16), _mm256_max_ps(zero, a2_2));
        _mm256_storeu_ps(h2_ptr.add(24), _mm256_max_ps(zero, a2_3));
        _mm256_storeu_ps(h2_ptr.add(32), _mm256_max_ps(zero, a2_4));
        _mm256_storeu_ps(h2_ptr.add(40), _mm256_max_ps(zero, a2_5));
        _mm256_storeu_ps(h2_ptr.add(48), _mm256_max_ps(zero, a2_6));
        _mm256_storeu_ps(h2_ptr.add(56), _mm256_max_ps(zero, a2_7));

        // ---- Layer 3: h2[64] × W3[64×9] + b3[9] ----
        // First 8 outputs via SIMD, 9th output scalar.
        let mut a3 = _mm256_loadu_ps(self.b3.as_ptr());
        let w3_ptr = self.w3.as_ptr();

        for i in 0..64 {
            let x = _mm256_set1_ps(*h2_ptr.add(i));
            // Load 8 weights starting at w3[i*9] — outputs 0..7 for input i
            a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(w3_ptr.add(i * 9)), a3);
        }

        let mut out = [0.0_f32; 9];
        _mm256_storeu_ps(out.as_mut_ptr(), a3);

        // 9th output: scalar dot product
        let mut sum8 = self.b3[8];
        for i in 0..64 {
            sum8 += h2[i] * *w3_ptr.add(i * 9 + 8);
        }
        out[8] = sum8;

        out
    }

    /// Run inference and produce typed output.
    pub fn predict(&self, input: &[f32]) -> StudentOutput {
        let out = self.forward_raw(input);

        let personality = PersonalityProfile {
            aggression: sigmoid(out[0]),
            risk_tolerance: sigmoid(out[1]),
            discipline: sigmoid(out[2]),
            control_bias: sigmoid(out[3]),
            altruism: sigmoid(out[4]),
            patience: sigmoid(out[5]),
        };

        let mut formation_probs = [0.0_f32; 3];
        softmax(&out[6..9], &mut formation_probs);

        let formation = match argmax3(formation_probs) {
            0 => FormationMode::Hold,
            1 => FormationMode::Advance,
            _ => FormationMode::Retreat,
        };

        StudentOutput {
            personality,
            formation,
            formation_probs,
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-10.0, 10.0)).exp())
}

#[inline]
pub(super) fn softmax(input: &[f32], output: &mut [f32; 3]) {
    let max = input[0].max(input[1]).max(input[2]);
    let e0 = (input[0] - max).exp();
    let e1 = (input[1] - max).exp();
    let e2 = (input[2] - max).exp();
    let sum = e0 + e1 + e2;
    output[0] = e0 / sum;
    output[1] = e1 / sum;
    output[2] = e2 / sum;
}

#[inline]
pub(super) fn argmax3(v: [f32; 3]) -> usize {
    if v[0] >= v[1] && v[0] >= v[2] {
        0
    } else if v[1] >= v[2] {
        1
    } else {
        2
    }
}

pub(super) fn flatten(mat: &[Vec<f32>], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    if mat.len() != rows {
        return Err(format!("expected {rows} rows, got {}", mat.len()));
    }
    let mut flat = Vec::with_capacity(rows * cols);
    for row in mat {
        if row.len() != cols {
            return Err(format!("expected {cols} cols, got {}", row.len()));
        }
        flat.extend_from_slice(row);
    }
    Ok(flat)
}
