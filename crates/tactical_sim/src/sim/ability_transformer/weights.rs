//! Transformer weight loading and inference.
//!
//! Implements multi-head self-attention, layer norm, and feed-forward layers
//! for frozen inference.  Follows the same patterns as EvalWeights and FlatMLP
//! in the existing codebase.

use serde::Deserialize;

pub(super) use super::weights_math::{dot_product, sigmoid, gelu, softmax_inplace};

// ---------------------------------------------------------------------------
// JSON schema (matches training/export_weights.py output)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct LinearWeights {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct LayerNormWeights {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct TransformerLayerJson {
    pub self_attn_in_proj: LinearWeights,
    pub self_attn_out_proj: LinearWeights,
    pub ff_linear1: LinearWeights,
    pub ff_linear2: LinearWeights,
    pub norm1: LayerNormWeights,
    pub norm2: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
pub(super) struct DecisionHeadJson {
    pub urgency: HeadPairJson,
    pub target: HeadPairJson,
}

#[derive(Debug, Deserialize)]
pub(super) struct HeadPairJson {
    pub linear1: LinearWeights,
    pub linear2: LinearWeights,
}

#[derive(Debug, Deserialize)]
pub(super) struct HeadJson {
    pub linear1: LinearWeights,
    pub linear2: LinearWeights,
}

#[derive(Debug, Deserialize)]
pub(super) struct ActorCriticArchJson {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    #[allow(dead_code)]
    #[serde(default)]
    pub game_state_dim: usize,
    #[serde(default)]
    pub num_base_actions: usize,
    pub pad_id: usize,
    #[allow(dead_code)]
    pub cls_id: usize,
    #[serde(default)]
    pub external_cls_dim: usize,
}

#[derive(Debug, Deserialize)]
pub(super) struct CrossAttnJson {
    pub attn_in_proj: LinearWeights,
    pub attn_out_proj: LinearWeights,
    pub norm_q: LayerNormWeights,
    pub norm_kv: LayerNormWeights,
    pub ff_linear1: LinearWeights,
    pub ff_linear2: LinearWeights,
    pub norm_ff: LayerNormWeights,
}

// ---------------------------------------------------------------------------
// Flattened runtime types (shared across all weight modules)
// ---------------------------------------------------------------------------

pub(super) const ENTITY_DIM: usize = 30;
pub(super) const NUM_ENTITIES: usize = 7;
pub(super) const NUM_ENTITY_TYPES: usize = 3;
/// Entity type IDs: [self, enemy, enemy, enemy, ally, ally, ally]
pub(super) const ENTITY_TYPE_IDS: [usize; NUM_ENTITIES] = [0, 1, 1, 1, 2, 2, 2];

pub(super) const NUM_BASE_ACTIONS: usize = 6;
pub(super) const AC_MAX_ABILITIES: usize = 8;
/// Total actions: 3 attack + 8 ability + 2 move + 1 hold = 14
pub const AC_NUM_ACTIONS: usize = 14;
pub(super) const NUM_ACTION_TYPES: usize = 11; // attack=0, move=1, hold=2, ability_0..7=3..10

#[derive(Debug, Clone)]
pub(super) struct FlatLinear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>,  // [out_dim x in_dim] row-major (w[i] = row i of weight matrix)
    pub b: Vec<f32>,  // [out_dim]
}

impl FlatLinear {
    pub fn from_json(lw: &LinearWeights) -> Self {
        let out_dim = lw.w.len();
        let in_dim = if out_dim > 0 { lw.w[0].len() } else { 0 };
        // PyTorch Linear stores [out_dim, in_dim], already row-major
        let w: Vec<f32> = lw.w.iter().flat_map(|row| row.iter().copied()).collect();
        Self {
            in_dim,
            out_dim,
            w,
            b: lw.b.clone(),
        }
    }

    /// y = Wx + b
    #[inline]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.out_dim {
            let row = &self.w[i * self.in_dim..(i + 1) * self.in_dim];
            output[i] = self.b[i] + dot_product(row, input);
        }
    }

    /// y = ReLU(Wx + b)  -- used for hidden layers
    pub fn forward_relu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = v.max(0.0);
        }
    }

    /// y = GELU(Wx + b)  -- used for transformer FFN
    pub fn forward_gelu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = gelu(*v);
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct FlatLayerNorm {
    pub dim: usize,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl FlatLayerNorm {
    pub fn from_json(ln: &LayerNormWeights) -> Self {
        Self {
            dim: ln.gamma.len(),
            gamma: ln.gamma.clone(),
            beta: ln.beta.clone(),
        }
    }

    pub fn forward(&self, x: &mut [f32]) {
        let n = self.dim as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for i in 0..self.dim {
            x[i] = (x[i] - mean) * inv_std * self.gamma[i] + self.beta[i];
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransformerLayer {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    // Self-attention: in_proj combines Q, K, V projections [3*d_model, d_model]
    attn_in_proj: FlatLinear,
    attn_out_proj: FlatLinear,
    // Feed-forward
    ff1: FlatLinear,
    ff2: FlatLinear,
    // Pre-norm
    norm1: FlatLayerNorm,
    norm2: FlatLayerNorm,
}

impl TransformerLayer {
    pub fn from_json(lj: &TransformerLayerJson, d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            head_dim: d_model / n_heads,
            attn_in_proj: FlatLinear::from_json(&lj.self_attn_in_proj),
            attn_out_proj: FlatLinear::from_json(&lj.self_attn_out_proj),
            ff1: FlatLinear::from_json(&lj.ff_linear1),
            ff2: FlatLinear::from_json(&lj.ff_linear2),
            norm1: FlatLayerNorm::from_json(&lj.norm1),
            norm2: FlatLayerNorm::from_json(&lj.norm2),
        }
    }

    /// Forward pass for one transformer layer (pre-norm style).
    /// `seq` is [seq_len x d_model] in row-major.
    /// `mask` has `true` for positions to attend, `false` for padding.
    /// `scratch` is a reusable buffer to avoid per-call allocations.
    pub fn forward(&self, seq: &mut [f32], seq_len: usize, mask: &[bool], scratch: &mut TransformerScratch) {
        let d = self.d_model;
        scratch.ensure(seq_len, d, self.ff1.out_dim);

        // --- Pre-norm self-attention ---
        // 1. Layer norm
        for t in 0..seq_len {
            scratch.normed[t * d..(t + 1) * d].copy_from_slice(&seq[t * d..(t + 1) * d]);
            self.norm1.forward(&mut scratch.normed[t * d..(t + 1) * d]);
        }

        // 2. QKV projection
        for t in 0..seq_len {
            self.attn_in_proj.forward(
                &scratch.normed[t * d..(t + 1) * d],
                &mut scratch.qkv[t * 3 * d..(t + 1) * 3 * d],
            );
        }

        // 3. Multi-head attention
        for v in scratch.attn_out[..seq_len * d].iter_mut() { *v = 0.0; }
        self.multi_head_attention(&scratch.qkv, &mut scratch.attn_out, seq_len, mask);

        // 4. Output projection + residual
        for t in 0..seq_len {
            self.attn_out_proj.forward(
                &scratch.attn_out[t * d..(t + 1) * d],
                &mut scratch.proj_out[..d],
            );
            for i in 0..d {
                seq[t * d + i] += scratch.proj_out[i];
            }
        }

        // --- Pre-norm feed-forward ---
        // 1. Layer norm
        for t in 0..seq_len {
            scratch.normed[t * d..(t + 1) * d].copy_from_slice(&seq[t * d..(t + 1) * d]);
            self.norm2.forward(&mut scratch.normed[t * d..(t + 1) * d]);
        }

        // 2. FFN: GELU(x W1 + b1) W2 + b2
        for t in 0..seq_len {
            self.ff1.forward_gelu(&scratch.normed[t * d..(t + 1) * d], &mut scratch.ff_hidden);
            self.ff2.forward(&scratch.ff_hidden, &mut scratch.ff_out[..d]);
            for i in 0..d {
                seq[t * d + i] += scratch.ff_out[i];
            }
        }
    }

    fn multi_head_attention(
        &self,
        qkv: &[f32],       // [seq_len x 3*d_model]
        output: &mut [f32], // [seq_len x d_model]
        seq_len: usize,
        mask: &[bool],
    ) {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        // For each head
        for head in 0..h {
            let offset = head * hd;

            // Compute attention scores
            for qi in 0..seq_len {
                // Softmax over keys
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for ki in 0..seq_len {
                    if !mask[ki] {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for f in 0..hd {
                        let q = qkv[qi * 3 * d + offset + f];
                        let k = qkv[ki * 3 * d + d + offset + f];
                        dot += q * k;
                    }
                    scores[ki] = dot * scale;
                }

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted sum of values -> output
                for f in 0..hd {
                    let mut sum = 0.0f32;
                    for vi in 0..seq_len {
                        let v = qkv[vi * 3 * d + 2 * d + offset + f];
                        sum += scores[vi] * v;
                    }
                    output[qi * d + offset + f] = sum;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reusable scratch buffers (avoid per-layer per-tick allocations)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub(super) struct TransformerScratch {
    pub normed: Vec<f32>,
    pub qkv: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub ff_hidden: Vec<f32>,
    pub ff_out: Vec<f32>,
}

impl TransformerScratch {
    pub fn ensure(&mut self, seq_len: usize, d_model: usize, d_ff: usize) {
        let n = seq_len * d_model;
        if self.normed.len() < n {
            self.normed.resize(n, 0.0);
        }
        let qkv_n = seq_len * 3 * d_model;
        if self.qkv.len() < qkv_n {
            self.qkv.resize(qkv_n, 0.0);
        }
        if self.attn_out.len() < n {
            self.attn_out.resize(n, 0.0);
        }
        if self.proj_out.len() < d_model {
            self.proj_out.resize(d_model, 0.0);
        }
        if self.ff_hidden.len() < d_ff {
            self.ff_hidden.resize(d_ff, 0.0);
        }
        if self.ff_out.len() < d_model {
            self.ff_out.resize(d_model, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-attention runtime type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct FlatCrossAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    attn_in_proj: FlatLinear,   // [3*d_model, d_model] -- Q from query, K/V from kv
    attn_out_proj: FlatLinear,
    norm_q: FlatLayerNorm,
    norm_kv: FlatLayerNorm,
    ff1: FlatLinear,
    ff2: FlatLinear,
    norm_ff: FlatLayerNorm,
}

impl FlatCrossAttention {
    pub fn from_json(cj: &CrossAttnJson, d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            head_dim: d_model / n_heads,
            attn_in_proj: FlatLinear::from_json(&cj.attn_in_proj),
            attn_out_proj: FlatLinear::from_json(&cj.attn_out_proj),
            norm_q: FlatLayerNorm::from_json(&cj.norm_q),
            norm_kv: FlatLayerNorm::from_json(&cj.norm_kv),
            ff1: FlatLinear::from_json(&cj.ff_linear1),
            ff2: FlatLinear::from_json(&cj.ff_linear2),
            norm_ff: FlatLayerNorm::from_json(&cj.norm_ff),
        }
    }

    /// Cross-attention: query (d_model) attends to kv (n_entities x d_model).
    /// Returns updated query embedding (d_model).
    pub fn forward(
        &self,
        query: &[f32],          // [d_model]
        kv: &[f32],             // [n_entities x d_model]
        kv_mask: &[bool],       // [n_entities] -- true = attend
        n_entities: usize,
    ) -> Vec<f32> {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        // Pre-norm query
        let mut normed_q = query.to_vec();
        self.norm_q.forward(&mut normed_q);

        // Pre-norm key/value
        let mut normed_kv = kv.to_vec();
        for ent in 0..n_entities {
            self.norm_kv.forward(&mut normed_kv[ent * d..(ent + 1) * d]);
        }

        // QKV projection for query (only Q used from query)
        let mut q_qkv = vec![0.0f32; 3 * d];
        self.attn_in_proj.forward(&normed_q, &mut q_qkv);

        // QKV projection for each kv entity (K and V used)
        let mut kv_qkv = vec![0.0f32; n_entities * 3 * d];
        for ent in 0..n_entities {
            self.attn_in_proj.forward(
                &normed_kv[ent * d..(ent + 1) * d],
                &mut kv_qkv[ent * 3 * d..(ent + 1) * 3 * d],
            );
        }

        // Multi-head cross-attention
        let mut attn_out = vec![0.0f32; d];
        for head in 0..h {
            let offset = head * hd;

            // Scores: Q[head] * K[head] for each entity
            let mut scores = vec![f32::NEG_INFINITY; n_entities];
            for ki in 0..n_entities {
                if !kv_mask[ki] {
                    continue;
                }
                let mut dot = 0.0f32;
                for f in 0..hd {
                    let q = q_qkv[offset + f];                          // Q from query
                    let k = kv_qkv[ki * 3 * d + d + offset + f];       // K from kv
                    dot += q * k;
                }
                scores[ki] = dot * scale;
            }

            softmax_inplace(&mut scores);

            // Weighted sum of V
            for f in 0..hd {
                let mut sum = 0.0f32;
                for vi in 0..n_entities {
                    let v = kv_qkv[vi * 3 * d + 2 * d + offset + f];   // V from kv
                    sum += scores[vi] * v;
                }
                attn_out[offset + f] = sum;
            }
        }

        // Output projection
        let mut proj_out = vec![0.0f32; d];
        self.attn_out_proj.forward(&attn_out, &mut proj_out);

        // Residual: query + attn_output
        let mut x = vec![0.0f32; d];
        for i in 0..d {
            x[i] = query[i] + proj_out[i];
        }

        // Feed-forward with residual
        let mut ff_normed = x.clone();
        self.norm_ff.forward(&mut ff_normed);
        let mut ff_hidden = vec![0.0f32; self.ff1.out_dim];
        self.ff1.forward_gelu(&ff_normed, &mut ff_hidden);
        let mut ff_out = vec![0.0f32; d];
        self.ff2.forward(&ff_hidden, &mut ff_out);
        for i in 0..d {
            x[i] += ff_out[i];
        }

        x
    }
}

// AbilityTransformerWeights, EmbeddingRegistry, and tests are in weights_base.rs
