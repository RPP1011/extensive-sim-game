//! Transformer weight loading and inference.
//!
//! Implements multi-head self-attention, layer norm, and feed-forward layers
//! for frozen inference.  Follows the same patterns as EvalWeights and FlatMLP
//! in the existing codebase.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// SIMD-friendly dot product — auto-vectorized by LLVM
// ---------------------------------------------------------------------------

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Process in chunks of 8 for autovectorization
    let n = a.len();
    let chunks = n / 8;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    for c in 0..chunks {
        let base = c * 8;
        sum0 += a[base] * b[base] + a[base + 1] * b[base + 1];
        sum1 += a[base + 2] * b[base + 2] + a[base + 3] * b[base + 3];
        sum2 += a[base + 4] * b[base + 4] + a[base + 5] * b[base + 5];
        sum3 += a[base + 6] * b[base + 6] + a[base + 7] * b[base + 7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

// ---------------------------------------------------------------------------
// JSON schema (matches training/export_weights.py output)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LinearWeights {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct LayerNormWeights {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TransformerLayerJson {
    self_attn_in_proj: LinearWeights,
    self_attn_out_proj: LinearWeights,
    ff_linear1: LinearWeights,
    ff_linear2: LinearWeights,
    norm1: LayerNormWeights,
    norm2: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
struct DecisionHeadJson {
    urgency: HeadPairJson,
    target: HeadPairJson,
}

#[derive(Debug, Deserialize)]
struct HeadPairJson {
    linear1: LinearWeights,
    linear2: LinearWeights,
}

#[derive(Debug, Deserialize)]
struct ArchitectureJson {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    max_seq_len: usize,
    #[serde(default)]
    game_state_dim: usize,
    #[serde(default)]
    n_targets: usize,
    pad_id: usize,
    cls_id: usize,
}

#[derive(Debug, Deserialize)]
struct EntityEncoderJson {
    proj: LinearWeights,
    type_emb: Vec<Vec<f32>>,  // [3 × d_model]
    input_norm: LayerNormWeights,
    /// Self-attention layers over entities (from pre-training).
    /// Empty if no pre-trained encoder loaded.
    #[serde(default)]
    self_attn_layers: Vec<TransformerLayerJson>,
    out_norm: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
struct CrossAttnJson {
    attn_in_proj: LinearWeights,
    attn_out_proj: LinearWeights,
    norm_q: LayerNormWeights,
    norm_kv: LayerNormWeights,
    ff_linear1: LinearWeights,
    ff_linear2: LinearWeights,
    norm_ff: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
struct TransformerFileJson {
    architecture: ArchitectureJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder: Option<EntityEncoderJson>,
    cross_attn: Option<CrossAttnJson>,
    decision_head: Option<DecisionHeadJson>,
}

// ---------------------------------------------------------------------------
// Flattened runtime types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlatLinear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>,  // [out_dim × in_dim] row-major (w[i] = row i of weight matrix)
    b: Vec<f32>,  // [out_dim]
}

impl FlatLinear {
    fn from_json(lw: &LinearWeights) -> Self {
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
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.out_dim {
            let row = &self.w[i * self.in_dim..(i + 1) * self.in_dim];
            output[i] = self.b[i] + dot_product(row, input);
        }
    }

    /// y = ReLU(Wx + b)  — used for hidden layers
    fn forward_relu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = v.max(0.0);
        }
    }

    /// y = GELU(Wx + b)  — used for transformer FFN
    fn forward_gelu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = gelu(*v);
        }
    }
}

#[derive(Debug, Clone)]
struct FlatLayerNorm {
    dim: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl FlatLayerNorm {
    fn from_json(ln: &LayerNormWeights) -> Self {
        Self {
            dim: ln.gamma.len(),
            gamma: ln.gamma.clone(),
            beta: ln.beta.clone(),
        }
    }

    fn forward(&self, x: &mut [f32]) {
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
struct TransformerLayer {
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
    fn from_json(lj: &TransformerLayerJson, d_model: usize, n_heads: usize) -> Self {
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
    /// `seq` is [seq_len × d_model] in row-major.
    /// `mask` has `true` for positions to attend, `false` for padding.
    /// `scratch` is a reusable buffer to avoid per-call allocations.
    fn forward(&self, seq: &mut [f32], seq_len: usize, mask: &[bool], scratch: &mut TransformerScratch) {
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
        qkv: &[f32],       // [seq_len × 3*d_model]
        output: &mut [f32], // [seq_len × d_model]
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

                // Weighted sum of values → output
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
struct TransformerScratch {
    normed: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    ff_hidden: Vec<f32>,
    ff_out: Vec<f32>,
}

impl TransformerScratch {
    fn ensure(&mut self, seq_len: usize, d_model: usize, d_ff: usize) {
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
// Entity encoder + cross-attention runtime types
// ---------------------------------------------------------------------------

const ENTITY_DIM: usize = 30;
const NUM_ENTITIES: usize = 7;
const NUM_ENTITY_TYPES: usize = 3;
/// Entity type IDs: [self, enemy, enemy, enemy, ally, ally, ally]
const ENTITY_TYPE_IDS: [usize; NUM_ENTITIES] = [0, 1, 1, 1, 2, 2, 2];

#[derive(Debug, Clone)]
struct FlatEntityEncoder {
    proj: FlatLinear,          // (ENTITY_DIM → d_model)
    type_emb: Vec<f32>,        // [NUM_ENTITY_TYPES × d_model]
    input_norm: FlatLayerNorm,
    self_attn_layers: Vec<TransformerLayer>,
    out_norm: FlatLayerNorm,
    d_model: usize,
}

impl FlatEntityEncoder {
    fn from_json(ej: &EntityEncoderJson, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            proj: FlatLinear::from_json(&ej.proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode game_state (210 floats) into entity tokens [7 × d_model].
    /// Returns (entity_tokens, entity_mask) where mask[i] = true means attend.
    fn forward(&self, game_state: &[f32]) -> (Vec<f32>, Vec<bool>) {
        let d = self.d_model;
        let mut tokens = vec![0.0f32; NUM_ENTITIES * d];
        let mut mask = vec![false; NUM_ENTITIES];

        for ent in 0..NUM_ENTITIES {
            let entity_feats = &game_state[ent * ENTITY_DIM..(ent + 1) * ENTITY_DIM];

            // exists = feature index 29
            let exists = entity_feats[29] > 0.5;
            mask[ent] = exists;

            // Project entity features to d_model
            self.proj.forward(entity_feats, &mut tokens[ent * d..(ent + 1) * d]);

            // Add type embedding
            let type_id = ENTITY_TYPE_IDS[ent];
            for i in 0..d {
                tokens[ent * d + i] += self.type_emb[type_id * d + i];
            }

            // Input layer norm per entity
            self.input_norm.forward(&mut tokens[ent * d..(ent + 1) * d]);
        }

        // Self-attention over entities (from pre-training)
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, NUM_ENTITIES, &mask, &mut scratch);
        }

        // Output norm per entity
        for ent in 0..NUM_ENTITIES {
            self.out_norm.forward(&mut tokens[ent * d..(ent + 1) * d]);
        }

        (tokens, mask)
    }
}

#[derive(Debug, Clone)]
struct FlatCrossAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    attn_in_proj: FlatLinear,   // [3*d_model, d_model] — Q from query, K/V from kv
    attn_out_proj: FlatLinear,
    norm_q: FlatLayerNorm,
    norm_kv: FlatLayerNorm,
    ff1: FlatLinear,
    ff2: FlatLinear,
    norm_ff: FlatLayerNorm,
}

impl FlatCrossAttention {
    fn from_json(cj: &CrossAttnJson, d_model: usize, n_heads: usize) -> Self {
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

    /// Cross-attention: query (d_model) attends to kv (n_entities × d_model).
    /// Returns updated query embedding (d_model).
    fn forward(
        &self,
        query: &[f32],          // [d_model]
        kv: &[f32],             // [n_entities × d_model]
        kv_mask: &[bool],       // [n_entities] — true = attend
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

            // Scores: Q[head] · K[head] for each entity
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

// ---------------------------------------------------------------------------
// Top-level inference struct
// ---------------------------------------------------------------------------

/// Frozen ability transformer for inference.
#[derive(Debug, Clone)]
pub struct AbilityTransformerWeights {
    d_model: usize,
    n_heads: usize,
    max_seq_len: usize,
    pad_id: usize,
    _cls_id: usize,
    n_targets: usize,

    // Embeddings: [vocab_size × d_model] and [max_seq_len × d_model]
    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    vocab_size: usize,

    // Output norm
    out_norm: FlatLayerNorm,

    // Transformer layers
    layers: Vec<TransformerLayer>,

    // Cross-attention with game state entities
    entity_encoder: Option<FlatEntityEncoder>,
    cross_attn: Option<FlatCrossAttention>,

    // Decision head (None for encoder-only / Phase 1 exports)
    urgency_l1: Option<FlatLinear>,
    urgency_l2: Option<FlatLinear>,
    target_l1: Option<FlatLinear>,
    target_l2: Option<FlatLinear>,
}

/// Output of transformer inference.
#[derive(Debug, Clone)]
pub struct TransformerOutput {
    pub urgency: f32,
    pub target_logits: Vec<f32>,
}

impl AbilityTransformerWeights {
    /// Load from JSON string exported by `training/export_weights.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: TransformerFileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        // Flatten embeddings to [vocab_size * d_model]
        let token_emb: Vec<f32> = file
            .token_embedding
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let pos_emb: Vec<f32> = file
            .position_embedding
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err(format!(
                "token_emb size mismatch: {} vs {}×{}",
                token_emb.len(),
                arch.vocab_size,
                arch.d_model
            ));
        }

        let layers: Vec<TransformerLayer> = file
            .layers
            .iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = file.entity_encoder.as_ref()
            .map(|ej| FlatEntityEncoder::from_json(ej, arch.d_model, arch.n_heads));
        let cross_attn_block = file.cross_attn.as_ref()
            .map(|cj| FlatCrossAttention::from_json(cj, arch.d_model, arch.n_heads));

        Ok(Self {
            d_model: arch.d_model,
            n_heads: arch.n_heads,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            _cls_id: arch.cls_id,
            n_targets: arch.n_targets,
            token_emb,
            pos_emb,
            vocab_size: arch.vocab_size,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn: cross_attn_block,
            urgency_l1: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.urgency.linear1)),
            urgency_l2: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.urgency.linear2)),
            target_l1: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.target.linear1)),
            target_l2: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.target.linear2)),
        })
    }

    /// Encode ability tokens into a [CLS] embedding.
    ///
    /// This is **static per ability** — token sequences never change during
    /// a fight. Cache the result at fight init and reuse every tick.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] =
                    self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Predict from a cached [CLS] embedding and precomputed entity tokens.
    ///
    /// This is the **hot path** — runs every tick per ability. Only
    /// cross-attention + decision head, no transformer encoder.
    pub fn predict_from_cls(
        &self,
        cls: &[f32],
        entities: Option<&EncodedEntities>,
    ) -> TransformerOutput {
        let d = self.d_model;

        // Cross-attention with entity tokens
        let head_input = if let (Some(ca), Some(ent)) = (&self.cross_attn, entities) {
            ca.forward(cls, &ent.tokens, &ent.mask, NUM_ENTITIES)
        } else {
            cls.to_vec()
        };

        // Decision heads (require Phase 2 weights)
        let ul1 = self.urgency_l1.as_ref().expect("predict_from_cls requires Phase 2 weights");
        let ul2 = self.urgency_l2.as_ref().unwrap();
        let tl1 = self.target_l1.as_ref().unwrap();
        let tl2 = self.target_l2.as_ref().unwrap();

        let mut u_hidden = vec![0.0f32; ul1.out_dim];
        ul1.forward_gelu(&head_input, &mut u_hidden);
        let mut u_out = vec![0.0f32; 1];
        ul2.forward(&u_hidden, &mut u_out);

        let mut t_hidden = vec![0.0f32; tl1.out_dim];
        tl1.forward_gelu(&head_input, &mut t_hidden);
        let mut target_logits = vec![0.0f32; self.n_targets];
        tl2.forward(&t_hidden, &mut target_logits);

        TransformerOutput {
            urgency: sigmoid(u_out[0]),
            target_logits,
        }
    }

    /// Run full forward inference (convenience, no caching).
    ///
    /// * `token_ids` — token IDs (with [CLS] at position 0)
    /// * `game_state` — optional game state features
    ///
    /// Returns urgency ∈ [0, 1] and target logits.
    pub fn predict(
        &self,
        token_ids: &[u32],
        game_state: Option<&[f32]>,
    ) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        self.predict_from_cls(&cls, entities.as_ref())
    }

    /// Convenience: get the argmax target index.
    pub fn predict_target(&self, token_ids: &[u32], game_state: Option<&[f32]>) -> (f32, usize) {
        let out = self.predict(token_ids, game_state);
        let best = out
            .target_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (out.urgency, best)
    }

    // -----------------------------------------------------------------------
    // Batch API: precompute entity tokens once, evaluate multiple abilities
    // -----------------------------------------------------------------------

    /// Pre-computed entity encoding for a single unit's game state.
    /// Reusable across all ability evaluations for that unit.
    pub fn encode_entities(&self, game_state: &[f32]) -> Option<EncodedEntities> {
        let enc = self.entity_encoder.as_ref()?;
        let (tokens, mask) = enc.forward(game_state);
        Some(EncodedEntities { tokens, mask })
    }

    /// Evaluate a single ability using precomputed entity tokens.
    ///
    /// Avoids redundant entity encoder calls when evaluating
    /// multiple abilities for the same unit.
    pub fn predict_with_entities(
        &self,
        token_ids: &[u32],
        entities: Option<&EncodedEntities>,
    ) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        self.predict_from_cls(&cls, entities)
    }

    /// Evaluate all abilities for a single unit at once.
    ///
    /// Computes entity encoding once, then evaluates each ability against it.
    pub fn predict_batch(
        &self,
        abilities: &[&[u32]],
        game_state: Option<&[f32]>,
    ) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        abilities
            .iter()
            .map(|tokens| self.predict_with_entities(tokens, entities.as_ref()))
            .collect()
    }

    /// Evaluate all abilities from cached [CLS] embeddings.
    ///
    /// **Optimal per-tick path.** Cache [CLS] embeddings at fight init
    /// with `encode_cls()`, then call this every tick. Only runs
    /// entity encoder (once) + cross-attention + decision head per ability.
    ///
    /// For 4 heroes × 8 abilities: 4 entity encoder calls + 32 cross-attn calls.
    /// No transformer encoder calls at all during the fight.
    pub fn predict_batch_cached(
        &self,
        cached_cls: &[&[f32]],
        game_state: Option<&[f32]>,
    ) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        cached_cls
            .iter()
            .map(|cls| self.predict_from_cls(cls, entities.as_ref()))
            .collect()
    }
}

/// Pre-computed entity encoding, reusable across ability evaluations.
#[derive(Debug, Clone)]
pub struct EncodedEntities {
    /// Flat entity tokens: [NUM_ENTITIES × d_model]
    pub tokens: Vec<f32>,
    /// Entity existence mask: [NUM_ENTITIES], true = entity exists
    pub mask: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-10.0, 10.0)).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Approximation: x * σ(1.702 * x)
    x * sigmoid(1.702 * x)
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Actor-critic inference (full 14-action policy)
// ---------------------------------------------------------------------------

const NUM_BASE_ACTIONS: usize = 6;
const AC_MAX_ABILITIES: usize = 8;
/// Total actions: 3 attack + 8 ability + 2 move + 1 hold = 14
pub const AC_NUM_ACTIONS: usize = 14;

#[derive(Debug, Deserialize)]
struct HeadJson {
    linear1: LinearWeights,
    linear2: LinearWeights,
}

#[derive(Debug, Deserialize)]
struct ActorCriticFileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder: EntityEncoderJson,
    cross_attn: CrossAttnJson,
    base_head: HeadJson,
    ability_proj: HeadJson,
    /// Optional projection for external CLS embeddings (e.g. 128→32 behavioral).
    external_cls_proj: Option<LinearWeights>,
}

#[derive(Debug, Deserialize)]
struct ActorCriticArchJson {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    max_seq_len: usize,
    #[allow(dead_code)]
    #[serde(default)]
    game_state_dim: usize,
    #[serde(default)]
    num_base_actions: usize,
    pad_id: usize,
    #[allow(dead_code)]
    cls_id: usize,
    #[serde(default)]
    external_cls_dim: usize,
}

/// Frozen actor-critic weights for 14-action policy inference.
#[derive(Debug, Clone)]
pub struct ActorCriticWeights {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoder,
    cross_attn: FlatCrossAttention,

    // Base head: pooled entity state → 6 logits
    base_l1: FlatLinear,
    base_l2: FlatLinear,
    // Ability projection: cross-attended CLS → 1 logit
    ability_l1: FlatLinear,
    ability_l2: FlatLinear,
    // Optional projection for external CLS (e.g. 128-dim behavioral → d_model)
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeights {
    /// Load from JSON string exported by `training/export_actor_critic.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticFileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err(format!("token_emb size mismatch"));
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoder::from_json(
            &file.entity_encoder, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );

        if arch.num_base_actions != NUM_BASE_ACTIONS {
            return Err(format!(
                "num_base_actions mismatch: {} vs {}",
                arch.num_base_actions, NUM_BASE_ACTIONS
            ));
        }

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            base_l1: FlatLinear::from_json(&file.base_head.linear1),
            base_l2: FlatLinear::from_json(&file.base_head.linear2),
            ability_l1: FlatLinear::from_json(&file.ability_proj.linear1),
            ability_l2: FlatLinear::from_json(&file.ability_proj.linear2),
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    /// Project external CLS embedding (e.g. 128-dim behavioral) to d_model.
    /// If no projection layer exists, returns the input unchanged.
    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            // GELU activation not needed for single linear projection
            out
        } else {
            cls.to_vec()
        }
    }

    /// Encode ability tokens → [CLS] embedding. Cache at fight start.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Encode game state → entity tokens + mask + mean-pooled state.
    pub fn encode_entities(&self, game_state: &[f32]) -> EntityState {
        let d = self.d_model;
        let (tokens, mask) = self.entity_encoder.forward(game_state);

        // Mean pool over existing entities
        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for ent in 0..NUM_ENTITIES {
            if mask[ent] {
                count += 1.0;
                for i in 0..d {
                    pooled[i] += tokens[ent * d + i];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityState { tokens, mask, pooled }
    }

    /// Compute 14-action logits.
    ///
    /// * `entity_state` — precomputed entity encoding for this tick
    /// * `ability_cls` — list of cached [CLS] embeddings per ability slot (None if slot empty)
    pub fn action_logits(
        &self,
        entity_state: &EntityState,
        ability_cls: &[Option<&[f32]>],
    ) -> [f32; AC_NUM_ACTIONS] {
        let d = self.d_model;
        let mut logits = [f32::NEG_INFINITY; AC_NUM_ACTIONS];

        // Base action logits from pooled entity state
        let mut base_hidden = vec![0.0f32; self.base_l1.out_dim];
        self.base_l1.forward_gelu(&entity_state.pooled, &mut base_hidden);
        let mut base_out = vec![0.0f32; NUM_BASE_ACTIONS];
        self.base_l2.forward(&base_hidden, &mut base_out);

        // Map base actions: [attack_near, attack_weak, attack_focus, move_toward, move_away, hold]
        // → action indices [0, 1, 2, 11, 12, 13]
        logits[0] = base_out[0];   // attack nearest
        logits[1] = base_out[1];   // attack weakest
        logits[2] = base_out[2];   // attack focus
        logits[11] = base_out[3];  // move toward
        logits[12] = base_out[4];  // move away
        logits[13] = base_out[5];  // hold

        // Ability logits: cross-attend each ability [CLS] with entity tokens
        let mut ability_hidden = vec![0.0f32; self.ability_l1.out_dim];
        let mut ability_out = vec![0.0f32; 1];

        for (idx, cls_opt) in ability_cls.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, NUM_ENTITIES,
                );
                self.ability_l1.forward_gelu(&cross_emb, &mut ability_hidden);
                self.ability_l2.forward(&ability_hidden, &mut ability_out);
                logits[3 + idx] = ability_out[0];
            }
        }

        logits
    }
}

/// Pre-computed entity state for a single tick.
#[derive(Debug, Clone)]
pub struct EntityState {
    pub tokens: Vec<f32>,
    pub mask: Vec<bool>,
    pub pooled: Vec<f32>,
}

// ---------------------------------------------------------------------------
// V2 entity encoder: variable-length entities + threats
// ---------------------------------------------------------------------------

const THREAT_DIM: usize = 8;
const NUM_ENTITY_TYPES_V2: usize = 4; // self=0, enemy=1, ally=2, threat=3

#[derive(Debug, Deserialize)]
struct EntityEncoderV2Json {
    entity_proj: LinearWeights,
    threat_proj: LinearWeights,
    type_emb: Vec<Vec<f32>>,  // [4 × d_model]
    input_norm: LayerNormWeights,
    #[serde(default)]
    self_attn_layers: Vec<TransformerLayerJson>,
    out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
struct FlatEntityEncoderV2 {
    entity_proj: FlatLinear,   // (30 → d_model)
    threat_proj: FlatLinear,   // (8 → d_model)
    type_emb: Vec<f32>,        // [4 × d_model]
    input_norm: FlatLayerNorm,
    self_attn_layers: Vec<TransformerLayer>,
    out_norm: FlatLayerNorm,
    d_model: usize,
}

impl FlatEntityEncoderV2 {
    fn from_json(ej: &EntityEncoderV2Json, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            entity_proj: FlatLinear::from_json(&ej.entity_proj),
            threat_proj: FlatLinear::from_json(&ej.threat_proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode variable-length entities + threats into tokens.
    /// Returns (tokens [n_total × d_model], mask [n_total], n_total).
    fn forward(
        &self,
        entities: &[&[f32]],     // each 30-dim
        entity_types: &[usize],  // 0/1/2 per entity
        threats: &[&[f32]],      // each 8-dim
    ) -> (Vec<f32>, Vec<bool>, usize) {
        let d = self.d_model;
        let n_ents = entities.len();
        let n_threats = threats.len();
        let n_total = n_ents + n_threats;

        let mut tokens = vec![0.0f32; n_total * d];
        let mut mask = vec![true; n_total]; // true = attend

        // Project entities
        for (i, (feats, &type_id)) in entities.iter().zip(entity_types).enumerate() {
            let exists = feats.len() >= 30 && feats[29] > 0.5;
            mask[i] = exists;

            self.entity_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);

            let tid = type_id.min(NUM_ENTITY_TYPES_V2 - 1);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[tid * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project threats (type_id = 3)
        for (ti, feats) in threats.iter().enumerate() {
            let i = n_ents + ti;
            let exists = feats.len() >= 8 && feats[7] > 0.5;
            mask[i] = exists;

            self.threat_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);

            for j in 0..d {
                tokens[i * d + j] += self.type_emb[3 * d + j]; // threat type
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Self-attention with reused scratch buffer
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, n_total, &mask, &mut scratch);
        }

        // Output norm
        for i in 0..n_total {
            self.out_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        (tokens, mask, n_total)
    }
}

// ---------------------------------------------------------------------------
// V2 actor-critic inference
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV2FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v2: EntityEncoderV2Json,
    cross_attn: CrossAttnJson,
    base_head: HeadJson,
    ability_proj: HeadJson,
}

/// V2 actor-critic weights with variable-length entity encoder.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV2 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoderV2,
    cross_attn: FlatCrossAttention,

    base_l1: FlatLinear,
    base_l2: FlatLinear,
    ability_l1: FlatLinear,
    ability_l2: FlatLinear,
}

impl ActorCriticWeightsV2 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV2FileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err("token_emb size mismatch".to_string());
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoderV2::from_json(
            &file.entity_encoder_v2, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );

        if arch.num_base_actions != NUM_BASE_ACTIONS {
            return Err(format!(
                "num_base_actions mismatch: {} vs {}",
                arch.num_base_actions, NUM_BASE_ACTIONS
            ));
        }

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            base_l1: FlatLinear::from_json(&file.base_head.linear1),
            base_l2: FlatLinear::from_json(&file.base_head.linear2),
            ability_l1: FlatLinear::from_json(&file.ability_proj.linear1),
            ability_l2: FlatLinear::from_json(&file.ability_proj.linear2),
        })
    }

    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Encode v2 game state → entity state with variable-length tokens.
    pub fn encode_entities_v2(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
    ) -> EntityState {
        let d = self.d_model;
        let (tokens, mask, n_total) = self.entity_encoder.forward(
            entities, entity_types, threats,
        );

        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for i in 0..n_total {
            if mask[i] {
                count += 1.0;
                for j in 0..d {
                    pooled[j] += tokens[i * d + j];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityState { tokens, mask, pooled }
    }

    /// Compute 14-action logits using v2 entity state.
    pub fn action_logits(
        &self,
        entity_state: &EntityState,
        ability_cls: &[Option<&[f32]>],
    ) -> [f32; AC_NUM_ACTIONS] {
        let _d = self.d_model;
        let n_entities = entity_state.mask.len();
        let mut logits = [f32::NEG_INFINITY; AC_NUM_ACTIONS];

        let mut base_hidden = vec![0.0f32; self.base_l1.out_dim];
        self.base_l1.forward_gelu(&entity_state.pooled, &mut base_hidden);
        let mut base_out = vec![0.0f32; NUM_BASE_ACTIONS];
        self.base_l2.forward(&base_hidden, &mut base_out);

        logits[0] = base_out[0];
        logits[1] = base_out[1];
        logits[2] = base_out[2];
        logits[11] = base_out[3];
        logits[12] = base_out[4];
        logits[13] = base_out[5];

        let mut ability_hidden = vec![0.0f32; self.ability_l1.out_dim];
        let mut ability_out = vec![0.0f32; 1];

        for (idx, cls_opt) in ability_cls.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_entities,
                );
                self.ability_l1.forward_gelu(&cross_emb, &mut ability_hidden);
                self.ability_l2.forward(&ability_hidden, &mut ability_out);
                logits[3 + idx] = ability_out[0];
            }
        }

        logits
    }
}

// ---------------------------------------------------------------------------
// V3: Pointer-based action space with position tokens
// ---------------------------------------------------------------------------

const POSITION_DIM: usize = 8;
const NUM_ENTITY_TYPES_V3: usize = 5; // self=0, enemy=1, ally=2, threat=3, position=4
const NUM_ACTION_TYPES: usize = 11; // attack=0, move=1, hold=2, ability_0..7=3..10

#[derive(Debug, Deserialize)]
struct EntityEncoderV3Json {
    entity_proj: LinearWeights,
    threat_proj: LinearWeights,
    position_proj: LinearWeights,
    type_emb: Vec<Vec<f32>>,  // [5 × d_model]
    input_norm: LayerNormWeights,
    #[serde(default)]
    self_attn_layers: Vec<TransformerLayerJson>,
    out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
struct FlatEntityEncoderV3 {
    entity_proj: FlatLinear,     // (30 → d_model)
    threat_proj: FlatLinear,     // (8 → d_model)
    position_proj: FlatLinear,   // (8 → d_model)
    type_emb: Vec<f32>,          // [5 × d_model]
    input_norm: FlatLayerNorm,
    self_attn_layers: Vec<TransformerLayer>,
    out_norm: FlatLayerNorm,
    d_model: usize,
}

impl FlatEntityEncoderV3 {
    fn from_json(ej: &EntityEncoderV3Json, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            entity_proj: FlatLinear::from_json(&ej.entity_proj),
            threat_proj: FlatLinear::from_json(&ej.threat_proj),
            position_proj: FlatLinear::from_json(&ej.position_proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode variable-length entities + threats + positions into tokens.
    /// Returns (tokens [n_total × d_model], mask [n_total], n_total, entity_type_ids [n_total]).
    fn forward(
        &self,
        entities: &[&[f32]],     // each 30-dim
        entity_types: &[usize],  // 0/1/2 per entity
        threats: &[&[f32]],      // each 8-dim
        positions: &[&[f32]],    // each 8-dim
    ) -> (Vec<f32>, Vec<bool>, usize, Vec<usize>) {
        let d = self.d_model;
        let n_ents = entities.len();
        let n_threats = threats.len();
        let n_positions = positions.len();
        let n_total = n_ents + n_threats + n_positions;

        let mut tokens = vec![0.0f32; n_total * d];
        let mut mask = vec![true; n_total]; // true = attend
        let mut full_type_ids = vec![0usize; n_total];

        // Project entities
        for (i, (feats, &type_id)) in entities.iter().zip(entity_types).enumerate() {
            let exists = feats.len() >= 30 && feats[29] > 0.5;
            mask[i] = exists;
            full_type_ids[i] = type_id;

            self.entity_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            let tid = type_id.min(NUM_ENTITY_TYPES_V3 - 1);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[tid * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project threats (type_id = 3)
        for (ti, feats) in threats.iter().enumerate() {
            let i = n_ents + ti;
            let exists = feats.len() >= THREAT_DIM && feats[7] > 0.5;
            mask[i] = exists;
            full_type_ids[i] = 3;

            self.threat_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[3 * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project positions (type_id = 4)
        for (pi, feats) in positions.iter().enumerate() {
            let i = n_ents + n_threats + pi;
            mask[i] = true; // positions always exist (no padding in this direction)
            full_type_ids[i] = 4;

            self.position_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[4 * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Self-attention
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, n_total, &mask, &mut scratch);
        }

        // Output norm
        for i in 0..n_total {
            self.out_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        (tokens, mask, n_total, full_type_ids)
    }
}

/// Pointer output from V3 action head.
#[derive(Debug)]
pub struct PointerOutput {
    /// Action type logits (11 elements).
    pub type_logits: Vec<f32>,
    /// Attack pointer logits over all tokens.
    pub attack_ptr: Vec<f32>,
    /// Move pointer logits over all tokens.
    pub move_ptr: Vec<f32>,
    /// Per-ability pointer logits (None if ability slot empty).
    pub ability_ptrs: Vec<Option<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct PointerHeadJson {
    action_type_head: HeadJson,
    pointer_key: LinearWeights,
    attack_query: LinearWeights,
    move_query: LinearWeights,
    ability_queries: Vec<LinearWeights>,
}

#[derive(Debug, Clone)]
struct FlatPointerHead {
    action_type_l1: FlatLinear,
    action_type_l2: FlatLinear,
    pointer_key: FlatLinear,
    attack_query: FlatLinear,
    move_query: FlatLinear,
    ability_queries: Vec<FlatLinear>,
    d_model: usize,
    scale: f32,
}

impl FlatPointerHead {
    fn from_json(pj: &PointerHeadJson, d_model: usize) -> Self {
        Self {
            action_type_l1: FlatLinear::from_json(&pj.action_type_head.linear1),
            action_type_l2: FlatLinear::from_json(&pj.action_type_head.linear2),
            pointer_key: FlatLinear::from_json(&pj.pointer_key),
            attack_query: FlatLinear::from_json(&pj.attack_query),
            move_query: FlatLinear::from_json(&pj.move_query),
            ability_queries: pj.ability_queries.iter()
                .map(|q| FlatLinear::from_json(q))
                .collect(),
            d_model,
            scale: (d_model as f32).powf(-0.5),
        }
    }

    fn forward(
        &self,
        pooled: &[f32],
        tokens: &[f32],
        mask: &[bool],
        n_total: usize,
        type_ids: &[usize],
        ability_cross_embs: &[Option<Vec<f32>>],
    ) -> PointerOutput {
        let d = self.d_model;

        // Action type logits
        let mut type_hidden = vec![0.0f32; self.action_type_l1.out_dim];
        self.action_type_l1.forward_gelu(pooled, &mut type_hidden);
        let mut type_logits = vec![0.0f32; NUM_ACTION_TYPES];
        self.action_type_l2.forward(&type_hidden, &mut type_logits);

        // Compute keys for all tokens
        let mut keys = vec![0.0f32; n_total * d];
        for i in 0..n_total {
            self.pointer_key.forward(
                &tokens[i * d..(i + 1) * d],
                &mut keys[i * d..(i + 1) * d],
            );
        }

        // Attack pointer: Q·K^T, masked to enemies only
        let mut atk_query = vec![0.0f32; d];
        self.attack_query.forward(pooled, &mut atk_query);
        let mut attack_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] == 1 {
                attack_ptr[i] = dot_product(&atk_query, &keys[i * d..(i + 1) * d]) * self.scale;
            }
        }

        // Move pointer: Q·K^T, masked to non-self tokens
        let mut mv_query = vec![0.0f32; d];
        self.move_query.forward(pooled, &mut mv_query);
        let mut move_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] != 0 {
                move_ptr[i] = dot_product(&mv_query, &keys[i * d..(i + 1) * d]) * self.scale;
            }
        }

        // Ability pointers
        let mut ability_ptrs = Vec::with_capacity(AC_MAX_ABILITIES);
        for (idx, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if idx < self.ability_queries.len() {
                    let mut ab_query = vec![0.0f32; d];
                    self.ability_queries[idx].forward(cross_emb, &mut ab_query);
                    let mut ab_ptr = vec![f32::NEG_INFINITY; n_total];
                    for i in 0..n_total {
                        if mask[i] {
                            ab_ptr[i] = dot_product(&ab_query, &keys[i * d..(i + 1) * d]) * self.scale;
                        }
                    }
                    ability_ptrs.push(Some(ab_ptr));
                } else {
                    ability_ptrs.push(None);
                }
            } else {
                ability_ptrs.push(None);
            }
        }

        PointerOutput { type_logits, attack_ptr, move_ptr, ability_ptrs }
    }
}

// ---------------------------------------------------------------------------
// V3 actor-critic JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV3FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v3: EntityEncoderV3Json,
    cross_attn: CrossAttnJson,
    pointer_head: PointerHeadJson,
    /// Optional projection for external CLS embeddings (e.g. 128→32 behavioral).
    external_cls_proj: Option<LinearWeights>,
}

/// V3 actor-critic weights with pointer-based action space.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV3 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoderV3,
    cross_attn: FlatCrossAttention,
    pointer_head: FlatPointerHead,
    // Optional projection for external CLS (e.g. 128-dim behavioral → d_model)
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeightsV3 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV3FileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err("token_emb size mismatch".to_string());
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoderV3::from_json(
            &file.entity_encoder_v3, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );
        let pointer_head = FlatPointerHead::from_json(
            &file.pointer_head, arch.d_model,
        );

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            pointer_head,
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    /// Project external CLS embedding (e.g. 128-dim behavioral) to d_model.
    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            out
        } else {
            cls.to_vec()
        }
    }

    /// Encode ability token IDs to [CLS] embedding.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Encode v3 game state → entity state with variable-length tokens + positions.
    pub fn encode_entities_v3(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
        positions: &[&[f32]],
    ) -> EntityStateV3 {
        let d = self.d_model;
        let (tokens, mask, n_total, type_ids) = self.entity_encoder.forward(
            entities, entity_types, threats, positions,
        );

        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for i in 0..n_total {
            if mask[i] {
                count += 1.0;
                for j in 0..d {
                    pooled[j] += tokens[i * d + j];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityStateV3 { tokens, mask, pooled, type_ids, n_total }
    }

    /// Compute pointer action logits using v3 entity state.
    pub fn pointer_logits(
        &self,
        entity_state: &EntityStateV3,
        ability_cls: &[Option<&[f32]>],
    ) -> PointerOutput {
        let n_total = entity_state.n_total;

        // Cross-attend each ability CLS to entity tokens
        // NOTE: CLS is already projected to d_model at cache time (project_external_cls
        // is called in transformer_rl.rs during CLS caching), so no projection here.
        let mut ability_cross_embs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for cls_opt in ability_cls.iter().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_total,
                );
                ability_cross_embs.push(Some(cross_emb));
            } else {
                ability_cross_embs.push(None);
            }
        }

        self.pointer_head.forward(
            &entity_state.pooled,
            &entity_state.tokens,
            &entity_state.mask,
            n_total,
            &entity_state.type_ids,
            &ability_cross_embs,
        )
    }
}

/// V3 entity state: tokens + mask + pooled + type IDs.
#[derive(Debug)]
pub struct EntityStateV3 {
    pub tokens: Vec<f32>,
    pub mask: Vec<bool>,
    pub pooled: Vec<f32>,
    pub type_ids: Vec<usize>,
    pub n_total: usize,
}

// ---------------------------------------------------------------------------
// V4: Dual-head actor-critic (directional movement + combat pointer)
// ---------------------------------------------------------------------------

/// V4 output: separate move direction + combat action.
#[derive(Debug)]
pub struct DualHeadOutput {
    /// Movement direction logits (9 elements: 8 cardinal + stay).
    pub move_logits: Vec<f32>,
    /// Combat type logits (10 elements: attack(0), hold(1), ability_0..7(2..9)).
    pub combat_logits: Vec<f32>,
    /// Attack pointer logits over all tokens.
    pub attack_ptr: Vec<f32>,
    /// Per-ability pointer logits (None if ability slot empty).
    pub ability_ptrs: Vec<Option<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct CombatHeadJson {
    combat_type_head: HeadJson,
    pointer_key: LinearWeights,
    attack_query: LinearWeights,
    ability_queries: Vec<LinearWeights>,
}

#[derive(Debug, Clone)]
struct FlatCombatHead {
    combat_type_l1: FlatLinear,
    combat_type_l2: FlatLinear,
    pointer_key: FlatLinear,
    attack_query: FlatLinear,
    ability_queries: Vec<FlatLinear>,
    d_model: usize,
    scale: f32,
}

impl FlatCombatHead {
    fn from_json(j: &CombatHeadJson, d_model: usize) -> Self {
        Self {
            combat_type_l1: FlatLinear::from_json(&j.combat_type_head.linear1),
            combat_type_l2: FlatLinear::from_json(&j.combat_type_head.linear2),
            pointer_key: FlatLinear::from_json(&j.pointer_key),
            attack_query: FlatLinear::from_json(&j.attack_query),
            ability_queries: j.ability_queries.iter().map(FlatLinear::from_json).collect(),
            d_model,
            scale: (d_model as f32).sqrt().recip(),
        }
    }

    fn forward(
        &self,
        pooled: &[f32],
        tokens: &[f32],
        mask: &[bool],
        n_total: usize,
        type_ids: &[usize],
        ability_cross_embs: &[Option<Vec<f32>>],
    ) -> (Vec<f32>, Vec<f32>, Vec<Option<Vec<f32>>>) {
        let d = self.d_model;

        // Combat type logits
        let mut h = vec![0.0f32; d];
        self.combat_type_l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_types = self.combat_type_l2.out_dim;
        let mut combat_logits = vec![0.0f32; n_types];
        self.combat_type_l2.forward(&h, &mut combat_logits);

        // Compute keys for all tokens
        let mut keys = vec![0.0f32; n_total * d];
        for i in 0..n_total {
            self.pointer_key.forward(
                &tokens[i * d..(i + 1) * d],
                &mut keys[i * d..(i + 1) * d],
            );
        }

        // Attack pointer: query from pooled, masked to enemies only (type_id == 1)
        let mut atk_q = vec![0.0f32; d];
        self.attack_query.forward(pooled, &mut atk_q);
        let mut attack_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] == 1 {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += atk_q[j] * keys[i * d + j];
                }
                attack_ptr[i] = dot * self.scale;
            }
        }

        // Ability pointers
        let mut ability_ptrs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for (ab_idx, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if ab_idx < self.ability_queries.len() {
                    let mut ab_q = vec![0.0f32; d];
                    self.ability_queries[ab_idx].forward(cross_emb, &mut ab_q);
                    let mut ab_ptr = vec![f32::NEG_INFINITY; n_total];
                    for i in 0..n_total {
                        if mask[i] {
                            let mut dot = 0.0f32;
                            for j in 0..d {
                                dot += ab_q[j] * keys[i * d + j];
                            }
                            ab_ptr[i] = dot * self.scale;
                        }
                    }
                    ability_ptrs.push(Some(ab_ptr));
                } else {
                    ability_ptrs.push(None);
                }
            } else {
                ability_ptrs.push(None);
            }
        }

        (combat_logits, attack_ptr, ability_ptrs)
    }
}

#[derive(Debug, Clone)]
struct FlatMoveHead {
    l1: FlatLinear,
    l2: FlatLinear,
}

impl FlatMoveHead {
    fn from_json(j: &HeadJson) -> Self {
        Self {
            l1: FlatLinear::from_json(&j.linear1),
            l2: FlatLinear::from_json(&j.linear2),
        }
    }

    fn forward(&self, pooled: &[f32]) -> Vec<f32> {
        let d = self.l1.out_dim;
        let mut h = vec![0.0f32; d];
        self.l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_out = self.l2.out_dim;
        let mut out = vec![0.0f32; n_out];
        self.l2.forward(&h, &mut out);
        out
    }
}

#[derive(Debug, Deserialize)]
struct ActorCriticV4FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v3: EntityEncoderV3Json,
    cross_attn: CrossAttnJson,
    move_head: HeadJson,
    combat_head: CombatHeadJson,
    external_cls_proj: Option<LinearWeights>,
}

/// V4 actor-critic weights with dual-head action space.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV4 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoderV3,
    cross_attn: FlatCrossAttention,
    move_head: FlatMoveHead,
    combat_head: FlatCombatHead,
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeightsV4 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV4FileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err("token_emb size mismatch".to_string());
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoderV3::from_json(
            &file.entity_encoder_v3, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );
        let move_head = FlatMoveHead::from_json(&file.move_head);
        let combat_head = FlatCombatHead::from_json(&file.combat_head, arch.d_model);

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            move_head,
            combat_head,
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read V4 weights: {e}"))?;
        Self::from_json(&data)
    }

    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            out
        } else {
            cls.to_vec()
        }
    }

    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    pub fn encode_entities_v3(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
        positions: &[&[f32]],
    ) -> EntityStateV3 {
        let d = self.d_model;
        let (tokens, mask, n_total, type_ids) = self.entity_encoder.forward(
            entities, entity_types, threats, positions,
        );

        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for i in 0..n_total {
            if mask[i] {
                count += 1.0;
                for j in 0..d {
                    pooled[j] += tokens[i * d + j];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityStateV3 { tokens, mask, pooled, type_ids, n_total }
    }

    /// Compute dual-head action logits: move direction + combat pointer.
    pub fn dual_head_logits(
        &self,
        entity_state: &EntityStateV3,
        ability_cls: &[Option<&[f32]>],
    ) -> DualHeadOutput {
        let n_total = entity_state.n_total;

        // Cross-attend each ability CLS to entity tokens
        let mut ability_cross_embs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for cls_opt in ability_cls.iter().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_total,
                );
                ability_cross_embs.push(Some(cross_emb));
            } else {
                ability_cross_embs.push(None);
            }
        }

        // Move direction logits (9-way)
        let move_logits = self.move_head.forward(&entity_state.pooled);

        // Combat pointer logits
        let (combat_logits, attack_ptr, ability_ptrs) = self.combat_head.forward(
            &entity_state.pooled,
            &entity_state.tokens,
            &entity_state.mask,
            n_total,
            &entity_state.type_ids,
            &ability_cross_embs,
        );

        DualHeadOutput { move_logits, combat_logits, attack_ptr, ability_ptrs }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_layer_norm() {
        let ln = FlatLayerNorm {
            dim: 4,
            gamma: vec![1.0; 4],
            beta: vec![0.0; 4],
        };
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        ln.forward(&mut x);
        // Mean should be ~0, std ~1
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_load_and_predict() {
        // Load exported weights from Python
        let json_path = "generated/ability_transformer_weights.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found (run export_weights.py first)");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = AbilityTransformerWeights::from_json(&json_str).unwrap();

        // Fireball ability token IDs (from Python tokenizer)
        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155, 14, 232, 63, 14, 249, 15, 38, 14, 248, 96, 14, 68, 68, 239, 12, 7, 14, 239, 13, 9];
        let out = model.predict(&tokens, None);
        assert!(out.urgency >= 0.0 && out.urgency <= 1.0, "urgency={}", out.urgency);
        assert_eq!(out.target_logits.len(), 3);
        println!("urgency={:.6}, targets={:?}", out.urgency, out.target_logits);
        // Compare with Python output: urgency=0.526941, targets=[2.69, -1.68, -2.58]
        assert!((out.urgency - 0.527).abs() < 0.02, "urgency mismatch: {}", out.urgency);
    }

    #[test]
    fn test_load_and_predict_cross_attn() {
        let json_path = "generated/ability_transformer_weights_xattn.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = match AbilityTransformerWeights::from_json(&json_str) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Skipping: {json_path} failed to parse (needs re-export): {e}");
                return;
            }
        };

        // Fireball tokens
        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155, 14, 232, 63, 14, 249, 15, 38, 14, 248, 96, 14, 68, 68, 239, 12, 7, 14, 239, 13, 9];

        // Game state: 210 floats (7 entities × 30 features)
        let mut gs = vec![0.0f32; 210];
        // Self entity: hp=1.0, exists=1.0 (index 29)
        gs[0] = 1.0;   // hp_pct
        gs[12] = 0.5;  // auto_dps
        gs[29] = 1.0;  // exists
        // Enemy 0: hp=0.5, exists=1.0
        gs[30] = 0.5;
        gs[42] = 0.8;
        gs[59] = 1.0;
        // Enemy 1: hp=1.0, exists=1.0
        gs[60] = 1.0;
        gs[72] = 0.4;
        gs[89] = 1.0;
        // Ally 0: hp=0.3, exists=1.0
        gs[120] = 0.3;
        gs[149] = 1.0;

        let out = model.predict(&tokens, Some(&gs));
        println!("Rust urgency: {:.6}, targets: {:?}", out.urgency, out.target_logits);

        // Just verify it loads and produces valid output
        assert!(
            out.urgency >= 0.0 && out.urgency <= 1.0,
            "urgency out of range: {}",
            out.urgency
        );
        assert_eq!(out.target_logits.len(), 3);
    }

    #[test]
    fn test_batch_api_matches_single() {
        let json_path = "generated/ability_transformer_weights_xattn.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = match AbilityTransformerWeights::from_json(&json_str) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Skipping: parse error (needs re-export): {e}");
                return;
            }
        };

        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155];
        let tokens2: Vec<u32> = vec![1, 1, 63, 14, 249, 15, 38, 14, 248, 96];

        let mut gs = vec![0.0f32; 210];
        gs[0] = 1.0; gs[29] = 1.0;
        gs[30] = 0.5; gs[59] = 1.0;

        // Single predict
        let out1 = model.predict(&tokens, Some(&gs));
        let out2 = model.predict(&tokens2, Some(&gs));

        // Batch predict
        let batch = model.predict_batch(&[&tokens, &tokens2], Some(&gs));
        assert_eq!(batch.len(), 2);
        assert!((batch[0].urgency - out1.urgency).abs() < 1e-6,
            "batch[0] urgency mismatch: {} vs {}", batch[0].urgency, out1.urgency);
        assert!((batch[1].urgency - out2.urgency).abs() < 1e-6,
            "batch[1] urgency mismatch: {} vs {}", batch[1].urgency, out2.urgency);
    }

    #[test]
    fn test_flat_linear() {
        // Identity-ish: 2→2 with identity weights
        let fl = FlatLinear {
            in_dim: 2,
            out_dim: 2,
            w: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![0.0, 0.0],
        };
        let mut out = vec![0.0; 2];
        fl.forward(&[3.0, 5.0], &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_v3_weights() {
        let json_path = "/tmp/test_v3_weights.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found (run export_actor_critic_v3.py first)");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = ActorCriticWeightsV3::from_json(&json_str).unwrap();

        // Test entity encoding with positions
        let self_ent = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0];
        let enemy = vec![0.8, 0.0, 1.0, 0.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0,
                         0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 1.0];
        let pos = vec![0.1, 0.0, 0.15, 0.2, 0.33, 0.6, 0.0, 0.0];

        let entities: Vec<&[f32]> = vec![&self_ent, &enemy];
        let entity_types = vec![0, 1];
        let threats: Vec<&[f32]> = Vec::new();
        let positions: Vec<&[f32]> = vec![&pos];

        let state = model.encode_entities_v3(&entities, &entity_types, &threats, &positions);
        assert_eq!(state.n_total, 3); // 2 entities + 0 threats + 1 position
        assert_eq!(state.type_ids, vec![0, 1, 4]);

        // Test pointer logits
        let ability_cls: Vec<Option<&[f32]>> = vec![None; 8];
        let ptr_out = model.pointer_logits(&state, &ability_cls);
        assert_eq!(ptr_out.type_logits.len(), NUM_ACTION_TYPES);
        assert_eq!(ptr_out.attack_ptr.len(), 3);
        assert_eq!(ptr_out.move_ptr.len(), 3);

        // Attack mask: only enemy (index 1) should be valid
        assert!(ptr_out.attack_ptr[0] < -1e8, "self should be masked");
        assert!(ptr_out.attack_ptr[1] > -1e8, "enemy should be unmasked");
        assert!(ptr_out.attack_ptr[2] < -1e8, "position should be masked for attack");

        // Move mask: self (index 0) should be masked, rest valid
        assert!(ptr_out.move_ptr[0] < -1e8, "self should be masked for move");
        assert!(ptr_out.move_ptr[1] > -1e8, "enemy should be valid move target");
        assert!(ptr_out.move_ptr[2] > -1e8, "position should be valid move target");

        println!("V3 type_logits: {:?}", ptr_out.type_logits);
        println!("V3 attack_ptr: {:?}", ptr_out.attack_ptr);
        println!("V3 move_ptr: {:?}", ptr_out.move_ptr);
    }
}

// ---------------------------------------------------------------------------
// Embedding Registry — pre-computed CLS lookup by ability name
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct EmbeddingRegistryJson {
    model_hash: String,
    d_model: usize,
    n_abilities: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
    #[serde(default)]
    outcome_mean: Vec<f32>,
    #[serde(default)]
    outcome_std: Vec<f32>,
}

/// Pre-computed CLS embeddings for known abilities.
///
/// Runtime = HashMap lookup, no transformer needed.
#[derive(Debug, Clone)]
pub struct EmbeddingRegistry {
    pub model_hash: String,
    pub d_model: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
}

impl EmbeddingRegistry {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: EmbeddingRegistryJson =
            serde_json::from_str(json_str).map_err(|e| format!("Registry parse error: {e}"))?;
        assert_eq!(
            file.embeddings.len(),
            file.n_abilities,
            "n_abilities mismatch"
        );
        Ok(Self {
            model_hash: file.model_hash,
            d_model: file.d_model,
            embeddings: file.embeddings,
        })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read registry: {e}"))?;
        Self::from_json(&data)
    }

    /// Look up CLS embedding by ability name. Returns None if not found.
    pub fn get(&self, ability_name: &str) -> Option<&[f32]> {
        self.embeddings.get(ability_name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }
}
