//! Autoregressive tree decoder for grammar-constrained ability generation.
//!
//! Generates grammar space vectors one dimension at a time, where each
//! decision conditions on all previous decisions + text encoding.
//! Cross-attention on text encoder outputs at every step.
//!
//! Architecture:
//!   Text → TextEncoder → memory [T, D]
//!   For each grammar dim i:
//!     [prev_decisions] → TransformerDecoder (cross-attn on memory) → logits for dim[i]
//!     Sample or argmax → dim[i] value
//!     Append to sequence
//!
//! Each dimension is discretized into N_BINS values. The model predicts
//! a categorical distribution over bins, then we convert back to [0,1].

use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig,
    transformer::{
        TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput,
    },
};
use burn::prelude::*;
use burn::tensor::activation;

use super::grammar_space::GRAMMAR_DIM;
use super::text_encoder::STATIC_EMBED_DIM;

/// Number of bins for categorical dimensions.
const N_BINS: usize = 32;

/// Model dimension.
const D_MODEL: usize = 256;
const N_HEADS: usize = 8;
const N_LAYERS: usize = 4;
const D_FF: usize = 512;

/// Per-dimension prediction type.
#[derive(Clone, Copy, PartialEq)]
pub enum DimType {
    /// Predict via softmax over N bins (targeting, hint, effect type, etc.)
    Categorical(usize), // number of valid bins
    /// Predict via regression to [0,1] (range, cooldown, params, etc.)
    Continuous,
}

/// Map each grammar space dimension to its prediction type.
/// Categorical dims use softmax, continuous dims use regression.
pub fn dim_types() -> [DimType; GRAMMAR_DIM] {
    use DimType::*;
    [
        // Header (8 dims)
        Categorical(2),   // 0: type (active/passive)
        Categorical(2),   // 1: domain (combat/campaign)
        Categorical(15),  // 2: targeting mode
        Continuous,       // 3: range
        Continuous,       // 4: cooldown
        Continuous,       // 5: cast time
        Categorical(9),   // 6: hint
        Continuous,       // 7: cost
        // Delivery (4 dims)
        Categorical(7),   // 8: delivery type
        Continuous,       // 9: delivery param 0
        Continuous,       // 10: delivery param 1
        Categorical(4),   // 11: n_effects (1-4)
        // Effect 0 (8 dims)
        Categorical(N_BINS), // 12: effect 0 type (many effects, keep full bins)
        Continuous,       // 13: effect 0 param
        Continuous,       // 14: effect 0 duration
        Categorical(7),   // 15: effect 0 area type
        Continuous,       // 16: effect 0 area param
        Categorical(10),  // 17: effect 0 tag
        Continuous,       // 18: effect 0 tag power
        Categorical(10),  // 19: effect 0 condition
        // Effect 1 (8 dims)
        Categorical(N_BINS), // 20
        Continuous,       // 21
        Continuous,       // 22
        Categorical(7),   // 23
        Continuous,       // 24
        Categorical(10),  // 25
        Continuous,       // 26
        Categorical(10),  // 27
        // Effect 2 (8 dims)
        Categorical(N_BINS), // 28
        Continuous,       // 29
        Continuous,       // 30
        Categorical(7),   // 31
        Continuous,       // 32
        Categorical(10),  // 33
        Continuous,       // 34
        Categorical(10),  // 35
        // Effect 3 (8 dims)
        Categorical(N_BINS), // 36
        Continuous,       // 37
        Continuous,       // 38
        Categorical(7),   // 39
        Continuous,       // 40
        Categorical(10),  // 41
        Continuous,       // 42
        Categorical(10),  // 43
        // Passive/scaling (4 dims)
        Categorical(10),  // 44: trigger type
        Continuous,       // 45: trigger param
        Categorical(8),   // 46: scaling stat
        Continuous,       // 47: scaling pct
    ]
}

/// Max bins across all categorical dims.
const MAX_CAT_BINS: usize = N_BINS;

// ---------------------------------------------------------------------------
// Tree Decoder Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct TreeDecoder<B: Backend> {
    /// Embedding for discretized grammar values (bin IDs).
    bin_emb: Embedding<B>,
    /// Embedding for which dimension we're predicting (position in the tree).
    dim_emb: Embedding<B>,
    /// Project text encoder output to decoder's d_model.
    memory_proj: Linear<B>,
    /// Causal transformer decoder with cross-attention on text memory.
    decoder: TransformerDecoder<B>,
    /// Categorical output head: predict bin logits.
    cat_head: Linear<B>,
    /// Continuous output head: predict scalar [0,1].
    cont_head: Linear<B>,
    /// Auxiliary classification heads on text encoder output.
    cls_hint: Linear<B>,
    cls_element: Linear<B>,
    cls_targeting: Linear<B>,
    cls_domain: Linear<B>,
}

impl<B: Backend> TreeDecoder<B> {
    pub fn new(text_dim: usize, device: &B::Device) -> Self {
        let decoder = TransformerDecoderConfig::new(D_MODEL, D_FF, N_HEADS, N_LAYERS)
            .with_dropout(0.1)
            .init(device);

        Self {
            bin_emb: EmbeddingConfig::new(N_BINS + 1, D_MODEL).init(device), // +1 for [START]
            dim_emb: EmbeddingConfig::new(GRAMMAR_DIM + 1, D_MODEL).init(device),
            memory_proj: LinearConfig::new(text_dim, D_MODEL).init(device),
            decoder,
            cat_head: LinearConfig::new(D_MODEL, MAX_CAT_BINS).init(device),
            cont_head: LinearConfig::new(D_MODEL, 1).init(device),
            cls_hint: LinearConfig::new(text_dim, 9).init(device),
            cls_element: LinearConfig::new(text_dim, 8).init(device),
            cls_targeting: LinearConfig::new(text_dim, 15).init(device),
            cls_domain: LinearConfig::new(text_dim, 2).init(device),
        }
    }

    /// Teacher-forced forward pass for training.
    ///
    /// Returns (cat_logits [B, GRAMMAR_DIM, MAX_CAT_BINS], cont_preds [B, GRAMMAR_DIM])
    /// — categorical logits and continuous regression predictions for each dim.
    pub fn forward_train(
        &self,
        target_bins: Tensor<B, 2, Int>,  // [B, GRAMMAR_DIM]
        text_memory: Tensor<B, 3>,       // [B, T, text_dim]
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, seq_len] = target_bins.dims();
        let device = target_bins.device();

        let memory = self.memory_proj.forward(text_memory);

        // Shift right: prepend [START] token
        let start_token = Tensor::<B, 2, Int>::full([batch, 1], N_BINS as i64, &device);
        let shifted = Tensor::cat(vec![start_token, target_bins.slice([0..batch, 0..seq_len - 1])], 1);

        let tok_emb = self.bin_emb.forward(shifted);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch as i64, seq_len as i64]);
        let pos_emb = self.dim_emb.forward(positions);
        let target_emb = tok_emb + pos_emb;

        let causal_mask = burn::nn::attention::generate_autoregressive_mask::<B>(
            batch, seq_len, &device,
        );
        let dec_input = TransformerDecoderInput::new(target_emb, memory)
            .target_mask_attn(causal_mask);
        let decoded = self.decoder.forward(dec_input); // [B, GRAMMAR_DIM, D_MODEL]

        // Two heads: categorical logits + continuous scalar
        let cat_logits = self.cat_head.forward(decoded.clone()); // [B, GRAMMAR_DIM, MAX_CAT_BINS]
        let cont_preds = self.cont_head.forward(decoded)
            .reshape([batch, seq_len]); // [B, GRAMMAR_DIM] — sigmoid to [0,1]
        let cont_preds = burn::tensor::activation::sigmoid(cont_preds);

        (cat_logits, cont_preds)
    }

    /// Autoregressive generation with KV cache.
    ///
    /// Uses burn's built-in autoregressive cache to avoid recomputing
    /// previous positions. O(T) decoder passes instead of O(T²).
    ///
    /// `text_memory`: [B, T, text_dim] — text encoder outputs
    /// `temperature`: sampling temperature (0 = greedy, >0 = stochastic)
    pub fn generate(
        &self,
        text_memory: Tensor<B, 3>,
        temperature: f32,
    ) -> (Vec<Vec<u32>>, Vec<[f32; GRAMMAR_DIM]>) {
        let [batch, _t, _d] = text_memory.dims();
        let device = text_memory.device();

        let memory = self.memory_proj.forward(text_memory);

        let mut generated_bins: Vec<Vec<u32>> = (0..batch).map(|_| Vec::with_capacity(GRAMMAR_DIM)).collect();

        // Build sequence incrementally: start with [START]
        let mut current_seq = Tensor::<B, 2, Int>::full([batch, 1], N_BINS as i64, &device);

        for step in 0..GRAMMAR_DIM {
            let slen = step + 1;

            // Embed full sequence + positions
            let tok_emb = self.bin_emb.forward(current_seq.clone());
            let pos_ids = Tensor::<B, 1, Int>::arange(0..slen as i64, &device)
                .unsqueeze::<2>().expand([batch as i64, slen as i64]);
            let pos_emb = self.dim_emb.forward(pos_ids);
            let target_emb = tok_emb + pos_emb;

            // Decoder forward (full recompute — burn 0.20 KV cache is broken)
            let dec_input = TransformerDecoderInput::new(target_emb, memory.clone());
            let decoded = self.decoder.forward(dec_input);
            let last = decoded.slice([0..batch, step..step + 1]).reshape([batch, D_MODEL]);

            // Use the right head based on dim type
            let dtypes = dim_types();
            let next_bins = match dtypes[step] {
                DimType::Categorical(n_bins) => {
                    let logits = self.cat_head.forward(last)
                        .slice([0..batch, 0..n_bins]); // [B, n_bins]
                    if temperature <= 0.0 {
                        logits.argmax(1)
                    } else {
                        let scaled = logits / temperature;
                        let probs = activation::softmax(scaled, 1);
                        let gumbel = Tensor::<B, 2>::random(
                            [batch, n_bins],
                            burn::tensor::Distribution::Uniform(1e-10, 1.0),
                            &device,
                        ).log().neg().log().neg();
                        (probs.log() + gumbel).argmax(1)
                    }
                }
                DimType::Continuous => {
                    let pred = self.cont_head.forward(last).reshape([batch]); // [B]
                    let pred = burn::tensor::activation::sigmoid(pred);
                    // Convert continuous [0,1] to bin index
                    let bins = (pred * N_BINS as f32).int();
                    bins.clamp(0, N_BINS as i64 - 1).unsqueeze_dim::<2>(1) // [B, 1]
                }
            };

            let next_data: Vec<i64> = next_bins.clone().reshape([batch]).to_data().to_vec().unwrap();
            for (bi, &bin) in next_data.iter().enumerate() {
                generated_bins[bi].push(bin as u32);
            }

            // Append to sequence
            current_seq = Tensor::cat(vec![current_seq, next_bins.reshape([batch, 1])], 1);
        }

        // Convert bins to continuous [0,1] values
        let mut results = Vec::with_capacity(batch);
        for bins in &generated_bins {
            let mut v = [0.0f32; GRAMMAR_DIM];
            for (d, &bin) in bins.iter().enumerate() {
                v[d] = (bin as f32 + 0.5) / N_BINS as f32;
            }
            results.push(v);
        }

        (generated_bins, results)
    }
}

// ---------------------------------------------------------------------------
// Training loss
// ---------------------------------------------------------------------------

pub struct TreeDecoderOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub loss_val: f64,
    pub accuracy: f64,
    pub cls_loss_val: f64,
}

/// Precomputed dim indices for fast loss computation.
pub struct DimIndices {
    pub cat_indices: Vec<i64>,  // which dims are categorical
    pub cont_indices: Vec<i64>, // which dims are continuous
}

impl DimIndices {
    pub fn new() -> Self {
        let dtypes = dim_types();
        let mut cat = Vec::new();
        let mut cont = Vec::new();
        for d in 0..GRAMMAR_DIM {
            match dtypes[d] {
                DimType::Categorical(_) => cat.push(d as i64),
                DimType::Continuous => cont.push(d as i64),
            }
        }
        Self { cat_indices: cat, cont_indices: cont }
    }
}

/// Compute mixed loss: CE for categorical dims, MSE for continuous dims.
/// Uses index_select to pull out cat/cont subsets — no per-dim loops, no masks.
pub fn tree_decoder_loss<B: Backend>(
    decoder: &TreeDecoder<B>,
    target_bins: Tensor<B, 2, Int>,    // [B, GRAMMAR_DIM]
    target_continuous: Tensor<B, 2>,   // [B, GRAMMAR_DIM]
    text_memory: Tensor<B, 3>,
    text_cls: Tensor<B, 2>,
    target_labels: Tensor<B, 2, Int>,  // [B, 4]
) -> TreeDecoderOutput<B> {
    let [batch, _seq_len] = target_bins.dims();
    let device = target_bins.device();

    let (cat_logits, cont_preds) = decoder.forward_train(target_bins.clone(), text_memory);
    // cat_logits: [B, 48, MAX_CAT_BINS], cont_preds: [B, 48]

    let indices = DimIndices::new();
    let n_cat = indices.cat_indices.len();
    let n_cont = indices.cont_indices.len();

    // --- Categorical loss: select cat dims, compute CE ---
    let cat_idx = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(indices.cat_indices.clone(), [n_cat]), &device,
    );
    // Select categorical dim logits: [B, n_cat, MAX_CAT_BINS]
    let cat_logits_sel = cat_logits.clone().select(1, cat_idx.clone());
    let cat_targets_sel = target_bins.clone().select(1, cat_idx); // [B, n_cat]

    let cat_logits_flat = cat_logits_sel.reshape([batch * n_cat, MAX_CAT_BINS]);
    let cat_targets_flat = cat_targets_sel.reshape([batch * n_cat]);

    let log_probs = activation::log_softmax(cat_logits_flat.clone(), 1);
    let target_lp = log_probs.gather(1, cat_targets_flat.clone().unsqueeze_dim::<2>(1));
    let cat_loss = target_lp.neg().mean();

    // Accuracy (categorical only)
    let preds = cat_logits_flat.argmax(1).reshape([batch * n_cat]);
    let accuracy: f64 = preds.equal(cat_targets_flat).float().mean().into_scalar().elem();

    // --- Continuous loss: select cont dims, compute MSE ---
    let cont_idx = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(indices.cont_indices.clone(), [n_cont]), &device,
    );
    let cont_preds_sel = cont_preds.select(1, cont_idx.clone()); // [B, n_cont]
    let cont_targets_sel = target_continuous.select(1, cont_idx); // [B, n_cont]

    let cont_diff = cont_preds_sel - cont_targets_sel;
    let cont_loss = (cont_diff.clone() * cont_diff).mean();

    // Combined
    let ar_loss = cat_loss + cont_loss * 10.0;
    let ar_loss_val: f64 = ar_loss.clone().into_scalar().elem();

    // --- Auxiliary classification ---
    let hint_logits = decoder.cls_hint.forward(text_cls.clone());
    let elem_logits = decoder.cls_element.forward(text_cls.clone());
    let tgt_logits = decoder.cls_targeting.forward(text_cls.clone());
    let dom_logits = decoder.cls_domain.forward(text_cls);

    let hint_labels = target_labels.clone().slice([0..batch, 0..1]).reshape([batch]);
    let elem_labels = target_labels.clone().slice([0..batch, 1..2]).reshape([batch]);
    let tgt_labels = target_labels.clone().slice([0..batch, 2..3]).reshape([batch]);
    let dom_labels = target_labels.slice([0..batch, 3..4]).reshape([batch]);

    let cls_loss = ce_loss(hint_logits, hint_labels)
        + ce_loss(elem_logits, elem_labels)
        + ce_loss(tgt_logits, tgt_labels)
        + ce_loss(dom_logits, dom_labels);
    let cls_loss = cls_loss * 0.25;
    let cls_loss_val: f64 = cls_loss.clone().into_scalar().elem();

    let total_loss = ar_loss + cls_loss * 0.5;

    TreeDecoderOutput {
        loss: total_loss.unsqueeze(),
        loss_val: ar_loss_val,
        accuracy,
        cls_loss_val,
    }
}

fn ce_loss<B: Backend>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    let log_probs = activation::log_softmax(logits, 1);
    let target_lp = log_probs.gather(1, targets.unsqueeze_dim::<2>(1));
    target_lp.neg().mean().unsqueeze()
}

// ---------------------------------------------------------------------------
// Utility: continuous [0,1] → bin IDs
// ---------------------------------------------------------------------------

/// Convert a continuous grammar space vector to discretized bin IDs.
pub fn to_bins(v: &[f32; GRAMMAR_DIM]) -> Vec<i64> {
    v.iter()
        .map(|&x| ((x.clamp(0.0, 0.9999) * N_BINS as f32) as i64).min(N_BINS as i64 - 1))
        .collect()
}

/// Convert bin IDs back to continuous [0,1].
pub fn from_bins(bins: &[u32]) -> [f32; GRAMMAR_DIM] {
    let mut v = [0.0f32; GRAMMAR_DIM];
    for (d, &b) in bins.iter().enumerate().take(GRAMMAR_DIM) {
        v[d] = (b as f32 + 0.5) / N_BINS as f32;
    }
    v
}

pub const TREE_N_BINS: usize = N_BINS;
pub const TREE_D_MODEL: usize = D_MODEL;
