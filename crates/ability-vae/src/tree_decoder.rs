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

/// Number of bins per grammar dimension.
/// Higher = finer resolution but harder to learn.
const N_BINS: usize = 32;

/// Model dimension.
const D_MODEL: usize = 256;
const N_HEADS: usize = 8;
const N_LAYERS: usize = 4;
const D_FF: usize = 512;

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
    /// Output head: predict bin logits for next dimension.
    output_head: Linear<B>,
    /// Auxiliary classification heads on text encoder output.
    /// These force the encoder to extract semantically meaningful features.
    cls_hint: Linear<B>,      // predict hint category (damage/heal/cc/defense/utility/economy/diplomacy)
    cls_element: Linear<B>,   // predict element (none/physical/magic/fire/ice/dark/holy/poison)
    cls_targeting: Linear<B>, // predict targeting mode
    cls_domain: Linear<B>,    // predict combat vs campaign
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
            output_head: LinearConfig::new(D_MODEL, N_BINS).init(device),
            cls_hint: LinearConfig::new(text_dim, 9).init(device),       // 9 hint categories
            cls_element: LinearConfig::new(text_dim, 8).init(device),    // 8 elements (incl. none)
            cls_targeting: LinearConfig::new(text_dim, 15).init(device), // 15 targeting modes
            cls_domain: LinearConfig::new(text_dim, 2).init(device),     // combat vs campaign
        }
    }

    /// Teacher-forced forward pass for training.
    ///
    /// `target_bins`: [B, GRAMMAR_DIM] — ground truth bin IDs (0..N_BINS-1)
    /// `text_memory`: [B, T, text_dim] — text encoder token-level outputs
    ///
    /// Returns logits [B, GRAMMAR_DIM, N_BINS] for each dimension.
    pub fn forward_train(
        &self,
        target_bins: Tensor<B, 2, Int>,  // [B, GRAMMAR_DIM]
        text_memory: Tensor<B, 3>,       // [B, T, text_dim]
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = target_bins.dims();
        let device = target_bins.device();

        // Project text memory to d_model
        let memory = self.memory_proj.forward(text_memory); // [B, T, D_MODEL]

        // Build decoder input: shift right (prepend [START] token = N_BINS)
        let start_token = Tensor::<B, 2, Int>::full([batch, 1], N_BINS as i64, &device);
        let shifted = Tensor::cat(vec![start_token, target_bins.slice([0..batch, 0..seq_len - 1])], 1);
        // shifted is [B, GRAMMAR_DIM] — starts with START, then target[0..46]

        // Embed bins + add positional (dimension) embedding
        let tok_emb = self.bin_emb.forward(shifted);
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .expand([batch as i64, seq_len as i64]);
        let pos_emb = self.dim_emb.forward(positions);
        let target_emb = tok_emb + pos_emb; // [B, GRAMMAR_DIM, D_MODEL]

        // Causal mask
        let causal_mask = burn::nn::attention::generate_autoregressive_mask::<B>(
            batch, seq_len, &device,
        );

        // Decode with cross-attention on text memory
        let dec_input = TransformerDecoderInput::new(target_emb, memory)
            .target_mask_attn(causal_mask);
        let decoded = self.decoder.forward(dec_input); // [B, GRAMMAR_DIM, D_MODEL]

        // Predict bin logits for each position
        self.output_head.forward(decoded) // [B, GRAMMAR_DIM, N_BINS]
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
        let mut cache = self.decoder.new_autoregressive_cache();

        // Current token: start with [START]
        let mut current_token = Tensor::<B, 2, Int>::full([batch, 1], N_BINS as i64, &device);

        for step in 0..GRAMMAR_DIM {
            // Embed just the current token
            let tok_emb = self.bin_emb.forward(current_token.clone());
            let pos_id = Tensor::<B, 2, Int>::full([batch, 1], step as i64, &device);
            let pos_emb = self.dim_emb.forward(pos_id);
            let target_emb = tok_emb + pos_emb; // [B, 1, D_MODEL]

            // Cached decoder forward — only processes new token, reuses KV from previous steps
            let dec_input = TransformerDecoderInput::new(target_emb, memory.clone());
            let decoded = self.decoder.forward_autoregressive_inference(dec_input, &mut cache);

            // Cache returns accumulated [B, step+1, D]. Take last position.
            let [_b, seq_so_far, _d] = decoded.dims();
            let last = decoded.slice([0..batch, seq_so_far - 1..seq_so_far])
                .reshape([batch, D_MODEL]);
            let logits = self.output_head.forward(last); // [B, N_BINS]

            // Sample or argmax
            let next_bins = if temperature <= 0.0 {
                logits.argmax(1)
            } else {
                let scaled = logits / temperature;
                let probs = activation::softmax(scaled, 1);
                let gumbel = Tensor::<B, 2>::random(
                    [batch, N_BINS],
                    burn::tensor::Distribution::Uniform(1e-10, 1.0),
                    &device,
                ).log().neg().log().neg();
                (probs.log() + gumbel).argmax(1)
            };

            let next_data: Vec<i64> = next_bins.clone().reshape([batch]).to_data().to_vec().unwrap();
            for (bi, &bin) in next_data.iter().enumerate() {
                generated_bins[bi].push(bin as u32);
            }

            // Next input token
            current_token = next_bins.reshape([batch, 1]);
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

/// Compute cross-entropy loss for tree decoder training.
///
/// `text_cls`: [B, text_dim] — CLS-pooled text encoder output (for auxiliary classification)
/// `target_labels`: [B, 4] — ground truth labels [hint, element, targeting, domain] as bin indices
pub fn tree_decoder_loss<B: Backend>(
    decoder: &TreeDecoder<B>,
    target_bins: Tensor<B, 2, Int>,  // [B, GRAMMAR_DIM]
    text_memory: Tensor<B, 3>,       // [B, T, text_dim]
    text_cls: Tensor<B, 2>,          // [B, text_dim] — for classification heads
    target_labels: Tensor<B, 2, Int>, // [B, 4] — hint, element, targeting, domain
) -> TreeDecoderOutput<B> {
    let [batch, seq_len] = target_bins.dims();

    let logits = decoder.forward_train(target_bins.clone(), text_memory);

    // Reshape for cross-entropy
    let logits_for_acc = logits.clone();
    let logits_flat = logits.reshape([batch * seq_len, N_BINS]);
    let targets_flat = target_bins.reshape([batch * seq_len]);

    // Autoregressive CE loss
    let log_probs = activation::log_softmax(logits_flat, 1);
    let target_lp = log_probs.gather(1, targets_flat.clone().unsqueeze_dim::<2>(1));
    let ar_loss = target_lp.neg().mean();

    // Accuracy
    let predictions = logits_for_acc.reshape([batch * seq_len, N_BINS]).argmax(1).reshape([batch * seq_len]);
    let correct = predictions.equal(targets_flat).float().mean();
    let accuracy: f64 = correct.into_scalar().elem();

    // Auxiliary classification losses on text encoder CLS output
    let hint_logits = decoder.cls_hint.forward(text_cls.clone());       // [B, 9]
    let elem_logits = decoder.cls_element.forward(text_cls.clone());    // [B, 8]
    let tgt_logits = decoder.cls_targeting.forward(text_cls.clone());   // [B, 15]
    let dom_logits = decoder.cls_domain.forward(text_cls);               // [B, 2]

    let hint_labels = target_labels.clone().slice([0..batch, 0..1]).reshape([batch]);
    let elem_labels = target_labels.clone().slice([0..batch, 1..2]).reshape([batch]);
    let tgt_labels = target_labels.clone().slice([0..batch, 2..3]).reshape([batch]);
    let dom_labels = target_labels.slice([0..batch, 3..4]).reshape([batch]);

    let cls_loss = ce_loss(hint_logits, hint_labels, 9)
        + ce_loss(elem_logits, elem_labels, 8)
        + ce_loss(tgt_logits, tgt_labels, 15)
        + ce_loss(dom_logits, dom_labels, 2);
    let cls_loss = cls_loss * 0.25; // average over 4 heads

    let cls_loss_val: f64 = cls_loss.clone().into_scalar().elem();
    let ar_loss_val: f64 = ar_loss.clone().into_scalar().elem();

    // Total: AR loss + 0.5 * classification loss
    let total_loss = ar_loss + cls_loss * 0.5;

    TreeDecoderOutput {
        loss: total_loss.unsqueeze(),
        loss_val: ar_loss_val,
        accuracy,
        cls_loss_val,
    }
}

/// Simple cross-entropy for classification.
fn ce_loss<B: Backend>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>, n_classes: usize) -> Tensor<B, 1> {
    let [batch] = targets.dims();
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
