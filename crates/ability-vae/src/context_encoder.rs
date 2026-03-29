//! Context encoder: structured game state → conditioning vector for tree decoder.
//!
//! Encodes BehaviorLedger, class info, archetype, settlement, and existing
//! abilities into a dense vector that the tree decoder cross-attends to.
//!
//! Input features (fixed size, known semantics):
//!   - BehaviorLedger: 24 floats (12 lifetime + 12 recent)
//!   - Class: hash embedding (32d) + level/100 + tier/7
//!   - Archetype: hash embedding (32d)
//!   - Settlement context: variable-length (tag, value) pairs → pooled (16d)
//!   - Existing abilities: set of hashes → pooled embedding (16d)
//!   - Narrative: tension, near_death_count, consecutive_retreats (3 floats)
//!
//! Total raw input: ~125 dims → 2-layer transformer → 128-dim conditioning

use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::prelude::*;

/// Conditioning dimension output.
pub const COND_DIM: usize = 128;

/// Number of hash embedding buckets (for class names, archetypes, ability hashes).
const HASH_BUCKETS: usize = 512;
/// Embedding dimension for hash-based lookups.
const HASH_EMB_DIM: usize = 32;

/// Number of behavior ledger features.
const BEHAVIOR_DIM: usize = 24;
/// Number of scalar features (level, tier, tension, near_death, retreats).
const SCALAR_DIM: usize = 5;
/// Max settlement context tags.
const MAX_SETTLEMENT: usize = 8;
/// Max existing abilities for set embedding.
const MAX_EXISTING: usize = 16;

/// Total feature groups fed as a sequence to the transformer encoder.
/// Each group becomes one "token" in the sequence:
///   [behavior(24→64), class_emb(32), archetype_emb(32), scalars(5→64),
///    settlement_pool(16→64), existing_pool(16→64)]
/// = 6 tokens of 64 dims each
const N_FEATURE_GROUPS: usize = 6;
const GROUP_DIM: usize = 64;

#[derive(Module, Debug)]
pub struct ContextEncoder<B: Backend> {
    // Hash embeddings
    class_emb: Embedding<B>,
    archetype_emb: Embedding<B>,
    ability_emb: Embedding<B>,     // for existing abilities set
    settlement_emb: Embedding<B>,  // for settlement context tags

    // Projections: map each feature group to GROUP_DIM
    behavior_proj: Linear<B>,     // 24 → 64
    scalar_proj: Linear<B>,       // 5 → 64
    class_proj: Linear<B>,        // 32 → 64
    archetype_proj: Linear<B>,    // 32 → 64
    settlement_proj: Linear<B>,   // 32 → 64 (after pooling)
    existing_proj: Linear<B>,     // 32 → 64 (after pooling)

    // Transformer: self-attention over feature groups
    encoder: TransformerEncoder<B>,

    // Output: pool → conditioning vector
    output_proj: Linear<B>,       // 64 → COND_DIM
    output_norm: LayerNorm<B>,
}

impl<B: Backend> ContextEncoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let encoder = TransformerEncoderConfig::new(GROUP_DIM, GROUP_DIM * 2, 4, 2)
            .with_dropout(0.1)
            .init(device);

        Self {
            class_emb: EmbeddingConfig::new(HASH_BUCKETS, HASH_EMB_DIM).init(device),
            archetype_emb: EmbeddingConfig::new(HASH_BUCKETS, HASH_EMB_DIM).init(device),
            ability_emb: EmbeddingConfig::new(HASH_BUCKETS, HASH_EMB_DIM).init(device),
            settlement_emb: EmbeddingConfig::new(HASH_BUCKETS, HASH_EMB_DIM / 2).init(device),

            behavior_proj: LinearConfig::new(BEHAVIOR_DIM, GROUP_DIM).init(device),
            scalar_proj: LinearConfig::new(SCALAR_DIM, GROUP_DIM).init(device),
            class_proj: LinearConfig::new(HASH_EMB_DIM, GROUP_DIM).init(device),
            archetype_proj: LinearConfig::new(HASH_EMB_DIM, GROUP_DIM).init(device),
            settlement_proj: LinearConfig::new(HASH_EMB_DIM, GROUP_DIM).init(device),
            existing_proj: LinearConfig::new(HASH_EMB_DIM, GROUP_DIM).init(device),

            encoder,
            output_proj: LinearConfig::new(GROUP_DIM, COND_DIM).init(device),
            output_norm: LayerNormConfig::new(COND_DIM).init(device),
        }
    }

    /// Encode a batch of game contexts into conditioning vectors.
    ///
    /// Returns [B, COND_DIM] conditioning + [B, N_FEATURE_GROUPS, GROUP_DIM] memory
    /// for cross-attention in the tree decoder.
    pub fn forward(&self, input: &ContextInput<B>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let [batch] = input.behavior.dims()[..1] else { panic!("bad batch dim") };
        let device = input.behavior.device();

        // --- Project each feature group to GROUP_DIM ---

        // Behavior: [B, 24] → [B, 64]
        let behavior = burn::tensor::activation::gelu(
            self.behavior_proj.forward(input.behavior.clone())
        );

        // Class embedding: hash → [B, 32] → [B, 64]
        let class = self.class_emb.forward(input.class_hash.clone())
            .reshape([batch, HASH_EMB_DIM]);
        let class = burn::tensor::activation::gelu(self.class_proj.forward(class));

        // Archetype embedding: hash → [B, 32] → [B, 64]
        let archetype = self.archetype_emb.forward(input.archetype_hash.clone())
            .reshape([batch, HASH_EMB_DIM]);
        let archetype = burn::tensor::activation::gelu(self.archetype_proj.forward(archetype));

        // Scalars: [B, 5] → [B, 64]
        let scalars = burn::tensor::activation::gelu(
            self.scalar_proj.forward(input.scalars.clone())
        );

        // Settlement: [B, MAX_SETTLEMENT] hashes → embed → mean pool → [B, 32] → [B, 64]
        let settlement = self.settlement_emb.forward(input.settlement_hashes.clone()); // [B, MAX, 16]
        let settlement_mask = input.settlement_mask.clone().float().unsqueeze_dim::<3>(2); // [B, MAX, 1]
        let settlement_sum = (settlement * settlement_mask.clone()).sum_dim(1)
            .reshape([batch, HASH_EMB_DIM / 2]);
        let settlement_count = settlement_mask.sum_dim(1).reshape([batch, 1]).clamp_min(1.0);
        let settlement_pooled = settlement_sum / settlement_count;
        // Pad to HASH_EMB_DIM and project
        let settlement_padded = Tensor::cat(vec![
            settlement_pooled.clone(), settlement_pooled
        ], 1); // [B, 32]
        let settlement = burn::tensor::activation::gelu(
            self.settlement_proj.forward(settlement_padded)
        );

        // Existing abilities: [B, MAX_EXISTING] hashes → embed → mean pool → [B, 32] → [B, 64]
        let existing = self.ability_emb.forward(input.existing_hashes.clone()); // [B, MAX, 32]
        let existing_mask = input.existing_mask.clone().float().unsqueeze_dim::<3>(2); // [B, MAX, 1]
        let existing_sum = (existing * existing_mask.clone()).sum_dim(1)
            .reshape([batch, HASH_EMB_DIM]);
        let existing_count = existing_mask.sum_dim(1).reshape([batch, 1]).clamp_min(1.0);
        let existing_pooled = existing_sum / existing_count;
        let existing = burn::tensor::activation::gelu(
            self.existing_proj.forward(existing_pooled)
        );

        // --- Stack into sequence: [B, 6, 64] ---
        let sequence = Tensor::cat(vec![
            behavior.unsqueeze_dim::<3>(1),
            class.unsqueeze_dim::<3>(1),
            archetype.unsqueeze_dim::<3>(1),
            scalars.unsqueeze_dim::<3>(1),
            settlement.unsqueeze_dim::<3>(1),
            existing.unsqueeze_dim::<3>(1),
        ], 1); // [B, 6, 64]

        // --- Self-attention over feature groups ---
        let enc_input = TransformerEncoderInput::new(sequence);
        let encoded = self.encoder.forward(enc_input); // [B, 6, 64]

        // --- Pool: mean over groups → [B, 64] → project to COND_DIM ---
        let pooled = encoded.clone().sum_dim(1).reshape([batch, GROUP_DIM])
            / N_FEATURE_GROUPS as f32;
        let cond = self.output_norm.forward(self.output_proj.forward(pooled));

        (cond, encoded) // cond for classification heads, encoded as memory for cross-attn
    }
}

/// Batched input for the context encoder.
/// All tensors are [B, ...] where B is batch size.
pub struct ContextInput<B: Backend> {
    /// BehaviorLedger floats: [B, 24]
    pub behavior: Tensor<B, 2>,
    /// Class name hash mod HASH_BUCKETS: [B, 1]
    pub class_hash: Tensor<B, 2, Int>,
    /// Archetype hash mod HASH_BUCKETS: [B, 1]
    pub archetype_hash: Tensor<B, 2, Int>,
    /// [level/100, tier/7, tension, near_death/10, retreats/5]: [B, 5]
    pub scalars: Tensor<B, 2>,
    /// Settlement context tag hashes: [B, MAX_SETTLEMENT]
    pub settlement_hashes: Tensor<B, 2, Int>,
    /// Settlement mask (true where valid): [B, MAX_SETTLEMENT]
    pub settlement_mask: Tensor<B, 2, Bool>,
    /// Existing ability hashes: [B, MAX_EXISTING]
    pub existing_hashes: Tensor<B, 2, Int>,
    /// Existing mask: [B, MAX_EXISTING]
    pub existing_mask: Tensor<B, 2, Bool>,
}

// ---------------------------------------------------------------------------
// Helper: build ContextInput from raw game state
// ---------------------------------------------------------------------------

/// Convert raw game state into a ContextInput tensor batch (single item).
pub fn encode_context<B: Backend>(
    class_name_hash: u32,
    class_level: u16,
    tier: u32,
    behavior_tags: &[u32],
    behavior_values: &[f32],
    archetype: &str,
    settlement_context: &[(u32, f32)],
    existing_abilities: &[u32],
    device: &B::Device,
) -> ContextInput<B> {
    // Behavior: pad to 24 dims
    let mut behavior_data = vec![0.0f32; BEHAVIOR_DIM];
    for (i, &v) in behavior_values.iter().enumerate().take(BEHAVIOR_DIM) {
        behavior_data[i] = v;
    }
    let behavior = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(behavior_data, [BEHAVIOR_DIM]), device,
    ).unsqueeze::<2>(); // [1, 24]

    // Class hash
    let class_hash = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![(class_name_hash % HASH_BUCKETS as u32) as i64], [1, 1]),
        device,
    );

    // Archetype hash
    let arch_hash = archetype.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let archetype_hash = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![(arch_hash % HASH_BUCKETS as u64) as i64], [1, 1]),
        device,
    );

    // Scalars
    let scalars = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(vec![
            class_level as f32 / 100.0,
            tier as f32 / 7.0,
            0.0, // tension (would come from BehaviorLedger)
            0.0, // near_death
            0.0, // retreats
        ], [SCALAR_DIM]),
        device,
    ).unsqueeze::<2>();

    // Settlement
    let mut settle_data = vec![0i64; MAX_SETTLEMENT];
    let mut settle_mask = vec![false; MAX_SETTLEMENT];
    for (i, &(tag, _)) in settlement_context.iter().enumerate().take(MAX_SETTLEMENT) {
        settle_data[i] = (tag % HASH_BUCKETS as u32) as i64;
        settle_mask[i] = true;
    }
    let settlement_hashes = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(settle_data, [MAX_SETTLEMENT]), device,
    ).unsqueeze::<2>();
    let settlement_mask = Tensor::<B, 1, Bool>::from_data(
        burn::tensor::TensorData::new(settle_mask, [MAX_SETTLEMENT]), device,
    ).unsqueeze::<2>();

    // Existing abilities
    let mut exist_data = vec![0i64; MAX_EXISTING];
    let mut exist_mask = vec![false; MAX_EXISTING];
    for (i, &hash) in existing_abilities.iter().enumerate().take(MAX_EXISTING) {
        exist_data[i] = (hash % HASH_BUCKETS as u32) as i64;
        exist_mask[i] = true;
    }
    let existing_hashes = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(exist_data, [MAX_EXISTING]), device,
    ).unsqueeze::<2>();
    let existing_mask = Tensor::<B, 1, Bool>::from_data(
        burn::tensor::TensorData::new(exist_mask, [MAX_EXISTING]), device,
    ).unsqueeze::<2>();

    ContextInput {
        behavior,
        class_hash,
        archetype_hash,
        scalars,
        settlement_hashes,
        settlement_mask,
        existing_hashes,
        existing_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    type TestB = burn::backend::LibTorch;
    #[cfg(feature = "cpu")]
    type TestB = burn::backend::NdArray;

    #[test]
    fn test_context_encoder_shapes() {
        let device = Default::default();
        let encoder = ContextEncoder::<TestB>::new(&device);

        let input = encode_context::<TestB>(
            42, 30, 4,
            &[0, 1, 2], &[0.8, 0.1, 0.1],
            "knight",
            &[(100, 0.5), (200, 0.3)],
            &[1000, 2000, 3000],
            &device,
        );

        let (cond, memory) = encoder.forward(&input);
        assert_eq!(cond.dims(), [1, COND_DIM]);
        assert_eq!(memory.dims(), [1, N_FEATURE_GROUPS, GROUP_DIM]);
    }
}
