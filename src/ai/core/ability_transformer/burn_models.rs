//! Burn port of `training/model.py`.
//!
//! All model architectures (AbilityTransformer, EntityEncoder V1-V3,
//! CrossAttentionBlock, ActorCritic V1-V3, PointerHead) ported from PyTorch
//! to Burn for native Rust training with autodiff.
//!
//! These models are compatible with the existing JSON weight export pipeline
//! and can replace the Python training loop entirely.

use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, Sigmoid,
};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Constants (must match training/model.py)
// ---------------------------------------------------------------------------

pub const NUM_ACTIONS: usize = 14;
pub const NUM_BASE_ACTIONS: usize = 6;
pub const MAX_ABILITIES: usize = 8;
pub const NUM_ACTION_TYPES: usize = 3 + MAX_ABILITIES; // 11
pub const ENTITY_DIM: usize = 30;
pub const THREAT_DIM: usize = 8;
pub const POSITION_DIM: usize = 8;

// ---------------------------------------------------------------------------
// AbilityTransformer — core transformer encoder for ability DSL tokens
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityTransformer<B: Backend> {
    pub d_model: usize,
    pub pad_id: usize,
    pub cls_id: usize,
    pub token_emb: Embedding<B>,
    pub pos_emb: Embedding<B>,
    pub encoder: TransformerEncoder<B>,
    pub out_norm: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct AbilityTransformerConfig {
    pub vocab_size: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
    #[config(default = "0")]
    pub pad_id: usize,
    #[config(default = "1")]
    pub cls_id: usize,
}

impl AbilityTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityTransformer<B> {
        let token_emb = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_emb = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);

        let encoder = TransformerEncoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);

        let out_norm = LayerNormConfig::new(self.d_model).init(device);

        AbilityTransformer {
            d_model: self.d_model,
            pad_id: self.pad_id,
            cls_id: self.cls_id,
            token_emb,
            pos_emb,
            encoder,
            out_norm,
        }
    }
}

impl<B: Backend> AbilityTransformer<B> {
    /// Encode token sequence.
    ///
    /// * `input_ids`: `[batch, seq_len]` int tensor
    /// * `attention_mask`: `[batch, seq_len]` bool tensor, true=attend
    ///
    /// Returns `[batch, seq_len, d_model]`.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();
        let device = &input_ids.device();

        // Positional indices: [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device)
            .unsqueeze::<2>()
            .expand([batch, seq_len]);

        // Embed tokens + positions
        let x = self.token_emb.forward(input_ids.clone()) + self.pos_emb.forward(positions);

        // Build padding mask for transformer encoder
        let pad_mask = match attention_mask {
            Some(mask) => mask,
            None => input_ids.not_equal_elem(self.pad_id as i32),
        };

        let encoder_input = TransformerEncoderInput::new(x).mask_pad(pad_mask);
        let x = self.encoder.forward(encoder_input);
        self.out_norm.forward(x)
    }

    /// Extract [CLS] embedding (position 0).
    ///
    /// Returns `[batch, d_model]`.
    pub fn cls_embedding(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let hidden = self.forward(input_ids, attention_mask);
        let batch = hidden.dims()[0];
        hidden.slice([0..batch, 0..1]).squeeze::<2>()
    }
}

// ---------------------------------------------------------------------------
// MLMHead — masked language model head for pre-training
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct MLMHead<B: Backend> {
    pub dense: Linear<B>,
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub gelu: Gelu,
}

#[derive(Config, Debug)]
pub struct MLMHeadConfig {
    pub d_model: usize,
    pub vocab_size: usize,
}

impl MLMHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLMHead<B> {
        MLMHead {
            dense: LinearConfig::new(self.d_model, self.d_model).init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            proj: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            gelu: Gelu::new(),
        }
    }
}

impl<B: Backend> MLMHead<B> {
    /// Predict masked tokens.
    ///
    /// * `hidden`: `[batch, seq_len, d_model]`
    ///
    /// Returns `[batch, seq_len, vocab_size]`.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.dense.forward(hidden);
        let x = self.gelu.forward(x);
        let x = self.norm.forward(x);
        self.proj.forward(x)
    }
}

// ---------------------------------------------------------------------------
// DecisionHead — urgency + target prediction from [CLS] embedding
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DecisionHead<B: Backend> {
    pub urgency_l1: Linear<B>,
    pub urgency_l2: Linear<B>,
    pub target_l1: Linear<B>,
    pub target_l2: Linear<B>,
    pub gelu: Gelu,
    pub sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct DecisionHeadConfig {
    pub d_model: usize,
    #[config(default = "3")]
    pub n_targets: usize,
}

impl DecisionHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecisionHead<B> {
        DecisionHead {
            urgency_l1: LinearConfig::new(self.d_model, self.d_model).init(device),
            urgency_l2: LinearConfig::new(self.d_model, 1).init(device),
            target_l1: LinearConfig::new(self.d_model, self.d_model).init(device),
            target_l2: LinearConfig::new(self.d_model, self.n_targets).init(device),
            gelu: Gelu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}

impl<B: Backend> DecisionHead<B> {
    /// Predict urgency and target scores.
    ///
    /// * `cls_emb`: `[batch, d_model]`
    ///
    /// Returns `(urgency [batch, 1], target_logits [batch, n_targets])`.
    pub fn forward(&self, cls_emb: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let urgency = self.urgency_l1.forward(cls_emb.clone());
        let urgency = self.gelu.forward(urgency);
        let urgency = self.urgency_l2.forward(urgency);
        let urgency = self.sigmoid.forward(urgency);

        let target = self.target_l1.forward(cls_emb);
        let target = self.gelu.forward(target);
        let target = self.target_l2.forward(target);

        (urgency, target)
    }
}

// ---------------------------------------------------------------------------
// HintClassificationHead — auxiliary head for ability category prediction
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct HintClassificationHead<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub gelu: Gelu,
}

#[derive(Config, Debug)]
pub struct HintClassificationHeadConfig {
    pub d_model: usize,
    #[config(default = "6")]
    pub n_classes: usize,
}

impl HintClassificationHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HintClassificationHead<B> {
        HintClassificationHead {
            linear1: LinearConfig::new(self.d_model, self.d_model).init(device),
            linear2: LinearConfig::new(self.d_model, self.n_classes).init(device),
            gelu: Gelu::new(),
        }
    }
}

impl<B: Backend> HintClassificationHead<B> {
    /// Predict hint category logits from [CLS] embedding.
    ///
    /// * `cls_emb`: `[batch, d_model]`
    ///
    /// Returns `[batch, n_classes]`.
    pub fn forward(&self, cls_emb: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(cls_emb);
        let x = self.gelu.forward(x);
        self.linear2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// EntityEncoder V1 — fixed 7-entity game state encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct EntityEncoder<B: Backend> {
    pub d_model: usize,
    pub proj: Linear<B>,
    pub type_emb: Embedding<B>,
    pub input_norm: LayerNorm<B>,
    pub encoder: TransformerEncoder<B>,
    pub out_norm: LayerNorm<B>,
}

/// Entity type IDs: [self, enemy, enemy, enemy, ally, ally, ally]
const V1_TYPE_IDS: [i64; 7] = [0, 1, 1, 1, 2, 2, 2];
const V1_NUM_ENTITIES: usize = 7;
const V1_NUM_TYPES: usize = 3;

#[derive(Config, Debug)]
pub struct EntityEncoderConfig {
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
}

impl EntityEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EntityEncoder<B> {
        let encoder = TransformerEncoderConfig::new(
            self.d_model,
            self.d_model * 2,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);

        EntityEncoder {
            d_model: self.d_model,
            proj: LinearConfig::new(ENTITY_DIM, self.d_model).init(device),
            type_emb: EmbeddingConfig::new(V1_NUM_TYPES, self.d_model).init(device),
            input_norm: LayerNormConfig::new(self.d_model).init(device),
            encoder,
            out_norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> EntityEncoder<B> {
    /// Encode game state entities.
    ///
    /// * `game_state`: `[batch, 210]` (7 entities x 30 features)
    ///
    /// Returns `(entity_tokens [batch, 7, d_model], entity_mask [batch, 7] bool)`.
    pub fn forward(
        &self,
        game_state: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let [batch, _] = game_state.dims();
        let device = &game_state.device();

        // Reshape to [batch, 7, 30]
        let entities = game_state.reshape([batch, V1_NUM_ENTITIES, ENTITY_DIM]);

        // Project to d_model
        let tokens = self.proj.forward(entities.clone());

        // Add type embeddings
        let type_ids = Tensor::<B, 1, Int>::from_data(
            TensorData::from(V1_TYPE_IDS.as_slice()),
            device,
        )
        .unsqueeze::<2>();
        let type_tokens = self.type_emb.forward(type_ids);
        let tokens = tokens + type_tokens;
        let tokens = self.input_norm.forward(tokens);

        // Entity mask: exists feature is index 29
        let exists = entities.slice([0..batch, 0..V1_NUM_ENTITIES, 29..30]).squeeze::<2>();
        let entity_mask = exists.greater_equal_elem(0.5);

        let encoder_input = TransformerEncoderInput::new(tokens).mask_pad(entity_mask.clone());
        let tokens = self.encoder.forward(encoder_input);
        let tokens = self.out_norm.forward(tokens);

        // Return inverted mask for compatibility (True = ignore/padding)
        let padding_mask = entity_mask.bool_not();
        (tokens, padding_mask)
    }
}

// ---------------------------------------------------------------------------
// EntityEncoder V2 — variable-length entities + threats
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct EntityEncoderV2<B: Backend> {
    pub d_model: usize,
    pub entity_proj: Linear<B>,
    pub threat_proj: Linear<B>,
    pub type_emb: Embedding<B>,
    pub input_norm: LayerNorm<B>,
    pub encoder: TransformerEncoder<B>,
    pub out_norm: LayerNorm<B>,
}

const V2_NUM_TYPES: usize = 4; // self=0, enemy=1, ally=2, threat=3

#[derive(Config, Debug)]
pub struct EntityEncoderV2Config {
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "4")]
    pub n_layers: usize,
}

impl EntityEncoderV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EntityEncoderV2<B> {
        let encoder = TransformerEncoderConfig::new(
            self.d_model,
            self.d_model * 2,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);

        EntityEncoderV2 {
            d_model: self.d_model,
            entity_proj: LinearConfig::new(ENTITY_DIM, self.d_model).init(device),
            threat_proj: LinearConfig::new(THREAT_DIM, self.d_model).init(device),
            type_emb: EmbeddingConfig::new(V2_NUM_TYPES, self.d_model).init(device),
            input_norm: LayerNormConfig::new(self.d_model).init(device),
            encoder,
            out_norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> EntityEncoderV2<B> {
    /// Encode variable-length entities + threats.
    ///
    /// * `entity_features`: `[B, max_entities, 30]`
    /// * `entity_type_ids`: `[B, max_entities]` (0/1/2)
    /// * `threat_features`: `[B, max_threats, 8]`
    /// * `entity_mask`: `[B, max_entities]` bool, True=padding
    /// * `threat_mask`: `[B, max_threats]` bool, True=padding
    ///
    /// Returns `(tokens [B, E+T, d_model], full_mask [B, E+T] True=padding)`.
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let [batch, _n_entities, _] = entity_features.dims();
        let [_, n_threats, _] = threat_features.dims();
        let device = &entity_features.device();

        // Project entities and threats
        let ent_tokens = self.entity_proj.forward(entity_features);
        let threat_tokens = self.threat_proj.forward(threat_features);

        // Add type embeddings
        let ent_tokens = ent_tokens + self.type_emb.forward(entity_type_ids);
        let threat_type_ids = Tensor::<B, 2, Int>::full(
            [batch, n_threats],
            3, // threat type
            device,
        );
        let threat_tokens = threat_tokens + self.type_emb.forward(threat_type_ids);

        // Concatenate into single sequence
        let tokens = Tensor::cat(vec![ent_tokens, threat_tokens], 1);
        let tokens = self.input_norm.forward(tokens);
        let full_mask = Tensor::cat(vec![entity_mask, threat_mask], 1);

        // mask_pad expects true=attend, full_mask is true=padding, so invert
        let attend_mask = full_mask.clone().bool_not();
        let encoder_input = TransformerEncoderInput::new(tokens).mask_pad(attend_mask);
        let tokens = self.encoder.forward(encoder_input);
        let tokens = self.out_norm.forward(tokens);

        (tokens, full_mask)
    }
}

// ---------------------------------------------------------------------------
// EntityEncoder V3 — entities + threats + position tokens
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct EntityEncoderV3<B: Backend> {
    pub d_model: usize,
    pub entity_proj: Linear<B>,
    pub threat_proj: Linear<B>,
    pub position_proj: Linear<B>,
    pub type_emb: Embedding<B>,
    pub input_norm: LayerNorm<B>,
    pub encoder: TransformerEncoder<B>,
    pub out_norm: LayerNorm<B>,
}

const V3_NUM_TYPES: usize = 5; // self=0, enemy=1, ally=2, threat=3, position=4

#[derive(Config, Debug)]
pub struct EntityEncoderV3Config {
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "4")]
    pub n_layers: usize,
}

impl EntityEncoderV3Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EntityEncoderV3<B> {
        let encoder = TransformerEncoderConfig::new(
            self.d_model,
            self.d_model * 2,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);

        EntityEncoderV3 {
            d_model: self.d_model,
            entity_proj: LinearConfig::new(ENTITY_DIM, self.d_model).init(device),
            threat_proj: LinearConfig::new(THREAT_DIM, self.d_model).init(device),
            position_proj: LinearConfig::new(POSITION_DIM, self.d_model).init(device),
            type_emb: EmbeddingConfig::new(V3_NUM_TYPES, self.d_model).init(device),
            input_norm: LayerNormConfig::new(self.d_model).init(device),
            encoder,
            out_norm: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> EntityEncoderV3<B> {
    /// Encode entities + threats + positions via shared self-attention.
    ///
    /// * `entity_features`: `[B, E, 30]`
    /// * `entity_type_ids`: `[B, E]` long
    /// * `threat_features`: `[B, T, 8]`
    /// * `entity_mask`: `[B, E]` bool True=padding
    /// * `threat_mask`: `[B, T]` bool True=padding
    /// * `position_features`: `[B, P, 8]` or None
    /// * `position_mask`: `[B, P]` bool True=padding or None
    ///
    /// Returns `(tokens [B, E+T+P, d_model], full_mask [B, E+T+P] True=padding)`.
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
        position_features: Option<Tensor<B, 3>>,
        position_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let [batch, _n_entities, _] = entity_features.dims();
        let [_, n_threats, _] = threat_features.dims();
        let device = &entity_features.device();

        // Project entities and threats
        let ent_tokens = self.entity_proj.forward(entity_features);
        let threat_tokens = self.threat_proj.forward(threat_features);

        // Add type embeddings
        let ent_tokens = ent_tokens + self.type_emb.forward(entity_type_ids);
        let threat_type_ids =
            Tensor::<B, 2, Int>::full([batch, n_threats], 3, device);
        let threat_tokens = threat_tokens + self.type_emb.forward(threat_type_ids);

        let mut parts = vec![ent_tokens, threat_tokens];
        let mut masks = vec![entity_mask, threat_mask];

        // Optional position tokens
        if let Some(pos_feats) = position_features {
            let [_, n_pos, _] = pos_feats.dims();
            if n_pos > 0 {
                let pos_tokens = self.position_proj.forward(pos_feats);
                let pos_type_ids =
                    Tensor::<B, 2, Int>::full([batch, n_pos], 4, device);
                let pos_tokens = pos_tokens + self.type_emb.forward(pos_type_ids);
                parts.push(pos_tokens);

                let pos_mask = position_mask.unwrap_or_else(|| {
                    Tensor::<B, 2, Bool>::full([batch, n_pos], false, device)
                });
                masks.push(pos_mask);
            }
        }

        let tokens = Tensor::cat(parts, 1);
        let tokens = self.input_norm.forward(tokens);
        let full_mask = Tensor::cat(masks, 1);

        // mask_pad expects true=attend, full_mask is true=padding, so invert
        let attend_mask = full_mask.clone().bool_not();
        let encoder_input = TransformerEncoderInput::new(tokens).mask_pad(attend_mask);
        let tokens = self.encoder.forward(encoder_input);
        let tokens = self.out_norm.forward(tokens);

        (tokens, full_mask)
    }
}

// ---------------------------------------------------------------------------
// CrossAttentionBlock — ability [CLS] attends to game state entity tokens
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    pub cross_attn: MultiHeadAttention<B>,
    pub norm_q: LayerNorm<B>,
    pub norm_kv: LayerNorm<B>,
    pub ff_l1: Linear<B>,
    pub ff_l2: Linear<B>,
    pub norm_ff: LayerNorm<B>,
    pub gelu: Gelu,
}

#[derive(Config, Debug)]
pub struct CrossAttentionBlockConfig {
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
}

impl CrossAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttentionBlock<B> {
        CrossAttentionBlock {
            cross_attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(0.0)
                .init(device),
            norm_q: LayerNormConfig::new(self.d_model).init(device),
            norm_kv: LayerNormConfig::new(self.d_model).init(device),
            ff_l1: LinearConfig::new(self.d_model, self.d_model * 2).init(device),
            ff_l2: LinearConfig::new(self.d_model * 2, self.d_model).init(device),
            norm_ff: LayerNormConfig::new(self.d_model).init(device),
            gelu: Gelu::new(),
        }
    }
}

impl<B: Backend> CrossAttentionBlock<B> {
    /// Cross-attention: query attends to key/value tokens.
    ///
    /// * `query`: `[batch, d_model]` — [CLS] embedding
    /// * `kv`: `[batch, n_entities, d_model]` — entity tokens
    /// * `kv_mask`: `[batch, n_entities]` bool True=padding (optional)
    ///
    /// Returns `[batch, d_model]`.
    pub fn forward(
        &self,
        query: Tensor<B, 2>,
        kv: Tensor<B, 3>,
        kv_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        // Expand query to [batch, 1, d_model]
        let q = self.norm_q.forward(query.clone()).unsqueeze_dim(1);
        let kv_normed = self.norm_kv.forward(kv);

        // Build MHA input with cross-attention (Q from query, K/V from kv)
        let mut mha_input = MhaInput::new(q, kv_normed.clone(), kv_normed);
        if let Some(mask) = kv_mask {
            // Invert: burn expects true=attend
            mha_input = mha_input.mask_pad(mask.bool_not());
        }
        let attn_out = self.cross_attn.forward(mha_input);

        // Squeeze back to [batch, d_model]
        let attn_out = attn_out.context.squeeze::<2>();

        // Residual + FF
        let x = query + attn_out;
        let ff_in = self.norm_ff.forward(x.clone());
        let ff_out = self.ff_l1.forward(ff_in);
        let ff_out = self.gelu.forward(ff_out);
        let ff_out = self.ff_l2.forward(ff_out);

        x + ff_out
    }
}

// ---------------------------------------------------------------------------
// AbilityTransformerMLM — pre-training: transformer + MLM head
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityTransformerMLM<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub mlm_head: MLMHead<B>,
}

#[derive(Config, Debug)]
pub struct AbilityTransformerMLMConfig {
    pub vocab_size: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
}

impl AbilityTransformerMLMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityTransformerMLM<B> {
        let transformer = AbilityTransformerConfig::new(self.vocab_size)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_max_seq_len(self.max_seq_len)
            .init(device);

        let mlm_head = MLMHeadConfig::new(self.d_model, self.vocab_size).init(device);

        AbilityTransformerMLM {
            transformer,
            mlm_head,
        }
    }
}

impl<B: Backend> AbilityTransformerMLM<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let hidden = self.transformer.forward(input_ids, attention_mask);
        self.mlm_head.forward(hidden)
    }
}

// ---------------------------------------------------------------------------
// AbilityTransformerDecision — fine-tuning: transformer + cross-attn + decision
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityTransformerDecision<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: Option<EntityEncoder<B>>,
    pub cross_attn: Option<CrossAttentionBlock<B>>,
    pub decision_head: DecisionHead<B>,
}

#[derive(Config, Debug)]
pub struct AbilityTransformerDecisionConfig {
    pub vocab_size: usize,
    #[config(default = "0")]
    pub game_state_dim: usize,
    #[config(default = "3")]
    pub n_targets: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
}

impl AbilityTransformerDecisionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityTransformerDecision<B> {
        let transformer = AbilityTransformerConfig::new(self.vocab_size)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_max_seq_len(self.max_seq_len)
            .init(device);

        let (entity_encoder, cross_attn) = if self.game_state_dim > 0 {
            let ee = EntityEncoderConfig::new(self.d_model)
                .with_n_heads(self.n_heads)
                .with_n_layers(self.n_layers)
                .init(device);
            let ca = CrossAttentionBlockConfig::new(self.d_model)
                .with_n_heads(self.n_heads)
                .init(device);
            (Some(ee), Some(ca))
        } else {
            (None, None)
        };

        let decision_head = DecisionHeadConfig::new(self.d_model)
            .with_n_targets(self.n_targets)
            .init(device);

        AbilityTransformerDecision {
            transformer,
            entity_encoder,
            cross_attn,
            decision_head,
        }
    }
}

impl<B: Backend> AbilityTransformerDecision<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        game_state: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut cls_emb = self.transformer.cls_embedding(input_ids, attention_mask);

        if let (Some(ee), Some(ca), Some(gs)) =
            (&self.entity_encoder, &self.cross_attn, game_state)
        {
            let (entity_tokens, entity_mask) = ee.forward(gs);
            cls_emb = ca.forward(cls_emb, entity_tokens, Some(entity_mask));
        }

        self.decision_head.forward(cls_emb)
    }
}

// ---------------------------------------------------------------------------
// AbilityActorCritic V1 — 14-action actor-critic with fixed entity encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityActorCritic<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoder<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub base_l1: Linear<B>,
    pub base_l2: Linear<B>,
    pub ability_l1: Linear<B>,
    pub ability_l2: Linear<B>,
    pub value_l1: Linear<B>,
    pub value_l2: Linear<B>,
    pub gelu: Gelu,
    pub d_model: usize,
}

#[derive(Config, Debug)]
pub struct AbilityActorCriticConfig {
    pub vocab_size: usize,
    #[config(default = "210")]
    pub game_state_dim: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
}

impl AbilityActorCriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityActorCritic<B> {
        let transformer = AbilityTransformerConfig::new(self.vocab_size)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_max_seq_len(self.max_seq_len)
            .init(device);

        let entity_encoder = EntityEncoderConfig::new(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .init(device);

        let cross_attn = CrossAttentionBlockConfig::new(self.d_model)
            .with_n_heads(self.n_heads)
            .init(device);

        let d = self.d_model;
        AbilityActorCritic {
            transformer,
            entity_encoder,
            cross_attn,
            base_l1: LinearConfig::new(d, d).init(device),
            base_l2: LinearConfig::new(d, NUM_BASE_ACTIONS).init(device),
            ability_l1: LinearConfig::new(d, d).init(device),
            ability_l2: LinearConfig::new(d, 1).init(device),
            value_l1: LinearConfig::new(d, d).init(device),
            value_l2: LinearConfig::new(d, 1).init(device),
            gelu: Gelu::new(),
            d_model: d,
        }
    }
}

impl<B: Backend> AbilityActorCritic<B> {
    /// Encode game state → (entity_tokens, entity_mask, pooled_state).
    pub fn encode_entities(
        &self,
        game_state: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 2>) {
        let (entity_tokens, padding_mask) = self.entity_encoder.forward(game_state);

        // Mean pool over existing (non-padding) entities
        let exists = padding_mask.clone().bool_not().float().unsqueeze_dim(2); // [B, 7, 1]
        let sum = (entity_tokens.clone() * exists.clone()).sum_dim(1);
        let count = exists.sum_dim(1).clamp_min(1.0);
        let pooled = sum / count;
        let pooled = pooled.squeeze::<2>();

        (entity_tokens, padding_mask, pooled)
    }

    /// Compute action logits.
    ///
    /// * `game_state`: `[B, 210]`
    /// * `ability_cls`: list of MAX_ABILITIES tensors, each `[B, d_model]` or None
    ///
    /// Returns `[B, 14]` logits.
    pub fn forward_policy(
        &self,
        game_state: Tensor<B, 2>,
        ability_cls: &[Option<Tensor<B, 2>>],
    ) -> Tensor<B, 2> {
        let [batch, _] = game_state.dims();
        let device = &game_state.device();
        let (entity_tokens, entity_mask, pooled) = self.encode_entities(game_state);

        // Base action logits
        let base_logits = self.base_l1.forward(pooled.clone());
        let base_logits = self.gelu.forward(base_logits);
        let base_logits = self.base_l2.forward(base_logits); // [B, 6]

        // Ability logits
        let mut ability_logits =
            Tensor::<B, 2>::full([batch, MAX_ABILITIES], -1e9, device);

        for (i, cls_opt) in ability_cls.iter().enumerate().take(MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb =
                    self.cross_attn
                        .forward(cls.clone(), entity_tokens.clone(), Some(entity_mask.clone()));
                let logit = self.ability_l1.forward(cross_emb);
                let logit = self.gelu.forward(logit);
                let logit = self.ability_l2.forward(logit); // [B, 1]
                // Place into column i of ability_logits
                ability_logits = ability_logits.slice_assign(
                    [0..batch, i..i + 1],
                    logit,
                );
            }
        }

        // Combine: [attack×3, ability×8, move×2, hold]
        let attack = base_logits.clone().slice([0..batch, 0..3]);
        let move_hold = base_logits.slice([0..batch, 3..6]);
        Tensor::cat(vec![attack, ability_logits, move_hold], 1)
    }

    /// Compute state value V(s).
    pub fn forward_value(&self, game_state: Tensor<B, 2>) -> Tensor<B, 2> {
        let (_, _, pooled) = self.encode_entities(game_state);
        let v = self.value_l1.forward(pooled);
        let v = self.gelu.forward(v);
        self.value_l2.forward(v) // [B, 1]
    }

    /// Returns (action_logits [B, 14], state_value [B, 1]).
    pub fn forward(
        &self,
        game_state: Tensor<B, 2>,
        ability_cls: &[Option<Tensor<B, 2>>],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _] = game_state.dims();
        let device = &game_state.device();
        let (entity_tokens, entity_mask, pooled) = self.encode_entities(game_state);

        let base_logits = self.base_l1.forward(pooled.clone());
        let base_logits = self.gelu.forward(base_logits);
        let base_logits = self.base_l2.forward(base_logits);

        let mut ability_logits =
            Tensor::<B, 2>::full([batch, MAX_ABILITIES], -1e9, device);

        for (i, cls_opt) in ability_cls.iter().enumerate().take(MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb =
                    self.cross_attn
                        .forward(cls.clone(), entity_tokens.clone(), Some(entity_mask.clone()));
                let logit = self.ability_l1.forward(cross_emb);
                let logit = self.gelu.forward(logit);
                let logit = self.ability_l2.forward(logit);
                ability_logits = ability_logits.slice_assign(
                    [0..batch, i..i + 1],
                    logit,
                );
            }
        }

        let attack = base_logits.clone().slice([0..batch, 0..3]);
        let move_hold = base_logits.slice([0..batch, 3..6]);
        let logits = Tensor::cat(vec![attack, ability_logits, move_hold], 1);

        let value = self.value_l1.forward(pooled);
        let value = self.gelu.forward(value);
        let value = self.value_l2.forward(value);

        (logits, value)
    }
}

// ---------------------------------------------------------------------------
// AbilityActorCritic V2 — variable-length entity encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityActorCriticV2<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoderV2<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub base_l1: Linear<B>,
    pub base_l2: Linear<B>,
    pub ability_l1: Linear<B>,
    pub ability_l2: Linear<B>,
    pub value_l1: Linear<B>,
    pub value_l2: Linear<B>,
    pub gelu: Gelu,
    pub d_model: usize,
}

#[derive(Config, Debug)]
pub struct AbilityActorCriticV2Config {
    pub vocab_size: usize,
    #[config(default = "4")]
    pub entity_encoder_layers: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
}

impl AbilityActorCriticV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityActorCriticV2<B> {
        let transformer = AbilityTransformerConfig::new(self.vocab_size)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_max_seq_len(self.max_seq_len)
            .init(device);

        let entity_encoder = EntityEncoderV2Config::new()
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.entity_encoder_layers)
            .init(device);

        let cross_attn = CrossAttentionBlockConfig::new(self.d_model)
            .with_n_heads(self.n_heads)
            .init(device);

        let d = self.d_model;
        AbilityActorCriticV2 {
            transformer,
            entity_encoder,
            cross_attn,
            base_l1: LinearConfig::new(d, d).init(device),
            base_l2: LinearConfig::new(d, NUM_BASE_ACTIONS).init(device),
            ability_l1: LinearConfig::new(d, d).init(device),
            ability_l2: LinearConfig::new(d, 1).init(device),
            value_l1: LinearConfig::new(d, d).init(device),
            value_l2: LinearConfig::new(d, 1).init(device),
            gelu: Gelu::new(),
            d_model: d,
        }
    }
}

impl<B: Backend> AbilityActorCriticV2<B> {
    /// Encode v2 game state → (tokens, mask, pooled).
    pub fn encode_entities(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 2>) {
        let (tokens, full_mask) = self.entity_encoder.forward(
            entity_features,
            entity_type_ids,
            threat_features,
            entity_mask,
            threat_mask,
        );

        let exist = full_mask.clone().bool_not().float().unsqueeze_dim(2);
        let sum = (tokens.clone() * exist.clone()).sum_dim(1);
        let count = exist.sum_dim(1).clamp_min(1.0);
        let pooled = (sum / count).squeeze::<2>();

        (tokens, full_mask, pooled)
    }

    /// Compute state value V(s).
    pub fn forward_value(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 2> {
        let (_, _, pooled) = self.encode_entities(
            entity_features,
            entity_type_ids,
            threat_features,
            entity_mask,
            threat_mask,
        );
        let v = self.value_l1.forward(pooled);
        let v = self.gelu.forward(v);
        self.value_l2.forward(v)
    }

    /// Returns (action_logits [B, 14], state_value [B, 1]).
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
        ability_cls: &[Option<Tensor<B, 2>>],
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, ..] = entity_features.dims();
        let device = &entity_features.device();

        let (tokens, full_mask, pooled) = self.encode_entities(
            entity_features,
            entity_type_ids,
            threat_features,
            entity_mask,
            threat_mask,
        );

        let base_logits = self.base_l1.forward(pooled.clone());
        let base_logits = self.gelu.forward(base_logits);
        let base_logits = self.base_l2.forward(base_logits);

        let mut ability_logits =
            Tensor::<B, 2>::full([batch, MAX_ABILITIES], -1e9, device);

        for (i, cls_opt) in ability_cls.iter().enumerate().take(MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb =
                    self.cross_attn
                        .forward(cls.clone(), tokens.clone(), Some(full_mask.clone()));
                let logit = self.ability_l1.forward(cross_emb);
                let logit = self.gelu.forward(logit);
                let logit = self.ability_l2.forward(logit);
                ability_logits = ability_logits.slice_assign(
                    [0..batch, i..i + 1],
                    logit,
                );
            }
        }

        let attack = base_logits.clone().slice([0..batch, 0..3]);
        let move_hold = base_logits.slice([0..batch, 3..6]);
        let logits = Tensor::cat(vec![attack, ability_logits, move_hold], 1);

        let value = self.value_l1.forward(pooled);
        let value = self.gelu.forward(value);
        let value = self.value_l2.forward(value);

        (logits, value)
    }
}

// ---------------------------------------------------------------------------
// PointerHead — pointer-based target selection for V3
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PointerHead<B: Backend> {
    pub d_model: usize,
    pub action_type_l1: Linear<B>,
    pub action_type_l2: Linear<B>,
    pub pointer_key: Linear<B>,
    pub attack_query: Linear<B>,
    pub move_query: Linear<B>,
    pub ability_queries: Vec<Linear<B>>,
    pub gelu: Gelu,
}

#[derive(Config, Debug)]
pub struct PointerHeadConfig {
    pub d_model: usize,
}

impl PointerHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PointerHead<B> {
        let d = self.d_model;
        let ability_queries = (0..MAX_ABILITIES)
            .map(|_| LinearConfig::new(d, d).init(device))
            .collect();

        PointerHead {
            d_model: d,
            action_type_l1: LinearConfig::new(d, d).init(device),
            action_type_l2: LinearConfig::new(d, NUM_ACTION_TYPES).init(device),
            pointer_key: LinearConfig::new(d, d).init(device),
            attack_query: LinearConfig::new(d, d).init(device),
            move_query: LinearConfig::new(d, d).init(device),
            ability_queries,
            gelu: Gelu::new(),
        }
    }
}

/// Output from PointerHead forward pass.
pub struct BurnPointerOutput<B: Backend> {
    /// Action type logits `[B, 11]`.
    pub type_logits: Tensor<B, 2>,
    /// Attack pointer logits `[B, N]`.
    pub attack_ptr: Tensor<B, 2>,
    /// Move pointer logits `[B, N]`.
    pub move_ptr: Tensor<B, 2>,
    /// Per-ability pointer logits, each `[B, N]` or None.
    pub ability_ptrs: Vec<Option<Tensor<B, 2>>>,
}

impl<B: Backend> PointerHead<B> {
    /// Compute action type logits and pointer distributions.
    ///
    /// * `pooled`: `[B, d_model]`
    /// * `entity_tokens`: `[B, N, d_model]`
    /// * `entity_mask`: `[B, N]` bool True=padding
    /// * `ability_cross_embs`: per-ability cross-attended embeddings
    /// * `entity_type_ids_full`: `[B, N]` int (0-4)
    pub fn forward(
        &self,
        pooled: Tensor<B, 2>,
        entity_tokens: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        ability_cross_embs: &[Option<Tensor<B, 2>>],
        entity_type_ids_full: Tensor<B, 2, Int>,
    ) -> BurnPointerOutput<B> {
        let [batch, n_tokens, _] = entity_tokens.dims();
        let device = &entity_tokens.device();
        let scale = (self.d_model as f64).powf(-0.5);

        // Action type logits
        let type_hidden = self.action_type_l1.forward(pooled.clone());
        let type_hidden = self.gelu.forward(type_hidden);
        let type_logits = self.action_type_l2.forward(type_hidden);

        // Compute keys for all tokens: [B, N, d]
        let keys = self.pointer_key.forward(entity_tokens);

        // Attack pointer: only enemies (type=1) are valid
        let atk_q = self.attack_query.forward(pooled.clone()); // [B, d]
        let atk_q = atk_q.unsqueeze_dim(1); // [B, 1, d]
        let atk_ptr = atk_q
            .matmul(keys.clone().transpose())
            .squeeze::<2>()
            .mul_scalar(scale); // [B, N]
        let atk_invalid =
            entity_type_ids_full.clone().not_equal_elem(1).bool_or(entity_mask.clone());
        let atk_ptr = atk_ptr.mask_fill(atk_invalid, -1e9);

        // Move pointer: non-self (type!=0) and non-padding
        let mv_q = self.move_query.forward(pooled.clone());
        let mv_q = mv_q.unsqueeze_dim(1);
        let mv_ptr = mv_q
            .matmul(keys.clone().transpose())
            .squeeze::<2>()
            .mul_scalar(scale);
        let mv_invalid =
            entity_type_ids_full.clone().equal_elem(0).bool_or(entity_mask.clone());
        let mv_ptr = mv_ptr.mask_fill(mv_invalid, -1e9);

        // Ability pointers
        let mut ability_ptrs = Vec::with_capacity(MAX_ABILITIES);
        for (i, cross_emb_opt) in ability_cross_embs
            .iter()
            .enumerate()
            .take(MAX_ABILITIES)
        {
            if let Some(cross_emb) = cross_emb_opt {
                let ab_q = self.ability_queries[i].forward(cross_emb.clone());
                let ab_q = ab_q.unsqueeze_dim(1);
                let ab_ptr = ab_q
                    .matmul(keys.clone().transpose())
                    .squeeze::<2>()
                    .mul_scalar(scale);
                let ab_ptr = ab_ptr.mask_fill(entity_mask.clone(), -1e9);
                ability_ptrs.push(Some(ab_ptr));
            } else {
                ability_ptrs.push(None);
            }
        }

        BurnPointerOutput {
            type_logits,
            attack_ptr: atk_ptr,
            move_ptr: mv_ptr,
            ability_ptrs,
        }
    }
}

// ---------------------------------------------------------------------------
// AbilityActorCritic V3 — pointer-based action space + position tokens
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityActorCriticV3<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoderV3<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub pointer_head: PointerHead<B>,
    pub value_l1: Linear<B>,
    pub value_l2: Linear<B>,
    pub gelu: Gelu,
    pub d_model: usize,
}

#[derive(Config, Debug)]
pub struct AbilityActorCriticV3Config {
    pub vocab_size: usize,
    #[config(default = "4")]
    pub entity_encoder_layers: usize,
    #[config(default = "64")]
    pub d_model: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_ff: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
}

impl AbilityActorCriticV3Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityActorCriticV3<B> {
        let transformer = AbilityTransformerConfig::new(self.vocab_size)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_max_seq_len(self.max_seq_len)
            .init(device);

        let entity_encoder = EntityEncoderV3Config::new()
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.entity_encoder_layers)
            .init(device);

        let cross_attn = CrossAttentionBlockConfig::new(self.d_model)
            .with_n_heads(self.n_heads)
            .init(device);

        let pointer_head = PointerHeadConfig::new(self.d_model).init(device);

        let d = self.d_model;
        AbilityActorCriticV3 {
            transformer,
            entity_encoder,
            cross_attn,
            pointer_head,
            value_l1: LinearConfig::new(d, d).init(device),
            value_l2: LinearConfig::new(d, 1).init(device),
            gelu: Gelu::new(),
            d_model: d,
        }
    }
}

impl<B: Backend> AbilityActorCriticV3<B> {
    /// Encode v3 game state → (tokens, mask, pooled).
    pub fn encode_entities(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
        position_features: Option<Tensor<B, 3>>,
        position_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>, Tensor<B, 2>) {
        let (tokens, full_mask) = self.entity_encoder.forward(
            entity_features,
            entity_type_ids,
            threat_features,
            entity_mask,
            threat_mask,
            position_features,
            position_mask,
        );

        let exist = full_mask.clone().bool_not().float().unsqueeze_dim(2);
        let sum = (tokens.clone() * exist.clone()).sum_dim(1);
        let count = exist.sum_dim(1).clamp_min(1.0);
        let pooled = (sum / count).squeeze::<2>();

        (tokens, full_mask, pooled)
    }

    /// Build full type IDs for the token sequence (entities+threats+positions).
    fn build_full_type_ids(
        entity_type_ids: Tensor<B, 2, Int>,
        n_threats: usize,
        n_positions: usize,
        batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        let threat_types = Tensor::<B, 2, Int>::full([batch, n_threats], 3, device);
        let mut parts = vec![entity_type_ids, threat_types];
        if n_positions > 0 {
            let pos_types =
                Tensor::<B, 2, Int>::full([batch, n_positions], 4, device);
            parts.push(pos_types);
        }
        Tensor::cat(parts, 1)
    }

    /// Compute state value V(s).
    pub fn forward_value(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
        position_features: Option<Tensor<B, 3>>,
        position_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let (_, _, pooled) = self.encode_entities(
            entity_features,
            entity_type_ids,
            threat_features,
            entity_mask,
            threat_mask,
            position_features,
            position_mask,
        );
        let v = self.value_l1.forward(pooled);
        let v = self.gelu.forward(v);
        self.value_l2.forward(v)
    }

    /// Returns (pointer_output, state_value [B, 1]).
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        threat_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_mask: Tensor<B, 2, Bool>,
        ability_cls: &[Option<Tensor<B, 2>>],
        position_features: Option<Tensor<B, 3>>,
        position_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (BurnPointerOutput<B>, Tensor<B, 2>) {
        let [batch, ..] = entity_features.dims();
        let [_, n_threats, _] = threat_features.dims();
        let n_positions = position_features
            .as_ref()
            .map(|p| p.dims()[1])
            .unwrap_or(0);
        let device = &entity_features.device();

        let (tokens, full_mask, pooled) = self.encode_entities(
            entity_features,
            entity_type_ids.clone(),
            threat_features,
            entity_mask,
            threat_mask,
            position_features,
            position_mask,
        );

        // Cross-attend each ability CLS to entity tokens
        let mut ability_cross_embs: Vec<Option<Tensor<B, 2>>> =
            Vec::with_capacity(MAX_ABILITIES);
        for cls_opt in ability_cls.iter().take(MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb =
                    self.cross_attn
                        .forward(cls.clone(), tokens.clone(), Some(full_mask.clone()));
                ability_cross_embs.push(Some(cross_emb));
            } else {
                ability_cross_embs.push(None);
            }
        }

        // Build full type IDs for pointer masking
        let full_type_ids = Self::build_full_type_ids(
            entity_type_ids,
            n_threats,
            n_positions,
            batch,
            device,
        );

        let pointer_out = self.pointer_head.forward(
            pooled.clone(),
            tokens,
            full_mask,
            &ability_cross_embs,
            full_type_ids,
        );

        let value = self.value_l1.forward(pooled);
        let value = self.gelu.forward(value);
        let value = self.value_l2.forward(value);

        (pointer_out, value)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_ability_transformer_shapes() {
        let device = Default::default();
        let model = AbilityTransformerConfig::new(252)
            .with_d_model(32)
            .with_n_heads(4)
            .with_n_layers(2)
            .with_d_ff(64)
            .with_max_seq_len(64)
            .init::<TestBackend>(&device);

        let input_ids =
            Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);
        let hidden = model.forward(input_ids.clone(), None);
        assert_eq!(hidden.dims(), [2, 10, 32]);

        let cls = model.cls_embedding(input_ids, None);
        assert_eq!(cls.dims(), [2, 32]);
    }

    #[test]
    fn test_mlm_head_shapes() {
        let device = Default::default();
        let head = MLMHeadConfig::new(32, 252).init::<TestBackend>(&device);
        let hidden = Tensor::<TestBackend, 3>::zeros([2, 10, 32], &device);
        let logits = head.forward(hidden);
        assert_eq!(logits.dims(), [2, 10, 252]);
    }

    #[test]
    fn test_decision_head_shapes() {
        let device = Default::default();
        let head = DecisionHeadConfig::new(32)
            .with_n_targets(3)
            .init::<TestBackend>(&device);
        let cls = Tensor::<TestBackend, 2>::zeros([2, 32], &device);
        let (urgency, targets) = head.forward(cls);
        assert_eq!(urgency.dims(), [2, 1]);
        assert_eq!(targets.dims(), [2, 3]);
    }

    #[test]
    fn test_entity_encoder_v1_shapes() {
        let device = Default::default();
        let enc = EntityEncoderConfig::new(32)
            .with_n_heads(4)
            .with_n_layers(2)
            .init::<TestBackend>(&device);
        let gs = Tensor::<TestBackend, 2>::zeros([2, 210], &device);
        let (tokens, mask) = enc.forward(gs);
        assert_eq!(tokens.dims(), [2, 7, 32]);
        assert_eq!(mask.dims(), [2, 7]);
    }

    #[test]
    fn test_cross_attention_shapes() {
        let device = Default::default();
        let ca = CrossAttentionBlockConfig::new(32)
            .with_n_heads(4)
            .init::<TestBackend>(&device);
        let query = Tensor::<TestBackend, 2>::zeros([2, 32], &device);
        let kv = Tensor::<TestBackend, 3>::zeros([2, 7, 32], &device);
        let out = ca.forward(query, kv, None);
        assert_eq!(out.dims(), [2, 32]);
    }

    #[test]
    fn test_actor_critic_v1_shapes() {
        let device = Default::default();
        let model = AbilityActorCriticConfig::new(252)
            .with_d_model(32)
            .with_n_heads(4)
            .with_n_layers(2)
            .with_d_ff(64)
            .init::<TestBackend>(&device);

        let gs = Tensor::<TestBackend, 2>::zeros([2, 210], &device);
        let cls = Tensor::<TestBackend, 2>::zeros([2, 32], &device);
        let ability_cls: Vec<Option<Tensor<TestBackend, 2>>> =
            vec![Some(cls.clone()), None, None, None, None, None, None, None];

        let (logits, value) = model.forward(gs, &ability_cls);
        assert_eq!(logits.dims(), [2, 14]);
        assert_eq!(value.dims(), [2, 1]);
    }

    #[test]
    fn test_mlm_pretraining_model() {
        let device = Default::default();
        let model = AbilityTransformerMLMConfig::new(252)
            .with_d_model(32)
            .with_n_heads(4)
            .with_n_layers(2)
            .with_d_ff(64)
            .init::<TestBackend>(&device);

        let input_ids =
            Tensor::<TestBackend, 2, Int>::zeros([2, 10], &device);
        let logits = model.forward(input_ids, None);
        assert_eq!(logits.dims(), [2, 10, 252]);
    }

    #[test]
    fn test_pointer_head_shapes() {
        let device = Default::default();
        let head = PointerHeadConfig::new(32).init::<TestBackend>(&device);
        let pooled = Tensor::<TestBackend, 2>::zeros([2, 32], &device);
        let tokens = Tensor::<TestBackend, 3>::zeros([2, 12, 32], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::full([2, 12], false, &device);
        let type_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 12], &device);
        let cls = Tensor::<TestBackend, 2>::zeros([2, 32], &device);
        let ability_embs: Vec<Option<Tensor<TestBackend, 2>>> =
            vec![Some(cls), None, None, None, None, None, None, None];

        let out = head.forward(pooled, tokens, mask, &ability_embs, type_ids);
        assert_eq!(out.type_logits.dims(), [2, 11]);
        assert_eq!(out.attack_ptr.dims(), [2, 12]);
        assert_eq!(out.move_ptr.dims(), [2, 12]);
        assert!(out.ability_ptrs[0].is_some());
        assert!(out.ability_ptrs[1].is_none());
    }
}
