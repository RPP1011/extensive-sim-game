//! V5 Actor-Critic: top-level model combining all components.
//!
//! entity encoder → (latent interface) → cross-attention → CfC → decision heads
//!
//! The latent interface is optional (GPU-only during training).
//! For CPU inference, we skip it and mean-pool directly.

use burn::module::Module;
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::config::*;
use super::entity_encoder::{EntityEncoderV5, EntityEncoderV5Config};
use super::cross_attention::{CrossAttentionBlock, CrossAttentionConfig};
use super::cfc_cell::{CfCCell, CfCCellConfig};
use super::combat_head::{CombatPointerHead, CombatPointerHeadConfig, CombatOutput};

// ---------------------------------------------------------------------------
// Ability Transformer (for CLS embeddings from DSL tokens)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AbilityTransformer<B: Backend> {
    token_emb: burn::nn::Embedding<B>,
    pos_emb: burn::nn::Embedding<B>,
    encoder: TransformerEncoder<B>,
    out_norm: LayerNorm<B>,
    d_model: usize,
    pad_id: usize,
}

#[derive(Config, Debug)]
pub struct AbilityTransformerConfig {
    pub vocab_size: usize,
    #[config(default = "D_MODEL")]
    pub d_model: usize,
    #[config(default = "N_HEADS")]
    pub n_heads: usize,
    #[config(default = "4")]
    pub n_layers: usize,
    #[config(default = "256")]
    pub max_seq_len: usize,
    #[config(default = "1")]
    pub pad_id: usize,
}

impl AbilityTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AbilityTransformer<B> {
        let d = self.d_model;
        AbilityTransformer {
            token_emb: burn::nn::EmbeddingConfig::new(self.vocab_size, d).init(device),
            pos_emb: burn::nn::EmbeddingConfig::new(self.max_seq_len, d).init(device),
            encoder: TransformerEncoderConfig::new(d, d * 2, self.n_heads, self.n_layers)
                .with_norm_first(true)
                .with_dropout(0.0)
                .init(device),
            out_norm: LayerNormConfig::new(d).init(device),
            d_model: d,
            pad_id: self.pad_id,
        }
    }
}

impl<B: Backend> AbilityTransformer<B> {
    /// Encode ability token IDs to CLS embedding.
    /// token_ids: [B, S] (padded with pad_id)
    /// Returns: [B, d_model] (CLS = first token)
    pub fn encode_cls(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch, seq_len] = token_ids.dims();
        let device = token_ids.device();

        // Position IDs
        let pos_ids: Tensor<B, 2, Int> = Tensor::<B, 1, Int>::arange(0..(seq_len as i64), &device)
            .unsqueeze_dim::<2>(0)
            .expand([batch, seq_len]);

        // Embed
        let tokens = self.token_emb.forward(token_ids.clone())
            + self.pos_emb.forward(pos_ids);

        // Padding mask (true = padded)
        let pad_mask = token_ids.equal_elem(self.pad_id as i64);

        let input = TransformerEncoderInput::new(tokens).mask_pad(pad_mask);
        let encoded = self.encoder.forward(input);

        // CLS = first token, then norm
        let cls: Tensor<B, 2> = encoded.slice([0..batch, 0..1]).squeeze_dim::<2>(1);
        self.out_norm.forward(cls)
    }
}

// ---------------------------------------------------------------------------
// Full Actor-Critic V5
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ActorCriticV5<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoderV5<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub temporal_cell: CfCCell<B>,
    pub position_head_l1: Linear<B>,
    pub position_head_l2: Linear<B>,
    pub combat_head: CombatPointerHead<B>,
    pub external_cls_proj: Option<Linear<B>>,
    gelu: Gelu,
    d_model: usize,
}

#[derive(Config, Debug)]
pub struct ActorCriticV5Config {
    pub vocab_size: usize,
    #[config(default = "D_MODEL")]
    pub d_model: usize,
    #[config(default = "D_FF")]
    pub d_ff: usize,
    #[config(default = "N_HEADS")]
    pub n_heads: usize,
    #[config(default = "4")]
    pub n_layers: usize,
    #[config(default = "N_ENCODER_LAYERS")]
    pub entity_encoder_layers: usize,
    #[config(default = "0")]
    pub external_cls_dim: usize,
    #[config(default = "H_DIM")]
    pub h_dim: usize,
}

impl ActorCriticV5Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorCriticV5<B> {
        let d = self.d_model;
        let cls_proj = if self.external_cls_dim > 0 && self.external_cls_dim != d {
            Some(LinearConfig::new(self.external_cls_dim, d).init(device))
        } else {
            None
        };

        ActorCriticV5 {
            transformer: AbilityTransformerConfig {
                vocab_size: self.vocab_size,
                d_model: d,
                n_heads: self.n_heads,
                n_layers: self.n_layers,
                max_seq_len: 256,
                pad_id: 1,
            }
            .init(device),
            entity_encoder: EntityEncoderV5Config {
                d_model: d,
                n_heads: self.n_heads,
                n_layers: self.entity_encoder_layers,
            }
            .init(device),
            cross_attn: CrossAttentionConfig {
                d_model: d,
                n_heads: self.n_heads,
            }
            .init(device),
            temporal_cell: CfCCellConfig {
                d_model: d,
                h_dim: self.h_dim,
            }
            .init(device),
            position_head_l1: LinearConfig::new(d, d).init(device),
            position_head_l2: LinearConfig::new(d, 2).init(device),
            combat_head: CombatPointerHeadConfig { d_model: d }.init(device),
            external_cls_proj: cls_proj,
            gelu: Gelu::new(),
            d_model: d,
        }
    }
}

/// Encoded game state (intermediate representation).
pub struct EncodedState<B: Backend> {
    pub pooled: Tensor<B, 2>,         // [B, d]
    pub tokens: Tensor<B, 3>,         // [B, S, d]
    pub mask: Tensor<B, 2, Bool>,     // [B, S]
    pub type_ids: Tensor<B, 2, Int>,  // [B, S]
}

/// Full model output.
pub struct ActorCriticOutput<B: Backend> {
    /// Target waypoint (normalized x/20, y/20): [B, 2]
    pub target_pos: Tensor<B, 2>,
    /// Combat action outputs
    pub combat: CombatOutput<B>,
}

impl<B: Backend> ActorCriticV5<B> {
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Project external CLS embedding to d_model.
    pub fn project_cls(&self, cls: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self.external_cls_proj {
            Some(proj) => proj.forward(cls),
            None => cls,
        }
    }

    /// Encode game state: entity encoder → mean pool.
    /// No latent interface or CfC (those are separate steps).
    pub fn encode_state(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        zone_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        zone_mask: Tensor<B, 2, Bool>,
        aggregate_features: Option<Tensor<B, 2>>,
    ) -> EncodedState<B> {
        let (tokens, mask) = self.entity_encoder.forward(
            entity_features,
            entity_type_ids.clone(),
            zone_features.clone(),
            entity_mask.clone(),
            zone_mask.clone(),
            aggregate_features,
        );

        // Mean pool over valid tokens
        let mask_float: Tensor<B, 3> = mask.clone().float().unsqueeze_dim::<3>(2); // [B, S, 1]
        let sum_tokens: Tensor<B, 2> = (tokens.clone() * mask_float.clone()).sum_dim(1).squeeze_dim::<2>(1);
        let count: Tensor<B, 2> = mask_float.sum_dim(1).squeeze_dim::<2>(1).clamp_min(1.0);
        let pooled = sum_tokens / count;

        // Build full type IDs: [entities..., zones..., maybe aggregate]
        let device = entity_type_ids.device();
        let [batch, n_zones, _] = zone_features.dims();
        let zone_type_ids = Tensor::<B, 2, Int>::full([batch, n_zones], 3, &device);
        let mut type_parts = vec![entity_type_ids, zone_type_ids];
        // If aggregate was included, add type_id=4
        let [_, total_tokens, _] = tokens.dims();
        let n_ent_plus_zone: usize = entity_mask.dims()[1] + n_zones;
        if total_tokens > n_ent_plus_zone {
            let agg_type = Tensor::<B, 2, Int>::full([batch, 1], 4, &device);
            type_parts.push(agg_type);
        }
        let type_ids = Tensor::cat(type_parts, 1);

        EncodedState { pooled, tokens, mask, type_ids }
    }

    /// Cross-attend ability CLS embeddings to entity tokens.
    pub fn cross_attend_abilities(
        &self,
        ability_cls: &[Option<Tensor<B, 2>>],
        tokens: &Tensor<B, 3>,
        mask: &Tensor<B, 2, Bool>,
    ) -> Vec<Option<Tensor<B, 2>>> {
        ability_cls.iter().map(|cls_opt| {
            cls_opt.as_ref().map(|cls| {
                let projected = self.project_cls(cls.clone());
                self.cross_attn.forward(projected, tokens.clone(), mask.clone())
            })
        }).collect()
    }

    /// Run decision heads on pooled state.
    pub fn decide(
        &self,
        pooled: Tensor<B, 2>,
        tokens: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        type_ids: Tensor<B, 2, Int>,
        ability_cross_embs: &[Option<Tensor<B, 2>>],
    ) -> ActorCriticOutput<B> {
        // Position head: [B, 2] normalized waypoint
        let pos_h = self.gelu.forward(self.position_head_l1.forward(pooled.clone()));
        let target_pos = self.position_head_l2.forward(pos_h);

        // Combat head
        let combat = self.combat_head.forward(
            pooled, tokens, mask, type_ids, ability_cross_embs,
        );

        ActorCriticOutput { target_pos, combat }
    }

    /// Full forward pass (encode + CfC + decide).
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        zone_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        zone_mask: Tensor<B, 2, Bool>,
        ability_cls: &[Option<Tensor<B, 2>>],
        aggregate_features: Option<Tensor<B, 2>>,
        h_prev: Option<Tensor<B, 2>>,
    ) -> (ActorCriticOutput<B>, Tensor<B, 2>) {
        let enc = self.encode_state(
            entity_features, entity_type_ids, zone_features,
            entity_mask, zone_mask, aggregate_features,
        );

        // Cross-attend abilities
        let ability_cross_embs = self.cross_attend_abilities(
            ability_cls, &enc.tokens, &enc.mask,
        );

        // CfC temporal cell
        let h_prev = h_prev.unwrap_or_else(|| {
            let [batch, _] = enc.pooled.dims();
            Tensor::zeros([batch, self.temporal_cell.h_dim()], &enc.pooled.device())
        });
        let (pooled, h_new) = self.temporal_cell.forward(enc.pooled, h_prev, 1.0);

        // Decision heads
        let output = self.decide(pooled, enc.tokens, enc.mask, enc.type_ids, &ability_cross_embs);

        (output, h_new)
    }
}
