//! V6 Actor-Critic: full model with latent interface + spatial cross-attention.
//!
//! Pipeline:
//!   entity encoder → spatial cross-attention → latent interface → CfC → decision heads
//!
//! New over V5:
//! - SpatialCrossAttention: entity tokens attend to visible corner tokens
//! - LatentInterface: ELIT-style learned bottleneck with Read/Write/tail-dropping
//! - ValueHead: two-headed value prediction for curriculum pretraining
//! - h_dim=256 for CfC temporal cell

use burn::module::{Module, Param};
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

use super::config::*;
use super::entity_encoder::{EntityEncoderV5, EntityEncoderV5Config};
use super::cross_attention::{CrossAttentionBlock, CrossAttentionConfig};
use super::cfc_cell::{CfCCell, CfCCellConfig};
use super::combat_head::{CombatPointerHead, CombatPointerHeadConfig, CombatOutput};
use super::latent_interface::{LatentInterface, LatentInterfaceConfig};
use super::spatial_cross_attn::{SpatialCrossAttention, SpatialCrossAttentionConfig};
use super::value_head::{ValueHead, ValueHeadConfig, ValueOutput};
use super::actor_critic::{
    AbilityTransformer, AbilityTransformerConfig,
    ActorCriticOutput, EncodedState,
};

/// V6 h_dim (CfC hidden state dimension).
pub const V6_H_DIM: usize = 256;

#[derive(Module, Debug)]
pub struct ActorCriticV6<B: Backend> {
    pub transformer: AbilityTransformer<B>,
    pub entity_encoder: EntityEncoderV5<B>,
    pub spatial_cross_attn: SpatialCrossAttention<B>,
    pub latent_interface: LatentInterface<B>,
    pub cross_attn: CrossAttentionBlock<B>,
    pub temporal_cell: CfCCell<B>,
    pub position_head_l1: Linear<B>,
    pub position_head_l2: Linear<B>,
    /// Learnable log standard deviation for movement Gaussian policy [2].
    /// σ = exp(move_log_std). Initialized to log(1.0) = 0 → σ=1.0 in normalized coords.
    pub move_log_std: Param<Tensor<B, 1>>,
    pub combat_head: CombatPointerHead<B>,
    pub value_head: ValueHead<B>,
    pub external_cls_proj: Option<Linear<B>>,
    gelu: Gelu,
    d_model: usize,
}

#[derive(Config, Debug)]
pub struct ActorCriticV6Config {
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
    #[config(default = "V6_H_DIM")]
    pub h_dim: usize,
    #[config(default = "N_LATENT_TOKENS")]
    pub n_latents: usize,
    #[config(default = "N_LATENT_BLOCKS")]
    pub n_latent_blocks: usize,
}

impl ActorCriticV6Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActorCriticV6<B> {
        let d = self.d_model;
        let cls_proj = if self.external_cls_dim > 0 && self.external_cls_dim != d {
            Some(LinearConfig::new(self.external_cls_dim, d).init(device))
        } else {
            None
        };

        ActorCriticV6 {
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
            spatial_cross_attn: SpatialCrossAttentionConfig {
                d_model: d,
                n_heads: self.n_heads,
            }
            .init(device),
            latent_interface: LatentInterfaceConfig {
                d_model: d,
                n_heads: self.n_heads,
                n_latents: self.n_latents,
                n_latent_blocks: self.n_latent_blocks,
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
            // log(σ) initialized to 0 → σ=1.0 in normalized coords (= 20 world units)
            move_log_std: Param::from_tensor(Tensor::zeros([2], device)),
            combat_head: CombatPointerHeadConfig { d_model: d }.init(device),
            value_head: ValueHeadConfig { d_model: d }.init(device),
            external_cls_proj: cls_proj,
            gelu: Gelu::new(),
            d_model: d,
        }
    }
}

impl<B: Backend> ActorCriticV6<B> {
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

    /// Encode game state: entity encoder → spatial cross-attention → latent interface.
    pub fn encode_state(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        zone_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        zone_mask: Tensor<B, 2, Bool>,
        aggregate_features: Option<Tensor<B, 2>>,
        corner_tokens: Option<Tensor<B, 3>>,     // [B, C, 11]
        corner_mask: Option<Tensor<B, 2, Bool>>,  // [B, C]
        n_latents_override: Option<usize>,
    ) -> (EncodedState<B>, Tensor<B, 2>) {
        let device = entity_features.device();
        let [batch, _, _] = entity_features.dims();

        // Step 1: Entity encoder
        let (tokens, mask) = self.entity_encoder.forward(
            entity_features,
            entity_type_ids.clone(),
            zone_features.clone(),
            entity_mask.clone(),
            zone_mask.clone(),
            aggregate_features,
        );

        // Step 2: Spatial cross-attention (if corner tokens provided)
        let tokens = if let (Some(corners), Some(cmask)) = (corner_tokens, corner_mask) {
            self.spatial_cross_attn.forward(tokens, corners, cmask)
        } else {
            tokens
        };

        // Step 3: Latent interface
        let (tokens, pooled_latents) = self.latent_interface.forward(
            tokens.clone(), mask.clone(), n_latents_override,
        );

        // Build type IDs for combat pointer head
        let [_, n_zones, _] = zone_features.dims();
        let zone_type_ids = Tensor::<B, 2, Int>::full([batch, n_zones], 3, &device);
        let mut type_parts = vec![entity_type_ids, zone_type_ids];
        let [_, total_tokens, _] = tokens.dims();
        let n_ent_plus_zone: usize = entity_mask.dims()[1] + n_zones;
        if total_tokens > n_ent_plus_zone {
            let agg_type = Tensor::<B, 2, Int>::full([batch, 1], 4, &device);
            type_parts.push(agg_type);
        }
        let type_ids = Tensor::cat(type_parts, 1);

        let enc = EncodedState { pooled: pooled_latents.clone(), tokens, mask, type_ids };
        (enc, pooled_latents)
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
        let pos_h = self.gelu.forward(self.position_head_l1.forward(pooled.clone()));
        let target_pos = self.position_head_l2.forward(pos_h);
        let move_log_std = Some(self.move_log_std.val());
        let combat = self.combat_head.forward(
            pooled, tokens, mask, type_ids, ability_cross_embs,
        );
        ActorCriticOutput { target_pos, move_log_std, combat }
    }

    /// Full forward pass: encode → spatial → latent → CfC → decide.
    ///
    /// Returns: (action output, new hidden state, optional value output)
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
        corner_tokens: Option<Tensor<B, 3>>,
        corner_mask: Option<Tensor<B, 2, Bool>>,
        n_latents_override: Option<usize>,
    ) -> (ActorCriticOutput<B>, Tensor<B, 2>, ValueOutput<B>) {
        let (enc, pooled_latents) = self.encode_state(
            entity_features, entity_type_ids, zone_features,
            entity_mask, zone_mask, aggregate_features,
            corner_tokens, corner_mask, n_latents_override,
        );

        // Cross-attend abilities
        let ability_cross_embs = self.cross_attend_abilities(
            ability_cls, &enc.tokens, &enc.mask,
        );

        // CfC temporal cell (uses pooled latents, not mean-pooled tokens)
        let h_prev = h_prev.unwrap_or_else(|| {
            let [batch, _] = pooled_latents.dims();
            Tensor::zeros([batch, self.temporal_cell.h_dim()], &pooled_latents.device())
        });
        let (temporal_out, h_new) = self.temporal_cell.forward(pooled_latents.clone(), h_prev, 1.0);

        // Value head (from pooled latents, before temporal cell mixes with hidden state)
        let value = self.value_head.forward(pooled_latents);

        // Decision heads (from temporal output)
        let output = self.decide(temporal_out, enc.tokens, enc.mask, enc.type_ids, &ability_cross_embs);

        (output, h_new, value)
    }

    /// Forward pass without value head (inference mode).
    pub fn forward_inference(
        &self,
        entity_features: Tensor<B, 3>,
        entity_type_ids: Tensor<B, 2, Int>,
        zone_features: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        zone_mask: Tensor<B, 2, Bool>,
        ability_cls: &[Option<Tensor<B, 2>>],
        aggregate_features: Option<Tensor<B, 2>>,
        h_prev: Option<Tensor<B, 2>>,
        corner_tokens: Option<Tensor<B, 3>>,
        corner_mask: Option<Tensor<B, 2, Bool>>,
        n_latents: Option<usize>,
    ) -> (ActorCriticOutput<B>, Tensor<B, 2>) {
        let (enc, pooled_latents) = self.encode_state(
            entity_features, entity_type_ids, zone_features,
            entity_mask, zone_mask, aggregate_features,
            corner_tokens, corner_mask, n_latents,
        );

        let ability_cross_embs = self.cross_attend_abilities(
            ability_cls, &enc.tokens, &enc.mask,
        );

        let h_prev = h_prev.unwrap_or_else(|| {
            let [batch, _] = pooled_latents.dims();
            Tensor::zeros([batch, self.temporal_cell.h_dim()], &pooled_latents.device())
        });
        let (temporal_out, h_new) = self.temporal_cell.forward(pooled_latents, h_prev, 1.0);

        let output = self.decide(temporal_out, enc.tokens, enc.mask, enc.type_ids, &ability_cross_embs);
        (output, h_new)
    }
}
