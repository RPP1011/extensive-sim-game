//! V5 Entity Encoder in Burn.
//!
//! Takes entities (34-dim), zones (12-dim), and optional aggregate (16-dim),
//! projects them to d_model, adds type embeddings, runs self-attention.
//! Entity types: self=0, enemy=1, ally=2, zone=3, aggregate=4.

use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::prelude::*;

use super::config::*;

#[derive(Module, Debug)]
pub struct EntityEncoderV5<B: Backend> {
    entity_proj: Linear<B>,
    zone_proj: Linear<B>,
    agg_proj: Linear<B>,
    type_emb: Embedding<B>,
    input_norm: LayerNorm<B>,
    encoder: TransformerEncoder<B>,
    out_norm: LayerNorm<B>,
    d_model: usize,
}

#[derive(Config, Debug)]
pub struct EntityEncoderV5Config {
    #[config(default = "D_MODEL")]
    pub d_model: usize,
    #[config(default = "N_HEADS")]
    pub n_heads: usize,
    #[config(default = "N_ENCODER_LAYERS")]
    pub n_layers: usize,
}

impl EntityEncoderV5Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EntityEncoderV5<B> {
        let d = self.d_model;
        EntityEncoderV5 {
            entity_proj: LinearConfig::new(ENTITY_DIM, d).init(device),
            zone_proj: LinearConfig::new(ZONE_DIM, d).init(device),
            agg_proj: LinearConfig::new(AGG_DIM, d).init(device),
            type_emb: EmbeddingConfig::new(NUM_ENTITY_TYPES, d).init(device),
            input_norm: LayerNormConfig::new(d).init(device),
            encoder: TransformerEncoderConfig::new(d, d * 2, self.n_heads, self.n_layers)
                .with_norm_first(true)
                .with_dropout(0.0)
                .init(device),
            out_norm: LayerNormConfig::new(d).init(device),
            d_model: d,
        }
    }
}

impl<B: Backend> EntityEncoderV5<B> {
    /// Forward pass: project entities + zones + aggregate, run self-attention.
    ///
    /// Returns (tokens [B, S, d_model], mask [B, S] where true = valid).
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,   // [B, E, 34]
        entity_type_ids: Tensor<B, 2, Int>, // [B, E]
        zone_features: Tensor<B, 3>,      // [B, Z, 12]
        entity_mask: Tensor<B, 2, Bool>,  // [B, E] true = valid
        zone_mask: Tensor<B, 2, Bool>,    // [B, Z] true = valid
        aggregate_features: Option<Tensor<B, 2>>, // [B, 16]
    ) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let device = entity_features.device();
        let [batch, n_ent, _] = entity_features.dims();
        let [_, n_zones, _] = zone_features.dims();

        // Project entities + type embeddings
        let ent_tokens = self.entity_proj.forward(entity_features)
            + self.type_emb.forward(entity_type_ids);

        // Project zones (type_id = 3)
        let zone_type_ids = Tensor::<B, 2, Int>::full([batch, n_zones], 3, &device);
        let zone_tokens = self.zone_proj.forward(zone_features)
            + self.type_emb.forward(zone_type_ids);

        // Concatenate tokens and masks
        let (mut tokens, mut mask) = (
            Tensor::cat(vec![ent_tokens, zone_tokens], 1),
            Tensor::cat(vec![entity_mask.clone(), zone_mask.clone()], 1),
        );

        // Aggregate token (type_id = 4)
        if let Some(agg) = aggregate_features {
            let agg_type = Tensor::<B, 2, Int>::full([batch, 1], 4, &device);
            let agg_token: Tensor<B, 3> = self.agg_proj.forward(agg).unsqueeze_dim::<3>(1)
                + self.type_emb.forward(agg_type);
            tokens = Tensor::cat(vec![tokens, agg_token], 1);
            // Aggregate is always valid
            let agg_mask = Tensor::<B, 2, Bool>::full([batch, 1], true, &device);
            mask = Tensor::cat(vec![mask, agg_mask], 1);
        }

        // Input norm
        tokens = self.input_norm.forward(tokens);

        // Self-attention (Burn expects mask where true = padded, but we have true = valid)
        let pad_mask = mask.clone().bool_not();
        let input = TransformerEncoderInput::new(tokens).mask_pad(pad_mask);
        tokens = self.encoder.forward(input);

        // Output norm
        tokens = self.out_norm.forward(tokens);

        (tokens, mask)
    }
}
