//! ELIT-style Latent Interface for V6.
//!
//! K learned latent tokens compress the entity token sequence via Read/Write
//! cross-attention steps with latent self-attention in between.
//!
//! - Read: latents attend to (spatially-enriched) entity tokens
//! - 2x latent transformer blocks (self-attention)
//! - Write: entity tokens updated by latents (zero-init output proj at init)
//!
//! Tail dropping: during training, randomly use a subset of latents.
//! At inference, pick J based on compute budget.

use burn::module::{Module, Param};
use burn::nn::{
    Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::config::*;

#[derive(Module, Debug)]
pub struct LatentInterface<B: Backend> {
    /// Learned latent tokens: [K, d_model]
    latents: Param<Tensor<B, 2>>,
    /// Read: latents (Q) attend to entity tokens (K/V)
    read_q: Linear<B>,
    read_k: Linear<B>,
    read_v: Linear<B>,
    read_out: Linear<B>,
    read_norm: LayerNorm<B>,
    /// Latent self-attention blocks
    latent_blocks: TransformerEncoder<B>,
    /// Write: entity tokens (Q) attend to latents (K/V)
    write_q: Linear<B>,
    write_k: Linear<B>,
    write_v: Linear<B>,
    write_out: Linear<B>,
    write_norm: LayerNorm<B>,
    n_heads: usize,
    d_model: usize,
    n_latents: usize,
    scale: f32,
}

#[derive(Config, Debug)]
pub struct LatentInterfaceConfig {
    #[config(default = "D_MODEL")]
    pub d_model: usize,
    #[config(default = "N_HEADS")]
    pub n_heads: usize,
    #[config(default = "N_LATENT_TOKENS")]
    pub n_latents: usize,
    #[config(default = "N_LATENT_BLOCKS")]
    pub n_latent_blocks: usize,
}

impl LatentInterfaceConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LatentInterface<B> {
        let d = self.d_model;
        let head_dim = d / self.n_heads;

        // Initialize latents with small normal
        let latents = Tensor::random(
            [self.n_latents, d],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );

        // Read cross-attention
        let read_q = LinearConfig::new(d, d).init(device);
        let read_k = LinearConfig::new(d, d).init(device);
        let read_v = LinearConfig::new(d, d).init(device);
        let read_out = LinearConfig::new(d, d).init(device);
        let read_norm = LayerNormConfig::new(d).init(device);

        // Latent self-attention blocks
        let latent_blocks = TransformerEncoderConfig::new(d, d * 2, self.n_heads, self.n_latent_blocks)
            .with_norm_first(true)
            .with_dropout(0.0)
            .init(device);

        // Write cross-attention — zero-init output proj for identity passthrough
        let write_q = LinearConfig::new(d, d).init(device);
        let write_k = LinearConfig::new(d, d).init(device);
        let write_v = LinearConfig::new(d, d).init(device);
        let mut write_out = LinearConfig::new(d, d).init(device);
        // Zero-init write output projection: at init, write step is identity
        write_out.weight = Param::from_tensor(Tensor::zeros([d, d], device));
        write_out.bias = write_out.bias.map(|_| Param::from_tensor(Tensor::zeros([d], device)));

        let write_norm = LayerNormConfig::new(d).init(device);

        LatentInterface {
            latents: Param::from_tensor(latents),
            read_q, read_k, read_v, read_out, read_norm,
            latent_blocks,
            write_q, write_k, write_v, write_out, write_norm,
            n_heads: self.n_heads,
            d_model: d,
            n_latents: self.n_latents,
            scale: (head_dim as f32).sqrt().recip(),
        }
    }
}

impl<B: Backend> LatentInterface<B> {
    pub fn n_latents(&self) -> usize {
        self.n_latents
    }

    /// Forward pass through the latent interface.
    ///
    /// entity_tokens: [B, S, d] — output of entity encoder (+ spatial enrichment)
    /// entity_mask: [B, S] — true = valid token
    /// n_latents_override: if Some, use this many latents (for inference budget control)
    ///
    /// Returns: (updated_entity_tokens [B, S, d], pooled_latents [B, d])
    pub fn forward(
        &self,
        entity_tokens: Tensor<B, 3>,
        entity_mask: Tensor<B, 2, Bool>,
        n_latents_override: Option<usize>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, seq_len, _] = entity_tokens.dims();
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;

        // Determine number of latents to use
        let n_lat = n_latents_override.unwrap_or(self.n_latents);
        let n_lat = n_lat.min(self.n_latents);

        // Expand latents: [K, d] -> [B, n_lat, d]
        let latents = self.latents.val()
            .slice([0..n_lat])
            .unsqueeze_dim::<3>(0)
            .expand([batch, n_lat, d]);

        // --- Read step: latents attend to entity tokens ---
        let read_q: Tensor<B, 4> = self.read_q.forward(latents.clone())
            .reshape([batch, n_lat, h, dh]).swap_dims(1, 2); // [B, H, K, dh]
        let read_k: Tensor<B, 4> = self.read_k.forward(entity_tokens.clone())
            .reshape([batch, seq_len, h, dh]).swap_dims(1, 2); // [B, H, S, dh]
        let read_v: Tensor<B, 4> = self.read_v.forward(entity_tokens.clone())
            .reshape([batch, seq_len, h, dh]).swap_dims(1, 2); // [B, H, S, dh]

        let read_scores = read_q.matmul(read_k.swap_dims(2, 3)) * self.scale; // [B, H, K, S]

        // Mask padded entity tokens
        let pad_mask = entity_mask.clone().bool_not()
            .reshape([batch, 1, 1, seq_len])
            .expand([batch, h, n_lat, seq_len]);
        let read_scores = read_scores.mask_fill(pad_mask, -1e9);

        let read_weights = softmax(read_scores, 3);
        let read_out = read_weights.matmul(read_v); // [B, H, K, dh]
        let read_out: Tensor<B, 3> = read_out.swap_dims(1, 2).reshape([batch, n_lat, d]);
        let read_out = self.read_out.forward(read_out);

        // Residual + norm
        let latents = self.read_norm.forward(latents + read_out);

        // --- Latent self-attention blocks ---
        let input = TransformerEncoderInput::new(latents);
        let latents = self.latent_blocks.forward(input);

        // --- Write step: entity tokens attend to latents ---
        let write_q: Tensor<B, 4> = self.write_q.forward(entity_tokens.clone())
            .reshape([batch, seq_len, h, dh]).swap_dims(1, 2); // [B, H, S, dh]
        let write_k: Tensor<B, 4> = self.write_k.forward(latents.clone())
            .reshape([batch, n_lat, h, dh]).swap_dims(1, 2); // [B, H, K, dh]
        let write_v: Tensor<B, 4> = self.write_v.forward(latents.clone())
            .reshape([batch, n_lat, h, dh]).swap_dims(1, 2); // [B, H, K, dh]

        let write_scores = write_q.matmul(write_k.swap_dims(2, 3)) * self.scale; // [B, H, S, K]
        // No masking needed — all latents are valid
        let write_weights = softmax(write_scores, 3);
        let write_out = write_weights.matmul(write_v); // [B, H, S, dh]
        let write_out: Tensor<B, 3> = write_out.swap_dims(1, 2).reshape([batch, seq_len, d]);
        let write_out = self.write_out.forward(write_out);

        // Residual + norm (zero-init write_out means this starts as identity)
        let entity_tokens = self.write_norm.forward(entity_tokens + write_out);

        // Pool latents for temporal cell input
        let pooled: Tensor<B, 2> = latents.mean_dim(1).squeeze_dim::<2>(1); // [B, d]

        (entity_tokens, pooled)
    }
}
