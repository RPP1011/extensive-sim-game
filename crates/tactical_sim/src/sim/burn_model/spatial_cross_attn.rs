//! Spatial Cross-Attention for V6.
//!
//! Entity tokens attend to visible corner tokens via dedicated cross-attention.
//! Corner tokens appear only as K/V — not part of the main token sequence.
//!
//! Zero-initialized output projection: at init, this is a pure identity
//! passthrough via the residual connection. Spatial reasoning trains from
//! zero without regressing initial performance.

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::config::*;

/// Corner token feature dimension (11 floats per corner).
pub const CORNER_DIM: usize = 11;
/// Maximum corners per acting unit.
pub const MAX_CORNERS: usize = 8;

#[derive(Module, Debug)]
pub struct SpatialCrossAttention<B: Backend> {
    corner_proj: Linear<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    norm: LayerNorm<B>,
    n_heads: usize,
    d_model: usize,
    scale: f32,
}

#[derive(Config, Debug)]
pub struct SpatialCrossAttentionConfig {
    #[config(default = "D_MODEL")]
    pub d_model: usize,
    #[config(default = "N_HEADS")]
    pub n_heads: usize,
}

impl SpatialCrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpatialCrossAttention<B> {
        let d = self.d_model;
        let head_dim = d / self.n_heads;

        let corner_proj = LinearConfig::new(CORNER_DIM, d).init(device);
        let q_proj = LinearConfig::new(d, d).init(device);
        let k_proj = LinearConfig::new(d, d).init(device);
        let v_proj = LinearConfig::new(d, d).init(device);

        // Zero-init output projection for identity passthrough at init
        let mut out_proj = LinearConfig::new(d, d).init(device);
        out_proj.weight = Param::from_tensor(Tensor::zeros([d, d], device));
        out_proj.bias = out_proj.bias.map(|_| Param::from_tensor(Tensor::zeros([d], device)));

        let norm = LayerNormConfig::new(d).init(device);

        SpatialCrossAttention {
            corner_proj, q_proj, k_proj, v_proj, out_proj, norm,
            n_heads: self.n_heads,
            d_model: d,
            scale: (head_dim as f32).sqrt().recip(),
        }
    }
}

impl<B: Backend> SpatialCrossAttention<B> {
    /// Cross-attend entity tokens to visible corner tokens.
    ///
    /// entity_tokens: [B, S, d] — main token sequence from entity encoder
    /// corner_tokens: [B, C, 11] — visible corners for acting unit (zero-padded)
    /// corner_mask: [B, C] — true = valid corner, false = padding
    ///
    /// Returns: spatially-enriched entity tokens [B, S, d]
    pub fn forward(
        &self,
        entity_tokens: Tensor<B, 3>,
        corner_tokens: Tensor<B, 3>,
        corner_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = entity_tokens.dims();
        let [_, n_corners, _] = corner_tokens.dims();
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;

        // Project corners to d_model
        let corners = self.corner_proj.forward(corner_tokens); // [B, C, d]

        // Q from entity tokens, K/V from corners
        let q: Tensor<B, 4> = self.q_proj.forward(entity_tokens.clone())
            .reshape([batch, seq_len, h, dh]).swap_dims(1, 2); // [B, H, S, dh]
        let k: Tensor<B, 4> = self.k_proj.forward(corners.clone())
            .reshape([batch, n_corners, h, dh]).swap_dims(1, 2); // [B, H, C, dh]
        let v: Tensor<B, 4> = self.v_proj.forward(corners)
            .reshape([batch, n_corners, h, dh]).swap_dims(1, 2); // [B, H, C, dh]

        // Scaled dot-product attention: [B, H, S, C]
        let attn_scores = q.matmul(k.swap_dims(2, 3)) * self.scale;

        // Mask padded corners
        let pad_mask = corner_mask.bool_not()
            .reshape([batch, 1, 1, n_corners])
            .expand([batch, h, seq_len, n_corners]);
        let attn_scores = attn_scores.mask_fill(pad_mask, -1e9);

        let attn_weights = softmax(attn_scores, 3);
        let attn_out = attn_weights.matmul(v); // [B, H, S, dh]
        let attn_out: Tensor<B, 3> = attn_out.swap_dims(1, 2).reshape([batch, seq_len, d]);
        let attn_out = self.out_proj.forward(attn_out);

        // Residual + norm (zero-init out_proj means this starts as identity)
        self.norm.forward(entity_tokens + attn_out)
    }
}
