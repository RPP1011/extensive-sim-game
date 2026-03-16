//! Cross-attention block for ability CLS → entity token attention.
//!
//! Pre-norm, multi-head, with feedforward residual.

use burn::module::Module;
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    norm_q: LayerNorm<B>,
    norm_kv: LayerNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    ff_l1: Linear<B>,
    ff_l2: Linear<B>,
    norm_ff: LayerNorm<B>,
    gelu: Gelu,
    n_heads: usize,
    d_model: usize,
    scale: f32,
}

#[derive(Config, Debug)]
pub struct CrossAttentionConfig {
    pub d_model: usize,
    #[config(default = "8")]
    pub n_heads: usize,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrossAttentionBlock<B> {
        let d = self.d_model;
        let head_dim = d / self.n_heads;
        CrossAttentionBlock {
            norm_q: LayerNormConfig::new(d).init(device),
            norm_kv: LayerNormConfig::new(d).init(device),
            q_proj: LinearConfig::new(d, d).init(device),
            k_proj: LinearConfig::new(d, d).init(device),
            v_proj: LinearConfig::new(d, d).init(device),
            out_proj: LinearConfig::new(d, d).init(device),
            ff_l1: LinearConfig::new(d, d * 2).init(device),
            ff_l2: LinearConfig::new(d * 2, d).init(device),
            norm_ff: LayerNormConfig::new(d).init(device),
            gelu: Gelu::new(),
            n_heads: self.n_heads,
            d_model: d,
            scale: (head_dim as f32).sqrt().recip(),
        }
    }
}

impl<B: Backend> CrossAttentionBlock<B> {
    /// Cross-attend query (CLS embedding) to key-value (entity tokens).
    ///
    /// query: [B, d_model] (ability CLS)
    /// kv_tokens: [B, S, d_model] (entity tokens)
    /// kv_mask: [B, S] (true = valid)
    ///
    /// Returns: [B, d_model]
    pub fn forward(
        &self,
        query: Tensor<B, 2>,        // [B, d]
        kv_tokens: Tensor<B, 3>,    // [B, S, d]
        kv_mask: Tensor<B, 2, Bool>, // [B, S]
    ) -> Tensor<B, 2> {
        let [batch, seq_len, _] = kv_tokens.dims();
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;

        // Pre-norm
        let q: Tensor<B, 3> = self.norm_q.forward(query.clone().unsqueeze_dim::<3>(1));
        let kv: Tensor<B, 3> = self.norm_kv.forward(kv_tokens);

        // Project Q, K, V -> reshape to [B, H, S, dh]
        let q: Tensor<B, 4> = self.q_proj.forward(q).reshape([batch, 1, h, dh]);
        let q = q.swap_dims(1, 2); // [B, H, 1, dh]
        let k: Tensor<B, 4> = self.k_proj.forward(kv.clone()).reshape([batch, seq_len, h, dh]);
        let k = k.swap_dims(1, 2); // [B, H, S, dh]
        let v: Tensor<B, 4> = self.v_proj.forward(kv).reshape([batch, seq_len, h, dh]);
        let v = v.swap_dims(1, 2); // [B, H, S, dh]

        // Scaled dot-product attention
        let attn_scores = q.matmul(k.swap_dims(2, 3)) * self.scale; // [B, H, 1, S]

        // Apply mask (set padded positions to -inf)
        let mask_expanded = kv_mask
            .clone()
            .bool_not() // true = padded
            .reshape([batch, 1, 1, seq_len])
            .expand([batch, h, 1, seq_len]);
        let attn_scores = attn_scores.mask_fill(mask_expanded, -1e9);

        let attn_weights = burn::tensor::activation::softmax(attn_scores, 3);
        let attn_out = attn_weights.matmul(v); // [B, H, 1, dh]
        let attn_out: Tensor<B, 3> = attn_out.swap_dims(1, 2).reshape([batch, 1, d]);
        let attn_out: Tensor<B, 2> = self.out_proj.forward(attn_out).squeeze_dim::<2>(1);

        // Residual + output
        let x = query + attn_out;

        // Feedforward with residual
        let ff_in = self.norm_ff.forward(x.clone());
        let ff_out = self.ff_l2.forward(self.gelu.forward(self.ff_l1.forward(ff_in)));
        x + ff_out
    }
}
