//! Combat pointer head: combat type classification + pointer attention for targeting.

use burn::module::Module;
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use super::config::*;

#[derive(Module, Debug)]
pub struct CombatPointerHead<B: Backend> {
    /// Combat type classifier: pooled -> d_model -> NUM_COMBAT_TYPES
    type_l1: Linear<B>,
    type_l2: Linear<B>,
    /// Pointer key projection: token -> d_model
    pointer_key: Linear<B>,
    /// Attack query: pooled -> d_model
    attack_query: Linear<B>,
    /// Per-ability queries: ability cross-emb -> d_model
    ability_queries: Vec<Linear<B>>,
    gelu: Gelu,
    d_model: usize,
    scale: f32,
}

#[derive(Config, Debug)]
pub struct CombatPointerHeadConfig {
    pub d_model: usize,
}

impl CombatPointerHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CombatPointerHead<B> {
        let d = self.d_model;
        CombatPointerHead {
            type_l1: LinearConfig::new(d, d).init(device),
            type_l2: LinearConfig::new(d, NUM_COMBAT_TYPES).init(device),
            pointer_key: LinearConfig::new(d, d).init(device),
            attack_query: LinearConfig::new(d, d).init(device),
            ability_queries: (0..MAX_ABILITIES)
                .map(|_| LinearConfig::new(d, d).init(device))
                .collect(),
            gelu: Gelu::new(),
            d_model: d,
            scale: (d as f32).sqrt().recip(),
        }
    }
}

/// Output from combat pointer head.
pub struct CombatOutput<B: Backend> {
    /// Combat type logits: [B, NUM_COMBAT_TYPES]
    pub combat_logits: Tensor<B, 2>,
    /// Attack pointer logits: [B, S] (masked to enemies)
    pub attack_ptr: Tensor<B, 2>,
    /// Per-ability pointer logits: Vec of [B, S] (one per ability, None if not available)
    pub ability_ptrs: Vec<Option<Tensor<B, 2>>>,
}

impl<B: Backend> CombatPointerHead<B> {
    /// Compute combat type + pointer logits.
    ///
    /// pooled: [B, d] mean-pooled entity state
    /// tokens: [B, S, d] all encoder tokens
    /// mask: [B, S] true = valid
    /// type_ids: [B, S] entity type per token (0=self, 1=enemy, 2=ally, 3=zone, 4=agg)
    /// ability_cross_embs: per-ability cross-attended embeddings [B, d] or None
    pub fn forward(
        &self,
        pooled: Tensor<B, 2>,
        tokens: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        type_ids: Tensor<B, 2, Int>,
        ability_cross_embs: &[Option<Tensor<B, 2>>],
    ) -> CombatOutput<B> {
        let [batch, seq_len, _] = tokens.dims();
        let d = self.d_model;

        // Combat type logits
        let h = self.gelu.forward(self.type_l1.forward(pooled.clone()));
        let combat_logits = self.type_l2.forward(h);

        // Pointer keys for all tokens
        let keys = self.pointer_key.forward(tokens); // [B, S, d]

        // Attack query from pooled state
        let atk_q = self.attack_query.forward(pooled.clone()); // [B, d]
        let atk_q: Tensor<B, 3> = atk_q.unsqueeze_dim::<3>(1); // [B, 1, d]
        let mut attack_ptr = atk_q.matmul(keys.clone().swap_dims(1, 2)).squeeze_dim::<2>(1) * self.scale; // [B, S]

        // Mask: only enemies (type_id == 1) are valid attack targets
        let enemy_mask = type_ids.clone().equal_elem(1).bool_and(mask.clone());
        let atk_pad = enemy_mask.bool_not();
        attack_ptr = attack_ptr.mask_fill(atk_pad, -1e9);

        // Ability pointers
        let mut ability_ptrs = Vec::with_capacity(MAX_ABILITIES);
        for (i, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if i < self.ability_queries.len() {
                    let ab_q: Tensor<B, 3> = self.ability_queries[i].forward(cross_emb.clone()).unsqueeze_dim::<3>(1); // [B, 1, d]
                    let mut ab_ptr = ab_q.matmul(keys.clone().swap_dims(1, 2)).squeeze_dim::<2>(1) * self.scale; // [B, S]
                    // Abilities can target any valid token
                    let pad = mask.clone().bool_not();
                    ab_ptr = ab_ptr.mask_fill(pad, -1e9);
                    ability_ptrs.push(Some(ab_ptr));
                } else {
                    ability_ptrs.push(None);
                }
            } else {
                ability_ptrs.push(None);
            }
        }

        CombatOutput { combat_logits, attack_ptr, ability_ptrs }
    }
}
