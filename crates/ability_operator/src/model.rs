//! Ability Latent Operator model architecture.
//!
//! Two-component system:
//! 1. StateEncoder — maps raw sim state (entities + threats + positions + ability slots)
//!    into latent entity tokens via a shared transformer encoder.
//! 2. AbilityOperator — transforms the encoded state according to a specific ability cast,
//!    predicting the latent state after the ability resolves.
//!
//! All at d_model=64, n_heads=8, d_ff=128.

use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::prelude::*;

// ── Constants ──────────────────────────────────────────────────────────────

/// Entity feature dimension (23 dims after removing collapsed ability scalars).
pub const ENTITY_DIM: usize = 23;
/// Threat feature dimension.
pub const THREAT_DIM: usize = 8;
/// Position feature dimension.
pub const POSITION_DIM: usize = 8;
/// Ability slot token dimension: 128 (frozen CLS) + 1 (is_ready) + 1 (cooldown_fraction) = 130.
pub const ABILITY_SLOT_DIM: usize = 130;
/// Frozen ability [CLS] embedding dimension.
pub const ABILITY_CLS_DIM: usize = 128;

/// Max entity slots: self(1) + enemies(3) + allies(3) = 7.
pub const MAX_ENTITIES: usize = 7;
/// Max threat slots.
pub const MAX_THREATS: usize = 8;
/// Max position slots.
pub const MAX_POSITIONS: usize = 8;
/// Max abilities per entity.
pub const MAX_ABILITIES_PER_ENTITY: usize = 8;
/// Max total ability tokens: 7 entities × 8 abilities.
pub const MAX_ABILITY_TOKENS: usize = MAX_ENTITIES * MAX_ABILITIES_PER_ENTITY;

/// Token type IDs:
/// 0 = self entity, 1 = enemy entity, 2 = ally entity,
/// 3 = threat, 4 = position,
/// 5 = self ability, 6 = enemy ability, 7 = ally ability.
pub const NUM_TOKEN_TYPES: usize = 8;

/// Max sequence length: 7 entities + 8 threats + 8 positions + 56 abilities = 79.
pub const MAX_SEQ_LEN: usize = MAX_ENTITIES + MAX_THREATS + MAX_POSITIONS + MAX_ABILITY_TOKENS;

/// Maximum ability effect window in milliseconds.
pub const MAX_WINDOW_MS: f32 = 6000.0;

// ── Default hyperparameters ────────────────────────────────────────────────

pub const D_MODEL: usize = 64;
pub const N_HEADS: usize = 8;
pub const D_FF: usize = 128;
pub const ENCODER_LAYERS: usize = 4;
pub const OPERATOR_LAYERS: usize = 2;

// ── StateEncoder ───────────────────────────────────────────────────────────

/// Encodes raw sim state into latent entity tokens.
///
/// Processes a heterogeneous token sequence (entities, threats, positions,
/// ability slots) through a shared transformer encoder. Outputs the first
/// 7 tokens (entity slots) as the latent state representation.
#[derive(Module, Debug)]
pub struct StateEncoder<B: Backend> {
    entity_proj: Linear<B>,
    threat_proj: Linear<B>,
    position_proj: Linear<B>,
    ability_proj: Linear<B>,
    type_emb: Embedding<B>,
    input_norm: LayerNorm<B>,
    encoder: TransformerEncoder<B>,
    out_norm: LayerNorm<B>,
}

impl<B: Backend> StateEncoder<B> {
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, device: &B::Device) -> Self {
        StateEncoder {
            entity_proj: LinearConfig::new(ENTITY_DIM, d_model).init(device),
            threat_proj: LinearConfig::new(THREAT_DIM, d_model).init(device),
            position_proj: LinearConfig::new(POSITION_DIM, d_model).init(device),
            ability_proj: LinearConfig::new(ABILITY_SLOT_DIM, d_model).init(device),
            type_emb: EmbeddingConfig::new(NUM_TOKEN_TYPES, d_model).init(device),
            input_norm: LayerNormConfig::new(d_model).init(device),
            encoder: TransformerEncoderConfig::new(d_model, D_FF, n_heads, n_layers)
                .with_norm_first(true)
                .with_dropout(0.0)
                .init(device),
            out_norm: LayerNormConfig::new(d_model).init(device),
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `entity_features` — (B, E, 23) entity features
    /// * `entity_types` — (B, E) int type IDs (0=self, 1=enemy, 2=ally)
    /// * `entity_mask` — (B, E) bool, True = padding
    /// * `threat_features` — (B, T, 8)
    /// * `threat_mask` — (B, T) bool
    /// * `position_features` — (B, P, 8)
    /// * `position_mask` — (B, P) bool
    /// * `ability_features` — (B, A, 34) ability slot tokens
    /// * `ability_types` — (B, A) int type IDs (5=self, 6=enemy, 7=ally)
    /// * `ability_mask` — (B, A) bool
    ///
    /// # Returns
    /// Entity tokens (B, E, d_model) after transformer encoding.
    pub fn forward(
        &self,
        entity_features: Tensor<B, 3>,
        entity_types: Tensor<B, 2, Int>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_features: Tensor<B, 3>,
        threat_mask: Tensor<B, 2, Bool>,
        position_features: Tensor<B, 3>,
        position_mask: Tensor<B, 2, Bool>,
        ability_features: Tensor<B, 3>,
        ability_types: Tensor<B, 2, Int>,
        ability_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, n_ent, _] = entity_features.dims();
        let [_, n_thr, _] = threat_features.dims();
        let [_, n_pos, _] = position_features.dims();
        let [_, n_abl, _] = ability_features.dims();
        let device = entity_features.device();

        // Project each token type to d_model
        let ent_tokens = self.entity_proj.forward(entity_features);
        let ent_type_embs = self.type_emb.forward(entity_types);
        let ent_tokens = ent_tokens + ent_type_embs;

        let thr_tokens = self.threat_proj.forward(threat_features);
        let thr_type_id = Tensor::<B, 2, Int>::full([batch, n_thr], 3, &device);
        let thr_type_embs = self.type_emb.forward(thr_type_id);
        let thr_tokens = thr_tokens + thr_type_embs;

        let pos_tokens = self.position_proj.forward(position_features);
        let pos_type_id = Tensor::<B, 2, Int>::full([batch, n_pos], 4, &device);
        let pos_type_embs = self.type_emb.forward(pos_type_id);
        let pos_tokens = pos_tokens + pos_type_embs;

        let abl_tokens = self.ability_proj.forward(ability_features);
        let abl_type_embs = self.type_emb.forward(ability_types);
        let abl_tokens = abl_tokens + abl_type_embs;

        // Concatenate all tokens: (B, E+T+P+A, d_model)
        let all_tokens = Tensor::cat(
            vec![ent_tokens, thr_tokens, pos_tokens, abl_tokens],
            1,
        );
        let all_tokens = self.input_norm.forward(all_tokens);

        // Concatenate masks
        let full_mask = Tensor::cat(
            vec![entity_mask, threat_mask, position_mask, ability_mask],
            1,
        );

        // Transformer encoder
        let input = TransformerEncoderInput::new(all_tokens).mask_pad(full_mask);
        let encoded = self.encoder.forward(input);
        let encoded = self.out_norm.forward(encoded);

        // Extract entity tokens [0:E]
        let seq_len = encoded.dims()[1];
        let d = encoded.dims()[2];
        encoded.slice([0..batch, 0..n_ent, 0..d])
    }
}

// ── AbilityOperator ────────────────────────────────────────────────────────

/// Transforms encoded entity state according to a specific ability cast.
///
/// The ability is injected as an additional token appended to the entity
/// sequence. The transformer learns asymmetric attention patterns: entities
/// inside an AoE radius attend strongly to the ability token while
/// out-of-range entities learn to ignore it.
#[derive(Module, Debug)]
pub struct AbilityOperator<B: Backend> {
    /// Projects frozen 32d ability CLS to d_model.
    ability_cls_proj: Linear<B>,
    /// Embedding for which entity slot is casting (0-6).
    caster_slot_emb: Embedding<B>,
    /// Pre-computed sinusoidal duration encoding projected to d_model.
    duration_proj: Linear<B>,
    /// LayerNorm on the constructed ability token.
    ability_norm: LayerNorm<B>,
    /// Transformer encoder for the operator.
    encoder: TransformerEncoder<B>,
    out_norm: LayerNorm<B>,
}

/// Number of sinusoidal frequency bands for duration encoding.
const DURATION_FREQS: usize = 16;

impl<B: Backend> AbilityOperator<B> {
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, device: &B::Device) -> Self {
        AbilityOperator {
            ability_cls_proj: LinearConfig::new(ABILITY_CLS_DIM, d_model).init(device),
            caster_slot_emb: EmbeddingConfig::new(MAX_ENTITIES, d_model).init(device),
            duration_proj: LinearConfig::new(DURATION_FREQS * 2, d_model).init(device),
            ability_norm: LayerNormConfig::new(d_model).init(device),
            encoder: TransformerEncoderConfig::new(d_model, D_FF, n_heads, n_layers)
                .with_norm_first(true)
                .with_dropout(0.0)
                .init(device),
            out_norm: LayerNormConfig::new(d_model).init(device),
        }
    }

    /// Compute sinusoidal duration encoding.
    ///
    /// duration_norm: (B,) — window_ms / MAX_WINDOW_MS in [0, 1].
    /// Returns: (B, DURATION_FREQS * 2) — sin/cos at exponentially spaced frequencies.
    fn duration_encoding<const D: usize>(
        &self,
        duration_norm: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let [batch] = duration_norm.dims();
        let device = duration_norm.device();

        // Frequencies: 2^(0..DURATION_FREQS) * pi
        let mut freq_data = [0.0f32; DURATION_FREQS];
        for i in 0..DURATION_FREQS {
            freq_data[i] = (2.0f32).powi(i as i32) * std::f32::consts::PI;
        }
        let freqs = Tensor::<B, 1>::from_data(
            TensorData::from(freq_data.as_slice()),
            &device,
        ); // (F,)

        // (B, 1) * (1, F) → (B, F)
        let dur = duration_norm.unsqueeze_dim(1); // (B, 1)
        let freqs = freqs.unsqueeze_dim(0); // (1, F)
        let angles = dur * freqs; // (B, F)

        // Concat sin and cos: (B, 2F)
        let sin_part = angles.clone().sin();
        let cos_part = angles.cos();
        Tensor::cat(vec![sin_part, cos_part], 1)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `z_before` — (B, 7, d_model) encoded entity tokens from StateEncoder
    /// * `ability_cls` — (B, 32) frozen ability CLS embedding
    /// * `caster_slot` — (B,) int — which entity slot [0-6] is casting
    /// * `duration_norm` — (B,) float — window_ms / MAX_WINDOW_MS
    /// * `entity_mask` — (B, 7) bool — True = padding (dead/absent entities)
    ///
    /// # Returns
    /// z_after: (B, 7, d_model) — entity tokens after ability effect.
    pub fn forward(
        &self,
        z_before: Tensor<B, 3>,
        ability_cls: Tensor<B, 2>,
        caster_slot: Tensor<B, 1, Int>,
        duration_norm: Tensor<B, 1>,
        entity_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, n_ent, d_model] = z_before.dims();
        let device = z_before.device();

        // Construct ability token: CLS proj + caster slot emb + duration emb
        let cls_proj = self.ability_cls_proj.forward(ability_cls); // (B, d)
        let slot_emb = self.caster_slot_emb.forward(caster_slot.unsqueeze_dim(1)); // (B, 1, d)
        let slot_emb: Tensor<B, 2> = slot_emb.reshape([batch, d_model]); // (B, d)
        let dur_enc = self.duration_encoding::<0>(duration_norm); // (B, 2F)
        let dur_proj = self.duration_proj.forward(dur_enc); // (B, d)

        let ability_token = cls_proj + slot_emb + dur_proj; // (B, d)
        let ability_token = self.ability_norm.forward(ability_token.unsqueeze_dim(1)); // (B, 1, d)

        // Append ability token to entity sequence: (B, E+1, d)
        let operator_input = Tensor::cat(vec![z_before, ability_token], 1);

        // Extend mask: ability token is never padded → False
        let abl_mask = Tensor::<B, 2, Bool>::full([batch, 1], false, &device);
        let full_mask = Tensor::cat(vec![entity_mask, abl_mask], 1);

        // Transformer encoder
        let input = TransformerEncoderInput::new(operator_input).mask_pad(full_mask);
        let encoded = self.encoder.forward(input);
        let encoded = self.out_norm.forward(encoded);

        // Extract entity tokens [0:E], discard ability token
        encoded.slice([0..batch, 0..n_ent, 0..d_model])
    }
}

// ── DecoderHeads ───────────────────────────────────────────────────────────

/// Prediction head outputting (mean, log_var) for beta-NLL loss.
#[derive(Module, Debug)]
pub struct GaussianHead<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    n_features: usize,
}

impl<B: Backend> GaussianHead<B> {
    pub fn new(d_in: usize, n_features: usize, d_hidden: usize, device: &B::Device) -> Self {
        GaussianHead {
            w1: LinearConfig::new(d_in, d_hidden).init(device),
            w2: LinearConfig::new(d_hidden, n_features * 2).init(device),
            n_features,
        }
    }

    /// Returns (mean, log_var), each (B, E, n_features).
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, n_ent, _] = x.dims();
        let h = Gelu.forward(self.w1.forward(x));
        let out = self.w2.forward(h);

        let mean = out.clone().slice([0..batch, 0..n_ent, 0..self.n_features]);
        let log_var = out
            .slice([0..batch, 0..n_ent, self.n_features..self.n_features * 2])
            .clamp(-10.0, 10.0);
        (mean, log_var)
    }
}

/// Binary prediction head outputting logits for BCE loss.
#[derive(Module, Debug)]
pub struct BinaryHead<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
}

impl<B: Backend> BinaryHead<B> {
    pub fn new(d_in: usize, n_features: usize, d_hidden: usize, device: &B::Device) -> Self {
        BinaryHead {
            w1: LinearConfig::new(d_in, d_hidden).init(device),
            w2: LinearConfig::new(d_hidden, n_features).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = Gelu.forward(self.w1.forward(x));
        self.w2.forward(h)
    }
}

/// Decoder heads that predict feature-group deltas from operator output.
#[derive(Module, Debug)]
pub struct DecoderHeads<B: Backend> {
    /// hp_pct, shield_pct, resource_pct: 3 means + 3 log_vars.
    pub hp_head: GaussianHead<B>,
    /// cc_remaining_norm, is_stunned: Gaussian + BCE.
    pub cc_head: GaussianHead<B>,
    /// is_stunned logit (BCE).
    pub cc_stun_head: BinaryHead<B>,
    /// delta_x, delta_y: 2 means + 2 log_vars.
    pub pos_head: GaussianHead<B>,
    /// Death logit (BCE).
    pub exists_head: BinaryHead<B>,
}

impl<B: Backend> DecoderHeads<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        let d_hidden = d_model * 2; // 128

        DecoderHeads {
            hp_head: GaussianHead::new(d_model, 3, d_hidden, device),
            cc_head: GaussianHead::new(d_model, 1, d_hidden, device),
            cc_stun_head: BinaryHead::new(d_model, 1, d_hidden, device),
            pos_head: GaussianHead::new(d_model, 2, d_hidden, device),
            exists_head: BinaryHead::new(d_model, 1, d_model, device),
        }
    }

    /// Forward pass on entity tokens from the operator.
    ///
    /// z_after: (B, 7, d_model)
    /// Returns DecoderOutput with per-group predictions.
    pub fn forward(&self, z_after: Tensor<B, 3>) -> DecoderOutput<B> {
        let (hp_mean, hp_logvar) = self.hp_head.forward(z_after.clone());
        let (cc_mean, cc_logvar) = self.cc_head.forward(z_after.clone());
        let cc_stun_logits = self.cc_stun_head.forward(z_after.clone());
        let (pos_mean, pos_logvar) = self.pos_head.forward(z_after.clone());
        let exists_logits = self.exists_head.forward(z_after);

        DecoderOutput {
            hp_mean,
            hp_logvar,
            cc_mean,
            cc_logvar,
            cc_stun_logits,
            pos_mean,
            pos_logvar,
            exists_logits,
        }
    }
}

/// Output from decoder heads.
pub struct DecoderOutput<B: Backend> {
    /// (B, E, 3) — delta hp_pct, shield_pct, resource_pct means.
    pub hp_mean: Tensor<B, 3>,
    /// (B, E, 3) — log variance for hp group.
    pub hp_logvar: Tensor<B, 3>,
    /// (B, E, 1) — cc_remaining_norm mean.
    pub cc_mean: Tensor<B, 3>,
    /// (B, E, 1) — cc log variance.
    pub cc_logvar: Tensor<B, 3>,
    /// (B, E, 1) — is_stunned logit.
    pub cc_stun_logits: Tensor<B, 3>,
    /// (B, E, 2) — delta_x, delta_y means.
    pub pos_mean: Tensor<B, 3>,
    /// (B, E, 2) — position log variance.
    pub pos_logvar: Tensor<B, 3>,
    /// (B, E, 1) — death probability logit.
    pub exists_logits: Tensor<B, 3>,
}

// ── Full Model ─────────────────────────────────────────────────────────────

/// Complete Ability Latent Operator model.
///
/// Combines StateEncoder + AbilityOperator + DecoderHeads.
#[derive(Module, Debug)]
pub struct AbilityLatentOperator<B: Backend> {
    pub encoder: StateEncoder<B>,
    pub operator: AbilityOperator<B>,
    pub decoder: DecoderHeads<B>,
}

impl<B: Backend> AbilityLatentOperator<B> {
    pub fn new(device: &B::Device) -> Self {
        AbilityLatentOperator {
            encoder: StateEncoder::new(D_MODEL, N_HEADS, ENCODER_LAYERS, device),
            operator: AbilityOperator::new(D_MODEL, N_HEADS, OPERATOR_LAYERS, device),
            decoder: DecoderHeads::new(D_MODEL, device),
        }
    }

    /// Full forward pass: encode → operate → decode.
    ///
    /// Returns DecoderOutput with predicted feature-group deltas.
    pub fn forward(
        &self,
        // Encoder inputs
        entity_features: Tensor<B, 3>,
        entity_types: Tensor<B, 2, Int>,
        entity_mask: Tensor<B, 2, Bool>,
        threat_features: Tensor<B, 3>,
        threat_mask: Tensor<B, 2, Bool>,
        position_features: Tensor<B, 3>,
        position_mask: Tensor<B, 2, Bool>,
        ability_features: Tensor<B, 3>,
        ability_types: Tensor<B, 2, Int>,
        ability_mask: Tensor<B, 2, Bool>,
        // Operator inputs
        ability_cls: Tensor<B, 2>,
        caster_slot: Tensor<B, 1, Int>,
        duration_norm: Tensor<B, 1>,
    ) -> DecoderOutput<B> {
        // Encode state
        let z_before = self.encoder.forward(
            entity_features,
            entity_types,
            entity_mask.clone(),
            threat_features,
            threat_mask,
            position_features,
            position_mask,
            ability_features,
            ability_types,
            ability_mask,
        );

        // Apply ability operator
        let z_after = self.operator.forward(
            z_before,
            ability_cls,
            caster_slot,
            duration_norm,
            entity_mask,
        );

        // Decode predictions
        self.decoder.forward(z_after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_forward_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let model = AbilityLatentOperator::<TestBackend>::new(&device);

        let batch = 2;
        let n_ent = MAX_ENTITIES;
        let n_thr = MAX_THREATS;
        let n_pos = MAX_POSITIONS;
        let n_abl = 16; // 2 entities × 8 abilities

        let entity_features = Tensor::zeros([batch, n_ent, ENTITY_DIM], &device);
        let entity_types = Tensor::<TestBackend, 2, Int>::zeros([batch, n_ent], &device);
        let entity_mask = Tensor::<TestBackend, 2, Bool>::full([batch, n_ent], false, &device);

        let threat_features = Tensor::zeros([batch, n_thr, THREAT_DIM], &device);
        let threat_mask = Tensor::<TestBackend, 2, Bool>::full([batch, n_thr], false, &device);

        let position_features = Tensor::zeros([batch, n_pos, POSITION_DIM], &device);
        let position_mask = Tensor::<TestBackend, 2, Bool>::full([batch, n_pos], false, &device);

        let ability_features = Tensor::zeros([batch, n_abl, ABILITY_SLOT_DIM], &device);
        let ability_types = Tensor::<TestBackend, 2, Int>::full([batch, n_abl], 5, &device);
        let ability_mask = Tensor::<TestBackend, 2, Bool>::full([batch, n_abl], false, &device);

        let ability_cls = Tensor::zeros([batch, ABILITY_CLS_DIM], &device);
        let caster_slot = Tensor::<TestBackend, 1, Int>::zeros([batch], &device);
        let duration_norm = Tensor::full([batch], 0.5, &device);

        let output = model.forward(
            entity_features,
            entity_types,
            entity_mask,
            threat_features,
            threat_mask,
            position_features,
            position_mask,
            ability_features,
            ability_types,
            ability_mask,
            ability_cls,
            caster_slot,
            duration_norm,
        );

        assert_eq!(output.hp_mean.dims(), [batch, n_ent, 3]);
        assert_eq!(output.hp_logvar.dims(), [batch, n_ent, 3]);
        assert_eq!(output.cc_mean.dims(), [batch, n_ent, 1]);
        assert_eq!(output.cc_stun_logits.dims(), [batch, n_ent, 1]);
        assert_eq!(output.pos_mean.dims(), [batch, n_ent, 2]);
        assert_eq!(output.pos_logvar.dims(), [batch, n_ent, 2]);
        assert_eq!(output.exists_logits.dims(), [batch, n_ent, 1]);
    }
}
