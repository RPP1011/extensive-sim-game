//! V5 actor-critic weights and inference.
//!
//! V5 extends V4 with:
//! - d_model=128, 8 heads (was 32/4)
//! - Entity features: 34-dim (30 base + 4 spatial summary)
//! - Threat features: 10-dim (8 base + kind + LOS)
//! - 6 type embeddings (adds aggregate type=5)
//! - Aggregate token projection (16 → d_model)
//! - No external_cls_proj when CLS dim matches d_model
//! - Latent interface runs on GPU only (not in Rust CPU inference)

use serde::Deserialize;
use super::weights::{
    gelu, sigmoid,
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    FlatCrossAttention, CrossAttnJson,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    AC_MAX_ABILITIES,
    HeadJson, ActorCriticArchJson,
};
use super::weights_actor_critic_v3::EntityStateV3;

// ---------------------------------------------------------------------------
// V5 entity encoder: entities(34) + zones(12) + aggregate(16)
// Also supports legacy threats(10) + positions(8) for backward compat.
// ---------------------------------------------------------------------------

const NUM_ENTITY_TYPES_V5: usize = 5; // self=0, enemy=1, ally=2, zone=3, aggregate=4

#[derive(Debug, Deserialize)]
struct EntityEncoderV5Json {
    entity_proj: LinearWeights,
    #[serde(default)]
    threat_proj: Option<LinearWeights>,
    #[serde(default)]
    position_proj: Option<LinearWeights>,
    #[serde(default)]
    zone_proj: Option<LinearWeights>,
    agg_proj: LinearWeights,
    type_emb: Vec<Vec<f32>>,  // [5 x d_model] (or 6 for legacy)
    input_norm: LayerNormWeights,
    #[serde(default, alias = "self_attn_layers")]
    layers: Vec<TransformerLayerJson>,
    out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
struct FlatEntityEncoderV5 {
    entity_proj: FlatLinear,     // (34 -> d_model)
    threat_proj: Option<FlatLinear>,     // (10 -> d_model) legacy
    position_proj: Option<FlatLinear>,   // (8 -> d_model) legacy
    zone_proj: Option<FlatLinear>,       // (12 -> d_model) unified zones
    agg_proj: FlatLinear,        // (16 -> d_model)
    type_emb: Vec<f32>,          // [5 x d_model] (zone=3, aggregate=4)
    input_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,
    out_norm: FlatLayerNorm,
    d_model: usize,
}

impl FlatEntityEncoderV5 {
    fn from_json(ej: &EntityEncoderV5Json, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let layers: Vec<TransformerLayer> = ej.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            entity_proj: FlatLinear::from_json(&ej.entity_proj),
            threat_proj: ej.threat_proj.as_ref().map(FlatLinear::from_json),
            position_proj: ej.position_proj.as_ref().map(FlatLinear::from_json),
            zone_proj: ej.zone_proj.as_ref().map(FlatLinear::from_json),
            agg_proj: FlatLinear::from_json(&ej.agg_proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode variable-length entities + zones + optional aggregate.
    /// Supports both legacy (threats+positions) and unified (zones) paths.
    /// Returns (tokens, mask, n_total, type_ids).
    fn forward(
        &self,
        entities: &[&[f32]],     // each 34-dim
        entity_types: &[usize],  // 0/1/2 per entity
        threats: &[&[f32]],      // each 10-dim (legacy)
        positions: &[&[f32]],    // each 8-dim (legacy)
        zones: &[&[f32]],        // each 12-dim (unified)
        aggregate: Option<&[f32]>, // 16-dim or None
    ) -> (Vec<f32>, Vec<bool>, usize, Vec<usize>) {
        let d = self.d_model;
        let n_ents = entities.len();
        let use_zones = self.zone_proj.is_some() && !zones.is_empty();
        let n_spatial = if use_zones { zones.len() } else { threats.len() + positions.len() };
        let has_agg = aggregate.is_some();
        let n_total = n_ents + n_spatial + if has_agg { 1 } else { 0 };

        let mut tokens = vec![0.0f32; n_total * d];
        let mut mask = vec![true; n_total];
        let mut full_type_ids = vec![0usize; n_total];

        // Project entities
        for (i, (feats, &type_id)) in entities.iter().zip(entity_types).enumerate() {
            let exists = feats.len() >= 30 && feats[29] > 0.5;
            mask[i] = exists;
            full_type_ids[i] = type_id;

            self.entity_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            let tid = type_id.min(NUM_ENTITY_TYPES_V5 - 1);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[tid * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        if use_zones {
            // Unified zone path (type_id = 3 = zone)
            let zone_proj = self.zone_proj.as_ref().unwrap();
            for (zi, feats) in zones.iter().enumerate() {
                let i = n_ents + zi;
                let exists = feats.len() >= 12 && feats[11] > 0.5;
                mask[i] = exists;
                full_type_ids[i] = 3; // zone

                zone_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
                for j in 0..d {
                    tokens[i * d + j] += self.type_emb[3 * d + j];
                }
                self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
            }
        } else {
            // Legacy path: threats (type_id = 3) + positions (type_id = 4)
            let n_threats = threats.len();
            if let Some(ref threat_proj) = self.threat_proj {
                for (ti, feats) in threats.iter().enumerate() {
                    let i = n_ents + ti;
                    let exists = feats.len() >= 10 && feats[7] > 0.5;
                    mask[i] = exists;
                    full_type_ids[i] = 3;

                    threat_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
                    for j in 0..d {
                        tokens[i * d + j] += self.type_emb[3 * d + j];
                    }
                    self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
                }
            }

            if let Some(ref position_proj) = self.position_proj {
                let type_id_for_pos = if self.type_emb.len() / d > 4 { 4 } else { 3 };
                for (pi, feats) in positions.iter().enumerate() {
                    let i = n_ents + n_threats + pi;
                    mask[i] = true;
                    full_type_ids[i] = type_id_for_pos;

                    position_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
                    if type_id_for_pos * d + d <= self.type_emb.len() {
                        for j in 0..d {
                            tokens[i * d + j] += self.type_emb[type_id_for_pos * d + j];
                        }
                    }
                    self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
                }
            }
        }

        // Project aggregate (type_id = 4 for unified, 5 for legacy)
        if let Some(agg_feats) = aggregate {
            let i = n_ents + n_spatial;
            mask[i] = true;
            let agg_type = if use_zones { 4 } else { 5.min(self.type_emb.len() / d - 1) };
            full_type_ids[i] = agg_type;

            self.agg_proj.forward(agg_feats, &mut tokens[i * d..(i + 1) * d]);
            if agg_type * d + d <= self.type_emb.len() {
                for j in 0..d {
                    tokens[i * d + j] += self.type_emb[agg_type * d + j];
                }
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Self-attention
        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut tokens, n_total, &mask, &mut scratch);
        }

        // Output norm
        for i in 0..n_total {
            self.out_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        (tokens, mask, n_total, full_type_ids)
    }
}

// ---------------------------------------------------------------------------
// Combat pointer head (same as V4 but at d=128)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CombatHeadV5Json {
    combat_type_head: HeadJson,
    pointer_key: LinearWeights,
    attack_query: LinearWeights,
    ability_queries: Vec<LinearWeights>,
}

#[derive(Debug, Clone)]
struct FlatCombatHeadV5 {
    combat_type_l1: FlatLinear,
    combat_type_l2: FlatLinear,
    pointer_key: FlatLinear,
    attack_query: FlatLinear,
    ability_queries: Vec<FlatLinear>,
    d_model: usize,
    scale: f32,
}

impl FlatCombatHeadV5 {
    fn from_json(j: &CombatHeadV5Json, d_model: usize) -> Self {
        Self {
            combat_type_l1: FlatLinear::from_json(&j.combat_type_head.linear1),
            combat_type_l2: FlatLinear::from_json(&j.combat_type_head.linear2),
            pointer_key: FlatLinear::from_json(&j.pointer_key),
            attack_query: FlatLinear::from_json(&j.attack_query),
            ability_queries: j.ability_queries.iter().map(FlatLinear::from_json).collect(),
            d_model,
            scale: (d_model as f32).sqrt().recip(),
        }
    }

    fn forward(
        &self,
        pooled: &[f32],
        tokens: &[f32],
        mask: &[bool],
        n_total: usize,
        type_ids: &[usize],
        ability_cross_embs: &[Option<Vec<f32>>],
    ) -> (Vec<f32>, Vec<f32>, Vec<Option<Vec<f32>>>) {
        let d = self.d_model;

        // Combat type logits
        let mut h = vec![0.0f32; d];
        self.combat_type_l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_types = self.combat_type_l2.out_dim;
        let mut combat_logits = vec![0.0f32; n_types];
        self.combat_type_l2.forward(&h, &mut combat_logits);

        // Compute keys for all tokens
        let mut keys = vec![0.0f32; n_total * d];
        for i in 0..n_total {
            self.pointer_key.forward(
                &tokens[i * d..(i + 1) * d],
                &mut keys[i * d..(i + 1) * d],
            );
        }

        // Attack pointer: query from pooled, masked to enemies only (type_id == 1)
        let mut atk_q = vec![0.0f32; d];
        self.attack_query.forward(pooled, &mut atk_q);
        let mut attack_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] == 1 {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += atk_q[j] * keys[i * d + j];
                }
                attack_ptr[i] = dot * self.scale;
            }
        }

        // Ability pointers
        let mut ability_ptrs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for (ab_idx, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if ab_idx < self.ability_queries.len() {
                    let mut ab_q = vec![0.0f32; d];
                    self.ability_queries[ab_idx].forward(cross_emb, &mut ab_q);
                    let mut ab_ptr = vec![f32::NEG_INFINITY; n_total];
                    for i in 0..n_total {
                        if mask[i] {
                            let mut dot = 0.0f32;
                            for j in 0..d {
                                dot += ab_q[j] * keys[i * d + j];
                            }
                            ab_ptr[i] = dot * self.scale;
                        }
                    }
                    ability_ptrs.push(Some(ab_ptr));
                } else {
                    ability_ptrs.push(None);
                }
            } else {
                ability_ptrs.push(None);
            }
        }

        (combat_logits, attack_ptr, ability_ptrs)
    }
}

// ---------------------------------------------------------------------------
// Move head
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlatMoveHeadV5 {
    l1: FlatLinear,
    l2: FlatLinear,
}

impl FlatMoveHeadV5 {
    fn from_json(j: &HeadJson) -> Self {
        Self {
            l1: FlatLinear::from_json(&j.linear1),
            l2: FlatLinear::from_json(&j.linear2),
        }
    }

    fn forward(&self, pooled: &[f32]) -> Vec<f32> {
        let d = self.l1.out_dim;
        let mut h = vec![0.0f32; d];
        self.l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_out = self.l2.out_dim;
        let mut out = vec![0.0f32; n_out];
        self.l2.forward(&h, &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// CfC temporal cell (replaces GRU)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CfCCellJson {
    f_gate: LinearWeights,
    h_gate: LinearWeights,
    t_a: LinearWeights,
    t_b: LinearWeights,
    proj: LinearWeights,
    h_dim: usize,
}

#[derive(Debug, Clone)]
pub struct FlatCfCCell {
    f_gate: FlatLinear,   // (d_model + h_dim) -> h_dim
    h_gate: FlatLinear,   // (d_model + h_dim) -> h_dim
    t_a: FlatLinear,      // (d_model + h_dim) -> h_dim
    t_b: FlatLinear,      // (d_model + h_dim) -> h_dim
    proj: FlatLinear,     // h_dim -> d_model
    h_dim: usize,
}

impl FlatCfCCell {
    fn from_json(j: &CfCCellJson) -> Self {
        Self {
            f_gate: FlatLinear::from_json(&j.f_gate),
            h_gate: FlatLinear::from_json(&j.h_gate),
            t_a: FlatLinear::from_json(&j.t_a),
            t_b: FlatLinear::from_json(&j.t_b),
            proj: FlatLinear::from_json(&j.proj),
            h_dim: j.h_dim,
        }
    }

    /// CfC forward pass.
    /// x: input (d_model), h_prev: hidden state (h_dim), delta_t: time step scalar.
    /// Returns (output (d_model), h_new (h_dim)).
    pub fn forward(&self, x: &[f32], h_prev: &[f32], delta_t: f32) -> (Vec<f32>, Vec<f32>) {
        let h_dim = self.h_dim;
        let x_dim = x.len();

        // combined = cat([x, h_prev])
        let mut combined = Vec::with_capacity(x_dim + h_dim);
        combined.extend_from_slice(x);
        combined.extend_from_slice(h_prev);

        // f = sigmoid(f_gate(combined))
        let mut f = vec![0.0f32; h_dim];
        self.f_gate.forward(&combined, &mut f);
        for v in f.iter_mut() { *v = sigmoid(*v); }

        // candidate = tanh(h_gate(combined))
        let mut candidate = vec![0.0f32; h_dim];
        self.h_gate.forward(&combined, &mut candidate);
        for v in candidate.iter_mut() { *v = v.tanh(); }

        // t = sigmoid(t_a(combined)) * delta_t + t_b(combined)
        let mut t_a_out = vec![0.0f32; h_dim];
        self.t_a.forward(&combined, &mut t_a_out);
        for v in t_a_out.iter_mut() { *v = sigmoid(*v); }

        let mut t_b_out = vec![0.0f32; h_dim];
        self.t_b.forward(&combined, &mut t_b_out);

        // t = sigmoid(t_a) * delta_t + t_b
        let mut t = vec![0.0f32; h_dim];
        for i in 0..h_dim {
            t[i] = t_a_out[i] * delta_t + t_b_out[i];
        }

        // h_new = tanh(f * h_prev + (1 - f) * candidate * t)
        let mut h_new = vec![0.0f32; h_dim];
        for i in 0..h_dim {
            h_new[i] = (f[i] * h_prev[i] + (1.0 - f[i]) * candidate[i] * t[i]).tanh();
        }

        // output = proj(h_new)
        let out_dim = self.proj.out_dim;
        let mut output = vec![0.0f32; out_dim];
        self.proj.forward(&h_new, &mut output);

        (output, h_new)
    }

    /// Return h_dim for initializing hidden state.
    pub fn h_dim(&self) -> usize {
        self.h_dim
    }
}

// ---------------------------------------------------------------------------
// V5 actor-critic JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV5FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v5: EntityEncoderV5Json,
    cross_attn: CrossAttnJson,
    move_head: HeadJson,
    combat_head: CombatHeadV5Json,
    #[serde(default)]
    external_cls_proj: Option<LinearWeights>,
    #[serde(default)]
    temporal_cell: Option<CfCCellJson>,
}

/// V5 actor-critic weights: d=128, 8 heads, aggregate token, 34-dim entities, 10-dim threats.
///
/// Note: The latent interface runs on GPU only. The CfC temporal cell is
/// optionally available for CPU-side temporal inference.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV5 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    // Ability transformer (for CLS embeddings)
    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    // Entity encoder V5
    entity_encoder: FlatEntityEncoderV5,

    // Cross-attention for abilities
    cross_attn: FlatCrossAttention,

    // Decision heads
    move_head: FlatMoveHeadV5,
    combat_head: FlatCombatHeadV5,

    // CLS projection (None when external_cls_dim == d_model)
    external_cls_proj: Option<FlatLinear>,

    // CfC temporal cell (optional, for CPU-side temporal inference)
    temporal_cell: Option<FlatCfCCell>,
}

impl ActorCriticWeightsV5 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV5FileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err("token_emb size mismatch".to_string());
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoderV5::from_json(
            &file.entity_encoder_v5, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );
        let move_head = FlatMoveHeadV5::from_json(&file.move_head);
        let combat_head = FlatCombatHeadV5::from_json(&file.combat_head, arch.d_model);

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            move_head,
            combat_head,
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
            temporal_cell: file.temporal_cell.as_ref().map(FlatCfCCell::from_json),
        })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read V5 weights: {e}"))?;
        Self::from_json(&data)
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            out
        } else {
            cls.to_vec()
        }
    }

    /// Encode ability tokens into a [CLS] embedding via the transformer encoder.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Encode game state with V5 entity encoder.
    /// Accepts 34-dim entities, 10-dim threats, 8-dim positions, 12-dim zones, optional 16-dim aggregate.
    /// When zones are non-empty and model has zone_proj, uses unified zone path.
    pub fn encode_entities(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
        positions: &[&[f32]],
        aggregate: Option<&[f32]>,
    ) -> EntityStateV3 {
        self.encode_entities_with_zones(entities, entity_types, threats, positions, &[], aggregate)
    }

    /// Encode game state with unified zone tokens.
    pub fn encode_entities_with_zones(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
        positions: &[&[f32]],
        zones: &[&[f32]],
        aggregate: Option<&[f32]>,
    ) -> EntityStateV3 {
        let d = self.d_model;
        let (tokens, mask, n_total, type_ids) = self.entity_encoder.forward(
            entities, entity_types, threats, positions, zones, aggregate,
        );

        // Mean pool over valid tokens
        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for i in 0..n_total {
            if mask[i] {
                count += 1.0;
                for j in 0..d {
                    pooled[j] += tokens[i * d + j];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityStateV3 { tokens, mask, pooled, type_ids, n_total }
    }

    /// Compute dual-head action logits: move direction + combat pointer.
    pub fn dual_head_logits(
        &self,
        entity_state: &EntityStateV3,
        ability_cls: &[Option<&[f32]>],
    ) -> super::weights_actor_critic_v4::DualHeadOutput {
        let n_total = entity_state.n_total;

        // Cross-attend each ability CLS to entity tokens
        let mut ability_cross_embs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for cls_opt in ability_cls.iter().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_total,
                );
                ability_cross_embs.push(Some(cross_emb));
            } else {
                ability_cross_embs.push(None);
            }
        }

        // Move direction logits (9-way)
        let move_logits = self.move_head.forward(&entity_state.pooled);

        // Combat pointer logits
        let (combat_logits, attack_ptr, ability_ptrs) = self.combat_head.forward(
            &entity_state.pooled,
            &entity_state.tokens,
            &entity_state.mask,
            n_total,
            &entity_state.type_ids,
            &ability_cross_embs,
        );

        super::weights_actor_critic_v4::DualHeadOutput {
            move_logits, combat_logits, attack_ptr, ability_ptrs,
        }
    }

    /// Access the optional CfC temporal cell for CPU-side temporal inference.
    pub fn temporal_cell(&self) -> Option<&FlatCfCCell> {
        self.temporal_cell.as_ref()
    }
}
