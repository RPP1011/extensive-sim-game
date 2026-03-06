//! Frozen autoencoder (loaded from JSON weights).
//!
//! FlatMLP, AbilityEncoder, AbilityDecoder, AutoencoderFile, load_autoencoder.

use serde::{Deserialize, Serialize};

use crate::ai::effects::AbilitySlot;

use super::properties::{
    ABILITY_EMBED_DIM, ABILITY_PROP_DIM, ABILITY_SLOT_DIM,
    extract_ability_properties,
};

// ---------------------------------------------------------------------------
// Weight types
// ---------------------------------------------------------------------------

/// 2-layer weight block (shared shape for encoder and decoder halves).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoLayerWeights {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
}

/// Top-level JSON: autoencoder format (encoder + decoder) or legacy encoder-only.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AutoencoderFile {
    encoder: Option<TwoLayerWeights>,
    decoder: Option<TwoLayerWeights>,
    // Legacy flat format (encoder-only, no nesting)
    w1: Option<Vec<Vec<f32>>>,
    b1: Option<Vec<f32>>,
    w2: Option<Vec<Vec<f32>>>,
    b2: Option<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// FlatMLP
// ---------------------------------------------------------------------------

/// Flat cache-friendly 2-layer MLP.
#[derive(Debug, Clone)]
struct FlatMLP {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl FlatMLP {
    fn from_weights(w: &TwoLayerWeights, in_dim: usize, out_dim: usize) -> Result<Self, String> {
        let hidden = w.b1.len();
        if w.w1.len() != in_dim {
            return Err(format!("w1: expected {in_dim} rows, got {}", w.w1.len()));
        }
        if w.b2.len() != out_dim {
            return Err(format!("b2: expected {out_dim}, got {}", w.b2.len()));
        }
        let w1 = flatten_2d(&w.w1, in_dim, hidden)?;
        let w2 = flatten_2d(&w.w2, hidden, out_dim)?;
        Ok(Self { in_dim, hidden, out_dim, w1, b1: w.b1.clone(), w2, b2: w.b2.clone() })
    }

    /// Forward: input → ReLU hidden → output (no activation on output).
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let mut h = vec![0.0f32; self.hidden];
        for j in 0..self.hidden {
            let mut sum = self.b1[j];
            for k in 0..self.in_dim {
                sum += input[k] * self.w1[k * self.hidden + j];
            }
            h[j] = sum.max(0.0);
        }
        for j in 0..self.out_dim {
            let mut sum = self.b2[j];
            for k in 0..self.hidden {
                sum += h[k] * self.w2[k * self.out_dim + j];
            }
            output[j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// AbilityEncoder
// ---------------------------------------------------------------------------

/// Frozen encoder: properties (80) → L2-normalized embedding (32).
#[derive(Debug, Clone)]
pub struct AbilityEncoder {
    mlp: FlatMLP,
}

/// Frozen decoder: embedding (32) → reconstructed properties (80).
#[derive(Debug, Clone)]
pub struct AbilityDecoder {
    mlp: FlatMLP,
}

/// Load both encoder and decoder from a single JSON file.
/// Supports the autoencoder format (`{"encoder": {...}, "decoder": {...}}`)
/// and the legacy encoder-only format (`{"w1": ..., "b1": ..., ...}`).
pub fn load_autoencoder(json_str: &str) -> Result<(AbilityEncoder, Option<AbilityDecoder>), String> {
    let file: AutoencoderFile =
        serde_json::from_str(json_str).map_err(|e| format!("parse error: {e}"))?;

    let encoder = if let Some(ref enc_w) = file.encoder {
        // Autoencoder format
        AbilityEncoder {
            mlp: FlatMLP::from_weights(enc_w, ABILITY_PROP_DIM, ABILITY_EMBED_DIM)?,
        }
    } else if let (Some(w1), Some(b1), Some(w2), Some(b2)) =
        (file.w1.as_ref(), file.b1.as_ref(), file.w2.as_ref(), file.b2.as_ref())
    {
        // Legacy flat format
        let legacy = TwoLayerWeights {
            w1: w1.clone(), b1: b1.clone(), w2: w2.clone(), b2: b2.clone(),
        };
        AbilityEncoder {
            mlp: FlatMLP::from_weights(&legacy, ABILITY_PROP_DIM, ABILITY_EMBED_DIM)?,
        }
    } else {
        return Err("JSON must have 'encoder' key or flat w1/b1/w2/b2 keys".into());
    };

    let decoder = if let Some(ref dec_w) = file.decoder {
        Some(AbilityDecoder {
            mlp: FlatMLP::from_weights(dec_w, ABILITY_EMBED_DIM, ABILITY_PROP_DIM)?,
        })
    } else {
        None
    };

    Ok((encoder, decoder))
}

impl AbilityEncoder {
    /// Load encoder from JSON (autoencoder or legacy format). Discards decoder.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let (encoder, _) = load_autoencoder(json_str)?;
        Ok(encoder)
    }

    /// Encode raw properties → L2-normalized embedding.
    pub fn encode(&self, props: &[f32; ABILITY_PROP_DIM]) -> [f32; ABILITY_EMBED_DIM] {
        let mut embed = [0.0f32; ABILITY_EMBED_DIM];
        self.mlp.forward(props, &mut embed);

        // L2 normalize
        let norm = embed.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut embed {
            *v /= norm;
        }
        embed
    }

    /// Convenience: encode an AbilityDef directly.
    pub fn encode_def(&self, def: &crate::ai::effects::AbilityDef) -> [f32; ABILITY_EMBED_DIM] {
        let props = extract_ability_properties(def);
        self.encode(&props)
    }

    /// Encode an ability slot: embedding + runtime features.
    pub fn encode_slot(
        &self,
        slot: &AbilitySlot,
        unit_resource: i32,
    ) -> [f32; ABILITY_SLOT_DIM] {
        let embed = self.encode_def(&slot.def);
        let mut out = [0.0f32; ABILITY_SLOT_DIM];
        out[..ABILITY_EMBED_DIM].copy_from_slice(&embed);

        let ready = slot.is_ready()
            && (slot.def.resource_cost <= 0 || unit_resource >= slot.def.resource_cost);
        out[ABILITY_EMBED_DIM] = if ready { 1.0 } else { 0.0 };

        if slot.def.cooldown_ms > 0 {
            out[ABILITY_EMBED_DIM + 1] =
                slot.cooldown_remaining_ms as f32 / slot.def.cooldown_ms as f32;
        }

        out
    }
}

impl AbilityDecoder {
    /// Decode embedding → reconstructed property vector.
    pub fn decode(&self, embed: &[f32; ABILITY_EMBED_DIM]) -> [f32; ABILITY_PROP_DIM] {
        let mut props = [0.0f32; ABILITY_PROP_DIM];
        self.mlp.forward(embed, &mut props);
        props
    }

    /// Round-trip: encode an ability, then decode the embedding back.
    /// Returns (embedding, reconstructed_properties).
    pub fn round_trip(
        &self,
        encoder: &AbilityEncoder,
        def: &crate::ai::effects::AbilityDef,
    ) -> ([f32; ABILITY_EMBED_DIM], [f32; ABILITY_PROP_DIM]) {
        let embed = encoder.encode_def(def);
        let recon = self.decode(&embed);
        (embed, recon)
    }

    /// Compute reconstruction MSE for an ability.
    pub fn reconstruction_error(
        &self,
        encoder: &AbilityEncoder,
        def: &crate::ai::effects::AbilityDef,
    ) -> f32 {
        let original = extract_ability_properties(def);
        let (_, recon) = self.round_trip(encoder, def);
        original.iter().zip(recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>() / ABILITY_PROP_DIM as f32
    }

    /// Interpolate between two ability embeddings and decode.
    /// `t` in [0, 1]: 0 = ability_a, 1 = ability_b.
    pub fn interpolate(
        &self,
        embed_a: &[f32; ABILITY_EMBED_DIM],
        embed_b: &[f32; ABILITY_EMBED_DIM],
        t: f32,
    ) -> [f32; ABILITY_PROP_DIM] {
        let mut interp = [0.0f32; ABILITY_EMBED_DIM];
        for i in 0..ABILITY_EMBED_DIM {
            interp[i] = embed_a[i] * (1.0 - t) + embed_b[i] * t;
        }
        // Re-normalize
        let norm = interp.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut interp {
            *v /= norm;
        }
        self.decode(&interp)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn flatten_2d(mat: &[Vec<f32>], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    if mat.len() != rows {
        return Err(format!("expected {rows} rows, got {}", mat.len()));
    }
    let mut flat = Vec::with_capacity(rows * cols);
    for row in mat {
        if row.len() != cols {
            return Err(format!("expected {cols} cols, got {}", row.len()));
        }
        flat.extend_from_slice(row);
    }
    Ok(flat)
}
