//! VAE inference — load trained weights and run forward pass in Rust.
//!
//! Loads JSON weights exported by `training/export_content_vae.py`,
//! runs the encoder→z→decoder pipeline, and produces slot vectors
//! that the serializer converts to valid DSL.

use serde::Deserialize;

use super::vae_features::VAE_INPUT_DIM;
use super::vae_serialize;
use super::vae_slots::{ABILITY_SLOT_DIM, CLASS_SLOT_DIM};

// ---------------------------------------------------------------------------
// Math primitives
// ---------------------------------------------------------------------------

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();
    for i in 0..n {
        x[i] = gamma[i] * (x[i] - mean) / std + beta[i];
    }
}

fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = bias[o];
        for i in 0..in_dim {
            sum += input[i] * weight[o * in_dim + i];
        }
        output[o] = sum;
    }
    output
}

fn linear_gelu(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = linear(input, weight, bias, out_dim);
    for v in out.iter_mut() {
        *v = gelu(*v);
    }
    out
}

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ParamEntry {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ExportedWeights {
    config: VaeConfig,
    params: std::collections::HashMap<String, ParamEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct VaeConfig {
    input_dim: usize,
    latent_dim: usize,
    hidden_dim: usize,
    ability_slot_dim: usize,
    class_slot_dim: usize,
    n_layers: usize,
}

// ---------------------------------------------------------------------------
// Loaded model
// ---------------------------------------------------------------------------

/// A residual block's weights.
struct ResBlockWeights {
    norm_gamma: Vec<f32>,
    norm_beta: Vec<f32>,
    linear1_w: Vec<f32>,
    linear1_b: Vec<f32>,
    linear2_w: Vec<f32>,
    linear2_b: Vec<f32>,
}

/// The full VAE model weights for inference.
pub struct ContentVaeWeights {
    config: VaeConfig,

    // Encoder
    enc_proj_w: Vec<f32>,
    enc_proj_b: Vec<f32>,
    enc_blocks: Vec<ResBlockWeights>,
    enc_norm_gamma: Vec<f32>,
    enc_norm_beta: Vec<f32>,
    fc_mu_w: Vec<f32>,
    fc_mu_b: Vec<f32>,

    // Content type head
    ct_linear1_w: Vec<f32>,
    ct_linear1_b: Vec<f32>,
    ct_linear2_w: Vec<f32>,
    ct_linear2_b: Vec<f32>,

    // Ability decoder
    ab_proj_w: Vec<f32>,
    ab_proj_b: Vec<f32>,
    ab_blocks: Vec<ResBlockWeights>,
    ab_norm_gamma: Vec<f32>,
    ab_norm_beta: Vec<f32>,
    ab_head_w: Vec<f32>,
    ab_head_b: Vec<f32>,

    // Class decoder
    cl_proj_w: Vec<f32>,
    cl_proj_b: Vec<f32>,
    cl_blocks: Vec<ResBlockWeights>,
    cl_norm_gamma: Vec<f32>,
    cl_norm_beta: Vec<f32>,
    cl_head_w: Vec<f32>,
    cl_head_b: Vec<f32>,
}

impl std::fmt::Debug for ContentVaeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentVaeWeights(h={}, z={})", self.config.hidden_dim, self.config.latent_dim)
    }
}

impl ContentVaeWeights {
    /// Load weights from JSON file.
    pub fn load(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let reader = std::io::BufReader::new(file);
        let exported: ExportedWeights = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse {}: {}", path, e))?;

        let p = &exported.params;
        let cfg = &exported.config;
        let get = |key: &str| -> Result<Vec<f32>, String> {
            p.get(key)
                .map(|e| e.data.clone())
                .ok_or_else(|| format!("Missing param: {}", key))
        };

        let load_blocks = |prefix: &str, n: usize| -> Result<Vec<ResBlockWeights>, String> {
            let mut blocks = Vec::new();
            for i in 0..n {
                blocks.push(ResBlockWeights {
                    norm_gamma: get(&format!("{}.{}.net.0.weight", prefix, i))?,
                    norm_beta: get(&format!("{}.{}.net.0.bias", prefix, i))?,
                    linear1_w: get(&format!("{}.{}.net.1.weight", prefix, i))?,
                    linear1_b: get(&format!("{}.{}.net.1.bias", prefix, i))?,
                    linear2_w: get(&format!("{}.{}.net.4.weight", prefix, i))?,
                    linear2_b: get(&format!("{}.{}.net.4.bias", prefix, i))?,
                });
            }
            Ok(blocks)
        };

        Ok(ContentVaeWeights {
            enc_proj_w: get("enc_proj.weight")?,
            enc_proj_b: get("enc_proj.bias")?,
            enc_blocks: load_blocks("enc_blocks", cfg.n_layers)?,
            enc_norm_gamma: get("enc_norm.weight")?,
            enc_norm_beta: get("enc_norm.bias")?,
            fc_mu_w: get("fc_mu.weight")?,
            fc_mu_b: get("fc_mu.bias")?,

            ct_linear1_w: get("content_type_head.0.weight")?,
            ct_linear1_b: get("content_type_head.0.bias")?,
            ct_linear2_w: get("content_type_head.2.weight")?,
            ct_linear2_b: get("content_type_head.2.bias")?,

            ab_proj_w: get("ab_proj.weight")?,
            ab_proj_b: get("ab_proj.bias")?,
            ab_blocks: load_blocks("ab_blocks", cfg.n_layers)?,
            ab_norm_gamma: get("ab_norm.weight")?,
            ab_norm_beta: get("ab_norm.bias")?,
            ab_head_w: get("ab_head.weight")?,
            ab_head_b: get("ab_head.bias")?,

            cl_proj_w: get("cl_proj.weight")?,
            cl_proj_b: get("cl_proj.bias")?,
            cl_blocks: load_blocks("cl_blocks", cfg.n_layers)?,
            cl_norm_gamma: get("cl_norm.weight")?,
            cl_norm_beta: get("cl_norm.bias")?,
            cl_head_w: get("cl_head.weight")?,
            cl_head_b: get("cl_head.bias")?,

            config: cfg.clone(),
        })
    }

    /// Run the VAE encoder to get μ (deterministic mode — no sampling).
    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let h = self.config.hidden_dim;

        // Project input → hidden
        let mut x = linear_gelu(input, &self.enc_proj_w, &self.enc_proj_b, h);

        // Residual blocks
        for block in &self.enc_blocks {
            let mut residual = x.clone();
            layer_norm(&mut residual, &block.norm_gamma, &block.norm_beta);
            residual = linear_gelu(&residual, &block.linear1_w, &block.linear1_b, h);
            residual = linear(&residual, &block.linear2_w, &block.linear2_b, h);
            for i in 0..h {
                x[i] += residual[i];
            }
        }

        // Final norm
        layer_norm(&mut x, &self.enc_norm_gamma, &self.enc_norm_beta);

        // μ only (no sampling for inference — use the mean)
        linear(&x, &self.fc_mu_w, &self.fc_mu_b, self.config.latent_dim)
    }

    /// Run the decoder from latent z to slot vectors.
    fn decode_ability(&self, z: &[f32]) -> Vec<f32> {
        self.run_decoder(z, &self.ab_proj_w, &self.ab_proj_b,
            &self.ab_blocks, &self.ab_norm_gamma, &self.ab_norm_beta,
            &self.ab_head_w, &self.ab_head_b, self.config.ability_slot_dim)
    }

    fn decode_class(&self, z: &[f32]) -> Vec<f32> {
        self.run_decoder(z, &self.cl_proj_w, &self.cl_proj_b,
            &self.cl_blocks, &self.cl_norm_gamma, &self.cl_norm_beta,
            &self.cl_head_w, &self.cl_head_b, self.config.class_slot_dim)
    }

    fn decode_content_type(&self, z: &[f32]) -> usize {
        let h = self.config.hidden_dim / 4;
        let hidden = linear_gelu(z, &self.ct_linear1_w, &self.ct_linear1_b, h);
        let logits = linear(&hidden, &self.ct_linear2_w, &self.ct_linear2_b, 2);
        if logits[1] > logits[0] { 1 } else { 0 }
    }

    fn run_decoder(&self, z: &[f32],
        proj_w: &[f32], proj_b: &[f32],
        blocks: &[ResBlockWeights],
        norm_gamma: &[f32], norm_beta: &[f32],
        head_w: &[f32], head_b: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        let h = self.config.hidden_dim;
        let mut x = linear_gelu(z, proj_w, proj_b, h);

        for block in blocks {
            let mut residual = x.clone();
            layer_norm(&mut residual, &block.norm_gamma, &block.norm_beta);
            residual = linear_gelu(&residual, &block.linear1_w, &block.linear1_b, h);
            residual = linear(&residual, &block.linear2_w, &block.linear2_b, h);
            for i in 0..h {
                x[i] += residual[i];
            }
        }

        layer_norm(&mut x, norm_gamma, norm_beta);
        linear(&x, head_w, head_b, out_dim)
    }

    /// Concatenate z and input for conditional decoding.
    fn concat_zx(z: &[f32], input: &[f32]) -> Vec<f32> {
        let mut zx = Vec::with_capacity(z.len() + input.len());
        zx.extend_from_slice(z);
        zx.extend_from_slice(input);
        zx
    }

    /// Generate an ability DSL string from a 124-dim input vector.
    pub fn generate_ability(&self, input: &[f32; VAE_INPUT_DIM], name: &str) -> String {
        let z = self.encode(input);
        let zx = Self::concat_zx(&z, input);
        let slots = self.decode_ability(&zx);
        vae_serialize::slots_to_ability_dsl(&slots, name)
    }

    /// Generate a class DSL string from a 124-dim input vector.
    pub fn generate_class(&self, input: &[f32; VAE_INPUT_DIM], name: &str) -> String {
        let z = self.encode(input);
        let zx = Self::concat_zx(&z, input);
        let slots = self.decode_class(&zx);
        vae_serialize::slots_to_class_dsl(&slots, name)
    }

    /// Generate content — auto-selects ability vs class based on the model's prediction.
    pub fn generate(&self, input: &[f32; VAE_INPUT_DIM], name: &str) -> (String, &'static str) {
        let z = self.encode(input);
        let zx = Self::concat_zx(&z, input);
        let ct = self.decode_content_type(&zx);
        if ct == 0 {
            let slots = self.decode_ability(&zx);
            (vae_serialize::slots_to_ability_dsl(&slots, name), "ability")
        } else {
            let slots = self.decode_class(&zx);
            (vae_serialize::slots_to_class_dsl(&slots, name), "class")
        }
    }
}
