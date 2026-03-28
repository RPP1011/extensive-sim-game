//! End-to-end training: text encoder + flow model jointly.
//!
//! Description → TextEncoder → embedding → FlowModel conditioning → grammar space
//! Single loss (flow matching), gradients through both models.
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin train-e2e

use std::io::Write;
use std::fs::OpenOptions;

use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::nn::{Linear, LinearConfig};

/// Log to both stdout and a file, flushing immediately.
macro_rules! log {
    ($file:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        print!("{}", msg);
        if let Some(ref mut f) = $file {
            write!(f, "{}", msg).ok();
            f.flush().ok();
        }
    }};
}

#[cfg(feature = "gpu")]
use burn::backend::{Autodiff, LibTorch};
#[cfg(feature = "gpu")]
type B = Autodiff<LibTorch>;

#[cfg(feature = "cpu")]
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "cpu")]
type B = Autodiff<NdArray>;

use ability_vae::grammar_space::{self, GRAMMAR_DIM};
use ability_vae::text_encoder::*;
use ability_vae::diffusion::{self, FiLMFlowModel};
use ability_vae::grokfast::GrokfastEma;

use tactical_sim::effects::dsl::parse_abilities;

const EMBED_DIM: usize = STATIC_EMBED_DIM; // 256
const COND_DIM: usize = 128; // project embedding for flow conditioning
const EPOCHS: usize = 500;
const BATCH_SIZE: usize = 128; // smaller batch for bigger model
const LR: f64 = 1e-4; // lower LR for bigger model
const CFG_DROP: f32 = 0.1;
const MAX_TEXT_LEN: usize = 32;
const FLOW_STEPS: usize = 50;
const EMA_ALPHA: f32 = 0.98;
const EMA_LAMB: f32 = 2.0;

// ---------------------------------------------------------------------------
// Joint model: TextEncoder + projection + FlowModel
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct E2EModel<B: Backend> {
    text_encoder: StaticEmbedder<B>,
    cond_proj: Linear<B>,
    flow: FiLMFlowModel<B>,  // FiLM conditioning instead of concat
}

impl<B: Backend> E2EModel<B> {
    fn new(vocab_size: usize, device: &B::Device) -> Self {
        Self {
            text_encoder: StaticEmbedder::new(vocab_size, EMBED_DIM, device),
            cond_proj: LinearConfig::new(EMBED_DIM, COND_DIM).init(device),
            flow: FiLMFlowModel::new(COND_DIM, device),
        }
    }

    /// Encode text → conditioning vector for the flow model.
    fn encode_text(
        &self,
        token_ids: Tensor<B, 2, Int>,
        lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        let emb = self.text_encoder.forward(token_ids, lengths); // [B, 128]
        self.cond_proj.forward(emb) // [B, COND_DIM]
    }
}

fn main() {
    #[cfg(feature = "gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "cpu")]
    let device = Default::default();

    let mut logfile: Option<std::fs::File> = OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open("generated/e2e_training.log").ok();

    log!(logfile, "=== E2E: Text Encoder + Flow Matching ===\n");

    // Load descriptions + grammar vectors
    let llm_desc_path = if std::path::Path::new("dataset/ability_descriptions_v2.jsonl").exists() {
        "dataset/ability_descriptions_v2.jsonl"
    } else {
        "dataset/ability_descriptions.jsonl"
    };

    let mut pairs: Vec<(String, [f32; GRAMMAR_DIM])> = Vec::new();
    if let Ok(content) = std::fs::read_to_string(llm_desc_path) {
        for line in content.lines() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                let desc = parsed["description"].as_str().unwrap_or("").to_string();
                let dsl = parsed["dsl"].as_str().unwrap_or("");
                if desc.len() > 10 {
                    if let Some(v) = grammar_space::encode(dsl) {
                        pairs.push((desc, v));
                    }
                }
            }
        }
    }
    log!(logfile, "  Loaded {} (description, vector) pairs", pairs.len());

    // Also add template descriptions
    let ability_files = find_ability_files("dataset/abilities");
    for path in &ability_files {
        if let Ok(content) = std::fs::read_to_string(path) {
            for block in split_ability_blocks(&content) {
                if let Some(v) = grammar_space::encode(&block) {
                    for desc in describe_ability(&v) {
                        pairs.push((desc, v));
                    }
                }
            }
        }
    }
    log!(logfile, "  Total: {} pairs", pairs.len());

    // Build tokenizer
    let texts: Vec<String> = pairs.iter().map(|(d, _)| d.clone()).collect();
    let tokenizer = WordTokenizer::fit(&texts, 3);
    log!(logfile, "  Vocab: {} tokens", tokenizer.vocab_size());

    // Tokenize
    let tokens: Vec<Vec<u32>> = pairs.iter().map(|(d, _)| tokenizer.encode(d)).collect();
    let targets: Vec<[f32; GRAMMAR_DIM]> = pairs.iter().map(|(_, v)| *v).collect();

    // Pre-compute batches on GPU
    log!(logfile, "  Pre-computing batches...");
    let mut gpu_batches: Vec<(Tensor<B, 2, Int>, Tensor<B, 1, Int>, Tensor<B, 2>)> = Vec::new();
    for chunk_start in (0..pairs.len()).step_by(BATCH_SIZE) {
        let chunk_end = (chunk_start + BATCH_SIZE).min(pairs.len());
        let batch = chunk_end - chunk_start;
        if batch < 4 { continue; }
        let indices: Vec<usize> = (chunk_start..chunk_end).collect();
        let (ids, lens, tgt) = prepare_batch::<B>(&tokens, &targets, &indices, MAX_TEXT_LEN, &device);
        gpu_batches.push((ids, lens, tgt));
    }
    log!(logfile, "  {} batches on GPU", gpu_batches.len());

    // Build model
    let mut model: E2EModel<B> = E2EModel::new(tokenizer.vocab_size(), &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, E2EModel<B>>();

    // Grokfast EMA gradient filter (operates on inner backend tensors)
    #[cfg(feature = "gpu")]
    let mut grokfast: GrokfastEma<burn::backend::LibTorch> = GrokfastEma::new(EMA_ALPHA, EMA_LAMB);
    #[cfg(feature = "cpu")]
    let mut grokfast: GrokfastEma<burn::backend::NdArray> = GrokfastEma::new(EMA_ALPHA, EMA_LAMB);

    log!(logfile, "\n  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}");
    log!(logfile, "  Text encoder: 6-layer transformer, d={EMBED_DIM}");
    log!(logfile, "  Flow model: conditioned on {COND_DIM}-dim text embedding");
    log!(logfile, "  Grokfast EMA: alpha={EMA_ALPHA}, lambda={EMA_LAMB}\n");

    // Training loop
    for epoch in 0..EPOCHS {
        let t0 = std::time::Instant::now();
        let mut total_loss = 0.0f64;
        let mut steps = 0;

        for (ids, lens, x_data) in &gpu_batches {
            let [batch, _] = ids.dims();

            // Text → conditioning
            let cond = model.encode_text(ids.clone(), lens.clone()); // [B, COND_DIM]

            // CFG dropout: randomly zero out conditioning
            let cond = if CFG_DROP > 0.0 {
                let drop_mask = Tensor::<B, 2>::random(
                    [batch, 1],
                    burn::tensor::Distribution::Uniform(0.0, 1.0),
                    &device,
                ).lower_elem(CFG_DROP).float();
                cond * (drop_mask.neg() + 1.0)
            } else {
                cond
            };

            // Truncate target to GRAMMAR_DIM (it's padded to EMBED_DIM in prepare_batch)
            let x_0 = x_data.clone().slice([0..batch, 0..GRAMMAR_DIM]);

            // FiLM flow matching loss
            let output = diffusion::film_flow_loss(&model.flow, x_0, cond, &device);

            let grads = output.loss.backward();
            let mut grads = GradientsParams::from_grads(grads, &model);
            grads = grokfast.apply(grads, &model);
            model = optim.step(LR, model, grads);

            total_loss += output.loss_val;
            steps += 1;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let dt = t0.elapsed().as_secs_f64();
            log!(logfile, "  Epoch {:>3}/{} | loss={:.6} | {:.1}s | {:.0} pairs/s\n",
                epoch + 1, EPOCHS,
                total_loss / steps.max(1) as f64,
                dt,
                pairs.len() as f64 / dt.max(0.001));
        }
    }

    // =========================================================================
    // Evaluation: generate from text prompts
    // =========================================================================
    log!(logfile, "\n=== Text → Ability Generation ===");

    let prompts = &[
        "fire damage AoE with stun",
        "healing ally support",
        "dark melee assassin strike",
        "army-wide leadership buff",
        "passive that triggers on kill and gives a shield",
        "devastating ice ultimate",
        "trade embargo economic warfare",
        "quick ranged projectile",
        "holy healing aura that protects allies",
        "poison DoT with slow",
    ];

    for prompt in prompts {
        let cond = model.text_encoder.embed_text(prompt, &tokenizer, &device);
        let cond = model.cond_proj.forward(cond.unsqueeze::<2>()); // [1, COND_DIM]

        let samples = diffusion::film_sample(&model.flow, 3, cond, FLOW_STEPS, &device);

        log!(logfile, "\n  \"{}\":\n", prompt);
        for (i, v) in samples.iter().enumerate() {
            let dsl = grammar_space::decode(v);
            let parsed = parse_abilities(&dsl).is_ok();
            let first_line = dsl.lines().nth(1).unwrap_or("").trim();
            let second_line = dsl.lines().nth(5).unwrap_or("").trim();
            log!(logfile, "    {}: {} | {} [{}]\n", i,
                first_line, second_line,
                if parsed { "OK" } else { "FAIL" });
        }
    }

    log!(logfile, "\nDone!");
}

// (flow_loss_conditioned and sample_conditioned replaced by FiLM versions in diffusion.rs)

// ---------------------------------------------------------------------------
// Helpers (copied from other binaries)
// ---------------------------------------------------------------------------

fn find_ability_files(dir: &str) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    walk_dir(std::path::Path::new(dir), &mut files);
    files
}

fn walk_dir(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() { walk_dir(&p, files); }
            else if p.extension().is_some_and(|e| e == "ability") { files.push(p); }
        }
    }
}

fn split_ability_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0i32;
    let mut in_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_block && (trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.is_empty()) { continue; }
        if !in_block && (trimmed.starts_with("ability ") || trimmed.starts_with("passive ")) {
            in_block = true;
            current.clear();
        }
        if in_block {
            current.push_str(line);
            current.push('\n');
            for ch in trimmed.chars() { match ch { '{' => brace_depth += 1, '}' => brace_depth -= 1, _ => {} } }
            if brace_depth <= 0 && current.contains('{') {
                blocks.push(current.trim().to_string());
                current.clear();
                brace_depth = 0;
                in_block = false;
            }
        }
    }
    blocks
}
