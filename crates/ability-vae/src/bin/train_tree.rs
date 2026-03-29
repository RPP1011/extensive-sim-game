//! E2E training: Text Encoder + Tree Decoder with classification heads.
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin train-tree

use std::io::Write;
use std::fs::OpenOptions;

use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::nn::{Linear, LinearConfig};

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
use ability_vae::tree_decoder::*;

use tactical_sim::effects::dsl::parse_abilities;

const EMBED_DIM: usize = STATIC_EMBED_DIM;
const EPOCHS: usize = 300;
const BATCH_SIZE: usize = 128;
const LR: f64 = 3e-4;
const MAX_TEXT_LEN: usize = 32;

macro_rules! log {
    ($file:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        print!("{}", msg);
        std::io::stdout().flush().ok();
        if let Some(ref mut f) = $file {
            write!(f, "{}", msg).ok();
            f.flush().ok();
        }
    }};
}

// ---------------------------------------------------------------------------
// Joint model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct E2ETreeModel<B: Backend> {
    text_encoder: StaticEmbedder<B>,
    tree_decoder: TreeDecoder<B>,
}

impl<B: Backend> E2ETreeModel<B> {
    fn new(vocab_size: usize, device: &B::Device) -> Self {
        Self {
            text_encoder: StaticEmbedder::new(vocab_size, EMBED_DIM, device),
            tree_decoder: TreeDecoder::new(EMBED_DIM, device),
        }
    }
}

fn main() {
    #[cfg(feature = "gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "cpu")]
    let device = Default::default();

    let mut logfile: Option<std::fs::File> = OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open("generated/tree_mixed_training.log").ok();

    log!(logfile, "=== E2E: Text Encoder + Tree Decoder ===\n");

    // Load data
    // Use filtered descriptions (only those with game-relevant keywords)
    let llm_desc_path = if std::path::Path::new("dataset/ability_descriptions_filtered.jsonl").exists() {
        "dataset/ability_descriptions_filtered.jsonl"
    } else if std::path::Path::new("dataset/ability_descriptions_v2.jsonl").exists() {
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
    log!(logfile, "  LFM descriptions: {} pairs\n", pairs.len());

    // Add template descriptions
    for path in find_ability_files("dataset/abilities") {
        if let Ok(content) = std::fs::read_to_string(&path) {
            for block in split_ability_blocks(&content) {
                if let Some(v) = grammar_space::encode(&block) {
                    for desc in describe_ability(&v) {
                        pairs.push((desc, v));
                    }
                }
            }
        }
    }
    log!(logfile, "  Total: {} pairs\n", pairs.len());

    // Tokenizer
    let texts: Vec<String> = pairs.iter().map(|(d, _)| d.clone()).collect();
    let tokenizer = WordTokenizer::fit(&texts, 3);
    log!(logfile, "  Vocab: {} tokens\n", tokenizer.vocab_size());

    // Tokenize text
    let tokens: Vec<Vec<u32>> = pairs.iter().map(|(d, _)| tokenizer.encode(d)).collect();

    // Convert grammar vectors to bins + extract classification labels
    let target_bins: Vec<Vec<i64>> = pairs.iter().map(|(_, v)| to_bins(v)).collect();
    let target_labels: Vec<[i64; 4]> = pairs.iter().map(|(_, v)| {
        let hint_bin = ((v[6] * 9.0) as i64).clamp(0, 8);
        let elem_bin = ((v[17] * 8.0) as i64).clamp(0, 7);
        let tgt_bin = ((v[2] * 15.0) as i64).clamp(0, 14);
        let domain_bin = if v[1] > 0.5 { 1i64 } else { 0 };
        [hint_bin, elem_bin, tgt_bin, domain_bin]
    }).collect();

    // Pre-compute batches on GPU
    log!(logfile, "  Pre-computing batches...\n");
    struct Batch<B2: Backend> {
        text_ids: Tensor<B2, 2, Int>,
        text_lens: Tensor<B2, 1, Int>,
        target_bins: Tensor<B2, 2, Int>,
        target_continuous: Tensor<B2, 2>,
        target_labels: Tensor<B2, 2, Int>,
    }

    let mut gpu_batches: Vec<Batch<B>> = Vec::new();
    for chunk_start in (0..pairs.len()).step_by(BATCH_SIZE) {
        let chunk_end = (chunk_start + BATCH_SIZE).min(pairs.len());
        let batch = chunk_end - chunk_start;
        if batch < 4 { continue; }

        // Text
        let mut ids_data = vec![0i64; batch * MAX_TEXT_LEN];
        let mut len_data = vec![0i64; batch];
        for (bi, idx) in (chunk_start..chunk_end).enumerate() {
            let toks = &tokens[idx];
            let len = toks.len().min(MAX_TEXT_LEN);
            len_data[bi] = len as i64;
            for ti in 0..len {
                ids_data[bi * MAX_TEXT_LEN + ti] = toks[ti] as i64;
            }
        }
        let text_ids = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(ids_data, [batch * MAX_TEXT_LEN]), &device,
        ).reshape([batch, MAX_TEXT_LEN]);
        let text_lens = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(len_data, [batch]), &device,
        );

        // Target bins
        let mut bins_data = vec![0i64; batch * GRAMMAR_DIM];
        for (bi, idx) in (chunk_start..chunk_end).enumerate() {
            for (d, &b) in target_bins[idx].iter().enumerate() {
                bins_data[bi * GRAMMAR_DIM + d] = b;
            }
        }
        let tgt_bins = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(bins_data, [batch * GRAMMAR_DIM]), &device,
        ).reshape([batch, GRAMMAR_DIM]);

        // Continuous target values (raw [0,1])
        let mut cont_data = vec![0.0f32; batch * GRAMMAR_DIM];
        for (bi, idx) in (chunk_start..chunk_end).enumerate() {
            let (_, v) = &pairs[idx];
            for d in 0..GRAMMAR_DIM {
                cont_data[bi * GRAMMAR_DIM + d] = v[d];
            }
        }
        let tgt_continuous = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(cont_data, [batch * GRAMMAR_DIM]), &device,
        ).reshape([batch, GRAMMAR_DIM]);

        // Classification labels
        let mut labels_data = vec![0i64; batch * 4];
        for (bi, idx) in (chunk_start..chunk_end).enumerate() {
            for (li, &l) in target_labels[idx].iter().enumerate() {
                labels_data[bi * 4 + li] = l;
            }
        }
        let tgt_labels = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(labels_data, [batch * 4]), &device,
        ).reshape([batch, 4]);

        gpu_batches.push(Batch {
            text_ids,
            text_lens,
            target_bins: tgt_bins,
            target_continuous: tgt_continuous,
            target_labels: tgt_labels,
        });
    }
    log!(logfile, "  {} batches on GPU\n", gpu_batches.len());

    // Build model
    let mut model: E2ETreeModel<B> = E2ETreeModel::new(tokenizer.vocab_size(), &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, E2ETreeModel<B>>();

    log!(logfile, "\n  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}\n");
    log!(logfile, "  Text encoder: 6-layer transformer, d={EMBED_DIM}\n");
    log!(logfile, "  Tree decoder: 4-layer, {GRAMMAR_DIM} steps × {} bins\n", TREE_N_BINS);
    log!(logfile, "  Classification heads: hint(9) + element(8) + targeting(15) + domain(2)\n\n");

    // Training loop
    for epoch in 0..EPOCHS {
        let t0 = std::time::Instant::now();
        let mut total_loss = 0.0f64;
        let mut total_acc = 0.0f64;
        let mut total_cls = 0.0f64;
        let mut steps = 0;

        for batch in &gpu_batches {
            let [b, _] = batch.text_ids.dims();

            // Text encoder: get both token-level memory and CLS output
            let text_emb = model.text_encoder.forward(
                batch.text_ids.clone(), batch.text_lens.clone(),
            ); // [B, EMBED_DIM] — CLS pooled

            // For cross-attention memory, we need token-level outputs.
            // Re-run encoder to get the full sequence (before CLS pooling).
            // TODO: refactor StaticEmbedder to return both.
            // For now, expand CLS to a 1-token memory.
            let text_memory = text_emb.clone().unsqueeze_dim::<3>(1); // [B, 1, EMBED_DIM]

            // Tree decoder loss (mixed: CE for categorical, MSE for continuous)
            let output = tree_decoder_loss(
                &model.tree_decoder,
                batch.target_bins.clone(),
                batch.target_continuous.clone(),
                text_memory,
                text_emb,
                batch.target_labels.clone(),
            );

            let grads = output.loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LR, model, grads);

            total_loss += output.loss_val;
            total_acc += output.accuracy;
            total_cls += output.cls_loss_val;
            steps += 1;
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            let dt = t0.elapsed().as_secs_f64();
            log!(logfile, "  Epoch {:>3}/{} | ar_loss={:.4} acc={:.1}% cls={:.4} | {:.1}s\n",
                epoch + 1, EPOCHS,
                total_loss / steps.max(1) as f64,
                total_acc / steps.max(1) as f64 * 100.0,
                total_cls / steps.max(1) as f64,
                dt);
        }
    }

    // =========================================================================
    // Evaluation
    // =========================================================================
    log!(logfile, "\n=== Text → Ability Generation ===\n");

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
        // Encode text
        let ids = tokenizer.encode(prompt);
        let len = ids.len().min(MAX_TEXT_LEN);
        let ids_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                ids.into_iter().take(len).map(|x| x as i64).collect::<Vec<_>>(), [len],
            ), &device,
        ).unsqueeze::<2>();
        let lens_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![len as i64], [1]), &device,
        );

        let text_emb = model.text_encoder.forward(ids_t, lens_t);
        let text_memory = text_emb.unsqueeze_dim::<3>(1); // [1, 1, D]

        // Generate 3 samples
        let (_, grammar_vecs) = model.tree_decoder.generate(text_memory, 0.7);

        log!(logfile, "\n  \"{}\":\n", prompt);
        for (i, v) in grammar_vecs.iter().enumerate() {
            let dsl = grammar_space::decode(v);
            let parsed = parse_abilities(&dsl).is_ok();
            let first_line = dsl.lines().nth(1).unwrap_or("").trim();
            let effect_line = dsl.lines()
                .find(|l| {
                    let t = l.trim();
                    !t.is_empty() && !t.starts_with("ability") && !t.starts_with("passive")
                        && !t.starts_with("target") && !t.starts_with("cooldown")
                        && !t.starts_with("hint") && !t.starts_with("cost")
                        && !t.starts_with("trigger") && !t.starts_with("deliver")
                        && !t.starts_with("on_hit") && !t.starts_with("{") && !t.starts_with("}")
                        && !t.starts_with("//") && !t.starts_with("requires")
                })
                .unwrap_or("").trim();
            log!(logfile, "    {}: {} | {} [{}]\n", i,
                first_line, effect_line,
                if parsed { "OK" } else { "FAIL" });
        }
    }

    log!(logfile, "\nDone!\n");
}

// ---------------------------------------------------------------------------
// Helpers
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
