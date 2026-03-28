//! Train the static text encoder with MRL.
//!
//! Phase 1: Pretrain on STS-B sentence pairs (learn word semantics)
//! Phase 2: Fine-tune on (ability description, grammar vector) pairs
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin train-text-encoder

use std::io::Write;

use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

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

const EMBED_DIM: usize = STATIC_EMBED_DIM; // 128
const PRETRAIN_EPOCHS: usize = 30;
const FINETUNE_EPOCHS: usize = 500;
const BATCH_SIZE: usize = 512;
const PRETRAIN_LR: f64 = 1e-3;    // Lower for transformer (attention needs smaller LR)
const FINETUNE_LR: f64 = 5e-4;
const MAX_TEXT_LEN: usize = 32;

fn main() {
    #[cfg(feature = "gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "cpu")]
    let device = Default::default();

    println!("=== Static Text Encoder with MRL ===");
    println!();

    // =========================================================================
    // Phase 1: Load STS-B for pretraining
    // =========================================================================
    println!("--- Phase 1: Pretraining on STS-B ---");

    let stsb_path = "dataset/stsb/stsbenchmark.tsv";
    let stsb_pairs = if std::path::Path::new(stsb_path).exists() {
        load_stsb(stsb_path)
    } else {
        println!("  STS-B not found at {}, downloading...", stsb_path);
        download_stsb();
        if std::path::Path::new(stsb_path).exists() {
            load_stsb(stsb_path)
        } else {
            println!("  Download failed, skipping pretraining.");
            Vec::new()
        }
    };
    println!("  STS-B pairs: {}", stsb_pairs.len());

    // Load TWI skills as additional pretraining pairs
    let twi_path = "dataset/twi_skills.json";
    let twi_pairs = if std::path::Path::new(twi_path).exists() {
        load_twi_skill_pairs(twi_path)
    } else {
        println!("  TWI skills not found, skipping.");
        Vec::new()
    };
    println!("  TWI skill pairs: {}", twi_pairs.len());

    // Combine STS-B + TWI pairs for pretraining
    let mut all_pretrain_pairs = stsb_pairs;
    all_pretrain_pairs.extend(twi_pairs);
    println!("  Total pretrain pairs: {}", all_pretrain_pairs.len());

    // =========================================================================
    // Phase 2: Generate ability descriptions for fine-tuning
    // =========================================================================
    println!("\n--- Generating ability descriptions ---");

    let ability_files = find_ability_files("dataset/abilities");
    println!("  Found {} .ability files", ability_files.len());

    let mut desc_pairs: Vec<(String, [f32; GRAMMAR_DIM])> = Vec::new();

    // Load LFM-generated descriptions if available
    // Prefer v2 (includes gap-filling abilities), fall back to v1
    let llm_desc_path = if std::path::Path::new("dataset/ability_descriptions_v2.jsonl").exists() {
        "dataset/ability_descriptions_v2.jsonl"
    } else {
        "dataset/ability_descriptions.jsonl"
    };
    if std::path::Path::new(llm_desc_path).exists() {
        let content = std::fs::read_to_string(llm_desc_path).unwrap_or_default();
        let mut llm_count = 0;
        for line in content.lines() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                let desc = parsed["description"].as_str().unwrap_or("").to_string();
                let dsl = parsed["dsl"].as_str().unwrap_or("");
                if desc.len() > 10 {
                    if let Some(v) = grammar_space::encode(dsl) {
                        desc_pairs.push((desc, v));
                        llm_count += 1;
                    }
                }
            }
        }
        println!("  From LFM descriptions: {} pairs", llm_count);
    }

    // Also add template-generated descriptions for coverage
    for path in &ability_files {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        for block in split_ability_blocks(&content) {
            if let Some(v) = grammar_space::encode(&block) {
                let descriptions = describe_ability(&v);
                for desc in descriptions {
                    desc_pairs.push((desc, v));
                }
            }
        }
    }
    println!("  + template descriptions");
    println!("  Total: {} (description, vector) pairs", desc_pairs.len());

    // =========================================================================
    // Build tokenizer from all texts
    // =========================================================================
    let mut all_texts: Vec<String> = desc_pairs.iter().map(|(d, _)| d.clone()).collect();
    for (a, b, _) in &all_pretrain_pairs {
        all_texts.push(a.clone());
        all_texts.push(b.clone());
    }
    let tokenizer = WordTokenizer::fit(&all_texts, 5); // higher min_freq to reduce vocab
    println!("  Vocabulary: {} words", tokenizer.vocab_size());

    // Tokenize everything
    let desc_tokens: Vec<Vec<u32>> = desc_pairs.iter().map(|(d, _)| tokenizer.encode(d)).collect();
    let stsb_tokens_a: Vec<Vec<u32>> = all_pretrain_pairs.iter().map(|(a, _, _)| tokenizer.encode(a)).collect();
    let stsb_tokens_b: Vec<Vec<u32>> = all_pretrain_pairs.iter().map(|(_, b, _)| tokenizer.encode(b)).collect();

    // =========================================================================
    // Build model
    // =========================================================================
    let mut model: StaticEmbedder<B> = StaticEmbedder::new(tokenizer.vocab_size(), EMBED_DIM, &device);
    println!("  Embed dim: {EMBED_DIM}, MRL dims: {:?}", &[32, 64, 128]);
    println!();

    // =========================================================================
    // Phase 1: Pretrain on STS-B
    // =========================================================================
    if !all_pretrain_pairs.is_empty() {
        println!("Pretraining on {} pairs (STS-B + TWI)...", all_pretrain_pairs.len());
        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.01)
            .init::<B, StaticEmbedder<B>>();

        // Pre-compute pretrain batches on GPU
        println!("  Pre-computing pretrain batches...");
        let mut pretrain_gpu_batches: Vec<(
            Tensor<B, 2, Int>, Tensor<B, 1, Int>,
            Tensor<B, 2, Int>, Tensor<B, 1, Int>,
            Tensor<B, 1>,
        )> = Vec::new();
        for chunk_start in (0..all_pretrain_pairs.len()).step_by(BATCH_SIZE) {
            let chunk_end = (chunk_start + BATCH_SIZE).min(all_pretrain_pairs.len());
            let batch = chunk_end - chunk_start;
            if batch < 2 { continue; }
            let (ids_a, lens_a) = pad_batch(&stsb_tokens_a[chunk_start..chunk_end], MAX_TEXT_LEN, &device);
            let (ids_b, lens_b) = pad_batch(&stsb_tokens_b[chunk_start..chunk_end], MAX_TEXT_LEN, &device);
            let scores: Vec<f32> = all_pretrain_pairs[chunk_start..chunk_end]
                .iter().map(|(_, _, s)| *s / 5.0)
                .collect();
            let target_sim = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(scores, [batch]),
                &device,
            );
            pretrain_gpu_batches.push((ids_a, lens_a, ids_b, lens_b, target_sim));
        }
        println!("  {} pretrain batches cached", pretrain_gpu_batches.len());

        for epoch in 0..PRETRAIN_EPOCHS {
            let t0 = std::time::Instant::now();
            let mut total_loss = 0.0f64;
            let mut steps = 0;

            for (ids_a, lens_a, ids_b, lens_b, target_sim) in &pretrain_gpu_batches {
                let batch = ids_a.dims()[0];

                // Forward
                let emb_a = model.forward(ids_a.clone(), lens_a.clone());
                let emb_b = model.forward(ids_b.clone(), lens_b.clone());

                // Cosine similarity loss at each MRL dim
                let mut loss = Tensor::<B, 1>::zeros([1], &device);
                for &dim in &[32usize, 64, 128] {
                    let ea = emb_a.clone().slice([0..batch, 0..dim]);
                    let eb = emb_b.clone().slice([0..batch, 0..dim]);
                    let ea_n = l2_norm_2d(ea);
                    let eb_n = l2_norm_2d(eb);
                    let cos = (ea_n * eb_n).sum_dim(1).reshape([batch]); // [B]
                    let diff = cos - (*target_sim).clone();
                    let mse = (diff.clone() * diff).mean();
                    loss = loss + mse.unsqueeze();
                }

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(PRETRAIN_LR, model, grads);

                total_loss += loss.into_scalar().elem::<f64>();
                steps += 1;
            }

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                print!("  Pretrain {:>3}/{} | loss={:.6} | {:.1}s | {:.0} pairs/s\n",
                    epoch + 1, PRETRAIN_EPOCHS,
                    total_loss / steps.max(1) as f64,
                    t0.elapsed().as_secs_f64(),
                    all_pretrain_pairs.len() as f64 / t0.elapsed().as_secs_f64().max(0.001));
                std::io::stdout().flush().ok();
            }
        }
    }

    // =========================================================================
    // Phase 2: Fine-tune on ability descriptions
    // =========================================================================
    println!("\nFine-tuning on {} ability descriptions...", desc_pairs.len());
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, StaticEmbedder<B>>();

    let targets: Vec<[f32; GRAMMAR_DIM]> = desc_pairs.iter().map(|(_, v)| *v).collect();

    // Pre-compute all batches on GPU once
    println!("  Pre-computing batches on GPU...");
    let mut gpu_batches: Vec<(Tensor<B, 2, Int>, Tensor<B, 1, Int>, Tensor<B, 2>)> = Vec::new();
    for chunk_start in (0..desc_pairs.len()).step_by(BATCH_SIZE) {
        let chunk_end = (chunk_start + BATCH_SIZE).min(desc_pairs.len());
        let batch = chunk_end - chunk_start;
        if batch < 2 { continue; }
        let indices: Vec<usize> = (chunk_start..chunk_end).collect();
        let (ids, lens, target_vecs) = prepare_batch::<B>(
            &desc_tokens, &targets, &indices, MAX_TEXT_LEN, &device,
        );
        gpu_batches.push((ids, lens, target_vecs));
    }
    println!("  {} batches cached on GPU", gpu_batches.len());

    for epoch in 0..FINETUNE_EPOCHS {
        let t0 = std::time::Instant::now();
        let mut total_loss = 0.0f64;
        let mut steps = 0;

        for (ids, lens, target_vecs) in &gpu_batches {
            let emb = model.forward(ids.clone(), lens.clone());
            let (loss, loss_val) = mrl_loss(emb, target_vecs.clone());

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(FINETUNE_LR, model, grads);

            total_loss += loss_val;
            steps += 1;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            print!("  Finetune {:>3}/{} | loss={:.6} | {:.1}s | {:.0} pairs/s\n",
                epoch + 1, FINETUNE_EPOCHS,
                total_loss / steps.max(1) as f64,
                t0.elapsed().as_secs_f64(),
                desc_pairs.len() as f64 / t0.elapsed().as_secs_f64().max(0.001));
            std::io::stdout().flush().ok();
        }
    }

    // =========================================================================
    // Evaluate
    // =========================================================================
    println!("\n=== Evaluation ===");

    let test_prompts = &[
        "fire damage AoE",
        "healing ally support",
        "dark melee stun",
        "army-wide leadership buff",
        "passive on kill shield",
        "devastating ice ultimate",
        "trade embargo economic warfare",
        "quick ranged projectile",
    ];

    for prompt in test_prompts {
        let emb = model.embed_text(prompt, &tokenizer, &device);
        let emb_data: Vec<f32> = emb.to_data().to_vec().unwrap();

        // Find nearest ability description by cosine similarity
        let mut best_sim = -1.0f32;
        let mut best_desc = String::new();
        for (desc, target) in &desc_pairs {
            let mut sim = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for d in 0..32.min(EMBED_DIM) { // use 32-dim prefix (MRL)
                sim += emb_data[d] * target[d];
                norm_a += emb_data[d] * emb_data[d];
                norm_b += target[d] * target[d];
            }
            let cos = sim / (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
            if cos > best_sim {
                best_sim = cos;
                best_desc = desc.clone();
            }
        }
        println!("  \"{}\"", prompt);
        println!("    → nearest: \"{}\" (sim={:.3})", &best_desc[..best_desc.len().min(80)], best_sim);
        println!();
    }

    println!("Done!");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn l2_norm_2d<B2: Backend>(x: Tensor<B2, 2>) -> Tensor<B2, 2> {
    let norm = (x.clone() * x.clone()).sum_dim(1).sqrt().clamp_min(1e-8);
    x / norm
}

fn pad_batch<B2: Backend>(
    tokens: &[Vec<u32>],
    max_len: usize,
    device: &B2::Device,
) -> (Tensor<B2, 2, Int>, Tensor<B2, 1, Int>) {
    let batch = tokens.len();
    let mut ids = vec![0i64; batch * max_len];
    let mut lens = vec![0i64; batch];
    for (bi, seq) in tokens.iter().enumerate() {
        let len = seq.len().min(max_len);
        lens[bi] = len as i64;
        for ti in 0..len {
            ids[bi * max_len + ti] = seq[ti] as i64;
        }
    }
    let ids_t = Tensor::<B2, 1, Int>::from_data(
        burn::tensor::TensorData::new(ids, [batch * max_len]),
        device,
    ).reshape([batch, max_len]);
    let lens_t = Tensor::<B2, 1, Int>::from_data(
        burn::tensor::TensorData::new(lens, [batch]),
        device,
    );
    (ids_t, lens_t)
}

/// Load STS-B TSV file: (sentence_a, sentence_b, score)
fn load_stsb(path: &str) -> Vec<(String, String, f32)> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut pairs = Vec::new();
    for line in content.lines().skip(1) { // skip header
        let cols: Vec<&str> = line.split('\t').collect();
        // Format: split genre dataset year sid score sentence1 sentence2
        if cols.len() >= 8 {
            if let Ok(score) = cols[5].parse::<f32>() {
                let s1 = cols[6].to_string();
                let s2 = cols[7].to_string();
                if !s1.is_empty() && !s2.is_empty() {
                    pairs.push((s1, s2, score));
                }
            }
        }
    }
    pairs
}

/// Load TWI skills and create contrastive pairs.
/// Skills with similar keywords (damage+damage, heal+heal) get high similarity.
/// Skills from different categories get low similarity.
fn load_twi_skill_pairs(path: &str) -> Vec<(String, String, f32)> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let raw: Vec<String> = serde_json::from_str(&content).unwrap_or_default();

    // Parse into (name, description) pairs
    let mut skills: Vec<(String, String)> = Vec::new();
    for s in &raw {
        if let Some((name, desc)) = s.split_once(" ||| ") {
            let desc = desc.trim();
            if desc.len() > 10 {
                skills.push((name.to_string(), desc.to_string()));
            }
        }
    }

    // Categorize each skill
    let categories: &[(&str, &[&str])] = &[
        ("damage", &["damage", "strike", "slash", "attack", "arrow", "shot", "hurt", "destroy"]),
        ("heal", &["heal", "restore", "mend", "regenerat", "recover", "cure"]),
        ("defense", &["shield", "armor", "protect", "defend", "block", "resist", "tough"]),
        ("cc", &["stun", "root", "slow", "fear", "silence", "immobil", "freeze", "disable"]),
        ("movement", &["speed", "dash", "teleport", "move", "leap", "charge", "sprint"]),
        ("buff", &["strength", "enhance", "boost", "increase", "improve", "empower"]),
        ("summon", &["summon", "conjure", "create", "raise", "spawn"]),
        ("stealth", &["stealth", "invisible", "hidden", "sneak", "cloak"]),
        ("aura", &["aura", "radius", "nearby", "around"]),
        ("army", &["army", "troops", "soldiers", "command", "formation"]),
    ];

    fn categorize<'a>(desc: &str, cats: &'a [(&str, &[&str])]) -> Option<&'a str> {
        let dl = desc.to_lowercase();
        for (cat, keywords) in cats {
            if keywords.iter().any(|kw| dl.contains(kw)) {
                return Some(cat);
            }
        }
        None
    }

    let mut pairs = Vec::new();

    // Create positive pairs: skills in same category get similarity 4.0-5.0
    // Create negative pairs: skills in different categories get 0.0-1.0
    let categorized: Vec<_> = skills.iter()
        .filter_map(|(_, desc)| {
            categorize(desc, categories).map(|c| (c, desc.clone()))
        })
        .collect();

    let mut rng: u64 = 42;
    for i in 0..categorized.len() {
        // Positive pair: find another skill in same category
        for j in (i+1)..categorized.len().min(i + 20) {
            if categorized[i].0 == categorized[j].0 {
                pairs.push((categorized[i].1.clone(), categorized[j].1.clone(), 4.5));
                break;
            }
        }

        // Negative pair: random skill from different category
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % categorized.len();
        if categorized[i].0 != categorized[j].0 {
            pairs.push((categorized[i].1.clone(), categorized[j].1.clone(), 0.5));
        }

        if pairs.len() > 3000 { break; } // cap size
    }

    pairs
}

fn download_stsb() {
    std::fs::create_dir_all("dataset/stsb").ok();
    let url = "https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsbenchmark.tsv";
    let result = std::process::Command::new("curl")
        .args(["-sL", "-o", "dataset/stsb/stsbenchmark.tsv", url])
        .status();
    match result {
        Ok(s) if s.success() => println!("  Downloaded STS-B."),
        _ => println!("  Failed to download STS-B."),
    }
}

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
