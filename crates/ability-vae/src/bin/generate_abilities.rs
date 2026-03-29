//! Generate abilities from natural language descriptions using the trained tree decoder.
//!
//! Loads checkpoint, encodes prompts, generates abilities autoregressively.
//! Benchmarks throughput.

use std::time::Instant;

use burn::prelude::*;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};

#[cfg(feature = "gpu")]
use burn::backend::LibTorch;
#[cfg(feature = "gpu")]
type B = LibTorch;

#[cfg(feature = "cpu")]
use burn::backend::NdArray;
#[cfg(feature = "cpu")]
type B = NdArray;

use ability_vae::grammar_space::{self, GRAMMAR_DIM};
use ability_vae::text_encoder::*;
use ability_vae::tree_decoder::*;

use tactical_sim::effects::dsl::parse_abilities;

const EMBED_DIM: usize = STATIC_EMBED_DIM;
const MAX_TEXT_LEN: usize = 32;

/// Must match the struct used during training exactly.
#[derive(Module, Debug, Clone)]
struct E2ETreeModel<B2: Backend> {
    text_encoder: StaticEmbedder<B2>,
    tree_decoder: TreeDecoder<B2>,
}

impl<B2: Backend> E2ETreeModel<B2> {
    fn new(vocab_size: usize, device: &B2::Device) -> Self {
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

    let prompts = vec![
        "fire damage AoE with stun",
        "healing ally support ability",
        "dark melee assassin strike",
        "army-wide leadership buff",
        "passive that triggers on kill and gives a shield",
        "devastating ice ultimate with crowd control",
        "trade embargo economic warfare",
        "quick ranged projectile damage",
        "holy healing aura that protects allies from fear",
        "poison DoT with slow and debuff",
    ];
    let count = 5usize;
    let temperature = 0.7f32;

    // Build tokenizer (must match training)
    eprintln!("Building tokenizer...");
    let desc_path = if std::path::Path::new("dataset/ability_descriptions_filtered.jsonl").exists() {
        "dataset/ability_descriptions_filtered.jsonl"
    } else {
        "dataset/ability_descriptions_v2.jsonl"
    };

    let mut all_texts: Vec<String> = Vec::new();
    if let Ok(content) = std::fs::read_to_string(desc_path) {
        for line in content.lines() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(desc) = parsed["description"].as_str() {
                    if desc.len() > 10 { all_texts.push(desc.to_string()); }
                }
            }
        }
    }
    for path in find_ability_files("dataset/abilities") {
        if let Ok(content) = std::fs::read_to_string(&path) {
            for block in split_ability_blocks(&content) {
                if let Some(v) = grammar_space::encode(&block) {
                    for desc in describe_ability(&v) { all_texts.push(desc); }
                }
            }
        }
    }
    let tokenizer = WordTokenizer::fit(&all_texts, 3);
    eprintln!("Tokenizer: {} words", tokenizer.vocab_size());

    // Load model from checkpoint
    let checkpoint = "generated/tree_checkpoint_e150";
    eprintln!("Loading checkpoint: {}.mpk", checkpoint);
    let model: E2ETreeModel<B> = E2ETreeModel::new(tokenizer.vocab_size(), &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model: E2ETreeModel<B> = match <E2ETreeModel<B> as burn::module::Module<B>>::load_file::<NamedMpkFileRecorder<FullPrecisionSettings>, &str>(model, checkpoint, &recorder, &device) {
        Ok(m) => { eprintln!("Checkpoint loaded successfully."); m }
        Err(e) => {
            eprintln!("Failed to load checkpoint: {}", e);
            eprintln!("Using random weights (grammar space still guarantees valid output).");
            E2ETreeModel::new(tokenizer.vocab_size(), &device)
        }
    };
    eprintln!();

    // Warmup
    {
        let ids = tokenizer.encode("warmup");
        let len = ids.len().min(MAX_TEXT_LEN);
        let ids_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                ids.into_iter().take(len).map(|x| x as i64).collect::<Vec<_>>(), [len],
            ), &device,
        ).unsqueeze::<2>();
        let lens_t = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![len as i64], [1]), &device,
        );
        let emb = model.text_encoder.forward(ids_t, lens_t);
        let mem = emb.unsqueeze_dim::<3>(1);
        let _ = model.tree_decoder.generate(mem, temperature);
        eprintln!("Warmup complete.");
    }

    // Batch ALL prompts × count as a single batch for maximum GPU utilization
    let total_batch = prompts.len() * count;
    eprintln!("Batching {} prompts × {} samples = {} total", prompts.len(), count, total_batch);

    // Encode all prompts
    let encode_start = Instant::now();
    let mut all_ids = vec![0i64; total_batch * MAX_TEXT_LEN];
    let mut all_lens = vec![0i64; total_batch];

    for (pi, prompt) in prompts.iter().enumerate() {
        let ids = tokenizer.encode(prompt);
        let len = ids.len().min(MAX_TEXT_LEN);
        for sample_i in 0..count {
            let bi = pi * count + sample_i;
            all_lens[bi] = len as i64;
            for ti in 0..len {
                all_ids[bi * MAX_TEXT_LEN + ti] = ids[ti] as i64;
            }
        }
    }

    let ids_t = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(all_ids, [total_batch * MAX_TEXT_LEN]), &device,
    ).reshape([total_batch, MAX_TEXT_LEN]);
    let lens_t = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(all_lens, [total_batch]), &device,
    );

    let text_emb = model.text_encoder.forward(ids_t, lens_t); // [total_batch, D]
    let text_memory = text_emb.unsqueeze_dim::<3>(1); // [total_batch, 1, D]
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Text encoding: {:.1} ms for {} prompts", encode_ms, total_batch);

    // Generate all at once
    let gen_start = Instant::now();
    let (_, grammar_vecs) = model.tree_decoder.generate(text_memory, temperature);
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Tree decoding: {:.1} ms for {} abilities ({:.1} ms/ability)",
        gen_ms, total_batch, gen_ms / total_batch as f64);

    // Decode and display
    let total_start = Instant::now();
    let mut total_generated = 0usize;
    let mut total_parsed = 0usize;

    for (pi, prompt) in prompts.iter().enumerate() {
        println!("=== \"{}\" ===", prompt);
        for sample_i in 0..count {
            let bi = pi * count + sample_i;
            let v = &grammar_vecs[bi];
            let dsl = grammar_space::decode(v);
            let parsed = parse_abilities(&dsl).is_ok();
            total_generated += 1;
            if parsed { total_parsed += 1; }

            println!("  --- Sample {} [{}] ---", sample_i, if parsed { "OK" } else { "FAIL" });
            for line in dsl.lines() {
                println!("  {}", line);
            }
            println!();
        }
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("=== Benchmark Summary ===");
    eprintln!("  Prompts: {}", prompts.len());
    eprintln!("  Abilities generated: {}", total_generated);
    eprintln!("  Parse rate: {}/{} ({:.0}%)", total_parsed, total_generated,
        total_parsed as f64 / total_generated.max(1) as f64 * 100.0);
    eprintln!("  Total time: {:.0} ms", total_ms);
    eprintln!("  Avg per ability: {:.1} ms", total_ms / total_generated.max(1) as f64);
    eprintln!("  Throughput: {:.0} abilities/sec", total_generated as f64 / (total_ms / 1000.0));
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
            in_block = true; current.clear();
        }
        if in_block {
            current.push_str(line); current.push('\n');
            for ch in trimmed.chars() { match ch { '{' => brace_depth += 1, '}' => brace_depth -= 1, _ => {} } }
            if brace_depth <= 0 && current.contains('{') {
                blocks.push(current.trim().to_string()); current.clear(); brace_depth = 0; in_block = false;
            }
        }
    }
    blocks
}
