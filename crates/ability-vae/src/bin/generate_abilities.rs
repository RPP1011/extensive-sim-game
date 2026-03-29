//! Generate abilities from natural language descriptions using the trained tree decoder.
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin generate-abilities

use std::time::Instant;

use burn::prelude::*;

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
    let count = 5; // per prompt
    let temperature = 0.7f32;

    // Build tokenizer
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

    // Create model (randomly initialized — no checkpoint loading for now)
    // The grammar space guarantees 100% validity regardless of model quality
    let encoder: StaticEmbedder<B> = StaticEmbedder::new(tokenizer.vocab_size(), EMBED_DIM, &device);
    let decoder: TreeDecoder<B> = TreeDecoder::new(EMBED_DIM, &device);
    eprintln!("Model initialized (random weights — benchmarking pipeline speed)");
    eprintln!();

    // Generate
    let total_start = Instant::now();
    let mut total_generated = 0usize;
    let mut total_parsed = 0usize;

    for prompt in &prompts {
        let prompt_start = Instant::now();

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

        let text_emb = encoder.forward(ids_t, lens_t);
        let text_memory = text_emb.unsqueeze_dim::<3>(1);

        println!("=== \"{}\" ===", prompt);

        for sample_i in 0..count {
            let sample_start = Instant::now();
            let (_, grammar_vecs) = decoder.generate(text_memory.clone(), temperature);

            for v in &grammar_vecs {
                let dsl = grammar_space::decode(v);
                let parsed = parse_abilities(&dsl).is_ok();
                total_generated += 1;
                if parsed { total_parsed += 1; }
                let ms = sample_start.elapsed().as_millis();

                println!("  --- Sample {} ({} ms) [{}] ---", sample_i, ms,
                    if parsed { "OK" } else { "FAIL" });
                for line in dsl.lines() {
                    println!("  {}", line);
                }
                println!();
            }
        }

        let prompt_ms = prompt_start.elapsed().as_millis();
        println!("  [{} samples in {} ms, {:.1} ms/sample]\n",
            count, prompt_ms, prompt_ms as f64 / count as f64);
    }

    let total_ms = total_start.elapsed().as_millis();
    eprintln!("=== Benchmark Summary ===");
    eprintln!("  Prompts: {}", prompts.len());
    eprintln!("  Abilities generated: {}", total_generated);
    eprintln!("  Parse rate: {}/{} ({:.0}%)", total_parsed, total_generated,
        total_parsed as f64 / total_generated.max(1) as f64 * 100.0);
    eprintln!("  Total time: {} ms", total_ms);
    eprintln!("  Avg per ability: {:.1} ms", total_ms as f64 / total_generated.max(1) as f64);
    eprintln!("  Throughput: {:.0} abilities/sec", total_generated as f64 / (total_ms as f64 / 1000.0));
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
