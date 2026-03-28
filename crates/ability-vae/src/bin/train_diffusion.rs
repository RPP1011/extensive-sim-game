//! Train the diffusion model on the grammar-defined ability space.
//!
//! Usage:
//!   cargo run -p ability-vae --release --bin train-diffusion

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
use ability_vae::diffusion::{self, FlowModel};

use tactical_sim::effects::dsl::parse_abilities;

const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 500;
const LR: f64 = 1e-3;
const CFG_DROP: f32 = 0.1;

fn main() {
    #[cfg(feature = "gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "cpu")]
    let device = Default::default();

    println!("=== Ability Diffusion Model ===");
    println!("Loading dataset...");

    // Load all .ability files, encode them to grammar space vectors
    let ability_files = find_ability_files("dataset/abilities");
    println!("  Found {} .ability files", ability_files.len());

    let mut data_points: Vec<([f32; GRAMMAR_DIM], usize)> = Vec::new(); // (vector, hint_label)
    let mut parse_count = 0;
    let mut encode_count = 0;

    for path in &ability_files {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Split into individual ability blocks
        let blocks = split_ability_blocks(&content);
        for block in &blocks {
            parse_count += 1;
            if let Some(v) = grammar_space::encode(block) {
                // Extract hint label from dim 5
                let hint_idx = (v[5] * 9.0) as usize;
                data_points.push((v, hint_idx.min(8)));
                encode_count += 1;
            }
        }
    }

    println!("  Parsed: {}, Encoded: {}", parse_count, encode_count);
    let n_total = data_points.len();
    let n_val = n_total / 10;
    let n_train = n_total - n_val;
    println!("  Train: {}, Val: {}", n_train, n_val);
    println!();

    // Validate grammar space roundtrip
    println!("=== Grammar Space Validation ===");
    let (ok, fail) = grammar_space::validate_random(1000);
    println!("  Random decode→parse: {}/{} ({:.0}%)", ok, ok + fail,
        ok as f64 / (ok + fail) as f64 * 100.0);
    println!();

    // Build model
    let denoiser: FlowModel<B> = FlowModel::new(&device);
    println!("  Flow model params: ~{}", denoiser.num_params());
    println!("  Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}");
    println!("  CFG dropout: {CFG_DROP}");
    println!();

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, FlowModel<B>>();

    let mut denoiser = denoiser;
    let mut best_val_loss = f64::INFINITY;

    // Shuffle data
    let mut rng: u64 = 42;
    let mut train_indices: Vec<usize> = (0..n_train).collect();

    for epoch in 0..EPOCHS {
        let t0 = std::time::Instant::now();

        // Shuffle train indices
        for i in (1..train_indices.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            train_indices.swap(i, j);
        }

        // Train
        let mut train_loss = 0.0f64;
        let mut train_steps = 0usize;

        for chunk in train_indices.chunks(BATCH_SIZE) {
            let batch = chunk.len();
            let mut x_data = vec![0.0f32; batch * GRAMMAR_DIM];
            let mut label_data = vec![0.0f32; batch * 10];

            for (bi, &idx) in chunk.iter().enumerate() {
                let (v, hint) = &data_points[idx];
                for d in 0..GRAMMAR_DIM {
                    x_data[bi * GRAMMAR_DIM + d] = v[d];
                }
                if *hint < 10 {
                    label_data[bi * 10 + hint] = 1.0;
                }
            }

            let x_0 = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(x_data, [batch, GRAMMAR_DIM]),
                &device,
            );
            let labels = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(label_data, [batch, 10]),
                &device,
            );

            let output = diffusion::flow_loss(&denoiser, x_0, labels, CFG_DROP, &device);
            let loss = output.loss.clone();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &denoiser);
            denoiser = optim.step(LR, denoiser, grads);

            train_loss += output.loss_val;
            train_steps += 1;
        }

        // Validate
        let mut val_loss = 0.0f64;
        let mut val_steps = 0usize;

        for chunk in (n_train..n_total).collect::<Vec<_>>().chunks(BATCH_SIZE) {
            let batch = chunk.len();
            let mut x_data = vec![0.0f32; batch * GRAMMAR_DIM];
            let mut label_data = vec![0.0f32; batch * 10];

            for (bi, &idx) in chunk.iter().enumerate() {
                let (v, hint) = &data_points[idx];
                for d in 0..GRAMMAR_DIM {
                    x_data[bi * GRAMMAR_DIM + d] = v[d];
                }
                if *hint < 10 {
                    label_data[bi * 10 + hint] = 1.0;
                }
            }

            let x_0 = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(x_data, [batch, GRAMMAR_DIM]),
                &device,
            );
            let labels = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(label_data, [batch, 10]),
                &device,
            );

            let output = diffusion::flow_loss(&denoiser, x_0, labels, 0.0, &device);
            val_loss += output.loss_val;
            val_steps += 1;
        }

        let avg_val = val_loss / val_steps.max(1) as f64;
        if avg_val < best_val_loss { best_val_loss = avg_val; }

        let dt = t0.elapsed().as_secs_f64();
        if (epoch + 1) % 50 == 0 || epoch == 0 {
            print!(
                "Epoch {:>3}/{} | loss={:.6} val={:.6} | {:.1}s\n",
                epoch + 1, EPOCHS,
                train_loss / train_steps.max(1) as f64,
                avg_val, dt,
            );
            std::io::stdout().flush().ok();
        }
    }

    println!("\nBest val loss: {:.6}", best_val_loss);

    // Sample and evaluate
    println!("\n=== Unconditional Samples ===");
    let samples = diffusion::sample(&denoiser, 20, None, 1.0, 50, &device);
    let mut parse_ok = 0;
    for (i, v) in samples.iter().enumerate() {
        let dsl = grammar_space::decode(v);
        let parsed = parse_abilities(&dsl).is_ok();
        if parsed { parse_ok += 1; }
        if i < 5 {
            let status = if parsed { "OK" } else { "FAIL" };
            println!("  Sample {} [{}]:", i, status);
            for line in dsl.lines().take(6) {
                println!("    {}", line);
            }
            if dsl.lines().count() > 6 { println!("    ..."); }
            println!();
        }
    }
    println!("  Parse rate: {}/20 ({:.0}%)", parse_ok, parse_ok as f64 / 20.0 * 100.0);

    // Conditional samples
    println!("\n=== Conditional Samples (damage) ===");
    let damage_label = diffusion::make_label::<B>(diffusion::LABEL_DAMAGE, 10, &device);
    let damage_samples = diffusion::sample(&denoiser, 10, Some(damage_label), 3.0, 50, &device);
    for (i, v) in damage_samples.iter().enumerate().take(3) {
        let dsl = grammar_space::decode(v);
        println!("  Damage {}:", i);
        for line in dsl.lines().take(6) {
            println!("    {}", line);
        }
        println!();
    }

    println!("\n=== Conditional Samples (economy) ===");
    let econ_label = diffusion::make_label::<B>(diffusion::LABEL_ECONOMY, 10, &device);
    let econ_samples = diffusion::sample(&denoiser, 10, Some(econ_label), 3.0, 50, &device);
    for (i, v) in econ_samples.iter().enumerate().take(3) {
        let dsl = grammar_space::decode(v);
        println!("  Economy {}:", i);
        for line in dsl.lines().take(6) {
            println!("    {}", line);
        }
        println!();
    }

    println!("Done!");
}

fn find_ability_files(dir: &str) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let path = std::path::Path::new(dir);
    if !path.exists() { return files; }
    walk_dir(path, &mut files);
    files
}

fn walk_dir(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk_dir(&p, files);
            } else if p.extension().is_some_and(|e| e == "ability") {
                files.push(p);
            }
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
        if !in_block && (trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.is_empty()) {
            continue;
        }
        if !in_block && (trimmed.starts_with("ability ") || trimmed.starts_with("passive ")) {
            in_block = true;
            current.clear();
        }
        if in_block {
            current.push_str(line);
            current.push('\n');
            for ch in trimmed.chars() {
                match ch {
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    _ => {}
                }
            }
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
