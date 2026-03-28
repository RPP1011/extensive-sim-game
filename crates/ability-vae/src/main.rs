//! Token-sequence VAE for ability DSL generation.
//!
//! Encodes ability DSL text as token sequences, compresses to a small latent
//! space, and decodes back to token sequences. The latent space captures
//! ability *meaning* — navigating it produces valid, coherent abilities.
//!
//! Architecture:
//!   Encoder: token_ids → Embedding → TransformerEncoder → [CLS] pool → MLP → z (μ, σ)
//!   Decoder: z → expand → TransformerDecoder (cross-attn on z) → Linear → logits
//!
//! Usage:
//!   cargo run -p ability-vae --release

mod model;
mod data;

use burn::prelude::*;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

#[cfg(feature = "gpu")]
use burn::backend::{Autodiff, LibTorch};
#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<LibTorch>;

#[cfg(feature = "cpu")]
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "cpu")]
type TrainBackend = Autodiff<NdArray>;

use std::io::Write;

use data::AbilityDataset;
use model::AbilityVAE;
use tactical_sim::effects::dsl::parse_abilities;

macro_rules! flush_println {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        std::io::stdout().flush().ok();
    }};
}


const LATENT_DIM: usize = 16;
const D_MODEL: usize = 128;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 4;
const D_FF: usize = 256;
const MAX_SEQ_LEN: usize = 128;
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 500;
const LR: f64 = 3e-4;
const KL_WEIGHT_MAX: f64 = 0.1;
const KL_WARMUP_EPOCHS: usize = 50;
const WORD_DROPOUT: f32 = 0.5; // force decoder to rely on z

fn main() {
    #[cfg(feature = "gpu")]
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    #[cfg(feature = "cpu")]
    let device = Default::default();

    // Load dataset
    println!("=== Token-Sequence Ability VAE (burn) ===");
    println!("Loading ability dataset...");

    let dataset = AbilityDataset::load::<TrainBackend>("dataset/abilities", &device);
    let n_total = dataset.num_samples();
    let n_val = n_total / 10;
    let n_train = n_total - n_val;

    println!("  Total abilities: {}", n_total);
    println!("  Train: {}, Val: {}", n_train, n_val);
    println!("  Vocab size: {}", dataset.vocab_size());
    println!("  Max seq len: {}", MAX_SEQ_LEN);
    println!();
    println!("  Architecture: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, d_ff={D_FF}");
    println!("  Latent dim: {LATENT_DIM}");
    println!("  Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}");
    println!();

    // Build model
    let mut model: AbilityVAE<TrainBackend> = AbilityVAE::new(
        dataset.vocab_size(),
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        LATENT_DIM,
        MAX_SEQ_LEN,
        &device,
    );

    let n_params: usize = model.num_params();
    println!("  Model params: {}", n_params);

    // Optimizer
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<TrainBackend, AbilityVAE<TrainBackend>>();

    let mut best_val_loss = f64::INFINITY;

    for epoch in 0..EPOCHS {
        let t0 = std::time::Instant::now();

        // KL warmup
        let kl_weight = if epoch < KL_WARMUP_EPOCHS {
            KL_WEIGHT_MAX * (epoch as f64) / (KL_WARMUP_EPOCHS as f64)
        } else {
            KL_WEIGHT_MAX
        };

        // Train
        let mut train_recon = 0.0f64;
        let mut train_kl = 0.0f64;
        let mut train_steps = 0usize;

        let train_batches = dataset.batches(0..n_train, BATCH_SIZE, &device);
        for (input_ids, target_ids, mask) in &train_batches {
            let output = model.forward_train(
                input_ids.clone(),
                target_ids.clone(),
                mask.clone(),
                kl_weight as f32,
                WORD_DROPOUT,
            );

            let loss = output.loss.clone();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LR, model, grads);

            train_recon += output.recon_loss_val;
            train_kl += output.kl_loss_val;
            train_steps += 1;
        }

        // Validate (no word dropout)
        let mut val_recon = 0.0f64;
        let mut val_kl = 0.0f64;
        let mut val_steps = 0usize;

        let val_batches = dataset.batches(n_train..n_total, BATCH_SIZE, &device);
        for (input_ids, target_ids, mask) in &val_batches {
            let output = model.valid().forward_train(
                input_ids.clone(),
                target_ids.clone(),
                mask.clone(),
                kl_weight as f32,
                0.0, // no word dropout for validation
            );
            val_recon += output.recon_loss_val;
            val_kl += output.kl_loss_val;
            val_steps += 1;
        }

        let avg_val = if val_steps > 0 {
            (val_recon + val_kl * kl_weight) / val_steps as f64
        } else {
            0.0
        };

        if avg_val < best_val_loss {
            best_val_loss = avg_val;
            // TODO: save checkpoint
        }

        let dt = t0.elapsed().as_secs_f64();
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            flush_println!(
                "Epoch {:>3}/{} | recon={:.4} kl={:.4} β={:.4} val={:.4} | {:.1}s",
                epoch + 1,
                EPOCHS,
                train_recon / train_steps.max(1) as f64,
                train_kl / train_steps.max(1) as f64,
                kl_weight,
                avg_val,
                dt,
            );
        }
    }

    println!("\nBest val loss: {:.4}", best_val_loss);

    // =========================================================================
    // Evaluation: roundtrip + random sampling + parse validation
    // =========================================================================

    println!("\n=== Roundtrip Reconstruction ===");
    let val_batches = dataset.batches(n_train..n_train + 10.min(n_val), 10, &device);
    if let Some((input_ids, _target_ids, mask)) = val_batches.first() {
        let reconstructed = model.roundtrip(input_ids.clone(), mask.clone());
        let originals = dataset.get_sequences(n_train..n_train + reconstructed.len().min(5));

        for (i, (orig, recon)) in originals.iter().zip(reconstructed.iter()).enumerate().take(5) {
            let orig_dsl = AbilityVAE::<TrainBackend>::tokens_to_dsl(orig);
            let recon_dsl = AbilityVAE::<TrainBackend>::tokens_to_dsl(recon);
            let token_match = orig.iter().zip(recon.iter())
                .take_while(|(a, b)| a == b).count();
            let orig_len = orig.iter().take_while(|&&t| t != 0).count();
            println!("  Sample {} ({}/{} tokens match):", i, token_match, orig_len);
            println!("    ORIG:  {}", orig_dsl.lines().next().unwrap_or(""));
            println!("    RECON: {}", recon_dsl.lines().next().unwrap_or(""));
        }
    }

    println!("\n=== Random Samples from Prior ===");
    let n_samples = 20;
    let z_random = Tensor::<TrainBackend, 2>::random(
        [n_samples, LATENT_DIM],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let sampled = model.generate(z_random, 80);

    let mut parse_ok = 0;
    let mut parse_fail = 0;

    for (i, tokens) in sampled.iter().enumerate() {
        let dsl = AbilityVAE::<TrainBackend>::tokens_to_dsl(tokens);

        // Try parsing
        let parse_result = parse_abilities(&dsl);
        let parsed = parse_result.is_ok();
        if parsed { parse_ok += 1; } else { parse_fail += 1; }

        if i < 5 || !parsed {
            let status = if parsed { "OK" } else { "FAIL" };
            println!("  Sample {} [{}]:", i, status);
            for line in dsl.lines().take(4) {
                println!("    {}", line);
            }
            if dsl.lines().count() > 4 {
                println!("    ...");
            }
            if !parsed {
                if let Err(e) = &parse_result {
                    println!("    Parse error: {}", e);
                }
            }
            println!();
        }
    }

    println!("=== Parse Results ===");
    println!("  Random samples: {}/{} parsed successfully ({:.0}%)",
        parse_ok, n_samples, parse_ok as f64 / n_samples as f64 * 100.0);

    // Roundtrip-parse a small subset of validation
    println!("\n=== Roundtrip Parse Validation (20 val samples) ===");
    let mut rt_ok = 0;
    let mut rt_total = 0;
    let val_batches_small = dataset.batches(n_train..n_train + 20.min(n_val), 4, &device);
    for (input_ids, _target_ids, mask) in &val_batches_small {
        let recons = model.roundtrip(input_ids.clone(), mask.clone());
        for tokens in &recons {
            let dsl = AbilityVAE::<TrainBackend>::tokens_to_dsl(tokens);
            if parse_abilities(&dsl).is_ok() {
                rt_ok += 1;
            }
            rt_total += 1;
        }
    }
    println!("  Roundtrip: {}/{} parsed successfully ({:.0}%)",
        rt_ok, rt_total, rt_ok as f64 / rt_total.max(1) as f64 * 100.0);

    println!("\nDone!");
}
