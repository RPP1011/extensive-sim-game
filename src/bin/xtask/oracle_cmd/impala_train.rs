//! IMPALA V-trace training command using Burn V6 model.
//!
//! `xtask scenario oracle transformer-rl impala-train <scenarios> [options]`
//!
//! Runs the full IMPALA loop in-process:
//! 1. Generate episodes using Burn V6 inference (no SHM, no Python)
//! 2. Extract trajectories, compute V-trace targets
//! 3. Train with Burn autodiff (Autodiff<LibTorch>)
//! 4. Save checkpoint, log metrics
//! 5. Repeat

use std::process::ExitCode;
use std::collections::HashMap;

use crate::cli::ImpalaTrainArgs;

#[cfg(feature = "burn-gpu")]
use rl4burn::{ReplayBuffer, Logger, CompositeLogger, PrintLogger, TensorBoardLogger};

pub(crate) fn run_impala_train(args: ImpalaTrainArgs) -> ExitCode {
    #[cfg(not(feature = "burn-gpu"))]
    {
        eprintln!("impala-train requires --features burn-gpu");
        return ExitCode::from(1);
    }

    #[cfg(feature = "burn-gpu")]
    {
        run_impala_train_inner(args)
    }
}

#[cfg(feature = "burn-gpu")]
fn run_impala_train_inner(args: ImpalaTrainArgs) -> ExitCode {
    use bevy_game::ai::core::burn_model::{
        ActorCriticV6Config,
        actor_critic_v6::V6_H_DIM,
        checkpoint,
        training::{self, ImpalaConfig, TrainingSample, vtrace_targets, train_step, train_step_bc, predict_values},
    };
    use burn::backend::{Autodiff, LibTorch};
    use burn::backend::libtorch::LibTorchDevice;
    use burn::module::AutodiffModule;
    use burn::optim::{AdamWConfig, GradientsParams};
    use burn::prelude::*;

    type TrainBackend = Autodiff<LibTorch>;

    let device = LibTorchDevice::Cuda(0);

    // Collect scenario paths
    let scenario_paths: Vec<_> = args.path.iter()
        .flat_map(|p| super::collect_toml_paths(p))
        .collect();
    if scenario_paths.is_empty() {
        eprintln!("No *.toml scenario files found");
        return ExitCode::from(1);
    }
    eprintln!("Scenarios: {}", scenario_paths.len());

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).ok();

    // Model config
    let model_config = ActorCriticV6Config {
        vocab_size: 256,
        d_model: 128,
        d_ff: 256,
        n_heads: 8,
        n_layers: 4,
        entity_encoder_layers: 4,
        external_cls_dim: 0,
        h_dim: V6_H_DIM,
        n_latents: 12,
        n_latent_blocks: 2,
    };

    // Initialize or load model
    let mut model = if let Some(ref ckpt_path) = args.checkpoint {
        match checkpoint::load_v6::<TrainBackend>(&model_config, ckpt_path, &device) {
            Ok(m) => { eprintln!("Loaded checkpoint: {}", ckpt_path.display()); m }
            Err(e) => { eprintln!("Failed to load checkpoint: {e}"); return ExitCode::from(1); }
        }
    } else {
        model_config.init::<TrainBackend>(&device)
    };

    let params: usize = 0; // TODO: count parameters
    eprintln!("Model initialized: d=128, h={V6_H_DIM}, K=12, device=CUDA:0");

    // Training config
    let train_config = ImpalaConfig {
        lr: args.lr,
        batch_size: args.batch_size,
        gamma: 0.99,
        step_interval: args.step_interval,
        entropy_coef: args.entropy_coef,
        value_coef: args.value_coef,
        ..Default::default()
    };

    // Optimizer
    let optim_config = AdamWConfig::new()
        .with_beta_1(train_config.beta_1)
        .with_beta_2(train_config.beta_2)
        .with_weight_decay(train_config.weight_decay);
    let mut optimizer = optim_config.init();

    // Episode output path
    let episode_path = args.output_dir.join("episodes.jsonl");

    // Logger: print + TensorBoard
    let tb_dir = args.output_dir.join("tb");
    std::fs::create_dir_all(&tb_dir).ok();
    let mut loggers: Vec<Box<dyn Logger>> = vec![
        Box::new(PrintLogger::new(1)),
    ];
    match TensorBoardLogger::new(&tb_dir) {
        Ok(tb) => loggers.push(Box::new(tb)),
        Err(e) => eprintln!("Warning: TensorBoard logger init failed: {e}"),
    }
    let mut logger = CompositeLogger::new(loggers);

    // CSV log (kept as fallback)
    let log_path = args.output_dir.join("training_log.csv");
    let mut log_file = std::fs::File::create(&log_path).expect("create log file");
    use std::io::Write;
    writeln!(log_file, "iter,episodes,steps,win_rate,loss,policy_loss,value_loss").ok();

    let mut rng_state = 42u64;

    // Replay buffer: FIFO eviction when over capacity.
    // Re-score advantages every RESCORE_INTERVAL iterations using current value head.
    const REPLAY_MAX: usize = 100_000;
    const RESCORE_INTERVAL: usize = 5;
    use rand::SeedableRng;
    let mut replay_buffer: ReplayBuffer<TrainingSample, rand::rngs::StdRng> =
        ReplayBuffer::new(REPLAY_MAX, rand::rngs::StdRng::seed_from_u64(42));

    for iteration in 1..=args.iters {
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Iteration {iteration}/{}", args.iters);

        // 1. Generate episodes using xtask generate --burn-v6
        let t0 = std::time::Instant::now();
        // Burn's BinFileRecorder appends .bin automatically, so pass path without extension
        let prev_ckpt = if iteration > 1 {
            Some(args.output_dir.join(format!("v6_iter{:04}", iteration - 1)))
        } else {
            None
        };
        let ckpt_for_gen = args.checkpoint.as_deref()
            .or(prev_ckpt.as_deref());
        let gen_status = generate_episodes(
            &scenario_paths,
            &episode_path,
            &args,
            iteration,
            ckpt_for_gen,
        );
        if !gen_status {
            eprintln!("Episode generation failed at iteration {iteration}");
            return ExitCode::from(1);
        }
        let gen_time = t0.elapsed().as_secs_f64();

        // 2. Load episodes and extract training data
        let episodes = load_episodes(&episode_path);
        let wins = episodes.iter().filter(|e| e.outcome == "Victory").count();
        let win_rate = wins as f32 / episodes.len().max(1) as f32;

        let new_samples = extract_training_samples(&episodes, &train_config, &model, &device);
        eprintln!("  Episodes: {}, steps: {}, WR: {:.1}%, gen: {:.1}s",
            episodes.len(), new_samples.len(), win_rate * 100.0, gen_time);

        if new_samples.is_empty() {
            eprintln!("  No training samples, skipping");
            continue;
        }

        // Add new samples; FIFO eviction is automatic when over capacity
        let n_new = new_samples.len();
        replay_buffer.extend(new_samples);

        // Re-score V-trace advantages periodically using current value head
        if iteration % RESCORE_INTERVAL == 0 {
            use bevy_game::ai::core::burn_model::training::rescore_replay_buffer;
            let inner_model = model.valid();
            let inner_device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
            let gamma_eff = train_config.gamma.powi(train_config.step_interval as i32);
            rescore_replay_buffer(
                replay_buffer.samples_mut(), &inner_model, &inner_device,
                gamma_eff, train_config.clip_rho, train_config.clip_c,
            );
            eprintln!("  Re-scored {} samples with V-trace", replay_buffer.len());
        }

        eprintln!("  Replay buffer: {} samples (+{} new)", replay_buffer.len(), n_new);

        // 3. Train on replay buffer (not just current iteration's samples)
        let t1 = std::time::Instant::now();
        let mut total_metrics = training::TrainMetrics::default();

        for step_i in 0..args.train_steps {
            // Sample random minibatch from replay buffer
            let batch: Vec<TrainingSample> = replay_buffer.sample_cloned(args.batch_size);

            // Diagnostic: log advantage stats for first batch of each iteration
            if step_i == 0 {
                let adv_min = batch.iter().map(|s| s.advantage).fold(f32::INFINITY, f32::min);
                let adv_max = batch.iter().map(|s| s.advantage).fold(f32::NEG_INFINITY, f32::max);
                let adv_mean = batch.iter().map(|s| s.advantage).sum::<f32>() / batch.len() as f32;
                let adv_var = batch.iter().map(|s| (s.advantage - adv_mean).powi(2)).sum::<f32>() / batch.len() as f32;
                let rew_mean = batch.iter().map(|s| s.step_reward).sum::<f32>() / batch.len() as f32;
                eprintln!("  batch[0] adv: min={adv_min:.3} max={adv_max:.3} mean={adv_mean:.3} std={:.3} | rew_mean={rew_mean:.4}", adv_var.sqrt());
            }

            let (new_model, metrics) = if args.bc {
                train_step_bc(model, &mut optimizer, &batch, &train_config, &device, &mut rng_state)
            } else {
                train_step(model, &mut optimizer, &batch, &train_config, &device, &mut rng_state)
            };
            model = new_model;

            total_metrics.total_loss += metrics.total_loss;
            total_metrics.policy_loss += metrics.policy_loss;
            total_metrics.value_loss += metrics.value_loss;
        }

        let n = args.train_steps as f32;
        total_metrics.total_loss /= n;
        total_metrics.policy_loss /= n;
        total_metrics.value_loss /= n;

        let train_time = t1.elapsed().as_secs_f64();
        eprintln!("  Train: {} steps in {:.1}s, loss={:.4}, policy={:.4}, value={:.4}",
            args.train_steps, train_time,
            total_metrics.total_loss, total_metrics.policy_loss,
            total_metrics.value_loss);

        // 4. Save checkpoint
        let ckpt_path = args.output_dir.join(format!("v6_iter{iteration:04}"));
        if let Err(e) = checkpoint::save_v6(&model, &ckpt_path) {
            eprintln!("  Failed to save checkpoint: {e}");
        } else {
            eprintln!("  Saved: {}.bin", ckpt_path.display());
        }

        // 5. Log
        let step = iteration as u64;
        logger.log_scalar("loss/total", total_metrics.total_loss as f64, step);
        logger.log_scalar("loss/policy", total_metrics.policy_loss as f64, step);
        logger.log_scalar("loss/value", total_metrics.value_loss as f64, step);
        logger.log_scalar("episode/win_rate", win_rate as f64, step);
        logger.log_scalar("episode/count", episodes.len() as f64, step);
        logger.log_scalar("replay/size", replay_buffer.len() as f64, step);
        logger.flush();

        writeln!(log_file, "{},{},{},{:.4},{:.6},{:.6},{:.6}",
            iteration, episodes.len(), replay_buffer.len(), win_rate,
            total_metrics.total_loss, total_metrics.policy_loss,
            total_metrics.value_loss).ok();
        log_file.flush().ok();
    }

    eprintln!("\nDone. Log: {}", log_path.display());
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Episode generation (shells out to self for now)
// ---------------------------------------------------------------------------

#[cfg(feature = "burn-gpu")]
fn generate_episodes(
    scenario_paths: &[std::path::PathBuf],
    output_path: &std::path::Path,
    args: &ImpalaTrainArgs,
    iteration: usize,
    checkpoint: Option<&std::path::Path>,
) -> bool {
    let exe = std::env::current_exe().unwrap_or_else(|_| "cargo".into());

    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("scenario").arg("oracle").arg("transformer-rl").arg("generate");

    for p in scenario_paths {
        cmd.arg(p);
    }

    // Use --burn-v6 when checkpoint available, --policy combined otherwise.
    if let Some(ckpt) = checkpoint {
        cmd.arg("--burn-v6");
        cmd.arg("--burn-checkpoint").arg(ckpt);
    } else {
        cmd.arg("--policy").arg("combined");
    }
    cmd.arg("--output").arg(output_path);
    cmd.arg("--episodes").arg(args.episodes.to_string());
    cmd.arg("--threads").arg(args.threads.to_string());
    cmd.arg("--temperature").arg(args.temperature.to_string());
    cmd.arg("--step-interval").arg(args.step_interval.to_string());
    if args.self_play {
        cmd.arg("--swap-sides");
    }
    if let Some(ref reg) = args.embedding_registry {
        cmd.arg("--embedding-registry").arg(reg);
    }

    let status = cmd.status();
    match status {
        Ok(s) => s.success(),
        Err(e) => {
            eprintln!("Failed to run episode generation: {e}");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Episode loading + training data extraction
// ---------------------------------------------------------------------------

#[cfg(feature = "burn-gpu")]
use super::transformer_rl::{RlEpisode, RlStep};

#[cfg(feature = "burn-gpu")]
fn load_episodes(path: &std::path::Path) -> Vec<RlEpisode> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read episodes from {}: {e}", path.display());
            return Vec::new();
        }
    };
    content.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[cfg(feature = "burn-gpu")]
fn extract_training_samples(
    episodes: &[RlEpisode],
    config: &bevy_game::ai::core::burn_model::training::ImpalaConfig,
    model: &bevy_game::ai::core::burn_model::ActorCriticV6<burn::backend::Autodiff<burn::backend::LibTorch>>,
    device: &burn::backend::libtorch::LibTorchDevice,
) -> Vec<bevy_game::ai::core::burn_model::training::TrainingSample> {
    use bevy_game::ai::core::burn_model::training::{TrainingSample, vtrace_targets, predict_values};
    use burn::module::AutodiffModule;

    let gamma_eff = config.gamma.powi(config.step_interval as i32);

    // First pass: collect all raw samples (without V-trace targets)
    static NEXT_TRAJ_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let mut raw_samples: Vec<(TrainingSample, usize, usize)> = Vec::new();
    let mut traj_boundaries: Vec<(usize, usize, bool)> = Vec::new();

    for (ep_idx, ep) in episodes.iter().enumerate() {
        let mut unit_steps: HashMap<u32, Vec<&RlStep>> = HashMap::new();
        for step in &ep.steps {
            if step.target_move_pos.is_some() || step.move_dir.is_some() {
                unit_steps.entry(step.unit_id).or_default().push(step);
            }
        }

        let is_terminal = ep.outcome == "Victory" || ep.outcome == "Defeat";

        for (_uid, steps) in &unit_steps {
            let mut steps = steps.clone();
            steps.sort_by_key(|s| s.tick);
            if steps.is_empty() { continue; }

            let traj_id = NEXT_TRAJ_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let traj_start = raw_samples.len();
            for (i, step) in steps.iter().enumerate() {
                let entities = step.entities.clone().unwrap_or_default();
                let entity_types = step.entity_types.clone().unwrap_or_default();
                let zones = step.zones.clone().unwrap_or_default();
                let agg = step.aggregate_features.clone().unwrap_or_default();
                let target_move_pos = step.target_move_pos.unwrap_or([0.0, 0.0]);
                let combat_type = step.combat_type.unwrap_or(1);
                let target_idx = step.target_idx.unwrap_or(0);

                let mask = &step.mask;
                let mut combat_mask = vec![false; 10];
                if mask.len() >= 3 {
                    combat_mask[0] = mask[0] || mask[1] || mask[2];
                    combat_mask[1] = true;
                    for j in 0..8.min(mask.len().saturating_sub(3)) {
                        combat_mask[2 + j] = mask[3 + j];
                    }
                }

                let behav_lp_move = step.lp_move.unwrap_or(0.0);
                let behav_lp_combat = step.lp_combat.unwrap_or(0.0)
                    + step.lp_pointer.unwrap_or(0.0);

                raw_samples.push((TrainingSample {
                    entities, entity_types, zones,
                    aggregate_features: agg,
                    corner_tokens: Vec::new(),
                    target_move_pos,
                    behavior_lp_move: behav_lp_move,
                    step_reward: step.step_reward,
                    combat_type, target_idx, combat_mask,
                    behavior_log_prob: behav_lp_combat,
                    value_target: 0.0,
                    advantage: 0.0,
                    traj_id,
                    traj_pos: i,
                    traj_terminal: is_terminal,
                }, ep_idx, i));
            }
            traj_boundaries.push((traj_start, steps.len(), is_terminal));
        }
    }

    if raw_samples.is_empty() {
        return Vec::new();
    }

    // Second pass: predict V(s) for all samples using no-grad model
    let inner_model = model.valid();
    let inner_device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    let all_values = {
        let batch_size = 256;
        let mut values = Vec::with_capacity(raw_samples.len());
        let samples_only: Vec<&TrainingSample> = raw_samples.iter().map(|(s, _, _)| s).collect();
        for chunk in samples_only.chunks(batch_size) {
            let chunk_vec: Vec<TrainingSample> = chunk.iter().map(|s| (*s).clone()).collect();
            let chunk_vals = predict_values(&inner_model, &chunk_vec, &inner_device);
            values.extend(chunk_vals);
        }
        values
    };

    // Third pass: compute V-trace per trajectory
    for &(start, len, is_terminal) in &traj_boundaries {
        let traj_values = &all_values[start..start + len];
        let rewards: Vec<f32> = raw_samples[start..start + len]
            .iter()
            .map(|(s, _, _)| s.step_reward)
            .collect();

        let log_rhos = vec![0.0f32; len]; // On-policy approximation
        let discounts = vec![gamma_eff; len];
        let bootstrap = if is_terminal { 0.0 } else { *traj_values.last().unwrap_or(&0.0) };

        let (vs_targets, advantages) = vtrace_targets(
            &log_rhos, &discounts, &rewards, traj_values,
            bootstrap, config.clip_rho, config.clip_c,
        );

        for i in 0..len {
            raw_samples[start + i].0.value_target = vs_targets[i];
            raw_samples[start + i].0.advantage = advantages[i];
        }
    }

    // Return raw (unnormalized) advantages — normalization happens per-minibatch in compute_loss
    raw_samples.into_iter().map(|(s, _, _)| s).collect()
}

