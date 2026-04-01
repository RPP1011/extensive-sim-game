//! Episode generation CLI for transformer RL.

use std::io::Write;
use std::process::ExitCode;

use rayon::prelude::*;

use super::collect_toml_paths;
use super::transformer_rl::{Policy, RlEpisode};
use super::rl_policies::run_single_episode;

pub(crate) fn run_generate(args: crate::cli::TransformerRlGenerateArgs) -> ExitCode {
    use game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use game::scenario::load_scenario_file;

    let policy = if args.random_policy {
        Policy::Random
    } else if args.burn_v6 {
        #[cfg(not(feature = "burn-gpu"))]
        {
            eprintln!("--burn-v6 requires building with --features burn-gpu");
            return ExitCode::from(1);
        }
        #[cfg(feature = "burn-gpu")]
        {
            use game::ai::core::burn_model::{ActorCriticV6Config, inference_v6::BurnInferenceClientV6, checkpoint};
            use burn::backend::libtorch::LibTorchDevice;
            let device = LibTorchDevice::Cuda(0);
            let config = ActorCriticV6Config {
                vocab_size: 256,
                d_model: 128,
                d_ff: 256,
                n_heads: 8,
                n_layers: 4,
                entity_encoder_layers: 4,
                external_cls_dim: 0,
                h_dim: 256,
                n_latents: 12,
                n_latent_blocks: 2,
            };
            let model = if let Some(ref ckpt_path) = args.burn_checkpoint {
                match checkpoint::load_v6::<burn::backend::LibTorch>(&config, ckpt_path, &device) {
                    Ok(m) => { eprintln!("Loaded V6 checkpoint: {}", ckpt_path.display()); m }
                    Err(e) => { eprintln!("Failed to load V6 checkpoint: {e}"); return ExitCode::from(1); }
                }
            } else {
                config.init::<burn::backend::LibTorch>(&device)
            };
            let client = BurnInferenceClientV6::new(model, device, 1024, 1);
            eprintln!("Burn V6 GPU inference: d=128, h=256, K=12, device=CUDA:0");
            Policy::BurnServerV6(client)
        }
    } else if args.burn {
        #[cfg(not(feature = "burn-gpu"))]
        {
            eprintln!("--burn requires building with --features burn-gpu");
            return ExitCode::from(1);
        }
        #[cfg(feature = "burn-gpu")]
        {
            use game::ai::core::burn_model::{ActorCriticV5Config, inference::BurnInferenceClient};
            use burn::backend::libtorch::LibTorchDevice;
            let device = LibTorchDevice::Cuda(0);
            let model = ActorCriticV5Config {
                vocab_size: 256,
                d_model: 128,
                d_ff: 256,
                n_heads: 8,
                n_layers: 4,
                entity_encoder_layers: 4,
                external_cls_dim: 0,
                h_dim: 64,
            }
            .init::<burn::backend::LibTorch>(&device);
            let client = BurnInferenceClient::new(model, device, 1024, 1);
            eprintln!("Burn GPU inference: d=128, h=64, device=CUDA:0");
            Policy::BurnServer(client)
        }
    } else if args.policy == "combined" {
        Policy::Combined
    } else {
        let weights_path = match &args.weights {
            Some(p) => p,
            None => { eprintln!("--weights is required for transformer policy"); return ExitCode::from(1); }
        };
        match Policy::load(weights_path) {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to load weights: {e}"); return ExitCode::from(1); }
        }
    };
    let policy_type = match &policy {
        Policy::ActorCriticV5(_) => "actor-critic-v5 (d=128, aggregate)",
        #[cfg(feature = "burn-gpu")]
        Policy::BurnServer(_) => "burn-gpu-v5 (in-process)",
        #[cfg(feature = "burn-gpu")]
        Policy::BurnServerV6(_) => "burn-gpu-v6 (spatial+latent)",
        Policy::Combined => "combined (squad AI)",
        Policy::Random => "random",
    };

    let tokenizer = AbilityTokenizer::new();

    let paths: Vec<_> = args.path.iter().flat_map(|p| collect_toml_paths(p)).collect();
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    eprintln!("Generating RL episodes: {} scenarios x {} episodes, temp={:.2}, policy={}",
        paths.len(), args.episodes, args.temperature, policy_type);

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("{e}"); None }
        }
    }).collect();

    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Load embedding registry if provided
    let registry = if let Some(ref reg_path) = args.embedding_registry {
        match game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded embedding registry: {} abilities, hash={}",
                    r.len(), r.model_hash);
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    // Load enemy policy for self-play
    let enemy_policy: Option<Policy> = if let Some(ref ew_path) = args.enemy_weights {
        match Policy::load(ew_path) {
            Ok(p) => {
                eprintln!("Loaded enemy policy from {}", ew_path.display());
                Some(p)
            }
            Err(e) => { eprintln!("Failed to load enemy policy: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };
    let enemy_registry = if let Some(ref reg_path) = args.enemy_registry {
        match game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded enemy embedding registry: {} abilities", r.len());
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load enemy registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let registry_ref = registry.as_ref();
    let enemy_policy_ref = enemy_policy.as_ref();
    let enemy_registry_ref = enemy_registry.as_ref();
    let step_interval = args.step_interval;
    let temperature = args.temperature;
    let max_ticks_override = args.max_ticks;

    // --swap-sides: duplicate scenarios with hero/enemy templates swapped
    let mut scenarios = scenarios;
    if args.swap_sides {
        let n = scenarios.len();
        let mut swapped = Vec::with_capacity(n);
        for sf in &scenarios {
            let mut cfg = sf.scenario.clone();
            std::mem::swap(&mut cfg.hero_templates, &mut cfg.enemy_hero_templates);
            cfg.name = format!("{}_swapped", cfg.name);
            std::mem::swap(&mut cfg.hero_count, &mut cfg.enemy_count);
            swapped.push(game::scenario::ScenarioFile { scenario: cfg, assert: None });
        }
        scenarios.extend(swapped);
        eprintln!("Swap-sides: {} original + {} swapped = {} total scenarios",
            n, n, scenarios.len());
    }

    let episode_tasks: Vec<(usize, usize)> = scenarios.iter().enumerate()
        .flat_map(|(si, _)| (0..args.episodes as usize).map(move |ei| (si, ei)))
        .collect();

    // Precompute all scenarios once (parse DSL, generate rooms, tokenize abilities)
    let precompute_t0 = std::time::Instant::now();
    let precomputed = std::sync::Arc::new(super::transformer_rl::precompute_scenarios(
        &scenarios, max_ticks_override, tokenizer_ref, registry_ref,
    ));
    eprintln!("Precomputed {} scenarios in {:.1}s",
        precomputed.len(), precompute_t0.elapsed().as_secs_f64());

    let ep_t0 = std::time::Instant::now();
    let episodes: Vec<RlEpisode> = pool.install(|| {
        episode_tasks.par_iter().map(|&(si, ei)| {
            run_single_episode(
                &precomputed[si], si, ei, true,
                policy_ref, tokenizer_ref, temperature, step_interval,
                registry_ref,
                enemy_policy_ref, enemy_registry_ref,
            )
        }).collect()
    });
    eprintln!("Episode generation: {:.1}s ({} threads)",
        ep_t0.elapsed().as_secs_f64(), threads);

    // Print per-function profiling
    {
        use super::rl_episode::*;
        use std::sync::atomic::Ordering;
        let ticks = PROF_TICKS.load(Ordering::Relaxed).max(1);
        let ms = |v: &std::sync::atomic::AtomicU64| v.load(Ordering::Relaxed) as f64 / 1e6;
        let total = ms(&PROF_INTENTS_NS) + ms(&PROF_POLICY_NS) + ms(&PROF_STEP_NS) + ms(&PROF_TERRAIN_NS);
        eprintln!("\n--- Episode Profiling ({ticks} ticks, {threads} threads) ---");
        eprintln!("  intents:   {:>8.1}ms ({:.1}%)", ms(&PROF_INTENTS_NS), ms(&PROF_INTENTS_NS) / total * 100.0);
        eprintln!("  policy:    {:>8.1}ms ({:.1}%)", ms(&PROF_POLICY_NS), ms(&PROF_POLICY_NS) / total * 100.0);
        eprintln!("  step:      {:>8.1}ms ({:.1}%)", ms(&PROF_STEP_NS), ms(&PROF_STEP_NS) / total * 100.0);
        eprintln!("  terrain:   {:>8.1}ms ({:.1}%)", ms(&PROF_TERRAIN_NS), ms(&PROF_TERRAIN_NS) / total * 100.0);
        eprintln!("  total:     {:>8.1}ms (sum of thread-time)", total);
        eprintln!("-----------------------------------------------");
    }

    let wins = episodes.iter().filter(|e| e.outcome == "Victory").count();
    let losses = episodes.iter().filter(|e| e.outcome == "Defeat").count();
    let timeouts = episodes.iter().filter(|e| e.outcome == "Timeout").count();
    let total_steps: usize = episodes.iter().map(|e| e.steps.len()).sum();
    let mean_reward: f32 = episodes.iter().map(|e| e.reward).sum::<f32>() / episodes.len().max(1) as f32;

    eprintln!("Episodes: {}  Wins: {}  Losses: {}  Timeouts: {}  Win rate: {:.1}%",
        episodes.len(), wins, losses, timeouts,
        wins as f64 / episodes.len().max(1) as f64 * 100.0);
    eprintln!("Total steps: {}  Mean reward: {:.3}", total_steps, mean_reward);

    let file = std::fs::File::create(&args.output).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    for ep in &episodes {
        let line = serde_json::to_string(ep).unwrap();
        writeln!(writer, "{}", line).unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Wrote {} episodes to {}", episodes.len(), args.output.display());

    ExitCode::SUCCESS
}
