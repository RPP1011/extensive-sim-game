//! Combat AI trainer using rl4burn PPO.
//!
//! Trains an actor-critic MLP to control a single hero unit in the tactical
//! combat simulator, using Proximal Policy Optimization with action masking.

mod env;
mod model;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use rand::SeedableRng;

use rl4burn::nn::dist::ActionDist;
use rl4burn::{masked_ppo_collect, masked_ppo_update, PpoConfig, SyncVecEnv};

use env::{load_hero_templates_from_dir, CombatEnv};
use model::CombatPolicy;

type TrainB = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Load hero templates for training diversity
    // -----------------------------------------------------------------------
    let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let hero_dir = project_root.join("assets/hero_templates");
    let lol_dir = project_root.join("assets/lol_heroes");

    let mut hero_templates = load_hero_templates_from_dir(&hero_dir);
    let lol_templates = load_hero_templates_from_dir(&lol_dir);
    hero_templates.extend(lol_templates.clone());

    let enemy_templates = hero_templates.clone();
    // If no templates found, use a minimal fallback
    if hero_templates.is_empty() {
        eprintln!("Warning: no hero templates found. Using minimal fallback.");
        // The env will just create empty sims; training won't be useful
        // but the binary will still compile and run.
        return;
    }

    eprintln!(
        "Loaded {} hero templates, {} enemy templates",
        hero_templates.len(),
        enemy_templates.len()
    );

    // -----------------------------------------------------------------------
    // Hyperparameters
    // -----------------------------------------------------------------------
    let n_envs = 32;
    let n_steps = 128; // steps per env per rollout
    let total_steps = 10_000_000;
    let checkpoint_interval = 100; // iterations between checkpoints

    // -----------------------------------------------------------------------
    // Create vectorized environments
    // -----------------------------------------------------------------------
    let envs: Vec<CombatEnv> = (0..n_envs)
        .map(|i| {
            CombatEnv::new(
                hero_templates.clone(),
                enemy_templates.clone(),
                1000 + i as u64,
            )
        })
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // -----------------------------------------------------------------------
    // Model and optimizer
    // -----------------------------------------------------------------------
    let mut model: CombatPolicy<TrainB> = CombatPolicy::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new()
        .with_n_steps(n_steps)
        .with_minibatch_size(256)
        .with_update_epochs(4)
        .with_ent_coef(0.01)
        .with_clip_eps(0.2)
        .with_lr(3e-4);

    let action_dist = ActionDist::Discrete(14);
    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent_rewards: Vec<f32> = Vec::new();
    let window = 100;

    eprintln!("=== Combat AI Training (PPO + Action Masking) ===");
    eprintln!();
    eprintln!(
        "Training for {} steps ({} envs, {} iters, {} steps/iter)",
        total_steps, n_envs, n_iters, steps_per_iter,
    );
    eprintln!(
        "Obs dim: 210, Action space: Discrete(14), Minibatch: {}",
        config.minibatch_size,
    );
    eprintln!("{:-<80}", "");

    for iter in 0..n_iters {
        // Linear LR annealing with floor (don't go below 1e-5)
        let lr = (config.lr * (1.0 - iter as f64 / n_iters as f64)).max(1e-5);

        // Collect rollout with action masking
        let rollout = masked_ppo_collect::<NdArray, _, _>(
            &model.valid(),
            &mut vec_env,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        // Track episode returns
        for &ret in &rollout.episode_returns {
            recent_rewards.push(ret);
        }
        if recent_rewards.len() > window {
            recent_rewards = recent_rewards[recent_rewards.len() - window..].to_vec();
        }

        // PPO update
        let stats;
        (model, stats) = masked_ppo_update(
            model,
            &mut optim,
            &rollout,
            &action_dist,
            &config,
            lr,
            &device,
            &mut rng,
        );

        // Logging
        if (iter + 1) % 5 == 0 && !recent_rewards.is_empty() {
            let avg_ret: f32 =
                recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
            let n_eps = rollout.episode_returns.len();

            eprintln!(
                "step {:>7} | ret {:>+7.2} | eps {:>3} | ploss {:.4} | vloss {:.4} | ent {:.4} | kl {:.4} | lr {:.2e}",
                (iter + 1) * steps_per_iter,
                avg_ret,
                n_eps,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.approx_kl,
                lr,
            );
        }

        // Checkpoint
        if (iter + 1) % checkpoint_interval == 0 {
            let step = (iter + 1) * steps_per_iter;
            eprintln!("  [checkpoint at step {}]", step);
            // TODO: save model weights to disk
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    eprintln!("{:-<80}", "");
    eprintln!("Training complete.");
    if !recent_rewards.is_empty() {
        let avg: f32 = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
        eprintln!(
            "Last {} episodes: avg return {:+.2}",
            recent_rewards.len(),
            avg,
        );
    }
}
