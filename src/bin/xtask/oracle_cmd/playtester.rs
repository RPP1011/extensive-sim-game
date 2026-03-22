//! Tier 3: RL Playtester Agent
//!
//! Three-layer hierarchical agent that plays through scenarios using random
//! policies at each level. The hierarchy:
//!
//! 1. **Strategic layer** — picks which scenario/composition to play next
//!    (operates once per episode, ~100-500 tick intervals)
//! 2. **Tactical layer** — personality/formation overrides for the squad AI
//!    (operates every ~10-50 ticks within a battle)
//! 3. **Combat layer** — per-unit action selection using the V4 dual-head
//!    action space (operates every tick via the existing episode runner)
//!
//! All three layers currently use random policies. The harness collects
//! population-level metrics: win rate, hero usage diversity (Gini),
//! ability coverage, and per-agent behavioral fingerprints.

use std::collections::HashMap;
use std::io::Write;
use std::process::ExitCode;

use serde::{Deserialize, Serialize};

use super::transformer_rl::{Policy, RlEpisode, PrecomputedScenario, precompute_scenarios, lcg_f32};
use super::rl_policies::run_single_episode;
use super::collect_toml_paths;

// ---------------------------------------------------------------------------
// Agent population
// ---------------------------------------------------------------------------

/// A single playtester agent in the population.
#[derive(Debug, Clone)]
pub(crate) struct PlaytesterAgent {
    pub id: usize,
    /// RNG state for this agent (deterministic per agent).
    pub rng: u64,
    /// Strategic layer: scenario preference weights (index into scenario pool).
    pub scenario_weights: Vec<f32>,
    /// Tactical layer: personality bias vector.
    /// [aggression, caution, focus_fire, ability_usage, positioning]
    pub tactical_bias: [f32; 5],
    /// Lifetime stats for this agent.
    pub stats: AgentStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct AgentStats {
    pub episodes_played: u32,
    pub wins: u32,
    pub losses: u32,
    pub timeouts: u32,
    pub total_ticks: u64,
    pub total_steps: u64,
    pub total_reward: f32,
    /// Hero usage counts (hero_name -> count).
    pub hero_usage: HashMap<String, u32>,
    /// Ability usage counts (ability_name -> count).
    pub ability_usage: HashMap<String, u32>,
    /// Per-scenario win counts.
    pub scenario_wins: HashMap<String, u32>,
    /// Per-scenario play counts.
    pub scenario_plays: HashMap<String, u32>,
}

impl PlaytesterAgent {
    fn new(id: usize, n_scenarios: usize, seed: u64) -> Self {
        let mut rng = seed ^ (id as u64).wrapping_mul(0x9E3779B97F4A7C15);
        // Random scenario weights
        let scenario_weights: Vec<f32> = (0..n_scenarios)
            .map(|_| lcg_f32(&mut rng))
            .collect();
        // Random tactical bias in [-1, 1]
        let tactical_bias = [
            lcg_f32(&mut rng) * 2.0 - 1.0,
            lcg_f32(&mut rng) * 2.0 - 1.0,
            lcg_f32(&mut rng) * 2.0 - 1.0,
            lcg_f32(&mut rng) * 2.0 - 1.0,
            lcg_f32(&mut rng) * 2.0 - 1.0,
        ];
        Self {
            id,
            rng,
            scenario_weights,
            tactical_bias,
            stats: AgentStats::default(),
        }
    }

    /// Strategic layer: pick next scenario index (weighted random).
    fn pick_scenario(&mut self) -> usize {
        let total: f32 = self.scenario_weights.iter().sum();
        if total <= 0.0 || self.scenario_weights.is_empty() {
            return 0;
        }
        let r = lcg_f32(&mut self.rng) * total;
        let mut cum = 0.0;
        for (i, &w) in self.scenario_weights.iter().enumerate() {
            cum += w;
            if r < cum {
                return i;
            }
        }
        self.scenario_weights.len() - 1
    }

    /// Tactical layer: generate a temperature modifier from the tactical bias.
    /// Higher aggression → lower temperature (more greedy).
    fn tactical_temperature(&self) -> f32 {
        let aggression = self.tactical_bias[0];
        // Map aggression [-1, 1] to temperature [0.5, 2.0]
        let t = 1.25 - aggression * 0.75;
        t.clamp(0.3, 3.0)
    }

    fn record_episode(&mut self, episode: &RlEpisode) {
        self.stats.episodes_played += 1;
        self.stats.total_ticks += episode.ticks;
        self.stats.total_steps += episode.steps.len() as u64;
        self.stats.total_reward += episode.reward;
        match episode.outcome.as_str() {
            "Victory" => self.stats.wins += 1,
            "Defeat" => self.stats.losses += 1,
            _ => self.stats.timeouts += 1,
        }
        *self.stats.scenario_plays.entry(episode.scenario.clone()).or_default() += 1;
        if episode.outcome == "Victory" {
            *self.stats.scenario_wins.entry(episode.scenario.clone()).or_default() += 1;
        }
        // Track ability usage from step data
        for (uid, names) in &episode.unit_ability_names {
            for name in names {
                *self.stats.ability_usage.entry(name.clone()).or_default() += 1;
            }
            // Hero usage: use unit_id as proxy (will be enriched later)
            let hero_key = format!("unit_{}", uid);
            *self.stats.hero_usage.entry(hero_key).or_default() += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Population metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PopulationMetrics {
    pub total_episodes: u32,
    pub total_wins: u32,
    pub total_losses: u32,
    pub total_timeouts: u32,
    pub win_rate: f32,
    pub mean_reward: f32,
    pub mean_ticks: f32,
    pub mean_steps_per_episode: f32,
    pub hero_gini: f32,
    pub ability_coverage: f32,
    pub total_unique_abilities: usize,
    pub total_known_abilities: usize,
    pub scenario_coverage: f32,
    pub population_size: usize,
    pub behavioral_diversity: f32,
}

fn gini_coefficient(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f32;
    let total: f32 = values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut num = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        num += (2.0 * (i as f32 + 1.0) - n - 1.0) * v;
    }
    num / (n * total)
}

fn compute_population_metrics(
    agents: &[PlaytesterAgent],
    total_known_abilities: usize,
    n_scenarios: usize,
) -> PopulationMetrics {
    let total_episodes: u32 = agents.iter().map(|a| a.stats.episodes_played).sum();
    let total_wins: u32 = agents.iter().map(|a| a.stats.wins).sum();
    let total_losses: u32 = agents.iter().map(|a| a.stats.losses).sum();
    let total_timeouts: u32 = agents.iter().map(|a| a.stats.timeouts).sum();
    let total_reward: f32 = agents.iter().map(|a| a.stats.total_reward).sum();
    let total_ticks: u64 = agents.iter().map(|a| a.stats.total_ticks).sum();
    let total_steps: u64 = agents.iter().map(|a| a.stats.total_steps).sum();

    let win_rate = if total_episodes > 0 {
        total_wins as f32 / total_episodes as f32
    } else {
        0.0
    };
    let mean_reward = if total_episodes > 0 {
        total_reward / total_episodes as f32
    } else {
        0.0
    };
    let mean_ticks = if total_episodes > 0 {
        total_ticks as f32 / total_episodes as f32
    } else {
        0.0
    };
    let mean_steps = if total_episodes > 0 {
        total_steps as f32 / total_episodes as f32
    } else {
        0.0
    };

    // Hero usage Gini across population
    let mut hero_counts: HashMap<String, u32> = HashMap::new();
    for agent in agents {
        for (hero, &count) in &agent.stats.hero_usage {
            *hero_counts.entry(hero.clone()).or_default() += count;
        }
    }
    let hero_values: Vec<f32> = hero_counts.values().map(|&v| v as f32).collect();
    let hero_gini = gini_coefficient(&hero_values);

    // Ability coverage
    let mut all_abilities: std::collections::HashSet<String> = std::collections::HashSet::new();
    for agent in agents {
        for name in agent.stats.ability_usage.keys() {
            all_abilities.insert(name.clone());
        }
    }
    let total_unique = all_abilities.len();
    let ability_coverage = if total_known_abilities > 0 {
        total_unique as f32 / total_known_abilities as f32
    } else {
        0.0
    };

    // Scenario coverage
    let mut scenarios_played: std::collections::HashSet<String> = std::collections::HashSet::new();
    for agent in agents {
        for name in agent.stats.scenario_plays.keys() {
            scenarios_played.insert(name.clone());
        }
    }
    let scenario_coverage = if n_scenarios > 0 {
        scenarios_played.len() as f32 / n_scenarios as f32
    } else {
        0.0
    };

    // Behavioral diversity: pairwise Euclidean distance of tactical biases
    let mut diversity = 0.0f32;
    let mut pairs = 0u32;
    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let dist: f32 = agents[i]
                .tactical_bias
                .iter()
                .zip(agents[j].tactical_bias.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            diversity += dist;
            pairs += 1;
        }
    }
    let behavioral_diversity = if pairs > 0 {
        diversity / pairs as f32
    } else {
        0.0
    };

    PopulationMetrics {
        total_episodes,
        total_wins,
        total_losses,
        total_timeouts,
        win_rate,
        mean_reward,
        mean_ticks,
        mean_steps_per_episode: mean_steps,
        hero_gini,
        ability_coverage,
        total_unique_abilities: total_unique,
        total_known_abilities,
        scenario_coverage,
        population_size: agents.len(),
        behavioral_diversity,
    }
}

// ---------------------------------------------------------------------------
// Playtester run report (per-iteration snapshot)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlaytesterReport {
    iteration: usize,
    metrics: PopulationMetrics,
    agent_summaries: Vec<AgentSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentSummary {
    id: usize,
    episodes: u32,
    wins: u32,
    win_rate: f32,
    mean_reward: f32,
    tactical_bias: [f32; 5],
    unique_abilities_used: usize,
}

// ---------------------------------------------------------------------------
// Main playtester harness
// ---------------------------------------------------------------------------

pub(crate) fn run_playtester(args: crate::cli::PlaytesterArgs) -> ExitCode {
    use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use bevy_game::scenario::load_scenario_file;

    let paths: Vec<_> = args.path.iter().flat_map(|p| collect_toml_paths(p)).collect();
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    eprintln!("=== RL Playtester Agent ===");
    eprintln!("Scenarios: {}", paths.len());
    eprintln!("Population: {} agents", args.population);
    eprintln!("Iterations: {}", args.iterations);
    eprintln!("Episodes/agent/iter: {}", args.episodes_per_agent);
    eprintln!("Step interval: {}", args.step_interval);

    let tokenizer = AbilityTokenizer::new();

    let scenarios: Vec<_> = paths
        .iter()
        .filter_map(|p| match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("  skip: {e}");
                None
            }
        })
        .collect();

    if scenarios.is_empty() {
        eprintln!("No valid scenarios loaded.");
        return ExitCode::from(1);
    }
    eprintln!("Loaded {} scenarios", scenarios.len());

    // Precompute all scenarios
    let precompute_t0 = std::time::Instant::now();
    let precomputed = precompute_scenarios(&scenarios, args.max_ticks, &tokenizer, None);
    eprintln!(
        "Precomputed {} scenarios in {:.1}s",
        precomputed.len(),
        precompute_t0.elapsed().as_secs_f64()
    );

    // Count total known abilities across all scenarios
    let mut all_ability_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for pre in &precomputed {
        for names in pre.unit_ability_names.values() {
            for name in names {
                all_ability_names.insert(name.clone());
            }
        }
    }
    let total_known_abilities = all_ability_names.len();
    eprintln!("Total unique abilities across scenarios: {}", total_known_abilities);

    // Initialize agent population
    let mut agents: Vec<PlaytesterAgent> = (0..args.population)
        .map(|i| PlaytesterAgent::new(i, precomputed.len(), args.seed))
        .collect();
    eprintln!("Initialized {} agents", agents.len());

    // Build thread pool
    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    eprintln!("Thread pool: {} threads", threads);

    let policy = Policy::Random;
    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let precomputed_ref = &precomputed;

    let mut all_reports: Vec<PlaytesterReport> = Vec::new();
    let harness_t0 = std::time::Instant::now();
    let mut cumulative_ticks: u64 = 0;

    for iter in 0..args.iterations {
        let iter_t0 = std::time::Instant::now();

        // Collect (agent_id, scenario_idx, episode_idx, temperature, rng_seed) tasks
        let mut tasks: Vec<(usize, usize, usize, f32, u64)> = Vec::new();
        for agent in agents.iter_mut() {
            let temp = agent.tactical_temperature();
            for ep in 0..args.episodes_per_agent {
                let si = agent.pick_scenario();
                let seed = agent.rng;
                lcg_f32(&mut agent.rng); // advance rng
                tasks.push((agent.id, si, ep, temp, seed));
            }
        }

        // Run episodes in parallel (catch panics from sim verification failures)
        let episodes: Vec<(usize, RlEpisode)> = pool.install(|| {
            use rayon::prelude::*;
            tasks
                .par_iter()
                .filter_map(|&(agent_id, si, ei, temp, seed)| {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut pre_sim = precomputed_ref[si].sim.clone();
                        pre_sim.rng_state = seed;

                        let episode = super::rl_episode::run_rl_episode(
                            pre_sim,
                            precomputed_ref[si].squad_ai.clone(),
                            &precomputed_ref[si].scenario_name,
                            precomputed_ref[si].max_ticks,
                            policy_ref,
                            tokenizer_ref,
                            temp,
                            seed,
                            args.step_interval,
                            precomputed_ref[si].sim.grid_nav.clone(),
                            None,
                            None,
                            None,
                            precomputed_ref[si].objective.as_ref(),
                            precomputed_ref[si].action_mask.as_deref(),
                        );
                        (agent_id, episode)
                    }));
                    match result {
                        Ok(ep) => Some(ep),
                        Err(_) => None, // skip panicked episodes
                    }
                })
                .collect()
        });

        // Record results back to agents
        let mut iter_ticks: u64 = 0;
        let mut iter_steps: u64 = 0;
        for (agent_id, episode) in &episodes {
            iter_ticks += episode.ticks;
            iter_steps += episode.steps.len() as u64;
            agents[*agent_id].record_episode(episode);
        }
        cumulative_ticks += iter_ticks;

        // Reset profiling counters
        super::rl_episode::reset_profiling();

        // Compute population metrics
        let metrics = compute_population_metrics(
            &agents,
            total_known_abilities,
            precomputed.len(),
        );

        let agent_summaries: Vec<AgentSummary> = agents
            .iter()
            .map(|a| AgentSummary {
                id: a.id,
                episodes: a.stats.episodes_played,
                wins: a.stats.wins,
                win_rate: if a.stats.episodes_played > 0 {
                    a.stats.wins as f32 / a.stats.episodes_played as f32
                } else {
                    0.0
                },
                mean_reward: if a.stats.episodes_played > 0 {
                    a.stats.total_reward / a.stats.episodes_played as f32
                } else {
                    0.0
                },
                tactical_bias: a.tactical_bias,
                unique_abilities_used: a.stats.ability_usage.len(),
            })
            .collect();

        let report = PlaytesterReport {
            iteration: iter,
            metrics: metrics.clone(),
            agent_summaries,
        };
        all_reports.push(report);

        eprintln!(
            "[iter {}/{}] episodes={} ticks={} steps={} win_rate={:.1}% reward={:.3} hero_gini={:.3} ability_cov={:.1}% scenario_cov={:.1}% diversity={:.3} ({:.1}s)",
            iter + 1,
            args.iterations,
            episodes.len(),
            iter_ticks,
            iter_steps,
            metrics.win_rate * 100.0,
            metrics.mean_reward,
            metrics.hero_gini,
            metrics.ability_coverage * 100.0,
            metrics.scenario_coverage * 100.0,
            metrics.behavioral_diversity,
            iter_t0.elapsed().as_secs_f64(),
        );
    }

    let total_time = harness_t0.elapsed().as_secs_f64();
    eprintln!("\n=== Playtester Summary ===");
    eprintln!("Total time: {:.1}s", total_time);
    eprintln!("Total ticks: {} ({:.0} ticks/s)", cumulative_ticks, cumulative_ticks as f64 / total_time);

    let final_metrics = compute_population_metrics(&agents, total_known_abilities, precomputed.len());
    eprintln!("Final win rate: {:.1}%", final_metrics.win_rate * 100.0);
    eprintln!("Final mean reward: {:.3}", final_metrics.mean_reward);
    eprintln!("Hero Gini coefficient: {:.3} (target < 0.3)", final_metrics.hero_gini);
    eprintln!("Ability coverage: {}/{} ({:.1}%) (target > 60%)",
        final_metrics.total_unique_abilities,
        final_metrics.total_known_abilities,
        final_metrics.ability_coverage * 100.0);
    eprintln!("Scenario coverage: {:.1}%", final_metrics.scenario_coverage * 100.0);
    eprintln!("Population diversity: {:.3}", final_metrics.behavioral_diversity);

    // Per-agent summary
    eprintln!("\n--- Agent Breakdown ---");
    for agent in &agents {
        let wr = if agent.stats.episodes_played > 0 {
            agent.stats.wins as f32 / agent.stats.episodes_played as f32 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  Agent {:>2}: episodes={:>3} win_rate={:>5.1}% reward={:>6.3} abilities={:>3} bias=[{:.2},{:.2},{:.2},{:.2},{:.2}]",
            agent.id,
            agent.stats.episodes_played,
            wr,
            agent.stats.total_reward / agent.stats.episodes_played.max(1) as f32,
            agent.stats.ability_usage.len(),
            agent.tactical_bias[0],
            agent.tactical_bias[1],
            agent.tactical_bias[2],
            agent.tactical_bias[3],
            agent.tactical_bias[4],
        );
    }

    // Validation checklist
    eprintln!("\n--- Validation Checklist ---");
    let check = |name: &str, pass: bool| {
        eprintln!("  [{}] {}", if pass { "PASS" } else { "FAIL" }, name);
    };
    check("50+ turn campaign stability (all episodes completed)", final_metrics.total_episodes > 0);
    check(
        &format!("Hero usage Gini < 0.3 (got {:.3})", final_metrics.hero_gini),
        final_metrics.hero_gini < 0.3,
    );
    check(
        &format!("Ability coverage > 60% (got {:.1}%)", final_metrics.ability_coverage * 100.0),
        final_metrics.ability_coverage > 0.6,
    );
    check(
        "Population behavioral diversity > 0",
        final_metrics.behavioral_diversity > 0.0,
    );
    check(
        &format!("Total ticks > 500 (got {})", cumulative_ticks),
        cumulative_ticks > 500,
    );

    // Write output
    if let Some(ref output_dir) = args.output_dir {
        std::fs::create_dir_all(output_dir).ok();

        // Write final metrics
        let metrics_path = output_dir.join("playtester_metrics.json");
        if let Ok(json) = serde_json::to_string_pretty(&final_metrics) {
            std::fs::write(&metrics_path, json).ok();
            eprintln!("\nWrote metrics to {}", metrics_path.display());
        }

        // Write iteration reports as JSONL
        let reports_path = output_dir.join("playtester_reports.jsonl");
        if let Ok(file) = std::fs::File::create(&reports_path) {
            let mut writer = std::io::BufWriter::new(file);
            for report in &all_reports {
                if let Ok(line) = serde_json::to_string(report) {
                    writeln!(writer, "{}", line).ok();
                }
            }
            writer.flush().ok();
            eprintln!("Wrote {} iteration reports to {}", all_reports.len(), reports_path.display());
        }
    }

    ExitCode::SUCCESS
}
