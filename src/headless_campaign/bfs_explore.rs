//! BFS state-space exploration with cluster-and-prune.
//!
//! 1. Start from a set of root states
//! 2. At each root, try every ACTION TYPE (grouped — not every target variant)
//! 3. Step forward N ticks per branch to let the action play out
//! 4. Record (root_tokens, action_type, leaf_tokens, leaf_value)
//! 5. Cluster the leaf states by feature similarity
//! 6. Pick the median state per cluster as next-wave roots
//! 7. Repeat for K waves
//!
//! This gives full action coverage without exponential blowup.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::actions::*;
use super::batch::heuristic_policy;
use super::config::CampaignConfig;
use super::heuristic_bc::action_type_name;
use super::state::*;
use super::step::step_campaign;
use super::tokens::EntityToken;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single BFS exploration sample: "from this state, taking this action type
/// leads to this outcome after N ticks."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BfsSample {
    /// Tokens at the root state (before action).
    pub root_tokens: Vec<EntityToken>,
    /// Action type taken.
    pub action_type: String,
    /// Specific action (for reconstruction).
    pub action_detail: String,
    /// Tokens at the leaf state (after action + N ticks).
    pub leaf_tokens: Vec<EntityToken>,
    /// Heuristic value estimate at the leaf.
    pub leaf_value: f32,
    /// Campaign outcome if terminal, else None.
    pub leaf_outcome: Option<String>,
    /// Root tick.
    pub root_tick: u64,
    /// Leaf tick.
    pub leaf_tick: u64,
    /// Seed of the originating campaign.
    pub seed: u64,
    /// Wave number in the BFS.
    pub wave: u32,
    /// Which cluster this leaf was assigned to.
    pub cluster_id: u32,
}

/// A state snapshot used as a BFS root.
#[derive(Clone)]
struct RootState {
    state: CampaignState,
    seed: u64,
    wave: u32,
}

/// Configuration for BFS exploration.
#[derive(Clone, Debug)]
pub struct BfsConfig {
    /// Maximum BFS waves (0 = unlimited, run until all leaves are terminal).
    pub max_waves: u32,
    /// Ticks to advance after each action (let it play out).
    pub ticks_per_branch: u64,
    /// Number of clusters per wave (controls width).
    pub clusters_per_wave: usize,
    /// Number of initial root states to generate.
    pub initial_roots: usize,
    /// Max ticks for initial heuristic trajectory.
    pub trajectory_max_ticks: u64,
    /// Interval for sampling roots from heuristic trajectory.
    pub root_sample_interval: u64,
    /// Campaign config.
    pub campaign_config: CampaignConfig,
    /// Base seed.
    pub base_seed: u64,
    /// Number of threads.
    pub threads: usize,
    /// Output JSONL path.
    pub output_path: String,
    /// Optional LLM config for content generation.
    pub llm_config: Option<super::llm::LlmConfig>,
    /// Optional VAE model for instant content generation.
    pub vae_model: Option<std::sync::Arc<super::vae_inference::ContentVaeWeights>>,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_waves: 0, // 0 = run until completion
            ticks_per_branch: 200, // ~20s game time per branch
            clusters_per_wave: 20,
            initial_roots: 50,
            trajectory_max_ticks: 15000,
            root_sample_interval: 300,
            campaign_config: CampaignConfig::default(),
            base_seed: 2026,
            threads: 0,
            output_path: "generated/bfs_explore.jsonl".into(),
            llm_config: None,
            vae_model: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Action grouping
// ---------------------------------------------------------------------------

/// Get one representative action per action type from valid actions.
/// This reduces branching from ~40 to ~10-12.
fn group_actions(valid_actions: &[CampaignAction]) -> Vec<(String, CampaignAction)> {
    let mut seen: HashMap<String, CampaignAction> = HashMap::new();
    for action in valid_actions {
        let type_name = action_type_name(action);
        // Keep the first instance of each type (or best by some heuristic)
        seen.entry(type_name).or_insert_with(|| action.clone());
    }
    seen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Leaf value estimation
// ---------------------------------------------------------------------------

/// Quick heuristic value for a campaign state.
/// Higher = better. Combines multiple signals.
fn estimate_value(state: &CampaignState) -> f32 {
    let alive = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count() as f32;
    let gold = (state.guild.gold / 100.0).min(5.0);
    let rep = state.guild.reputation / 100.0;
    let progress = state.overworld.campaign_progress;
    let quests_done = state
        .completed_quests
        .iter()
        .filter(|q| q.result == QuestResult::Victory)
        .count() as f32;

    alive * 0.15 + gold * 0.1 + rep * 0.2 + progress * 0.3 + (quests_done / 25.0).min(1.0) * 0.25
}

/// Feature vector for clustering leaf states.
/// Low-dimensional summary for distance computation.
fn state_features(state: &CampaignState) -> Vec<f32> {
    let alive = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count() as f32;
    let idle = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .count() as f32;
    let gold_bucket = (state.guild.gold / 50.0).floor().min(10.0);
    let rep = state.guild.reputation / 20.0;
    let active_quests = state.active_quests.len() as f32;
    let active_battles = state.active_battles.len() as f32;
    let parties = state.parties.len() as f32;
    let progress = state.overworld.campaign_progress * 10.0;
    let threat = state.overworld.global_threat_level / 20.0;
    let pending_choices = state.pending_choices.len() as f32;
    let unlocks = state.unlocks.len() as f32;
    let mean_stress = if alive > 0.0 {
        state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.stress)
            .sum::<f32>() / alive
    } else { 0.0 } / 20.0;

    vec![
        alive, idle, gold_bucket, rep, active_quests, active_battles,
        parties, progress, threat, pending_choices, unlocks, mean_stress,
    ]
}

// ---------------------------------------------------------------------------
// Clustering (simple k-medoids)
// ---------------------------------------------------------------------------

/// Cluster leaf states into k groups, return the median state index per cluster.
fn cluster_and_select_medians(
    leaves: &[(CampaignState, Vec<f32>)],
    k: usize,
) -> Vec<usize> {
    if leaves.len() <= k {
        return (0..leaves.len()).collect();
    }

    // Simple k-medoids: pick k initial medoids evenly spaced
    let step = leaves.len() / k;
    let mut medoid_indices: Vec<usize> = (0..k).map(|i| i * step).collect();

    // Assign each leaf to nearest medoid, iterate a few times
    for _ in 0..5 {
        // Assignment
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, (_, features)) in leaves.iter().enumerate() {
            let nearest = medoid_indices
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    let da = distance(&leaves[a].1, features);
                    let db = distance(&leaves[b].1, features);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            clusters[nearest].push(i);
        }

        // Update medoids: pick member closest to cluster centroid
        for (ci, members) in clusters.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            let dim = leaves[0].1.len();
            let centroid: Vec<f32> = (0..dim)
                .map(|d| {
                    members.iter().map(|&m| leaves[m].1[d]).sum::<f32>() / members.len() as f32
                })
                .collect();
            let best = members
                .iter()
                .min_by(|&&a, &&b| {
                    let da = distance(&leaves[a].1, &centroid);
                    let db = distance(&leaves[b].1, &centroid);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(members[0]);
            medoid_indices[ci] = best;
        }
    }

    medoid_indices
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// BFS exploration
// ---------------------------------------------------------------------------

/// Run the full BFS exploration pipeline.
pub fn run_bfs_exploration(config: &BfsConfig) -> BfsStats {
    let threads = if config.threads == 0 {
        rayon::current_num_threads()
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build rayon pool");

    // Output
    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let writer = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&config.output_path).expect("Failed to create output file"),
    ));

    // Create shared LLM content store if LLM is enabled
    let llm_store = config.llm_config.as_ref().map(|cfg| {
        std::sync::Arc::new(super::llm::create_store(cfg))
    });

    let mut stats = BfsStats::default();
    let t0 = std::time::Instant::now();

    let max_waves_str = if config.max_waves == 0 {
        "unlimited (until completion)".to_string()
    } else {
        config.max_waves.to_string()
    };
    eprintln!("=== BFS State-Space Exploration ===");
    eprintln!("Max waves: {}, Clusters/wave: {}, Ticks/branch: {}",
        max_waves_str, config.clusters_per_wave, config.ticks_per_branch);
    eprintln!("Initial roots: {}, Threads: {}", config.initial_roots, threads);

    // Phase 1: Generate initial roots from diverse heuristic trajectories
    let mut roots = generate_initial_roots(config, &llm_store);
    eprintln!("Generated {} initial root states", roots.len());
    stats.initial_roots = roots.len();

    // Phase 2: BFS waves — continue until all leaves are terminal or max_waves reached
    let mut wave: u32 = 0;
    let mut total_terminal = 0u64;
    let mut total_victories = 0u64;
    let mut total_defeats = 0u64;

    loop {
        if config.max_waves > 0 && wave >= config.max_waves {
            eprintln!("  Reached max waves ({}), stopping", config.max_waves);
            break;
        }
        if roots.is_empty() {
            eprintln!("  No more roots to expand (all terminal), stopping");
            break;
        }

        let wave_t0 = std::time::Instant::now();
        let num_roots = roots.len();

        // Expand all roots in parallel
        let (samples, leaves): (Vec<Vec<BfsSample>>, Vec<Vec<(CampaignState, Vec<f32>, u64)>>) = pool.install(|| {
            use rayon::prelude::*;
            roots
                .par_iter()
                .map(|root| expand_root(root, config, wave))
                .unzip()
        });

        // Flatten
        let all_samples: Vec<BfsSample> = samples.into_iter().flatten().collect();
        let all_leaves: Vec<(CampaignState, Vec<f32>, u64)> =
            leaves.into_iter().flatten().collect();

        // Count terminal outcomes in this wave's samples
        let wave_terminal: usize = all_samples.iter().filter(|s| s.leaf_outcome.is_some()).count();
        let wave_victories: usize = all_samples.iter()
            .filter(|s| s.leaf_outcome.as_deref() == Some("Victory")).count();
        let wave_defeats: usize = all_samples.iter()
            .filter(|s| s.leaf_outcome.as_deref() == Some("Defeat")).count();
        total_terminal += wave_terminal as u64;
        total_victories += wave_victories as u64;
        total_defeats += wave_defeats as u64;

        // Write samples
        {
            use std::io::Write;
            let mut w = writer.lock().unwrap();
            for sample in &all_samples {
                if let Ok(json) = serde_json::to_string(sample) {
                    writeln!(w, "{}", json).ok();
                }
            }
        }

        stats.total_samples += all_samples.len();
        stats.total_leaves += all_leaves.len();

        // If no non-terminal leaves, we're done
        if all_leaves.is_empty() {
            let unique_action_types: std::collections::HashSet<&str> = all_samples
                .iter()
                .map(|s| s.action_type.as_str())
                .collect();
            eprintln!(
                "  Wave {}: {} roots → {} samples (all terminal: {}V/{}D), {} action types, {:.1}s",
                wave, num_roots, all_samples.len(),
                wave_victories, wave_defeats,
                unique_action_types.len(),
                wave_t0.elapsed().as_secs_f64(),
            );
            eprintln!("  All branches reached terminal states, stopping");
            break;
        }

        // Cluster non-terminal leaves and select medians for next wave
        let leaf_states_features: Vec<(CampaignState, Vec<f32>)> = all_leaves
            .iter()
            .map(|(s, f, _)| (s.clone(), f.clone()))
            .collect();

        let median_indices =
            cluster_and_select_medians(&leaf_states_features, config.clusters_per_wave);

        // Build next wave roots from medians
        roots = median_indices
            .iter()
            .filter_map(|&idx| {
                all_leaves.get(idx).map(|(state, _, seed)| RootState {
                    state: state.clone(),
                    seed: *seed,
                    wave: wave + 1,
                })
            })
            .collect();

        let unique_action_types: std::collections::HashSet<&str> = all_samples
            .iter()
            .map(|s| s.action_type.as_str())
            .collect();

        eprintln!(
            "  Wave {}: {} roots → {} samples ({} terminal: {}V/{}D), {} live leaves → {} clusters → {} next roots ({} action types, {:.1}s)",
            wave, num_roots,
            all_samples.len(), wave_terminal, wave_victories, wave_defeats,
            all_leaves.len(),
            config.clusters_per_wave,
            roots.len(),
            unique_action_types.len(),
            wave_t0.elapsed().as_secs_f64(),
        );

        wave += 1;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    stats.elapsed_secs = elapsed;
    stats.total_terminal = total_terminal as usize;
    stats.total_victories = total_victories as usize;
    stats.total_defeats = total_defeats as usize;
    stats.waves_completed = wave;

    eprintln!("\n=== BFS Exploration Complete ===");
    eprintln!("Waves: {}", wave);
    eprintln!("Total samples: {}", stats.total_samples);
    eprintln!("Terminal outcomes: {} ({} victories, {} defeats)",
        total_terminal, total_victories, total_defeats);
    eprintln!("Total leaves explored: {}", stats.total_leaves);
    eprintln!("Rate: {:.0} samples/s", stats.total_samples as f64 / elapsed);
    eprintln!("Output: {} ({:.1} MB)",
        config.output_path,
        std::fs::metadata(&config.output_path).map(|m| m.len() as f64 / 1_000_000.0).unwrap_or(0.0));

    if let Some(ref store) = llm_store {
        let (total, hits, valid) = store.stats();
        eprintln!("LLM: {} requests, {} cache hits, {} valid generations", total, hits, valid);
    }

    stats
}

#[derive(Debug, Default)]
pub struct BfsStats {
    pub initial_roots: usize,
    pub total_samples: usize,
    pub total_leaves: usize,
    pub total_terminal: usize,
    pub total_victories: usize,
    pub total_defeats: usize,
    pub waves_completed: u32,
    pub elapsed_secs: f64,
}

// ---------------------------------------------------------------------------
// Diverse policy for human-like root generation
// ---------------------------------------------------------------------------

/// Policy that simulates varied human playstyles for root generation.
/// Randomizes creation choices, occasionally rests, trains, or diplomats
/// instead of always taking the greedy quest-focused heuristic path.
fn diverse_root_policy(state: &CampaignState, rng: &mut u64) -> Option<CampaignAction> {
    // Pre-game: randomize creation and starting package choices
    if state.phase != CampaignPhase::Playing {
        if let Some(choice) = state.pending_choices.first() {
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            let idx = (*rng as usize) % choice.options.len().max(1);
            return Some(CampaignAction::RespondToChoice {
                choice_id: choice.id,
                option_index: idx,
            });
        }
        if state.phase == CampaignPhase::ChoosingStartingPackage {
            if !state.available_starting_choices.is_empty() {
                *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
                let idx = (*rng as usize) % state.available_starting_choices.len();
                return Some(CampaignAction::ChooseStartingPackage {
                    choice: state.available_starting_choices[idx].clone(),
                });
            }
        }
        return None;
    }

    // Randomize choice responses during play
    if let Some(choice) = state.pending_choices.first() {
        *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
        let idx = (*rng as usize) % choice.options.len().max(1);
        return Some(CampaignAction::RespondToChoice {
            choice_id: choice.id,
            option_index: idx,
        });
    }

    // ~10% chance: rest if there are pending progression items
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    if (*rng % 10) == 0 && !state.pending_progression.is_empty() {
        return Some(CampaignAction::Rest);
    }

    // ~5% chance: train a random adventurer if we have gold
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    if (*rng % 20) == 0 && state.guild.gold > state.config.economy.training_cost {
        let idle: Vec<u32> = state.adventurers.iter()
            .filter(|a| a.status == AdventurerStatus::Idle)
            .map(|a| a.id)
            .collect();
        if !idle.is_empty() {
            let idx = (*rng as usize) % idle.len();
            let training_types = [TrainingType::Combat, TrainingType::Exploration,
                                  TrainingType::Leadership, TrainingType::Survival];
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            let ti = (*rng as usize) % training_types.len();
            return Some(CampaignAction::TrainAdventurer {
                adventurer_id: idle[idx],
                training_type: training_types[ti],
            });
        }
    }

    // ~5% chance: diplomatic action if we have factions
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    if (*rng % 20) == 0 && !state.factions.is_empty() {
        let fi = (*rng as usize) % state.factions.len();
        return Some(CampaignAction::DiplomaticAction {
            faction_id: state.factions[fi].id,
            action_type: DiplomacyActionType::ImproveRelations,
        });
    }

    // ~3% chance: decline a quest instead of accepting
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    if (*rng % 30) == 0 {
        if let Some(quest) = state.request_board.first() {
            return Some(CampaignAction::DeclineQuest { request_id: quest.id });
        }
    }

    // Otherwise: use the standard heuristic
    heuristic_policy(state)
}

// ---------------------------------------------------------------------------
// Root generation
// ---------------------------------------------------------------------------

/// Generate initial roots from diverse campaigns.
fn generate_initial_roots(
    config: &BfsConfig,
    llm_store: &Option<std::sync::Arc<super::llm::ContentStore>>,
) -> Vec<RootState> {
    let mut roots = Vec::new();
    let campaigns_needed = (config.initial_roots / 10).max(5);

    for i in 0..campaigns_needed as u64 {
        let seed = config.base_seed.wrapping_add(i * 7919); // spread seeds
        let mut state = CampaignState::with_config(seed, config.campaign_config.clone());
        state.llm_config = config.llm_config.clone();
        state.llm_store = llm_store.clone();
        state.vae_model = config.vae_model.clone();
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        // Run heuristic trajectory with diverse character creation
        for _tick in 0..config.trajectory_max_ticks {
            let action = diverse_root_policy(&state, &mut rng);
            let result = step_campaign(&mut state, action);

            if state.tick > 0 && state.tick % config.root_sample_interval == 0 {
                roots.push(RootState {
                    state: state.clone(),
                    seed,
                    wave: 0,
                });
            }

            if result.outcome.is_some() {
                break;
            }
        }
    }

    // Deduplicate by clustering if we have too many
    if roots.len() > config.initial_roots * 2 {
        let features: Vec<(CampaignState, Vec<f32>)> = roots
            .iter()
            .map(|r| (r.state.clone(), state_features(&r.state)))
            .collect();
        let medians = cluster_and_select_medians(&features, config.initial_roots);
        roots = medians.into_iter().filter_map(|idx| roots.get(idx).cloned()).collect();
    }

    roots
}

// ---------------------------------------------------------------------------
// Root expansion
// ---------------------------------------------------------------------------

/// Expand a single root: try every action type, advance ticks_per_branch,
/// record samples.
fn expand_root(
    root: &RootState,
    config: &BfsConfig,
    wave: u32,
) -> (Vec<BfsSample>, Vec<(CampaignState, Vec<f32>, u64)>) {
    let valid_actions = root.state.valid_actions();
    let grouped = group_actions(&valid_actions);
    let root_tokens = root.state.to_tokens();
    let root_tick = root.state.tick;

    let mut samples = Vec::new();
    let mut leaves = Vec::new();

    for (action_type, action) in &grouped {
        // Clone state and apply this action
        let mut branch_state = root.state.clone();
        let action_detail = format!("{:?}", action);
        step_campaign(&mut branch_state, Some(action.clone()));

        // Advance ticks_per_branch with heuristic policy
        let mut terminal = false;
        let mut outcome_str = None;
        for _ in 0..config.ticks_per_branch {
            let h_action = heuristic_policy(&branch_state);
            let result = step_campaign(&mut branch_state, h_action);
            if let Some(outcome) = result.outcome {
                outcome_str = Some(format!("{:?}", outcome));
                terminal = true;
                break;
            }
        }

        let leaf_tokens = branch_state.to_tokens();
        let leaf_value = if terminal {
            match outcome_str.as_deref() {
                Some("Victory") => 1.0,
                Some("Defeat") => 0.0,
                _ => 0.5,
            }
        } else {
            estimate_value(&branch_state)
        };

        let leaf_features = state_features(&branch_state);

        samples.push(BfsSample {
            root_tokens: root_tokens.clone(),
            action_type: action_type.clone(),
            action_detail: action_detail.clone(),
            leaf_tokens,
            leaf_value,
            leaf_outcome: outcome_str,
            root_tick,
            leaf_tick: branch_state.tick,
            seed: root.seed,
            wave,
            cluster_id: 0, // filled during clustering
        });

        if !terminal {
            leaves.push((branch_state, leaf_features, root.seed));
        }
    }

    (samples, leaves)
}
