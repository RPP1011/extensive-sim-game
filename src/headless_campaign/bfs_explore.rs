//! BFS state-space exploration with cluster-and-prune.
//!
//! 1. Start from a set of root states
//! 2. At each root, try every ACTION TYPE (grouped — not every target variant)
//! 3. Step forward N ticks per branch to let the action play out
//! 4. Record (root_tokens, action_type, leaf_tokens, leaf_value)
//! 5. Cluster the leaf states by feature similarity, preserving extremes
//! 6. Pick diverse representatives per cluster as next-wave roots
//! 7. Repeat for K waves
//!
//! This gives full action coverage without exponential blowup.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::actions::*;
use super::batch::heuristic_policy;
use super::config::{CampaignConfig, Difficulty};
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
    /// Difficulty level of the originating campaign.
    pub difficulty: Difficulty,
    /// Heuristic value at the root (for computing spread).
    pub root_value: f32,
    /// Game phase tag (early/mid/late).
    pub phase_tag: String,
}

/// A state snapshot used as a BFS root.
#[derive(Clone)]
struct RootState {
    state: CampaignState,
    seed: u64,
    wave: u32,
    difficulty: Difficulty,
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
// Action grouping — with strategic diversity
// ---------------------------------------------------------------------------

/// Categorize actions into strategic buckets for ensuring diversity.
fn strategic_bucket(action_type: &str) -> &'static str {
    match action_type {
        "AcceptQuest" | "DeclineQuest" | "DispatchQuest" | "AssignToPool" | "UnassignFromPool" => "quest_mgmt",
        "TrainAdventurer" | "EquipGear" => "development",
        "DiplomaticAction" | "ProposeCoalition" | "RequestCoalitionAid" => "diplomacy",
        "HireScout" => "scouting",
        "PurchaseSupplies" | "SendRunner" => "logistics",
        "HireMercenary" | "CallRescue" | "InterceptChampion" => "military",
        "SetSpendPriority" => "economy",
        "UseAbility" => "ability",
        "Rest" => "rest",
        "Wait" => "wait",
        _ => "other",
    }
}

/// Get one representative action per action type from valid actions.
/// For diplomacy actions, split by stance-specific subtypes to preserve diversity.
fn group_actions(valid_actions: &[CampaignAction]) -> Vec<(String, CampaignAction)> {
    let mut seen: HashMap<String, CampaignAction> = HashMap::new();
    for action in valid_actions {
        let type_name = match action {
            // Split diplomacy by action subtype so all diplomatic moves get branches
            CampaignAction::DiplomaticAction { action_type, .. } => {
                format!("DiplomaticAction_{:?}", action_type)
            }
            // Split spend priority by variant — each is strategically distinct
            CampaignAction::SetSpendPriority { priority } => {
                format!("SetSpendPriority_{:?}", priority)
            }
            _ => action_type_name(action),
        };
        // Keep the first instance of each type
        seen.entry(type_name).or_insert_with(|| action.clone());
    }
    seen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Leaf value estimation
// ---------------------------------------------------------------------------

/// Quick heuristic value for a campaign state.
/// Higher = better. Range roughly 0-5 with multiplicative factors for spread.
fn estimate_value(state: &CampaignState) -> f32 {
    let adventurers: Vec<_> = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .collect();
    let alive = adventurers.len() as f32;

    if alive == 0.0 {
        return 0.0; // everyone dead = worst case
    }

    // --- Roster health (0-1): how healthy are our adventurers? ---
    let mean_health = adventurers.iter()
        .map(|a| {
            let injury_penalty = a.injury / 100.0;
            let stress_penalty = a.stress / 200.0;
            let morale_bonus = a.morale / 200.0;
            (1.0 - injury_penalty - stress_penalty + morale_bonus).clamp(0.0, 1.0)
        })
        .sum::<f32>() / alive;

    // --- Resource position (0-1) ---
    let gold_score = (state.guild.gold / 200.0).clamp(0.0, 1.0);
    let supply_score = (state.guild.supplies / 100.0).clamp(0.0, 1.0);
    let rep_score = state.guild.reputation / 100.0;

    // --- Progress (0-1) ---
    let progress = state.overworld.campaign_progress;
    let quests_won = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Victory).count() as f32;
    let quests_lost = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Defeat).count() as f32;
    let win_rate = if quests_won + quests_lost > 0.0 {
        quests_won / (quests_won + quests_lost)
    } else {
        0.5
    };

    // --- Threat pressure (0-1, higher = worse) ---
    let threat = (state.overworld.global_threat_level / 100.0).clamp(0.0, 1.0);
    let crisis_pressure = (state.overworld.active_crises.len() as f32 * 0.25).min(1.0);

    // --- Operational capacity ---
    let idle = adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .count() as f32;
    let _capacity = (idle / alive.max(1.0)).clamp(0.0, 1.0);
    let quest_load = (state.active_quests.len() as f32 / state.guild.active_quest_capacity.max(1) as f32)
        .clamp(0.0, 1.0);

    // --- Mean level (growth indicator) ---
    let mean_level = adventurers.iter().map(|a| a.level as f32).sum::<f32>() / alive;
    let level_score = (mean_level / 20.0).min(1.0);

    // --- Penalties for negative states (amplify differences) ---
    let injured_count = adventurers.iter()
        .filter(|a| a.injury > 40.0).count() as f32;
    let deserted = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Dead).count() as f32;

    // Multiplicative components -- create larger spreads
    let roster_factor = (alive / (alive + deserted).max(1.0)).powf(0.5);
    let health_factor = mean_health.powf(1.5);
    let economy_factor = (gold_score * 0.6 + supply_score * 0.2 + rep_score * 0.2).powf(0.8);

    // Additive progress
    let progress_score = progress * 2.0 + (quests_won / 25.0).min(1.0) + win_rate;

    // --- War pressure ---
    let war_penalty = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .map(|f| {
            let strength_ratio = f.military_strength / 50.0;
            strength_ratio.min(3.0) * 0.3
        })
        .sum::<f32>();

    // Territory safety
    let guild_faction_id = state.diplomacy.guild_faction_id;
    let guild_regions = state.overworld.regions.iter()
        .filter(|r| r.owner_faction_id == guild_faction_id)
        .count() as f32;
    let total_regions = state.overworld.regions.len().max(1) as f32;
    let territory_score = guild_regions / total_regions;
    let territory_penalty = if territory_score < 0.5 { (1.0 - territory_score) * 0.8 } else { 0.0 };

    // --- Scouting visibility: information is power ---
    let mean_visibility = if state.overworld.regions.is_empty() {
        0.5
    } else {
        state.overworld.regions.iter().map(|r| r.visibility).sum::<f32>()
            / state.overworld.regions.len() as f32
    };
    let scouting_bonus = (mean_visibility - 0.3).max(0.0) * 0.4; // reward above baseline 0.3

    // --- Rival guild pressure: falling behind the rival is bad ---
    let rival_penalty = if state.rival_guild.active {
        let rep_gap = (state.rival_guild.reputation - state.guild.reputation) / 100.0;
        let quest_gap = (state.rival_guild.quests_completed as f32
            - state.completed_quests.len() as f32) / 20.0;
        (rep_gap.max(0.0) * 0.4 + quest_gap.max(0.0) * 0.3).min(1.0)
    } else {
        0.0
    };

    // --- Building investment: infrastructure signals long-term strength ---
    let buildings = &state.guild_buildings;
    let total_building_tiers = (buildings.training_grounds
        + buildings.watchtower + buildings.trade_post
        + buildings.barracks + buildings.infirmary
        + buildings.war_room) as f32;
    let building_bonus = (total_building_tiers / 18.0).min(1.0) * 0.3; // 18 = 6 buildings * 3 max

    // --- Bond network: strong bonds = resilient party ---
    let bond_count = state.adventurer_bonds.len() as f32;
    let mean_bond = if bond_count > 0.0 {
        state.adventurer_bonds.values().sum::<f32>() / bond_count / 100.0
    } else {
        0.0
    };
    let bond_bonus = mean_bond * 0.15;

    // --- Coalition strength: allies amplify military capacity ---
    let coalition_allies = state.factions.iter()
        .filter(|f| f.coalition_member)
        .count() as f32;
    let coalition_bonus = (coalition_allies * 0.15).min(0.5);

    // --- Trade income: passive gold generation compounds over time ---
    let trade_bonus = (state.guild.total_trade_income / 100.0).min(0.3);

    // --- Season awareness: winter is harsher, summer is opportunity ---
    let season_modifier = match state.overworld.season {
        Season::Winter => -0.1,
        Season::Summer => 0.05,
        _ => 0.0,
    };

    // --- Loyalty risk: low-loyalty adventurers may desert ---
    let desertion_risk = adventurers.iter()
        .filter(|a| a.loyalty < 30.0)
        .count() as f32 * 0.1;

    // Penalties
    let injured_penalty = injured_count * 0.15;
    let threat_penalty = threat * 0.5 + crisis_pressure * 0.4;
    let bankruptcy_penalty = if state.guild.gold < 20.0 { 0.5 } else { 0.0 };

    // Combine with multiplicative factors for larger dynamic range
    let base = progress_score + level_score * 0.5 + quest_load * 0.3
        + territory_score * 0.5
        + scouting_bonus + building_bonus + bond_bonus
        + coalition_bonus + trade_bonus + season_modifier;
    let v = base * roster_factor * health_factor * economy_factor
        - injured_penalty - threat_penalty - bankruptcy_penalty
        - war_penalty - territory_penalty - rival_penalty - desertion_risk;

    v.clamp(0.0, 5.0)
}

/// Feature vector for clustering leaf states.
/// Expanded to capture scouting, buildings, bonds, rival, and season.
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
    let (mean_stress, mean_injury, mean_morale) = if alive > 0.0 {
        let advs: Vec<_> = state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .collect();
        (
            advs.iter().map(|a| a.stress).sum::<f32>() / alive / 20.0,
            advs.iter().map(|a| a.injury).sum::<f32>() / alive / 20.0,
            advs.iter().map(|a| a.morale).sum::<f32>() / alive / 20.0,
        )
    } else { (0.0, 0.0, 0.0) };

    let mean_level = if alive > 0.0 {
        state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.level as f32).sum::<f32>() / alive / 10.0
    } else { 0.0 };

    let crises = state.overworld.active_crises.len() as f32;
    let supplies = (state.guild.supplies / 20.0).min(5.0);
    let quests_done = (state.completed_quests.len() as f32 / 5.0).min(10.0);

    // New: scouting visibility
    let mean_visibility = if state.overworld.regions.is_empty() {
        0.5
    } else {
        state.overworld.regions.iter().map(|r| r.visibility).sum::<f32>()
            / state.overworld.regions.len() as f32
    };

    // New: building tiers (normalized)
    let b = &state.guild_buildings;
    let building_total = (b.training_grounds + b.watchtower + b.trade_post
        + b.barracks + b.infirmary + b.war_room) as f32 / 6.0;

    // New: rival guild state
    let rival_rep = if state.rival_guild.active {
        state.rival_guild.reputation / 20.0
    } else {
        0.0
    };

    // New: bond network density
    let bond_density = state.adventurer_bonds.len() as f32 / (alive.max(1.0) * (alive.max(1.0) - 1.0) / 2.0).max(1.0);

    // New: coalition count
    let coalition_count = state.factions.iter()
        .filter(|f| f.coalition_member).count() as f32;

    // New: factions at war
    let war_count = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar).count() as f32;

    // New: season (cyclical encoding)
    let season_idx = match state.overworld.season {
        Season::Spring => 0.0,
        Season::Summer => 1.0,
        Season::Autumn => 2.0,
        Season::Winter => 3.0,
    };

    // New: territory control ratio
    let guild_faction_id = state.diplomacy.guild_faction_id;
    let guild_regions = state.overworld.regions.iter()
        .filter(|r| r.owner_faction_id == guild_faction_id)
        .count() as f32;
    let territory_ratio = guild_regions / state.overworld.regions.len().max(1) as f32;

    vec![
        alive, idle, gold_bucket, rep, active_quests, active_battles,
        parties, progress, threat, pending_choices, unlocks, mean_stress,
        mean_injury, mean_morale, mean_level, crises, supplies, quests_done,
        // Extended dimensions:
        mean_visibility, building_total, rival_rep, bond_density,
        coalition_count, war_count, season_idx, territory_ratio,
    ]
}

// ---------------------------------------------------------------------------
// Clustering (k-medoids with extreme preservation)
// ---------------------------------------------------------------------------

/// Cluster leaf states into k groups, return representative indices.
/// Always preserves the best and worst leaves to keep extreme outcomes.
fn cluster_and_select_diverse(
    leaves: &[(CampaignState, Vec<f32>)],
    k: usize,
    leaf_values: &[f32],
) -> Vec<usize> {
    if leaves.len() <= k {
        return (0..leaves.len()).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(k);

    // Reserve 2 slots for extremes: best and worst leaf by value
    if let Some(best_idx) = leaf_values.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        selected.push(best_idx);
    }
    if let Some(worst_idx) = leaf_values.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        if !selected.contains(&worst_idx) {
            selected.push(worst_idx);
        }
    }

    // Remaining slots via k-medoids on features
    let remaining_k = k.saturating_sub(selected.len());
    if remaining_k > 0 {
        let medoids = cluster_and_select_medians(leaves, remaining_k + selected.len());
        for idx in medoids {
            if !selected.contains(&idx) && selected.len() < k {
                selected.push(idx);
            }
        }
    }

    selected
}

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
// Game phase tagging
// ---------------------------------------------------------------------------

/// Classify tick into game phase for coverage tracking.
fn phase_tag(tick: u64, progress: f32) -> String {
    if progress < 0.2 || tick < 2000 {
        "early".to_string()
    } else if progress < 0.6 || tick < 8000 {
        "mid".to_string()
    } else {
        "late".to_string()
    }
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

    // Track action type coverage across all waves
    let mut global_action_types: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Track phase coverage
    let mut phase_counts: HashMap<String, usize> = HashMap::new();

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
        let (samples, leaves): (Vec<Vec<BfsSample>>, Vec<Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)>>) = pool.install(|| {
            use rayon::prelude::*;
            roots
                .par_iter()
                .map(|root| expand_root(root, config, wave))
                .unzip()
        });

        // Flatten
        let all_samples: Vec<BfsSample> = samples.into_iter().flatten().collect();
        let all_leaves: Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)> =
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

        // Track coverage
        for s in &all_samples {
            global_action_types.insert(s.action_type.clone());
            *phase_counts.entry(s.phase_tag.clone()).or_insert(0) += 1;
        }

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
                "  Wave {}: {} roots -> {} samples (all terminal: {}V/{}D), {} action types, {:.1}s",
                wave, num_roots, all_samples.len(),
                wave_victories, wave_defeats,
                unique_action_types.len(),
                wave_t0.elapsed().as_secs_f64(),
            );
            eprintln!("  All branches reached terminal states, stopping");
            break;
        }

        // Cluster non-terminal leaves with extreme preservation
        let leaf_states_features: Vec<(CampaignState, Vec<f32>)> = all_leaves
            .iter()
            .map(|(s, f, _, _, _)| (s.clone(), f.clone()))
            .collect();
        let leaf_values: Vec<f32> = all_leaves.iter().map(|(_, _, v, _, _)| *v).collect();

        let selected_indices =
            cluster_and_select_diverse(&leaf_states_features, config.clusters_per_wave, &leaf_values);

        // Build next wave roots from selected
        roots = selected_indices
            .iter()
            .filter_map(|&idx| {
                all_leaves.get(idx).map(|(state, _, _, seed, diff)| RootState {
                    state: state.clone(),
                    seed: *seed,
                    wave: wave + 1,
                    difficulty: *diff,
                })
            })
            .collect();

        let unique_action_types: std::collections::HashSet<&str> = all_samples
            .iter()
            .map(|s| s.action_type.as_str())
            .collect();

        // Compute spread stats for this wave
        let mut root_spreads: Vec<f32> = Vec::new();
        let mut per_root: HashMap<u64, (f32, f32)> = HashMap::new();
        for s in &all_samples {
            let key = s.root_tick ^ s.seed;
            let entry = per_root.entry(key).or_insert((f32::MAX, f32::MIN));
            entry.0 = entry.0.min(s.leaf_value);
            entry.1 = entry.1.max(s.leaf_value);
        }
        for (_, (lo, hi)) in &per_root {
            root_spreads.push(hi - lo);
        }
        let mean_spread = if root_spreads.is_empty() { 0.0 }
            else { root_spreads.iter().sum::<f32>() / root_spreads.len() as f32 };

        eprintln!(
            "  Wave {}: {} roots -> {} samples ({} terminal: {}V/{}D), {} live -> {} next ({} types, spread={:.3}, {:.1}s)",
            wave, num_roots,
            all_samples.len(), wave_terminal, wave_victories, wave_defeats,
            all_leaves.len(),
            roots.len(),
            unique_action_types.len(),
            mean_spread,
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
    stats.unique_action_types = global_action_types.len();
    stats.phase_coverage = phase_counts;

    eprintln!("\n=== BFS Exploration Complete ===");
    eprintln!("Waves: {}", wave);
    eprintln!("Total samples: {}", stats.total_samples);
    eprintln!("Terminal outcomes: {} ({} victories, {} defeats)",
        total_terminal, total_victories, total_defeats);
    eprintln!("Total leaves explored: {}", stats.total_leaves);
    eprintln!("Unique action types: {}", stats.unique_action_types);
    eprintln!("Phase coverage: {:?}", stats.phase_coverage);
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
    /// Number of distinct action types seen across all waves.
    pub unique_action_types: usize,
    /// Samples per game phase (early/mid/late).
    pub phase_coverage: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Diverse policy for human-like root generation
// ---------------------------------------------------------------------------

/// Policy that simulates varied human playstyles for root generation.
/// Expanded to cover more action types: scouting, spend priority, equip,
/// purchase supplies, coalition, trade, threaten, and ability usage.
fn diverse_root_policy(state: &CampaignState, rng: &mut u64) -> Option<CampaignAction> {
    // Pre-game: randomize creation and starting package choices
    if state.phase != CampaignPhase::Playing {
        if let Some(choice) = state.pending_choices.first() {
            xorshift(rng);
            let idx = (*rng as usize) % choice.options.len().max(1);
            return Some(CampaignAction::RespondToChoice {
                choice_id: choice.id,
                option_index: idx,
            });
        }
        if state.phase == CampaignPhase::ChoosingStartingPackage {
            if !state.available_starting_choices.is_empty() {
                xorshift(rng);
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
        xorshift(rng);
        let idx = (*rng as usize) % choice.options.len().max(1);
        return Some(CampaignAction::RespondToChoice {
            choice_id: choice.id,
            option_index: idx,
        });
    }

    // Roll a d100 for action selection — guarantees coverage of rare actions
    xorshift(rng);
    let roll = *rng % 100;

    // ~8% chance: rest (especially if stressed/injured roster or pending progression)
    if roll < 8 {
        let has_stressed = state.adventurers.iter()
            .any(|a| a.status != AdventurerStatus::Dead && (a.stress > 50.0 || a.injury > 30.0));
        if has_stressed || !state.pending_progression.is_empty() {
            return Some(CampaignAction::Rest);
        }
    }

    // ~5% chance: train a random adventurer
    if roll < 13 && roll >= 8 && state.guild.gold > state.config.economy.training_cost {
        let idle: Vec<u32> = state.adventurers.iter()
            .filter(|a| a.status == AdventurerStatus::Idle)
            .map(|a| a.id)
            .collect();
        if !idle.is_empty() {
            xorshift(rng);
            let idx = (*rng as usize) % idle.len();
            let training_types = [TrainingType::Combat, TrainingType::Exploration,
                                  TrainingType::Leadership, TrainingType::Survival];
            xorshift(rng);
            let ti = (*rng as usize) % training_types.len();
            return Some(CampaignAction::TrainAdventurer {
                adventurer_id: idle[idx],
                training_type: training_types[ti],
            });
        }
    }

    // ~7% chance: diplomatic action (expanded to cover all diplomacy subtypes)
    if roll < 20 && roll >= 13 && !state.factions.is_empty() {
        xorshift(rng);
        let fi = (*rng as usize) % state.factions.len();
        let faction = &state.factions[fi];
        let action_type = match faction.diplomatic_stance {
            DiplomaticStance::AtWar => DiplomacyActionType::ProposeCeasefire,
            DiplomaticStance::Hostile => {
                xorshift(rng);
                if *rng % 2 == 0 { DiplomacyActionType::ImproveRelations }
                else { DiplomacyActionType::Threaten }
            }
            DiplomaticStance::Neutral => {
                xorshift(rng);
                if *rng % 2 == 0 { DiplomacyActionType::ImproveRelations }
                else { DiplomacyActionType::TradeAgreement }
            }
            DiplomaticStance::Friendly => {
                xorshift(rng);
                if *rng % 3 == 0 && faction.relationship_to_guild > 60.0 && !faction.coalition_member {
                    // Try to propose coalition instead
                    return Some(CampaignAction::ProposeCoalition {
                        faction_id: faction.id,
                    });
                }
                DiplomacyActionType::TradeAgreement
            }
            DiplomaticStance::Coalition => {
                if faction.military_strength > 20.0 {
                    return Some(CampaignAction::RequestCoalitionAid {
                        faction_id: faction.id,
                    });
                }
                DiplomacyActionType::TradeAgreement
            }
        };
        return Some(CampaignAction::DiplomaticAction {
            faction_id: faction.id,
            action_type,
        });
    }

    // ~4% chance: hire scout for unscouted location
    if roll < 24 && roll >= 20
        && state.guild.gold >= state.config.economy.scout_cost
    {
        let unscouted: Vec<usize> = state.overworld.locations.iter()
            .filter(|l| !l.scouted)
            .map(|l| l.id)
            .collect();
        if !unscouted.is_empty() {
            xorshift(rng);
            let idx = (*rng as usize) % unscouted.len();
            return Some(CampaignAction::HireScout {
                location_id: unscouted[idx],
            });
        }
    }

    // ~3% chance: change spend priority
    if roll < 27 && roll >= 24 {
        xorshift(rng);
        let priorities = [
            SpendPriority::Balanced,
            SpendPriority::SaveForEmergencies,
            SpendPriority::InvestInGrowth,
            SpendPriority::MilitaryFocus,
        ];
        let pi = (*rng as usize) % priorities.len();
        return Some(CampaignAction::SetSpendPriority {
            priority: priorities[pi],
        });
    }

    // ~3% chance: equip gear if inventory has items
    if roll < 30 && roll >= 27 && !state.guild.inventory.is_empty() {
        let idle: Vec<u32> = state.adventurers.iter()
            .filter(|a| a.status == AdventurerStatus::Idle)
            .map(|a| a.id)
            .collect();
        if !idle.is_empty() {
            xorshift(rng);
            let ai = (*rng as usize) % idle.len();
            xorshift(rng);
            let ii = (*rng as usize) % state.guild.inventory.len();
            return Some(CampaignAction::EquipGear {
                adventurer_id: idle[ai],
                item_id: state.guild.inventory[ii].id,
            });
        }
    }

    // ~3% chance: purchase supplies for in-field party
    if roll < 33 && roll >= 30
        && state.guild.gold >= state.config.economy.supply_cost_per_unit * 10.0
    {
        let field_parties: Vec<u32> = state.parties.iter()
            .filter(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting))
            .map(|p| p.id)
            .collect();
        if !field_parties.is_empty() {
            xorshift(rng);
            let pi = (*rng as usize) % field_parties.len();
            return Some(CampaignAction::PurchaseSupplies {
                party_id: field_parties[pi],
                amount: 10.0,
            });
        }
    }

    // ~2% chance: send runner
    if roll < 35 && roll >= 33
        && state.guild.gold >= state.config.economy.runner_cost
    {
        let field_parties: Vec<u32> = state.parties.iter()
            .filter(|p| matches!(p.status, PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting))
            .map(|p| p.id)
            .collect();
        if !field_parties.is_empty() {
            xorshift(rng);
            let pi = (*rng as usize) % field_parties.len();
            xorshift(rng);
            let payload = if *rng % 2 == 0 {
                RunnerPayload::Supplies(20.0)
            } else {
                RunnerPayload::Message
            };
            return Some(CampaignAction::SendRunner {
                party_id: field_parties[pi],
                payload,
            });
        }
    }

    // ~2% chance: use an active ability
    if roll < 37 && roll >= 35 {
        let ready_abilities: Vec<&UnlockInstance> = state.unlocks.iter()
            .filter(|u| u.active && !u.properties.is_passive
                && u.cooldown_remaining_ms == 0
                && state.guild.gold >= u.properties.resource_cost)
            .collect();
        if !ready_abilities.is_empty() {
            xorshift(rng);
            let ai = (*rng as usize) % ready_abilities.len();
            let unlock = ready_abilities[ai];
            let target = match unlock.properties.target_type {
                TargetType::GuildSelf => Some(AbilityTarget::Party(0)),
                TargetType::Adventurer => {
                    let live: Vec<u32> = state.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .map(|a| a.id).collect();
                    if live.is_empty() { None } else {
                        xorshift(rng);
                        Some(AbilityTarget::Adventurer(live[(*rng as usize) % live.len()]))
                    }
                }
                TargetType::Party => {
                    if state.parties.is_empty() { None } else {
                        xorshift(rng);
                        Some(AbilityTarget::Party(state.parties[(*rng as usize) % state.parties.len()].id))
                    }
                }
                TargetType::Quest => {
                    if state.active_quests.is_empty() { None } else {
                        xorshift(rng);
                        Some(AbilityTarget::Quest(state.active_quests[(*rng as usize) % state.active_quests.len()].id))
                    }
                }
                TargetType::Area => {
                    if state.overworld.locations.is_empty() { None } else {
                        xorshift(rng);
                        Some(AbilityTarget::Location(state.overworld.locations[(*rng as usize) % state.overworld.locations.len()].id))
                    }
                }
                TargetType::Faction => {
                    if state.factions.is_empty() { None } else {
                        xorshift(rng);
                        Some(AbilityTarget::Faction(state.factions[(*rng as usize) % state.factions.len()].id))
                    }
                }
            };
            if let Some(t) = target {
                return Some(CampaignAction::UseAbility {
                    unlock_id: unlock.id,
                    target: t,
                });
            }
        }
    }

    // ~3% chance: decline a quest instead of accepting
    if roll < 40 && roll >= 37 {
        if let Some(quest) = state.request_board.first() {
            return Some(CampaignAction::DeclineQuest { request_id: quest.id });
        }
    }

    // ~2% chance: hire mercenary for active quest
    if roll < 42 && roll >= 40
        && state.guild.gold >= state.config.economy.mercenary_cost
    {
        let eligible: Vec<u32> = state.active_quests.iter()
            .filter(|q| matches!(q.status, ActiveQuestStatus::InProgress | ActiveQuestStatus::InCombat))
            .map(|q| q.id)
            .collect();
        if !eligible.is_empty() {
            xorshift(rng);
            let qi = (*rng as usize) % eligible.len();
            return Some(CampaignAction::HireMercenary { quest_id: eligible[qi] });
        }
    }

    // ~1% chance: call rescue for losing battle
    if roll < 43 && roll >= 42 {
        for battle in &state.active_battles {
            if battle.status == BattleStatus::Active && battle.party_health_ratio < 0.4 {
                let can_afford = state.guild.gold >= state.config.economy.rescue_bribe_cost;
                let has_free = state.npc_relationships.iter()
                    .any(|r| r.rescue_available && r.relationship_score > 50.0);
                if can_afford || has_free {
                    return Some(CampaignAction::CallRescue { battle_id: battle.id });
                }
            }
        }
    }

    // Otherwise (57%): use the standard heuristic
    heuristic_policy(state)
}

/// Inline xorshift64 step.
fn xorshift(rng: &mut u64) {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
}

// ---------------------------------------------------------------------------
// Root generation
// ---------------------------------------------------------------------------

/// Generate initial roots from diverse campaigns.
/// Ensures full game timeline coverage (early/mid/late) and crisis states.
fn generate_initial_roots(
    config: &BfsConfig,
    llm_store: &Option<std::sync::Arc<super::llm::ContentStore>>,
) -> Vec<RootState> {
    let mut roots = Vec::new();
    let campaigns_needed = (config.initial_roots / 3).max(10);

    // Track phase coverage to ensure balance
    let mut early_count = 0usize;
    let mut mid_count = 0usize;
    let mut late_count = 0usize;
    let target_per_phase = config.initial_roots / 3;

    for i in 0..campaigns_needed as u64 {
        let seed = config.base_seed.wrapping_add(i * 7919);
        // Distribution: 15% Easy, 25% Normal, 35% Hard, 25% Brutal
        let difficulty = match i as usize % 20 {
            0..=2 => Difficulty::Easy,
            3..=7 => Difficulty::Normal,
            8..=14 => Difficulty::Hard,
            _ => Difficulty::Brutal,
        };
        let campaign_config = CampaignConfig::with_difficulty(difficulty);
        let mut state = CampaignState::with_config(seed, campaign_config);
        state.llm_config = config.llm_config.clone();
        state.llm_store = llm_store.clone();
        state.vae_model = config.vae_model.clone();
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        // Run heuristic trajectory with diverse character creation
        // Extend max_tick to allow late-game sampling
        let extended_max = config.trajectory_max_ticks;
        for _tick in 0..extended_max {
            let action = diverse_root_policy(&state, &mut rng);
            let result = step_campaign(&mut state, action);

            let alive = state.adventurers.iter()
                .filter(|a| a.status != AdventurerStatus::Dead).count();
            let is_viable = alive >= 2 && state.guild.gold > 5.0; // relaxed gold threshold

            // Phase-aware sampling
            let phase = phase_tag(state.tick, state.overworld.campaign_progress);
            let phase_has_room = match phase.as_str() {
                "early" => early_count < target_per_phase * 2,
                "mid" => mid_count < target_per_phase * 2,
                "late" => late_count < target_per_phase * 2,
                _ => true,
            };

            // Stratified sampling: tag roots by what actions are available
            let has_quest_board = !state.request_board.is_empty();
            let has_active_quest = !state.active_quests.is_empty();
            let has_battle = !state.active_battles.is_empty();
            let has_injured = state.adventurers.iter().any(|a| a.injury > 30.0);
            let has_crisis = !state.overworld.active_crises.is_empty();
            let has_pending_choice = !state.pending_choices.is_empty();
            let has_rival = state.rival_guild.active;
            let has_war = state.factions.iter().any(|f| f.diplomatic_stance == DiplomaticStance::AtWar);
            let has_low_loyalty = state.adventurers.iter()
                .any(|a| a.status != AdventurerStatus::Dead && a.loyalty < 30.0);

            let has_tension = is_viable && (
                has_quest_board || has_active_quest || has_battle
                || has_injured || has_crisis || has_pending_choice
                || has_rival || has_war || has_low_loyalty
                || state.guild.gold < 150.0
                || state.overworld.global_threat_level > 30.0
                || state.completed_quests.len() >= 3
            );

            // Crisis states are always interesting — sample them even off-cadence
            let is_crisis_state = has_crisis || has_war || has_battle
                || (state.overworld.global_threat_level > 60.0);

            let on_cadence = state.tick % config.root_sample_interval == 0;
            let crisis_sample = is_crisis_state && state.tick % (config.root_sample_interval / 3).max(1) == 0;

            let min_tick = config.root_sample_interval; // reduced from 3x — allow earlier roots
            if state.tick >= min_tick && has_tension && phase_has_room
                && (on_cadence || crisis_sample)
            {
                roots.push(RootState {
                    state: state.clone(),
                    seed,
                    wave: 0,
                    difficulty,
                });
                match phase.as_str() {
                    "early" => early_count += 1,
                    "mid" => mid_count += 1,
                    "late" => late_count += 1,
                    _ => {}
                }
            }

            if result.outcome.is_some() {
                break;
            }
        }
    }

    eprintln!("  Root phase distribution: early={}, mid={}, late={}", early_count, mid_count, late_count);

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
/// record samples. Returns (samples, leaves) where leaves include value for
/// extreme-preserving clustering.
fn expand_root(
    root: &RootState,
    config: &BfsConfig,
    wave: u32,
) -> (Vec<BfsSample>, Vec<(CampaignState, Vec<f32>, f32, u64, Difficulty)>) {
    // Skip roots where the campaign is effectively over (value ~ 0)
    let root_value = estimate_value(&root.state);
    if root_value < 0.05 {
        return (vec![], vec![]);
    }

    let valid_actions = root.state.valid_actions();
    let grouped = group_actions(&valid_actions);
    let root_tokens = root.state.to_tokens();
    let root_tick = root.state.tick;
    let root_phase = phase_tag(root_tick, root.state.overworld.campaign_progress);

    // Track which strategic buckets have been covered
    let mut bucket_covered: HashMap<&str, bool> = HashMap::new();
    for (action_type, _) in &grouped {
        let bucket = strategic_bucket(action_type);
        bucket_covered.entry(bucket).or_insert(false);
    }

    let mut samples = Vec::new();
    let mut leaves = Vec::new();

    for (action_type, action) in &grouped {
        // Clone state and apply this action
        let mut branch_state = root.state.clone();
        let action_detail = format!("{:?}", action);
        step_campaign(&mut branch_state, Some(action.clone()));

        // Mark bucket as covered
        let bucket = strategic_bucket(action_type);
        bucket_covered.insert(bucket, true);

        // Advance ticks_per_branch with diverse policy for realistic follow-up
        let mut terminal = false;
        let mut outcome_str = None;
        // Vary the branch RNG by action type hash so different actions
        // produce different follow-up trajectories
        let action_hash = action_type.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut branch_rng = root.seed.wrapping_mul(6364136223846793005)
            .wrapping_add(root.state.tick)
            .wrapping_add(action_hash);
        for _ in 0..config.ticks_per_branch {
            let h_action = diverse_root_policy(&branch_state, &mut branch_rng);
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
                Some("Victory") => 5.0, // max value to match non-terminal scale
                Some("Defeat") => 0.0,
                _ => 2.5,
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
            cluster_id: 0,
            difficulty: root.difficulty,
            root_value,
            phase_tag: root_phase.clone(),
        });

        if !terminal {
            leaves.push((branch_state, leaf_features, leaf_value, root.seed, root.difficulty));
        }
    }

    (samples, leaves)
}
