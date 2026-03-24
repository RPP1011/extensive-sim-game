//! Heuristic behavioral cloning data pipeline.
//!
//! Records every decision the heuristic policy makes across millions of
//! campaigns as (state_tokens, action, valid_actions, outcome) tuples.
//! Also tracks state space coverage metrics to detect degenerate patterns.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::actions::*;
use super::batch::heuristic_policy;
use super::config::CampaignConfig;
use super::state::*;
use super::step::step_campaign;
use super::tokens::EntityToken;

// ---------------------------------------------------------------------------
// BC sample
// ---------------------------------------------------------------------------

/// A single behavioral cloning training sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BcSample {
    pub seed: u64,
    pub tick: u64,
    /// Entity tokens at the decision point.
    pub tokens: Vec<EntityToken>,
    /// Action the heuristic chose.
    pub action: String,
    /// Index of the chosen action in the valid actions list.
    pub action_index: usize,
    /// Total number of valid actions at this point.
    pub num_valid_actions: usize,
    /// Campaign outcome (filled after campaign completes).
    pub outcome: String,
    /// Starting choice name.
    pub starting_choice: String,
    /// World template (derived from region names).
    pub world_template: String,
}

// ---------------------------------------------------------------------------
// Coverage tracker
// ---------------------------------------------------------------------------

/// Tracks what the heuristic policy explores and where it's blind.
#[derive(Debug, Default)]
pub struct CoverageTracker {
    /// Action type → count of times chosen.
    pub action_type_counts: HashMap<String, u64>,
    /// Action type → count of times available but NOT chosen.
    pub action_type_skipped: HashMap<String, u64>,
    /// (starting_choice, outcome) → count.
    pub outcome_by_start: HashMap<(String, String), u64>,
    /// Tick bucket (0-999, 1000-1999, ...) → action type distribution.
    pub action_by_phase: HashMap<u64, HashMap<String, u64>>,
    /// Quest type accepted vs declined.
    pub quest_type_acceptance: HashMap<String, (u64, u64)>, // (accepted, declined)
    /// Choice option index selected (per choice source category).
    pub choice_option_distribution: HashMap<String, HashMap<usize, u64>>,
    /// How many unique action sequences (first 5 actions) observed.
    pub opening_sequences: HashMap<Vec<String>, u64>,
    /// Campaigns where adventurers died.
    pub death_campaigns: u64,
    /// Campaigns that timed out.
    pub timeout_campaigns: u64,
    /// Total campaigns.
    pub total_campaigns: u64,
    /// Campaigns per world template.
    pub campaigns_by_world: HashMap<String, u64>,
}

impl CoverageTracker {
    fn record_action(
        &mut self,
        action: &CampaignAction,
        valid_actions: &[CampaignAction],
        tick: u64,
    ) {
        let action_type = action_type_name(action);
        *self.action_type_counts.entry(action_type.clone()).or_default() += 1;

        // Track what was available but not chosen
        for va in valid_actions {
            let va_type = action_type_name(va);
            if va_type != action_type {
                *self.action_type_skipped.entry(va_type).or_default() += 1;
            }
        }

        // Phase-based tracking (1000-tick buckets)
        let phase = tick / 1000;
        let phase_map = self.action_by_phase.entry(phase).or_default();
        *phase_map.entry(action_type).or_default() += 1;
    }

    fn record_quest_decision(&mut self, quest_type: &str, accepted: bool) {
        let entry = self.quest_type_acceptance.entry(quest_type.to_string()).or_default();
        if accepted {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    fn record_choice_response(&mut self, source: &str, option_index: usize) {
        let entry = self.choice_option_distribution.entry(source.to_string()).or_default();
        *entry.entry(option_index).or_default() += 1;
    }

    /// Print a coverage report.
    pub fn report(&self) {
        eprintln!("\n======== COVERAGE REPORT ========");
        eprintln!("Total campaigns: {}", self.total_campaigns);
        eprintln!("Deaths: {} ({:.1}%)", self.death_campaigns,
            self.death_campaigns as f64 / self.total_campaigns.max(1) as f64 * 100.0);
        eprintln!("Timeouts: {} ({:.1}%)", self.timeout_campaigns,
            self.timeout_campaigns as f64 / self.total_campaigns.max(1) as f64 * 100.0);

        // Outcome by starting choice
        eprintln!("\n--- Outcome by Starting Choice ---");
        let mut starts: HashMap<String, HashMap<String, u64>> = HashMap::new();
        for ((start, outcome), count) in &self.outcome_by_start {
            *starts.entry(start.clone()).or_default().entry(outcome.clone()).or_default() += count;
        }
        for (start, outcomes) in &starts {
            let total: u64 = outcomes.values().sum();
            let wins = outcomes.get("Victory").copied().unwrap_or(0);
            let losses = outcomes.get("Defeat").copied().unwrap_or(0);
            let timeouts = outcomes.get("Timeout").copied().unwrap_or(0);
            eprintln!("  {}: {} total, {:.1}% win, {:.1}% loss, {:.1}% timeout",
                start, total,
                wins as f64 / total.max(1) as f64 * 100.0,
                losses as f64 / total.max(1) as f64 * 100.0,
                timeouts as f64 / total.max(1) as f64 * 100.0);
        }

        // World template distribution
        eprintln!("\n--- World Template Distribution ---");
        for (world, count) in &self.campaigns_by_world {
            eprintln!("  {}: {}", world, count);
        }

        // Action type frequency
        eprintln!("\n--- Action Type Frequency (chosen / available) ---");
        let total_chosen: u64 = self.action_type_counts.values().sum();
        let mut action_types: Vec<_> = self.action_type_counts.iter().collect();
        action_types.sort_by(|a, b| b.1.cmp(a.1));
        for (action, count) in &action_types {
            let skipped = self.action_type_skipped.get(*action).copied().unwrap_or(0);
            let available = *count + skipped;
            let selection_rate = **count as f64 / available.max(1) as f64 * 100.0;
            eprintln!("  {:<25} chosen {:>6} / available {:>8} ({:>5.1}%)",
                action, count, available, selection_rate);
        }

        // Never-chosen actions
        let chosen_types: std::collections::HashSet<_> = self.action_type_counts.keys().collect();
        let available_types: std::collections::HashSet<_> = self.action_type_skipped.keys().collect();
        let never_chosen: Vec<_> = available_types.difference(&chosen_types).collect();
        if !never_chosen.is_empty() {
            eprintln!("\n  BLIND SPOTS (available but NEVER chosen):");
            for action in &never_chosen {
                let skipped = self.action_type_skipped.get(**action).copied().unwrap_or(0);
                eprintln!("    {} (skipped {} times)", action, skipped);
            }
        }

        // Quest type acceptance
        eprintln!("\n--- Quest Type Acceptance ---");
        for (qt, (accepted, declined)) in &self.quest_type_acceptance {
            let total = accepted + declined;
            eprintln!("  {:<15} accepted {:>5} / total {:>5} ({:.1}%)",
                qt, accepted, total, *accepted as f64 / total.max(1) as f64 * 100.0);
        }

        // Choice option distribution
        eprintln!("\n--- Choice Option Distribution ---");
        for (source, options) in &self.choice_option_distribution {
            let total: u64 = options.values().sum();
            let mut opts: Vec<_> = options.iter().collect();
            opts.sort_by_key(|(idx, _)| *idx);
            let dist: Vec<String> = opts.iter()
                .map(|(idx, count)| format!("opt{}={:.0}%", idx, **count as f64 / total.max(1) as f64 * 100.0))
                .collect();
            eprintln!("  {}: {} ({})", source, total, dist.join(", "));
        }

        // Phase-based action distribution
        eprintln!("\n--- Action Distribution by Phase ---");
        let mut phases: Vec<_> = self.action_by_phase.keys().collect();
        phases.sort();
        for phase in phases.iter().take(8) {
            let actions = &self.action_by_phase[*phase];
            let total: u64 = actions.values().sum();
            let mut sorted: Vec<_> = actions.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let top3: Vec<String> = sorted.iter().take(3)
                .map(|(a, c)| format!("{}={:.0}%", a, **c as f64 / total.max(1) as f64 * 100.0))
                .collect();
            eprintln!("  phase {} (tick {}-{}): {} actions, top: {}",
                phase, *phase * 1000, (*phase + 1) * 1000 - 1, total, top3.join(", "));
        }

        // Opening sequence diversity
        eprintln!("\n--- Opening Diversity ---");
        eprintln!("  Unique opening sequences (first 5 actions): {}", self.opening_sequences.len());
        let mut seqs: Vec<_> = self.opening_sequences.iter().collect();
        seqs.sort_by(|a, b| b.1.cmp(a.1));
        for (seq, count) in seqs.iter().take(5) {
            eprintln!("    {} x{}", seq.join(" → "), count);
        }
    }
}

/// Extract a short action type name from a CampaignAction.
pub fn action_type_name(action: &CampaignAction) -> String {
    match action {
        CampaignAction::Wait => "Wait".into(),
        CampaignAction::Rest => "Rest".into(),
        CampaignAction::AcceptQuest { .. } => "AcceptQuest".into(),
        CampaignAction::DeclineQuest { .. } => "DeclineQuest".into(),
        CampaignAction::AssignToPool { .. } => "AssignToPool".into(),
        CampaignAction::UnassignFromPool { .. } => "UnassignFromPool".into(),
        CampaignAction::DispatchQuest { .. } => "DispatchQuest".into(),
        CampaignAction::PurchaseSupplies { .. } => "PurchaseSupplies".into(),
        CampaignAction::TrainAdventurer { .. } => "TrainAdventurer".into(),
        CampaignAction::EquipGear { .. } => "EquipGear".into(),
        CampaignAction::SendRunner { .. } => "SendRunner".into(),
        CampaignAction::HireMercenary { .. } => "HireMercenary".into(),
        CampaignAction::CallRescue { .. } => "CallRescue".into(),
        CampaignAction::HireScout { .. } => "HireScout".into(),
        CampaignAction::DiplomaticAction { .. } => "DiplomaticAction".into(),
        CampaignAction::UseAbility { .. } => "UseAbility".into(),
        CampaignAction::SetSpendPriority { .. } => "SetSpendPriority".into(),
        CampaignAction::ChooseStartingPackage { .. } => "StartingChoice".into(),
        CampaignAction::RespondToChoice { .. } => "RespondToChoice".into(),
        CampaignAction::ProposeCoalition { .. } => "ProposeCoalition".into(),
        CampaignAction::RequestCoalitionAid { .. } => "RequestCoalitionAid".into(),
        CampaignAction::InterceptChampion { .. } => "InterceptChampion".into(),
    }
}

/// Detect the world template from region names.
fn detect_world_template(state: &CampaignState) -> String {
    if state.overworld.regions.is_empty() {
        return "unknown".into();
    }
    let first_region = &state.overworld.regions[0].name;
    if first_region.contains("Greenhollow") {
        "Frontier".into()
    } else if first_region.contains("Crown") || first_region.contains("Landing") {
        "Civil War".into()
    } else if first_region.contains("Coral") || first_region.contains("Tidebreak") {
        "Archipelago".into()
    } else {
        format!("Unknown({})", first_region)
    }
}

// ---------------------------------------------------------------------------
// BC data generation
// ---------------------------------------------------------------------------

/// Configuration for BC data generation.
#[derive(Clone, Debug)]
pub struct BcConfig {
    pub campaigns: u64,
    pub max_ticks: u64,
    pub threads: usize,
    pub base_seed: u64,
    pub report_interval: u64,
    pub output_path: String,
    pub campaign_config: CampaignConfig,
    /// Record every N-th decision (1 = all, 10 = every 10th).
    /// Reduces output size without losing coverage info.
    pub sample_rate: usize,
}

impl Default for BcConfig {
    fn default() -> Self {
        Self {
            campaigns: 10_000,
            max_ticks: 30_000,
            threads: 0,
            base_seed: 2026,
            report_interval: 1000,
            output_path: "generated/heuristic_bc.jsonl".into(),
            campaign_config: CampaignConfig::default(),
            sample_rate: 1,
        }
    }
}

/// Run BC data generation: play campaigns with heuristic policy,
/// record every decision with tokens, measure coverage.
pub fn run_bc_generation(config: &BcConfig) -> CoverageTracker {
    let threads = if config.threads == 0 {
        rayon::current_num_threads()
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build rayon pool");

    // Output file
    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let writer = Mutex::new(
        std::io::BufWriter::new(
            std::fs::File::create(&config.output_path)
                .expect("Failed to create output file"),
        ),
    );

    let coverage = Mutex::new(CoverageTracker::default());
    let total_samples = AtomicU64::new(0);
    let completed = AtomicU64::new(0);

    eprintln!("=== Heuristic BC Data Generation ===");
    eprintln!("Campaigns: {}", config.campaigns);
    eprintln!("Threads: {}", threads);
    eprintln!("Sample rate: 1/{}", config.sample_rate);
    eprintln!("Output: {}", config.output_path);

    let t0 = std::time::Instant::now();

    let seeds: Vec<u64> = (0..config.campaigns)
        .map(|i| config.base_seed.wrapping_add(i))
        .collect();

    pool.install(|| {
        use rayon::prelude::*;

        seeds.par_iter().for_each(|&seed| {
            let (samples, local_coverage) = run_single_bc_campaign(
                seed,
                config.max_ticks,
                &config.campaign_config,
                config.sample_rate,
            );

            // Write samples
            if !samples.is_empty() {
                use std::io::Write;
                let mut w = writer.lock().unwrap();
                for sample in &samples {
                    if let Ok(json) = serde_json::to_string(sample) {
                        writeln!(w, "{}", json).ok();
                    }
                }
            }

            total_samples.fetch_add(samples.len() as u64, Ordering::Relaxed);

            // Merge coverage
            {
                let mut cov = coverage.lock().unwrap();
                merge_coverage(&mut cov, &local_coverage);
            }

            let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if n % config.report_interval == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "[{}/{}] samples={} rate={:.0} campaigns/s  {:.0} samples/s",
                    n, config.campaigns,
                    total_samples.load(Ordering::Relaxed),
                    n as f64 / elapsed,
                    total_samples.load(Ordering::Relaxed) as f64 / elapsed,
                );
            }
        });
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let total = total_samples.load(Ordering::Relaxed);
    eprintln!("\n=== BC Generation Complete ===");
    eprintln!("Campaigns: {}", config.campaigns);
    eprintln!("Samples: {}", total);
    eprintln!("Rate: {:.0} campaigns/s, {:.0} samples/s",
        config.campaigns as f64 / elapsed, total as f64 / elapsed);
    eprintln!("Output: {} ({:.1} MB)",
        config.output_path,
        std::fs::metadata(&config.output_path).map(|m| m.len() as f64 / 1_000_000.0).unwrap_or(0.0));

    let cov = coverage.into_inner().unwrap();
    cov.report();
    cov
}

/// Run a single campaign, recording BC samples and coverage.
fn run_single_bc_campaign(
    seed: u64,
    max_ticks: u64,
    config: &CampaignConfig,
    sample_rate: usize,
) -> (Vec<BcSample>, CoverageTracker) {
    let mut state = CampaignState::with_config(seed, config.clone());
    let mut samples = Vec::new();
    let mut coverage = CoverageTracker::default();
    let mut decision_count = 0usize;
    let mut opening_actions = Vec::new();
    let mut had_death = false;

    let world_template = detect_world_template(&state);
    let mut starting_choice_name = String::new();

    loop {
        if state.tick > max_ticks {
            break;
        }

        let action = heuristic_policy(&state);

        if let Some(ref action) = action {
            let valid = state.valid_actions();
            let action_type = action_type_name(action);

            // Track coverage
            coverage.record_action(action, &valid, state.tick);

            // Track quest acceptance
            match action {
                CampaignAction::AcceptQuest { request_id } => {
                    if let Some(req) = state.request_board.iter().find(|r| r.id == *request_id) {
                        coverage.record_quest_decision(&format!("{:?}", req.quest_type), true);
                    }
                }
                CampaignAction::DeclineQuest { request_id } => {
                    if let Some(req) = state.request_board.iter().find(|r| r.id == *request_id) {
                        coverage.record_quest_decision(&format!("{:?}", req.quest_type), false);
                    }
                }
                CampaignAction::RespondToChoice { choice_id, option_index } => {
                    if let Some(choice) = state.pending_choices.iter().find(|c| c.id == *choice_id) {
                        let source = format!("{:?}", choice.source).split('{').next().unwrap_or("unknown").trim().to_string();
                        coverage.record_choice_response(&source, *option_index);
                    }
                }
                CampaignAction::ChooseStartingPackage { choice } => {
                    starting_choice_name = choice.name.clone();
                }
                _ => {}
            }

            // Track opening
            if opening_actions.len() < 5 {
                opening_actions.push(action_type.clone());
            }

            // Find action index in valid actions
            let action_index = valid.iter().position(|va| {
                action_type_name(va) == action_type
            }).unwrap_or(0);

            // Record sample (at sample_rate)
            decision_count += 1;
            if decision_count % sample_rate == 0 {
                samples.push(BcSample {
                    seed,
                    tick: state.tick,
                    tokens: state.to_tokens(),
                    action: action_type,
                    action_index,
                    num_valid_actions: valid.len(),
                    outcome: String::new(), // filled later
                    starting_choice: starting_choice_name.clone(),
                    world_template: world_template.clone(),
                });
            }
        }

        let result = step_campaign(&mut state, action);

        // Track deaths
        for event in &result.events {
            if matches!(event, WorldEvent::AdventurerDied { .. }) {
                had_death = true;
            }
        }

        if let Some(outcome) = result.outcome {
            let outcome_str = format!("{:?}", outcome);

            // Fill outcome in all samples
            for s in &mut samples {
                s.outcome = outcome_str.clone();
            }

            // Coverage
            coverage.outcome_by_start
                .entry((starting_choice_name.clone(), outcome_str.clone()))
                .and_modify(|c| *c += 1)
                .or_insert(1);
            if had_death {
                coverage.death_campaigns += 1;
            }
            if outcome == CampaignOutcome::Timeout {
                coverage.timeout_campaigns += 1;
            }
            break;
        }
    }

    coverage.total_campaigns = 1;
    *coverage.campaigns_by_world.entry(world_template).or_default() += 1;

    if opening_actions.len() >= 5 {
        *coverage.opening_sequences.entry(opening_actions).or_default() += 1;
    }

    (samples, coverage)
}

/// Merge local coverage into global.
fn merge_coverage(global: &mut CoverageTracker, local: &CoverageTracker) {
    for (k, v) in &local.action_type_counts {
        *global.action_type_counts.entry(k.clone()).or_default() += v;
    }
    for (k, v) in &local.action_type_skipped {
        *global.action_type_skipped.entry(k.clone()).or_default() += v;
    }
    for (k, v) in &local.outcome_by_start {
        *global.outcome_by_start.entry(k.clone()).or_default() += v;
    }
    for (phase, actions) in &local.action_by_phase {
        let entry = global.action_by_phase.entry(*phase).or_default();
        for (a, c) in actions {
            *entry.entry(a.clone()).or_default() += c;
        }
    }
    for (qt, (a, d)) in &local.quest_type_acceptance {
        let entry = global.quest_type_acceptance.entry(qt.clone()).or_default();
        entry.0 += a;
        entry.1 += d;
    }
    for (src, opts) in &local.choice_option_distribution {
        let entry = global.choice_option_distribution.entry(src.clone()).or_default();
        for (idx, c) in opts {
            *entry.entry(*idx).or_default() += c;
        }
    }
    for (seq, c) in &local.opening_sequences {
        *global.opening_sequences.entry(seq.clone()).or_default() += c;
    }
    global.death_campaigns += local.death_campaigns;
    global.timeout_campaigns += local.timeout_campaigns;
    global.total_campaigns += local.total_campaigns;
    for (w, c) in &local.campaigns_by_world {
        *global.campaigns_by_world.entry(w.clone()).or_default() += c;
    }
}
