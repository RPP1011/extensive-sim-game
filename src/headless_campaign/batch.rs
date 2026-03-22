//! Batch campaign runner for large-scale headless simulation.
//!
//! Runs N campaigns in parallel with a simple heuristic policy (no LFM).
//! Used for:
//! - Validating the sim works at scale (100K+ runs)
//! - Generating training data for MCTS bootstrap
//! - Smoke-testing balance before involving learned policies

use std::sync::atomic::{AtomicU64, Ordering};

use super::actions::*;
use super::config::CampaignConfig;
use super::state::*;
use super::step::step_campaign;
use super::trace::{CampaignTrace, TraceRecorder};

/// Result of a single campaign run.
#[derive(Clone, Debug)]
pub struct CampaignRunResult {
    pub seed: u64,
    pub outcome: CampaignOutcome,
    pub ticks: u64,
    pub quests_completed: usize,
    pub quests_failed: usize,
    pub adventurers_lost: usize,
    pub gold_earned: f32,
    pub unlocks_gained: usize,
    pub violations: Vec<String>,
}

/// Summary statistics from a batch of campaigns.
#[derive(Clone, Debug, Default)]
pub struct BatchSummary {
    pub total_runs: u64,
    pub victories: u64,
    pub defeats: u64,
    pub timeouts: u64,
    pub total_ticks: u64,
    pub total_quests_completed: u64,
    pub total_quests_failed: u64,
    pub total_adventurers_lost: u64,
    pub total_violations: u64,
    pub mean_ticks_per_campaign: f64,
    pub mean_quests_completed: f64,
}

/// Configuration for batch runs.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Target number of successful (non-violation) runs.
    pub target_successes: u64,
    /// Maximum ticks per campaign before timeout.
    pub max_ticks: u64,
    /// Number of parallel threads (0 = auto-detect).
    pub threads: usize,
    /// Base seed (each campaign gets `base_seed + run_index`).
    pub base_seed: u64,
    /// How often to print progress.
    pub report_interval: u64,
    /// Record traces for the first N campaigns. 0 = disabled.
    pub record_traces: u64,
    /// Snapshot interval for trace recording (ticks between keyframes).
    pub trace_snapshot_interval: u64,
    /// Output directory for trace files.
    pub trace_output_dir: String,
    /// Campaign balance config (overrides defaults).
    pub campaign_config: CampaignConfig,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            target_successes: 100_000,
            max_ticks: 50_000, // ~83 minutes of game time
            threads: 0,
            base_seed: 2026,
            report_interval: 1000,
            record_traces: 0,
            trace_snapshot_interval: 100,
            trace_output_dir: "generated/traces".into(),
            campaign_config: CampaignConfig::default(),
        }
    }
}

/// Run a single campaign with a simple heuristic policy.
///
/// The heuristic:
/// - Accepts quests with reward/threat > 1.5
/// - Assigns all idle adventurers to the highest-threat accepted quest
/// - Dispatches when pool has 2+ adventurers
/// - Sends runners when party supply < 20%
/// - Calls rescue when battle health < 0.3
/// - Otherwise waits
pub fn run_single_campaign(seed: u64, max_ticks: u64) -> CampaignRunResult {
    run_single_campaign_with_config(seed, max_ticks, &CampaignConfig::default())
}

/// Run a single campaign with a custom config.
pub fn run_single_campaign_with_config(seed: u64, max_ticks: u64, config: &CampaignConfig) -> CampaignRunResult {
    let mut state = CampaignState::with_config(seed, config.clone());
    let mut total_violations = Vec::new();
    let mut adventurers_lost = 0usize;

    for _ in 0..max_ticks {
        // Heuristic policy: choose an action
        let action = heuristic_policy(&state);

        let result = step_campaign(&mut state, action);

        if !result.violations.is_empty() {
            total_violations.extend(result.violations);
        }

        // Track adventurer losses
        for event in &result.events {
            match event {
                WorldEvent::AdventurerDied { .. } | WorldEvent::AdventurerDeserted { .. } => {
                    adventurers_lost += 1;
                }
                _ => {}
            }
        }

        if let Some(outcome) = result.outcome {
            return CampaignRunResult {
                seed,
                outcome,
                ticks: state.tick,
                quests_completed: state.completed_quests.iter().filter(|q| q.result == QuestResult::Victory).count(),
                quests_failed: state.completed_quests.iter().filter(|q| q.result != QuestResult::Victory).count(),
                adventurers_lost,
                gold_earned: state.guild.gold,
                unlocks_gained: state.unlocks.len(),
                violations: total_violations,
            };
        }
    }

    CampaignRunResult {
        seed,
        outcome: CampaignOutcome::Timeout,
        ticks: state.tick,
        quests_completed: state.completed_quests.iter().filter(|q| q.result == QuestResult::Victory).count(),
        quests_failed: state.completed_quests.iter().filter(|q| q.result != QuestResult::Victory).count(),
        adventurers_lost,
        gold_earned: state.guild.gold,
        unlocks_gained: state.unlocks.len(),
        violations: total_violations,
    }
}

/// Run a single campaign with trace recording.
///
/// Returns both the run result and the full trace.
pub fn run_single_campaign_with_trace(
    seed: u64,
    max_ticks: u64,
    snapshot_interval: u64,
) -> (CampaignRunResult, CampaignTrace) {
    run_single_campaign_with_trace_and_config(seed, max_ticks, snapshot_interval, &CampaignConfig::default())
}

/// Run a single campaign with trace recording and custom config.
pub fn run_single_campaign_with_trace_and_config(
    seed: u64,
    max_ticks: u64,
    snapshot_interval: u64,
    config: &CampaignConfig,
) -> (CampaignRunResult, CampaignTrace) {
    let mut state = CampaignState::with_config(seed, config.clone());
    let mut recorder = TraceRecorder::new(seed, snapshot_interval);
    let mut total_violations = Vec::new();
    let mut adventurers_lost = 0usize;

    for _ in 0..max_ticks {
        let action = heuristic_policy(&state);
        let result = recorder.step(&mut state, action);

        if !result.violations.is_empty() {
            total_violations.extend(result.violations);
        }

        for event in &result.events {
            match event {
                WorldEvent::AdventurerDied { .. } | WorldEvent::AdventurerDeserted { .. } => {
                    adventurers_lost += 1;
                }
                _ => {}
            }
        }

        if let Some(outcome) = result.outcome {
            let run_result = CampaignRunResult {
                seed,
                outcome,
                ticks: state.tick,
                quests_completed: state.completed_quests.iter().filter(|q| q.result == QuestResult::Victory).count(),
                quests_failed: state.completed_quests.iter().filter(|q| q.result != QuestResult::Victory).count(),
                adventurers_lost,
                gold_earned: state.guild.gold,
                unlocks_gained: state.unlocks.len(),
                violations: total_violations,
            };
            let trace = recorder.finish(Some(outcome));
            return (run_result, trace);
        }
    }

    let run_result = CampaignRunResult {
        seed,
        outcome: CampaignOutcome::Timeout,
        ticks: state.tick,
        quests_completed: state.completed_quests.iter().filter(|q| q.result == QuestResult::Victory).count(),
        quests_failed: state.completed_quests.iter().filter(|q| q.result != QuestResult::Victory).count(),
        adventurers_lost,
        gold_earned: state.guild.gold,
        unlocks_gained: state.unlocks.len(),
        violations: total_violations,
    };
    let trace = recorder.finish(Some(CampaignOutcome::Timeout));
    (run_result, trace)
}

/// Simple heuristic policy for batch runs.
fn heuristic_policy(state: &CampaignState) -> Option<CampaignAction> {
    // 0. Choose starting package if not initialized
    if !state.initialized {
        if let Some(choice) = state.available_starting_choices.last() {
            return Some(CampaignAction::ChooseStartingPackage {
                choice: choice.clone(),
            });
        }
    }

    // 0.5. Respond to pending choices (pick first non-default option if we can afford it)
    if let Some(choice) = state.pending_choices.first() {
        // Heuristic: pick option 0 (usually the most aggressive/rewarding)
        // unless it costs more gold than we have
        let best = choice.options.iter().enumerate().find(|(_, opt)| {
            opt.effects.iter().all(|e| match e {
                ChoiceEffect::Gold(amount) => state.guild.gold + amount >= 0.0,
                _ => true,
            })
        });
        let idx = best.map(|(i, _)| i).unwrap_or(choice.default_option);
        return Some(CampaignAction::RespondToChoice {
            choice_id: choice.id,
            option_index: idx,
        });
    }

    // 1. Accept quests (lower threshold = more aggressive)
    for req in &state.request_board {
        if state.active_quests.len() < state.guild.active_quest_capacity {
            return Some(CampaignAction::AcceptQuest { request_id: req.id });
        }
    }

    // 2. Assign idle adventurers to preparing quests
    for quest in &state.active_quests {
        if quest.status == ActiveQuestStatus::Preparing {
            for adv in &state.adventurers {
                if adv.status == AdventurerStatus::Idle
                    && !quest.assigned_pool.contains(&adv.id)
                {
                    return Some(CampaignAction::AssignToPool {
                        adventurer_id: adv.id,
                        quest_id: quest.id,
                    });
                }
            }

            // Dispatch if we have 2+ adventurers assigned
            if quest.assigned_pool.len() >= 2 {
                return Some(CampaignAction::DispatchQuest { quest_id: quest.id });
            }
        }
    }

    // 3. Send runner when party supply low
    for party in &state.parties {
        if party.supply_level < 20.0
            && matches!(
                party.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            )
            && state.guild.gold >= RUNNER_COST
        {
            return Some(CampaignAction::SendRunner {
                party_id: party.id,
                payload: RunnerPayload::Supplies(30.0),
            });
        }
    }

    // 4. Rescue when battle is going badly
    for battle in &state.active_battles {
        if battle.status == BattleStatus::Active && battle.party_health_ratio < 0.3 {
            let can_afford = state.guild.gold >= RESCUE_BRIBE_COST;
            let has_free = state.npc_relationships.iter().any(|r| r.rescue_available);
            if can_afford || has_free {
                return Some(CampaignAction::CallRescue {
                    battle_id: battle.id,
                });
            }
        }
    }

    // 5. Accept any remaining quest if we have capacity
    if let Some(req) = state.request_board.first() {
        if state.active_quests.len() < state.guild.active_quest_capacity {
            return Some(CampaignAction::AcceptQuest { request_id: req.id });
        }
    }

    None // Wait
}

/// Run a batch of campaigns in parallel.
///
/// Returns a summary. Prints progress to stderr.
pub fn run_batch(config: &BatchConfig) -> BatchSummary {
    let threads = if config.threads == 0 {
        rayon::current_num_threads()
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build rayon pool");

    let successes = AtomicU64::new(0);
    let total_runs = AtomicU64::new(0);
    let victories = AtomicU64::new(0);
    let defeats = AtomicU64::new(0);
    let timeouts = AtomicU64::new(0);
    let total_ticks = AtomicU64::new(0);
    let total_quests = AtomicU64::new(0);
    let total_violations = AtomicU64::new(0);

    // Create trace output directory if recording
    if config.record_traces > 0 {
        std::fs::create_dir_all(&config.trace_output_dir).ok();
        eprintln!("Recording traces for first {} campaigns to {}", config.record_traces, config.trace_output_dir);
    }

    eprintln!(
        "=== Headless Campaign Batch Runner ===\nTarget: {} successes\nThreads: {}\nMax ticks/campaign: {}",
        config.target_successes, threads, config.max_ticks
    );

    let t0 = std::time::Instant::now();
    let traces_recorded = AtomicU64::new(0);
    let trace_output_dir = config.trace_output_dir.clone();
    let trace_snapshot_interval = config.trace_snapshot_interval;
    let record_traces = config.record_traces;
    let campaign_config = config.campaign_config.clone();

    pool.install(|| {
        use rayon::prelude::*;

        // Generate seeds in chunks
        let chunk_size = (config.target_successes as usize).max(1000);
        let seeds: Vec<u64> = (0..chunk_size as u64 * 2)
            .map(|i| config.base_seed.wrapping_add(i))
            .collect();

        seeds.par_iter().for_each(|&seed| {
            if successes.load(Ordering::Relaxed) >= config.target_successes {
                return;
            }

            // Decide whether to record a trace for this campaign
            let should_trace = record_traces > 0
                && traces_recorded.load(Ordering::Relaxed) < record_traces;

            let result = if should_trace {
                let (run_result, trace) = run_single_campaign_with_trace_and_config(
                    seed,
                    config.max_ticks,
                    trace_snapshot_interval,
                    &campaign_config,
                );
                // Save trace file
                let path = std::path::PathBuf::from(&trace_output_dir)
                    .join(format!("campaign_{}.trace.json", seed));
                if let Err(e) = trace.save_to_file(&path) {
                    eprintln!("  Warning: failed to save trace {}: {}", seed, e);
                } else {
                    traces_recorded.fetch_add(1, Ordering::Relaxed);
                }
                run_result
            } else {
                run_single_campaign_with_config(seed, config.max_ticks, &campaign_config)
            };

            let run_num = total_runs.fetch_add(1, Ordering::Relaxed) + 1;

            total_ticks.fetch_add(result.ticks, Ordering::Relaxed);
            total_quests.fetch_add(result.quests_completed as u64, Ordering::Relaxed);

            if !result.violations.is_empty() {
                total_violations.fetch_add(result.violations.len() as u64, Ordering::Relaxed);
            }

            match result.outcome {
                CampaignOutcome::Victory => {
                    victories.fetch_add(1, Ordering::Relaxed);
                    successes.fetch_add(1, Ordering::Relaxed);
                }
                CampaignOutcome::Defeat => {
                    defeats.fetch_add(1, Ordering::Relaxed);
                    // Defeats without violations still count as "successful" runs
                    if result.violations.is_empty() {
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                }
                CampaignOutcome::Timeout => {
                    timeouts.fetch_add(1, Ordering::Relaxed);
                    if result.violations.is_empty() {
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            if run_num % config.report_interval == 0 {
                let s = successes.load(Ordering::Relaxed);
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = run_num as f64 / elapsed;
                eprintln!(
                    "[{}/{}] successes={} rate={:.0}/s elapsed={:.1}s violations={}",
                    run_num,
                    config.target_successes,
                    s,
                    rate,
                    elapsed,
                    total_violations.load(Ordering::Relaxed),
                );
            }
        });
    });

    let total = total_runs.load(Ordering::Relaxed);
    let elapsed = t0.elapsed().as_secs_f64();

    let summary = BatchSummary {
        total_runs: total,
        victories: victories.load(Ordering::Relaxed),
        defeats: defeats.load(Ordering::Relaxed),
        timeouts: timeouts.load(Ordering::Relaxed),
        total_ticks: total_ticks.load(Ordering::Relaxed),
        total_quests_completed: total_quests.load(Ordering::Relaxed),
        total_quests_failed: 0,
        total_adventurers_lost: 0,
        total_violations: total_violations.load(Ordering::Relaxed),
        mean_ticks_per_campaign: if total > 0 {
            total_ticks.load(Ordering::Relaxed) as f64 / total as f64
        } else {
            0.0
        },
        mean_quests_completed: if total > 0 {
            total_quests.load(Ordering::Relaxed) as f64 / total as f64
        } else {
            0.0
        },
    };

    eprintln!("\n=== Batch Complete ===");
    eprintln!("Total runs: {}", summary.total_runs);
    eprintln!(
        "Outcomes: {} victories, {} defeats, {} timeouts",
        summary.victories, summary.defeats, summary.timeouts
    );
    eprintln!("Mean ticks/campaign: {:.0}", summary.mean_ticks_per_campaign);
    eprintln!("Mean quests completed: {:.1}", summary.mean_quests_completed);
    eprintln!("Total violations: {}", summary.total_violations);
    eprintln!("Rate: {:.0} campaigns/s", total as f64 / elapsed);
    eprintln!("Total time: {:.1}s", elapsed);

    summary
}
