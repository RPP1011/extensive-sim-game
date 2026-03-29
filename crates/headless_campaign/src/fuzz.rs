//! Campaign fuzzer — stress-tests the headless campaign with randomized
//! configs and random action policies to find crashes, invariant violations,
//! hangs, and degenerate states.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::actions::*;
use super::config::CampaignConfig;
use super::state::*;
use super::step::step_campaign;
use super::systems::verify::verify_invariants;

/// A single fuzzer finding — a crash, violation, or degenerate state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuzzFinding {
    pub seed: u64,
    pub tick: u64,
    pub kind: FuzzFindingKind,
    pub description: String,
    /// The config mutation applied (field name → multiplier).
    pub config_mutations: Vec<(String, f64)>,
    /// Action history leading to the finding (last N actions).
    pub action_trace: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FuzzFindingKind {
    Panic,
    InvariantViolation,
    Hang,
    Degenerate,
}

/// Fuzzer configuration.
#[derive(Clone, Debug)]
pub struct FuzzConfig {
    /// Total campaigns to fuzz.
    pub campaigns: u64,
    /// Max ticks per campaign before declaring hang.
    pub max_ticks: u64,
    /// Number of parallel threads (0 = auto).
    pub threads: usize,
    /// Base RNG seed.
    pub base_seed: u64,
    /// How aggressively to mutate config (0.0 = no mutation, 1.0 = wild).
    pub mutation_strength: f64,
    /// Output JSONL file for findings.
    pub output_path: String,
    /// Report progress every N campaigns.
    pub report_interval: u64,
    /// Fraction of campaigns that use purely random actions (vs heuristic with noise).
    pub random_action_ratio: f64,
    /// Max action trace length to keep.
    pub max_trace_len: usize,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            campaigns: 10_000,
            max_ticks: 50_000,
            threads: 0,
            base_seed: 0xF022,
            mutation_strength: 0.5,
            output_path: "generated/fuzz_findings.jsonl".into(),
            report_interval: 500,
            random_action_ratio: 0.5,
            max_trace_len: 50,
        }
    }
}

/// Mutate a campaign config using a seeded RNG.
/// Returns the config and a list of (field_name, multiplier) for reproduction.
fn mutate_config(seed: u64, strength: f64) -> (CampaignConfig, Vec<(String, f64)>) {
    let mut config = CampaignConfig::default();
    let mut mutations = Vec::new();
    let mut rng = seed;

    // Simple xorshift for cheap RNG
    macro_rules! next_f64 {
        () => {{
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 % 1000.0) / 1000.0
        }};
    }

    macro_rules! mutate_f32 {
        ($field:expr, $name:expr) => {{
            let roll = next_f64!();
            if roll < strength {
                // Multiplier between 0.1x and 10x, log-uniform
                let log_mult = (next_f64!() - 0.5) * 2.0 * 2.3; // ±2.3 = ln(10)
                let mult = log_mult.exp();
                $field = ($field as f64 * mult) as f32;
                mutations.push(($name.to_string(), mult));
            }
        }};
    }

    macro_rules! mutate_u64 {
        ($field:expr, $name:expr) => {{
            let roll = next_f64!();
            if roll < strength {
                let log_mult = (next_f64!() - 0.5) * 2.0 * 2.3;
                let mult = log_mult.exp();
                $field = (($field as f64 * mult).max(1.0)) as u64;
                mutations.push(($name.to_string(), mult));
            }
        }};
    }

    macro_rules! mutate_u32 {
        ($field:expr, $name:expr) => {{
            let roll = next_f64!();
            if roll < strength {
                let log_mult = (next_f64!() - 0.5) * 2.0 * 2.3;
                let mult = log_mult.exp();
                $field = (($field as f64 * mult).max(1.0)) as u32;
                mutations.push(($name.to_string(), mult));
            }
        }};
    }

    macro_rules! mutate_usize {
        ($field:expr, $name:expr) => {{
            let roll = next_f64!();
            if roll < strength {
                let log_mult = (next_f64!() - 0.5) * 2.0 * 2.3;
                let mult = log_mult.exp();
                $field = (($field as f64 * mult).max(1.0)) as usize;
                mutations.push(($name.to_string(), mult));
            }
        }};
    }

    // Quest generation
    mutate_u64!(config.quest_generation.base_arrival_interval_ticks, "quest.arrival_interval");
    mutate_f32!(config.quest_generation.base_threat, "quest.base_threat");
    mutate_f32!(config.quest_generation.threat_variance, "quest.threat_variance");
    mutate_f32!(config.quest_generation.min_threat, "quest.min_threat");
    mutate_f32!(config.quest_generation.max_threat, "quest.max_threat");
    mutate_f32!(config.quest_generation.gold_per_threat, "quest.gold_per_threat");
    mutate_f32!(config.quest_generation.rep_per_threat, "quest.rep_per_threat");
    mutate_u64!(config.quest_generation.quest_deadline_ms, "quest.deadline_ms");

    // Combat
    mutate_f32!(config.combat.sigmoid_steepness, "combat.sigmoid_steepness");
    mutate_f32!(config.combat.level_power_bonus, "combat.level_power_bonus");
    mutate_f32!(config.combat.casualty_rate_multiplier, "combat.casualty_rate");

    // Battle
    mutate_u64!(config.battle.default_duration_ticks, "battle.duration");
    mutate_f32!(config.battle.victory_party_damage, "battle.victory_damage");
    mutate_f32!(config.battle.defeat_enemy_damage, "battle.defeat_damage");

    // Adventurer condition
    mutate_u64!(config.adventurer_condition.drift_interval_ticks, "condition.drift_interval");
    mutate_f32!(config.adventurer_condition.desertion_loyalty_threshold, "condition.desertion_loyalty");
    mutate_f32!(config.adventurer_condition.desertion_stress_threshold, "condition.desertion_stress");

    // Recovery
    mutate_u64!(config.adventurer_recovery.recovery_interval_ticks, "recovery.interval");
    mutate_f32!(config.adventurer_recovery.stress_recovery, "recovery.stress");
    mutate_f32!(config.adventurer_recovery.injury_recovery, "recovery.injury");
    mutate_f32!(config.adventurer_recovery.injury_threshold, "recovery.injury_threshold");

    // Supply
    mutate_f32!(config.supply.drain_per_member_per_sec, "supply.drain_rate");

    // Economy
    mutate_f32!(config.economy.passive_gold_per_sec, "economy.passive_gold");
    mutate_f32!(config.economy.runner_cost, "economy.runner_cost");
    mutate_f32!(config.economy.mercenary_cost, "economy.merc_cost");
    mutate_f32!(config.economy.supply_cost_per_unit, "economy.supply_cost");

    // Faction AI
    mutate_u64!(config.faction_ai.decision_interval_ticks, "faction.decision_interval");
    mutate_f32!(config.faction_ai.hostile_strength_gain, "faction.hostile_gain");
    mutate_f32!(config.faction_ai.war_declaration_threshold, "faction.war_threshold");

    // Recruitment
    mutate_u64!(config.recruitment.interval_ticks, "recruit.interval");
    mutate_f32!(config.recruitment.recruit_cost, "recruit.cost");
    mutate_usize!(config.recruitment.max_adventurers, "recruit.max_adventurers");

    // Campaign progress
    mutate_f32!(config.campaign_progress.victory_quest_count, "progress.victory_quests");
    mutate_f32!(config.campaign_progress.reputation_cap, "progress.rep_cap");

    // Quest lifecycle
    mutate_f32!(config.quest_lifecycle.death_chance, "lifecycle.death_chance");
    mutate_f32!(config.quest_lifecycle.incapacitation_threshold, "lifecycle.incap_threshold");
    mutate_f32!(config.quest_lifecycle.defeat_base_injury, "lifecycle.defeat_injury");
    mutate_u32!(config.quest_lifecycle.level_up_xp_multiplier, "lifecycle.level_xp_mult");
    mutate_f32!(config.quest_lifecycle.party_speed, "lifecycle.party_speed");

    // Starting state
    mutate_f32!(config.starting_state.gold, "start.gold");
    mutate_f32!(config.starting_state.supplies, "start.supplies");
    mutate_f32!(config.starting_state.reputation, "start.reputation");
    mutate_u64!(config.starting_state.early_game_protection_ticks, "start.protection_ticks");

    // Threat
    mutate_f32!(config.threat.hostile_faction_multiplier, "threat.hostile_mult");
    mutate_f32!(config.threat.progress_threat_weight, "threat.progress_weight");

    (config, mutations)
}

/// Pick a random action from valid_actions, or None.
fn random_action(state: &CampaignState, rng: &mut u64) -> Option<CampaignAction> {
    // Handle pre-game phases
    if state.phase != CampaignPhase::Playing {
        if let Some(choice) = state.pending_choices.first() {
            // Random option
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            let idx = (*rng as usize) % choice.options.len().max(1);
            return Some(CampaignAction::RespondToChoice {
                choice_id: choice.id,
                option_index: idx,
            });
        }
        if state.phase == CampaignPhase::ChoosingStartingPackage {
            if !state.available_starting_choices.is_empty() {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                let idx = (*rng as usize) % state.available_starting_choices.len();
                return Some(CampaignAction::ChooseStartingPackage {
                    choice: state.available_starting_choices[idx].clone(),
                });
            }
        }
        return None;
    }

    let actions = state.valid_actions();
    if actions.is_empty() {
        return None;
    }
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let idx = (*rng as usize) % actions.len();
    Some(actions[idx].clone())
}

/// Check for degenerate states beyond what verify_invariants catches.
fn check_degenerate(state: &CampaignState) -> Vec<String> {
    let mut issues = Vec::new();

    // All adventurers dead/deserted with no recruitment possible
    if state.adventurers.is_empty() && state.tick > 5000 {
        issues.push("No adventurers remaining after tick 5000".into());
    }

    // Extreme gold accumulation (economy broken)
    if state.guild.gold > 100_000.0 {
        issues.push(format!("Gold overflow: {:.0}", state.guild.gold));
    }

    // NaN/Inf checks on key fields
    if state.guild.gold.is_nan() || state.guild.gold.is_infinite() {
        issues.push(format!("Gold is NaN/Inf: {}", state.guild.gold));
    }
    if state.guild.supplies.is_nan() || state.guild.supplies.is_infinite() {
        issues.push(format!("Supplies NaN/Inf: {}", state.guild.supplies));
    }
    if state.guild.reputation.is_nan() || state.guild.reputation.is_infinite() {
        issues.push(format!("Reputation NaN/Inf: {}", state.guild.reputation));
    }

    // Check adventurer stats for NaN
    for adv in &state.adventurers {
        for (name, val) in [
            ("stress", adv.stress),
            ("fatigue", adv.fatigue),
            ("injury", adv.injury),
            ("loyalty", adv.loyalty),
            ("morale", adv.morale),
        ] {
            if val.is_nan() || val.is_infinite() {
                issues.push(format!("Adventurer {} {}: {}", adv.id, name, val));
            }
        }
    }

    // Global threat NaN
    let threat = state.overworld.global_threat_level;
    if threat.is_nan() || threat.is_infinite() {
        issues.push(format!("Global threat NaN/Inf: {}", threat));
    }

    issues
}

/// Result of fuzzing a single campaign.
struct FuzzRunResult {
    findings: Vec<FuzzFinding>,
    outcome: Option<CampaignOutcome>,
    ticks: u64,
}

/// Fuzz a single campaign.
///
/// Avoids `catch_unwind` — panics propagate and are caught at the thread pool
/// level. Checks invariants every 100 ticks, not every tick. Caps findings
/// per campaign to avoid memory blowup from noisy violations.
fn fuzz_single(seed: u64, config: &FuzzConfig) -> FuzzRunResult {
    const MAX_FINDINGS_PER_CAMPAIGN: usize = 20;
    const INVARIANT_CHECK_INTERVAL: u64 = 3;

    let use_random = {
        let mut r = seed;
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        (r as f64 / u64::MAX as f64) < config.random_action_ratio
    };

    let (campaign_config, mutations) = mutate_config(seed, config.mutation_strength);
    let mut state = CampaignState::with_config(seed, campaign_config);
    let mut findings = Vec::new();
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut last_tick = 0u64;
    let mut stall_count = 0u64;
    // Ring buffer for action trace (avoids Vec::remove(0) shifting)
    let mut action_ring = Vec::with_capacity(config.max_trace_len);

    for _ in 0..config.max_ticks {
        if findings.len() >= MAX_FINDINGS_PER_CAMPAIGN {
            break;
        }

        let action = if use_random {
            random_action(&state, &mut rng)
        } else {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if (rng % 10) < 3 {
                random_action(&state, &mut rng)
            } else {
                super::batch::heuristic_policy(&state)
            }
        };

        // Track action in ring buffer
        let action_str = match &action {
            Some(a) => format!("{:?}", std::mem::discriminant(a)),
            None => "Wait".into(),
        };
        if action_ring.len() < config.max_trace_len {
            action_ring.push(action_str);
        } else {
            let idx = (state.tick as usize) % config.max_trace_len;
            action_ring[idx] = action_str;
        }

        let step_result = step_campaign(&mut state, action);

        // Periodic invariant + degenerate checks (not every tick)
        if state.tick % INVARIANT_CHECK_INTERVAL == 0 {
            let violations = verify_invariants(&state);
            for v in violations {
                if findings.len() >= MAX_FINDINGS_PER_CAMPAIGN { break; }
                findings.push(FuzzFinding {
                    seed,
                    tick: state.tick,
                    kind: FuzzFindingKind::InvariantViolation,
                    description: v,
                    config_mutations: mutations.clone(),
                    action_trace: action_ring.clone(),
                });
            }

            let degens = check_degenerate(&state);
            for d in degens {
                if findings.len() >= MAX_FINDINGS_PER_CAMPAIGN { break; }
                findings.push(FuzzFinding {
                    seed,
                    tick: state.tick,
                    kind: FuzzFindingKind::Degenerate,
                    description: d,
                    config_mutations: mutations.clone(),
                    action_trace: action_ring.clone(),
                });
            }
        }

        // Hang detection
        if state.tick == last_tick {
            stall_count += 1;
            if stall_count > 100 {
                findings.push(FuzzFinding {
                    seed,
                    tick: state.tick,
                    kind: FuzzFindingKind::Hang,
                    description: format!("Tick stalled at {} for 100+ steps", state.tick),
                    config_mutations: mutations.clone(),
                    action_trace: action_ring,
                });
                return FuzzRunResult {
                    findings,
                    outcome: None,
                    ticks: state.tick,
                };
            }
        } else {
            stall_count = 0;
            last_tick = state.tick;
        }

        if let Some(outcome) = step_result.outcome {
            return FuzzRunResult {
                findings,
                outcome: Some(outcome),
                ticks: state.tick,
            };
        }
    }

    // Reached max ticks — that's a hang
    findings.push(FuzzFinding {
        seed,
        tick: state.tick,
        kind: FuzzFindingKind::Hang,
        description: format!("Reached max ticks ({})", config.max_ticks),
        config_mutations: mutations,
        action_trace: action_ring,
    });

    FuzzRunResult {
        findings,
        outcome: None,
        ticks: state.tick,
    }
}

/// Run the campaign fuzzer.
pub fn run_fuzz(config: &FuzzConfig) {
    // Cap threads — fuzzing is CPU+memory intensive, don't starve the system
    let threads = if config.threads == 0 {
        rayon::current_num_threads().min(4)
    } else {
        config.threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(4 * 1024 * 1024) // 4MB stack per thread
        .build()
        .expect("Failed to build thread pool");

    let completed = AtomicU64::new(0);
    let panics = AtomicU64::new(0);
    let violations = AtomicU64::new(0);
    let hangs = AtomicU64::new(0);
    let degenerates = AtomicU64::new(0);
    let victories = AtomicU64::new(0);
    let defeats = AtomicU64::new(0);
    let start = Instant::now();

    // Collect all findings into a shared vec
    let all_findings: Arc<Mutex<Vec<FuzzFinding>>> = Arc::new(Mutex::new(Vec::new()));

    eprintln!(
        "Campaign fuzzer: {} campaigns, {} threads, mutation={:.0}%, random_action={:.0}%",
        config.campaigns,
        threads,
        config.mutation_strength * 100.0,
        config.random_action_ratio * 100.0,
    );

    pool.install(|| {
        (0..config.campaigns).into_par_iter().for_each(|i| {
            let seed = config.base_seed.wrapping_add(i.wrapping_mul(7919));

            // Catch panics per campaign (not per tick)
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fuzz_single(seed, config)
            }));

            match result {
                Ok(result) => {
                    for f in &result.findings {
                        match f.kind {
                            FuzzFindingKind::Panic => { panics.fetch_add(1, Ordering::Relaxed); }
                            FuzzFindingKind::InvariantViolation => { violations.fetch_add(1, Ordering::Relaxed); }
                            FuzzFindingKind::Hang => { hangs.fetch_add(1, Ordering::Relaxed); }
                            FuzzFindingKind::Degenerate => { degenerates.fetch_add(1, Ordering::Relaxed); }
                        }
                    }
                    match result.outcome {
                        Some(CampaignOutcome::Victory) => { victories.fetch_add(1, Ordering::Relaxed); }
                        Some(CampaignOutcome::Defeat { .. }) => { defeats.fetch_add(1, Ordering::Relaxed); }
                        _ => {}
                    }
                    if !result.findings.is_empty() {
                        all_findings.lock().unwrap().extend(result.findings);
                    }
                }
                Err(panic_info) => {
                    panics.fetch_add(1, Ordering::Relaxed);
                    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".into()
                    };
                    let (_, mutations) = mutate_config(seed, config.mutation_strength);
                    all_findings.lock().unwrap().push(FuzzFinding {
                        seed,
                        tick: 0,
                        kind: FuzzFindingKind::Panic,
                        description: msg,
                        config_mutations: mutations,
                        action_trace: vec![],
                    });
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % config.report_interval == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                eprintln!(
                    "[{done}/{total}] {rate:.0}/s | panics={p} violations={v} hangs={h} degenerate={d} | W={w} L={l}",
                    total = config.campaigns,
                    p = panics.load(Ordering::Relaxed),
                    v = violations.load(Ordering::Relaxed),
                    h = hangs.load(Ordering::Relaxed),
                    d = degenerates.load(Ordering::Relaxed),
                    w = victories.load(Ordering::Relaxed),
                    l = defeats.load(Ordering::Relaxed),
                );
            }
        });
    });

    let elapsed = start.elapsed().as_secs_f64();
    let findings = all_findings.lock().unwrap();

    // Write findings
    if !findings.is_empty() {
        if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut file = std::fs::File::create(&config.output_path)
            .expect("Failed to create output file");
        for f in findings.iter() {
            if let Ok(json) = serde_json::to_string(f) {
                writeln!(file, "{}", json).ok();
            }
        }
    }

    eprintln!("\n=== Fuzz Results ({:.1}s) ===", elapsed);
    eprintln!("Campaigns: {}", config.campaigns);
    eprintln!("Rate: {:.0}/s", config.campaigns as f64 / elapsed);
    eprintln!("Victories: {}", victories.load(Ordering::Relaxed));
    eprintln!("Defeats: {}", defeats.load(Ordering::Relaxed));
    eprintln!("Panics: {}", panics.load(Ordering::Relaxed));
    eprintln!("Invariant violations: {}", violations.load(Ordering::Relaxed));
    eprintln!("Hangs: {}", hangs.load(Ordering::Relaxed));
    eprintln!("Degenerate states: {}", degenerates.load(Ordering::Relaxed));
    eprintln!("Total findings: {}", findings.len());
    if !findings.is_empty() {
        eprintln!("Written to: {}", config.output_path);

        // Summary by kind
        let mut by_kind: std::collections::HashMap<String, Vec<&FuzzFinding>> = std::collections::HashMap::new();
        for f in findings.iter() {
            by_kind.entry(format!("{:?}", f.kind)).or_default().push(f);
        }
        eprintln!("\n--- Finding Summary ---");
        for (kind, items) in &by_kind {
            // Deduplicate by description
            let mut descs: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
            for item in items {
                *descs.entry(&item.description).or_default() += 1;
            }
            eprintln!("  {} ({} total):", kind, items.len());
            let mut sorted: Vec<_> = descs.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (desc, count) in sorted.iter().take(10) {
                eprintln!("    [{count}x] {desc}");
            }
        }
    }
}
