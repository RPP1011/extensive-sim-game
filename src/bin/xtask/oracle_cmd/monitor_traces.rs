//! Offline stream monitor for saved RL episode traces.
//!
//! Reads JSONL episode files, reconstructs per-tick SimState snapshots,
//! and feeds them through `SimMonitor::observe()` to check temporal properties
//! without re-running simulations.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use game::ai::core::monitor::SimMonitor;
use game::ai::core::{step, SimState, UnitIntent, Team, FIXED_TICK_MS};

use serde::Deserialize;

/// Minimal episode record for trace replay.
/// Matches the JSONL format produced by `transformer-rl generate`.
#[derive(Debug, Deserialize)]
struct EpisodeRecord {
    scenario: Option<String>,
    initial_state: Option<SimState>,
    steps: Option<Vec<EpisodeStep>>,
}

#[derive(Debug, Deserialize)]
struct EpisodeStep {
    #[serde(default)]
    intents: Vec<UnitIntent>,
}

pub fn run_monitor_traces(paths: &[PathBuf], sample_pct: f32) -> ExitCode {
    let mut total_episodes = 0u64;
    let mut monitored_episodes = 0u64;
    let mut total_violations = 0u64;
    let mut violation_summary: Vec<(String, String, u64, usize)> = Vec::new(); // (scenario, property, tick, count)

    let jsonl_files = collect_jsonl_files(paths);
    if jsonl_files.is_empty() {
        eprintln!("No JSONL files found in provided paths");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "Monitoring {} JSONL files (sample={:.0}%)",
        jsonl_files.len(),
        sample_pct * 100.0
    );

    for path in &jsonl_files {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open {}: {e}", path.display());
                continue;
            }
        };
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }

            total_episodes += 1;

            // Sampling: use deterministic hash of episode index
            if sample_pct < 1.0 {
                let hash = simple_hash(total_episodes);
                if (hash % 10000) as f32 / 10000.0 >= sample_pct {
                    continue;
                }
            }

            let episode: EpisodeRecord = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let scenario_name = episode
                .scenario
                .as_deref()
                .unwrap_or("unknown")
                .to_string();

            let Some(initial_state) = episode.initial_state else {
                continue;
            };
            let steps = episode.steps.unwrap_or_default();
            if steps.is_empty() {
                continue;
            }

            monitored_episodes += 1;

            // Replay through monitor
            let mut monitor = SimMonitor::new(&initial_state, 100);
            let mut sim = initial_state;

            for ep_step in &steps {
                let (new_sim, events) = step(sim, &ep_step.intents, FIXED_TICK_MS);
                sim = new_sim;
                let new_violations = monitor.observe(&sim, &events, FIXED_TICK_MS);

                for v in new_violations {
                    total_violations += 1;
                    violation_summary.push((
                        scenario_name.clone(),
                        v.property.to_string(),
                        v.tick,
                        1,
                    ));
                }

                // Stop if fight is over
                let heroes_alive = sim
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Hero && u.hp > 0)
                    .count();
                let enemies_alive = sim
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Enemy && u.hp > 0)
                    .count();
                if heroes_alive == 0 || enemies_alive == 0 {
                    break;
                }
            }
        }
    }

    // Report
    eprintln!();
    eprintln!("=== Monitor Traces Report ===");
    eprintln!(
        "Episodes: {} total, {} monitored",
        total_episodes, monitored_episodes
    );
    eprintln!("Violations: {}", total_violations);

    if total_violations > 0 {
        // Aggregate by property type
        let mut by_property: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (_, prop, _, _) in &violation_summary {
            *by_property.entry(prop.clone()).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = by_property.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        eprintln!();
        eprintln!("By property:");
        for (prop, count) in &sorted {
            eprintln!("  {}: {}", prop, count);
        }

        // Show first few violations
        eprintln!();
        eprintln!("First violations (up to 10):");
        for (scenario, prop, tick, _) in violation_summary.iter().take(10) {
            eprintln!("  [{}] tick={}: {}", scenario, tick, prop);
        }
    }

    if total_violations > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn collect_jsonl_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for path in paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                        result.push(p);
                    }
                }
            }
        } else if path.exists() {
            result.push(path.clone());
        }
    }
    result.sort();
    result
}

fn simple_hash(x: u64) -> u64 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h
}
