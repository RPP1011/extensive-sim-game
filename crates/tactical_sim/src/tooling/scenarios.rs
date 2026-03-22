use std::collections::HashMap;

use crate::control as ai_phase4;
use crate::sim::{ReplayResult, SimEvent, Team};
use crate::personality::{run_phase5_with_personality_overrides, PersonalityProfile};
use crate::roles as ai_phase2;
use crate::squad as ai_phase3;

use super::types::{CcChainMetrics, ScenarioSummary, TuningResult};

pub fn run_scenario_matrix() -> Vec<ScenarioSummary> {
    let scenarios = vec![
        ("phase2_seed17", 2_u8, 17_u64, 260_u32),
        ("phase3_seed23", 3_u8, 23_u64, 280_u32),
        ("phase4_seed29", 4_u8, 29_u64, 320_u32),
        ("phase5_seed31", 5_u8, 31_u64, 320_u32),
    ];

    let mut out = Vec::new();
    for (name, phase, seed, ticks) in scenarios {
        let (a_event_hash, a_state_hash, winner, tfd, casts, heals) = match phase {
            2 => {
                let a = ai_phase2::run_phase2_sample(seed, ticks, crate::sim::FIXED_TICK_MS);
                (
                    a.event_log_hash,
                    a.final_state_hash,
                    format!("{:?}", a.metrics.winner),
                    a.metrics.tick_to_first_death,
                    a.metrics.casts_completed,
                    a.metrics.heals_completed,
                )
            }
            3 => {
                let a = ai_phase3::run_phase3_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                (
                    a.event_log_hash,
                    a.final_state_hash,
                    format!("{:?}", a.metrics.winner),
                    a.metrics.tick_to_first_death,
                    a.metrics.casts_completed,
                    a.metrics.heals_completed,
                )
            }
            4 => {
                let a = ai_phase4::run_phase4_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                (
                    a.event_log_hash,
                    a.final_state_hash,
                    format!("{:?}", a.metrics.winner),
                    a.metrics.tick_to_first_death,
                    a.metrics.casts_completed,
                    a.metrics.heals_completed,
                )
            }
            _ => {
                let a = crate::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::sim::FIXED_TICK_MS,
                )
                .replay;
                (
                    a.event_log_hash,
                    a.final_state_hash,
                    format!("{:?}", a.metrics.winner),
                    a.metrics.tick_to_first_death,
                    a.metrics.casts_completed,
                    a.metrics.heals_completed,
                )
            }
        };

        let (team_ttk, eliminated_team, hero_deaths, enemy_deaths) = match phase {
            2 => {
                let a = ai_phase2::run_phase2_sample(seed, ticks, crate::sim::FIXED_TICK_MS);
                derive_outcome_metrics(&a)
            }
            3 => {
                let a = ai_phase3::run_phase3_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                derive_outcome_metrics(&a)
            }
            4 => {
                let a = ai_phase4::run_phase4_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                derive_outcome_metrics(&a)
            }
            _ => {
                let a = crate::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::sim::FIXED_TICK_MS,
                )
                .replay;
                derive_outcome_metrics(&a)
            }
        };

        let (b_event_hash, b_state_hash) = match phase {
            2 => {
                let b = ai_phase2::run_phase2_sample(seed, ticks, crate::sim::FIXED_TICK_MS);
                (b.event_log_hash, b.final_state_hash)
            }
            3 => {
                let b = ai_phase3::run_phase3_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                (b.event_log_hash, b.final_state_hash)
            }
            4 => {
                let b = ai_phase4::run_phase4_sample(seed, ticks, crate::sim::FIXED_TICK_MS)
                    .replay;
                (b.event_log_hash, b.final_state_hash)
            }
            _ => {
                let b = crate::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::sim::FIXED_TICK_MS,
                )
                .replay;
                (b.event_log_hash, b.final_state_hash)
            }
        };

        out.push(ScenarioSummary {
            name: name.to_string(),
            winner,
            tick_to_first_death: tfd,
            team_ttk,
            eliminated_team,
            hero_deaths,
            enemy_deaths,
            event_hash: a_event_hash,
            state_hash: a_state_hash,
            deterministic: a_event_hash == b_event_hash && a_state_hash == b_state_hash,
            casts,
            heals,
        });
    }

    out
}

fn derive_outcome_metrics(replay: &ReplayResult) -> (Option<u64>, Option<String>, u32, u32) {
    let mut team_by_unit = HashMap::new();
    let mut alive_hero = 0_i32;
    let mut alive_enemy = 0_i32;
    for unit in &replay.final_state.units {
        team_by_unit.insert(unit.id, unit.team);
        match unit.team {
            Team::Hero => alive_hero += 1,
            Team::Enemy => alive_enemy += 1,
        }
    }

    let mut team_ttk = None;
    let mut eliminated_team = None;
    for event in &replay.events {
        if let SimEvent::UnitDied { tick, unit_id } = *event {
            match team_by_unit.get(&unit_id).copied() {
                Some(Team::Hero) => {
                    alive_hero -= 1;
                    if alive_hero == 0 && team_ttk.is_none() {
                        team_ttk = Some(tick);
                        eliminated_team = Some("Hero".to_string());
                    }
                }
                Some(Team::Enemy) => {
                    alive_enemy -= 1;
                    if alive_enemy == 0 && team_ttk.is_none() {
                        team_ttk = Some(tick);
                        eliminated_team = Some("Enemy".to_string());
                    }
                }
                None => {}
            }
        }
    }

    let hero_deaths = replay
        .final_state
        .units
        .iter()
        .filter(|u| u.team == Team::Hero && u.hp <= 0)
        .count() as u32;
    let enemy_deaths = replay
        .final_state
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp <= 0)
        .count() as u32;

    (team_ttk, eliminated_team, hero_deaths, enemy_deaths)
}

pub fn reservation_timeline_summary(seed: u64, ticks: u32) -> String {
    let run = ai_phase4::run_phase4_sample(seed, ticks, crate::sim::FIXED_TICK_MS);
    let mut windows_by_target: HashMap<u32, Vec<(u64, u64)>> = HashMap::new();
    for w in run.control_windows {
        windows_by_target
            .entry(w.target_id)
            .or_default()
            .push((w.start_tick, w.end_tick));
    }

    let mut lines = Vec::new();
    lines.push("CC Control Timeline".to_string());
    let mut targets = windows_by_target.keys().copied().collect::<Vec<_>>();
    targets.sort_unstable();
    for target in targets {
        let mut ranges = windows_by_target.remove(&target).unwrap_or_default();
        ranges.sort_by_key(|(s, _)| *s);
        let serialized = ranges
            .iter()
            .map(|(s, e)| format!("[{}-{}]", s, e))
            .collect::<Vec<_>>()
            .join(" ");
        lines.push(format!("target {}: {}", target, serialized));
    }
    lines.push(format!(
        "reservation ticks with entries: {}",
        run.reservation_history
            .iter()
            .filter(|v| !v.is_empty())
            .count()
    ));

    lines.join("\n")
}

pub fn analyze_phase4_cc_metrics(seed: u64, ticks: u32) -> CcChainMetrics {
    let run = ai_phase4::run_phase4_sample(seed, ticks, crate::sim::FIXED_TICK_MS);
    let mut count_by_target: HashMap<u32, usize> = HashMap::new();
    for w in &run.control_windows {
        *count_by_target.entry(w.target_id).or_insert(0) += 1;
    }
    let primary_target = count_by_target
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(target, _)| *target);

    let Some(target_id) = primary_target else {
        return CcChainMetrics {
            primary_target: None,
            windows: 0,
            links: 0,
            coverage_ratio: 0.0,
            overlap_ratio: 0.0,
            avg_gap_ticks: 0.0,
        };
    };

    let mut intervals = run
        .control_windows
        .iter()
        .filter(|w| w.target_id == target_id)
        .map(|w| (w.start_tick, w.end_tick))
        .collect::<Vec<_>>();
    intervals.sort_by_key(|(s, e)| (*s, *e));

    if intervals.is_empty() {
        return CcChainMetrics {
            primary_target: Some(target_id),
            windows: 0,
            links: 0,
            coverage_ratio: 0.0,
            overlap_ratio: 0.0,
            avg_gap_ticks: 0.0,
        };
    }

    let raw_sum = intervals
        .iter()
        .map(|(s, e)| e.saturating_sub(*s))
        .sum::<u64>();

    let mut merged = Vec::<(u64, u64)>::new();
    for (start, end) in intervals.iter().copied() {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        } else {
            merged.push((start, end));
        }
    }

    let merged_sum = merged
        .iter()
        .map(|(s, e)| e.saturating_sub(*s))
        .sum::<u64>();
    let overlap = raw_sum.saturating_sub(merged_sum);
    let span = merged
        .last()
        .unwrap()
        .1
        .saturating_sub(merged.first().unwrap().0);
    let coverage_ratio = if span == 0 {
        1.0
    } else {
        merged_sum as f32 / span as f32
    };
    let overlap_ratio = if raw_sum == 0 {
        0.0
    } else {
        overlap as f32 / raw_sum as f32
    };

    let mut links = 0_usize;
    let mut gap_total = 0_u64;
    for pair in intervals.windows(2) {
        let a = pair[0];
        let b = pair[1];
        links += 1;
        gap_total += b.0.saturating_sub(a.1);
    }
    let avg_gap_ticks = if links == 0 {
        0.0
    } else {
        gap_total as f32 / links as f32
    };

    CcChainMetrics {
        primary_target: Some(target_id),
        windows: intervals.len(),
        links,
        coverage_ratio,
        overlap_ratio,
        avg_gap_ticks,
    }
}

pub fn run_personality_grid_tuning() -> Vec<TuningResult> {
    let mut results = Vec::new();
    let vals = [0.35_f32, 0.55, 0.75];

    for aggression in vals {
        for control_bias in vals {
            for altruism in vals {
                let profile = PersonalityProfile {
                    aggression,
                    risk_tolerance: 0.55,
                    discipline: 0.70,
                    control_bias,
                    altruism,
                    patience: 0.60,
                };
                let override_map = (1_u32..=6_u32)
                    .map(|id| (id, profile))
                    .collect::<HashMap<_, _>>();
                let run = run_phase5_with_personality_overrides(
                    31,
                    260,
                    crate::sim::FIXED_TICK_MS,
                    override_map,
                );

                let hero_hp = run
                    .replay
                    .final_state
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Hero)
                    .map(|u| u.hp.max(0))
                    .sum::<i32>();
                let enemy_hp = run
                    .replay
                    .final_state
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Enemy)
                    .map(|u| u.hp.max(0))
                    .sum::<i32>();
                let winner_bonus = match run.replay.metrics.winner {
                    Some(Team::Hero) => 120,
                    Some(Team::Enemy) => -120,
                    _ => 0,
                };
                let score = hero_hp - enemy_hp + winner_bonus;

                results.push(TuningResult {
                    aggression,
                    control_bias,
                    altruism,
                    score,
                    event_hash: run.replay.event_log_hash,
                    winner: format!("{:?}", run.replay.metrics.winner),
                });
            }
        }
    }

    results.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.event_hash.cmp(&b.event_hash))
    });
    results
}
