use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io;

use serde::{Deserialize, Serialize};

use crate::ai::control as ai_phase4;
use crate::ai::core::{
    distance, move_away, position_at_range, run_replay, sim_vec2, step, IntentAction, ReplayResult,
    SimEvent, SimState, Team, UnitIntent, UnitState,
};
use crate::ai::pathing::GridNav;
use crate::ai::personality::{
    default_personalities, generate_scripted_intents, run_phase5_with_personality_overrides,
    sample_phase5_party_state, PersonalityProfile, UnitMode,
};
use crate::ai::roles as ai_phase2;
use crate::ai::squad as ai_phase3;

#[derive(Debug, Clone)]
pub struct ActionScoreDebug {
    pub action: IntentAction,
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct UnitDecisionDebug {
    pub unit_id: u32,
    pub mode: UnitMode,
    pub chosen: IntentAction,
    pub top_k: Vec<ActionScoreDebug>,
}

#[derive(Debug, Clone)]
pub struct TickDecisionDebug {
    pub tick: u64,
    pub decisions: Vec<UnitDecisionDebug>,
}

#[derive(Debug, Clone)]
pub struct ScenarioSummary {
    pub name: String,
    pub winner: String,
    pub tick_to_first_death: Option<u64>,
    pub team_ttk: Option<u64>,
    pub eliminated_team: Option<String>,
    pub hero_deaths: u32,
    pub enemy_deaths: u32,
    pub event_hash: u64,
    pub state_hash: u64,
    pub deterministic: bool,
    pub casts: u32,
    pub heals: u32,
}

#[derive(Debug, Clone)]
pub struct CcChainMetrics {
    pub primary_target: Option<u32>,
    pub windows: usize,
    pub links: usize,
    pub coverage_ratio: f32,
    pub overlap_ratio: f32,
    pub avg_gap_ticks: f32,
}

#[derive(Debug, Clone)]
pub struct TuningResult {
    pub aggression: f32,
    pub control_bias: f32,
    pub altruism: f32,
    pub score: i32,
    pub event_hash: u64,
    pub winner: String,
}

pub fn build_phase5_debug(seed: u64, ticks: u32, top_k: usize) -> Vec<TickDecisionDebug> {
    let initial = sample_phase5_party_state(seed);
    let personalities = default_personalities();
    let roles = crate::ai::roles::default_roles();
    let (script, mode_history) = generate_scripted_intents(
        &initial,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
        roles,
        personalities.clone(),
    );

    let mut state = initial;
    let mut out = Vec::with_capacity(ticks as usize);

    for tick in 0..ticks as usize {
        let intents = script.get(tick).cloned().unwrap_or_default();
        let modes = mode_history.get(tick).cloned().unwrap_or_default();

        let mut mode_by_unit = HashMap::new();
        for (unit_id, mode) in modes {
            mode_by_unit.insert(unit_id, mode);
        }

        let mut decisions = Vec::new();
        for intent in &intents {
            let unit_id = intent.unit_id;
            let Some(unit) = state.units.iter().find(|u| u.id == unit_id && u.hp > 0) else {
                continue;
            };
            let mode = *mode_by_unit.get(&unit_id).unwrap_or(&UnitMode::Aggressive);
            let p = *personalities
                .get(&unit_id)
                .unwrap_or(&PersonalityProfile::vanguard());
            let mut candidates = score_candidates(&state, unit, mode, p);
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| format!("{:?}", b.action).cmp(&format!("{:?}", a.action)))
            });
            candidates.truncate(top_k);
            decisions.push(UnitDecisionDebug {
                unit_id,
                mode,
                chosen: intent.action,
                top_k: candidates,
            });
        }

        out.push(TickDecisionDebug {
            tick: state.tick,
            decisions,
        });

        let (new_state, _) = step(state, &intents, crate::ai::core::FIXED_TICK_MS);
        state = new_state;
    }

    out
}

fn score_candidates(
    state: &crate::ai::core::SimState,
    unit: &crate::ai::core::UnitState,
    mode: UnitMode,
    p: PersonalityProfile,
) -> Vec<ActionScoreDebug> {
    let mut out = Vec::new();

    out.push(ActionScoreDebug {
        action: IntentAction::Hold,
        score: -2.0 + p.patience * 2.0,
        reason: "idle_patience".to_string(),
    });

    let nearest_enemy = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team != unit.team)
        .min_by(|a, b| {
            distance(unit.position, a.position)
                .partial_cmp(&distance(unit.position, b.position))
                .unwrap_or(Ordering::Equal)
        });

    if let Some(enemy) = nearest_enemy {
        let dist = distance(unit.position, enemy.position);
        let attack_score = 10.0 + p.aggression * 8.0 - dist * 2.5;
        out.push(ActionScoreDebug {
            action: IntentAction::Attack {
                target_id: enemy.id,
            },
            score: attack_score,
            reason: "attack_pressure".to_string(),
        });

        let ability_ready = unit.ability_cooldown_remaining_ms == 0 && unit.ability_damage > 0;
        let ability_score = 12.0 + p.control_bias * 7.0 - dist * 1.8;
        if ability_ready {
            out.push(ActionScoreDebug {
                action: IntentAction::CastAbility {
                    target_id: enemy.id,
                },
                score: ability_score,
                reason: "ability_ready".to_string(),
            });
        }

        let max_step = unit.move_speed_per_sec * 0.1;
        let move_pos = match mode {
            UnitMode::Defensive => move_away(unit.position, enemy.position, max_step),
            _ => position_at_range(unit.position, enemy.position, unit.attack_range * 0.9),
        };
        let mode_bias = match mode {
            UnitMode::Aggressive => 3.5,
            UnitMode::Defensive => 4.5,
            UnitMode::Protector => 2.0,
            UnitMode::Controller => 2.8,
        };
        out.push(ActionScoreDebug {
            action: IntentAction::MoveTo { position: move_pos },
            score: 6.0 + mode_bias,
            reason: "reposition_mode".to_string(),
        });
    }

    if unit.heal_amount > 0 {
        let weak_ally = state
            .units
            .iter()
            .filter(|u| u.hp > 0 && u.team == unit.team)
            .map(|ally| {
                (
                    ally,
                    ally.hp as f32 / ally.max_hp.max(1) as f32,
                    distance(unit.position, ally.position),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((ally, hp_pct, distance)) = weak_ally {
            let score = 14.0 + p.altruism * 10.0 - hp_pct * 8.0 - distance;
            out.push(ActionScoreDebug {
                action: IntentAction::CastHeal { target_id: ally.id },
                score,
                reason: "triage_low_hp".to_string(),
            });
        }
    }

    out
}

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
                let a = ai_phase2::run_phase2_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
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
                let a = ai_phase3::run_phase3_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
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
                let a = ai_phase4::run_phase4_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
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
                let a = crate::ai::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::ai::core::FIXED_TICK_MS,
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
                let a = ai_phase2::run_phase2_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
                derive_outcome_metrics(&a)
            }
            3 => {
                let a = ai_phase3::run_phase3_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
                    .replay;
                derive_outcome_metrics(&a)
            }
            4 => {
                let a = ai_phase4::run_phase4_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
                    .replay;
                derive_outcome_metrics(&a)
            }
            _ => {
                let a = crate::ai::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::ai::core::FIXED_TICK_MS,
                )
                .replay;
                derive_outcome_metrics(&a)
            }
        };

        let (b_event_hash, b_state_hash) = match phase {
            2 => {
                let b = ai_phase2::run_phase2_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
                (b.event_log_hash, b.final_state_hash)
            }
            3 => {
                let b = ai_phase3::run_phase3_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
                    .replay;
                (b.event_log_hash, b.final_state_hash)
            }
            4 => {
                let b = ai_phase4::run_phase4_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS)
                    .replay;
                (b.event_log_hash, b.final_state_hash)
            }
            _ => {
                let b = crate::ai::personality::run_phase5_sample(
                    seed,
                    ticks,
                    crate::ai::core::FIXED_TICK_MS,
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
    let run = ai_phase4::run_phase4_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
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
    let run = ai_phase4::run_phase4_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
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
                    crate::ai::core::FIXED_TICK_MS,
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

fn event_row(event: &SimEvent) -> (u64, String, String, String, String, String) {
    match *event {
        SimEvent::Moved {
            tick,
            unit_id,
            from_x100,
            from_y100,
            to_x100,
            to_y100,
        } => (
            tick,
            "Moved".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            format!(
                "({}, {}) -> ({}, {})",
                from_x100, from_y100, to_x100, to_y100
            ),
        ),
        SimEvent::CastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "CastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "attack cast start".to_string(),
        ),
        SimEvent::AbilityCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "ability cast start".to_string(),
        ),
        SimEvent::HealCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "heal cast start".to_string(),
        ),
        SimEvent::ControlCastStarted {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlCastStarted".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "control cast start".to_string(),
        ),
        SimEvent::ControlApplied {
            tick,
            source_id,
            target_id,
            duration_ms,
        } => (
            tick,
            "ControlApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            duration_ms.to_string(),
            format!("control for {}ms", duration_ms),
        ),
        SimEvent::UnitControlled { tick, unit_id } => (
            tick,
            "UnitControlled".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            "unit action locked".to_string(),
        ),
        SimEvent::DamageApplied {
            tick,
            source_id,
            target_id,
            amount,
            target_hp_after,
            ..
        } => (
            tick,
            "DamageApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            amount.to_string(),
            format!("target hp -> {}", target_hp_after),
        ),
        SimEvent::HealApplied {
            tick,
            source_id,
            target_id,
            amount,
            target_hp_after,
            ..
        } => (
            tick,
            "HealApplied".to_string(),
            source_id.to_string(),
            target_id.to_string(),
            amount.to_string(),
            format!("target hp -> {}", target_hp_after),
        ),
        SimEvent::UnitDied { tick, unit_id } => (
            tick,
            "UnitDied".to_string(),
            unit_id.to_string(),
            "-".to_string(),
            "-".to_string(),
            "unit died".to_string(),
        ),
        SimEvent::AttackBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "AttackBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::AbilityBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "AbilityBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::HealBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "HealBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::ControlBlockedCooldown {
            tick,
            unit_id,
            target_id,
            cooldown_remaining_ms,
        } => (
            tick,
            "ControlBlockedCooldown".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            format!("cd {}", cooldown_remaining_ms),
        ),
        SimEvent::AttackBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AttackBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::AbilityBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::HealBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::ControlBlockedInvalidTarget {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlBlockedInvalidTarget".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "invalid target".to_string(),
        ),
        SimEvent::AttackRepositioned {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AttackRepositioned".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "reposition".to_string(),
        ),
        SimEvent::AbilityBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "AbilityBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
        SimEvent::CastFailedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "CastFailedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "cast failed".to_string(),
        ),
        SimEvent::HealBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "HealBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
        SimEvent::ControlBlockedOutOfRange {
            tick,
            unit_id,
            target_id,
        } => (
            tick,
            "ControlBlockedOutOfRange".to_string(),
            unit_id.to_string(),
            target_id.to_string(),
            "-".to_string(),
            "out of range".to_string(),
        ),
    }
}

fn build_event_rows(replay: &ReplayResult) -> String {
    let mut rows = String::new();
    for event in &replay.events {
        let (tick, kind, src, dst, value, detail) = event_row(event);
        rows.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            tick, kind, src, dst, value, detail
        ));
    }
    rows
}

fn build_frame_rows(
    initial: &crate::ai::core::SimState,
    script: &[Vec<UnitIntent>],
    dt_ms: u32,
) -> String {
    let mut frame_rows = String::new();
    let mut state = initial.clone();
    for unit in &state.units {
        frame_rows.push_str(&format!(
            "{}\t{}\t{:?}\t{}\t{:.3}\t{:.3}\n",
            state.tick, unit.id, unit.team, unit.hp, unit.position.x, unit.position.y
        ));
    }
    for intents in script {
        let (new_state, _) = step(state, intents, dt_ms);
        state = new_state;
        for unit in &state.units {
            frame_rows.push_str(&format!(
                "{}\t{}\t{:?}\t{}\t{:.3}\t{:.3}\n",
                state.tick, unit.id, unit.team, unit.hp, unit.position.x, unit.position.y
            ));
        }
    }
    frame_rows
}

fn obstacle_rows_from_nav_cells(nav: &GridNav) -> String {
    let mut cells = nav.blocked.iter().copied().collect::<Vec<_>>();
    cells.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut rows = String::new();
    for (cx, cy) in cells {
        let min_x = nav.min_x + cx as f32 * nav.cell_size;
        let max_x = min_x + nav.cell_size;
        let min_y = nav.min_y + cy as f32 * nav.cell_size;
        let max_y = min_y + nav.cell_size;
        rows.push_str(&format!(
            "{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            min_x, max_x, min_y, max_y
        ));
    }
    rows
}

fn build_visualization_html(
    title: &str,
    subtitle: &str,
    replay: &ReplayResult,
    event_rows: &str,
    frame_rows: &str,
    obstacle_rows: &str,
    seed: u64,
    ticks: u32,
) -> String {
    let max_tick = replay.final_state.tick;

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{title}</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 16px; background: #111318; color: #e8eaf0; }}
  .header {{ display:flex; align-items:flex-end; justify-content:space-between; gap: 10px; margin-bottom: 8px; }}
  .tabs {{ display:flex; gap:6px; margin: 8px 0 10px; }}
  .tab-btn {{ background:#1b2233; border:1px solid #3a4259; color:#dbe4ff; padding:6px 10px; cursor:pointer; }}
  .tab-btn.active {{ background:#2a3552; }}
  .pane {{ display:none; }}
  .pane.active {{ display:block; }}
  .controls {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px; }}
  .map-controls {{ display: grid; grid-template-columns: 1fr auto auto auto; gap: 8px; align-items: center; margin: 8px 0 10px; }}
  input, select {{ background: #1e2230; color: #eef; border: 1px solid #3a4259; padding: 6px 8px; }}
  button {{ background: #1e2230; color: #eef; border: 1px solid #3a4259; padding: 6px 10px; cursor: pointer; }}
  #timeline {{ display: grid; grid-template-columns: repeat(100, 1fr); gap: 1px; margin: 10px 0 16px; }}
  #abilityTimeline {{ display: grid; grid-template-columns: repeat(100, 1fr); gap: 1px; margin: 8px 0 10px; }}
  .bar {{ height: 10px; background: #3f4a67; }}
  .ability-stack {{ display:flex; flex-direction:column; justify-content:flex-end; height:52px; background:#151b2a; }}
  .ability-seg {{ width:100%; }}
  #map-wrap {{ border: 1px solid #2a3042; background: #151927; padding: 8px; margin-bottom: 10px; }}
  #map {{ width: 100%; max-width: 900px; height: 360px; background: #0f1320; border: 1px solid #2f3850; }}
  .legend {{ color: #9ca7c5; font-size: 12px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border-bottom: 1px solid #2a3042; text-align: left; padding: 4px 6px; }}
  tr:hover {{ background: #1d2232; }}
  .meta {{ color: #9ca7c5; margin-bottom: 8px; }}
  .kpis {{ display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; margin: 10px 0; }}
  .kpi {{ background:#151b2a; border:1px solid #2a3042; padding:8px; }}
  .kpi .label {{ color:#9ca7c5; font-size:11px; }}
  .kpi .value {{ font-size:18px; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <h2 style="margin:0;">{title}</h2>
      <div class="meta">{subtitle}</div>
    </div>
    <div class="meta">seed={seed}, ticks={ticks}, event_hash={event_hash:016x}, state_hash={state_hash:016x}, max_tick={max_tick}</div>
  </div>
  <div class="tabs">
    <button class="tab-btn active" data-tab="map">Map</button>
    <button class="tab-btn" data-tab="events">Events</button>
    <button class="tab-btn" data-tab="metrics">Metrics</button>
  </div>
  <div id="pane-map" class="pane active">
    <div class="controls">
      <input id="search" placeholder="search detail/kind/source/target" />
      <select id="kind"></select>
      <select id="unit"></select>
    </div>
    <div id="map-wrap">
      <div class="map-controls">
        <input id="tick" type="range" min="0" max="{max_tick}" value="0" />
        <span id="tickLabel">tick 0</span>
        <span id="tickStats"></span>
        <select id="speedSel">
          <option value="220">0.5x</option>
          <option value="120" selected>1x</option>
          <option value="70">1.7x</option>
          <option value="40">3x</option>
        </select>
        <button id="playBtn">play</button>
      </div>
      <div class="map-controls">
        <label>link window</label>
        <input id="depth" type="range" min="1" max="30" value="12" />
        <span id="depthLabel">12</span>
      </div>
      <canvas id="map" width="900" height="360"></canvas>
      <div class="legend">units: blue=Hero, red=Enemy | links: red=damage, green=heal, amber=casts, cyan=reposition | walls: hatched blocks | gate: highlighted opening</div>
    </div>
    <div id="timeline"></div>
    <div class="legend">ability timeline: amber=abilities, green=heals, blue=attacks</div>
    <div id="abilityTimeline"></div>
  </div>
  <div id="pane-events" class="pane">
    <table>
      <thead>
        <tr><th>tick</th><th>kind</th><th>src</th><th>dst</th><th>value</th><th>detail</th></tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div id="pane-metrics" class="pane">
    <div class="kpis">
      <div class="kpi"><div class="label">Winner</div><div class="value">{winner}</div></div>
      <div class="kpi"><div class="label">First Death</div><div class="value">{first_death}</div></div>
      <div class="kpi"><div class="label">Casts Completed</div><div class="value">{casts_completed}</div></div>
      <div class="kpi"><div class="label">Heals Completed</div><div class="value">{heals_completed}</div></div>
      <div class="kpi"><div class="label">Repositions</div><div class="value">{repositions}</div></div>
      <div class="kpi"><div class="label">Invariant Violations</div><div class="value">{invariants}</div></div>
      <div class="kpi"><div class="label">Hero Alive</div><div class="value">{hero_alive}</div></div>
      <div class="kpi"><div class="label">Enemy Alive</div><div class="value">{enemy_alive}</div></div>
    </div>
    <h4>Ability Usage By Unit</h4>
    <table>
      <thead>
        <tr><th>unit</th><th>ability starts</th><th>heal starts</th><th>attack starts</th></tr>
      </thead>
      <tbody id="abilityUnitRows"></tbody>
    </table>
  </div>
<script>
const raw = `{rows}`;
const lines = raw.trim().split('\n').filter(Boolean);
const data = lines.map(line => {{
  const [tick, kind, src, dst, value, ...detail] = line.split('\t');
  return {{ tick: Number(tick), kind, src, dst, value, detail: detail.join('\t') }};
}});
const rawFrames = `{frame_rows}`;
const frameLines = rawFrames.trim().split('\n').filter(Boolean);
const frameData = frameLines.map(line => {{
  const [tick, id, team, hp, x, y] = line.split('\t');
  return {{ tick: Number(tick), id: Number(id), team, hp: Number(hp), x: Number(x), y: Number(y) }};
}});
const rawObstacles = `{obstacle_rows}`;
const obstacleData = rawObstacles.trim() ? rawObstacles.trim().split('\n').map(line => {{
  const [min_x,max_x,min_y,max_y] = line.split('\t').map(Number);
  return {{min_x,max_x,min_y,max_y}};
}}) : [];
const framesByTick = new Map();
for (const f of frameData) {{
  if (!framesByTick.has(f.tick)) framesByTick.set(f.tick, []);
  framesByTick.get(f.tick).push(f);
}}
for (const list of framesByTick.values()) list.sort((a,b) => a.id - b.id);

const kinds = ['ALL', ...new Set(data.map(d => d.kind)).values()];
const units = ['ALL', ...new Set(data.flatMap(d => [d.src, d.dst]).filter(v => v !== '-')).values()].sort((a,b)=>Number(a)-Number(b));
const kindSel = document.getElementById('kind');
const unitSel = document.getElementById('unit');
for (const k of kinds) {{ const o=document.createElement('option'); o.value=k; o.textContent=k; kindSel.appendChild(o); }}
for (const u of units) {{ const o=document.createElement('option'); o.value=u; o.textContent=u; unitSel.appendChild(o); }}

const rowsEl = document.getElementById('rows');
const timelineEl = document.getElementById('timeline');
const abilityTimelineEl = document.getElementById('abilityTimeline');
const abilityUnitRowsEl = document.getElementById('abilityUnitRows');
const searchEl = document.getElementById('search');
const tickEl = document.getElementById('tick');
const tickLabelEl = document.getElementById('tickLabel');
const tickStatsEl = document.getElementById('tickStats');
const playBtn = document.getElementById('playBtn');
const speedSel = document.getElementById('speedSel');
const depthEl = document.getElementById('depth');
const depthLabelEl = document.getElementById('depthLabel');
const mapEl = document.getElementById('map');
const ctx = mapEl.getContext('2d');

const world = frameData.reduce((acc, f) => {{
  acc.minX = Math.min(acc.minX, f.x);
  acc.maxX = Math.max(acc.maxX, f.x);
  acc.minY = Math.min(acc.minY, f.y);
  acc.maxY = Math.max(acc.maxY, f.y);
  return acc;
}}, {{ minX: 0, maxX: 1, minY: 0, maxY: 1 }});
const pad = 0.5;
world.minX -= pad; world.maxX += pad;
world.minY -= pad; world.maxY += pad;
if (Math.abs(world.maxX - world.minX) < 0.001) {{ world.maxX += 1; world.minX -= 1; }}
if (Math.abs(world.maxY - world.minY) < 0.001) {{ world.maxY += 1; world.minY -= 1; }}

function mapToCanvas(x, y) {{
  const w = mapEl.width, h = mapEl.height;
  const px = ((x - world.minX) / (world.maxX - world.minX)) * (w - 40) + 20;
  const py = h - ((((y - world.minY) / (world.maxY - world.minY)) * (h - 40)) + 20);
  return [px, py];
}}

function approxEq(a, b, eps) {{
  return Math.abs(a - b) <= eps;
}}

function detectChokepoints(obstacles) {{
  const eps = 0.12;
  const groups = [];
  for (const o of obstacles) {{
    let group = null;
    for (const g of groups) {{
      if (approxEq(g.min_x, o.min_x, eps) && approxEq(g.max_x, o.max_x, eps)) {{
        group = g;
        break;
      }}
    }}
    if (!group) {{
      group = {{ min_x: o.min_x, max_x: o.max_x, segs: [] }};
      groups.push(group);
    }}
    group.segs.push(o);
  }}

  const gates = [];
  for (const g of groups) {{
    if (g.segs.length < 2) continue;
    g.segs.sort((a, b) => a.min_y - b.min_y);
    for (let i = 0; i < g.segs.length - 1; i++) {{
      const a = g.segs[i];
      const b = g.segs[i + 1];
      const gap = b.min_y - a.max_y;
      if (gap > 0.25) {{
        gates.push({{
          min_x: g.min_x,
          max_x: g.max_x,
          min_y: a.max_y,
          max_y: b.min_y
        }});
      }}
    }}
  }}
  return gates;
}}

const chokepoints = detectChokepoints(obstacleData);

function eventColor(kind) {{
  if (kind === 'DamageApplied') return '#ff6b6b';
  if (kind === 'HealApplied') return '#4ade80';
  if (kind === 'ControlApplied' || kind === 'ControlCastStarted' || kind === 'UnitControlled') return '#c084fc';
  if (kind.includes('CastStarted')) return '#fbbf24';
  if (kind === 'AttackRepositioned' || kind === 'Moved') return '#22d3ee';
  return '#94a3b8';
}}

function drawMap(currentTick) {{
  ctx.clearRect(0, 0, mapEl.width, mapEl.height);
  ctx.fillStyle = '#0f1320';
  ctx.fillRect(0, 0, mapEl.width, mapEl.height);

  // Grid
  ctx.strokeStyle = '#1f273a';
  ctx.lineWidth = 1;
  for (let i=0; i<=10; i++) {{
    const x = 20 + (i/10) * (mapEl.width - 40);
    const y = 20 + (i/10) * (mapEl.height - 40);
    ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, mapEl.height - 20); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(20, y); ctx.lineTo(mapEl.width - 20, y); ctx.stroke();
  }}

  for (const o of obstacleData) {{
    const [ax, ay] = mapToCanvas(o.min_x, o.min_y);
    const [bx, by] = mapToCanvas(o.max_x, o.max_y);
    const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
    const minY = Math.min(ay, by), maxY = Math.max(ay, by);
    ctx.fillStyle = 'rgba(170, 178, 204, 0.48)';
    ctx.fillRect(minX, minY, maxX-minX, maxY-minY);
    ctx.strokeStyle = 'rgba(224, 230, 255, 0.82)';
    ctx.lineWidth = 1.6;
    ctx.strokeRect(minX, minY, maxX-minX, maxY-minY);

    // Diagonal hatching helps blocked regions stand out from path trails.
    ctx.save();
    ctx.beginPath();
    ctx.rect(minX, minY, maxX-minX, maxY-minY);
    ctx.clip();
    ctx.strokeStyle = 'rgba(230, 236, 255, 0.28)';
    ctx.lineWidth = 1.0;
    const step = 8;
    for (let x = minX - (maxY - minY); x < maxX + (maxY - minY); x += step) {{
      ctx.beginPath();
      ctx.moveTo(x, minY);
      ctx.lineTo(x + (maxY - minY), maxY);
      ctx.stroke();
    }}
    ctx.restore();
  }}

  for (const gate of chokepoints) {{
    const [ax, ay] = mapToCanvas(gate.min_x, gate.min_y);
    const [bx, by] = mapToCanvas(gate.max_x, gate.max_y);
    const minX = Math.min(ax, bx), maxX = Math.max(ax, bx);
    const minY = Math.min(ay, by), maxY = Math.max(ay, by);
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const width = Math.max(16, maxX - minX + 10);
    const height = Math.max(24, maxY - minY);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.95)';
    ctx.lineWidth = 2.4;
    ctx.strokeRect(cx - width * 0.5, cy - height * 0.5, width, height);
    ctx.beginPath();
    ctx.moveTo(cx - width * 0.5, cy);
    ctx.lineTo(cx + width * 0.5, cy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(251, 191, 36, 0.95)';
    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.fillText('CHOKEPOINT GATE', cx + 10, cy - 8);
  }}

  const unitsNow = framesByTick.get(currentTick) || [];
  const byId = new Map(unitsNow.map(u => [String(u.id), u]));
  const recentDepth = Number(depthEl.value || 12);
  const eventsNow = data.filter(d => d.tick === currentTick);
  const recentEvents = data.filter(d => d.tick <= currentTick && d.tick > currentTick - recentDepth);

  // Unit trails for recent motion context.
  for (const u of unitsNow) {{
    const trail = [];
    for (let t = Math.max(0, currentTick - 24); t <= currentTick; t++) {{
      const at = (framesByTick.get(t) || []).find(x => x.id === u.id);
      if (at) trail.push(at);
    }}
    if (trail.length >= 2) {{
      ctx.strokeStyle = u.team === 'Hero' ? '#3b82f6' : '#ef4444';
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i=0; i<trail.length; i++) {{
        const [tx, ty] = mapToCanvas(trail[i].x, trail[i].y);
        if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
      }}
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }}
  }}

  for (const e of recentEvents) {{
    if (e.src === '-' || e.dst === '-') continue;
    const a = byId.get(e.src), b = byId.get(e.dst);
    if (!a || !b) continue;
    const [ax, ay] = mapToCanvas(a.x, a.y);
    const [bx, by] = mapToCanvas(b.x, b.y);
    ctx.strokeStyle = eventColor(e.kind);
    const age = Math.max(0, currentTick - e.tick);
    ctx.globalAlpha = Math.max(0.18, 1.0 - age / recentDepth);
    ctx.lineWidth = e.tick === currentTick ? 2.6 : 1.4;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.globalAlpha = 1.0;
  }}

  for (const u of unitsNow) {{
    const [x, y] = mapToCanvas(u.x, u.y);
    ctx.fillStyle = u.team === 'Hero' ? '#60a5fa' : '#f87171';
    ctx.beginPath(); ctx.arc(x, y, u.hp > 0 ? 8 : 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(`${{u.id}} (hp:${{u.hp}})`, x + 10, y - 10);
  }}
}}

function render() {{
  const q = searchEl.value.trim().toLowerCase();
  const kind = kindSel.value;
  const unit = unitSel.value;
  const filtered = data.filter(d => {{
    if (kind !== 'ALL' && d.kind !== kind) return false;
    if (unit !== 'ALL' && d.src !== unit && d.dst !== unit) return false;
    if (q && !(`${{d.kind}} ${{d.src}} ${{d.dst}} ${{d.detail}}`.toLowerCase().includes(q))) return false;
    return true;
  }});

  rowsEl.innerHTML = filtered.map(d => `<tr><td>${{d.tick}}</td><td>${{d.kind}}</td><td>${{d.src}}</td><td>${{d.dst}}</td><td>${{d.value}}</td><td>${{d.detail}}</td></tr>`).join('');

  const maxTick = Math.max(...data.map(d => d.tick), 1);
  const bins = new Array(100).fill(0);
  for (const d of filtered) {{
    const idx = Math.min(99, Math.floor((d.tick / maxTick) * 100));
    bins[idx] += 1;
  }}
  const maxBin = Math.max(...bins, 1);
  timelineEl.innerHTML = bins.map(v => `<div class='bar' title='${{v}} events' style='height:${{Math.max(3, (v/maxBin)*42)}}px'></div>`).join('');

  const abilityBins = new Array(100).fill(0);
  const healBins = new Array(100).fill(0);
  const attackBins = new Array(100).fill(0);
  for (const d of filtered) {{
    const idx = Math.min(99, Math.floor((d.tick / maxTick) * 100));
    if (d.kind === 'AbilityCastStarted' || d.kind === 'ControlCastStarted') abilityBins[idx] += 1;
    if (d.kind === 'HealCastStarted') healBins[idx] += 1;
    if (d.kind === 'CastStarted') attackBins[idx] += 1;
  }}
  const maxAbilityBin = Math.max(
    ...abilityBins.map((v, i) => v + healBins[i] + attackBins[i]),
    1
  );
  abilityTimelineEl.innerHTML = abilityBins.map((_, i) => {{
    const a = abilityBins[i], h = healBins[i], atk = attackBins[i];
    const aH = Math.round((a / maxAbilityBin) * 50);
    const hH = Math.round((h / maxAbilityBin) * 50);
    const atkH = Math.round((atk / maxAbilityBin) * 50);
    return `<div class='ability-stack' title='abilities=${{a}} heals=${{h}} attacks=${{atk}}'>
      <div class='ability-seg' style='height:${{Math.max(0, atkH)}}px;background:#60a5fa;'></div>
      <div class='ability-seg' style='height:${{Math.max(0, hH)}}px;background:#4ade80;'></div>
      <div class='ability-seg' style='height:${{Math.max(0, aH)}}px;background:#fbbf24;'></div>
    </div>`;
  }}).join('');

  const perUnit = new Map();
  for (const d of data) {{
    if (!perUnit.has(d.src) && d.src !== '-') perUnit.set(d.src, {{ ability: 0, heal: 0, attack: 0 }});
    if (d.src === '-') continue;
    if (d.kind === 'AbilityCastStarted' || d.kind === 'ControlCastStarted') perUnit.get(d.src).ability += 1;
    if (d.kind === 'HealCastStarted') perUnit.get(d.src).heal += 1;
    if (d.kind === 'CastStarted') perUnit.get(d.src).attack += 1;
  }}
  const unitRows = Array.from(perUnit.entries())
    .sort((a,b) => Number(a[0]) - Number(b[0]))
    .map(([unit, v]) => `<tr><td>${{unit}}</td><td>${{v.ability}}</td><td>${{v.heal}}</td><td>${{v.attack}}</td></tr>`)
    .join('');
  abilityUnitRowsEl.innerHTML = unitRows;

  const tick = Number(tickEl.value);
  tickLabelEl.textContent = `tick ${{tick}}`;
  depthLabelEl.textContent = String(depthEl.value);
  const depth = Number(depthEl.value || 12);
  tickStatsEl.textContent = `${{data.filter(d=>d.tick===tick).length}} events @ tick | ${{data.filter(d=>d.tick<=tick && d.tick>tick-depth).length}} in last ${{depth}}`;
  drawMap(tick);
}}

searchEl.addEventListener('input', render);
kindSel.addEventListener('change', render);
unitSel.addEventListener('change', render);
tickEl.addEventListener('input', render);
depthEl.addEventListener('input', render);
let playTimer = null;
playBtn.addEventListener('click', () => {{
  if (playTimer) {{
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = 'play';
    return;
  }}
  playBtn.textContent = 'pause';
  playTimer = setInterval(() => {{
    let t = Number(tickEl.value) + 1;
    if (t > Number(tickEl.max)) t = 0;
    tickEl.value = String(t);
    render();
  }}, Number(speedSel.value || 120));
}});
speedSel.addEventListener('change', () => {{
  if (playTimer) {{
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = 'play';
  }}
}});
for (const btn of document.querySelectorAll('.tab-btn')) {{
  btn.addEventListener('click', () => {{
    for (const b of document.querySelectorAll('.tab-btn')) b.classList.remove('active');
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    for (const pane of document.querySelectorAll('.pane')) pane.classList.remove('active');
    document.getElementById(`pane-${{tab}}`).classList.add('active');
  }});
}}
render();
</script>
</body>
</html>"#,
        title = title,
        subtitle = subtitle,
        seed = seed,
        ticks = ticks,
        event_hash = replay.event_log_hash,
        state_hash = replay.final_state_hash,
        max_tick = max_tick,
        winner = format!("{:?}", replay.metrics.winner),
        first_death = replay
            .metrics
            .tick_to_first_death
            .map_or_else(|| "-".to_string(), |t| t.to_string()),
        casts_completed = replay.metrics.casts_completed,
        heals_completed = replay.metrics.heals_completed,
        repositions = replay.metrics.reposition_for_range_events,
        invariants = replay.metrics.invariant_violations,
        hero_alive = replay
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0)
            .count(),
        enemy_alive = replay
            .final_state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count(),
        rows = event_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$"),
        frame_rows = frame_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$"),
        obstacle_rows = obstacle_rows
            .replace('\\', r"\\")
            .replace('`', r"\`")
            .replace('$', r"\$")
    )
}

fn build_phase5_event_visualization_html(seed: u64, ticks: u32) -> String {
    let run =
        crate::ai::personality::run_phase5_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
    let replay = run.replay;
    let event_rows = build_event_rows(&replay);

    let initial = sample_phase5_party_state(seed);
    let roles = crate::ai::roles::default_roles();
    let personalities = default_personalities();
    let (script, _mode_history) = generate_scripted_intents(
        &initial,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
        roles,
        personalities,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);

    build_visualization_html(
        "AI Event Visualization",
        "Phase 5 personality timeline",
        &replay,
        &event_rows,
        &frame_rows,
        "",
        seed,
        ticks,
    )
}

pub fn export_phase5_event_visualization(path: &str, seed: u64, ticks: u32) -> io::Result<()> {
    let html = build_phase5_event_visualization_html(seed, ticks);
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

pub fn export_horde_chokepoint_visualization(path: &str, seed: u64, ticks: u32) -> io::Result<()> {
    let (initial, script) = crate::ai::advanced::build_horde_chokepoint_script(
        seed,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let replay = run_replay(
        initial.clone(),
        &script,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = crate::ai::advanced::horde_chokepoint_nav();
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);

    let html = build_visualization_html(
        "Pathing Horde Visualization",
        "Chokepoint wall/gate with A* waypointing",
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        seed,
        ticks,
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

pub fn export_horde_chokepoint_hero_favored_visualization(
    path: &str,
    seed: u64,
    ticks: u32,
) -> io::Result<()> {
    let (initial, script) = crate::ai::advanced::build_horde_chokepoint_hero_favored_script(
        seed,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let replay = run_replay(
        initial.clone(),
        &script,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = crate::ai::advanced::horde_chokepoint_nav();
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);

    let html = build_visualization_html(
        "Pathing Horde Visualization (Hero Favored)",
        "Chokepoint wall/gate with hero-favored roster and pressure tactics",
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        seed,
        ticks,
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioObstacle {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioUnit {
    pub id: u32,
    pub team: String,
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub elevation: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub move_speed: f32,
    pub attack_damage: i32,
    pub attack_range: f32,
    pub ability_damage: i32,
    pub ability_range: f32,
    pub heal_amount: i32,
    pub heal_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioElevationZone {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub elevation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSlopeZone {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub slope_cost_multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScenario {
    pub name: String,
    pub seed: u64,
    pub ticks: u32,
    pub world_min_x: f32,
    pub world_max_x: f32,
    pub world_min_y: f32,
    pub world_max_y: f32,
    pub cell_size: f32,
    #[serde(default)]
    pub elevation_zones: Vec<ScenarioElevationZone>,
    #[serde(default)]
    pub slope_zones: Vec<ScenarioSlopeZone>,
    pub obstacles: Vec<ScenarioObstacle>,
    pub units: Vec<ScenarioUnit>,
}

fn parse_team(label: &str) -> Team {
    if label.eq_ignore_ascii_case("hero") {
        Team::Hero
    } else {
        Team::Enemy
    }
}

fn custom_scenario_to_state(s: &CustomScenario) -> SimState {
    let mut units = s
        .units
        .iter()
        .map(|u| UnitState {
            id: u.id,
            team: parse_team(&u.team),
            hp: u.hp,
            max_hp: u.max_hp,
            position: sim_vec2(u.x, u.y),
            move_speed_per_sec: u.move_speed,
            attack_damage: u.attack_damage,
            attack_range: u.attack_range,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 250,
            cooldown_remaining_ms: 0,
            ability_damage: u.ability_damage,
            ability_range: u.ability_range,
            ability_cooldown_ms: 2_800,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount: u.heal_amount,
            heal_range: u.heal_range,
            heal_cooldown_ms: 2_100,
            heal_cast_time_ms: 380,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
        })
        .collect::<Vec<_>>();
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: s.seed,
        units,
    }
}

fn build_custom_nav(s: &CustomScenario) -> GridNav {
    let mut nav = GridNav::new(
        s.world_min_x,
        s.world_max_x,
        s.world_min_y,
        s.world_max_y,
        s.cell_size.max(0.2),
    );
    for o in &s.obstacles {
        nav.add_block_rect(o.min_x, o.max_x, o.min_y, o.max_y);
    }
    for zone in &s.elevation_zones {
        nav.set_elevation_rect(
            zone.min_x,
            zone.max_x,
            zone.min_y,
            zone.max_y,
            zone.elevation,
        );
    }
    for zone in &s.slope_zones {
        nav.set_slope_cost_rect(
            zone.min_x,
            zone.max_x,
            zone.min_y,
            zone.max_y,
            zone.slope_cost_multiplier,
        );
    }
    nav
}

fn custom_scenario_intents(state: &SimState, nav: &GridNav, dt_ms: u32) -> Vec<UnitIntent> {
    crate::ai::advanced::build_environment_reactive_intents(state, nav, dt_ms)
}

fn build_custom_scenario_script(
    s: &CustomScenario,
    dt_ms: u32,
) -> (SimState, Vec<Vec<UnitIntent>>) {
    let nav = build_custom_nav(s);
    let initial = custom_scenario_to_state(s);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(s.ticks as usize);
    for _ in 0..s.ticks {
        let intents = custom_scenario_intents(&state, &nav, dt_ms);
        script.push(intents.clone());
        let (next, _) = step(state, &intents, dt_ms);
        state = next;
    }
    (initial, script)
}

pub fn write_custom_scenario_template(path: &str) -> io::Result<()> {
    let scenario = CustomScenario {
        name: "chokepoint_demo".to_string(),
        seed: 777,
        ticks: 360,
        world_min_x: -20.0,
        world_max_x: 20.0,
        world_min_y: -10.0,
        world_max_y: 10.0,
        cell_size: 0.7,
        elevation_zones: vec![],
        slope_zones: vec![],
        obstacles: vec![ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: -9.0,
            max_y: 9.0,
        }],
        units: vec![
            ScenarioUnit {
                id: 1,
                team: "Hero".to_string(),
                x: -14.0,
                y: -0.6,
                elevation: 0.0,
                hp: 165,
                max_hp: 165,
                move_speed: 4.1,
                attack_damage: 16,
                attack_range: 1.4,
                ability_damage: 28,
                ability_range: 2.0,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 2,
                team: "Hero".to_string(),
                x: -15.2,
                y: 1.0,
                elevation: 0.0,
                hp: 96,
                max_hp: 96,
                move_speed: 4.3,
                attack_damage: 9,
                attack_range: 1.3,
                ability_damage: 0,
                ability_range: 0.0,
                heal_amount: 28,
                heal_range: 2.8,
            },
            ScenarioUnit {
                id: 10,
                team: "Enemy".to_string(),
                x: 12.0,
                y: -1.8,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 11,
                team: "Enemy".to_string(),
                x: 12.8,
                y: 0.0,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 12,
                team: "Enemy".to_string(),
                x: 12.0,
                y: 1.8,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
        ],
    };
    let body = serde_json::to_string_pretty(&scenario)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)?;
    Ok(())
}

pub fn export_custom_scenario_visualization(
    scenario_path: &str,
    output_path: &str,
) -> io::Result<()> {
    let text = fs::read_to_string(scenario_path)?;
    let scenario: CustomScenario = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let (initial, script) = build_custom_scenario_script(&scenario, crate::ai::core::FIXED_TICK_MS);
    let replay = run_replay(
        initial.clone(),
        &script,
        scenario.ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = build_custom_nav(&scenario);
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);
    let html = build_visualization_html(
        "Custom Scenario Visualization",
        &format!("{} | {}", scenario.name, scenario_path),
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        scenario.seed,
        scenario.ticks,
    );
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, html)?;
    Ok(())
}

pub fn build_custom_scenario_state_frames(s: &CustomScenario, dt_ms: u32) -> Vec<SimState> {
    let (initial, script) = build_custom_scenario_script(s, dt_ms);
    let mut frames = Vec::with_capacity(s.ticks as usize + 1);
    let mut state = initial;
    frames.push(state.clone());
    for intents in script.iter().take(s.ticks as usize) {
        let (next, _) = step(state, intents, dt_ms);
        state = next;
        frames.push(state.clone());
    }
    frames
}

pub fn export_visualization_index(path: &str, links: &[(String, String)]) -> io::Result<()> {
    let mut items = String::new();
    for (label, href) in links {
        items.push_str(&format!(
            "<li><a href=\"{}\">{}</a></li>",
            href.replace('"', ""),
            label
        ));
    }
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AI Visualization Index</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 20px; background:#0f1420; color:#e6ebff; }}
h2 {{ margin:0 0 8px 0; }}
ul {{ line-height: 1.9; }}
a {{ color:#8bc7ff; }}
code {{ background:#1a2436; padding:2px 5px; border-radius:4px; }}
</style></head>
<body>
<h2>AI Visualization Index</h2>
<p>Use <code>--phase6-viz</code>, <code>--pathing-viz</code>, or <code>--scenario-viz &lt;json&gt;</code> to generate pages.</p>
<ul>{}</ul>
</body></html>"#,
        items
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn scenario_matrix_is_deterministic_flagged() {
        let rows = run_scenario_matrix();
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|r| r.deterministic));
        assert!(rows
            .iter()
            .all(|r| r.hero_deaths <= 3 && r.enemy_deaths <= 3));
    }

    #[test]
    fn debug_output_contains_ranked_candidates() {
        let ticks = build_phase5_debug(31, 2, 3);
        assert_eq!(ticks.len(), 2);
        assert!(ticks
            .iter()
            .flat_map(|t| t.decisions.iter())
            .any(|d| !d.top_k.is_empty()));
    }

    #[test]
    fn tuning_grid_returns_sorted_results() {
        let rows = run_personality_grid_tuning();
        assert!(!rows.is_empty());
        for pair in rows.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn cc_metrics_are_within_expected_bounds() {
        let cc = analyze_phase4_cc_metrics(29, 320);
        assert!(cc.primary_target.is_some());
        assert!(cc.windows >= 3);
        assert!(cc.coverage_ratio >= 0.2 && cc.coverage_ratio <= 1.0);
        assert!(cc.overlap_ratio >= 0.0 && cc.overlap_ratio <= 1.0);
        assert!(cc.avg_gap_ticks >= 0.0);
    }

    #[test]
    fn scenario_matrix_hash_regression_snapshot() {
        let rows = run_scenario_matrix();
        let map = rows
            .into_iter()
            .map(|r| (r.name, (r.event_hash, r.state_hash)))
            .collect::<HashMap<_, _>>();

        assert_eq!(
            map.get("phase2_seed17"),
            Some(&(0x0122_c661_c895_0a47, 0xde59_67a3_db01_1c82))
        );
        assert_eq!(
            map.get("phase3_seed23"),
            Some(&(0xdb95_9f8e_2e84_480a, 0x673b_d6b7_96db_9a40))
        );
        assert_eq!(
            map.get("phase4_seed29"),
            Some(&(0xd903_0d7a_a128_07c1, 0xf4db_60f8_2f15_8573))
        );
        assert_eq!(
            map.get("phase5_seed31"),
            Some(&(0x1609_d64c_eeae_0632, 0xa90d_5423_a84b_0e9f))
        );
    }

    #[test]
    fn visualization_html_contains_expected_sections() {
        let html = build_phase5_event_visualization_html(31, 40);
        assert!(html.contains("AI Event Visualization"));
        assert!(html.contains("event_hash="));
        assert!(html.contains("timeline"));
        assert!(html.contains("DamageApplied"));
    }
}
