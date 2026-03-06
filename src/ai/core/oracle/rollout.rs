use super::{
    ScoredAction, OracleResult, DEFAULT_ROLLOUT_TICKS,
    KILL_BONUS_PER_DPS, CC_VALUE_FACTOR, EARLY_EXIT_THRESHOLD, EARLY_EXIT_CHECKPOINT,
    ATTACK_RANGE_FACTOR, FIXED_TICK_MS,
};
use crate::ai::core::{
    distance, is_alive, step, IntentAction, SimEvent, SimState, UnitIntent,
};
use crate::ai::effects::AbilityTarget;
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Threat
// ---------------------------------------------------------------------------

/// Compute the "threat per second" of a unit — a rough DPS estimate.
#[allow(dead_code)]
pub fn threat_per_sec(state: &SimState, unit_id: u32) -> f64 {
    let u = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return 0.0,
    };
    let mut dps = 0.0;
    if u.attack_cooldown_ms > 0 {
        dps += u.attack_damage as f64 / (u.attack_cooldown_ms as f64 / 1000.0);
    }
    if u.ability_cooldown_ms > 0 && u.ability_damage > 0 {
        dps += u.ability_damage as f64 / (u.ability_cooldown_ms as f64 / 1000.0);
    }
    if u.heal_cooldown_ms > 0 && u.heal_amount > 0 {
        dps += u.heal_amount as f64 / (u.heal_cooldown_ms as f64 / 1000.0) * 0.8;
    }
    dps
}

// ---------------------------------------------------------------------------
// Candidate enumeration
// ---------------------------------------------------------------------------

/// Enumerate candidate actions for a unit. Returns a pruned set with
/// cooldown and range gating to avoid wasting rollouts on impossible actions.
pub(crate) fn enumerate_candidates(
    state: &SimState,
    unit_id: u32,
    focus_target: Option<u32>,
) -> Vec<IntentAction> {
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return vec![],
    };
    let unit_team = unit.team;
    let unit_pos = unit.position;

    // Gather enemy ids sorted by distance (alive only)
    let mut enemies: Vec<(u32, f32)> = state
        .units
        .iter()
        .filter(|u| u.team != unit_team && is_alive(u))
        .map(|u| (u.id, distance(unit_pos, u.position)))
        .collect();
    enemies.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take 3 nearest + focus target (deduplicated)
    let mut target_ids: Vec<u32> = enemies.iter().take(3).map(|(id, _)| *id).collect();
    if let Some(ft) = focus_target {
        if !target_ids.contains(&ft) {
            if state.units.iter().any(|u| u.id == ft && is_alive(u)) {
                target_ids.push(ft);
            }
        }
    }

    // Gather allies for healing
    let allies: Vec<u32> = state
        .units
        .iter()
        .filter(|u| u.team == unit_team && is_alive(u) && u.id != unit_id)
        .filter(|u| u.hp < u.max_hp)
        .map(|u| u.id)
        .collect();

    let mut candidates = Vec::with_capacity(target_ids.len() * 4 + allies.len() + unit.abilities.len() * 4 + 6);

    let max_attack_dist = unit.attack_range * ATTACK_RANGE_FACTOR;

    for &tid in &target_ids {
        let dist = enemies.iter().find(|(id, _)| *id == tid).map(|(_, d)| *d).unwrap_or(f32::MAX);

        // [Range gate] Only add Attack if target is within reasonable distance
        if dist <= max_attack_dist {
            candidates.push(IntentAction::Attack { target_id: tid });
        }

        // CastAbility — skip only if cooldown > half the rollout window
        // (shorter CDs will come off during the rollout and still contribute)
        if unit.ability_damage > 0 {
            let ability_max_dist = unit.ability_range * ATTACK_RANGE_FACTOR;
            if dist <= ability_max_dist {
                candidates.push(IntentAction::CastAbility { target_id: tid });
            }
        }

        // CastControl — same logic
        if unit.control_duration_ms > 0 {
            let control_max_dist = unit.control_range * ATTACK_RANGE_FACTOR;
            if dist <= control_max_dist {
                candidates.push(IntentAction::CastControl { target_id: tid });
            }
        }
    }

    // Heal candidates
    if unit.heal_amount > 0 {
        for &aid in &allies {
            candidates.push(IntentAction::CastHeal { target_id: aid });
        }
    }

    // Hero abilities (already cooldown-gated)
    for (idx, slot) in unit.abilities.iter().enumerate() {
        if slot.cooldown_remaining_ms > 0 {
            continue;
        }
        // Resource gate
        if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
            continue;
        }
        use crate::ai::effects::AbilityTargeting;
        match slot.def.targeting {
            AbilityTargeting::TargetEnemy => {
                let ability_max_dist = slot.def.range * ATTACK_RANGE_FACTOR;
                for &tid in &target_ids {
                    let dist = enemies.iter().find(|(id, _)| *id == tid).map(|(_, d)| *d).unwrap_or(f32::MAX);
                    if dist <= ability_max_dist || ability_max_dist <= 0.0 {
                        candidates.push(IntentAction::UseAbility {
                            ability_index: idx,
                            target: AbilityTarget::Unit(tid),
                        });
                    }
                }
            }
            AbilityTargeting::TargetAlly => {
                for &aid in &allies {
                    candidates.push(IntentAction::UseAbility {
                        ability_index: idx,
                        target: AbilityTarget::Unit(aid),
                    });
                }
                candidates.push(IntentAction::UseAbility {
                    ability_index: idx,
                    target: AbilityTarget::Unit(unit_id),
                });
            }
            AbilityTargeting::SelfCast | AbilityTargeting::SelfAoe | AbilityTargeting::Global => {
                candidates.push(IntentAction::UseAbility {
                    ability_index: idx,
                    target: AbilityTarget::None,
                });
            }
            AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
                if let Some(&(tid, _)) = enemies.first() {
                    if let Some(t) = state.units.iter().find(|u| u.id == tid) {
                        candidates.push(IntentAction::UseAbility {
                            ability_index: idx,
                            target: AbilityTarget::Position(t.position),
                        });
                    }
                }
            }
        }
    }

    // Movement candidates
    if let Some(&(nearest_eid, nearest_dist)) = enemies.first() {
        if let Some(nearest_e) = state.units.iter().find(|u| u.id == nearest_eid) {
            let step_dist = unit.move_speed_per_sec * (FIXED_TICK_MS as f32 / 1000.0);

            // Move toward nearest enemy (close to attack range)
            if nearest_dist > unit.attack_range {
                let desired = crate::ai::core::position_at_range(unit_pos, nearest_e.position, unit.attack_range * 0.9);
                let next = crate::ai::core::move_towards(unit_pos, desired, step_dist);
                candidates.push(IntentAction::MoveTo { position: next });
            }

            // Move away from nearest enemy (kite/retreat)
            if nearest_dist < unit.attack_range * 2.0 {
                let away = crate::ai::core::move_away(unit_pos, nearest_e.position, step_dist);
                candidates.push(IntentAction::MoveTo { position: away });
            }
        }
    }

    // Move toward weakest ally (healer repositioning)
    if unit.heal_amount > 0 || unit.abilities.iter().any(|s| s.def.ai_hint == "heal") {
        let weakest_ally = state.units.iter()
            .filter(|u| u.team == unit_team && is_alive(u) && u.id != unit_id && u.hp < u.max_hp)
            .min_by(|a, b| {
                let ha = a.hp as f32 / a.max_hp.max(1) as f32;
                let hb = b.hp as f32 / b.max_hp.max(1) as f32;
                ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(ally) = weakest_ally {
            let step_dist = unit.move_speed_per_sec * (FIXED_TICK_MS as f32 / 1000.0);
            let next = crate::ai::core::move_towards(unit_pos, ally.position, step_dist);
            candidates.push(IntentAction::MoveTo { position: next });
        }
    }

    // Hold as baseline
    candidates.push(IntentAction::Hold);

    candidates
}

// ---------------------------------------------------------------------------
// Rollout execution
// ---------------------------------------------------------------------------

/// Run a single rollout: clone state, override the unit's intent on tick 0,
/// then let default AI drive for `rollout_ticks` ticks.
/// Supports early exit: if `best_score_so_far` is provided and the running
/// score falls behind by `EARLY_EXIT_THRESHOLD` after `EARLY_EXIT_CHECKPOINT`
/// ticks, aborts early.
pub(crate) fn run_rollout(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    action: &IntentAction,
    rollout_ticks: u64,
    best_score_so_far: Option<f64>,
) -> (i32, i32, u32, u32, i32) {
    let unit_team = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u.team,
        None => return (0, 0, 0, 0, 0),
    };

    // Snapshot HP before rollout
    let pre_enemy_hp: i32 = state
        .units
        .iter()
        .filter(|u| u.team != unit_team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();
    let pre_ally_hp: i32 = state
        .units
        .iter()
        .filter(|u| u.team == unit_team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();
    // Snapshot enemy healing done before rollout
    let pre_enemy_healing: i32 = state
        .units
        .iter()
        .filter(|u| u.team != unit_team)
        .map(|u| u.total_healing_done)
        .sum();

    let mut sim = state.clone();
    let mut ai = squad_ai.clone();
    let mut total_kills = 0u32;
    let mut total_cc_ms = 0u32;

    for tick_i in 0..rollout_ticks {
        let mut intents = generate_intents(&sim, &mut ai, FIXED_TICK_MS);

        // On the first tick, override our unit's action
        if tick_i == 0 {
            intents.retain(|i| i.unit_id != unit_id);
            intents.push(UnitIntent {
                unit_id,
                action: *action,
            });
        }

        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        for ev in &events {
            match ev {
                SimEvent::UnitDied { unit_id: dead_id, .. } => {
                    if let Some(u) = state.units.iter().find(|u| u.id == *dead_id) {
                        if u.team != unit_team {
                            total_kills += 1;
                        }
                    }
                }
                SimEvent::ControlApplied { duration_ms, target_id, .. } => {
                    if let Some(u) = state.units.iter().find(|u| u.id == *target_id) {
                        if u.team != unit_team {
                            total_cc_ms += duration_ms;
                        }
                    }
                }
                _ => {}
            }
        }

        // Early exit check at checkpoint
        if tick_i == EARLY_EXIT_CHECKPOINT {
            if let Some(best) = best_score_so_far {
                let mid_enemy_hp: i32 = sim.units.iter()
                    .filter(|u| u.team != unit_team && is_alive(u))
                    .map(|u| u.hp + u.shield_hp).sum();
                let mid_ally_hp: i32 = sim.units.iter()
                    .filter(|u| u.team == unit_team && is_alive(u))
                    .map(|u| u.hp + u.shield_hp).sum();
                let mid_score = ((pre_enemy_hp - mid_enemy_hp) - (pre_ally_hp - mid_ally_hp)) as f64
                    + total_kills as f64 * KILL_BONUS_PER_DPS
                    + total_cc_ms as f64 * CC_VALUE_FACTOR;
                if mid_score < best - EARLY_EXIT_THRESHOLD {
                    // This candidate is hopelessly behind — abort
                    let enemy_hp_lost = pre_enemy_hp - mid_enemy_hp;
                    let ally_hp_lost = pre_ally_hp - mid_ally_hp;
                    let enemy_heal_during: i32 = sim.units.iter()
                        .filter(|u| u.team != unit_team)
                        .map(|u| u.total_healing_done).sum::<i32>() - pre_enemy_healing;
                    return (enemy_hp_lost, ally_hp_lost, total_kills, total_cc_ms, enemy_heal_during);
                }
            }
        }
    }

    let post_enemy_hp: i32 = sim
        .units
        .iter()
        .filter(|u| u.team != unit_team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();
    let post_ally_hp: i32 = sim
        .units
        .iter()
        .filter(|u| u.team == unit_team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();

    let enemy_hp_lost = pre_enemy_hp - post_enemy_hp;
    let ally_hp_lost = pre_ally_hp - post_ally_hp;
    let enemy_heal_during: i32 = sim
        .units
        .iter()
        .filter(|u| u.team != unit_team)
        .map(|u| u.total_healing_done)
        .sum::<i32>() - pre_enemy_healing;

    (enemy_hp_lost, ally_hp_lost, total_kills, total_cc_ms, enemy_heal_during)
}

pub(crate) fn score_rollout(enemy_hp_lost: i32, ally_hp_lost: i32, kills: u32, cc_ms: u32, enemy_heal_suppressed: i32) -> f64 {
    let kill_bonus = kills as f64 * KILL_BONUS_PER_DPS;
    let cc_value = cc_ms as f64 * CC_VALUE_FACTOR;
    // Heal suppression: reward actions that reduce enemy healing output.
    // Negative enemy_heal_suppressed means enemies healed less → good.
    // Scale: -1 HP healed = +0.5 score (half weight of direct damage).
    let heal_sup_bonus = -(enemy_heal_suppressed as f64) * 0.5;
    (enemy_hp_lost - ally_hp_lost) as f64 + kill_bonus + cc_value + heal_sup_bonus
}

// ---------------------------------------------------------------------------
// Score actions
// ---------------------------------------------------------------------------

/// Score all candidate actions for a unit via rollout.
pub fn score_actions(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    focus_target: Option<u32>,
) -> OracleResult {
    score_actions_with_depth(state, squad_ai, unit_id, focus_target, DEFAULT_ROLLOUT_TICKS)
}

/// Score all candidate actions with configurable rollout depth.
/// Uses rayon for parallel rollouts and early exit for pruning.
pub fn score_actions_with_depth(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    focus_target: Option<u32>,
    rollout_ticks: u64,
) -> OracleResult {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    let candidates = enumerate_candidates(state, unit_id, focus_target);

    // Run Hold baseline first (sequential) to seed early-exit threshold
    let (hold_ehl, hold_ahl, hold_kills, hold_cc, hold_heal) =
        run_rollout(state, squad_ai, unit_id, &IntentAction::Hold, rollout_ticks, None);
    let hold_score = score_rollout(hold_ehl, hold_ahl, hold_kills, hold_cc, hold_heal);

    // Shared atomic best score for early exit across parallel rollouts.
    // We use i64 with fixed-point (score * 1000) since AtomicF64 doesn't exist.
    let best_score_fp = AtomicI64::new((hold_score * 1000.0) as i64);

    // Score remaining candidates in parallel
    let non_hold: Vec<_> = candidates.into_iter().filter(|a| !matches!(a, IntentAction::Hold)).collect();

    let mut scored: Vec<ScoredAction> = non_hold
        .into_par_iter()
        .map(|action| {
            let current_best = best_score_fp.load(Ordering::Relaxed) as f64 / 1000.0;
            let (enemy_hp_lost, ally_hp_lost, kills, cc_ms, heal_sup) =
                run_rollout(state, squad_ai, unit_id, &action, rollout_ticks, Some(current_best));

            let score = score_rollout(enemy_hp_lost, ally_hp_lost, kills, cc_ms, heal_sup);

            // Update shared best score
            let score_fp = (score * 1000.0) as i64;
            best_score_fp.fetch_max(score_fp, Ordering::Relaxed);

            ScoredAction {
                action,
                score,
                enemy_hp_lost,
                ally_hp_lost,
                kills,
                cc_ticks_applied: cc_ms / FIXED_TICK_MS,
            }
        })
        .collect();

    // Add Hold result
    scored.push(ScoredAction {
        action: IntentAction::Hold,
        score: hold_score,
        enemy_hp_lost: hold_ehl,
        ally_hp_lost: hold_ahl,
        kills: hold_kills,
        cc_ticks_applied: hold_cc / FIXED_TICK_MS,
    });

    // Sort descending by score
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    OracleResult {
        unit_id,
        tick: state.tick,
        scored_actions: scored,
    }
}
