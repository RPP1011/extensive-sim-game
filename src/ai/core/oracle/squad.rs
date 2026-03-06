use super::rollout::{enumerate_candidates, score_rollout, threat_per_sec};
use crate::ai::core::{
    is_alive, step, IntentAction, SimEvent, SimState, UnitIntent, FIXED_TICK_MS,
};
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Squad-level sequential greedy oracle
// ---------------------------------------------------------------------------

/// Result of squad-level oracle: committed action per hero.
#[derive(Debug, Clone)]
pub struct SquadPlan {
    pub actions: Vec<(u32, IntentAction)>, // (unit_id, action)
    pub score: f64,
}

/// Run a multi-unit rollout with specific actions locked for listed heroes.
/// All other units use default AI.
pub(crate) fn run_squad_rollout(
    state: &SimState,
    squad_ai: &SquadAiState,
    locked: &[(u32, IntentAction)],
    team: crate::ai::core::Team,
    rollout_ticks: u64,
) -> (i32, i32, u32, i32) {
    let pre_enemy_hp: i32 = state.units.iter()
        .filter(|u| u.team != team && is_alive(u))
        .map(|u| u.hp + u.shield_hp).sum();
    let pre_ally_hp: i32 = state.units.iter()
        .filter(|u| u.team == team && is_alive(u))
        .map(|u| u.hp + u.shield_hp).sum();
    let pre_enemy_healing: i32 = state.units.iter()
        .filter(|u| u.team != team)
        .map(|u| u.total_healing_done).sum();

    let mut sim = state.clone();
    let mut ai = squad_ai.clone();
    let mut kills = 0u32;

    for tick_i in 0..rollout_ticks {
        let mut intents = generate_intents(&sim, &mut ai, FIXED_TICK_MS);

        // On tick 0, override all locked heroes
        if tick_i == 0 {
            for &(uid, ref action) in locked {
                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent { unit_id: uid, action: *action });
            }
        }

        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        for ev in &events {
            if let SimEvent::UnitDied { unit_id: dead_id, .. } = ev {
                if let Some(u) = state.units.iter().find(|u| u.id == *dead_id) {
                    if u.team != team { kills += 1; }
                }
            }
        }
    }

    let post_enemy_hp: i32 = sim.units.iter()
        .filter(|u| u.team != team && is_alive(u))
        .map(|u| u.hp + u.shield_hp).sum();
    let post_ally_hp: i32 = sim.units.iter()
        .filter(|u| u.team == team && is_alive(u))
        .map(|u| u.hp + u.shield_hp).sum();
    let enemy_heal: i32 = sim.units.iter()
        .filter(|u| u.team != team)
        .map(|u| u.total_healing_done).sum::<i32>() - pre_enemy_healing;

    (pre_enemy_hp - post_enemy_hp, pre_ally_hp - post_ally_hp, kills, enemy_heal)
}

/// Sequential greedy squad oracle.
///
/// For each hero (sorted by threat, highest first), enumerate candidate actions,
/// rollout each with previously-committed heroes locked, pick the best.
/// Cost: ~(candidates_per_hero × num_heroes) rollouts ≈ 10 × 4 = 40.
pub fn squad_oracle(
    state: &SimState,
    squad_ai: &SquadAiState,
    team: crate::ai::core::Team,
    rollout_ticks: u64,
) -> SquadPlan {
    let focus_target = squad_ai.blackboard_for_team(team).and_then(|b| b.focus_target);

    // Gather hero unit IDs, sorted by threat (highest first for greedy ordering)
    let mut hero_ids: Vec<(u32, f64)> = state.units.iter()
        .filter(|u| u.team == team && is_alive(u))
        .filter(|u| u.casting.is_none() && u.control_remaining_ms == 0)
        .map(|u| (u.id, threat_per_sec(state, u.id)))
        .collect();
    hero_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut committed: Vec<(u32, IntentAction)> = Vec::new();

    for (uid, _threat) in &hero_ids {
        let candidates = enumerate_candidates(state, *uid, focus_target);
        if candidates.is_empty() {
            committed.push((*uid, IntentAction::Hold));
            continue;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_action = IntentAction::Hold;

        for candidate in &candidates {
            let mut trial = committed.clone();
            trial.push((*uid, *candidate));

            let (ehl, ahl, kills, heal_sup) =
                run_squad_rollout(state, squad_ai, &trial, team, rollout_ticks);
            let score = score_rollout(ehl, ahl, kills, 0, heal_sup);

            if score > best_score {
                best_score = score;
                best_action = *candidate;
            }
        }

        committed.push((*uid, best_action));
    }

    // Final score with all actions committed
    let (ehl, ahl, kills, heal_sup) =
        run_squad_rollout(state, squad_ai, &committed, team, rollout_ticks);
    let final_score = score_rollout(ehl, ahl, kills, 0, heal_sup);

    SquadPlan {
        actions: committed,
        score: final_score,
    }
}
