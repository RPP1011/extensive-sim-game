use super::rollout::score_rollout;
use crate::ai::core::{
    is_alive, step, SimEvent, SimState, FIXED_TICK_MS,
};
use crate::ai::squad::{generate_intents, SquadAiState};

// ---------------------------------------------------------------------------
// Joint focus-target search
// ---------------------------------------------------------------------------

/// Result of evaluating a candidate focus target.
#[derive(Debug, Clone)]
pub struct FocusCandidate {
    pub target_id: u32,
    pub score: f64,
    pub enemy_hp_lost: i32,
    pub ally_hp_lost: i32,
    pub kills: u32,
    pub enemy_heal_during: i32,
}

/// Evaluate each living enemy as a potential team focus target.
///
/// For each candidate, overrides the hero team's focus_target on the squad
/// blackboard, then runs a full-team rollout (all heroes use default AI with
/// that focus) for `rollout_ticks` ticks. Returns candidates sorted by score
/// (best first).
pub fn search_focus_target(
    state: &SimState,
    squad_ai: &SquadAiState,
    team: crate::ai::core::Team,
    rollout_ticks: u64,
) -> Vec<FocusCandidate> {
    let enemy_ids: Vec<u32> = state
        .units
        .iter()
        .filter(|u| u.team != team && is_alive(u))
        .map(|u| u.id)
        .collect();

    if enemy_ids.is_empty() {
        return Vec::new();
    }

    // Snapshot HP before rollout
    let pre_enemy_hp: i32 = state
        .units
        .iter()
        .filter(|u| u.team != team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();
    let pre_ally_hp: i32 = state
        .units
        .iter()
        .filter(|u| u.team == team && is_alive(u))
        .map(|u| u.hp + u.shield_hp)
        .sum();
    let pre_enemy_healing: i32 = state
        .units
        .iter()
        .filter(|u| u.team != team)
        .map(|u| u.total_healing_done)
        .sum();

    let mut candidates: Vec<FocusCandidate> = enemy_ids
        .iter()
        .map(|&eid| {
            let mut sim = state.clone();
            let mut ai = squad_ai.clone();

            // Override focus target for the hero team
            ai.set_focus_target(team, Some(eid));

            let mut total_kills = 0u32;

            for _ in 0..rollout_ticks {
                let intents = generate_intents(&sim, &mut ai, FIXED_TICK_MS);
                let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
                sim = new_sim;

                for ev in &events {
                    if let SimEvent::UnitDied { unit_id: dead_id, .. } = ev {
                        if let Some(u) = state.units.iter().find(|u| u.id == *dead_id) {
                            if u.team != team {
                                total_kills += 1;
                            }
                        }
                    }
                }
            }

            let post_enemy_hp: i32 = sim
                .units
                .iter()
                .filter(|u| u.team != team && is_alive(u))
                .map(|u| u.hp + u.shield_hp)
                .sum();
            let post_ally_hp: i32 = sim
                .units
                .iter()
                .filter(|u| u.team == team && is_alive(u))
                .map(|u| u.hp + u.shield_hp)
                .sum();
            let enemy_heal_during: i32 = sim
                .units
                .iter()
                .filter(|u| u.team != team)
                .map(|u| u.total_healing_done)
                .sum::<i32>() - pre_enemy_healing;

            let enemy_hp_lost = pre_enemy_hp - post_enemy_hp;
            let ally_hp_lost = pre_ally_hp - post_ally_hp;
            let score = score_rollout(enemy_hp_lost, ally_hp_lost, total_kills, 0, enemy_heal_during);

            FocusCandidate {
                target_id: eid,
                score,
                enemy_hp_lost,
                ally_hp_lost,
                kills: total_kills,
                enemy_heal_during,
            }
        })
        .collect();

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}
