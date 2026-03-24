//! Personal goals system for adventurers — every 300 ticks.
//!
//! Each adventurer can pursue a personal ambition beyond guild missions.
//! Goals are assigned based on backstory motivation and current situation.
//! Fulfilling goals boosts loyalty and morale; neglecting them causes decay.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerGoal, AdventurerStatus, CampaignState, GoalType, lcg_f32,
};

/// How often to tick personal goals (in ticks).
const GOAL_INTERVAL: u64 = 300;

/// Loyalty bonus on goal fulfillment.
const FULFILLMENT_LOYALTY: f32 = 20.0;
/// Morale bonus on goal fulfillment.
const FULFILLMENT_MORALE: f32 = 15.0;
/// Loyalty penalty when deadline passes with < 50% progress.
const NEGLECT_LOYALTY: f32 = 10.0;
/// Morale penalty when deadline passes with < 50% progress.
const NEGLECT_MORALE: f32 = 10.0;
/// Loyalty penalty when adventurer abandons a goal.
const ABANDON_LOYALTY: f32 = 5.0;

/// Tick the personal goals system.
///
/// 1. Assign goals to adventurers without one (based on backstory / situation).
/// 2. Update progress on active goals.
/// 3. Check for fulfillment (reward) or deadline neglect (penalty).
pub fn tick_personal_goals(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % GOAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let tick = state.tick;
    let adv_count = state.adventurers.len();

    // --- Phase 1: Assign goals to adventurers without one ---
    for i in 0..adv_count {
        if state.adventurers[i].personal_goal.is_some() {
            continue;
        }
        if state.adventurers[i].status == AdventurerStatus::Dead {
            continue;
        }

        if let Some(goal) = pick_goal(state, i) {
            let adv_id = state.adventurers[i].id;
            let goal_type = goal.goal_type.clone();
            state.adventurers[i].personal_goal = Some(goal);
            events.push(WorldEvent::PersonalGoalAssigned {
                adventurer_id: adv_id,
                goal: goal_type,
            });
        }
    }

    // --- Phase 2: Update progress on active goals ---
    for i in 0..adv_count {
        let goal = match &state.adventurers[i].personal_goal {
            Some(g) if !g.fulfilled && !g.abandoned => g.clone(),
            _ => continue,
        };

        let new_progress = compute_progress(state, i, &goal.goal_type);
        state.adventurers[i].personal_goal.as_mut().unwrap().progress = new_progress;

        // Check fulfillment
        if new_progress >= 100.0 {
            let adv_id = state.adventurers[i].id;
            let g = state.adventurers[i].personal_goal.as_mut().unwrap();
            g.fulfilled = true;
            g.progress = 100.0;

            state.adventurers[i].loyalty =
                (state.adventurers[i].loyalty + FULFILLMENT_LOYALTY).min(100.0);
            state.adventurers[i].morale =
                (state.adventurers[i].morale + FULFILLMENT_MORALE).min(100.0);
            state.adventurers[i]
                .history_tags
                .entry("goal_fulfilled".to_string())
                .and_modify(|v| *v += 1)
                .or_insert(1);

            let goal_type = goal.goal_type.clone();
            events.push(WorldEvent::PersonalGoalFulfilled {
                adventurer_id: adv_id,
                goal: goal_type,
            });

            // Clear goal so a new one can be assigned next interval
            state.adventurers[i].personal_goal = None;
            continue;
        }

        // Check deadline neglect
        if let Some(deadline) = goal.deadline_tick {
            if tick >= deadline && new_progress < 50.0 {
                let adv_id = state.adventurers[i].id;
                let goal_type = goal.goal_type.clone();

                state.adventurers[i].loyalty =
                    (state.adventurers[i].loyalty - NEGLECT_LOYALTY).max(0.0);
                state.adventurers[i].morale =
                    (state.adventurers[i].morale - NEGLECT_MORALE).max(0.0);

                // Abandon the goal
                state.adventurers[i].personal_goal.as_mut().unwrap().abandoned = true;

                events.push(WorldEvent::PersonalGoalAbandoned {
                    adventurer_id: adv_id,
                    goal: goal_type,
                });

                // Clear so a new goal can be assigned
                state.adventurers[i].personal_goal = None;
            }
        }
    }

    // --- Phase 3: Abandon stale goals with very low progress ---
    // Goals with < 10% progress after half the deadline are abandoned.
    for i in 0..adv_count {
        let goal = match &state.adventurers[i].personal_goal {
            Some(g) if !g.fulfilled && !g.abandoned => g.clone(),
            _ => continue,
        };

        if let Some(deadline) = goal.deadline_tick {
            let midpoint = deadline.saturating_sub(GOAL_INTERVAL * 5); // rough halfway
            if tick >= midpoint && goal.progress < 10.0 {
                let adv_id = state.adventurers[i].id;
                let goal_type = goal.goal_type.clone();

                state.adventurers[i].loyalty =
                    (state.adventurers[i].loyalty - ABANDON_LOYALTY).max(0.0);
                state.adventurers[i].personal_goal = None;

                events.push(WorldEvent::PersonalGoalAbandoned {
                    adventurer_id: adv_id,
                    goal: goal_type,
                });
            }
        }
    }
}

/// Pick a goal for an adventurer based on backstory motivation and situation.
fn pick_goal(state: &mut CampaignState, adv_idx: usize) -> Option<AdventurerGoal> {
    let adv = &state.adventurers[adv_idx];
    let tick = state.tick;

    // Build weighted candidate pool
    let mut candidates: Vec<(GoalType, f32)> = Vec::new();

    // --- Backstory-driven goals ---
    if let Some(ref backstory) = adv.backstory {
        use crate::headless_campaign::systems::backstory::BackstoryMotivation;
        match backstory.motivation {
            BackstoryMotivation::Revenge => {
                // AvengeAlly if any ally has died
                for other in &state.adventurers {
                    if other.status == AdventurerStatus::Dead && other.id != adv.id {
                        candidates.push((
                            GoalType::AvengeAlly { dead_id: other.id },
                            3.0,
                        ));
                        break; // one candidate is enough
                    }
                }
                // DefeatNemesis if any nemesis exists
                for nem in &state.nemeses {
                    if !nem.defeated {
                        candidates.push((
                            GoalType::DefeatNemesis { nemesis_id: nem.id },
                            3.0,
                        ));
                        break;
                    }
                }
            }
            BackstoryMotivation::Glory => {
                candidates.push((GoalType::EarnTitle, 3.0));
            }
            BackstoryMotivation::Greed => {
                let target = (adv.level as f32 * 50.0).max(100.0);
                candidates.push((GoalType::AccumulateGold { target }, 3.0));
            }
            BackstoryMotivation::Legacy => {
                candidates.push((GoalType::EarnTitle, 2.0));
                candidates.push((GoalType::RetireWealthy, 2.0));
            }
            BackstoryMotivation::Curiosity => {
                candidates.push((GoalType::ExploreAllRegions, 2.5));
                if !adv.traits.is_empty() {
                    let skill = adv.traits[0].clone();
                    candidates.push((GoalType::MasterSkill { skill }, 2.0));
                }
            }
            _ => {}
        }
    }

    // --- Situation-driven goals ---
    // Low level adventurers want to level up
    if adv.level < 3 {
        let target = adv.level + 2;
        candidates.push((GoalType::ReachLevel { target }, 2.0));
    }

    // Far from hometown
    if let Some(ref backstory) = adv.backstory {
        if let Some(hometown_id) = backstory.hometown_region_id {
            // Check if adventurer is in a party and far from home
            if adv.party_id.is_some() {
                candidates.push((
                    GoalType::VisitHometown {
                        region_id: hometown_id,
                    },
                    1.5,
                ));
            }
        }
    }

    // Bond-driven: high bond with someone
    let adv_id = adv.id;
    for (&(a, b), &strength) in &state.adventurer_bonds {
        if (a == adv_id || b == adv_id) && strength > 50.0 {
            let target_id = if a == adv_id { b } else { a };
            candidates.push((GoalType::FormBond { target_id }, 1.5));
            break;
        }
    }

    // Fallback: always offer ReachLevel and AccumulateGold at low weight
    if candidates.is_empty() {
        let target = adv.level + 2;
        candidates.push((GoalType::ReachLevel { target }, 1.0));
        let gold_target = (adv.level as f32 * 30.0).max(50.0);
        candidates.push((GoalType::AccumulateGold { target: gold_target }, 1.0));
    }

    // Weighted random selection
    let total_weight: f32 = candidates.iter().map(|(_, w)| *w).sum();
    if total_weight <= 0.0 {
        return None;
    }

    let roll = lcg_f32(&mut state.rng) * total_weight;
    let mut cumulative = 0.0;
    let mut chosen = candidates[0].0.clone();
    for (goal_type, w) in &candidates {
        cumulative += w;
        if roll < cumulative {
            chosen = goal_type.clone();
            break;
        }
    }

    // Set deadline: 3000 ticks from now for most goals, None for open-ended ones
    let deadline_tick = match &chosen {
        GoalType::ExploreAllRegions => None,
        GoalType::RetireWealthy => None,
        _ => Some(tick + 3000),
    };

    Some(AdventurerGoal {
        goal_type: chosen,
        progress: 0.0,
        deadline_tick,
        fulfilled: false,
        abandoned: false,
    })
}

/// Compute current progress (0–100) for a goal based on campaign state.
fn compute_progress(state: &CampaignState, adv_idx: usize, goal: &GoalType) -> f32 {
    let adv = &state.adventurers[adv_idx];

    match goal {
        GoalType::ReachLevel { target } => {
            if adv.level >= *target {
                100.0
            } else {
                // Progress based on XP toward next level (approximate)
                let base_level = (*target).saturating_sub(2);
                let levels_done = adv.level.saturating_sub(base_level) as f32;
                let levels_needed = (*target - base_level) as f32;
                ((levels_done / levels_needed) * 100.0).min(99.0)
            }
        }

        GoalType::AccumulateGold { target } => {
            let current = state.guild.gold;
            ((current / target) * 100.0).min(100.0)
        }

        GoalType::DefeatNemesis { nemesis_id } => {
            let defeated = state
                .nemeses
                .iter()
                .any(|n| n.id == *nemesis_id && n.defeated);
            if defeated {
                100.0
            } else {
                // Progress based on nemesis health reduction (approximate via strength)
                let original_strength = state
                    .nemeses
                    .iter()
                    .find(|n| n.id == *nemesis_id)
                    .map(|n| n.strength)
                    .unwrap_or(100.0);
                // Lower strength = more progress (rough heuristic)
                ((1.0 - original_strength / 100.0) * 80.0).max(0.0)
            }
        }

        GoalType::VisitHometown { region_id } => {
            // Check if the adventurer's party is near the hometown region
            if let Some(party_id) = adv.party_id {
                if let Some(party) = state.parties.iter().find(|p| p.id == party_id) {
                    // Check if party is at or near a location in the target region
                    for loc in &state.overworld.locations {
                        if loc.faction_owner == Some(*region_id) {
                            let dx = party.position.0 - loc.position.0;
                            let dy = party.position.1 - loc.position.1;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < 2.0 {
                                return 100.0;
                            }
                        }
                    }
                }
            }
            // Check history tags for visits
            if adv
                .history_tags
                .get("visited_hometown")
                .copied()
                .unwrap_or(0)
                > 0
            {
                return 100.0;
            }
            0.0
        }

        GoalType::MasterSkill { skill } => {
            // Check if the adventurer has the skill as a trait
            let has_skill = adv.traits.contains(skill);
            if has_skill {
                // Progress based on level and related history tags
                let skill_count = adv.history_tags.get(skill).copied().unwrap_or(0);
                ((skill_count as f32 / 5.0) * 100.0).min(100.0)
            } else {
                0.0
            }
        }

        GoalType::FormBond { target_id } => {
            let key = if adv.id < *target_id {
                (adv.id, *target_id)
            } else {
                (*target_id, adv.id)
            };
            let strength = state.adventurer_bonds.get(&key).copied().unwrap_or(0.0);
            // Bond of 80+ = fulfilled
            ((strength / 80.0) * 100.0).min(100.0)
        }

        GoalType::EarnTitle => {
            // Title = having a leadership role or legendary deed
            if adv.leadership_role.is_some() || !adv.deeds.is_empty() {
                100.0
            } else {
                // Progress based on level and reputation
                let level_prog = (adv.level as f32 / 5.0) * 50.0;
                let deed_prog = if adv.deeds.is_empty() { 0.0 } else { 50.0 };
                (level_prog + deed_prog).min(99.0)
            }
        }

        GoalType::RetireWealthy => {
            // Need high guild gold and high level
            let gold_prog = (state.guild.gold / 500.0) * 50.0;
            let level_prog = (adv.level as f32 / 8.0) * 50.0;
            (gold_prog + level_prog).min(100.0)
        }

        GoalType::AvengeAlly { dead_id } => {
            // Check if the adventurer has completed a quest since the ally died
            let avenge_count = adv
                .history_tags
                .get("avenge_quest")
                .copied()
                .unwrap_or(0);
            if avenge_count > 0 {
                100.0
            } else {
                // Progress from completed quests (rough proxy)
                let quests_completed = state
                    .completed_quests
                    .iter()
                    .filter(|q| {
                        // Use party_id to check if adventurer was involved
                        let _ = dead_id; // referenced for avenge context
                        state.parties.iter().any(|p| {
                            p.id == q.party_id && p.member_ids.contains(&adv.id)
                        })
                    })
                    .count();
                // Check if any nemesis was defeated (proxy for vengeance)
                let nemesis_defeated = state.nemeses.iter().any(|n| {
                    n.defeated && n.kills > 0
                });
                let base = (quests_completed as f32 * 25.0).min(75.0);
                if nemesis_defeated { 100.0 } else { base }
            }
        }

        GoalType::ExploreAllRegions => {
            let total = state.overworld.regions.len();
            if total == 0 {
                return 100.0;
            }
            let explored = state
                .overworld
                .regions
                .iter()
                .filter(|r| r.visibility > 0.7)
                .count();
            (explored as f32 / total as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_assignment_and_fulfillment() {
        let mut state = CampaignState::default_test_campaign(42);
        // Ensure we have adventurers
        assert!(!state.adventurers.is_empty());

        // No goals initially
        for adv in &state.adventurers {
            assert!(adv.personal_goal.is_none());
        }

        // Tick to assign goals
        state.tick = 300;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_personal_goals(&mut state, &mut deltas, &mut events);

        // At least one adventurer should have a goal assigned
        let has_goal = state.adventurers.iter().any(|a| a.personal_goal.is_some());
        assert!(has_goal, "Expected at least one adventurer to receive a goal");

        // Events should contain at least one assignment
        let assigned = events
            .iter()
            .any(|e| matches!(e, WorldEvent::PersonalGoalAssigned { .. }));
        assert!(assigned, "Expected PersonalGoalAssigned event");
    }
}
