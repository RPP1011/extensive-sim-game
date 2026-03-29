//! Adventurer rivalry system — every 300 ticks.
//!
//! Adventurers with low bonds develop grudges that affect party composition,
//! morale, and can escalate to duel challenges. Rivalries can be resolved
//! through mediation, forced cooperation, or natural attrition.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    lcg_f32, lcg_next, AdventurerStatus, CampaignState, ChoiceEffect, ChoiceEvent, ChoiceOption,
    ChoiceSource, QuestType, Rivalry, RivalryCause,
};
use crate::systems::bonds::bond_key;

/// Check if two adventurers have an active rivalry.
pub fn has_rivalry(state: &CampaignState, a: u32, b: u32) -> bool {
    let (lo, hi) = (a.min(b), a.max(b));
    state
        .rivalries
        .iter()
        .any(|r| r.adventurer_a == lo && r.adventurer_b == hi)
}

/// Get rivalry intensity between two adventurers (0 if no rivalry).
pub fn rivalry_intensity(state: &CampaignState, a: u32, b: u32) -> f32 {
    let (lo, hi) = (a.min(b), a.max(b));
    state
        .rivalries
        .iter()
        .find(|r| r.adventurer_a == lo && r.adventurer_b == hi)
        .map(|r| r.intensity)
        .unwrap_or(0.0)
}

/// Returns true if adventurer `a` refuses to party with adventurer `b` due to rivalry.
/// Threshold: intensity > 30.
pub fn refuses_party(state: &CampaignState, a: u32, b: u32) -> bool {
    rivalry_intensity(state, a, b) > 30.0
}

/// Main tick function. Called every 300 ticks.
///
/// 1. Form new rivalries from low bonds (bond < -10, 5% chance)
/// 2. Drift intensity: +1 if same guild, -0.5 if separated
/// 3. Apply morale penalties (intensity > 50, both at guild)
/// 4. Trigger duel challenges (intensity > 70)
/// 5. Clean up resolved rivalries (dead/retired adventurers)
pub fn tick_rivalries(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 10 != 0 {
        return;
    }

    // --- 1. Rivalry formation from low bonds ---
    // Collect adventurer pairs with bond < -10 that don't already have a rivalry.
    let alive_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    let mut new_rivalries = Vec::new();
    for (i, &a) in alive_ids.iter().enumerate() {
        for &b in &alive_ids[i + 1..] {
            let key = bond_key(a, b);
            let bond = *state.adventurer_bonds.get(&key).unwrap_or(&0.0);
            if bond < -10.0 && !has_rivalry(state, a, b) {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.05 {
                    // Select cause based on context
                    let cause = select_rivalry_cause(state, a, b);
                    let (lo, hi) = (a.min(b), a.max(b));
                    new_rivalries.push(Rivalry {
                        adventurer_a: lo,
                        adventurer_b: hi,
                        intensity: 10.0,
                        cause: cause.clone(),
                        started_tick: state.tick,
                    });
                    let cause_str = format!("{:?}", cause);
                    events.push(WorldEvent::RivalryFormed {
                        a: lo,
                        b: hi,
                        cause: cause_str,
                    });
                }
            }
        }
    }
    state.rivalries.extend(new_rivalries);

    // --- 2. Intensity drift ---
    // Both at guild (idle/assigned): +1.0 per tick cycle
    // Separated (one or both on quest/traveling): -0.5 per tick cycle
    let at_guild: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            matches!(
                a.status,
                AdventurerStatus::Idle | AdventurerStatus::Assigned
            )
        })
        .map(|a| a.id)
        .collect();

    for rivalry in &mut state.rivalries {
        let a_at_guild = at_guild.contains(&rivalry.adventurer_a);
        let b_at_guild = at_guild.contains(&rivalry.adventurer_b);
        if a_at_guild && b_at_guild {
            rivalry.intensity = (rivalry.intensity + 1.0).min(100.0);
        } else {
            rivalry.intensity = (rivalry.intensity - 0.5).max(0.0);
        }
    }

    // --- 3. Morale penalty when intensity > 50 and both at guild ---
    let morale_pairs: Vec<(u32, u32)> = state
        .rivalries
        .iter()
        .filter(|r| r.intensity > 50.0)
        .map(|r| (r.adventurer_a, r.adventurer_b))
        .collect();

    for (a, b) in &morale_pairs {
        let a_at_guild = at_guild.contains(a);
        let b_at_guild = at_guild.contains(b);
        if a_at_guild && b_at_guild {
            if let Some(adv) = state.adventurers.iter_mut().find(|x| x.id == *a) {
                adv.morale = (adv.morale - 5.0).max(0.0);
            }
            if let Some(adv) = state.adventurers.iter_mut().find(|x| x.id == *b) {
                adv.morale = (adv.morale - 5.0).max(0.0);
            }
        }
    }

    // --- 4. Duel challenges (intensity > 70) ---
    let duel_candidates: Vec<(u32, u32, f32)> = state
        .rivalries
        .iter()
        .filter(|r| r.intensity > 70.0)
        .map(|r| (r.adventurer_a, r.adventurer_b, r.intensity))
        .collect();

    for (a, b, _intensity) in duel_candidates {
        // Both must be idle/assigned (at guild) for a challenge
        let a_at_guild = at_guild.contains(&a);
        let b_at_guild = at_guild.contains(&b);
        if !a_at_guild || !b_at_guild {
            continue;
        }

        // 10% chance per tick cycle at intensity > 70
        let roll = lcg_f32(&mut state.rng);
        if roll >= 0.10 {
            continue;
        }

        // Pick challenger randomly
        let challenger;
        let challenged;
        if lcg_f32(&mut state.rng) < 0.5 {
            challenger = a;
            challenged = b;
        } else {
            challenger = b;
            challenged = a;
        }

        let challenger_name = state
            .adventurers
            .iter()
            .find(|x| x.id == challenger)
            .map(|x| x.name.clone())
            .unwrap_or_else(|| format!("Adventurer {}", challenger));
        let challenged_name = state
            .adventurers
            .iter()
            .find(|x| x.id == challenged)
            .map(|x| x.name.clone())
            .unwrap_or_else(|| format!("Adventurer {}", challenged));

        events.push(WorldEvent::RivalryDuel {
            challenger,
            challenged,
        });

        // Create choice event for the player
        let choice_id = state.next_event_id;
        state.next_event_id += 1;

        let choice = ChoiceEvent {
            id: choice_id,
            source: ChoiceSource::RivalryDuel {
                challenger,
                challenged,
            },
            prompt: format!(
                "{} has challenged {} to a duel! How do you respond?",
                challenger_name, challenged_name
            ),
            options: vec![
                ChoiceOption {
                    label: "Allow the duel".into(),
                    description: format!(
                        "Let them settle it. One will be injured, both lose morale."
                    ),
                    effects: vec![ChoiceEffect::Narrative(format!(
                        "{} and {} fight a duel.",
                        challenger_name, challenged_name
                    ))],
                },
                ChoiceOption {
                    label: "Mediate".into(),
                    description: "Step in and defuse the situation. Rivalry intensity -30."
                        .into(),
                    effects: vec![ChoiceEffect::Narrative(format!(
                        "You mediate between {} and {}.",
                        challenger_name, challenged_name
                    ))],
                },
                ChoiceOption {
                    label: "Punish the challenger".into(),
                    description: format!(
                        "Discipline {}. Their morale drops -15, but rivalry intensity -20.",
                        challenger_name
                    ),
                    effects: vec![ChoiceEffect::Narrative(format!(
                        "{} is disciplined for challenging {}.",
                        challenger_name, challenged_name
                    ))],
                },
            ],
            default_option: 1, // Mediate is the safe default
            deadline_ms: Some(state.elapsed_ms + 30_000), // 300 ticks deadline
            created_at_ms: state.elapsed_ms,
        };

        state.pending_choices.push(choice);
    }

    // --- 5. Clean up: remove rivalries involving dead adventurers or intensity <= 0 ---
    let dead_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    let mut resolved_events = Vec::new();
    state.rivalries.retain(|r| {
        if r.intensity <= 0.0
            || dead_ids.contains(&r.adventurer_a)
            || dead_ids.contains(&r.adventurer_b)
        {
            resolved_events.push(WorldEvent::RivalryResolved {
                a: r.adventurer_a,
                b: r.adventurer_b,
            });
            false
        } else {
            true
        }
    });
    events.extend(resolved_events);
}

/// Select a rivalry cause based on adventurer context.
fn select_rivalry_cause(state: &CampaignState, a: u32, b: u32) -> RivalryCause {
    // Check if both are assigned to same quest type
    let a_quest_type = find_quest_type_for_adventurer(state, a);
    let b_quest_type = find_quest_type_for_adventurer(state, b);
    if let (Some(aqt), Some(bqt)) = (a_quest_type, b_quest_type) {
        if aqt == bqt {
            return RivalryCause::QuestCompetition;
        }
    }

    // Check level difference for professional jealousy
    let a_level = state
        .adventurers
        .iter()
        .find(|x| x.id == a)
        .map(|x| x.level)
        .unwrap_or(1);
    let b_level = state
        .adventurers
        .iter()
        .find(|x| x.id == b)
        .map(|x| x.level)
        .unwrap_or(1);
    if a_level.abs_diff(b_level) >= 3 {
        return RivalryCause::ProfessionalJealousy;
    }

    // Random from remaining causes
    let roll = lcg_next(&mut state.rng.clone()) % 3;
    match roll {
        0 => RivalryCause::PersonalInsult,
        1 => RivalryCause::LoyaltyConflict,
        _ => RivalryCause::RomanticRivalry,
    }
}

/// Find the quest type an adventurer is working on (if any).
fn find_quest_type_for_adventurer(state: &CampaignState, adv_id: u32) -> Option<QuestType> {
    for quest in &state.active_quests {
        if quest.assigned_pool.contains(&adv_id) {
            return Some(quest.request.quest_type);
        }
    }
    None
}

/// Apply duel outcome based on choice option index.
/// Called from step.rs when a RivalryDuel choice is resolved.
pub fn apply_duel_outcome(
    state: &mut CampaignState,
    challenger: u32,
    challenged: u32,
    option_index: usize,
) {
    match option_index {
        0 => {
            // Allow duel — one gets injured, both lose morale
            // Determine winner via RNG
            let winner;
            let loser;
            if lcg_f32(&mut state.rng) < 0.5 {
                winner = challenger;
                loser = challenged;
            } else {
                winner = challenged;
                loser = challenger;
            }

            // Winner: +10 morale
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == winner) {
                adv.morale = (adv.morale + 10.0).min(100.0);
            }
            // Loser: -10 morale, injured
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == loser) {
                adv.morale = (adv.morale - 10.0).max(0.0);
                adv.injury = (adv.injury + 20.0).min(100.0);
                if adv.injury >= 90.0 {
                    adv.status = AdventurerStatus::Injured;
                }
            }

            // Rivalry intensity -10 (they settled it somewhat)
            let (lo, hi) = (challenger.min(challenged), challenger.max(challenged));
            if let Some(r) = state
                .rivalries
                .iter_mut()
                .find(|r| r.adventurer_a == lo && r.adventurer_b == hi)
            {
                r.intensity = (r.intensity - 10.0).max(0.0);
            }
        }
        1 => {
            // Mediate — intensity -30
            let (lo, hi) = (challenger.min(challenged), challenger.max(challenged));
            if let Some(r) = state
                .rivalries
                .iter_mut()
                .find(|r| r.adventurer_a == lo && r.adventurer_b == hi)
            {
                r.intensity = (r.intensity - 30.0).max(0.0);
            }
        }
        2 => {
            // Punish challenger — challenger morale -15, intensity -20
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == challenger) {
                adv.morale = (adv.morale - 15.0).max(0.0);
            }
            let (lo, hi) = (challenger.min(challenged), challenger.max(challenged));
            if let Some(r) = state
                .rivalries
                .iter_mut()
                .find(|r| r.adventurer_a == lo && r.adventurer_b == hi)
            {
                r.intensity = (r.intensity - 20.0).max(0.0);
            }
        }
        _ => {} // Invalid option — do nothing
    }
}

/// Reduce rivalry intensity when adventurers complete a quest together.
/// Called from quest lifecycle when a quest succeeds.
pub fn on_quest_completed_together(state: &mut CampaignState, member_ids: &[u32]) {
    for (i, &a) in member_ids.iter().enumerate() {
        for &b in &member_ids[i + 1..] {
            let (lo, hi) = (a.min(b), a.max(b));
            if let Some(r) = state
                .rivalries
                .iter_mut()
                .find(|r| r.adventurer_a == lo && r.adventurer_b == hi)
            {
                r.intensity = (r.intensity - 20.0).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::state::{
        Adventurer, AdventurerStats, CampaignPhase, Equipment,
    };

    fn make_test_adventurer(id: u32) -> Adventurer {
        Adventurer {
            id,
            name: format!("Adventurer {}", id),
            archetype: "knight".into(),
            level: 1,
            xp: 0,
            stats: AdventurerStats::default(),
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 50.0,
            stress: 10.0,
            fatigue: 5.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            ..Default::default()
        }
    }

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.adventurers.push(make_test_adventurer(1));
        state.adventurers.push(make_test_adventurer(2));
        state.adventurers.push(make_test_adventurer(3));
        state
    }

    #[test]
    fn rivalry_formation_from_low_bond() {
        let mut state = make_test_state();
        // Ensure at least 2 adventurers
        assert!(state.adventurers.len() >= 2);
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        // Set bond to very negative
        state
            .adventurer_bonds
            .insert(bond_key(a, b), -20.0);
        state.tick = 300;

        // Force RNG to produce a roll < 0.05
        // Run multiple times until we get a rivalry (or test the check)
        let mut formed = false;
        for seed in 0..1000u64 {
            let mut s = state.clone();
            s.rng = seed;
            let mut events = Vec::new();
            let mut deltas = StepDeltas::default();
            tick_rivalries(&mut s, &mut deltas, &mut events);
            if !s.rivalries.is_empty() {
                formed = true;
                assert!(has_rivalry(&s, a, b));
                assert!(events.iter().any(|e| matches!(e, WorldEvent::RivalryFormed { .. })));
                break;
            }
        }
        assert!(formed, "Should form a rivalry with low bond given enough RNG seeds");
    }

    #[test]
    fn no_rivalry_without_low_bond() {
        let mut state = make_test_state();
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        // Bond is positive — no rivalry should form
        state.adventurer_bonds.insert(bond_key(a, b), 50.0);
        state.tick = 300;

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_rivalries(&mut state, &mut deltas, &mut events);
        assert!(state.rivalries.is_empty());
    }

    #[test]
    fn rivalry_refuses_party_above_threshold() {
        let mut state = make_test_state();
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        state.rivalries.push(Rivalry {
            adventurer_a: a.min(b),
            adventurer_b: a.max(b),
            intensity: 35.0,
            cause: RivalryCause::PersonalInsult,
            started_tick: 0,
        });

        assert!(refuses_party(&state, a, b));
    }

    #[test]
    fn rivalry_removed_on_death() {
        let mut state = make_test_state();
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        state.rivalries.push(Rivalry {
            adventurer_a: a.min(b),
            adventurer_b: a.max(b),
            intensity: 50.0,
            cause: RivalryCause::QuestCompetition,
            started_tick: 0,
        });

        // Kill adventurer a
        if let Some(adv) = state.adventurers.iter_mut().find(|x| x.id == a) {
            adv.status = AdventurerStatus::Dead;
        }

        state.tick = 300;
        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_rivalries(&mut state, &mut deltas, &mut events);

        assert!(state.rivalries.is_empty());
        assert!(events.iter().any(|e| matches!(e, WorldEvent::RivalryResolved { .. })));
    }

    #[test]
    fn quest_completion_reduces_rivalry() {
        let mut state = make_test_state();
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        state.rivalries.push(Rivalry {
            adventurer_a: a.min(b),
            adventurer_b: a.max(b),
            intensity: 50.0,
            cause: RivalryCause::ProfessionalJealousy,
            started_tick: 0,
        });

        on_quest_completed_together(&mut state, &[a, b]);

        let r = state.rivalries.first().unwrap();
        assert!((r.intensity - 30.0).abs() < 0.01);
    }

    #[test]
    fn duel_mediation_reduces_intensity() {
        let mut state = make_test_state();
        let a = state.adventurers[0].id;
        let b = state.adventurers[1].id;

        state.rivalries.push(Rivalry {
            adventurer_a: a.min(b),
            adventurer_b: a.max(b),
            intensity: 80.0,
            cause: RivalryCause::PersonalInsult,
            started_tick: 0,
        });

        apply_duel_outcome(&mut state, a, b, 1); // mediate

        let r = state.rivalries.first().unwrap();
        assert!((r.intensity - 50.0).abs() < 0.01);
    }
}
