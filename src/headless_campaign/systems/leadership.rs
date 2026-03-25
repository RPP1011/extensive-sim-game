//! Guild leadership and succession system — fires every 500 ticks.
//!
//! The guild has a single leader whose `LeadershipStyle` provides global bonuses.
//! When the leader dies or retires (approval < 20), a succession crisis ensues:
//! 300 ticks of morale/performance penalties before a new leader is appointed
//! (either via council vote or auto-select highest level adventurer).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to run leadership logic (in ticks).
const LEADERSHIP_INTERVAL: u64 = 17;

/// Duration of succession crisis in ticks.
const SUCCESSION_CRISIS_TICKS: u64 = 10;

/// Approval threshold below which a leader voluntarily retires.
const LOW_APPROVAL_THRESHOLD: f32 = 20.0;

/// Main tick function. Runs every 500 ticks.
///
/// 1. If no leader exists: auto-appoint or handle succession crisis countdown
/// 2. Update approval rating based on guild performance
/// 3. Check for leader death or low-approval retirement
/// 4. Apply leadership bonuses
pub fn tick_leadership(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % LEADERSHIP_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Handle succession crisis countdown ---
    if let Some(ref mut crisis) = state.succession_crisis {
        if state.tick >= crisis.crisis_start_tick + SUCCESSION_CRISIS_TICKS {
            // Crisis period over — appoint new leader
            let new_leader_id = select_new_leader(state);
            if let Some(id) = new_leader_id {
                appoint_leader(state, id, events);
            }
            state.succession_crisis = None;
        } else {
            // Still in crisis — apply morale penalty
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale - 2.0).max(0.0);
                }
            }
        }
        return;
    }

    // --- Check if current leader is still alive ---
    if let Some(ref leader) = state.guild_leader {
        let leader_id = leader.adventurer_id;
        let leader_alive = state
            .adventurers
            .iter()
            .any(|a| a.id == leader_id && a.status != AdventurerStatus::Dead);

        if !leader_alive {
            // Leader died — trigger succession crisis
            events.push(WorldEvent::LeaderDied {
                adventurer_id: leader_id,
            });
            events.push(WorldEvent::SuccessionCrisis);
            state.guild_leader = None;
            state.succession_crisis = Some(SuccessionCrisis {
                crisis_start_tick: state.tick,
                previous_leader_id: leader_id,
            });
            return;
        }
    }

    // --- Auto-appoint if no leader ---
    if state.guild_leader.is_none() {
        let best_id = select_new_leader(state);
        if let Some(id) = best_id {
            appoint_leader(state, id, events);
        }
        return;
    }

    // --- Update approval rating ---
    update_approval(state);

    // --- Check low approval → retirement ---
    let should_retire = state
        .guild_leader
        .as_ref()
        .map(|l| l.approval_rating < LOW_APPROVAL_THRESHOLD)
        .unwrap_or(false);

    if should_retire {
        let leader_id = state.guild_leader.as_ref().unwrap().adventurer_id;
        let old_approval = state.guild_leader.as_ref().unwrap().approval_rating;
        events.push(WorldEvent::LeaderRetired {
            adventurer_id: leader_id,
            approval_rating: old_approval,
        });

        // Check if council can vote for replacement
        let has_council = state
            .adventurers
            .iter()
            .filter(|a| a.level >= 5 && a.status != AdventurerStatus::Dead)
            .count()
            >= 3;

        if has_council {
            // Council succession — short crisis
            events.push(WorldEvent::SuccessionCrisis);
            state.guild_leader = None;
            state.succession_crisis = Some(SuccessionCrisis {
                crisis_start_tick: state.tick,
                previous_leader_id: leader_id,
            });
        } else {
            // Immediate replacement — no crisis
            state.guild_leader = None;
            let new_id = select_new_leader(state);
            if let Some(id) = new_id {
                appoint_leader(state, id, events);
            }
        }
        return;
    }

    // --- Emit approval change event ---
    if let Some(ref leader) = state.guild_leader {
        events.push(WorldEvent::ApprovalChanged {
            adventurer_id: leader.adventurer_id,
            approval_rating: leader.approval_rating,
        });
    }
}

/// Select the best candidate for leadership: highest level alive adventurer.
/// Ties broken by highest loyalty.
fn select_new_leader(state: &CampaignState) -> Option<u32> {
    state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .max_by(|a, b| {
            a.level
                .cmp(&b.level)
                .then(a.loyalty.partial_cmp(&b.loyalty).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|a| a.id)
}

/// Determine leadership style from an adventurer's dominant history tags.
fn determine_style(adv: &Adventurer) -> LeadershipStyle {
    // Map history tag keywords to styles
    let style_scores = [
        (
            LeadershipStyle::Aggressive,
            tag_score(adv, &["combat", "attack", "kill", "war", "battle"]),
        ),
        (
            LeadershipStyle::Cautious,
            tag_score(adv, &["defense", "defend", "guard", "protect", "retreat"]),
        ),
        (
            LeadershipStyle::Diplomatic,
            tag_score(adv, &["diplomacy", "negotiate", "alliance", "relation", "peace"]),
        ),
        (
            LeadershipStyle::Mercantile,
            tag_score(adv, &["trade", "merchant", "gold", "market", "caravan"]),
        ),
        (
            LeadershipStyle::Scholarly,
            tag_score(adv, &["research", "study", "lore", "knowledge", "explore"]),
        ),
        (
            LeadershipStyle::Inspirational,
            tag_score(adv, &["morale", "inspire", "lead", "rally", "mentor"]),
        ),
    ];

    style_scores
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(style, _)| *style)
        .unwrap_or(LeadershipStyle::Inspirational)
}

/// Sum history tag counts that contain any of the given keywords.
fn tag_score(adv: &Adventurer, keywords: &[&str]) -> u32 {
    adv.history_tags
        .iter()
        .filter(|(tag, _)| keywords.iter().any(|kw| tag.contains(kw)))
        .map(|(_, count)| count)
        .sum()
}

/// Appoint a new leader and emit the event.
fn appoint_leader(state: &mut CampaignState, adventurer_id: u32, events: &mut Vec<WorldEvent>) {
    let adv = match state.adventurers.iter().find(|a| a.id == adventurer_id) {
        Some(a) => a,
        None => return,
    };

    let style = determine_style(adv);

    events.push(WorldEvent::LeaderAppointed {
        adventurer_id,
        style: format!("{:?}", style),
    });

    state.guild_leader = Some(GuildLeader {
        adventurer_id,
        appointed_tick: state.tick,
        leadership_style: style,
        approval_rating: 50.0,
        decisions_made: 0,
    });
}

/// Update approval rating based on recent guild performance.
fn update_approval(state: &mut CampaignState) {
    let leader = match state.guild_leader.as_mut() {
        Some(l) => l,
        None => return,
    };

    leader.decisions_made += 1;

    let mut approval_delta: f32 = 0.0;

    // Wins increase approval
    let recent_wins = state
        .completed_quests
        .iter()
        .rev()
        .take(5)
        .filter(|q| q.result == QuestResult::Victory)
        .count() as f32;
    approval_delta += recent_wins * 2.0;

    // Losses decrease approval
    let recent_losses = state
        .completed_quests
        .iter()
        .rev()
        .take(5)
        .filter(|q| q.result == QuestResult::Defeat || q.result == QuestResult::Abandoned)
        .count() as f32;
    approval_delta -= recent_losses * 3.0;

    // Gold growth → positive
    if state.guild.gold > 200.0 {
        approval_delta += 1.0;
    } else if state.guild.gold < 50.0 {
        approval_delta -= 2.0;
    }

    // Dead adventurers → negative
    let dead_count = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .count() as f32;
    approval_delta -= dead_count * 0.5;

    // Mean drift toward 50
    let drift = (50.0 - leader.approval_rating) * 0.05;
    approval_delta += drift;

    leader.approval_rating = (leader.approval_rating + approval_delta).clamp(0.0, 100.0);
}

/// Manually appoint a leader (called from `apply_action` for `AppointLeader`).
pub fn apply_appoint_leader(
    state: &mut CampaignState,
    adventurer_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<(), String> {
    let adv_alive = state
        .adventurers
        .iter()
        .any(|a| a.id == adventurer_id && a.status != AdventurerStatus::Dead);

    if !adv_alive {
        return Err("Adventurer not found or dead".into());
    }

    // If there is a current leader, they retire
    if let Some(ref old_leader) = state.guild_leader {
        events.push(WorldEvent::LeaderRetired {
            adventurer_id: old_leader.adventurer_id,
            approval_rating: old_leader.approval_rating,
        });
    }

    appoint_leader(state, adventurer_id, events);

    // Clear any active succession crisis
    state.succession_crisis = None;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_appoint_highest_level() {
        let mut state = CampaignState::default_test_campaign(42);
        // Ensure at least 2 adventurers with different levels
        if state.adventurers.len() >= 2 {
            state.adventurers[0].level = 5;
            state.adventurers[1].level = 10;
        }

        let best = select_new_leader(&state);
        assert!(best.is_some());
        // Should pick the level-10 adventurer
        if state.adventurers.len() >= 2 {
            assert_eq!(best.unwrap(), state.adventurers[1].id);
        }
    }

    #[test]
    fn style_from_tags() {
        let mut adv = Adventurer {
            id: 1,
            name: "Test".into(),
            archetype: "knight".into(),
            level: 5,
            xp: 0,
            stats: AdventurerStats::default(),
            equipment: Equipment::default(),
            traits: vec![],
            status: AdventurerStatus::Idle,
            loyalty: 80.0,
            stress: 10.0,
            fatigue: 10.0,
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
            backstory: None,
            deeds: vec![],
            hobbies: vec![],
            disease_status: Default::default(),
            mood_state: Default::default(),
            fears: vec![],
            personal_goal: None,
            journal: vec![],
            equipped_items: vec![],
            nicknames: vec![],
        };

        // No tags → default Inspirational
        assert_eq!(determine_style(&adv), LeadershipStyle::Inspirational);

        // Combat tags → Aggressive
        adv.history_tags.insert("combat".into(), 10);
        adv.history_tags.insert("battle".into(), 5);
        assert_eq!(determine_style(&adv), LeadershipStyle::Aggressive);

        // Trade tags dominant
        adv.history_tags.clear();
        adv.history_tags.insert("trade".into(), 20);
        assert_eq!(determine_style(&adv), LeadershipStyle::Mercantile);
    }

    #[test]
    fn succession_crisis_on_death() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();

        // Appoint leader
        state.tick = 500;
        state.guild_leader = None;
        tick_leadership(&mut state, &mut deltas, &mut events);
        assert!(state.guild_leader.is_some());

        // Kill the leader
        let leader_id = state.guild_leader.as_ref().unwrap().adventurer_id;
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == leader_id) {
            adv.status = AdventurerStatus::Dead;
        }

        events.clear();
        state.tick = 1000;
        tick_leadership(&mut state, &mut deltas, &mut events);
        assert!(state.guild_leader.is_none());
        assert!(state.succession_crisis.is_some());
        assert!(events.iter().any(|e| matches!(e, WorldEvent::LeaderDied { .. })));
        assert!(events.iter().any(|e| matches!(e, WorldEvent::SuccessionCrisis)));
    }
}
