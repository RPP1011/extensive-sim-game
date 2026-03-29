//! Guild council voting system — every 200 ticks.
//!
//! Senior adventurers (level >= 5) form the guild council and vote on major
//! decisions. Personality traits (loyalty, morale, stress) and bonds between
//! adventurers influence voting behavior deterministically via the campaign LCG.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;
use crate::systems::bonds::bond_strength;

/// Main tick function. Called every 200 ticks.
///
/// 1. Resolve any votes past their deadline
/// 2. Auto-propose votes when conditions are met
pub fn tick_council(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 || state.tick == 0 {
        return;
    }

    // Collect council member IDs (adventurers with level >= 5, alive).
    let council_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.level >= 5 && a.status != AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    if council_ids.is_empty() {
        return;
    }

    // --- Resolve votes past deadline ---
    resolve_pending_votes(state, &council_ids, events);

    // --- Auto-propose votes based on conditions ---
    auto_propose_votes(state, &council_ids, events);
}

/// Resolve all pending votes that have passed their deadline.
fn resolve_pending_votes(
    state: &mut CampaignState,
    council_ids: &[u32],
    events: &mut Vec<WorldEvent>,
) {
    let tick = state.tick;
    let bonds = state.adventurer_bonds.clone();

    // Snapshot adventurer data for voting decisions (avoids borrow issues).
    let adv_data: Vec<(u32, f32, f32, f32, f32, f32)> = state
        .adventurers
        .iter()
        .map(|a| (a.id, a.loyalty, a.morale, a.stress, a.fatigue, a.injury))
        .collect();

    // Indices of votes to resolve.
    let to_resolve: Vec<usize> = state
        .council_votes
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.resolved && tick >= v.deadline_tick)
        .map(|(i, _)| i)
        .collect();

    for idx in to_resolve {
        let vote = &mut state.council_votes[idx];

        // Cast votes from all council members who haven't voted yet.
        for &cid in council_ids {
            if vote.votes_for.contains(&cid) || vote.votes_against.contains(&cid) {
                continue;
            }
            let in_favor = should_vote_for(
                cid, &vote.topic, &adv_data, &bonds, council_ids,
            );
            if in_favor {
                vote.votes_for.push(cid);
            } else {
                vote.votes_against.push(cid);
            }
        }

        // Resolve by majority.
        let passed = vote.votes_for.len() > vote.votes_against.len();
        vote.resolved = true;
        vote.passed = passed;

        let topic_str = vote.topic.description();

        events.push(WorldEvent::CouncilVoteResolved {
            passed,
            topic: topic_str.clone(),
        });

        // Morale penalty for losing side.
        let losers: Vec<u32> = if passed {
            vote.votes_against.clone()
        } else {
            vote.votes_for.clone()
        };
        for &loser_id in &losers {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == loser_id) {
                adv.morale = (adv.morale - 5.0).max(0.0);
            }
        }

        // Apply vote effects if passed.
        if passed {
            apply_vote_effects(state, &state.council_votes[idx].topic.clone(), events);
        }
    }
}

/// Apply the effects of a passed vote.
fn apply_vote_effects(
    state: &mut CampaignState,
    topic: &VoteTopic,
    _events: &mut Vec<WorldEvent>,
) {
    match topic {
        VoteTopic::DeclareWar { faction_id } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *faction_id) {
                faction.diplomatic_stance = DiplomaticStance::AtWar;
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - 30.0).max(-100.0);
            }
        }
        VoteTopic::SuePeace { faction_id } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *faction_id) {
                if faction.diplomatic_stance == DiplomaticStance::AtWar {
                    faction.diplomatic_stance = DiplomaticStance::Hostile;
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild + 10.0).min(100.0);
                }
            }
        }
        VoteTopic::ExileAdventurer { adventurer_id } => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == *adventurer_id)
            {
                adv.status = AdventurerStatus::Dead; // Exiled = removed from play
                adv.loyalty = 0.0;
            }
            // Remove bonds involving exiled adventurer.
            let eid = *adventurer_id;
            state
                .adventurer_bonds
                .retain(|&(a, b), _| a != eid && b != eid);
        }
        VoteTopic::ChangePolicy { .. } => {
            // Policy changes are narrative — no mechanical effect beyond the vote itself.
            // Boost guild reputation slightly for democratic governance.
            state.guild.reputation = (state.guild.reputation + 2.0).min(100.0);
        }
        VoteTopic::MajorExpedition { region_id } => {
            // Scout the target region if not already scouted.
            if let Some(loc) = state
                .overworld
                .locations
                .iter_mut()
                .find(|l| l.id == *region_id)
            {
                loc.scouted = true;
            }
        }
        VoteTopic::AllianceProposal { faction_id } => {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *faction_id) {
                if faction.relationship_to_guild > 0.0 {
                    faction.diplomatic_stance = DiplomaticStance::Friendly;
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild + 15.0).min(100.0);
                }
            }
        }
    }
}

/// Auto-propose votes when certain conditions are met.
fn auto_propose_votes(
    state: &mut CampaignState,
    _council_ids: &[u32],
    events: &mut Vec<WorldEvent>,
) {
    let tick = state.tick;

    // Collect proposals first, then apply them (avoids borrow conflicts).
    let mut proposals: Vec<VoteTopic> = Vec::new();

    // Helper: check if a pending vote with this description already exists.
    let has_pending_desc = |votes: &[CouncilVote], desc: &str| -> bool {
        votes.iter().any(|v| !v.resolved && v.topic.description() == desc)
    };

    // War exhaustion → propose SuePeace.
    let factions_at_war: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .map(|f| f.id)
        .collect();

    for fid in factions_at_war {
        let alive_count = state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .count();
        if alive_count == 0 {
            continue;
        }
        let avg_exhaustion: f32 = state
            .adventurers
            .iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.fatigue + a.stress)
            .sum::<f32>()
            / alive_count as f32;

        if avg_exhaustion > 50.0 {
            let topic = VoteTopic::SuePeace { faction_id: fid };
            if !has_pending_desc(&state.council_votes, &topic.description()) {
                proposals.push(topic);
            }
        }
    }

    // Adventurer loyalty < 20 → propose ExileAdventurer.
    let disloyal: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.loyalty < 20.0
                && a.status != AdventurerStatus::Dead
                && a.level >= 1
        })
        .map(|a| a.id)
        .collect();

    for aid in disloyal {
        let topic = VoteTopic::ExileAdventurer { adventurer_id: aid };
        if !has_pending_desc(&state.council_votes, &topic.description()) {
            proposals.push(topic);
        }
    }

    // Now apply all proposals.
    for topic in proposals {
        propose_vote(state, topic, tick, events);
    }
}

/// Create a new council vote proposal.
fn propose_vote(
    state: &mut CampaignState,
    topic: VoteTopic,
    tick: u64,
    events: &mut Vec<WorldEvent>,
) {
    let id = state.next_vote_id;
    state.next_vote_id += 1;

    let description = topic.description();

    state.council_votes.push(CouncilVote {
        id,
        topic,
        proposed_tick: tick,
        deadline_tick: tick + 400, // 400 ticks = 40 seconds to vote
        votes_for: Vec::new(),
        votes_against: Vec::new(),
        resolved: false,
        passed: false,
    });

    events.push(WorldEvent::CouncilVoteProposed {
        topic: description,
    });
}

/// Determine whether an adventurer votes for a given topic.
/// Uses personality traits and bonds — fully deterministic (no RNG needed).
fn should_vote_for(
    adv_id: u32,
    topic: &VoteTopic,
    adv_data: &[(u32, f32, f32, f32, f32, f32)], // (id, loyalty, morale, stress, fatigue, injury)
    bonds: &std::collections::HashMap<(u32, u32), f32>,
    _council_ids: &[u32],
) -> bool {
    let (_, loyalty, morale, stress, fatigue, injury) = adv_data
        .iter()
        .find(|(id, ..)| *id == adv_id)
        .copied()
        .unwrap_or((adv_id, 50.0, 50.0, 25.0, 25.0, 0.0));

    match topic {
        VoteTopic::DeclareWar { .. } => {
            // High-loyalty adventurers vote for war, cautious (high stress) against.
            loyalty > 50.0 && stress < 40.0 && morale > 40.0
        }
        VoteTopic::SuePeace { .. } => {
            // Exhausted/injured adventurers vote for peace.
            fatigue > 30.0 || injury > 20.0 || stress > 50.0 || morale < 30.0
        }
        VoteTopic::ExileAdventurer { adventurer_id } => {
            // Bonded allies vote against exile, rivals vote for.
            let bond = bond_strength(bonds, adv_id, *adventurer_id);
            if bond > 30.0 {
                false // Ally — protect them
            } else if bond < 5.0 {
                // No bond — vote based on loyalty (loyal members exile traitors)
                loyalty > 40.0
            } else {
                // Weak bond — mixed feelings, lean toward exile if loyal
                loyalty > 60.0
            }
        }
        VoteTopic::ChangePolicy { .. } => {
            // Moderate adventurers vote for change, stubborn ones against.
            morale > 30.0 && stress < 60.0
        }
        VoteTopic::MajorExpedition { .. } => {
            // Adventurous (low fatigue, high morale) vote for expeditions.
            fatigue < 40.0 && morale > 50.0 && injury < 20.0
        }
        VoteTopic::AllianceProposal { .. } => {
            // Diplomatic types (high loyalty, low stress) vote for alliances.
            loyalty > 30.0 && stress < 50.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn war_vote_loyalty_matters() {
        let adv_data = vec![(1, 80.0, 70.0, 20.0, 10.0, 0.0)];
        let bonds = HashMap::new();
        let topic = VoteTopic::DeclareWar { faction_id: 0 };
        assert!(should_vote_for(1, &topic, &adv_data, &bonds, &[1]));

        // Low loyalty -> against war
        let adv_data = vec![(1, 30.0, 70.0, 20.0, 10.0, 0.0)];
        assert!(!should_vote_for(1, &topic, &adv_data, &bonds, &[1]));
    }

    #[test]
    fn peace_vote_exhaustion_matters() {
        let adv_data = vec![(1, 80.0, 70.0, 10.0, 50.0, 0.0)];
        let bonds = HashMap::new();
        let topic = VoteTopic::SuePeace { faction_id: 0 };
        // High fatigue -> votes for peace
        assert!(should_vote_for(1, &topic, &adv_data, &bonds, &[1]));

        // Low fatigue, low stress, no injury -> against peace
        let adv_data = vec![(1, 80.0, 70.0, 10.0, 10.0, 0.0)];
        assert!(!should_vote_for(1, &topic, &adv_data, &bonds, &[1]));
    }

    #[test]
    fn exile_vote_bonds_matter() {
        let mut bonds = HashMap::new();
        bonds.insert((1, 2), 60.0); // Strong bond between 1 and 2
        let adv_data = vec![(1, 80.0, 70.0, 20.0, 10.0, 0.0)];

        // Adventurer 1 should vote against exiling bonded ally 2
        let topic = VoteTopic::ExileAdventurer { adventurer_id: 2 };
        assert!(!should_vote_for(1, &topic, &adv_data, &bonds, &[1]));

        // Adventurer 1 should vote for exiling stranger 3 (loyal member)
        let topic = VoteTopic::ExileAdventurer { adventurer_id: 3 };
        assert!(should_vote_for(1, &topic, &adv_data, &bonds, &[1]));
    }

    #[test]
    fn vote_topic_description() {
        let topic = VoteTopic::DeclareWar { faction_id: 2 };
        assert!(topic.description().contains("war"));

        let topic = VoteTopic::SuePeace { faction_id: 1 };
        assert!(topic.description().contains("peace"));
    }
}
