#![allow(unused)]
//! Guild council voting system — every 7 ticks.
//!
//! Senior adventurers (level >= 5) form the guild council and vote on major
//! decisions. Personality traits (loyalty, morale, stress) and bonds between
//! adventurers influence voting behavior deterministically.
//!
//! Ported from `crates/headless_campaign/src/systems/council.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: council_votes: Vec<CouncilVote> on WorldState
//   CouncilVote { id: u32, topic: VoteTopic, proposed_tick: u64,
//                 deadline_tick: u64, votes_for: Vec<u32>,
//                 votes_against: Vec<u32>, resolved: bool, passed: bool }
// NEEDS STATE: VoteTopic enum { DeclareWar { faction_id }, SuePeace { faction_id },
//              ExileAdventurer { adventurer_id }, ChangePolicy { description },
//              MajorExpedition { region_id }, AllianceProposal { faction_id } }
// NEEDS STATE: next_vote_id: u32 on WorldState
// NEEDS STATE: adventurer_bonds: HashMap<(u32, u32), f32> on WorldState
// NEEDS STATE: Entity.npc extended with loyalty, morale, stress, fatigue, injury fields
// NEEDS STATE: factions: Vec<FactionState> on WorldState
// NEEDS STATE: DiplomaticStance enum

// NEEDS DELTA: CastVote { vote_id: u32, voter_id: u32, in_favor: bool }
// NEEDS DELTA: ResolveVote { vote_id: u32, passed: bool }
// NEEDS DELTA: ProposeVote { topic: VoteTopic, deadline_tick: u64 }
// NEEDS DELTA: AdjustMorale { entity_id: u32, delta: f32 }
// NEEDS DELTA: SetDiplomaticStance { faction_id: u32, stance: DiplomaticStance }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: KillEntity { entity_id: u32 }
// NEEDS DELTA: AdjustReputation { delta: f32 }
// NEEDS DELTA: ScoutRegion { region_id: u32 }

/// Cadence: every 7 ticks.
const COUNCIL_INTERVAL: u64 = 7;

/// Vote deadline: 400 ticks after proposal.
const VOTE_DEADLINE_TICKS: u64 = 400;

pub fn compute_council(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % COUNCIL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once the council state fields exist, this system will:
    // 1. Collect council member IDs (entities with level >= 5, alive NPCs)
    // 2. Resolve pending votes past their deadline
    // 3. Auto-propose new votes based on conditions

    /*
    // --- Collect council members ---
    let council_ids: Vec<u32> = state.entities.iter()
        .filter(|e| e.kind == crate::world_sim::state::EntityKind::Npc
                && e.alive
                && e.level >= 5)
        .map(|e| e.id)
        .collect();

    if council_ids.is_empty() {
        return;
    }

    // --- Resolve votes past deadline ---
    resolve_pending_votes(state, &council_ids, out);

    // --- Auto-propose votes ---
    auto_propose_votes(state, &council_ids, out);
    */
}

// ---------------------------------------------------------------------------
// Vote resolution
// ---------------------------------------------------------------------------

/*
fn resolve_pending_votes(
    state: &WorldState,
    council_ids: &[u32],
    out: &mut Vec<WorldDelta>,
) {
    let tick = state.tick;

    for vote in &state.council_votes {
        if vote.resolved || tick < vote.deadline_tick {
            continue;
        }

        // Cast votes from all council members who haven't voted yet
        for &cid in council_ids {
            if vote.votes_for.contains(&cid) || vote.votes_against.contains(&cid) {
                continue;
            }

            let in_favor = should_vote_for(cid, &vote.topic, state);
            out.push(WorldDelta::CastVote {
                vote_id: vote.id,
                voter_id: cid,
                in_favor,
            });
        }

        // Determine outcome: majority wins
        // Count existing votes plus newly cast ones (approximate: use current counts
        // since all council members who haven't voted will vote now)
        let mut total_for = vote.votes_for.len();
        let mut total_against = vote.votes_against.len();
        for &cid in council_ids {
            if vote.votes_for.contains(&cid) || vote.votes_against.contains(&cid) {
                continue;
            }
            if should_vote_for(cid, &vote.topic, state) {
                total_for += 1;
            } else {
                total_against += 1;
            }
        }

        let passed = total_for > total_against;
        out.push(WorldDelta::ResolveVote {
            vote_id: vote.id,
            passed,
        });

        // Morale penalty for losing side
        let loser_ids: Vec<u32> = if passed {
            // Those who voted against (including newly cast)
            let mut losers = vote.votes_against.clone();
            for &cid in council_ids {
                if !vote.votes_for.contains(&cid) && !vote.votes_against.contains(&cid) {
                    if !should_vote_for(cid, &vote.topic, state) {
                        losers.push(cid);
                    }
                }
            }
            losers
        } else {
            let mut losers = vote.votes_for.clone();
            for &cid in council_ids {
                if !vote.votes_for.contains(&cid) && !vote.votes_against.contains(&cid) {
                    if should_vote_for(cid, &vote.topic, state) {
                        losers.push(cid);
                    }
                }
            }
            losers
        };

        for &loser_id in &loser_ids {
            out.push(WorldDelta::AdjustMorale {
                entity_id: loser_id,
                delta: -5.0,
            });
        }

        // Apply vote effects if passed
        if passed {
            apply_vote_effects(&vote.topic, state, out);
        }
    }
}

fn apply_vote_effects(
    topic: &VoteTopic,
    state: &WorldState,
    out: &mut Vec<WorldDelta>,
) {
    match topic {
        VoteTopic::DeclareWar { faction_id } => {
            out.push(WorldDelta::SetDiplomaticStance {
                faction_id: *faction_id,
                stance: DiplomaticStance::AtWar,
            });
            out.push(WorldDelta::AdjustRelationship {
                faction_id: *faction_id,
                delta: -30.0,
            });
        }
        VoteTopic::SuePeace { faction_id } => {
            // Only if currently at war
            if let Some(faction) = state.factions.iter().find(|f| f.id == *faction_id) {
                if faction.diplomatic_stance == DiplomaticStance::AtWar {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: *faction_id,
                        stance: DiplomaticStance::Hostile,
                    });
                    out.push(WorldDelta::AdjustRelationship {
                        faction_id: *faction_id,
                        delta: 10.0,
                    });
                }
            }
        }
        VoteTopic::ExileAdventurer { adventurer_id } => {
            out.push(WorldDelta::Die { entity_id: *adventurer_id });
        }
        VoteTopic::ChangePolicy { .. } => {
            // Narrative effect only — boost guild reputation
            out.push(WorldDelta::AdjustReputation { delta: 2.0 });
        }
        VoteTopic::MajorExpedition { region_id } => {
            out.push(WorldDelta::ScoutRegion { region_id: *region_id });
        }
        VoteTopic::AllianceProposal { faction_id } => {
            if let Some(faction) = state.factions.iter().find(|f| f.id == *faction_id) {
                if faction.relationship_to_guild > 0.0 {
                    out.push(WorldDelta::SetDiplomaticStance {
                        faction_id: *faction_id,
                        stance: DiplomaticStance::Friendly,
                    });
                    out.push(WorldDelta::AdjustRelationship {
                        faction_id: *faction_id,
                        delta: 15.0,
                    });
                }
            }
        }
    }
}
*/

// ---------------------------------------------------------------------------
// Vote proposals
// ---------------------------------------------------------------------------

/*
fn auto_propose_votes(
    state: &WorldState,
    _council_ids: &[u32],
    out: &mut Vec<WorldDelta>,
) {
    let tick = state.tick;

    // Helper: check if a pending vote with matching topic already exists
    let has_pending = |topic_desc: &str| -> bool {
        state.council_votes.iter().any(|v| !v.resolved && v.topic.description() == topic_desc)
    };

    // War exhaustion -> propose SuePeace
    let factions_at_war: Vec<u32> = state.factions.iter()
        .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
        .map(|f| f.id)
        .collect();

    for fid in factions_at_war {
        // Check average guild adventurer exhaustion
        let guild_npcs: Vec<_> = state.entities.iter()
            .filter(|e| e.alive && e.npc.is_some()
                && e.team == crate::world_sim::state::WorldTeam::Friendly)
            .collect();

        if guild_npcs.is_empty() { continue; }

        // Approximate average exhaustion from entity HP ratio
        let avg_hp_ratio: f32 = guild_npcs.iter()
            .map(|e| e.hp / e.max_hp)
            .sum::<f32>() / guild_npcs.len() as f32;

        // Low average HP ratio suggests war weariness
        if avg_hp_ratio < 0.5 {
            let topic_desc = format!("Sue for peace with faction {}", fid);
            if !has_pending(&topic_desc) {
                out.push(WorldDelta::ProposeVote {
                    topic: VoteTopic::SuePeace { faction_id: fid },
                    deadline_tick: tick + VOTE_DEADLINE_TICKS,
                });
            }
        }
    }

    // Low-loyalty adventurers -> propose exile
    for entity in &state.entities {
        if !entity.alive { continue; }
        if let Some(ref npc) = entity.npc {
            // NPC loyalty check (needs loyalty field on NpcData)
            // if npc.loyalty < 20.0 && entity.level >= 1 {
            //     let topic_desc = format!("Exile adventurer {}", entity.id);
            //     if !has_pending(&topic_desc) {
            //         out.push(WorldDelta::ProposeVote {
            //             topic: VoteTopic::ExileAdventurer { adventurer_id: entity.id },
            //             deadline_tick: tick + VOTE_DEADLINE_TICKS,
            //         });
            //     }
            // }
        }
    }
}
*/

// ---------------------------------------------------------------------------
// Voting logic
// ---------------------------------------------------------------------------

/*
/// Determine whether an adventurer votes for a given topic.
/// Uses personality traits and bonds — fully deterministic (no RNG needed).
fn should_vote_for(
    adv_id: u32,
    topic: &VoteTopic,
    state: &WorldState,
) -> bool {
    // Look up adventurer entity
    let entity = match state.entities.iter().find(|e| e.id == adv_id) {
        Some(e) => e,
        None => return false,
    };

    // NPC personality traits (need extended NpcData fields)
    // Defaults: loyalty=50, morale=50, stress=25, fatigue=25, injury=0
    let loyalty = 50.0_f32;  // NEEDS STATE: npc.loyalty
    let morale = 50.0_f32;   // NEEDS STATE: npc.morale
    let stress = 25.0_f32;   // NEEDS STATE: npc.stress
    let fatigue = entity.hp / entity.max_hp * 100.0; // Approximate from HP
    let injury = if entity.hp < entity.max_hp * 0.5 { 50.0 } else { 0.0 };

    match topic {
        VoteTopic::DeclareWar { .. } => {
            loyalty > 50.0 && stress < 40.0 && morale > 40.0
        }
        VoteTopic::SuePeace { .. } => {
            fatigue > 30.0 || injury > 20.0 || stress > 50.0 || morale < 30.0
        }
        VoteTopic::ExileAdventurer { adventurer_id } => {
            // Check bond strength
            let bond = state.adventurer_bonds
                .get(&(adv_id, *adventurer_id))
                .or_else(|| state.adventurer_bonds.get(&(*adventurer_id, adv_id)))
                .copied()
                .unwrap_or(0.0);

            if bond > 30.0 {
                false // Ally — protect them
            } else if bond < 5.0 {
                loyalty > 40.0
            } else {
                loyalty > 60.0
            }
        }
        VoteTopic::ChangePolicy { .. } => {
            morale > 30.0 && stress < 60.0
        }
        VoteTopic::MajorExpedition { .. } => {
            fatigue < 40.0 && morale > 50.0 && injury < 20.0
        }
        VoteTopic::AllianceProposal { .. } => {
            loyalty > 30.0 && stress < 50.0
        }
    }
}
*/
