//! General party interception — every tick.
//!
//! Any guild party traveling on the map can be intercepted by an opposing
//! faction's party within range. This replaces hardcoded champion-specific
//! interception with a general proximity-based battle trigger.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Distance threshold (tiles) at which parties automatically engage.
const INTERCEPTION_RANGE: f32 = 5.0;

pub fn tick_interception(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // Only check every 10 ticks for performance (1 second cadence)
    if state.tick % 10 != 0 {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Collect guild parties (traveling/on-mission, members have no faction_id)
    let guild_parties: Vec<(u32, (f32, f32))> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission
            ) && p.member_ids.iter().any(|mid| {
                state
                    .adventurers
                    .iter()
                    .any(|a| a.id == *mid && a.faction_id.is_none())
            })
        })
        .map(|p| (p.id, p.position))
        .collect();

    if guild_parties.is_empty() {
        return;
    }

    // Collect hostile faction parties (traveling parties whose members belong
    // to factions at war with or hostile to the guild)
    let hostile_faction_ids: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| {
            f.id != guild_faction_id
                && (f.at_war_with.contains(&guild_faction_id)
                    || matches!(
                        f.diplomatic_stance,
                        DiplomaticStance::AtWar | DiplomaticStance::Hostile
                    ))
        })
        .map(|f| f.id)
        .collect();

    if hostile_faction_ids.is_empty() {
        return;
    }

    let hostile_parties: Vec<(u32, (f32, f32))> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission
            ) && p.member_ids.iter().any(|mid| {
                state
                    .adventurers
                    .iter()
                    .any(|a| {
                        a.id == *mid
                            && a.faction_id
                                .map(|fid| hostile_faction_ids.contains(&fid))
                                .unwrap_or(false)
                    })
            })
        })
        .map(|p| (p.id, p.position))
        .collect();

    if hostile_parties.is_empty() {
        return;
    }

    // Check proximity between guild parties and hostile parties
    let mut interceptions: Vec<(u32, u32)> = Vec::new();
    for &(guild_pid, guild_pos) in &guild_parties {
        for &(hostile_pid, hostile_pos) in &hostile_parties {
            let dx = guild_pos.0 - hostile_pos.0;
            let dy = guild_pos.1 - hostile_pos.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < INTERCEPTION_RANGE {
                interceptions.push((guild_pid, hostile_pid));
            }
        }
    }

    // Deduplicate: each party should only be in one interception per tick
    let mut engaged_guild: Vec<u32> = Vec::new();
    let mut engaged_hostile: Vec<u32> = Vec::new();

    for &(guild_pid, hostile_pid) in &interceptions {
        if engaged_guild.contains(&guild_pid) || engaged_hostile.contains(&hostile_pid) {
            continue;
        }
        engaged_guild.push(guild_pid);
        engaged_hostile.push(hostile_pid);

        // Set both parties to fighting
        if let Some(gp) = state.parties.iter_mut().find(|p| p.id == guild_pid) {
            gp.status = PartyStatus::Fighting;
        }
        if let Some(hp) = state.parties.iter_mut().find(|p| p.id == hostile_pid) {
            hp.status = PartyStatus::Fighting;
        }

        // Mark all members as fighting
        let all_member_ids: Vec<u32> = state
            .parties
            .iter()
            .filter(|p| p.id == guild_pid || p.id == hostile_pid)
            .flat_map(|p| p.member_ids.clone())
            .collect();
        for mid in &all_member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mid) {
                if adv.status != AdventurerStatus::Dead {
                    adv.status = AdventurerStatus::Fighting;
                }
            }
        }

        events.push(WorldEvent::CampaignMilestone {
            description: format!(
                "Guild party {} intercepted hostile party {} on the road!",
                guild_pid, hostile_pid
            ),
        });
    }
}
