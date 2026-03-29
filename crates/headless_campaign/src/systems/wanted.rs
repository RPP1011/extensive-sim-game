//! Wanted poster system.
//!
//! Factions issue bounties on guild adventurers who've wronged them.
//! Wanted adventurers face increased ambush chance in hostile territory
//! and bounty hunter encounters. Posters expire after 5000 ticks or
//! can be resolved through payment, combat, or diplomacy.
//!
//! Ticks every 300 ticks.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Cadence: wanted system ticks every 300 ticks.
const TICK_CADENCE: u64 = 10;

/// Posters expire after this many ticks.
const POSTER_EXPIRY_TICKS: u64 = 167;

/// Faction relation threshold above which posters are removed diplomatically.
const DIPLOMACY_REMOVAL_THRESHOLD: f32 = 30.0;

/// Ambush chance increase for wanted adventurers in hostile territory.
const WANTED_AMBUSH_BONUS: f32 = 0.20;

/// Chance per battle that a poster is issued when guild kills faction units.
const BATTLE_POSTER_CHANCE: f32 = 0.15;

/// Chance per espionage event that a poster is issued.
const ESPIONAGE_POSTER_CHANCE: f32 = 0.80;

/// Chance per completed quest against a faction that a poster is issued.
const QUEST_AGAINST_POSTER_CHANCE: f32 = 0.10;

/// Run the wanted poster system for one tick.
///
/// Every 300 ticks:
/// - Check for new poster triggers (battles, quests against factions)
/// - Dispatch hunters for active posters
/// - Resolve expired posters and diplomatic removals
/// - Generate hunter encounters for traveling wanted adventurers
pub fn tick_wanted(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    // --- Phase 1: Check for new poster triggers ---
    check_battle_triggers(state, events);
    check_quest_triggers(state, events);

    // --- Phase 2: Diplomatic removal ---
    remove_by_diplomacy(state, events);

    // --- Phase 3: Expire old posters ---
    expire_posters(state, events);

    // --- Phase 4: Dispatch hunters ---
    dispatch_hunters(state, events);

    // --- Phase 5: Hunter encounters ---
    hunter_encounters(state, events);
}

/// Check recently completed battles for poster triggers.
///
/// When a guild party wins a battle against a faction's forces, there's
/// a 15% chance the faction issues a wanted poster for a party member.
fn check_battle_triggers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Look at recently completed quests with victories that have a faction.
    // We use completed_quests and check for ones completed this cadence window.
    let window_start = state.tick.saturating_sub(TICK_CADENCE);
    let window_start_ms = window_start * CAMPAIGN_TURN_SECS as u64 * 1000;

    let recent_victories: Vec<(u32, Option<usize>)> = state
        .completed_quests
        .iter()
        .filter(|cq| {
            cq.result == QuestResult::Victory && cq.completed_at_ms > window_start_ms
        })
        .map(|cq| {
            // Find the original quest's faction from the active quest archive.
            // We look at the quest id and find the faction from the request.
            let faction_id = state
                .active_quests
                .iter()
                .find(|aq| aq.id == cq.id)
                .and_then(|aq| aq.request.source_faction_id);
            (cq.party_id, faction_id)
        })
        .collect();

    for (party_id, faction_id) in recent_victories {
        let faction_id = match faction_id {
            Some(fid) => fid,
            None => continue,
        };

        // Check if faction is hostile (relationship < 0 means they'd issue a poster)
        let faction_hostile = state
            .factions
            .iter()
            .any(|f| f.id == faction_id && f.relationship_to_guild < 0.0);
        if !faction_hostile {
            continue;
        }

        let roll = lcg_f32(&mut state.rng);
        if roll >= BATTLE_POSTER_CHANCE {
            continue;
        }

        // Pick an adventurer from the party
        let party_members: Vec<u32> = state
            .parties
            .iter()
            .find(|p| p.id == party_id)
            .map(|p| p.member_ids.clone())
            .unwrap_or_default();

        if party_members.is_empty() {
            continue;
        }

        let member_idx = lcg_next(&mut state.rng) as usize % party_members.len();
        let adventurer_id = party_members[member_idx];

        // Don't issue duplicate posters for same adventurer+faction
        let already_posted = state
            .wanted_posters
            .iter()
            .any(|wp| wp.adventurer_id == adventurer_id && wp.faction_id == faction_id);
        if already_posted {
            continue;
        }

        let adv_level = state
            .adventurers
            .iter()
            .find(|a| a.id == adventurer_id)
            .map(|a| a.level)
            .unwrap_or(1);

        let bounty = compute_bounty(adv_level, 1.0);
        issue_poster(state, events, adventurer_id, faction_id, bounty, "Killed faction forces in battle".into());
    }
}

/// Check recently completed quests that were against a faction.
fn check_quest_triggers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let window_start = state.tick.saturating_sub(TICK_CADENCE);
    let window_start_ms = window_start * CAMPAIGN_TURN_SECS as u64 * 1000;

    // Find recently completed quests that targeted a faction
    let recent_quests: Vec<(u32, usize)> = state
        .completed_quests
        .iter()
        .filter(|cq| cq.result == QuestResult::Victory && cq.completed_at_ms > window_start_ms)
        .filter_map(|cq| {
            // Find the original quest request's faction
            state
                .active_quests
                .iter()
                .find(|aq| aq.id == cq.id)
                .and_then(|aq| aq.request.source_faction_id)
                .map(|fid| (cq.party_id, fid))
        })
        .collect();

    for (party_id, faction_id) in recent_quests {
        // Only trigger for hostile factions
        let faction_hostile = state
            .factions
            .iter()
            .any(|f| f.id == faction_id && f.relationship_to_guild < -20.0);
        if !faction_hostile {
            continue;
        }

        let roll = lcg_f32(&mut state.rng);
        if roll >= QUEST_AGAINST_POSTER_CHANCE {
            continue;
        }

        let party_members: Vec<u32> = state
            .parties
            .iter()
            .find(|p| p.id == party_id)
            .map(|p| p.member_ids.clone())
            .unwrap_or_default();

        if party_members.is_empty() {
            continue;
        }

        let member_idx = lcg_next(&mut state.rng) as usize % party_members.len();
        let adventurer_id = party_members[member_idx];

        let already_posted = state
            .wanted_posters
            .iter()
            .any(|wp| wp.adventurer_id == adventurer_id && wp.faction_id == faction_id);
        if already_posted {
            continue;
        }

        let adv_level = state
            .adventurers
            .iter()
            .find(|a| a.id == adventurer_id)
            .map(|a| a.level)
            .unwrap_or(1);

        let bounty = compute_bounty(adv_level, 0.7);
        issue_poster(state, events, adventurer_id, faction_id, bounty, "Completed quest against faction".into());
    }
}

/// Remove posters when faction relation improves above threshold.
fn remove_by_diplomacy(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let factions_above_threshold: Vec<usize> = state
        .factions
        .iter()
        .filter(|f| f.relationship_to_guild > DIPLOMACY_REMOVAL_THRESHOLD)
        .map(|f| f.id)
        .collect();

    let mut removed = Vec::new();
    state.wanted_posters.retain(|wp| {
        if factions_above_threshold.contains(&wp.faction_id) {
            removed.push((wp.id, wp.adventurer_id, wp.faction_id, wp.bounty_amount));
            false
        } else {
            true
        }
    });

    for (poster_id, adventurer_id, faction_id, bounty_amount) in removed {
        events.push(WorldEvent::BountyPaidOff {
            poster_id,
            adventurer_id,
            faction_id,
            amount: bounty_amount,
        });
    }
}

/// Expire posters older than POSTER_EXPIRY_TICKS.
fn expire_posters(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    state
        .wanted_posters
        .retain(|wp| state.tick.saturating_sub(wp.posted_tick) < POSTER_EXPIRY_TICKS);
}

/// Dispatch bounty hunters for posters that haven't dispatched yet.
///
/// Hunter strength = bounty / 10.
fn dispatch_hunters(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect poster IDs that need dispatching
    let to_dispatch: Vec<(u32, u32, usize, f32)> = state
        .wanted_posters
        .iter()
        .filter(|wp| !wp.hunters_dispatched)
        .map(|wp| (wp.id, wp.adventurer_id, wp.faction_id, wp.bounty_amount))
        .collect();

    for (poster_id, adventurer_id, faction_id, bounty_amount) in to_dispatch {
        // Only dispatch if faction has enough military strength
        let has_strength = state
            .factions
            .iter()
            .any(|f| f.id == faction_id && f.military_strength > bounty_amount / 10.0);
        if !has_strength {
            continue;
        }

        // Mark as dispatched
        if let Some(wp) = state.wanted_posters.iter_mut().find(|wp| wp.id == poster_id) {
            wp.hunters_dispatched = true;
        }

        // Reduce faction military strength by hunter party cost
        let hunter_strength = bounty_amount / 10.0;
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
            faction.military_strength = (faction.military_strength - hunter_strength).max(0.0);
        }

        events.push(WorldEvent::BountyHunterDispatched {
            poster_id,
            adventurer_id,
            faction_id,
            hunter_strength,
        });
    }
}

/// Generate hunter encounters for wanted adventurers traveling through hostile territory.
///
/// Wanted adventurers in traveling parties that cross hostile faction territory
/// face combat encounters with bounty hunters.
fn hunter_encounters(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find wanted adventurers who are currently traveling
    let active_posters: Vec<(u32, usize, f32)> = state
        .wanted_posters
        .iter()
        .filter(|wp| wp.hunters_dispatched)
        .map(|wp| (wp.adventurer_id, wp.faction_id, wp.bounty_amount))
        .collect();

    if active_posters.is_empty() {
        return;
    }

    for (adventurer_id, faction_id, bounty_amount) in active_posters {
        // Check if adventurer is in a traveling party
        let party = state.parties.iter().find(|p| {
            p.member_ids.contains(&adventurer_id)
                && matches!(p.status, PartyStatus::Traveling)
        });

        let party_id = match party {
            Some(p) => p.id,
            None => continue,
        };

        let party_pos = match state.parties.iter().find(|p| p.id == party_id) {
            Some(p) => p.position,
            None => continue,
        };

        // Check if party is in territory controlled by the hostile faction
        let in_hostile_territory = in_faction_territory(party_pos, faction_id, state);

        if !in_hostile_territory {
            // Even outside hostile territory, there's a reduced chance
            let roll = lcg_f32(&mut state.rng);
            if roll >= WANTED_AMBUSH_BONUS * 0.5 {
                continue;
            }
        } else {
            // In hostile territory: higher chance
            let roll = lcg_f32(&mut state.rng);
            if roll >= WANTED_AMBUSH_BONUS {
                continue;
            }
        }

        // Generate a hunter encounter as a battle
        let battle_id = state.next_battle_id;
        state.next_battle_id += 1;

        let hunter_strength = bounty_amount / 10.0;

        // Create a battle for the hunter encounter
        let party_health = state
            .parties
            .iter()
            .find(|p| p.id == party_id)
            .map(|p| {
                let total_hp: f32 = p
                    .member_ids
                    .iter()
                    .filter_map(|mid| state.adventurers.iter().find(|a| a.id == *mid))
                    .map(|a| (100.0 - a.injury) / 100.0)
                    .sum();
                let count = p.member_ids.len().max(1) as f32;
                total_hp / count
            })
            .unwrap_or(1.0);

        // Predict outcome based on relative strengths
        let party_strength: f32 = state
            .parties
            .iter()
            .find(|p| p.id == party_id)
            .map(|p| {
                p.member_ids
                    .iter()
                    .filter_map(|mid| state.adventurers.iter().find(|a| a.id == *mid))
                    .map(|a| a.stats.attack + a.stats.defense)
                    .sum::<f32>()
            })
            .unwrap_or(10.0);

        let predicted = (party_strength - hunter_strength) / (party_strength + hunter_strength + 1.0);

        // We don't create a quest_id for bounty hunter battles — use 0 as sentinel.
        state.active_battles.push(BattleState {
            id: battle_id,
            quest_id: 0, // bounty hunter battle, not quest-related
            party_id,
            location: party_pos,
            party_health_ratio: party_health,
            enemy_health_ratio: 1.0,
            enemy_strength: hunter_strength,
            elapsed_ticks: 0,
            predicted_outcome: predicted,
            status: BattleStatus::Active,
            runner_sent: false,
            mercenary_hired: false,
            rescue_called: false,
        });

        // Set party to fighting
        if let Some(p) = state.parties.iter_mut().find(|p| p.id == party_id) {
            p.status = PartyStatus::Fighting;
        }

        events.push(WorldEvent::BattleStarted {
            battle_id,
            quest_id: 0,
            party_health: party_health,
            enemy_strength: hunter_strength,
        });
    }
}

/// Compute bounty amount based on adventurer level and a damage multiplier.
fn compute_bounty(level: u32, damage_mult: f32) -> f32 {
    let base = 20.0 + level as f32 * 10.0;
    (base * damage_mult).max(10.0)
}

/// Issue a wanted poster and emit event.
fn issue_poster(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    adventurer_id: u32,
    faction_id: usize,
    bounty_amount: f32,
    reason: String,
) {
    let poster_id = state.next_poster_id;
    state.next_poster_id += 1;

    state.wanted_posters.push(WantedPoster {
        id: poster_id,
        adventurer_id,
        faction_id,
        bounty_amount,
        reason: reason.clone(),
        posted_tick: state.tick,
        hunters_dispatched: false,
    });

    events.push(WorldEvent::WantedPosterIssued {
        poster_id,
        adventurer_id,
        faction_id,
        bounty_amount,
        reason,
    });
}

/// Apply the PayOffBounty action. Costs 2x the bounty amount.
pub fn apply_pay_off_bounty(
    state: &mut CampaignState,
    poster_id: u32,
    events: &mut Vec<WorldEvent>,
) -> super::super::actions::ActionResult {
    let poster = state.wanted_posters.iter().find(|wp| wp.id == poster_id);
    let (adventurer_id, faction_id, bounty_amount) = match poster {
        Some(wp) => (wp.adventurer_id, wp.faction_id, wp.bounty_amount),
        None => {
            return super::super::actions::ActionResult::InvalidAction(format!(
                "Wanted poster {} not found",
                poster_id
            ));
        }
    };

    let cost = bounty_amount * 2.0;
    if state.guild.gold < cost {
        return super::super::actions::ActionResult::Failed(format!(
            "Not enough gold (need {:.0}, have {:.0})",
            cost, state.guild.gold
        ));
    }

    state.guild.gold -= cost;
    state.wanted_posters.retain(|wp| wp.id != poster_id);

    // Slightly improve faction relation
    if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
        faction.relationship_to_guild = (faction.relationship_to_guild + 5.0).min(100.0);
    }

    events.push(WorldEvent::BountyPaidOff {
        poster_id,
        adventurer_id,
        faction_id,
        amount: cost,
    });

    super::super::actions::ActionResult::Success(format!(
        "Paid off bounty on adventurer {} for {:.0} gold",
        adventurer_id, cost
    ))
}

/// Handle bounty hunter defeat: increase bounty but earn reputation.
pub fn on_hunter_defeated(
    state: &mut CampaignState,
    adventurer_id: u32,
    events: &mut Vec<WorldEvent>,
) {
    // Find posters for this adventurer where hunters were dispatched
    let poster_ids: Vec<u32> = state
        .wanted_posters
        .iter()
        .filter(|wp| wp.adventurer_id == adventurer_id && wp.hunters_dispatched)
        .map(|wp| wp.id)
        .collect();

    for poster_id in poster_ids {
        if let Some(wp) = state.wanted_posters.iter_mut().find(|wp| wp.id == poster_id) {
            let old_bounty = wp.bounty_amount;
            // Increase bounty by 50%
            wp.bounty_amount *= 1.5;
            // Reset hunters so faction can dispatch again
            wp.hunters_dispatched = false;

            events.push(WorldEvent::BountyHunterDefeated {
                poster_id,
                adventurer_id,
                new_bounty: wp.bounty_amount,
            });

            // Earn reputation for defeating hunters
            state.guild.reputation = (state.guild.reputation + 2.0).min(100.0);

            let _ = old_bounty; // used for scaling if needed later
        }
    }
}

/// Check if a position is in territory controlled by a given faction.
///
/// Uses locations with `faction_owner` as territory markers. A party is
/// considered "in hostile territory" if it's within range of a faction-owned location.
fn in_faction_territory(pos: (f32, f32), faction_id: usize, state: &CampaignState) -> bool {
    const TERRITORY_RADIUS_SQ: f32 = 400.0; // 20 tiles
    state.overworld.locations.iter().any(|loc| {
        loc.faction_owner == Some(faction_id) && {
            let dx = pos.0 - loc.position.0;
            let dy = pos.1 - loc.position.1;
            dx * dx + dy * dy <= TERRITORY_RADIUS_SQ
        }
    })
}
