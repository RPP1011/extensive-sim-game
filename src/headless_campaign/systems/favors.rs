//! Favor economy system — factions request favors from the guild, completing
//! them banks goodwill that can be called in later for military aid, intel,
//! trade deals, or diplomatic support.
//!
//! Fires every 500 ticks. Friendly factions (relation > 30) generate favor
//! requests. Expired requests penalise relations. Accumulated favor balance
//! unlocks call-in options via `CampaignAction::CallInFavor`.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the favor system ticks (in campaign ticks).
const FAVOR_INTERVAL: u64 = 500;

/// Relation penalty when a favor request expires without being completed.
const EXPIRE_RELATION_PENALTY: f32 = 5.0;

/// Favor request lifetime in ticks before it expires.
const FAVOR_LIFETIME_TICKS: u64 = 3000;

/// Minimum faction relation to generate favor requests.
const MIN_RELATION_FOR_FAVORS: f32 = 30.0;

/// Base probability of a friendly faction generating a favor request each interval.
const FAVOR_GENERATION_CHANCE: f32 = 0.4;

// --- Favor call-in costs ---
pub const MILITARY_AID_COST: f32 = 30.0;
pub const INTEL_COST: f32 = 15.0;
pub const TRADE_DEAL_COST: f32 = 20.0;
pub const DIPLOMATIC_SUPPORT_COST: f32 = 25.0;

/// Duration of trade deal buff (in ticks).
pub const TRADE_DEAL_DURATION_TICKS: u64 = 5000;

/// Relation boost from diplomatic support call-in.
pub const DIPLOMATIC_SUPPORT_RELATION_BOOST: f32 = 15.0;

/// Tick the favor system: generate new requests, expire old ones.
pub fn tick_favors(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FAVOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire old favor requests ---
    let mut expired_faction_ids = Vec::new();
    let current_tick = state.tick;
    state.favor_requests.retain(|req| {
        if current_tick >= req.expires_tick {
            expired_faction_ids.push(req.faction_id);
            false
        } else {
            true
        }
    });

    // Penalise relations for expired requests
    for fid in &expired_faction_ids {
        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == *fid) {
            faction.relationship_to_guild =
                (faction.relationship_to_guild - EXPIRE_RELATION_PENALTY).max(-100.0);
        }
    }

    // --- Update active trade deal buffs (decay) ---
    for favor in &mut state.faction_favors {
        if favor.trade_deal_expires_tick > 0 && current_tick >= favor.trade_deal_expires_tick {
            favor.trade_deal_expires_tick = 0;
        }
    }

    // --- Generate new favor requests from friendly factions ---
    let faction_snapshots: Vec<(usize, f32)> = state
        .factions
        .iter()
        .filter(|f| f.relationship_to_guild > MIN_RELATION_FOR_FAVORS)
        .map(|f| (f.id, f.relationship_to_guild))
        .collect();

    for (fid, _relation) in faction_snapshots {
        // Already has an outstanding request from this faction? Skip.
        if state.favor_requests.iter().any(|r| r.faction_id == fid) {
            continue;
        }

        let roll = lcg_f32(&mut state.rng);
        if roll > FAVOR_GENERATION_CHANCE {
            continue;
        }

        // Pick a quest type deterministically
        let quest_roll = lcg_next(&mut state.rng) % 3;
        let (quest_type, description, reward_favor) = match quest_roll {
            0 => (
                "supply_aid".to_string(),
                "Deliver supplies to a struggling settlement".to_string(),
                10.0_f32,
            ),
            1 => (
                "military_escort".to_string(),
                "Escort a diplomatic envoy through hostile territory".to_string(),
                15.0_f32,
            ),
            _ => (
                "diplomatic_task".to_string(),
                "Mediate a dispute between regional leaders".to_string(),
                12.0_f32,
            ),
        };

        let req_id = state.next_quest_id;
        state.next_quest_id += 1;

        let request = FavorRequest {
            id: req_id,
            faction_id: fid,
            description: description.clone(),
            reward_favor,
            quest_type,
            expires_tick: current_tick + FAVOR_LIFETIME_TICKS,
        };

        events.push(WorldEvent::FavorRequested {
            request_id: req_id,
            faction_id: fid,
            description,
        });

        state.favor_requests.push(request);
    }
}

/// Complete a favor request: remove it, credit the faction's favor balance.
pub fn complete_favor(
    state: &mut CampaignState,
    request_id: u32,
    events: &mut Vec<WorldEvent>,
) -> bool {
    let idx = state.favor_requests.iter().position(|r| r.id == request_id);
    let req = match idx {
        Some(i) => state.favor_requests.remove(i),
        None => return false,
    };

    // Ensure FactionFavor entry exists
    let favor = get_or_create_faction_favor(&mut state.faction_favors, req.faction_id);
    favor.favor_balance += req.reward_favor;
    favor.favors_given += 1;

    // Improve relations as a side effect
    if let Some(faction) = state.factions.iter_mut().find(|f| f.id == req.faction_id) {
        faction.relationship_to_guild =
            (faction.relationship_to_guild + 5.0).min(100.0);
    }

    events.push(WorldEvent::FavorCompleted {
        request_id: req.id,
        faction_id: req.faction_id,
        reward_favor: req.reward_favor,
    });

    true
}

/// Call in a favor from a faction. Returns an error message on failure.
pub fn call_in_favor(
    state: &mut CampaignState,
    faction_id: usize,
    favor_type: &str,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    let cost = match favor_type {
        "military_aid" => MILITARY_AID_COST,
        "intel" => INTEL_COST,
        "trade_deal" => TRADE_DEAL_COST,
        "diplomatic_support" => DIPLOMATIC_SUPPORT_COST,
        _ => return Err(format!("Unknown favor type: {}", favor_type)),
    };

    // Check balance
    let favor = state
        .faction_favors
        .iter()
        .find(|f| f.faction_id == faction_id);
    let balance = favor.map(|f| f.favor_balance).unwrap_or(0.0);
    if balance < cost {
        return Err(format!(
            "Not enough favor with faction {} (have {:.0}, need {:.0})",
            faction_id, balance, cost
        ));
    }

    // Deduct
    let favor = get_or_create_faction_favor(&mut state.faction_favors, faction_id);
    favor.favor_balance -= cost;
    favor.favors_called += 1;

    // Apply the effect
    let description = match favor_type {
        "military_aid" => {
            // Faction sends troops: boost predicted outcome of active battles
            // and boost control of the weakest guild region
            for battle in &mut state.active_battles {
                if battle.status == BattleStatus::Active {
                    battle.predicted_outcome = (battle.predicted_outcome + 0.3).min(1.0);
                }
            }
            if let Some(region) = state
                .overworld
                .regions
                .iter_mut()
                .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
                .min_by(|a, b| {
                    a.control
                        .partial_cmp(&b.control)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                region.control = (region.control + 15.0).min(100.0);
            }
            "Military aid received — troops dispatched to bolster defenses"
        }
        "intel" => {
            // Reveal faction's enemies and boost visibility of all regions
            for region in &mut state.overworld.regions {
                region.visibility = (region.visibility + 0.3).min(1.0);
            }
            "Intel received — enemy movements and plans revealed"
        }
        "trade_deal" => {
            // Temporary +50% trade income
            let favor =
                get_or_create_faction_favor(&mut state.faction_favors, faction_id);
            favor.trade_deal_expires_tick = state.tick + TRADE_DEAL_DURATION_TICKS;
            "Trade deal activated — +50% trade income from this faction"
        }
        "diplomatic_support" => {
            // Faction vouches for guild to another faction (+15 relation)
            // Pick the faction with the worst relation that isn't the vouching faction
            let target = state
                .factions
                .iter()
                .filter(|f| f.id != faction_id)
                .min_by(|a, b| {
                    a.relationship_to_guild
                        .partial_cmp(&b.relationship_to_guild)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|f| f.id);

            if let Some(target_id) = target {
                if let Some(target_faction) =
                    state.factions.iter_mut().find(|f| f.id == target_id)
                {
                    target_faction.relationship_to_guild = (target_faction
                        .relationship_to_guild
                        + DIPLOMATIC_SUPPORT_RELATION_BOOST)
                        .min(100.0);
                }
            }
            "Diplomatic support — faction vouched for the guild"
        }
        _ => unreachable!(),
    };

    events.push(WorldEvent::FavorCalledIn {
        faction_id,
        favor_type: favor_type.to_string(),
        cost,
        description: description.to_string(),
    });

    Ok(description.to_string())
}

/// Count active trade deal buffs and return the income multiplier.
pub fn trade_deal_income_multiplier(state: &CampaignState) -> f32 {
    let active_deals = state
        .faction_favors
        .iter()
        .filter(|f| f.trade_deal_expires_tick > 0 && state.tick < f.trade_deal_expires_tick)
        .count();

    if active_deals > 0 {
        1.0 + 0.5 * active_deals as f32
    } else {
        1.0
    }
}

/// Ensure a `FactionFavor` entry exists for the given faction, returning a mutable reference.
fn get_or_create_faction_favor(
    favors: &mut Vec<FactionFavor>,
    faction_id: usize,
) -> &mut FactionFavor {
    if !favors.iter().any(|f| f.faction_id == faction_id) {
        favors.push(FactionFavor {
            faction_id,
            favor_balance: 0.0,
            favors_given: 0,
            favors_called: 0,
            trade_deal_expires_tick: 0,
        });
    }
    favors
        .iter_mut()
        .find(|f| f.faction_id == faction_id)
        .unwrap()
}
