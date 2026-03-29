//! Invariant verification — every step.
//!
//! Checks campaign state consistency. Returns a list of violation descriptions.

use crate::state::*;

pub fn verify_invariants(state: &CampaignState) -> Vec<String> {
    let mut violations = Vec::new();

    // Guild
    if state.guild.gold < 0.0 {
        violations.push(format!("Negative gold: {}", state.guild.gold));
    }
    if state.guild.supplies < 0.0 {
        violations.push(format!("Negative supplies: {}", state.guild.supplies));
    }
    if state.guild.reputation < 0.0 || state.guild.reputation > 100.0 {
        violations.push(format!(
            "Reputation out of range: {}",
            state.guild.reputation
        ));
    }

    // Adventurers
    let mut adv_ids = std::collections::HashSet::new();
    for adv in &state.adventurers {
        if !adv_ids.insert(adv.id) {
            violations.push(format!("Duplicate adventurer ID: {}", adv.id));
        }
        for (name, val) in [
            ("stress", adv.stress),
            ("fatigue", adv.fatigue),
            ("injury", adv.injury),
            ("loyalty", adv.loyalty),
            ("morale", adv.morale),
            ("resolve", adv.resolve),
        ] {
            if val < 0.0 || val > 100.0 {
                violations.push(format!(
                    "Adventurer {} {} out of range: {}",
                    adv.id, name, val
                ));
            }
        }
    }

    // Parties
    let mut party_ids = std::collections::HashSet::new();
    for party in &state.parties {
        if !party_ids.insert(party.id) {
            violations.push(format!("Duplicate party ID: {}", party.id));
        }
        if party.supply_level < 0.0 {
            violations.push(format!(
                "Party {} negative supply: {}",
                party.id, party.supply_level
            ));
        }
        // Verify member IDs exist
        for &mid in &party.member_ids {
            if !state.adventurers.iter().any(|a| a.id == mid) {
                violations.push(format!(
                    "Party {} references nonexistent adventurer {}",
                    party.id, mid
                ));
            }
        }
    }

    // Quests
    for quest in &state.active_quests {
        if let Some(pid) = quest.dispatched_party_id {
            if !state.parties.iter().any(|p| p.id == pid) {
                violations.push(format!(
                    "Quest {} references nonexistent party {}",
                    quest.id, pid
                ));
            }
        }
    }

    // Battles
    for battle in &state.active_battles {
        if !state.active_quests.iter().any(|q| q.id == battle.quest_id) {
            violations.push(format!(
                "Battle {} references nonexistent quest {}",
                battle.id, battle.quest_id
            ));
        }
    }

    // Regions
    for region in &state.overworld.regions {
        if region.unrest < 0.0 || region.unrest > 100.0 {
            violations.push(format!(
                "Region {} unrest out of range: {}",
                region.id, region.unrest
            ));
        }
        if region.control < 0.0 || region.control > 100.0 {
            violations.push(format!(
                "Region {} control out of range: {}",
                region.id, region.control
            ));
        }
    }

    // Diplomacy matrix
    let n = state.factions.len();
    if state.diplomacy.relations.len() != n {
        violations.push(format!(
            "Diplomacy matrix rows {} != faction count {}",
            state.diplomacy.relations.len(),
            n
        ));
    }

    violations
}
