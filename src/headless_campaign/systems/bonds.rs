//! Adventurer bond/relationship system — every 50 ticks.
//!
//! Adventurers who fight together develop bonds that affect morale,
//! combat effectiveness, and quest outcomes. Bonds decay slowly when apart.

use std::collections::HashMap;

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState, PartyStatus};

/// Canonical bond key: always (min, max) for symmetric lookup.
pub fn bond_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Look up bond strength between two adventurers.
pub fn bond_strength(bonds: &HashMap<(u32, u32), f32>, a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    *bonds.get(&bond_key(a, b)).unwrap_or(&0.0)
}

/// Average bond strength among a set of party member IDs.
/// Returns 0.0 for solo adventurers.
pub fn average_party_bond(bonds: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> f32 {
    if member_ids.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u32;
    for (i, &a) in member_ids.iter().enumerate() {
        for &b in &member_ids[i + 1..] {
            total += bond_strength(bonds, a, b);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f32 }
}

/// Morale bonus for an adventurer based on bonds with current party members.
/// Bond > 30: +5% morale (additive).
pub fn morale_bonus(bonds: &HashMap<(u32, u32), f32>, adv_id: u32, party_member_ids: &[u32]) -> f32 {
    let max_bond = party_member_ids
        .iter()
        .filter(|&&id| id != adv_id)
        .map(|&id| bond_strength(bonds, adv_id, id))
        .fold(0.0f32, f32::max);
    if max_bond > 30.0 { 5.0 } else { 0.0 }
}

/// Combat power bonus multiplier from bonds.
/// Bond > 60 with any party member: 1.10 (10% damage).
/// Bond > 80: 1.15 (15% defense, stacks → use max tier).
pub fn combat_power_multiplier(bonds: &HashMap<(u32, u32), f32>, adv_id: u32, party_member_ids: &[u32]) -> f32 {
    let max_bond = party_member_ids
        .iter()
        .filter(|&&id| id != adv_id)
        .map(|&id| bond_strength(bonds, adv_id, id))
        .fold(0.0f32, f32::max);
    if max_bond > 80.0 {
        1.15
    } else if max_bond > 60.0 {
        1.10
    } else {
        1.0
    }
}

/// Whether this adventurer has "Battle Brothers" status (bond > 80 with
/// any party member). Resists desertion.
pub fn has_battle_brothers(bonds: &HashMap<(u32, u32), f32>, adv_id: u32, party_member_ids: &[u32]) -> bool {
    party_member_ids
        .iter()
        .filter(|&&id| id != adv_id)
        .any(|&id| bond_strength(bonds, adv_id, id) > 80.0)
}

/// Main tick function. Called every 50 ticks.
///
/// 1. Decay all bonds by 0.1
/// 2. For each active party, increase bonds between member pairs by 0.5
/// 3. Quest-completion bonus (+2.0) applied via `on_quest_completed`
/// 4. Grief on death applied via `on_adventurer_died`
pub fn tick_bonds(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    _events: &mut Vec<WorldEvent>,
) {
    if state.tick % 1 != 0 {
        return;
    }

    // 1. Decay all bonds by 0.1 (slow drift toward 0).
    state.adventurer_bonds.values_mut().for_each(|v| {
        *v = (*v - 0.1).max(0.0);
    });
    // Remove zeroed-out entries to keep map clean.
    state.adventurer_bonds.retain(|_, v| *v > 0.0);

    // 2. Increase bonds for adventurers in the same active party.
    let active_parties: Vec<Vec<u32>> = state
        .parties
        .iter()
        .filter(|p| matches!(p.status, PartyStatus::OnMission | PartyStatus::Fighting | PartyStatus::Traveling))
        .map(|p| p.member_ids.clone())
        .collect();

    for members in &active_parties {
        for (i, &a) in members.iter().enumerate() {
            for &b in &members[i + 1..] {
                let key = bond_key(a, b);
                let entry = state.adventurer_bonds.entry(key).or_insert(0.0);
                *entry = (*entry + 0.5).min(100.0);
            }
        }
    }
}

/// Call when a quest is completed (victory). Boosts bonds for all party member pairs.
pub fn on_quest_completed(state: &mut CampaignState, party_member_ids: &[u32]) {
    for (i, &a) in party_member_ids.iter().enumerate() {
        for &b in &party_member_ids[i + 1..] {
            let key = bond_key(a, b);
            let entry = state.adventurer_bonds.entry(key).or_insert(0.0);
            *entry = (*entry + 2.0).min(100.0);
        }
    }
}

/// Call when an adventurer dies. Applies grief penalty to bonded adventurers.
/// Bond > 50: stress +20, morale -15.
pub fn on_adventurer_died(state: &mut CampaignState, dead_id: u32, events: &mut Vec<WorldEvent>) {
    // Collect bonded adventurers first to avoid borrow issues.
    let bonded: Vec<(u32, f32)> = state
        .adventurers
        .iter()
        .filter(|a| a.id != dead_id && a.status != AdventurerStatus::Dead)
        .filter_map(|a| {
            let strength = bond_strength(&state.adventurer_bonds, dead_id, a.id);
            if strength > 50.0 {
                Some((a.id, strength))
            } else {
                None
            }
        })
        .collect();

    for (adv_id, _strength) in &bonded {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adv_id) {
            adv.stress = (adv.stress + 20.0).min(100.0);
            adv.morale = (adv.morale - 15.0).max(0.0);
            events.push(WorldEvent::BondGrief {
                adventurer_id: *adv_id,
                dead_id,
                bond_strength: *_strength,
            });
        }
    }

    // Remove all bonds involving the dead adventurer.
    state
        .adventurer_bonds
        .retain(|&(a, b), _| a != dead_id && b != dead_id);
}

/// Bond-adjusted party power bonus (additive, in the same units as unlock_power_bonus).
/// Returns the average multiplier - 1.0 scaled to a bonus value.
/// e.g., if average multiplier is 1.10, returns 10% of average member power.
pub fn party_bond_power_bonus(
    bonds: &HashMap<(u32, u32), f32>,
    member_ids: &[u32],
    base_party_power: f32,
) -> f32 {
    if member_ids.len() < 2 {
        return 0.0;
    }
    let avg_mult: f32 = member_ids
        .iter()
        .map(|&id| combat_power_multiplier(bonds, id, member_ids))
        .sum::<f32>()
        / member_ids.len() as f32;
    // Convert multiplier to additive bonus on base power.
    (avg_mult - 1.0) * base_party_power
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_key_is_symmetric() {
        assert_eq!(bond_key(3, 7), bond_key(7, 3));
        assert_eq!(bond_key(3, 7), (3, 7));
    }

    #[test]
    fn bond_strength_returns_zero_for_unknown() {
        let bonds = HashMap::new();
        assert_eq!(bond_strength(&bonds, 1, 2), 0.0);
    }

    #[test]
    fn bond_strength_self_is_zero() {
        let bonds = HashMap::new();
        assert_eq!(bond_strength(&bonds, 5, 5), 0.0);
    }

    #[test]
    fn morale_bonus_threshold() {
        let mut bonds = HashMap::new();
        bonds.insert(bond_key(1, 2), 25.0);
        assert_eq!(morale_bonus(&bonds, 1, &[1, 2]), 0.0);

        bonds.insert(bond_key(1, 2), 35.0);
        assert_eq!(morale_bonus(&bonds, 1, &[1, 2]), 5.0);
    }

    #[test]
    fn combat_multiplier_tiers() {
        let mut bonds = HashMap::new();
        bonds.insert(bond_key(1, 2), 50.0);
        assert_eq!(combat_power_multiplier(&bonds, 1, &[1, 2]), 1.0);

        bonds.insert(bond_key(1, 2), 65.0);
        assert_eq!(combat_power_multiplier(&bonds, 1, &[1, 2]), 1.10);

        bonds.insert(bond_key(1, 2), 85.0);
        assert_eq!(combat_power_multiplier(&bonds, 1, &[1, 2]), 1.15);
    }

    #[test]
    fn battle_brothers_check() {
        let mut bonds = HashMap::new();
        bonds.insert(bond_key(1, 2), 85.0);
        assert!(has_battle_brothers(&bonds, 1, &[1, 2]));
        assert!(!has_battle_brothers(&bonds, 1, &[1, 3]));
    }

    #[test]
    fn average_party_bond_works() {
        let mut bonds = HashMap::new();
        bonds.insert(bond_key(1, 2), 40.0);
        bonds.insert(bond_key(1, 3), 60.0);
        bonds.insert(bond_key(2, 3), 80.0);
        let avg = average_party_bond(&bonds, &[1, 2, 3]);
        assert!((avg - 60.0).abs() < 0.01);
    }

    #[test]
    fn solo_adventurer_no_bonus() {
        let bonds = HashMap::new();
        assert_eq!(average_party_bond(&bonds, &[1]), 0.0);
        assert_eq!(morale_bonus(&bonds, 1, &[1]), 0.0);
        assert_eq!(combat_power_multiplier(&bonds, 1, &[1]), 1.0);
        assert_eq!(party_bond_power_bonus(&bonds, &[1], 100.0), 0.0);
    }
}
