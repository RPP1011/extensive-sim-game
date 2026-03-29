//! Party chemistry system — every 200 ticks.
//!
//! Adventurers who repeatedly team together develop chemistry that boosts
//! quest success, travel speed, and loot. Creates roster optimization tension
//! between specialization (deep chemistry in a few pairs) and flexibility
//! (spreading chemistry across many pairs).

use std::collections::HashMap;

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    AdventurerStatus, CampaignState, PartyStatus,
};
use crate::systems::bonds::bond_key;
use crate::systems::rivalries::has_rivalry;

/// Chemistry key: always (min, max) for symmetric lookup.
pub fn chemistry_key(a: u32, b: u32) -> (u32, u32) {
    bond_key(a, b)
}

/// Look up chemistry score between two adventurers.
pub fn chemistry_score(chemistry: &HashMap<(u32, u32), f32>, a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    *chemistry.get(&chemistry_key(a, b)).unwrap_or(&0.0)
}

/// Average chemistry among a set of party member IDs.
/// Returns 0.0 for solo adventurers.
pub fn average_party_chemistry(chemistry: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> f32 {
    if member_ids.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u32;
    for (i, &a) in member_ids.iter().enumerate() {
        for &b in &member_ids[i + 1..] {
            total += chemistry_score(chemistry, a, b);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f32 }
}

/// Quest success rate modifier based on mean party chemistry.
/// Returns a multiplier offset: e.g. 0.05 means +5%.
pub fn quest_success_modifier(chemistry: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> f32 {
    let mean = average_party_chemistry(chemistry, member_ids);
    if mean > 0.8 {
        0.15
    } else if mean > 0.6 {
        0.10
    } else if mean > 0.3 {
        0.05
    } else if mean < 0.1 {
        -0.05
    } else {
        0.0
    }
}

/// Travel time multiplier based on mean party chemistry.
/// Returns a multiplier: e.g. 0.9 means 10% faster.
pub fn travel_time_multiplier(chemistry: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> f32 {
    let mean = average_party_chemistry(chemistry, member_ids);
    if mean > 0.6 {
        0.9 // -10% travel time
    } else {
        1.0
    }
}

/// Loot bonus multiplier based on mean party chemistry.
/// Returns a multiplier: e.g. 1.1 means +10% loot.
pub fn loot_multiplier(chemistry: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> f32 {
    let mean = average_party_chemistry(chemistry, member_ids);
    if mean > 0.8 {
        1.1 // +10% loot
    } else {
        1.0
    }
}

/// Whether this party qualifies as a "legendary team" (mean chemistry > 0.8).
pub fn is_legendary_team(chemistry: &HashMap<(u32, u32), f32>, member_ids: &[u32]) -> bool {
    member_ids.len() >= 2 && average_party_chemistry(chemistry, member_ids) > 0.8
}

/// Main tick function. Called every 200 ticks.
///
/// 1. Grow chemistry for adventurers in the same active party (+0.02)
///    - Shared archetype bonus (+0.01)
/// 2. Decay chemistry for pairs not in the same party (-0.005)
///    - Rivalry penalty (-0.1)
/// 3. Track best-ever chemistry pairs
/// 4. Emit events for chemistry milestones
pub fn tick_party_chemistry(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 {
        return;
    }

    // Collect active party member lists.
    let active_parties: Vec<(u32, Vec<u32>)> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::OnMission | PartyStatus::Fighting | PartyStatus::Traveling
            )
        })
        .map(|p| (p.id, p.member_ids.clone()))
        .collect();

    // Build a set of all pairs currently in the same party.
    let mut paired: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for (_party_id, members) in &active_parties {
        for (i, &a) in members.iter().enumerate() {
            for &b in &members[i + 1..] {
                paired.insert(chemistry_key(a, b));
            }
        }
    }

    // Look up archetypes for shared-archetype bonus.
    let archetypes: HashMap<u32, String> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| (a.id, a.archetype.clone()))
        .collect();

    // --- 1. Grow chemistry for paired adventurers ---
    for &key in &paired {
        let (a, b) = key;
        let entry = state.party_chemistry.entry(key).or_insert(0.0);
        let mut growth = 0.02;

        // Shared archetype bonus
        if let (Some(arch_a), Some(arch_b)) = (archetypes.get(&a), archetypes.get(&b)) {
            if arch_a == arch_b {
                growth += 0.01;
            }
        }

        let old = *entry;
        *entry = (*entry + growth).min(1.0);
        let new = *entry;

        // Check for crossing the 0.5 threshold (ChemistryForged event)
        if old < 0.5 && new >= 0.5 {
            events.push(WorldEvent::ChemistryForged {
                adv_id_1: a,
                adv_id_2: b,
                score: new,
            });
        }

        // Track best-ever
        let best = state.best_chemistry.entry(key).or_insert(0.0);
        if new > *best {
            *best = new;
        }
    }

    // --- 2. Decay chemistry for non-paired adventurers ---
    // Collect keys to update (avoid borrow issues).
    let decay_keys: Vec<(u32, u32)> = state
        .party_chemistry
        .keys()
        .filter(|k| !paired.contains(k))
        .copied()
        .collect();

    for key in decay_keys {
        let (a, b) = key;
        let rivalry_active = has_rivalry(state, a, b);
        let decay = if rivalry_active { 0.1 } else { 0.005 };

        if let Some(entry) = state.party_chemistry.get_mut(&key) {
            let old = *entry;
            *entry = (*entry - decay).max(0.0);

            // Emit ChemistryBroken if it drops below 0.1 from above
            if old >= 0.1 && *entry < 0.1 && old >= 0.3 {
                let reason = if rivalry_active {
                    "rivalry".to_string()
                } else {
                    "separation".to_string()
                };
                events.push(WorldEvent::ChemistryBroken {
                    adv_id_1: a,
                    adv_id_2: b,
                    reason,
                });
            }
        }
    }

    // Remove zeroed-out entries to keep map clean.
    state.party_chemistry.retain(|_, v| *v > 0.0);

    // --- 3. Check for legendary teams ---
    for (party_id, members) in &active_parties {
        let mean = average_party_chemistry(&state.party_chemistry, members);
        if mean > 0.8 {
            events.push(WorldEvent::LegendaryTeamFormed {
                party_id: *party_id,
                mean_chemistry: mean,
            });
        }
    }
}

/// Call when a quest is completed. Boosts chemistry for all party member pairs.
pub fn on_quest_completed(state: &mut CampaignState, party_member_ids: &[u32]) {
    for (i, &a) in party_member_ids.iter().enumerate() {
        for &b in &party_member_ids[i + 1..] {
            let key = chemistry_key(a, b);
            let entry = state.party_chemistry.entry(key).or_insert(0.0);
            *entry = (*entry + 0.05).min(1.0);

            // Track best-ever
            let best = state.best_chemistry.entry(key).or_insert(0.0);
            if *entry > *best {
                *best = *entry;
            }
        }
    }
}

/// Call when a battle is survived together. Boosts chemistry more than quests.
pub fn on_battle_survived(state: &mut CampaignState, party_member_ids: &[u32]) {
    for (i, &a) in party_member_ids.iter().enumerate() {
        for &b in &party_member_ids[i + 1..] {
            let key = chemistry_key(a, b);
            let entry = state.party_chemistry.entry(key).or_insert(0.0);
            *entry = (*entry + 0.08).min(1.0);

            // Track best-ever
            let best = state.best_chemistry.entry(key).or_insert(0.0);
            if *entry > *best {
                *best = *entry;
            }
        }
    }
}

/// Remove all chemistry involving a dead adventurer.
pub fn on_adventurer_died(state: &mut CampaignState, dead_id: u32) {
    state
        .party_chemistry
        .retain(|&(a, b), _| a != dead_id && b != dead_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chemistry_key_is_symmetric() {
        assert_eq!(chemistry_key(3, 7), chemistry_key(7, 3));
        assert_eq!(chemistry_key(3, 7), (3, 7));
    }

    #[test]
    fn chemistry_score_returns_zero_for_unknown() {
        let chem = HashMap::new();
        assert_eq!(chemistry_score(&chem, 1, 2), 0.0);
    }

    #[test]
    fn chemistry_score_self_is_zero() {
        let chem = HashMap::new();
        assert_eq!(chemistry_score(&chem, 5, 5), 0.0);
    }

    #[test]
    fn average_party_chemistry_works() {
        let mut chem = HashMap::new();
        chem.insert(chemistry_key(1, 2), 0.4);
        chem.insert(chemistry_key(1, 3), 0.6);
        chem.insert(chemistry_key(2, 3), 0.8);
        let avg = average_party_chemistry(&chem, &[1, 2, 3]);
        assert!((avg - 0.6).abs() < 0.01);
    }

    #[test]
    fn solo_adventurer_no_chemistry() {
        let chem = HashMap::new();
        assert_eq!(average_party_chemistry(&chem, &[1]), 0.0);
        assert_eq!(quest_success_modifier(&chem, &[1]), 0.0);
        assert_eq!(travel_time_multiplier(&chem, &[1]), 1.0);
        assert_eq!(loot_multiplier(&chem, &[1]), 1.0);
        assert!(!is_legendary_team(&chem, &[1]));
    }

    #[test]
    fn quest_success_modifier_tiers() {
        let mut chem = HashMap::new();

        // Strangers penalty
        chem.insert(chemistry_key(1, 2), 0.05);
        assert_eq!(quest_success_modifier(&chem, &[1, 2]), -0.05);

        // Low chemistry — no bonus
        chem.insert(chemistry_key(1, 2), 0.2);
        assert_eq!(quest_success_modifier(&chem, &[1, 2]), 0.0);

        // Medium chemistry
        chem.insert(chemistry_key(1, 2), 0.4);
        assert_eq!(quest_success_modifier(&chem, &[1, 2]), 0.05);

        // High chemistry
        chem.insert(chemistry_key(1, 2), 0.7);
        assert_eq!(quest_success_modifier(&chem, &[1, 2]), 0.10);

        // Legendary chemistry
        chem.insert(chemistry_key(1, 2), 0.9);
        assert_eq!(quest_success_modifier(&chem, &[1, 2]), 0.15);
    }

    #[test]
    fn travel_time_multiplier_threshold() {
        let mut chem = HashMap::new();
        chem.insert(chemistry_key(1, 2), 0.5);
        assert_eq!(travel_time_multiplier(&chem, &[1, 2]), 1.0);

        chem.insert(chemistry_key(1, 2), 0.7);
        assert_eq!(travel_time_multiplier(&chem, &[1, 2]), 0.9);
    }

    #[test]
    fn loot_multiplier_threshold() {
        let mut chem = HashMap::new();
        chem.insert(chemistry_key(1, 2), 0.7);
        assert_eq!(loot_multiplier(&chem, &[1, 2]), 1.0);

        chem.insert(chemistry_key(1, 2), 0.9);
        assert_eq!(loot_multiplier(&chem, &[1, 2]), 1.1);
    }

    #[test]
    fn legendary_team_check() {
        let mut chem = HashMap::new();
        chem.insert(chemistry_key(1, 2), 0.9);
        assert!(is_legendary_team(&chem, &[1, 2]));
        assert!(!is_legendary_team(&chem, &[1])); // solo can't be legendary
    }

    #[test]
    fn on_quest_completed_boosts_chemistry() {
        let mut state = CampaignState::default_test_campaign(42);
        on_quest_completed(&mut state, &[1, 2, 3]);

        assert!((chemistry_score(&state.party_chemistry, 1, 2) - 0.05).abs() < 0.001);
        assert!((chemistry_score(&state.party_chemistry, 1, 3) - 0.05).abs() < 0.001);
        assert!((chemistry_score(&state.party_chemistry, 2, 3) - 0.05).abs() < 0.001);
    }

    #[test]
    fn on_battle_survived_boosts_more_than_quest() {
        let mut state = CampaignState::default_test_campaign(42);
        on_battle_survived(&mut state, &[1, 2]);

        let score = chemistry_score(&state.party_chemistry, 1, 2);
        assert!(score > 0.05, "Battle survived should give more than quest completion");
        assert!((score - 0.08).abs() < 0.001);
    }

    #[test]
    fn on_adventurer_died_removes_chemistry() {
        let mut state = CampaignState::default_test_campaign(42);
        state.party_chemistry.insert(chemistry_key(1, 2), 0.5);
        state.party_chemistry.insert(chemistry_key(1, 3), 0.3);
        state.party_chemistry.insert(chemistry_key(2, 3), 0.7);

        on_adventurer_died(&mut state, 1);

        assert_eq!(chemistry_score(&state.party_chemistry, 1, 2), 0.0);
        assert_eq!(chemistry_score(&state.party_chemistry, 1, 3), 0.0);
        assert!((chemistry_score(&state.party_chemistry, 2, 3) - 0.7).abs() < 0.001);
    }

    #[test]
    fn best_chemistry_tracks_peak() {
        let mut state = CampaignState::default_test_campaign(42);

        // Build chemistry
        on_battle_survived(&mut state, &[1, 2]);
        let peak = chemistry_score(&state.party_chemistry, 1, 2);
        assert!(peak > 0.0);

        // Verify best tracks it
        let best = *state.best_chemistry.get(&chemistry_key(1, 2)).unwrap_or(&0.0);
        assert!((best - peak).abs() < 0.001);
    }
}
