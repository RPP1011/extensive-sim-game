//! Animal companion system — every 300 ticks.
//!
//! Adventurers can acquire animal companions that provide combat, travel, or
//! utility bonuses. Each adventurer may have at most one companion. Bond level
//! grows when the owner is active and decays when idle. High bond (>70) doubles
//! the species bonus.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, CampaignState, Companion, CompanionSpecies, QuestResult, QuestType,
    lcg_f32, lcg_next,
};

/// Names for randomly generated companions.
const COMPANION_NAMES: [&str; 24] = [
    "Shadow", "Fang", "Storm", "Ember", "Frost", "Thorn",
    "Ash", "Blaze", "Dusk", "Gale", "Ivy", "Luna",
    "Onyx", "Pike", "Rune", "Sage", "Talon", "Vale",
    "Whisper", "Zephyr", "Copper", "Flint", "Moss", "Wren",
];

/// Main tick function. Called every 300 ticks.
///
/// 1. Bond growth: +1.0 when owner is active (fighting, on mission, traveling)
/// 2. Bond decay: -0.5 when owner is idle
/// 3. High bond (>70): companion bonus doubles (handled by consumers via `effective_multiplier`)
/// 4. Bond milestones at 25, 50, 70
pub fn tick_companions(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 10 != 0 {
        return;
    }

    // Collect owner statuses to avoid borrow conflicts.
    let owner_statuses: Vec<(u32, AdventurerStatus)> = state
        .adventurers
        .iter()
        .map(|a| (a.id, a.status))
        .collect();

    for companion in &mut state.companions {
        let status = owner_statuses
            .iter()
            .find(|(id, _)| *id == companion.owner_id)
            .map(|(_, s)| *s);

        let old_bond = companion.bond_level;

        match status {
            Some(
                AdventurerStatus::Fighting
                | AdventurerStatus::OnMission
                | AdventurerStatus::Traveling,
            ) => {
                companion.bond_level = (companion.bond_level + 1.0).min(100.0);
            }
            Some(AdventurerStatus::Idle | AdventurerStatus::Assigned) => {
                companion.bond_level = (companion.bond_level - 0.5).max(0.0);
            }
            // Injured/Dead owners — no bond change (handled separately).
            _ => {}
        }

        // Bond milestone events at 25, 50, 70.
        for &milestone in &[25, 50, 70] {
            let m = milestone as f32;
            if old_bond < m && companion.bond_level >= m {
                events.push(WorldEvent::CompanionBondMilestone {
                    adventurer_id: companion.owner_id,
                    name: companion.name.clone(),
                    level: milestone,
                });
            }
        }
    }
}

/// Called when an adventurer dies in battle. 20% chance the companion also dies.
/// If the companion survives, it becomes a stray (owner cleared) and can be
/// reassigned via `reassign_strays`.
pub fn on_adventurer_died(
    state: &mut CampaignState,
    dead_id: u32,
    events: &mut Vec<WorldEvent>,
) {
    // Find companion owned by the dead adventurer.
    let comp_idx = state
        .companions
        .iter()
        .position(|c| c.owner_id == dead_id);
    let Some(idx) = comp_idx else { return };

    let death_roll = lcg_f32(&mut state.rng);
    if death_roll < 0.20 {
        // Companion dies.
        let lost = state.companions.remove(idx);
        events.push(WorldEvent::CompanionLost {
            name: lost.name,
            reason: "Fell alongside their owner in battle".into(),
        });
        // Owner gets -15 morale + grief (owner is dead, so apply to bonded allies).
        // The grief for other adventurers is already handled by bonds::on_adventurer_died.
    } else {
        // Companion becomes a stray — set owner_id to 0 (invalid, means unowned).
        let companion = &mut state.companions[idx];
        companion.owner_id = 0;
        companion.bond_level = (companion.bond_level * 0.5).max(0.0);
        // Grief: reduce morale of bonded adventurers who knew the companion.
        // (Owner is dead, so no morale penalty to apply to them.)
    }
}

/// Try to assign stray companions (owner_id == 0) to adventurers without companions.
/// Called within `tick_companions`.
pub fn reassign_strays(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    // Collect IDs of adventurers who already have a companion.
    let owned: Vec<u32> = state
        .companions
        .iter()
        .filter(|c| c.owner_id != 0)
        .map(|c| c.owner_id)
        .collect();

    // Find eligible adventurers (alive, no companion).
    let eligible: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && !owned.contains(&a.id))
        .map(|a| a.id)
        .collect();

    let mut eligible_idx = 0;
    for companion in &mut state.companions {
        if companion.owner_id != 0 {
            continue;
        }
        if eligible_idx >= eligible.len() {
            break;
        }
        companion.owner_id = eligible[eligible_idx];
        companion.bond_level = 5.0; // Start with low bond for new owner.
        eligible_idx += 1;

        events.push(WorldEvent::CompanionAcquired {
            adventurer_id: companion.owner_id,
            species: companion.species,
            name: companion.name.clone(),
        });
    }
}

/// Called after a quest completes successfully.
/// Exploration quests: 10% chance to acquire a companion.
/// Combat quests in wilderness (distance > 15): 5% chance.
/// Only adventurers without a companion are eligible.
pub fn on_quest_completed(
    state: &mut CampaignState,
    party_member_ids: &[u32],
    quest_type: QuestType,
    quest_distance: f32,
    result: QuestResult,
    events: &mut Vec<WorldEvent>,
) {
    if result != QuestResult::Victory {
        return;
    }

    let acquisition_chance = match quest_type {
        QuestType::Exploration => 0.10,
        QuestType::Combat if quest_distance > 15.0 => 0.05,
        _ => return,
    };

    // Collect which adventurers already have companions.
    let has_companion: Vec<u32> = state
        .companions
        .iter()
        .map(|c| c.owner_id)
        .collect();

    // Check each party member for acquisition.
    for &member_id in party_member_ids {
        if has_companion.contains(&member_id) {
            continue;
        }
        // Dead adventurers can't acquire companions.
        let is_alive = state
            .adventurers
            .iter()
            .any(|a| a.id == member_id && a.status != AdventurerStatus::Dead);
        if !is_alive {
            continue;
        }

        let roll = lcg_f32(&mut state.rng);
        if roll < acquisition_chance {
            // Pick a random species.
            let species_idx =
                (lcg_next(&mut state.rng) as usize) % CompanionSpecies::ALL.len();
            let species = CompanionSpecies::ALL[species_idx];

            // Pick a random name.
            let name_idx = (lcg_next(&mut state.rng) as usize) % COMPANION_NAMES.len();
            let name = format!("{} the {}", COMPANION_NAMES[name_idx], species.name());

            let id = state.next_companion_id;
            state.next_companion_id += 1;

            state.companions.push(Companion {
                id,
                name: name.clone(),
                species,
                owner_id: member_id,
                bond_level: 10.0,
                acquired_tick: state.tick,
            });

            events.push(WorldEvent::CompanionAcquired {
                adventurer_id: member_id,
                species,
                name,
            });

            // Apply initial morale boost from Cat companions.
            if species == CompanionSpecies::Cat {
                if let Some(adv) = state
                    .adventurers
                    .iter_mut()
                    .find(|a| a.id == member_id)
                {
                    adv.morale = (adv.morale + species.morale_bonus()).min(100.0);
                }
            }

            // Only one companion per quest completion event per adventurer.
            break;
        }
    }
}

/// Returns the effective bonus multiplier for a companion.
/// High bond (>70) doubles the base bonus.
pub fn effective_multiplier(bond_level: f32) -> f32 {
    if bond_level > 70.0 { 2.0 } else { 1.0 }
}

/// Combat power bonus for an adventurer from their companion.
/// Returns additive multiplier (e.g. 0.10 for +10%).
pub fn companion_combat_bonus(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .companions
        .iter()
        .find(|c| c.owner_id == adventurer_id)
        .map(|c| c.species.combat_power_bonus() * effective_multiplier(c.bond_level))
        .unwrap_or(0.0)
}

/// Travel speed bonus for an adventurer from their companion.
/// Returns additive multiplier (e.g. 0.30 for +30%).
pub fn companion_travel_bonus(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .companions
        .iter()
        .find(|c| c.owner_id == adventurer_id)
        .map(|c| c.species.travel_speed_bonus() * effective_multiplier(c.bond_level))
        .unwrap_or(0.0)
}

/// Scouting range bonus for an adventurer from their companion.
/// Returns additive multiplier (e.g. 0.20 for +20%).
pub fn companion_scouting_bonus(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .companions
        .iter()
        .find(|c| c.owner_id == adventurer_id)
        .map(|c| c.species.scouting_bonus() * effective_multiplier(c.bond_level))
        .unwrap_or(0.0)
}

/// Morale bonus for an adventurer from their companion.
pub fn companion_morale_bonus(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .companions
        .iter()
        .find(|c| c.owner_id == adventurer_id)
        .map(|c| c.species.morale_bonus() * effective_multiplier(c.bond_level))
        .unwrap_or(0.0)
}

/// Intimidation penalty applied to enemies when this adventurer's companion is present.
pub fn companion_intimidation(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .companions
        .iter()
        .find(|c| c.owner_id == adventurer_id)
        .map(|c| c.species.intimidation() * effective_multiplier(c.bond_level))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::{
        Adventurer, AdventurerStats, CampaignState, Equipment,
    };

    /// Create a minimal test adventurer.
    fn test_adventurer(id: u32) -> Adventurer {
        Adventurer {
            id,
            name: format!("Test Adventurer {}", id),
            archetype: "ranger".into(),
            level: 3,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 100.0,
                attack: 15.0,
                defense: 10.0,
                speed: 12.0,
                ability_power: 8.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 70.0,
            stress: 10.0,
            fatigue: 5.0,
            injury: 0.0,
            resolve: 60.0,
            morale: 80.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),

            backstory: None,

            deeds: Vec::new(),

            hobbies: Vec::new(),

            disease_status: DiseaseStatus::Healthy,

            mood_state: MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
            gold: 0.0,
            home_location_id: None,
            economic_intent: Default::default(),
            ticks_since_income: 0,
        }
    }

    #[test]
    fn companion_bond_grows_when_active() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        let mut adv = test_adventurer(1);
        adv.status = AdventurerStatus::Fighting;
        state.adventurers.push(adv);
        let adv_id = 1;
        state.companions.push(Companion {
            id: 1,
            name: "Fang the Wolf".into(),
            species: CompanionSpecies::Wolf,
            owner_id: adv_id,
            bond_level: 20.0,
            acquired_tick: 0,
        });

        state.tick = 300;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_companions(&mut state, &mut deltas, &mut events);

        assert_eq!(state.companions[0].bond_level, 21.0);
    }

    #[test]
    fn companion_bond_decays_when_idle() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        let adv = test_adventurer(1);
        state.adventurers.push(adv);
        let adv_id = 1;
        state.companions.push(Companion {
            id: 1,
            name: "Luna the Cat".into(),
            species: CompanionSpecies::Cat,
            owner_id: adv_id,
            bond_level: 30.0,
            acquired_tick: 0,
        });

        state.tick = 300;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_companions(&mut state, &mut deltas, &mut events);

        assert_eq!(state.companions[0].bond_level, 29.5);
    }

    #[test]
    fn high_bond_doubles_bonus() {
        assert_eq!(effective_multiplier(50.0), 1.0);
        assert_eq!(effective_multiplier(70.1), 2.0);
        assert_eq!(effective_multiplier(100.0), 2.0);
    }

    #[test]
    fn wolf_combat_bonus() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;
        state.adventurers.push(test_adventurer(1));
        let adv_id = 1;

        // No companion -> no bonus
        assert_eq!(companion_combat_bonus(&state, adv_id), 0.0);

        // Wolf with low bond -> 10%
        state.companions.push(Companion {
            id: 1,
            name: "Shadow".into(),
            species: CompanionSpecies::Wolf,
            owner_id: adv_id,
            bond_level: 50.0,
            acquired_tick: 0,
        });
        assert!((companion_combat_bonus(&state, adv_id) - 0.10).abs() < 0.001);

        // High bond -> 20%
        state.companions[0].bond_level = 80.0;
        assert!((companion_combat_bonus(&state, adv_id) - 0.20).abs() < 0.001);
    }

    #[test]
    fn bond_milestone_events() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        let mut adv = test_adventurer(1);
        adv.status = AdventurerStatus::Fighting;
        state.adventurers.push(adv);
        let adv_id = 1;
        state.companions.push(Companion {
            id: 1,
            name: "Storm the Hawk".into(),
            species: CompanionSpecies::Hawk,
            owner_id: adv_id,
            bond_level: 24.5,
            acquired_tick: 0,
        });

        state.tick = 300;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_companions(&mut state, &mut deltas, &mut events);

        // Bond goes from 24.5 -> 25.5, crossing the 25 milestone.
        assert_eq!(state.companions[0].bond_level, 25.5);
        assert_eq!(events.len(), 1);
        match &events[0] {
            WorldEvent::CompanionBondMilestone { level, .. } => assert_eq!(*level, 25),
            other => panic!("Expected CompanionBondMilestone, got {:?}", other),
        }
    }

    #[test]
    fn max_one_companion_per_adventurer() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        state.adventurers.push(test_adventurer(1));
        let adv_id = 1;
        state.companions.push(Companion {
            id: 1,
            name: "Existing".into(),
            species: CompanionSpecies::Wolf,
            owner_id: adv_id,
            bond_level: 50.0,
            acquired_tick: 0,
        });

        let mut events = Vec::new();
        // Try to acquire — should fail since adventurer already has one.
        on_quest_completed(
            &mut state,
            &[adv_id],
            QuestType::Exploration,
            20.0,
            QuestResult::Victory,
            &mut events,
        );

        // Still only one companion.
        assert_eq!(state.companions.len(), 1);
    }
}
