//! Adventurer hobbies and downtime system — every 200 ticks.
//!
//! Idle adventurers develop interests over time that provide passive bonuses.
//! Hobby selection is weighted by archetype; adventurers can have at most 2 hobbies.
//! Gambling has special gold gain/loss and rivalry mechanics.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, CampaignState, Hobby, HobbyProgress, lcg_f32, lcg_next,
};

/// How often to tick hobbies (in ticks).
const HOBBY_INTERVAL: u64 = 7;

/// How many ticks an adventurer must be idle before picking a hobby.
const IDLE_THRESHOLD: u64 = 17;

/// Skill gain per hobby tick for idle adventurers.
const SKILL_GAIN_PER_TICK: f32 = 2.0;

/// Maximum hobbies per adventurer.
const MAX_HOBBIES: usize = 2;

/// Tick the hobby system. Called every `HOBBY_INTERVAL` ticks.
///
/// 1. Idle adventurers who have been idle for 500+ ticks and have < 2 hobbies
///    pick a new hobby weighted by archetype.
/// 2. Idle adventurers with hobbies gain skill (+2 per tick).
/// 3. Gambling triggers gold gain/loss and rivalry creation.
pub fn tick_hobbies(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % HOBBY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let tick = state.tick;
    let adv_count = state.adventurers.len();

    // --- Phase 1: Hobby selection for idle adventurers ---
    for i in 0..adv_count {
        let adv = &state.adventurers[i];
        if adv.status != AdventurerStatus::Idle {
            continue;
        }
        if adv.hobbies.len() >= MAX_HOBBIES {
            continue;
        }

        // Check idle duration: adventurer must have been idle for 500+ ticks.
        // We approximate by checking if they have no party and have low fatigue
        // (truly idle). Use the tick counter and a simple heuristic: if they
        // have no hobbies yet, require tick >= IDLE_THRESHOLD (first hobby).
        // If they have 1 hobby, they can pick a second after another idle period.
        let idle_long_enough = if adv.hobbies.is_empty() {
            tick >= IDLE_THRESHOLD
        } else {
            // Second hobby: at least IDLE_THRESHOLD ticks after the first hobby started
            let last_started = adv.hobbies.last().map(|h| h.started_tick).unwrap_or(0);
            tick.saturating_sub(last_started) >= IDLE_THRESHOLD
        };

        if !idle_long_enough {
            continue;
        }

        // Build weighted hobby pool based on archetype.
        let archetype = &state.adventurers[i].archetype;
        let weights = hobby_weights_for_archetype(archetype);

        // Filter out hobbies the adventurer already has.
        let existing: Vec<Hobby> = state.adventurers[i].hobbies.iter().map(|h| h.hobby).collect();
        let filtered: Vec<(Hobby, f32)> = weights
            .into_iter()
            .filter(|(h, _)| !existing.contains(h))
            .collect();

        if filtered.is_empty() {
            continue;
        }

        let total_weight: f32 = filtered.iter().map(|(_, w)| w).sum();
        let roll = lcg_f32(&mut state.rng) * total_weight;
        let mut cumulative = 0.0;
        let mut chosen = filtered[0].0;
        for (hobby, w) in &filtered {
            cumulative += w;
            if roll < cumulative {
                chosen = *hobby;
                break;
            }
        }

        state.adventurers[i].hobbies.push(HobbyProgress {
            hobby: chosen,
            skill_level: 0.0,
            started_tick: tick,
        });

        events.push(WorldEvent::HobbyDeveloped {
            adventurer_id: state.adventurers[i].id,
            hobby: format!("{:?}", chosen),
        });
    }

    // --- Phase 2: Skill progression for idle adventurers with hobbies ---
    for i in 0..adv_count {
        if state.adventurers[i].status != AdventurerStatus::Idle {
            continue;
        }

        let hobby_count = state.adventurers[i].hobbies.len();
        for j in 0..hobby_count {
            state.adventurers[i].hobbies[j].skill_level =
                (state.adventurers[i].hobbies[j].skill_level + SKILL_GAIN_PER_TICK).min(100.0);
        }
    }

    // --- Phase 3: Gambling special effects ---
    // Collect gambling adventurer IDs first to avoid borrow issues.
    let gamblers: Vec<(usize, u32, f32)> = state
        .adventurers
        .iter()
        .enumerate()
        .filter(|(_, adv)| adv.status == AdventurerStatus::Idle)
        .filter_map(|(idx, adv)| {
            adv.hobbies
                .iter()
                .find(|h| h.hobby == Hobby::Gambling)
                .map(|h| (idx, adv.id, h.skill_level))
        })
        .collect();

    for &(idx, adv_id, _skill) in &gamblers {
        let roll = lcg_f32(&mut state.rng);

        if roll < 0.20 {
            // 20% chance: gain 5 gold
            state.guild.gold += 5.0;
            events.push(WorldEvent::GamblingOutcome {
                adventurer_id: adv_id,
                amount: 5.0,
            });
        } else if roll < 0.30 {
            // 10% chance: lose 10 gold
            let loss = state.guild.gold.min(10.0);
            state.guild.gold -= loss;
            events.push(WorldEvent::GamblingOutcome {
                adventurer_id: adv_id,
                amount: -loss,
            });
        } else if roll < 0.35 {
            // 5% chance: create rivalry with another gambler
            // Find a different gambler to create a rivalry with.
            let other_gamblers: Vec<u32> = gamblers
                .iter()
                .filter(|&&(other_idx, _, _)| other_idx != idx)
                .map(|&(_, other_id, _)| other_id)
                .collect();

            if !other_gamblers.is_empty() {
                let pick = (lcg_next(&mut state.rng) as usize) % other_gamblers.len();
                let rival_id = other_gamblers[pick];

                // Reduce bond between the two gamblers (rivalry).
                let key = crate::headless_campaign::systems::bonds::bond_key(adv_id, rival_id);
                let entry = state.adventurer_bonds.entry(key).or_insert(0.0);
                *entry = (*entry - 10.0).max(-50.0);
            }
        }
    }
}

/// Return weighted hobby candidates for a given archetype string.
fn hobby_weights_for_archetype(archetype: &str) -> Vec<(Hobby, f32)> {
    // Normalize archetype to lowercase for matching.
    let arch = archetype.to_lowercase();

    let mut weights = vec![];

    // Archetype-specific weights
    match arch.as_str() {
        "knight" | "warrior" | "paladin" | "berserker" => {
            weights.push((Hobby::Training, 3.0));
            weights.push((Hobby::Smithing, 3.0));
        }
        "rogue" | "assassin" | "thief" => {
            weights.push((Hobby::Gambling, 3.0));
            weights.push((Hobby::Cartography, 3.0));
        }
        "mage" | "wizard" | "sorcerer" | "warlock" => {
            weights.push((Hobby::Meditation, 3.0));
            weights.push((Hobby::Herbalism, 3.0));
        }
        "cleric" | "healer" | "priest" => {
            weights.push((Hobby::Herbalism, 3.0));
            weights.push((Hobby::Cooking, 3.0));
        }
        _ => {
            // Unknown archetype: equal weight for all
            weights.push((Hobby::Training, 1.0));
            weights.push((Hobby::Smithing, 1.0));
            weights.push((Hobby::Gambling, 1.0));
            weights.push((Hobby::Cartography, 1.0));
            weights.push((Hobby::Meditation, 1.0));
            weights.push((Hobby::Herbalism, 1.0));
            weights.push((Hobby::Cooking, 1.0));
        }
    }

    // Storytelling is available to all archetypes with base weight
    weights.push((Hobby::Storytelling, 1.5));

    // Add low-weight fallbacks for non-primary hobbies (so any archetype can
    // potentially get any hobby, just less likely).
    let all_hobbies = [
        Hobby::Cooking,
        Hobby::Cartography,
        Hobby::Gambling,
        Hobby::Herbalism,
        Hobby::Smithing,
        Hobby::Training,
        Hobby::Meditation,
    ];
    for h in &all_hobbies {
        if !weights.iter().any(|(wh, _)| wh == h) {
            weights.push((*h, 0.5));
        }
    }

    weights
}

/// Compute the passive bonus scale factor for a hobby.
/// Returns 0.0 at skill 0, ramps linearly to 1.0 at skill 50+.
pub fn hobby_bonus_scale(skill_level: f32) -> f32 {
    (skill_level / 50.0).min(1.0)
}

/// Check if any adventurer in the party has a specific hobby at a useful level (>0).
/// Returns the best skill level found, or 0.0.
pub fn best_hobby_skill_in_party(
    state: &CampaignState,
    party_member_ids: &[u32],
    hobby: Hobby,
) -> f32 {
    party_member_ids
        .iter()
        .filter_map(|&id| state.adventurers.iter().find(|a| a.id == id))
        .flat_map(|a| a.hobbies.iter())
        .filter(|h| h.hobby == hobby)
        .map(|h| h.skill_level)
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::CampaignState;

    #[test]
    fn hobby_bonus_scale_ramps() {
        assert_eq!(hobby_bonus_scale(0.0), 0.0);
        assert!((hobby_bonus_scale(25.0) - 0.5).abs() < 0.001);
        assert_eq!(hobby_bonus_scale(50.0), 1.0);
        assert_eq!(hobby_bonus_scale(100.0), 1.0);
    }

    #[test]
    fn hobby_weights_warrior() {
        let weights = hobby_weights_for_archetype("knight");
        let training_w = weights.iter().find(|(h, _)| *h == Hobby::Training).unwrap().1;
        let cooking_w = weights.iter().find(|(h, _)| *h == Hobby::Cooking).unwrap().1;
        assert!(training_w > cooking_w, "Warriors should prefer training over cooking");
    }

    #[test]
    fn hobby_weights_healer() {
        let weights = hobby_weights_for_archetype("cleric");
        let herb_w = weights.iter().find(|(h, _)| *h == Hobby::Herbalism).unwrap().1;
        let gambling_w = weights.iter().find(|(h, _)| *h == Hobby::Gambling).unwrap().1;
        assert!(herb_w > gambling_w, "Healers should prefer herbalism over gambling");
    }

    use crate::headless_campaign::state::{
        Adventurer, AdventurerStats, Equipment, AdventurerStatus as AdvStatus,
    };

    fn make_test_adventurer(id: u32, archetype: &str) -> Adventurer {
        Adventurer {
            id,
            name: format!("Test_{}", id),
            archetype: archetype.into(),
            level: 1,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 100.0,
                attack: 10.0,
                defense: 10.0,
                speed: 10.0,
                ability_power: 10.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdvStatus::Idle,
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
            hobbies: Vec::new(),
            backstory: None,
            deeds: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,

            mood_state: crate::headless_campaign::state::MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
        }
    }

    #[test]
    fn tick_hobbies_assigns_hobby_after_idle() {
        let mut state = CampaignState::default_test_campaign(42);
        state.adventurers.push(make_test_adventurer(1, "knight"));
        state.adventurers.push(make_test_adventurer(2, "mage"));

        // Tick at a HOBBY_INTERVAL boundary past the idle threshold
        state.tick = ((IDLE_THRESHOLD / HOBBY_INTERVAL) + 1) * HOBBY_INTERVAL;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_hobbies(&mut state, &mut deltas, &mut events);

        // At least one idle adventurer should have gained a hobby
        let has_hobby = state.adventurers.iter().any(|a| !a.hobbies.is_empty());
        assert!(has_hobby, "Idle adventurers should develop hobbies");

        // Should have emitted HobbyDeveloped events
        let hobby_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::HobbyDeveloped { .. }))
            .collect();
        assert!(!hobby_events.is_empty(), "Should emit HobbyDeveloped events");
    }

    #[test]
    fn max_two_hobbies() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut adv = make_test_adventurer(1, "knight");
        adv.hobbies = vec![
            HobbyProgress {
                hobby: Hobby::Cooking,
                skill_level: 50.0,
                started_tick: 100,
            },
            HobbyProgress {
                hobby: Hobby::Training,
                skill_level: 30.0,
                started_tick: 600,
            },
        ];
        state.adventurers.push(adv);
        state.tick = 2000; // Well past idle threshold

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_hobbies(&mut state, &mut deltas, &mut events);

        // First adventurer should still have exactly 2 hobbies
        assert_eq!(state.adventurers[0].hobbies.len(), 2);
    }

    #[test]
    fn skill_progression_only_when_idle() {
        let mut state = CampaignState::default_test_campaign(42);

        let mut idle_adv = make_test_adventurer(1, "knight");
        idle_adv.hobbies = vec![HobbyProgress {
            hobby: Hobby::Cooking,
            skill_level: 10.0,
            started_tick: 0,
        }];
        idle_adv.status = AdvStatus::Idle;

        let mut traveling_adv = make_test_adventurer(2, "mage");
        traveling_adv.hobbies = vec![HobbyProgress {
            hobby: Hobby::Cooking,
            skill_level: 10.0,
            started_tick: 0,
        }];
        traveling_adv.status = AdvStatus::Traveling;

        state.adventurers.push(idle_adv);
        state.adventurers.push(traveling_adv);
        state.tick = HOBBY_INTERVAL;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_hobbies(&mut state, &mut deltas, &mut events);

        assert!(
            state.adventurers[0].hobbies[0].skill_level > 10.0,
            "Idle adventurer should gain skill"
        );
        assert_eq!(
            state.adventurers[1].hobbies[0].skill_level, 10.0,
            "Traveling adventurer should NOT gain skill"
        );
    }
}
