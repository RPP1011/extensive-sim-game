//! Adventurer mood/emotion system — every 200 ticks.
//!
//! Beyond morale, adventurers carry specific emotional states that affect
//! their behavior: combat effectiveness, morale drift, bond formation,
//! quest preferences, and decision-making.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, CampaignState, Mood, MoodCause, MoodState, lcg_f32,
};

/// Mood tick cadence (every 200 ticks = 20 seconds game time).
const MOOD_TICK_INTERVAL: u64 = 7;

/// Duration range for mood decay (ticks). Moods last 500-1000 ticks.
const MOOD_MIN_DURATION: u64 = 17;
const MOOD_MAX_DURATION: u64 = 33;

/// Grieving morale drain duration (ticks).
const GRIEF_DURATION: u64 = 17;

pub fn tick_moods(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % MOOD_TICK_INTERVAL != 0 {
        return;
    }

    let tick = state.tick;

    // --- Phase 1: Natural decay — moods expire after their duration ---
    let mut decay_events: Vec<(u32, Mood)> = Vec::new();
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        if adv.mood_state.mood != Mood::Neutral {
            if tick >= adv.mood_state.expires_at {
                let old = adv.mood_state.mood;
                adv.mood_state = MoodState::default();
                decay_events.push((adv.id, old));
            }
        }
    }
    for (id, old_mood) in decay_events {
        events.push(WorldEvent::MoodChanged {
            adventurer_id: id,
            old_mood,
            new_mood: Mood::Neutral,
            cause: MoodCause::NaturalDecay,
        });
    }

    // --- Phase 2: Apply ongoing mood effects ---
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        match adv.mood_state.mood {
            Mood::Grieving => {
                // -2 morale per mood tick for grief_duration ticks
                let elapsed = tick.saturating_sub(adv.mood_state.started_at);
                if elapsed <= GRIEF_DURATION {
                    adv.morale = (adv.morale - 2.0).max(0.0);
                }
            }
            Mood::Excited => {
                // +5 morale per mood tick
                adv.morale = (adv.morale + 5.0).min(100.0);
            }
            Mood::Melancholic => {
                // -1 morale per mood tick
                adv.morale = (adv.morale - 1.0).max(0.0);
            }
            _ => {}
        }
    }

    // --- Phase 3: Contagion — moods spread within parties ---
    // Collect party compositions and member moods.
    let party_info: Vec<(Vec<u32>, Vec<(u32, Mood)>)> = state
        .parties
        .iter()
        .map(|p| {
            let moods: Vec<(u32, Mood)> = p
                .member_ids
                .iter()
                .filter_map(|&id| {
                    state.adventurers.iter().find(|a| a.id == id).map(|a| (a.id, a.mood_state.mood))
                })
                .collect();
            (p.member_ids.clone(), moods)
        })
        .collect();

    for (member_ids, moods) in &party_info {
        // Check for Excited contagion (20% chance per neutral member)
        let has_excited = moods.iter().any(|(_, m)| *m == Mood::Excited);
        if has_excited {
            for &mid in member_ids {
                if let Some(adv) = state.adventurers.iter().find(|a| a.id == mid) {
                    if adv.mood_state.mood == Mood::Neutral {
                        let roll = lcg_f32(&mut state.rng);
                        if roll < 0.20 {
                            let duration = mood_duration(&mut state.rng);
                            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                                let old = adv.mood_state.mood;
                                adv.mood_state = MoodState {
                                    mood: Mood::Excited,
                                    started_at: tick,
                                    expires_at: tick + duration,
                                };
                                events.push(WorldEvent::MoodChanged {
                                    adventurer_id: mid,
                                    old_mood: old,
                                    new_mood: Mood::Excited,
                                    cause: MoodCause::Contagion,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Check for Fearful contagion (15% chance per neutral member)
        let has_fearful = moods.iter().any(|(_, m)| *m == Mood::Fearful);
        if has_fearful {
            for &mid in member_ids {
                if let Some(adv) = state.adventurers.iter().find(|a| a.id == mid) {
                    if adv.mood_state.mood == Mood::Neutral {
                        let roll = lcg_f32(&mut state.rng);
                        if roll < 0.15 {
                            let duration = mood_duration(&mut state.rng);
                            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                                let old = adv.mood_state.mood;
                                adv.mood_state = MoodState {
                                    mood: Mood::Fearful,
                                    started_at: tick,
                                    expires_at: tick + duration,
                                };
                                events.push(WorldEvent::MoodChanged {
                                    adventurer_id: mid,
                                    old_mood: old,
                                    new_mood: Mood::Fearful,
                                    cause: MoodCause::Contagion,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Trigger mood transitions based on a specific cause. Called from other systems
/// when relevant events occur (battle victory, defeat, ally death, etc.).
pub fn trigger_mood(
    state: &mut CampaignState,
    adventurer_id: u32,
    cause: MoodCause,
    events: &mut Vec<WorldEvent>,
) {
    let tick = state.tick;

    // Determine candidate mood and probability.
    let candidates: &[(Mood, f32)] = match cause {
        MoodCause::BattleVictory => &[
            (Mood::Inspired, 0.40),
            (Mood::Excited, 0.30),
        ],
        MoodCause::BattleDefeat => &[
            (Mood::Angry, 0.40),
            (Mood::Fearful, 0.30),
            (Mood::Determined, 0.30),
        ],
        MoodCause::AllyDeath => &[
            (Mood::Grieving, 0.70),
            (Mood::Angry, 0.30),
        ],
        MoodCause::LevelUp => &[
            (Mood::Inspired, 0.50),
            (Mood::Excited, 0.50),
        ],
        MoodCause::LongIdle => &[
            (Mood::Melancholic, 0.20),
        ],
        MoodCause::RivalEncounter => &[
            (Mood::Angry, 0.60),
        ],
        MoodCause::QuestSuccess => &[
            (Mood::Determined, 0.40),
            (Mood::Excited, 0.30),
        ],
        MoodCause::LowMoraleHighStress => &[
            (Mood::Fearful, 0.40),
        ],
        // These causes don't trigger new moods directly
        MoodCause::NaturalDecay | MoodCause::Contagion => return,
    };

    // Roll against each candidate in order — first hit wins.
    let roll = lcg_f32(&mut state.rng);
    let mut cumulative = 0.0;
    let mut chosen = None;
    for &(mood, prob) in candidates {
        cumulative += prob;
        if roll < cumulative {
            chosen = Some(mood);
            break;
        }
    }

    let new_mood = match chosen {
        Some(m) => m,
        None => return, // No mood triggered this roll.
    };

    let duration = mood_duration(&mut state.rng);

    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        if adv.status == AdventurerStatus::Dead {
            return;
        }
        let old_mood = adv.mood_state.mood;
        adv.mood_state = MoodState {
            mood: new_mood,
            started_at: tick,
            expires_at: tick + duration,
        };
        events.push(WorldEvent::MoodChanged {
            adventurer_id,
            old_mood,
            new_mood,
            cause,
        });
    }
}

/// Check for low-morale + high-stress condition and trigger Fearful mood.
/// Called during mood tick for idle adventurers.
pub fn check_stress_mood_triggers(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    // Collect candidates first to avoid borrow issues.
    let candidates: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status != AdventurerStatus::Dead
                && a.mood_state.mood == Mood::Neutral
                && a.morale < 30.0
                && a.stress > 70.0
        })
        .map(|a| a.id)
        .collect();

    for id in candidates {
        trigger_mood(state, id, MoodCause::LowMoraleHighStress, events);
    }
}

/// Check for long idle condition and trigger Melancholic mood.
pub fn check_idle_mood_triggers(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    // Adventurers idle for 1000+ ticks may become melancholic.
    let candidates: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status == AdventurerStatus::Idle
                && a.mood_state.mood == Mood::Neutral
                && a.party_id.is_none()
        })
        .map(|a| a.id)
        .collect();

    for id in candidates {
        trigger_mood(state, id, MoodCause::LongIdle, events);
    }
}

// --- Mood query helpers for other systems ---

/// Combat power multiplier from mood. Applied in battle resolution.
pub fn mood_combat_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Inspired => 1.15,  // +15% combat power
        Mood::Angry => 1.10,     // +10% attack (net positive but risky)
        Mood::Fearful => 0.90,   // -10% combat
        _ => 1.0,
    }
}

/// Defense multiplier from mood. Applied in battle resolution.
pub fn mood_defense_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Angry => 0.90, // -10% defense
        _ => 1.0,
    }
}

/// XP gain multiplier from mood.
pub fn mood_xp_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Inspired => 1.10, // +10% XP gain
        _ => 1.0,
    }
}

/// Loyalty modifier from mood (additive per tick).
pub fn mood_loyalty_modifier(mood: Mood) -> f32 {
    match mood {
        Mood::Determined => 0.10, // +10% loyalty drift
        _ => 0.0,
    }
}

/// Whether this mood increases rivalry formation chance.
pub fn mood_rivalry_bonus(mood: Mood) -> f32 {
    match mood {
        Mood::Angry => 0.20, // +20% rivalry formation chance
        _ => 0.0,
    }
}

/// Whether this mood increases bond growth (shared grief).
pub fn mood_bond_growth_bonus(mood: Mood) -> f32 {
    match mood {
        Mood::Grieving => 0.50, // +50% bond growth
        _ => 0.0,
    }
}

/// Whether this adventurer's mood makes them reckless (ignoring danger signals).
pub fn is_reckless(mood: Mood, rng: &mut u64) -> bool {
    match mood {
        Mood::Excited => lcg_f32(rng) < 0.10, // 10% chance
        _ => false,
    }
}

/// Whether this adventurer refuses high-threat quests due to fear.
pub fn refuses_high_threat(mood: Mood) -> bool {
    matches!(mood, Mood::Fearful)
}

/// Quest completion speed multiplier from mood.
pub fn mood_quest_speed_multiplier(mood: Mood) -> f32 {
    match mood {
        Mood::Determined => 1.20, // +20% quest completion speed
        _ => 1.0,
    }
}

/// Generate a random mood duration between MOOD_MIN_DURATION and MOOD_MAX_DURATION.
fn mood_duration(rng: &mut u64) -> u64 {
    let range = MOOD_MAX_DURATION - MOOD_MIN_DURATION;
    MOOD_MIN_DURATION + (lcg_f32(rng) * range as f32) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::{
        Adventurer, AdventurerStats, AdventurerStatus, CampaignState, Equipment,
    };

    /// Create a minimal test adventurer.
    fn test_adventurer(id: u32) -> Adventurer {
        Adventurer {
            id,
            name: format!("Test_{}", id),
            archetype: "ranger".into(),
            level: 3,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 80.0,
                attack: 14.0,
                defense: 8.0,
                speed: 12.0,
                ability_power: 6.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 60.0,
            stress: 10.0,
            fatigue: 5.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 40.0,
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
    fn mood_defaults_to_neutral() {
        let adv = test_adventurer(1);
        assert_eq!(adv.mood_state.mood, Mood::Neutral);
    }

    #[test]
    fn trigger_mood_sets_mood_on_adventurer() {
        let mut state = CampaignState::default_test_campaign(42);
        state.adventurers.push(test_adventurer(1));
        let mut events = Vec::new();

        // Force a mood via repeated triggers (probabilistic, but LCG is deterministic).
        for _ in 0..20 {
            trigger_mood(&mut state, 1, MoodCause::BattleVictory, &mut events);
        }

        // After 20 attempts with 70% combined chance, at least one should have hit.
        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!(
            adv.mood_state.mood == Mood::Inspired || adv.mood_state.mood == Mood::Excited,
            "Expected Inspired or Excited after battle victory triggers, got {:?}",
            adv.mood_state.mood
        );
        assert!(!events.is_empty());
    }

    #[test]
    fn mood_decays_to_neutral() {
        let mut state = CampaignState::default_test_campaign(42);
        state.adventurers.push(test_adventurer(1));
        let mut events = Vec::new();

        // Manually set a mood that expires soon.
        state.adventurers[0].mood_state = MoodState {
            mood: Mood::Inspired,
            started_at: 0,
            expires_at: 200, // Expires at tick 200
        };

        // Advance to tick 200.
        state.tick = 200;
        tick_moods(&mut state, &mut StepDeltas::default(), &mut events);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert_eq!(adv.mood_state.mood, Mood::Neutral);
        assert!(events.iter().any(|e| matches!(e,
            WorldEvent::MoodChanged { new_mood: Mood::Neutral, cause: MoodCause::NaturalDecay, .. }
        )));
    }

    #[test]
    fn combat_multipliers_correct() {
        assert_eq!(mood_combat_multiplier(Mood::Inspired), 1.15);
        assert_eq!(mood_combat_multiplier(Mood::Angry), 1.10);
        assert_eq!(mood_combat_multiplier(Mood::Fearful), 0.90);
        assert_eq!(mood_combat_multiplier(Mood::Neutral), 1.0);
    }

    #[test]
    fn mood_duration_in_range() {
        let mut rng: u64 = 12345;
        for _ in 0..100 {
            let d = mood_duration(&mut rng);
            assert!(d >= MOOD_MIN_DURATION && d <= MOOD_MAX_DURATION,
                "Duration {} out of range [{}, {}]", d, MOOD_MIN_DURATION, MOOD_MAX_DURATION);
        }
    }
}
