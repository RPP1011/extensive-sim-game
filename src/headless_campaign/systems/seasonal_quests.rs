//! Seasonal quest variant system — generates season-themed quests on season change.
//!
//! Quests are generated when the season changes (every 5000 ticks) and expire at the
//! end of that season. Completing all seasonal quests in a season earns a "Season
//! Champion" bonus (+5 rep, themed reward). Difficulty scales with game progression.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use crate::headless_campaign::systems::seasons::TICKS_PER_SEASON;

/// How often to check for seasonal quest generation/expiry (in ticks).
const CHECK_INTERVAL: u64 = 500;

/// Tick seasonal quests: generate on season change, expire old ones, check champion bonus.
pub fn tick_seasonal_quests(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Detect season change: season_tick was just reset (< CHECK_INTERVAL means fresh season).
    let just_changed = state.overworld.season_tick < CHECK_INTERVAL;

    // Expire quests from previous season.
    expire_seasonal_quests(state, events);

    // Generate new quests on season change.
    if just_changed {
        generate_seasonal_quests(state, events);
    }

    // Check season champion bonus.
    check_season_champion(state, events);
}

/// Remove expired seasonal quests (those whose season no longer matches).
fn expire_seasonal_quests(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    let current_season = state.overworld.season;
    let tick = state.tick;
    state.seasonal_quests.retain(|q| {
        q.season == current_season && tick < q.expires_tick
    });
}

/// Generate 2-3 season-appropriate quests.
fn generate_seasonal_quests(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let season = state.overworld.season;
    let tick = state.tick;
    let expires_tick = tick + TICKS_PER_SEASON;

    // Difficulty scales with progression: base 0.3 + tick-based ramp.
    let progression_scale = 1.0 + (state.tick as f32 / 10_000.0).min(3.0);

    // Determine how many quests: 2-3 via deterministic RNG.
    let roll = lcg_next(&mut state.rng);
    let quest_count = 2 + (roll % 2) as usize; // 2 or 3

    let templates = seasonal_templates(season);

    for i in 0..quest_count.min(templates.len()) {
        let template = &templates[i % templates.len()];

        let quest_id = state.next_quest_id;
        state.next_quest_id += 1;

        let threat = template.base_threat * progression_scale;
        let reward_gold = template.base_gold * progression_scale;

        let quest = SeasonalQuest {
            id: quest_id,
            name: template.name.to_string(),
            season,
            quest_type: template.quest_type.to_string(),
            threat_level: threat,
            reward_gold,
            reward_special: template.reward_special.to_string(),
            available_tick: tick,
            expires_tick,
            claimed: false,
        };

        state.seasonal_quests.push(quest);

        events.push(WorldEvent::SeasonalQuestAvailable {
            quest_id,
            name: template.name.to_string(),
            season,
        });
    }
}

/// Check if all seasonal quests for the current season have been claimed.
/// If so, award the Season Champion bonus.
fn check_season_champion(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let season = state.overworld.season;
    let seasonal = state
        .seasonal_quests
        .iter()
        .filter(|q| q.season == season)
        .collect::<Vec<_>>();

    if seasonal.is_empty() {
        return;
    }

    let all_claimed = seasonal.iter().all(|q| q.claimed);
    if !all_claimed {
        return;
    }

    // Check we haven't already awarded champion this season at this tick range.
    // Use a simple guard: remove all claimed quests (they're done) and award once.
    let already_awarded = state
        .seasonal_quests
        .iter()
        .filter(|q| q.season == season)
        .count()
        == 0;
    if already_awarded {
        return;
    }

    // Award Season Champion bonus.
    state.guild.reputation += 5.0;

    let themed_reward = match season {
        Season::Spring => "Blessed Seeds (+10 supplies)",
        Season::Summer => "Ancient Artifact Fragment",
        Season::Autumn => "Winter Stockpile (+20 supplies)",
        Season::Winter => "Frostforged Bonds (+5 morale all)",
    };

    // Apply themed reward effects.
    match season {
        Season::Spring | Season::Autumn => {
            let bonus = if season == Season::Spring { 10.0 } else { 20.0 };
            state.guild.supplies += bonus;
        }
        Season::Winter => {
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale + 5.0).min(100.0);
                }
            }
        }
        Season::Summer => {
            // Gold bonus for artifact fragment.
            state.guild.gold += 50.0;
        }
    }

    events.push(WorldEvent::SeasonChampionEarned {
        season,
        reward: themed_reward.to_string(),
    });

    // Clear seasonal quests for this season (all done).
    state.seasonal_quests.retain(|q| q.season != season);
}

/// Accept (claim) a seasonal quest by ID. Returns gold reward and special reward description.
pub fn accept_seasonal_quest(
    state: &mut CampaignState,
    quest_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<(f32, String), String> {
    let idx = state
        .seasonal_quests
        .iter()
        .position(|q| q.id == quest_id);

    let idx = match idx {
        Some(i) => i,
        None => return Err(format!("Seasonal quest {} not found", quest_id)),
    };

    if state.seasonal_quests[idx].claimed {
        return Err(format!("Seasonal quest {} already claimed", quest_id));
    }

    if state.seasonal_quests[idx].expires_tick <= state.tick {
        return Err(format!("Seasonal quest {} has expired", quest_id));
    }

    state.seasonal_quests[idx].claimed = true;
    let gold = state.seasonal_quests[idx].reward_gold;
    let special = state.seasonal_quests[idx].reward_special.clone();
    let name = state.seasonal_quests[idx].name.clone();

    state.guild.gold += gold;

    events.push(WorldEvent::SeasonalQuestCompleted {
        quest_id,
        name,
        reward_gold: gold,
        reward_special: special.clone(),
    });

    Ok((gold, special))
}

// ---------------------------------------------------------------------------
// Quest templates
// ---------------------------------------------------------------------------

struct QuestTemplate {
    name: &'static str,
    quest_type: &'static str,
    base_threat: f32,
    base_gold: f32,
    reward_special: &'static str,
}

fn seasonal_templates(season: Season) -> Vec<QuestTemplate> {
    match season {
        Season::Spring => vec![
            QuestTemplate {
                name: "Clear flooding routes",
                quest_type: "rescue",
                base_threat: 0.3,
                base_gold: 30.0,
                reward_special: "Rare herbs",
            },
            QuestTemplate {
                name: "Plant sacred grove",
                quest_type: "gather",
                base_threat: 0.2,
                base_gold: 20.0,
                reward_special: "Morale boost",
            },
            QuestTemplate {
                name: "Spring festival escort",
                quest_type: "escort",
                base_threat: 0.4,
                base_gold: 35.0,
                reward_special: "Faction favor",
            },
        ],
        Season::Summer => vec![
            QuestTemplate {
                name: "Hunt the great beast",
                quest_type: "combat",
                base_threat: 0.7,
                base_gold: 60.0,
                reward_special: "Trophy artifact",
            },
            QuestTemplate {
                name: "Explore ancient ruins",
                quest_type: "exploration",
                base_threat: 0.5,
                base_gold: 45.0,
                reward_special: "Ancient artifact",
            },
            QuestTemplate {
                name: "Trade fair security",
                quest_type: "escort",
                base_threat: 0.4,
                base_gold: 40.0,
                reward_special: "Trade contacts",
            },
        ],
        Season::Autumn => vec![
            QuestTemplate {
                name: "Harvest protection",
                quest_type: "defense",
                base_threat: 0.5,
                base_gold: 35.0,
                reward_special: "Food stores",
            },
            QuestTemplate {
                name: "Gather winter supplies",
                quest_type: "gather",
                base_threat: 0.3,
                base_gold: 25.0,
                reward_special: "Supply cache",
            },
            QuestTemplate {
                name: "Migration escort",
                quest_type: "escort",
                base_threat: 0.4,
                base_gold: 30.0,
                reward_special: "Reputation boost",
            },
        ],
        Season::Winter => vec![
            QuestTemplate {
                name: "Rescue stranded travelers",
                quest_type: "rescue",
                base_threat: 0.6,
                base_gold: 40.0,
                reward_special: "Rare materials",
            },
            QuestTemplate {
                name: "Dungeon delve",
                quest_type: "combat",
                base_threat: 0.8,
                base_gold: 70.0,
                reward_special: "Dungeon treasure",
            },
            QuestTemplate {
                name: "Guard the mountain pass",
                quest_type: "defense",
                base_threat: 0.6,
                base_gold: 45.0,
                reward_special: "Frostforged bonds",
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state
    }

    #[test]
    fn generates_quests_on_season_change() {
        let mut state = make_test_state();
        // Simulate a fresh season (season_tick < CHECK_INTERVAL).
        state.tick = 500;
        state.overworld.season_tick = 100;
        state.overworld.season = Season::Summer;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_seasonal_quests(&mut state, &mut deltas, &mut events);

        assert!(
            !state.seasonal_quests.is_empty(),
            "Expected seasonal quests to be generated"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, WorldEvent::SeasonalQuestAvailable { .. })),
            "Expected SeasonalQuestAvailable event"
        );
    }

    #[test]
    fn expires_quests_from_wrong_season() {
        let mut state = make_test_state();
        state.tick = 500;
        state.overworld.season = Season::Autumn;
        state.overworld.season_tick = 1000; // Not a fresh season.

        // Add a spring quest that should expire.
        state.seasonal_quests.push(SeasonalQuest {
            id: 99,
            name: "Old spring quest".into(),
            season: Season::Spring,
            quest_type: "rescue".into(),
            threat_level: 0.3,
            reward_gold: 30.0,
            reward_special: "herbs".into(),
            available_tick: 0,
            expires_tick: 5000,
            claimed: false,
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_seasonal_quests(&mut state, &mut deltas, &mut events);

        assert!(
            state.seasonal_quests.is_empty(),
            "Spring quest should be expired in Autumn"
        );
    }

    #[test]
    fn accept_seasonal_quest_awards_gold() {
        let mut state = make_test_state();
        state.tick = 100;
        let initial_gold = state.guild.gold;

        state.seasonal_quests.push(SeasonalQuest {
            id: 1,
            name: "Test quest".into(),
            season: Season::Spring,
            quest_type: "rescue".into(),
            threat_level: 0.3,
            reward_gold: 50.0,
            reward_special: "herbs".into(),
            available_tick: 0,
            expires_tick: 5000,
            claimed: false,
        });

        let mut events = Vec::new();
        let result = accept_seasonal_quest(&mut state, 1, &mut events);

        assert!(result.is_ok());
        let (gold, _special) = result.unwrap();
        assert!((gold - 50.0).abs() < 0.01);
        assert!((state.guild.gold - initial_gold - 50.0).abs() < 0.01);
        assert!(state.seasonal_quests[0].claimed);
    }

    #[test]
    fn season_champion_bonus() {
        let mut state = make_test_state();
        state.tick = 500;
        state.overworld.season = Season::Spring;
        state.overworld.season_tick = 1000;
        let initial_rep = state.guild.reputation;

        // Add completed seasonal quests.
        state.seasonal_quests.push(SeasonalQuest {
            id: 1,
            name: "Q1".into(),
            season: Season::Spring,
            quest_type: "rescue".into(),
            threat_level: 0.3,
            reward_gold: 30.0,
            reward_special: "herbs".into(),
            available_tick: 0,
            expires_tick: 5000,
            claimed: true,
        });
        state.seasonal_quests.push(SeasonalQuest {
            id: 2,
            name: "Q2".into(),
            season: Season::Spring,
            quest_type: "gather".into(),
            threat_level: 0.2,
            reward_gold: 20.0,
            reward_special: "morale".into(),
            available_tick: 0,
            expires_tick: 5000,
            claimed: true,
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_seasonal_quests(&mut state, &mut deltas, &mut events);

        assert!(
            (state.guild.reputation - initial_rep - 5.0).abs() < 0.01,
            "Season champion should grant +5 reputation"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, WorldEvent::SeasonChampionEarned { .. })),
            "Expected SeasonChampionEarned event"
        );
    }

    #[test]
    fn each_season_has_templates() {
        for season in [Season::Spring, Season::Summer, Season::Autumn, Season::Winter] {
            let templates = seasonal_templates(season);
            assert!(
                templates.len() >= 3,
                "Each season should have at least 3 quest templates"
            );
        }
    }
}
