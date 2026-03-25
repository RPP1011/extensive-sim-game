//! Chronicle / narrative log system — every 100 ticks.
//!
//! Records significant campaign events as narrative entries that feed back
//! into quest generation. The chronicle provides historical context for
//! revenge quests, memorial quests, faction-themed quests, and personal
//! adventurer legacies.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{CampaignState, ChronicleEntry, ChronicleType};

/// Maximum chronicle entries before pruning lowest-significance entries.
const MAX_CHRONICLE_ENTRIES: usize = 100;

/// Minimum significance to record an entry (avoids clutter).
const MIN_SIGNIFICANCE: f32 = 3.0;

/// Scan recent WorldEvents (since last chronicle tick) and convert significant
/// ones into narrative chronicle entries. Runs every 100 ticks.
pub fn tick_chronicle(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 3 != 0 {
        return;
    }

    // Collect new entries from recent events
    let mut new_entries: Vec<ChronicleEntry> = Vec::new();

    // Scan all events produced this step (they cover the last 100 ticks)
    for event in events.iter() {
        if let Some(entry) = convert_event(state, event) {
            if entry.significance >= MIN_SIGNIFICANCE {
                new_entries.push(entry);
            }
        }
    }

    // Also scan completed quests that finished recently
    for cq in &state.completed_quests {
        let completed_tick = cq.completed_at_ms / 100; // ms to ticks
        if completed_tick + 100 >= state.tick && completed_tick <= state.tick {
            let threat = cq.reward_applied.reputation; // proxy for significance
            if threat > 0.0 {
                let sig = 5.0 + threat / 20.0;
                if sig >= MIN_SIGNIFICANCE {
                    let quest_desc = format!("{:?}", cq.quest_type);
                    let text = format!(
                        "{} completed a {} quest, earning the guild great renown",
                        party_name_for_id(state, cq.party_id),
                        quest_desc.to_lowercase(),
                    );
                    // Avoid duplicates if we already have an entry for this quest
                    let dominated = new_entries
                        .iter()
                        .any(|e| e.entry_type == ChronicleType::QuestCompletion && e.tick == state.tick);
                    if !dominated {
                        new_entries.push(ChronicleEntry {
                            tick: state.tick,
                            entry_type: ChronicleType::QuestCompletion,
                            text,
                            participants: Vec::new(),
                            location_id: None,
                            faction_id: None,
                            significance: sig,
                        });
                    }
                }
            }
        }
    }

    // Emit WorldEvents for each recorded entry
    for entry in &new_entries {
        events.push(WorldEvent::ChronicleRecorded {
            text: entry.text.clone(),
            significance: entry.significance,
        });
    }

    // Append to state chronicle
    state.chronicle.extend(new_entries);

    // Prune if over capacity — remove lowest-significance entries first
    if state.chronicle.len() > MAX_CHRONICLE_ENTRIES {
        state
            .chronicle
            .sort_by(|a, b| b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal));
        state.chronicle.truncate(MAX_CHRONICLE_ENTRIES);
        // Re-sort by tick for chronological order
        state.chronicle.sort_by_key(|e| e.tick);
    }
}

/// Convert a WorldEvent into a ChronicleEntry, if significant enough.
fn convert_event(state: &CampaignState, event: &WorldEvent) -> Option<ChronicleEntry> {
    let season = format!("{:?}", state.overworld.season).to_lowercase();
    let tick = state.tick;

    match event {
        WorldEvent::AdventurerDied {
            adventurer_id,
            cause,
        } => {
            let adv_name = adventurer_name(state, *adventurer_id);
            let region = adventurer_region(state, *adventurer_id);
            let text = format!(
                "In the {} of tick {}, {} fell{}{}",
                season,
                tick,
                adv_name,
                region
                    .as_ref()
                    .map(|r| format!(" defending {}", r))
                    .unwrap_or_default(),
                if cause.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", cause)
                },
            );
            Some(ChronicleEntry {
                tick,
                entry_type: ChronicleType::Death,
                text,
                participants: vec![*adventurer_id],
                location_id: None,
                faction_id: None,
                significance: 8.0,
            })
        }

        WorldEvent::QuestCompleted { quest_id, result } => {
            // Find the completed quest for threat info
            let cq = state.completed_quests.iter().find(|q| q.id == *quest_id);
            let threat = cq.map(|q| q.reward_applied.reputation).unwrap_or(0.0);
            if threat <= 0.0 && *result != crate::headless_campaign::state::QuestResult::Victory {
                return None;
            }
            let sig = 5.0 + threat / 20.0;
            let quest_desc = cq
                .map(|q| format!("{:?}", q.quest_type).to_lowercase())
                .unwrap_or_else(|| "unknown".to_string());
            let text = match result {
                crate::headless_campaign::state::QuestResult::Victory => {
                    format!(
                        "The guild completed a legendary {} quest (id {}), earning great renown",
                        quest_desc, quest_id
                    )
                }
                crate::headless_campaign::state::QuestResult::Defeat => {
                    format!(
                        "The guild suffered a crushing defeat on {} quest (id {})",
                        quest_desc, quest_id
                    )
                }
                crate::headless_campaign::state::QuestResult::Abandoned => {
                    return None;
                }
            };
            Some(ChronicleEntry {
                tick,
                entry_type: ChronicleType::QuestCompletion,
                text,
                participants: Vec::new(),
                location_id: None,
                faction_id: cq.and_then(|q| q.reward_applied.relation_faction_id),
                significance: sig,
            })
        }

        WorldEvent::FactionActionTaken { faction_id, action } => {
            let action_lower = action.to_lowercase();
            if action_lower.contains("war") || action_lower.contains("attack") {
                let fname = faction_name(state, *faction_id);
                let text = format!(
                    "War erupted as {} took aggressive action: {}",
                    fname, action
                );
                Some(ChronicleEntry {
                    tick,
                    entry_type: ChronicleType::DiplomaticEvent,
                    text,
                    participants: Vec::new(),
                    location_id: None,
                    faction_id: Some(*faction_id),
                    significance: 6.0,
                })
            } else if action_lower.contains("trade") || action_lower.contains("alliance") {
                let fname = faction_name(state, *faction_id);
                let text = format!("{} established a trade agreement: {}", fname, action);
                Some(ChronicleEntry {
                    tick,
                    entry_type: ChronicleType::DiplomaticEvent,
                    text,
                    participants: Vec::new(),
                    location_id: None,
                    faction_id: Some(*faction_id),
                    significance: 3.0,
                })
            } else {
                None
            }
        }

        WorldEvent::RegionOwnerChanged {
            region_id,
            old_owner,
            new_owner,
        } => {
            let region_name = state
                .overworld
                .regions
                .get(*region_id)
                .map(|r| r.name.as_str())
                .unwrap_or("unknown lands");
            let old_name = faction_name(state, *old_owner);
            let new_name = faction_name(state, *new_owner);
            let text = format!(
                "Control of {} shifted from {} to {} in the {} of tick {}",
                region_name, old_name, new_name, season, tick
            );
            Some(ChronicleEntry {
                tick,
                entry_type: ChronicleType::DiplomaticEvent,
                text,
                participants: Vec::new(),
                location_id: Some(*region_id),
                faction_id: Some(*new_owner),
                significance: 6.0,
            })
        }

        WorldEvent::CalamityWarning { description } => Some(ChronicleEntry {
            tick,
            entry_type: ChronicleType::CrisisEvent,
            text: format!("Crisis looms: {}", description),
            participants: Vec::new(),
            location_id: None,
            faction_id: None,
            significance: 7.0,
        }),

        WorldEvent::BuildingUpgraded {
            building,
            new_tier,
            cost: _,
        } => {
            if *new_tier >= 2 {
                Some(ChronicleEntry {
                    tick,
                    entry_type: ChronicleType::Construction,
                    text: format!(
                        "The guild's {} was upgraded to tier {}",
                        building, new_tier
                    ),
                    participants: Vec::new(),
                    location_id: None,
                    faction_id: None,
                    significance: 2.0,
                })
            } else {
                None
            }
        }

        WorldEvent::AdventurerLevelUp {
            adventurer_id,
            new_level,
        } => {
            if *new_level >= 5 && *new_level % 5 == 0 {
                let name = adventurer_name(state, *adventurer_id);
                Some(ChronicleEntry {
                    tick,
                    entry_type: ChronicleType::HeroicDeed,
                    text: format!(
                        "{} achieved legendary level {}, cementing their place in guild history",
                        name, new_level
                    ),
                    participants: vec![*adventurer_id],
                    location_id: None,
                    faction_id: None,
                    significance: 4.0,
                })
            } else {
                None
            }
        }

        WorldEvent::BattleEnded { battle_id, result } => {
            let battle = state.active_battles.iter().find(|b| b.id == *battle_id);
            match result {
                crate::headless_campaign::state::BattleStatus::Victory => {
                    let quest_id = battle.map(|b| b.quest_id);
                    Some(ChronicleEntry {
                        tick,
                        entry_type: ChronicleType::BattleRecord,
                        text: format!(
                            "A great battle was won (battle {}, quest {:?})",
                            battle_id, quest_id
                        ),
                        participants: Vec::new(),
                        location_id: None,
                        faction_id: None,
                        significance: 4.0,
                    })
                }
                crate::headless_campaign::state::BattleStatus::Defeat => {
                    Some(ChronicleEntry {
                        tick,
                        entry_type: ChronicleType::Tragedy,
                        text: format!("The guild suffered a devastating defeat in battle {}", battle_id),
                        participants: Vec::new(),
                        location_id: None,
                        faction_id: None,
                        significance: 5.0,
                    })
                }
                _ => None,
            }
        }

        WorldEvent::RandomEvent { name, description } => {
            let text = format!("{}: {}", name, description);
            Some(ChronicleEntry {
                tick,
                entry_type: ChronicleType::Discovery,
                text,
                participants: Vec::new(),
                location_id: None,
                faction_id: None,
                significance: 3.5,
            })
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn adventurer_name(state: &CampaignState, id: u32) -> String {
    state
        .adventurers
        .iter()
        .find(|a| a.id == id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| format!("Adventurer #{}", id))
}

fn adventurer_region(state: &CampaignState, id: u32) -> Option<String> {
    // Try to find the adventurer's party and its current quest's faction region
    let adv = state.adventurers.iter().find(|a| a.id == id)?;
    let party_id = adv.party_id?;
    let party = state.parties.iter().find(|p| p.id == party_id)?;
    let quest_id = party.quest_id?;
    // Find the quest and its source area
    let quest = state
        .active_quests
        .iter()
        .find(|q| q.id == quest_id)?;
    let area_id = quest.request.source_area_id?;
    state
        .overworld
        .regions
        .get(area_id)
        .map(|r| r.name.clone())
}

fn faction_name(state: &CampaignState, id: usize) -> String {
    state
        .factions
        .get(id)
        .map(|f| f.name.clone())
        .unwrap_or_else(|| format!("Faction #{}", id))
}

fn party_name_for_id(state: &CampaignState, party_id: u32) -> String {
    let party = state.parties.iter().find(|p| p.id == party_id);
    if let Some(party) = party {
        let member_names: Vec<String> = party
            .member_ids
            .iter()
            .filter_map(|id| state.adventurers.iter().find(|a| a.id == *id))
            .map(|a| a.name.clone())
            .collect();
        if member_names.is_empty() {
            format!("Party #{}", party_id)
        } else {
            member_names.join(", ")
        }
    } else {
        format!("Party #{}", party_id)
    }
}

// ---------------------------------------------------------------------------
// Chronicle queries for quest generation
// ---------------------------------------------------------------------------

/// Recent tragedy entries (deaths, defeats) — useful for revenge/memorial quests.
pub fn recent_tragedies(state: &CampaignState) -> Vec<&ChronicleEntry> {
    state
        .chronicle
        .iter()
        .filter(|e| {
            matches!(
                e.entry_type,
                ChronicleType::Death | ChronicleType::Tragedy
            )
        })
        .collect()
}

/// All chronicle entries involving a specific faction — for faction-themed quests.
pub fn faction_history(state: &CampaignState, faction_id: usize) -> Vec<&ChronicleEntry> {
    state
        .chronicle
        .iter()
        .filter(|e| e.faction_id == Some(faction_id))
        .collect()
}

/// All chronicle entries involving a specific adventurer — for personal/legacy quests.
pub fn adventurer_legacy(state: &CampaignState, adv_id: u32) -> Vec<&ChronicleEntry> {
    state
        .chronicle
        .iter()
        .filter(|e| e.participants.contains(&adv_id))
        .collect()
}
