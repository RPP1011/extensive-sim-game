//! Guild archives / library system — fires every 500 ticks (~50s game time).
//!
//! The guild accumulates knowledge from completed quests, exploration,
//! scholar NPCs, chronicle entries, and vision fulfillment. Knowledge
//! unlocks research topics; completed research grants permanent bonuses.
//!
//! Assigning a librarian (idle adventurer) doubles knowledge gain and
//! adds a research speed bonus.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;
use super::class_system::effective_noncombat_stats;

/// How often the archives system ticks (in ticks).
const ARCHIVES_INTERVAL: u64 = 17;

/// Knowledge gained per quest completed in the window.
const QUEST_COMPLETION_KNOWLEDGE: f32 = 5.0;

/// Knowledge gained per exploration quest completed.
const EXPLORATION_KNOWLEDGE: f32 = 3.0;

/// Knowledge gained per tick when a Scholar NPC has high reputation.
const SCHOLAR_NPC_KNOWLEDGE: f32 = 2.0;

/// Knowledge gained per chronicle entry added in the window.
const CHRONICLE_KNOWLEDGE: f32 = 1.0;

/// Knowledge gained per vision fulfilled in the window.
const VISION_FULFILLED_KNOWLEDGE: f32 = 10.0;

/// Base research progress per tick (before librarian bonus).
const BASE_RESEARCH_RATE: f32 = 2.0;

/// Research speed multiplier when a librarian is assigned.
const LIBRARIAN_RESEARCH_MULTIPLIER: f32 = 1.5;

/// Knowledge gain multiplier when a librarian is assigned.
const LIBRARIAN_KNOWLEDGE_MULTIPLIER: f32 = 2.0;

/// Research topic definitions: (name, category, knowledge_threshold, reward_description).
const RESEARCH_CATALOG: &[(&str, &str, f32, &str)] = &[
    ("Monster Bestiary", "combat", 50.0, "+15% combat vs known species"),
    ("Alchemical Treatise", "crafting", 60.0, "+20% potion crafting quality"),
    ("Ancient Maps", "exploration", 75.0, "Reveal dungeon connections"),
    ("Trade Routes Compendium", "economy", 80.0, "+15% trade income"),
    ("Faction Histories", "diplomacy", 100.0, "+10% diplomacy effectiveness"),
    ("War Tactics", "combat", 150.0, "+10% party combat power"),
];

pub fn tick_archives(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % ARCHIVES_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Check librarian validity ---
    // If the assigned librarian is no longer idle (or dead), unassign.
    if let Some(lib_id) = state.archives.librarian_id {
        let still_valid = state
            .adventurers
            .iter()
            .any(|a| a.id == lib_id && a.status == AdventurerStatus::Idle);
        if !still_valid {
            state.archives.librarian_id = None;
        }
    }
    let has_librarian = state.archives.librarian_id.is_some();

    // --- Phase 2: Accumulate knowledge ---
    let mut knowledge_gained: f32 = 0.0;
    let mut sources: Vec<&str> = Vec::new();

    // Quest completions in the last interval window.
    let window_start_ms =
        state.elapsed_ms.saturating_sub(ARCHIVES_INTERVAL * CAMPAIGN_TURN_SECS as u64 * 1000);

    let quest_completions = state
        .completed_quests
        .iter()
        .filter(|q| q.completed_at_ms >= window_start_ms && q.result == QuestResult::Victory)
        .count() as f32;
    if quest_completions > 0.0 {
        knowledge_gained += quest_completions * QUEST_COMPLETION_KNOWLEDGE;
        sources.push("quests");
    }

    // Exploration quests (subset of completed).
    let exploration_completions = state
        .completed_quests
        .iter()
        .filter(|q| {
            q.completed_at_ms >= window_start_ms
                && q.result == QuestResult::Victory
                && q.quest_type == QuestType::Exploration
        })
        .count() as f32;
    if exploration_completions > 0.0 {
        knowledge_gained += exploration_completions * EXPLORATION_KNOWLEDGE;
        sources.push("exploration");
    }

    // Scholar NPCs with good reputation (>50).
    let scholar_count = state
        .named_npcs
        .iter()
        .filter(|npc| npc.role == NpcRole::Scholar && npc.reputation > 50.0)
        .count() as f32;
    if scholar_count > 0.0 {
        knowledge_gained += scholar_count * SCHOLAR_NPC_KNOWLEDGE;
        sources.push("scholars");
    }

    // Chronicle entries added in the window (use tick instead of ms).
    let window_start_tick = state.tick.saturating_sub(ARCHIVES_INTERVAL);
    let chronicle_count = state
        .chronicle
        .iter()
        .filter(|entry| entry.tick >= window_start_tick)
        .count() as f32;
    if chronicle_count > 0.0 {
        knowledge_gained += chronicle_count * CHRONICLE_KNOWLEDGE;
        sources.push("chronicle");
    }

    // Visions fulfilled (any that are fulfilled and recent by tick).
    let visions_fulfilled = state
        .visions
        .iter()
        .filter(|v| v.fulfilled && v.tick >= window_start_tick)
        .count() as f32;
    if visions_fulfilled > 0.0 {
        knowledge_gained += visions_fulfilled * VISION_FULFILLED_KNOWLEDGE;
        sources.push("visions");
    }

    // Librarian doubles knowledge gain.
    if has_librarian && knowledge_gained > 0.0 {
        knowledge_gained *= LIBRARIAN_KNOWLEDGE_MULTIPLIER;
    }

    // Scholarship bonus from adventurer class stats adds to knowledge gain
    let scholarship_bonus: f32 = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && a.faction_id.is_none())
        .map(|a| effective_noncombat_stats(a).4) // scholarship component
        .sum();
    knowledge_gained += scholarship_bonus;

    if knowledge_gained > 0.0 {
        state.archives.knowledge_points += knowledge_gained;
        events.push(WorldEvent::KnowledgeGained {
            amount: knowledge_gained,
            source: sources.join(", "),
        });
    }

    // --- Phase 3: Unlock new research topics ---
    for &(name, category, threshold, reward) in RESEARCH_CATALOG {
        if state.archives.knowledge_points < threshold {
            continue;
        }
        // Skip if already unlocked or completed.
        let already_unlocked = state
            .archives
            .research_topics
            .iter()
            .any(|t| t.name == name);
        let already_completed = state.archives.completed_research.iter().any(|n| n == name);
        if already_unlocked || already_completed {
            continue;
        }
        state.archives.research_topics.push(ResearchTopic {
            name: name.to_string(),
            category: category.to_string(),
            progress: 0.0,
            reward_description: reward.to_string(),
        });
    }

    // --- Phase 4: Advance research on active topics ---
    let base_rate = if has_librarian {
        BASE_RESEARCH_RATE * LIBRARIAN_RESEARCH_MULTIPLIER
    } else {
        BASE_RESEARCH_RATE
    };
    // Scholarship bonus also accelerates research progress
    let research_rate = base_rate + scholarship_bonus * 0.1;

    let mut completed_names: Vec<String> = Vec::new();
    for topic in &mut state.archives.research_topics {
        topic.progress = (topic.progress + research_rate).min(100.0);
        if topic.progress >= 100.0 {
            completed_names.push(topic.name.clone());
        }
    }

    // --- Phase 5: Complete research and grant bonuses ---
    for name in &completed_names {
        state.archives.research_topics.retain(|t| &t.name != name);
        state.archives.completed_research.push(name.clone());
        events.push(WorldEvent::ResearchCompleted {
            topic: name.clone(),
        });
        apply_research_bonus(state, name);
    }

    // --- Phase 6: Update system trackers ---
    state.system_trackers.archives_knowledge = state.archives.knowledge_points;
    state.system_trackers.archives_completed_count =
        state.archives.completed_research.len() as u32;
    state.system_trackers.archives_active_research =
        state.archives.research_topics.len() as u32;
}

/// Apply permanent bonuses from completed research.
fn apply_research_bonus(state: &mut CampaignState, name: &str) {
    match name {
        "Monster Bestiary" => {
            // +15% combat vs known species: boost all adventurer attack slightly.
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.stats.attack *= 1.05; // Distributed: cumulative across adventurers
                }
            }
        }
        "Faction Histories" => {
            // +10% diplomacy effectiveness: nudge all faction relations up.
            for faction in &mut state.factions {
                let old = faction.relationship_to_guild;
                faction.relationship_to_guild = (old + old.abs() * 0.10).clamp(-100.0, 100.0);
            }
        }
        "Ancient Maps" => {
            // Reveal dungeon connections: mark all dungeons as having known connections.
            for dungeon in &mut state.dungeons {
                dungeon.explored = (dungeon.explored + 15.0).min(100.0);
            }
        }
        "Alchemical Treatise" => {
            // +20% potion crafting quality: boost herb resources.
            if let Some(herbs) = state.resources.get_mut(&ResourceType::Herbs) {
                *herbs *= 1.20;
            }
        }
        "War Tactics" => {
            // +10% party combat power: boost defense for all adventurers.
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.stats.defense *= 1.05;
                }
            }
        }
        "Trade Routes Compendium" => {
            // +15% trade income: small gold bonus.
            state.guild.gold += 50.0;
        }
        _ => {}
    }
}
