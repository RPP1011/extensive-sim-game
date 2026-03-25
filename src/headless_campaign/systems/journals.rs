//! Adventurer journal system — every 500 ticks.
//!
//! Each adventurer keeps a personal journal that accumulates entries from
//! their experiences, affecting personality drift (morale, loyalty) over time.
//! Journal entries are auto-generated from recent world events and pruned
//! to a maximum of 20 entries per adventurer.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, BattleStatus, CampaignState, JournalEntry, JournalType, QuestResult,
    lcg_f32, lcg_next,
};

/// Maximum journal entries per adventurer before oldest are pruned.
const MAX_JOURNAL_ENTRIES: usize = 20;

/// Number of recent entries used for sentiment averaging.
const SENTIMENT_WINDOW: usize = 10;

/// Positive sentiment threshold for morale/loyalty boost.
const POSITIVE_THRESHOLD: f32 = 0.3;

/// Negative sentiment threshold for morale penalty.
const NEGATIVE_THRESHOLD: f32 = -0.3;

/// Main tick function. Called every 500 ticks.
pub fn tick_journals(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 {
        return;
    }

    // Collect context needed for journal generation.
    let tick = state.tick;

    // Gather recent completed quests (last 500 ticks).
    let recent_quests: Vec<(u32, QuestResult, u32)> = state
        .completed_quests
        .iter()
        .filter(|q| {
            let completed_tick = q.completed_at_ms / 100; // ms -> tick
            completed_tick + 500 >= tick
        })
        .map(|q| (q.party_id, q.result, q.casualties))
        .collect();

    // Gather recently ended battles.
    let recent_battle_results: Vec<(u32, BattleStatus)> = state
        .active_battles
        .iter()
        .filter(|b| matches!(b.status, BattleStatus::Victory | BattleStatus::Defeat))
        .map(|b| (b.party_id, b.status))
        .collect();

    // Gather recent level-ups from the event log.
    let recent_levelups: Vec<u32> = state
        .event_log
        .iter()
        .filter(|e| e.tick + 500 >= tick)
        .filter_map(|e| {
            if e.description.contains("leveled up") {
                // Extract adventurer id from description is fragile;
                // instead we'll check adventurer xp thresholds below.
                None
            } else {
                None
            }
        })
        .collect();
    let _ = recent_levelups; // Handled per-adventurer below.

    // Gather recent deaths.
    let dead_adventurer_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    // Gather bond milestones (bonds > 60 that crossed recently).
    let high_bonds: Vec<(u32, u32, f32)> = state
        .adventurer_bonds
        .iter()
        .filter(|&(_, &v)| v > 60.0)
        .map(|(&(a, b), &v)| (a, b, v))
        .collect();

    // Gather party membership for context.
    let party_members: Vec<(u32, Vec<u32>)> = state
        .parties
        .iter()
        .map(|p| (p.id, p.member_ids.clone()))
        .collect();

    // Region names for flavor text.
    let region_names: Vec<String> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.name.clone())
        .collect();

    // Build journal entries for each living adventurer.
    let adventurer_count = state.adventurers.len();
    for i in 0..adventurer_count {
        if state.adventurers[i].status == AdventurerStatus::Dead {
            continue;
        }

        let adv_id = state.adventurers[i].id;
        let adv_name = state.adventurers[i].name.clone();
        let mut new_entries: Vec<JournalEntry> = Vec::new();

        // Find which party this adventurer is in.
        let my_party = party_members
            .iter()
            .find(|(_, members)| members.contains(&adv_id));

        // Pick a region name for flavor.
        let region_idx = (lcg_next(&mut state.rng) as usize) % region_names.len().max(1);
        let region_name = region_names
            .get(region_idx)
            .cloned()
            .unwrap_or_else(|| "the frontier".to_string());

        // --- Battle memories ---
        if let Some((party_id, _)) = my_party {
            for &(battle_party_id, ref status) in &recent_battle_results {
                if battle_party_id == *party_id {
                    let (sentiment, text) = match status {
                        BattleStatus::Victory => (
                            0.5 + lcg_f32(&mut state.rng) * 0.3,
                            format!(
                                "We fought hard in {} and emerged victorious. I can still feel the rush.",
                                region_name
                            ),
                        ),
                        BattleStatus::Defeat => (
                            -0.6 - lcg_f32(&mut state.rng) * 0.3,
                            format!(
                                "The battle in {} went badly. We were overwhelmed and had to retreat.",
                                region_name
                            ),
                        ),
                        _ => continue,
                    };
                    new_entries.push(JournalEntry {
                        tick,
                        entry_type: JournalType::BattleMemory,
                        text,
                        sentiment,
                    });
                }
            }
        }

        // --- Quest reflections / triumphs / regrets ---
        if let Some((party_id, _)) = my_party {
            for &(quest_party_id, ref result, casualties) in &recent_quests {
                if quest_party_id == *party_id {
                    match result {
                        QuestResult::Victory if casualties == 0 => {
                            new_entries.push(JournalEntry {
                                tick,
                                entry_type: JournalType::Triumph,
                                text: format!(
                                    "Quest complete — everyone made it back alive. {} would be proud.",
                                    adv_name
                                ),
                                sentiment: 0.7 + lcg_f32(&mut state.rng) * 0.2,
                            });
                        }
                        QuestResult::Victory => {
                            new_entries.push(JournalEntry {
                                tick,
                                entry_type: JournalType::QuestReflection,
                                text: format!(
                                    "We completed the quest, but lost {} along the way. Was it worth it?",
                                    casualties
                                ),
                                sentiment: 0.1 + lcg_f32(&mut state.rng) * 0.2,
                            });
                        }
                        QuestResult::Defeat => {
                            new_entries.push(JournalEntry {
                                tick,
                                entry_type: JournalType::Regret,
                                text: format!(
                                    "We failed. The quest was too much for us. I keep replaying what went wrong."
                                ),
                                sentiment: -0.5 - lcg_f32(&mut state.rng) * 0.3,
                            });
                        }
                        QuestResult::Abandoned => {
                            new_entries.push(JournalEntry {
                                tick,
                                entry_type: JournalType::Regret,
                                text: "We had to abandon the quest. Sometimes retreat is the only option.".to_string(),
                                sentiment: -0.3 - lcg_f32(&mut state.rng) * 0.2,
                            });
                        }
                    }
                }
            }
        }

        // --- Grief entries for dead allies ---
        for &dead_id in &dead_adventurer_ids {
            if dead_id == adv_id {
                continue;
            }
            let bond = crate::headless_campaign::systems::bonds::bond_strength(
                &state.adventurer_bonds,
                adv_id,
                dead_id,
            );
            if bond > 30.0 {
                // Find dead adventurer's name.
                let dead_name = state
                    .adventurers
                    .iter()
                    .find(|a| a.id == dead_id)
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| "a comrade".to_string());

                // Only write grief if we haven't already written one for this death.
                let already_grieved = state.adventurers[i]
                    .journal
                    .iter()
                    .any(|e| {
                        e.entry_type == JournalType::GriefEntry
                            && e.text.contains(&dead_name)
                    });
                if !already_grieved {
                    new_entries.push(JournalEntry {
                        tick,
                        entry_type: JournalType::GriefEntry,
                        text: format!(
                            "{} is gone. The guild hall feels emptier without them.",
                            dead_name
                        ),
                        sentiment: -0.7 - lcg_f32(&mut state.rng) * 0.2,
                    });
                }
            }
        }

        // --- Bond moments ---
        for &(a, b, strength) in &high_bonds {
            let partner_id = if a == adv_id {
                b
            } else if b == adv_id {
                a
            } else {
                continue;
            };
            // Only write if bond crossed 60 recently (within a margin).
            if strength < 62.0 {
                let partner_name = state
                    .adventurers
                    .iter()
                    .find(|a| a.id == partner_id)
                    .map(|a| a.name.clone())
                    .unwrap_or_else(|| "my companion".to_string());
                new_entries.push(JournalEntry {
                    tick,
                    entry_type: JournalType::BondMoment,
                    text: format!(
                        "{} and I have been through a lot together. I trust them with my life.",
                        partner_name
                    ),
                    sentiment: 0.5 + lcg_f32(&mut state.rng) * 0.2,
                });
            }
        }

        // --- Ambition (level up proxy: check if level is a multiple of 2 and high) ---
        let adv_level = state.adventurers[i].level;
        if adv_level > 1 && adv_level % 2 == 0 {
            // Only write if no ambition entry at this level yet.
            let already_logged = state.adventurers[i]
                .journal
                .iter()
                .any(|e| {
                    e.entry_type == JournalType::Ambition
                        && e.text.contains(&format!("level {}", adv_level))
                });
            if !already_logged {
                new_entries.push(JournalEntry {
                    tick,
                    entry_type: JournalType::Ambition,
                    text: format!(
                        "I reached level {}. Each step forward makes the next challenge feel possible.",
                        adv_level
                    ),
                    sentiment: 0.4 + lcg_f32(&mut state.rng) * 0.3,
                });
            }
        }

        // --- Discovery (exploration flavor, random chance) ---
        let discovery_roll = lcg_f32(&mut state.rng);
        if discovery_roll < 0.05 && !region_names.is_empty() {
            new_entries.push(JournalEntry {
                tick,
                entry_type: JournalType::Discovery,
                text: format!(
                    "Found something curious while passing through {}. The world is full of secrets.",
                    region_name
                ),
                sentiment: 0.3 + lcg_f32(&mut state.rng) * 0.2,
            });
        }

        // Emit events and append entries.
        for entry in &new_entries {
            events.push(WorldEvent::JournalEntryWritten {
                adventurer_id: adv_id,
                entry_type: entry.entry_type.clone(),
                sentiment: entry.sentiment,
            });
        }

        state.adventurers[i].journal.extend(new_entries);

        // Prune to MAX_JOURNAL_ENTRIES (keep most recent).
        let journal = &mut state.adventurers[i].journal;
        if journal.len() > MAX_JOURNAL_ENTRIES {
            let excess = journal.len() - MAX_JOURNAL_ENTRIES;
            journal.drain(..excess);
        }

        // --- Personality drift from journal sentiment ---
        let recent_count = journal.len().min(SENTIMENT_WINDOW);
        if recent_count > 0 {
            let avg_sentiment: f32 = journal
                .iter()
                .rev()
                .take(SENTIMENT_WINDOW)
                .map(|e| e.sentiment)
                .sum::<f32>()
                / recent_count as f32;

            let adv = &mut state.adventurers[i];
            if avg_sentiment > POSITIVE_THRESHOLD {
                adv.morale = (adv.morale + 1.0).min(100.0);
                adv.loyalty = (adv.loyalty + 0.5).min(100.0);
            } else if avg_sentiment < NEGATIVE_THRESHOLD {
                adv.morale = (adv.morale - 1.0).max(0.0);
                // Negative journal sentiment increases stress slightly.
                adv.stress = (adv.stress + 0.5).min(100.0);
            }
            // Mixed sentiment: no drift (balanced adventurer).
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::CampaignState;

    #[test]
    fn tick_journals_fires_at_500() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Not a multiple of 500 — should be a no-op.
        state.tick = 100;
        tick_journals(&mut state, &mut deltas, &mut events);
        assert!(events.is_empty());

        // At tick 500 it should fire.
        state.tick = 500;
        tick_journals(&mut state, &mut deltas, &mut events);
        // May or may not produce entries depending on state, but shouldn't panic.
    }

    #[test]
    fn journal_prunes_to_max() {
        let mut state = CampaignState::default_test_campaign(42);
        // Manually fill journal past max.
        if let Some(adv) = state.adventurers.first_mut() {
            for i in 0..25 {
                adv.journal.push(JournalEntry {
                    tick: i,
                    entry_type: JournalType::Discovery,
                    text: format!("Entry {}", i),
                    sentiment: 0.1,
                });
            }
        }

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        state.tick = 500;
        tick_journals(&mut state, &mut deltas, &mut events);

        if let Some(adv) = state.adventurers.first() {
            assert!(
                adv.journal.len() <= MAX_JOURNAL_ENTRIES,
                "Journal should be pruned to {} entries, got {}",
                MAX_JOURNAL_ENTRIES,
                adv.journal.len()
            );
        }
    }

    #[test]
    fn positive_sentiment_boosts_morale() {
        let mut state = CampaignState::default_test_campaign(42);
        if let Some(adv) = state.adventurers.first_mut() {
            adv.morale = 50.0;
            adv.loyalty = 50.0;
            // Fill with positive entries.
            for i in 0..10 {
                adv.journal.push(JournalEntry {
                    tick: i,
                    entry_type: JournalType::Triumph,
                    text: "Victory!".to_string(),
                    sentiment: 0.8,
                });
            }
        }

        let morale_before = state.adventurers[0].morale;
        let loyalty_before = state.adventurers[0].loyalty;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        state.tick = 500;
        tick_journals(&mut state, &mut deltas, &mut events);

        assert!(
            state.adventurers[0].morale >= morale_before,
            "Morale should increase with positive sentiment"
        );
        assert!(
            state.adventurers[0].loyalty >= loyalty_before,
            "Loyalty should increase with positive sentiment"
        );
    }

    #[test]
    fn negative_sentiment_reduces_morale() {
        let mut state = CampaignState::default_test_campaign(42);
        if let Some(adv) = state.adventurers.first_mut() {
            adv.morale = 50.0;
            adv.stress = 10.0;
            // Fill with negative entries.
            for i in 0..10 {
                adv.journal.push(JournalEntry {
                    tick: i,
                    entry_type: JournalType::GriefEntry,
                    text: "Loss...".to_string(),
                    sentiment: -0.8,
                });
            }
        }

        let morale_before = state.adventurers[0].morale;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        state.tick = 500;
        tick_journals(&mut state, &mut deltas, &mut events);

        assert!(
            state.adventurers[0].morale <= morale_before,
            "Morale should decrease with negative sentiment"
        );
        assert!(
            state.adventurers[0].stress > 10.0,
            "Stress should increase with negative sentiment"
        );
    }
}
