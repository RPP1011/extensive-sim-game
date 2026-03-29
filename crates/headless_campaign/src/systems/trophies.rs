//! Guild trophy hall — trophies from major victories provide passive bonuses.
//!
//! Trophies are earned automatically when the guild achieves significant milestones:
//! defeating nemeses, resolving crises, conquering factions, discovering artifacts,
//! killing large monsters, or completing quest milestones. Each trophy applies a
//! passive bonus that stacks (up to 10 trophies max). Bonuses are applied every
//! 500 ticks.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to apply passive trophy bonuses (in ticks).
const TROPHY_TICK_INTERVAL: u64 = 17;

/// Maximum trophies the hall can hold.
const MAX_TROPHIES: usize = 10;

/// Tick the trophy hall system every `TROPHY_TICK_INTERVAL` ticks.
///
/// 1. Check for trophy-earning triggers and create new trophies.
/// 2. Apply passive bonuses from all held trophies.
pub fn tick_trophies(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TROPHY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Check for new trophy triggers ---
    check_trophy_triggers(state, events);

    // --- Apply passive bonuses from all trophies ---
    apply_trophy_bonuses(state, events);
}

/// Intermediate trophy candidate (no ID yet — avoids borrowing state mutably
/// while iterating over its collections).
struct TrophyCandidate {
    name: String,
    trophy_type: TrophyType,
    source_description: String,
    bonus: TrophyBonus,
}

/// Scan game state for trophy-earning conditions.
fn check_trophy_triggers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut candidates: Vec<TrophyCandidate> = Vec::new();

    let tick_window_start = state.tick.saturating_sub(TROPHY_TICK_INTERVAL);
    let tick_window_start_ms = tick_window_start * CAMPAIGN_TURN_SECS as u64 * 1000;
    let current_tick = state.tick;

    // --- NemesisSkull: high-threat combat quest victories ---
    for quest in &state.completed_quests {
        if quest.result != QuestResult::Victory || quest.completed_at_ms <= tick_window_start_ms {
            continue;
        }
        if quest.quest_type == QuestType::Combat && quest.threat_level >= 80.0 {
            let tag = format!("Quest#{}", quest.id);
            let already_has = state.trophy_hall.iter().any(|t| {
                t.trophy_type == TrophyType::NemesisSkull
                    && t.source_description.contains(&tag)
            });
            if !already_has {
                candidates.push(TrophyCandidate {
                    name: format!("Quest#{}'s Nemesis Skull", quest.id),
                    trophy_type: TrophyType::NemesisSkull,
                    source_description: format!(
                        "Quest#{} defeated (threat {})",
                        quest.id, quest.threat_level
                    ),
                    bonus: TrophyBonus::CombatBoost(0.05),
                });
            }
        }
    }

    // --- AncientArtifact: exploration quest victory in high-threat area ---
    for quest in &state.completed_quests {
        if quest.result != QuestResult::Victory || quest.completed_at_ms <= tick_window_start_ms {
            continue;
        }
        if quest.quest_type == QuestType::Exploration && quest.threat_level >= 60.0 {
            let tag = format!("Quest#{}", quest.id);
            let already_has = state.trophy_hall.iter().any(|t| {
                t.trophy_type == TrophyType::AncientArtifact
                    && t.source_description.contains(&tag)
            });
            if !already_has {
                candidates.push(TrophyCandidate {
                    name: format!("Quest#{}'s Ancient Artifact", quest.id),
                    trophy_type: TrophyType::AncientArtifact,
                    source_description: format!(
                        "Quest#{} ancient discovery (threat {})",
                        quest.id, quest.threat_level
                    ),
                    bonus: TrophyBonus::XpBoost(0.10),
                });
            }
        }
    }

    // --- MonsterTrophy: combat victory with moderately high threat ---
    for quest in &state.completed_quests {
        if quest.result != QuestResult::Victory || quest.completed_at_ms <= tick_window_start_ms {
            continue;
        }
        if quest.quest_type == QuestType::Combat
            && quest.threat_level >= 70.0
            && quest.threat_level < 80.0
        {
            let tag = format!("Quest#{}", quest.id);
            let already_has = state.trophy_hall.iter().any(|t| {
                t.trophy_type == TrophyType::MonsterTrophy
                    && t.source_description.contains(&tag)
            });
            if !already_has {
                candidates.push(TrophyCandidate {
                    name: format!("Quest#{}'s Monster Trophy", quest.id),
                    trophy_type: TrophyType::MonsterTrophy,
                    source_description: format!(
                        "Quest#{} monster slain (threat {})",
                        quest.id, quest.threat_level
                    ),
                    bonus: TrophyBonus::MoraleBoost(0.03),
                });
            }
        }
    }

    // --- CrisisRelic: crisis resolved ---
    if state.overworld.active_crises.is_empty() {
        let had_crisis_trophy = state
            .trophy_hall
            .iter()
            .any(|t| t.trophy_type == TrophyType::CrisisRelic);
        if !had_crisis_trophy
            && current_tick > 5000
            && state.guild.reputation > 60.0
            && state.overworld.campaign_progress > 0.5
        {
            candidates.push(TrophyCandidate {
                name: "Calamity's Relic".into(),
                trophy_type: TrophyType::CrisisRelic,
                source_description: "Crisis resolved through guild effort".into(),
                bonus: TrophyBonus::ReputationBoost(0.10),
            });
        }
    }

    // --- FactionBanner: faction with 0 territory (conquered) ---
    for faction in &state.factions {
        if faction.territory_size == 0 && faction.military_strength < 10.0 {
            let already_has = state.trophy_hall.iter().any(|t| {
                t.trophy_type == TrophyType::FactionBanner
                    && t.source_description.contains(&faction.name)
            });
            if !already_has {
                candidates.push(TrophyCandidate {
                    name: format!("{}'s Battle Standard", faction.name),
                    trophy_type: TrophyType::FactionBanner,
                    source_description: format!("Faction {} conquered", faction.name),
                    bonus: TrophyBonus::RecruitmentBoost(0.05),
                });
            }
        }
    }

    // --- QuestToken: every 10th completed quest ---
    let victory_count = state
        .completed_quests
        .iter()
        .filter(|q| q.result == QuestResult::Victory)
        .count();
    if victory_count > 0 && victory_count % 10 == 0 {
        let milestone_label = format!("{}th quest", victory_count);
        let already_has = state.trophy_hall.iter().any(|t| {
            t.trophy_type == TrophyType::QuestToken
                && t.source_description.contains(&milestone_label)
        });
        if !already_has {
            candidates.push(TrophyCandidate {
                name: format!("{}'s Quest Token", milestone_label),
                trophy_type: TrophyType::QuestToken,
                source_description: format!("{} milestone reached", milestone_label),
                bonus: TrophyBonus::GoldBoost(0.05),
            });
        }
    }

    // --- Allocate IDs and add trophies ---
    for candidate in candidates {
        let id = state.next_event_id;
        state.next_event_id += 1;
        let trophy = Trophy {
            id,
            name: candidate.name,
            trophy_type: candidate.trophy_type,
            source_description: candidate.source_description,
            earned_tick: current_tick,
            bonus: candidate.bonus,
        };
        add_trophy(state, trophy, events);
    }
}

/// Add a trophy to the hall, replacing the weakest if at capacity.
fn add_trophy(state: &mut CampaignState, trophy: Trophy, events: &mut Vec<WorldEvent>) {
    let bonus_desc = format_bonus(&trophy.bonus);
    let trophy_name = trophy.name.clone();

    if state.trophy_hall.len() >= MAX_TROPHIES {
        // Find and replace the weakest trophy (lowest bonus magnitude).
        let weakest_idx = state
            .trophy_hall
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                bonus_magnitude(&a.bonus)
                    .partial_cmp(&bonus_magnitude(&b.bonus))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(idx) = weakest_idx {
            if bonus_magnitude(&trophy.bonus) > bonus_magnitude(&state.trophy_hall[idx].bonus) {
                state.trophy_hall[idx] = trophy;
            } else {
                // New trophy is weaker than everything in the hall; discard it.
                return;
            }
        }
    } else {
        state.trophy_hall.push(trophy);
    }

    events.push(WorldEvent::TrophyEarned {
        name: trophy_name,
        bonus_description: bonus_desc,
    });
}

/// Apply passive bonuses from all trophies in the hall.
fn apply_trophy_bonuses(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.trophy_hall.is_empty() {
        return;
    }

    let mut total_recruitment = 0.0f32;
    let mut total_morale = 0.0f32;
    let mut total_reputation = 0.0f32;
    let mut total_combat = 0.0f32;
    let mut total_gold = 0.0f32;
    let mut total_xp = 0.0f32;

    for trophy in &state.trophy_hall {
        match trophy.bonus {
            TrophyBonus::RecruitmentBoost(v) => total_recruitment += v,
            TrophyBonus::MoraleBoost(v) => total_morale += v,
            TrophyBonus::ReputationBoost(v) => total_reputation += v,
            TrophyBonus::CombatBoost(v) => total_combat += v,
            TrophyBonus::GoldBoost(v) => total_gold += v,
            TrophyBonus::XpBoost(v) => total_xp += v,
        }
    }

    // Apply reputation boost
    if total_reputation > 0.0 {
        let rep_gain = state.guild.reputation * total_reputation;
        state.guild.reputation = (state.guild.reputation + rep_gain).min(100.0);
    }

    // Apply gold boost
    if total_gold > 0.0 {
        let gold_gain = 10.0 * total_gold; // Flat gold per application
        state.guild.gold += gold_gain;
    }

    // Apply morale boost to all alive adventurers
    if total_morale > 0.0 {
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale + total_morale * 100.0).min(100.0);
            }
        }
    }

    // Build summary string
    let mut parts = Vec::new();
    if total_recruitment > 0.0 {
        parts.push(format!("recruitment+{:.0}%", total_recruitment * 100.0));
    }
    if total_morale > 0.0 {
        parts.push(format!("morale+{:.0}%", total_morale * 100.0));
    }
    if total_reputation > 0.0 {
        parts.push(format!("reputation+{:.0}%", total_reputation * 100.0));
    }
    if total_combat > 0.0 {
        parts.push(format!("combat+{:.0}%", total_combat * 100.0));
    }
    if total_gold > 0.0 {
        parts.push(format!("gold+{:.0}%", total_gold * 100.0));
    }
    if total_xp > 0.0 {
        parts.push(format!("xp+{:.0}%", total_xp * 100.0));
    }

    if !parts.is_empty() {
        events.push(WorldEvent::TrophyBonusApplied {
            total_bonuses: parts.join(", "),
        });
    }
}

/// Get the magnitude of a bonus for comparison purposes.
fn bonus_magnitude(bonus: &TrophyBonus) -> f32 {
    match bonus {
        TrophyBonus::RecruitmentBoost(v)
        | TrophyBonus::MoraleBoost(v)
        | TrophyBonus::ReputationBoost(v)
        | TrophyBonus::CombatBoost(v)
        | TrophyBonus::GoldBoost(v)
        | TrophyBonus::XpBoost(v) => *v,
    }
}

/// Format a trophy bonus for display.
fn format_bonus(bonus: &TrophyBonus) -> String {
    match bonus {
        TrophyBonus::RecruitmentBoost(v) => format!("+{:.0}% recruitment", v * 100.0),
        TrophyBonus::MoraleBoost(v) => format!("+{:.0}% morale", v * 100.0),
        TrophyBonus::ReputationBoost(v) => format!("+{:.0}% reputation gain", v * 100.0),
        TrophyBonus::CombatBoost(v) => format!("+{:.0}% combat", v * 100.0),
        TrophyBonus::GoldBoost(v) => format!("+{:.0}% gold", v * 100.0),
        TrophyBonus::XpBoost(v) => format!("+{:.0}% XP", v * 100.0),
    }
}
