//! Progression system — triggers on quest completion.
//!
//! Evaluates playstyle from recent history and potentially grants an unlock.
//! In the full game, the LFM would judge playstyle. Here we use simple
//! heuristics as a placeholder.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Minimum quests completed before first unlock.
const UNLOCK_THRESHOLD: usize = 3;
/// Quests between unlock checks.
const UNLOCK_INTERVAL: usize = 5;

pub fn tick_progression(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let total = state.completed_quests.len();
    if total < UNLOCK_THRESHOLD {
        return;
    }
    if total % UNLOCK_INTERVAL != 0 {
        return;
    }
    // Only trigger once per quest count milestone
    let milestone_tag = format!("quests_{}", total);
    if state
        .progression_history
        .iter()
        .any(|p| p.description.starts_with(&milestone_tag))
    {
        return;
    }

    // Simple playstyle detection from recent quests
    let recent = &state.completed_quests[total.saturating_sub(5)..];
    let combat_count = recent
        .iter()
        .filter(|q| matches!(q.quest_type, QuestType::Combat))
        .count();
    let explore_count = recent
        .iter()
        .filter(|q| matches!(q.quest_type, QuestType::Exploration))
        .count();

    let (category, name, desc) = if combat_count >= 3 {
        (
            UnlockCategory::PassiveBuff,
            "Battle Hardened",
            "Party members deal 10% more damage",
        )
    } else if explore_count >= 3 {
        (
            UnlockCategory::Information,
            "Scout Network",
            "Automatically reveals nearby locations",
        )
    } else {
        (
            UnlockCategory::Economic,
            "Trade Routes",
            "5% reduction in all costs",
        )
    };

    let id = state.next_unlock_id;
    state.next_unlock_id += 1;

    let unlock = UnlockInstance {
        id,
        category,
        properties: UnlockProperties {
            cooldown_ms: if category == UnlockCategory::ActiveAbility {
                30_000
            } else {
                0
            },
            target_type: TargetType::GuildSelf,
            magnitude: 0.1,
            duration_ms: 0,
            resource_cost: 0.0,
            is_passive: !matches!(category, UnlockCategory::ActiveAbility),
            category_embedding: [0.0; 8],
        },
        name: name.into(),
        description: desc.into(),
        active: true,
        cooldown_remaining_ms: 0,
    };

    events.push(WorldEvent::ProgressionUnlocked {
        unlock_id: id,
        category,
        name: name.into(),
    });

    state.progression_history.push(ProgressionEvent {
        tick: state.tick,
        unlock_id: id,
        description: format!("quests_{}: {}", total, desc),
    });

    state.unlocks.push(unlock);
}
