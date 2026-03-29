//! Progression system — triggers on quest completion.
//!
//! Evaluates playstyle from recent history and potentially grants an unlock.
//! In the full game, the LFM would judge playstyle. Here we use simple
//! heuristics as a placeholder.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

pub fn tick_progression(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.progression;
    let total = state.completed_quests.len();
    if total < cfg.unlock_threshold {
        return;
    }
    if total % cfg.unlock_interval != 0 {
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
    let recent = &state.completed_quests[total.saturating_sub(cfg.recent_quest_window)..];
    let combat_count = recent
        .iter()
        .filter(|q| matches!(q.quest_type, QuestType::Combat))
        .count();
    let explore_count = recent
        .iter()
        .filter(|q| matches!(q.quest_type, QuestType::Exploration))
        .count();

    let themed_count = cfg.themed_unlock_count;
    let (category, name, desc) = if combat_count >= themed_count {
        (
            UnlockCategory::PassiveBuff,
            "Battle Hardened",
            "Party members deal 10% more damage",
        )
    } else if explore_count >= themed_count {
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
                state.config.progression.active_ability_cooldown_ms
            } else {
                0
            },
            target_type: TargetType::GuildSelf,
            magnitude: state.config.progression.default_magnitude,
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
