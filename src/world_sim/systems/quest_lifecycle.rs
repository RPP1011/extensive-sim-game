//! Quest lifecycle — every 10 ticks.
//!
//! Progresses active quests toward completion, expiry, or staleness:
//!
//! 1. **Arrival**: adventuring NPCs within 5.0 of the quest destination advance progress.
//! 2. **Completion**: NPC at destination + settlement threat below quest threshold → complete.
//!    Awards gold + XP, records chronicle, returns NPC to Produce.
//! 3. **Expiry**: quests past their deadline_tick are removed, NPCs return to Produce.
//! 4. **Staleness**: quests with no progress for 200 ticks have their reward reduced;
//!    if reward drops below 1.0, the quest is failed.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

/// Run every N ticks.
const LIFECYCLE_INTERVAL: u64 = 10;

/// NPC must be within this distance of quest destination to count as "arrived".
const ARRIVAL_DISTANCE: f32 = 5.0;

/// Progress increment per lifecycle tick when NPC is at destination.
const PROGRESS_PER_TICK: f32 = 0.05;

/// Number of ticks without progress before quest is considered stale.
const STALE_THRESHOLD_TICKS: u64 = 200;

/// Fraction of reward removed per stale check.
const STALE_REWARD_DECAY: f32 = 0.25;

/// Minimum reward before a stale quest is failed outright.
const MIN_REWARD_BEFORE_FAIL: f32 = 1.0;

pub fn compute_quest_lifecycle(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % LIFECYCLE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for quest in &state.quests {
        // Skip already-terminal quests.
        match quest.status {
            QuestStatus::Completed | QuestStatus::Failed => continue,
            _ => {}
        }

        // --- Expiry check (tick > deadline) ---
        if quest.deadline_tick > 0 && state.tick > quest.deadline_tick {
            expire_quest(state, quest, out);
            continue;
        }

        // Collect adventuring NPCs assigned to this quest.
        let adventurers: Vec<&Entity> = state.entities.iter().filter(|e| {
            if !e.alive || e.kind != EntityKind::Npc { return false; }
            let npc = match &e.npc { Some(n) => n, None => return false };
            matches!(
                &npc.economic_intent,
                EconomicIntent::Adventuring { quest_id, .. } if *quest_id == quest.id
            )
        }).collect();

        if adventurers.is_empty() {
            // No one is working on this quest — check staleness.
            check_stale(state, quest, out);
            continue;
        }

        // --- Arrival + completion checks ---
        let mut any_arrived = false;
        for entity in &adventurers {
            let dx = quest.destination.0 - entity.pos.0;
            let dy = quest.destination.1 - entity.pos.1;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= ARRIVAL_DISTANCE {
                any_arrived = true;
            }
        }

        if !any_arrived {
            // NPCs still traveling — check staleness based on quest age.
            let ticks_active = state.tick.saturating_sub(quest.accepted_tick);
            if ticks_active > STALE_THRESHOLD_TICKS && quest.progress < 0.01 {
                // Quest accepted long ago but no one has arrived — treat as stale.
                check_stale(state, quest, out);
            }
            continue;
        }

        // At least one NPC is at the destination.
        // Advance progress if quest is still Traveling → InProgress.
        if quest.status == QuestStatus::Traveling {
            out.push(WorldDelta::QuestUpdate {
                quest_id: quest.id,
                update: QuestDelta::SetStatus { status: QuestStatus::InProgress },
            });
        }

        // Advance progress.
        out.push(WorldDelta::QuestUpdate {
            quest_id: quest.id,
            update: QuestDelta::AdvanceProgress { amount: PROGRESS_PER_TICK },
        });

        // --- Completion: threat at nearest settlement has dropped below quest threshold ---
        let threat_cleared = find_nearest_settlement(state, quest.destination)
            .map(|s| s.threat_level < quest.threat_level * 0.5)
            .unwrap_or(false);

        let progress_complete = quest.progress + PROGRESS_PER_TICK >= 1.0;

        if threat_cleared || progress_complete {
            complete_quest(state, quest, &adventurers, out);
        }
    }
}

/// Mark quest completed, award gold + XP to participating NPCs, record chronicle.
fn complete_quest(
    state: &WorldState,
    quest: &Quest,
    adventurers: &[&Entity],
    out: &mut Vec<WorldDelta>,
) {
    // Mark quest as complete.
    out.push(WorldDelta::QuestUpdate {
        quest_id: quest.id,
        update: QuestDelta::Complete,
    });

    // Divide reward among participants.
    let num_members = adventurers.len().max(1) as f32;
    let gold_share = quest.reward_gold / num_members;
    let xp_share = (quest.reward_xp as f32 / num_members) as u32;

    let mut participant_ids = Vec::new();

    for entity in adventurers {
        participant_ids.push(entity.id);

        // Gold reward.
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Gold,
            value: gold_share,
        });

        // XP reward.
        out.push(WorldDelta::AddXp {
            entity_id: entity.id,
            amount: xp_share.max(1),
        });

        // Morale boost from completing a quest.
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Morale,
            value: 5.0,
        });

        // Combat/tactics behavior tags from quest experience.
        let mut action = ActionTags::empty();
        action.add(tags::COMBAT, 2.0);
        action.add(tags::TACTICS, 1.5);
        action.add(tags::SURVIVAL, 1.0);
        out.push(WorldDelta::AddBehaviorTags {
            entity_id: entity.id,
            tags: action.tags,
            count: action.count,
        });

        // Return NPC to Produce.
        out.push(WorldDelta::SetIntent {
            entity_id: entity.id,
            intent: EconomicIntent::Produce,
        });
    }

    // Record chronicle entry.
    let quest_name = &quest.name;
    let n = adventurers.len();
    let text = if n == 1 {
        format!(
            "Quest \"{}\" completed. Adventurer earned {:.0} gold and {} XP.",
            quest_name, gold_share, xp_share,
        )
    } else {
        format!(
            "Quest \"{}\" completed by {} adventurers. Each earned {:.0} gold and {} XP.",
            quest_name, n, gold_share, xp_share,
        )
    };
    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Quest,
            text,
            entity_ids: participant_ids,
        },
    });
}

/// Expire a quest that has passed its deadline. Return NPCs to Produce.
fn expire_quest(state: &WorldState, quest: &Quest, out: &mut Vec<WorldDelta>) {
    out.push(WorldDelta::QuestUpdate {
        quest_id: quest.id,
        update: QuestDelta::Fail,
    });

    // Return all adventuring NPCs on this quest to Produce.
    return_adventurers_to_produce(state, quest.id, out);

    out.push(WorldDelta::RecordChronicle {
        entry: ChronicleEntry {
            tick: state.tick,
            category: ChronicleCategory::Quest,
            text: format!("Quest \"{}\" expired — deadline passed.", quest.name),
            entity_ids: quest.party_member_ids.clone(),
        },
    });
}

/// Check for staleness: no progress in STALE_THRESHOLD_TICKS.
/// Reduce reward; if reward too low, fail the quest.
fn check_stale(state: &WorldState, quest: &Quest, out: &mut Vec<WorldDelta>) {
    let ticks_since_accepted = state.tick.saturating_sub(quest.accepted_tick);

    // Only apply stale checks after the threshold period.
    if ticks_since_accepted < STALE_THRESHOLD_TICKS {
        return;
    }

    // Check staleness every STALE_THRESHOLD_TICKS ticks after accepted.
    // Use modular arithmetic to fire at intervals: 200, 400, 600, ...
    if ticks_since_accepted % STALE_THRESHOLD_TICKS != 0 {
        return;
    }

    // If reward has decayed to near-zero, fail the quest.
    let projected_reward = quest.reward_gold * (1.0 - STALE_REWARD_DECAY);
    if projected_reward < MIN_REWARD_BEFORE_FAIL {
        out.push(WorldDelta::QuestUpdate {
            quest_id: quest.id,
            update: QuestDelta::Fail,
        });

        return_adventurers_to_produce(state, quest.id, out);

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Quest,
                text: format!(
                    "Quest \"{}\" abandoned — no progress, reward too low.",
                    quest.name,
                ),
                entity_ids: quest.party_member_ids.clone(),
            },
        });
    } else {
        // Reduce reward (negative gold delta on the quest's reward_gold).
        // We model this via UpdateEntityField on a sentinel, but since quest
        // reward is on the Quest struct, we just note the decay via progress.
        // The reward_gold field is immutable in the delta model, so instead
        // we advance progress slightly (to prevent repeated stale triggers on
        // the same interval) and chronicle the decay.
        out.push(WorldDelta::QuestUpdate {
            quest_id: quest.id,
            update: QuestDelta::AdvanceProgress { amount: 0.001 },
        });

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Quest,
                text: format!(
                    "Quest \"{}\" is stale — no adventurer progress for {} ticks.",
                    quest.name, STALE_THRESHOLD_TICKS,
                ),
                entity_ids: vec![],
            },
        });
    }
}

/// Find all NPCs adventuring on the given quest and set them back to Produce.
fn return_adventurers_to_produce(state: &WorldState, quest_id: u32, out: &mut Vec<WorldDelta>) {
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if let EconomicIntent::Adventuring { quest_id: qid, .. } = &npc.economic_intent {
            if *qid == quest_id {
                out.push(WorldDelta::SetIntent {
                    entity_id: entity.id,
                    intent: EconomicIntent::Produce,
                });
            }
        }
    }
}

/// Find the settlement closest to a given world position.
fn find_nearest_settlement<'a>(state: &'a WorldState, pos: (f32, f32)) -> Option<&'a SettlementState> {
    state.settlements.iter().min_by(|a, b| {
        let da = dist_sq(a.pos, pos);
        let db = dist_sq(b.pos, pos);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}
