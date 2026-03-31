//! Quest system — settlements post quests, NPCs accept and complete them.
//!
//! Flow:
//! 1. Settlements with high threat post "clear threat" quests (every 50 ticks)
//! 2. Combat-capable NPCs evaluate quests and accept the best one
//! 3. Accepted NPCs switch intent to Adventuring, move to quest destination
//! 4. At destination, NPCs on High-fidelity grids fight (handled by battles.rs)
//! 5. Quest completes when threat drops or timer expires → NPC gets reward
//!
//! **Gold conservation:** Quest rewards are paid from the posting settlement's
//! treasury. If the settlement cannot afford it, no gold is paid.
//!
//! Cadence: generation every 50 ticks, lifecycle every tick.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::*;

const QUEST_GEN_INTERVAL: u64 = 50;
const QUEST_LIFETIME_TICKS: u64 = 500;
const MIN_THREAT_FOR_QUEST: f32 = 0.3; // threat_level is 0.0-1.0
const QUEST_REWARD_PER_THREAT: f32 = 100.0; // scale up since threat is fractional
const QUEST_XP_PER_THREAT: f32 = 50.0;
const MIN_LEVEL_FOR_QUEST: u32 = 2;
const MAX_ACTIVE_QUESTS: usize = 20;



pub fn compute_quests(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // --- Phase 1: Quest generation (every 50 ticks) ---
    if state.tick % QUEST_GEN_INTERVAL == 0 && state.tick > 0 {
        generate_quests(state, out);
    }

    // --- Phase 2: Quest acceptance (NPCs evaluate available quests) ---
    if state.tick % QUEST_GEN_INTERVAL == 0 && state.tick > 0 {
        accept_quests(state, out);
    }

    // --- Phase 3: Quest lifecycle (every tick — move adventuring NPCs, check completion) ---
    run_quest_lifecycle(state, out);
}

pub fn compute_quests_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    // Quest gen and acceptance are global (read all settlements).
    // Per-settlement variant only handles lifecycle for NPCs at this settlement.
    run_quest_lifecycle_for_entities(state, settlement_id, entities, out);
}

/// Settlements with high threat post quests.
fn generate_quests(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.quest_board.len() + state.quests.len() >= MAX_ACTIVE_QUESTS {
        return;
    }

    for settlement in &state.settlements {
        if settlement.threat_level < MIN_THREAT_FOR_QUEST { continue; }
        if settlement.treasury < 10.0 { continue; } // can't afford to post

        // Don't post if there's already a quest for this settlement.
        let already_posted = state.quest_board.iter().any(|q| q.settlement_id == settlement.id)
            || state.quests.iter().any(|q| q.destination == settlement.pos);
        if already_posted { continue; }

        // Deterministic: only some settlements post per cycle.
        let roll = entity_hash_f32(settlement.id, state.tick, 0xBE57);
        if roll > 0.3 { continue; }

        let reward = settlement.threat_level * QUEST_REWARD_PER_THREAT;

        // Pay for quest posting from treasury.
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement.id,
            delta: -reward * 0.1, // 10% upfront cost
        });

        // Note: We can't push to quest_board directly (read-only snapshot).
        // Quest generation needs a SpawnQuest delta or runtime post-phase.
        // For now, emit a RecordEvent so the runtime can pick it up.
        out.push(WorldDelta::RecordEvent {
            event: WorldEvent::QuestPosted {
                settlement_id: settlement.id,
                threat_level: settlement.threat_level,
                reward_gold: reward,
            },
        });

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Quest,
                text: format!(
                    "{} posted a quest: clear threat (reward: {:.0} gold)",
                    settlement.name, reward,
                ),
                entity_ids: vec![],
            },
        });
    }
}

/// Combat-capable NPCs evaluate available quests and accept the best one.
fn accept_quests(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.quest_board.is_empty() { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);

        for entity in &state.entities[range] {
            if !entity.alive { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };

            // Only idle/producing NPCs consider quests.
            match &npc.economic_intent {
                EconomicIntent::Idle | EconomicIntent::Produce => {}
                _ => continue,
            }

            // Must be combat-capable (level >= MIN_LEVEL, has combat behavior).
            if entity.level < MIN_LEVEL_FOR_QUEST { continue; }
            let combat_skill = npc.behavior_value(tags::MELEE)
                + npc.behavior_value(tags::RANGED)
                + npc.behavior_value(tags::COMBAT);
            if combat_skill < 10.0 { continue; } // not a fighter

            // Risk aversion check.
            let risk_aversion = (1.0 + npc.stress * 0.005 - npc.resolve * 0.003).clamp(0.3, 3.0);

            // Find best quest (highest reward:risk ratio).
            let mut best_quest: Option<&QuestPosting> = None;
            let mut best_score = 0.0f32;

            for quest in &state.quest_board {
                if quest.expires_tick < state.tick { continue; }
                // Can this NPC handle the threat?
                let power = entity.attack_damage + entity.hp * 0.1;
                let difficulty = quest.threat_level * risk_aversion;
                if power < difficulty * 0.5 { continue; } // too dangerous

                let score = quest.reward_gold / (difficulty + 1.0);
                if score > best_score {
                    best_score = score;
                    best_quest = Some(quest);
                }
            }

            if let Some(quest) = best_quest {
                // Accept: change intent to Adventuring.
                out.push(WorldDelta::SetIntent {
                    entity_id: entity.id,
                    intent: EconomicIntent::Adventuring {
                        quest_id: quest.id,
                        destination: quest.destination,
                    },
                });

                // Movement toward quest destination is handled by goal system
                // setting entity.move_target from the Adventuring intent.

                // Behavior: taking on a quest.
                let mut action = ActionTags::empty();
                action.add(tags::COMBAT, 0.5);
                action.add(tags::SURVIVAL, 0.3);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: entity.id,
                    tags: action.tags,
                    count: action.count,
                });

                // Chronicle: NPC accepted a quest.
                let npc_name = entity_display_name(entity);
                let settlement_name = state.settlement(quest.settlement_id)
                    .map(|s| s.name.as_str())
                    .unwrap_or("unknown");
                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: ChronicleCategory::Quest,
                        text: format!(
                            "{} accepted a quest to clear threats near {}",
                            npc_name, settlement_name,
                        ),
                        entity_ids: vec![entity.id],
                    },
                });

                break; // one quest per NPC per decision cycle
            }
        }
    }
}

/// Move adventuring NPCs toward quest destinations, check for completion.
fn run_quest_lifecycle(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        run_quest_lifecycle_for_entities(state, settlement.id, &state.entities[range], out);
    }

    // Also handle unaffiliated adventuring NPCs.
    let unaffiliated = state.group_index.unaffiliated_entities();
    run_quest_lifecycle_for_entities(state, u32::MAX, &state.entities[unaffiliated], out);
}

fn run_quest_lifecycle_for_entities(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        let (quest_id, destination) = match &npc.economic_intent {
            EconomicIntent::Adventuring { quest_id, destination } => (*quest_id, *destination),
            _ => continue,
        };

        // Move toward destination.
        let dx = destination.0 - entity.pos.0;
        let dy = destination.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > 2.0 {
            // Still traveling — movement handled by move_target + advance_movement().
        } else {
            // At destination — check if quest is done.
            // Quest completes if:
            // 1. NPC has been at destination for a while (survived)
            // 2. Or no monsters nearby (threat cleared)
            //
            // Simple heuristic: if NPC is alive at destination and has been
            // fighting (combat behavior increased), the quest is progressing.
            // Complete after ~200 ticks at destination.

            let quest = state.quests.iter().find(|q| q.id == quest_id);
            let quest_posting = state.quest_board.iter().find(|q| q.id == quest_id);

            // Determine reward from either active quest or posting.
            let reward_gold = quest.map(|q| q.reward_gold)
                .or_else(|| quest_posting.map(|q| q.reward_gold))
                .unwrap_or(10.0);
            // Completion check: deterministic, ~5% chance per tick at destination.
            let roll = pair_hash_f32(entity.id, quest_id, state.tick, 0);
            if roll < 0.05 {
                // Quest reward paid from posting settlement's treasury.
                // Use quest posting's settlement, fall back to NPC's home settlement.
                let funding_sid = quest_posting.map(|q| q.settlement_id)
                    .or(npc.home_settlement_id)
                    .unwrap_or(settlement_id);
                let can_afford = state.settlement(funding_sid)
                    .map(|s| s.treasury > reward_gold)
                    .unwrap_or(false);

                if can_afford {
                    out.push(WorldDelta::TransferGold {
                        from_entity: funding_sid,
                        to_entity: entity.id,
                        amount: reward_gold,
                    });
                }
                // Combat behavior from quest completion.
                let mut action = ActionTags::empty();
                action.add(tags::COMBAT, 3.0);
                action.add(tags::SURVIVAL, 2.0);
                action.add(tags::RESILIENCE, 1.0);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: entity.id,
                    tags: action.tags,
                    count: action.count,
                });

                // Return to producing.
                out.push(WorldDelta::SetIntent {
                    entity_id: entity.id,
                    intent: EconomicIntent::Produce,
                });

                // Return movement toward home settlement is handled by
                // move_target set from the Produce intent by the goal system.
            }
        }
    }
}
