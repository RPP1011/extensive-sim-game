#![allow(unused)]
//! Social gathering — NPCs meet at taverns/temples for multi-tick conversations.
//!
//! NPCs with low social need and a Socialize goal walk to the nearest social
//! building (Inn, Temple, Market, GuildHall). When two+ NPCs are at the same
//! building, they start a conversation (visible as NpcAction::Socializing).
//! Conversations last 10-20 ticks and satisfy social need, build friendships,
//! and occasionally form beliefs about each other.
//!
//! Cadence: every 5 ticks (post-apply, needs &mut WorldState).

use crate::world_sim::state::*;
use crate::world_sim::systems::agent_inner::record_npc_event;

const SOCIAL_INTERVAL: u64 = 5;
/// Maximum distance between two NPCs to count as "at the same building".
const CONVERSATION_DIST_SQ: f32 = 100.0; // 10 units
/// How many ticks a conversation lasts.
const CONVERSATION_TICKS: u8 = 15;
/// Social need restored per conversation.
const SOCIAL_RESTORE: f32 = 25.0;
/// Chance per pair per check to start a conversation (if both available).
const CONVERSATION_CHANCE: f32 = 0.3;

/// Process social gatherings. Called post-apply from runtime.rs.
pub fn advance_social_gatherings(state: &mut WorldState) {
    if state.tick % SOCIAL_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Collect NPCs that are socializing (have Socialize goal and are near a social building).
    // We need: entity_index, entity_id, position, is_currently_socializing.
    let mut social_npcs: Vec<(usize, u32, (f32, f32), bool)> = Vec::new();

    for (i, entity) in state.entities.iter().enumerate() {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Must have Socialize goal active (or at least low social need).
        let is_socializing = matches!(npc.action, NpcAction::Socializing { .. });
        let wants_social = npc.goal_stack.has(&GoalKind::Socialize) || npc.needs.social < 40.0;

        if is_socializing || wants_social {
            social_npcs.push((i, entity.id, entity.pos, is_socializing));
        }
    }

    if social_npcs.len() < 2 { return; }

    // Find pairs of NPCs close enough to converse.
    let mut conversations: Vec<(usize, usize)> = Vec::new(); // (entity_idx_a, entity_idx_b)

    for i in 0..social_npcs.len() {
        for j in (i + 1)..social_npcs.len() {
            if conversations.len() >= 20 { break; } // cap per tick

            let (idx_a, id_a, pos_a, already_a) = social_npcs[i];
            let (idx_b, id_b, pos_b, already_b) = social_npcs[j];

            // Already socializing with each other? Skip (let it continue).
            if already_a || already_b { continue; }

            // Close enough?
            let dx = pos_a.0 - pos_b.0;
            let dy = pos_a.1 - pos_b.1;
            if dx * dx + dy * dy > CONVERSATION_DIST_SQ { continue; }

            // Deterministic chance check.
            let roll = entity_hash_f32(id_a, tick, id_b as u64);
            if roll > CONVERSATION_CHANCE { continue; }

            conversations.push((idx_a, idx_b));
        }
    }

    // Start conversations: update actions, needs, memories.
    for (idx_a, idx_b) in conversations {
        let id_a = state.entities[idx_a].id;
        let id_b = state.entities[idx_b].id;
        let pos_a = state.entities[idx_a].pos;

        // Set both NPCs to Socializing action.
        if let Some(npc_a) = state.entities[idx_a].npc.as_mut() {
            npc_a.action = NpcAction::Socializing {
                partner_id: id_b,
                ticks_remaining: CONVERSATION_TICKS,
            };
            npc_a.needs.social = (npc_a.needs.social + SOCIAL_RESTORE).min(100.0);
            // Record friendship event.
            record_npc_event(npc_a, MemEventType::MadeNewFriend(id_b),
                pos_a, vec![id_b], 0.4, tick);
        }

        if let Some(npc_b) = state.entities[idx_b].npc.as_mut() {
            npc_b.action = NpcAction::Socializing {
                partner_id: id_a,
                ticks_remaining: CONVERSATION_TICKS,
            };
            npc_b.needs.social = (npc_b.needs.social + SOCIAL_RESTORE).min(100.0);
            record_npc_event(npc_b, MemEventType::MadeNewFriend(id_a),
                pos_a, vec![id_a], 0.4, tick);
        }

        // Build relationship strength.
        // (Using behavior tags as a proxy for social bonding.)
        if let Some(npc_a) = state.entities[idx_a].npc.as_mut() {
            let mut tags = ActionTags::empty();
            tags.add(tags::DIPLOMACY, 0.5);
            tags.add(tags::LEADERSHIP, 0.2);
            npc_a.accumulate_tags(&tags);
        }
        if let Some(npc_b) = state.entities[idx_b].npc.as_mut() {
            let mut tags = ActionTags::empty();
            tags.add(tags::DIPLOMACY, 0.5);
            tags.add(tags::LEADERSHIP, 0.2);
            npc_b.accumulate_tags(&tags);
        }

        // --- Tavern Stories: share most dramatic memory ---
        // A shares their best story with B, and vice versa.
        // Creates HeardStory beliefs about the story's subject.

        // A's best story → B hears it.
        let story_a = state.entities[idx_a].npc.as_ref()
            .and_then(|n| n.memory.events.iter()
                .filter(|e| matches!(e.event_type,
                    MemEventType::WonFight | MemEventType::CompletedQuest
                    | MemEventType::FriendDied(_) | MemEventType::WasAttacked
                    | MemEventType::BuiltSomething | MemEventType::LearnedSkill))
                .max_by(|a, b| a.emotional_impact.abs().partial_cmp(&b.emotional_impact.abs())
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| (id_a, e.emotional_impact.abs()))
            );

        if let Some((about_id, impact)) = story_a {
            if impact > 0.3 { // only share dramatic stories
                if let Some(npc_b) = state.entities[idx_b].npc.as_mut() {
                    // Don't duplicate stories.
                    let already_heard = npc_b.memory.beliefs.iter().any(|b|
                        matches!(&b.belief_type, BeliefType::HeardStory { about } if *about == about_id)
                    );
                    if !already_heard {
                        npc_b.memory.beliefs.push(Belief {
                            belief_type: BeliefType::HeardStory { about: about_id },
                            confidence: impact.min(1.0),
                            formed_tick: tick,
                        });
                        // Storytelling tags for the teller.
                        if let Some(npc_a) = state.entities[idx_a].npc.as_mut() {
                            let mut tags = ActionTags::empty();
                            tags.add(tags::TEACHING, 1.0);  // storytelling = teaching
                            tags.add(tags::DIPLOMACY, 0.5);
                            npc_a.accumulate_tags(&tags);
                        }
                    }
                }
            }
        }

        // B's best story → A hears it.
        let story_b = state.entities[idx_b].npc.as_ref()
            .and_then(|n| n.memory.events.iter()
                .filter(|e| matches!(e.event_type,
                    MemEventType::WonFight | MemEventType::CompletedQuest
                    | MemEventType::FriendDied(_) | MemEventType::WasAttacked
                    | MemEventType::BuiltSomething | MemEventType::LearnedSkill))
                .max_by(|a, b| a.emotional_impact.abs().partial_cmp(&b.emotional_impact.abs())
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| (id_b, e.emotional_impact.abs()))
            );

        if let Some((about_id, impact)) = story_b {
            if impact > 0.3 {
                if let Some(npc_a) = state.entities[idx_a].npc.as_mut() {
                    let already_heard = npc_a.memory.beliefs.iter().any(|b|
                        matches!(&b.belief_type, BeliefType::HeardStory { about } if *about == about_id)
                    );
                    if !already_heard {
                        npc_a.memory.beliefs.push(Belief {
                            belief_type: BeliefType::HeardStory { about: about_id },
                            confidence: impact.min(1.0),
                            formed_tick: tick,
                        });
                        if let Some(npc_b) = state.entities[idx_b].npc.as_mut() {
                            let mut tags = ActionTags::empty();
                            tags.add(tags::TEACHING, 1.0);
                            tags.add(tags::DIPLOMACY, 0.5);
                            npc_b.accumulate_tags(&tags);
                        }
                    }
                }
            }
        }
    }

    // Tick down existing conversations.
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        if let NpcAction::Socializing { ticks_remaining, .. } = &mut npc.action {
            if *ticks_remaining > 0 {
                *ticks_remaining -= 1;
            } else {
                // Conversation complete — clear action, remove Socialize goal.
                npc.action = NpcAction::Idle;
                npc.goal_stack.remove_kind(&GoalKind::Socialize);
            }
        }
    }
}
