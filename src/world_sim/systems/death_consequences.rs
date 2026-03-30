#![allow(unused)]
//! Death consequences — funerals, mourning, inheritance, and memorials.
//!
//! When an NPC dies:
//! 1. **Funeral** at the settlement's Temple (or settlement center) for 20 ticks
//!    - Nearby NPCs attend, get grief + social satisfaction
//!    - Chronicle entry for notable NPCs
//! 2. **Mourning period** — friends (from memory) get grief spike + social need
//! 3. **Inheritance** — gold distributed to family (home building residents)
//!    or settlement treasury
//! 4. **Memorial** — notable NPCs (level 20+, 5+ chronicle mentions) get
//!    a permanent chronicle Narrative entry
//!
//! Death tracking: we monitor world_events for EntityDied this tick.
//!
//! Cadence: every tick (check for new deaths), funerals run for 20 ticks.

use crate::world_sim::state::*;
use crate::world_sim::systems::agent_inner::record_npc_event;

/// How long a funeral lasts (ticks).
const FUNERAL_DURATION: u16 = 20;
/// Fraction of gold inherited by home building co-residents.
const INHERITANCE_FRACTION: f32 = 0.7;
/// Minimum level for a memorial chronicle entry.
const MEMORIAL_LEVEL: u32 = 10;

/// Track active funerals. Stored on WorldState would be ideal, but for now
/// we use a simple approach: detect deaths from world_events and process
/// consequences immediately.

/// Process death consequences. Called post-apply from runtime.rs.
pub fn advance_death_consequences(state: &mut WorldState) {
    let tick = state.tick;

    // Collect deaths that happened this tick from world_events.
    let recent_deaths: Vec<u32> = state.world_events.iter()
        .filter_map(|e| match e {
            WorldEvent::EntityDied { entity_id, .. } => Some(*entity_id),
            _ => None,
        })
        .collect();

    if recent_deaths.is_empty() { return; }

    // Deduplicate: only process each death ONCE by checking if we already
    // have a funeral chronicle entry for this entity.
    let already_processed: Vec<u32> = state.chronicle.iter()
        .filter(|e| e.category == ChronicleCategory::Death && !e.entity_ids.is_empty())
        .map(|e| e.entity_ids[0])
        .collect();

    let recent_deaths: Vec<u32> = recent_deaths.into_iter()
        .filter(|id| !already_processed.contains(id))
        .collect();

    if recent_deaths.is_empty() { return; }

    // Cap to 5 deaths per tick to avoid chronicle spam.
    let deaths_to_process = recent_deaths.len().min(5);

    for &dead_id in &recent_deaths[..deaths_to_process] {
        let dead_info = state.entities.iter()
            .find(|e| e.id == dead_id && !e.alive)
            .map(|e| {
                let name = e.npc.as_ref().map(|n| n.name.clone()).unwrap_or_default();
                let level = e.level;
                let gold = e.npc.as_ref().map(|n| n.gold).unwrap_or(0.0);
                let home_sid = e.npc.as_ref().and_then(|n| n.home_settlement_id);
                let home_bid = e.npc.as_ref().and_then(|n| n.home_building_id);
                let pos = e.pos;
                let is_npc = e.kind == EntityKind::Npc;
                (name, level, gold, home_sid, home_bid, pos, is_npc)
            });

        let (name, level, gold, home_sid, home_bid, death_pos, is_npc) = match dead_info {
            Some(info) => info,
            None => continue,
        };

        if !is_npc || name.is_empty() { continue; }

        // --- 1. Inheritance: distribute gold ---
        if gold > 1.0 {
            let mut inherited = false;
            let inherit_amount = gold * INHERITANCE_FRACTION;

            // Try to give to home building co-residents.
            if let Some(hbid) = home_bid {
                let co_residents: Vec<usize> = state.entities.iter().enumerate()
                    .filter(|(_, e)| {
                        e.alive && e.kind == EntityKind::Npc && e.id != dead_id
                            && e.npc.as_ref().map(|n| n.home_building_id == Some(hbid)).unwrap_or(false)
                    })
                    .map(|(i, _)| i)
                    .collect();

                if !co_residents.is_empty() {
                    let share = inherit_amount / co_residents.len() as f32;
                    for &ci in &co_residents {
                        if let Some(npc) = state.entities[ci].npc.as_mut() {
                            npc.gold += share;
                        }
                    }
                    inherited = true;
                }
            }

            // Remainder (or all if no co-residents) goes to settlement treasury.
            let treasury_amount = if inherited { gold * (1.0 - INHERITANCE_FRACTION) } else { gold };
            if let Some(sid) = home_sid {
                if let Some(settlement) = state.settlements.iter_mut().find(|s| s.id == sid) {
                    settlement.treasury += treasury_amount;
                }
            }
        }

        // --- 2. Mourning: grief for friends at same settlement ---
        if let Some(sid) = home_sid {
            let mut mourners = 0u32;
            for entity in &mut state.entities {
                if !entity.alive || entity.kind != EntityKind::Npc || entity.id == dead_id { continue; }
                let npc = match &mut entity.npc { Some(n) => n, None => continue };
                if npc.home_settlement_id != Some(sid) { continue; }

                // Check if the dead NPC was a friend (in memory).
                let was_friend = npc.memory.events.iter().any(|e| {
                    matches!(&e.event_type, MemEventType::MadeNewFriend(fid) if *fid == dead_id)
                });

                if was_friend {
                    // Deep grief for friends.
                    npc.emotions.grief = (npc.emotions.grief + 0.7).min(1.0);
                    npc.emotions.anger = (npc.emotions.anger + 0.2).min(1.0);
                    npc.needs.social -= 20.0;
                    npc.needs.social = npc.needs.social.max(0.0);
                    mourners += 1;
                } else {
                    // Mild grief for settlement-mates.
                    npc.emotions.grief = (npc.emotions.grief + 0.2).min(1.0);
                    npc.needs.social -= 5.0;
                    npc.needs.social = npc.needs.social.max(0.0);
                }

                // Push social need (funeral attendance satisfies social).
                npc.needs.social = (npc.needs.social + 10.0).min(80.0);
            }

            // --- 2b. Apprentice Lineage: closest friend inherits 30% of behavior tags ---
            {
                // Get the dead NPC's behavior profile.
                let dead_profile: Vec<(u32, f32)> = state.entities.iter()
                    .find(|e| e.id == dead_id)
                    .and_then(|e| e.npc.as_ref())
                    .map(|n| n.behavior_profile.clone())
                    .unwrap_or_default();

                let dead_lineage: Vec<u32> = state.entities.iter()
                    .find(|e| e.id == dead_id)
                    .and_then(|e| e.npc.as_ref())
                    .map(|n| n.mentor_lineage.clone())
                    .unwrap_or_default();

                if !dead_profile.is_empty() {
                    // Find closest friend at same settlement (best apprentice candidate).
                    let mut best_friend: Option<usize> = None;
                    let mut best_friend_score = 0.0f32;

                    for (ei, entity) in state.entities.iter().enumerate() {
                        if !entity.alive || entity.kind != EntityKind::Npc || entity.id == dead_id { continue; }
                        let npc = match &entity.npc { Some(n) => n, None => continue };
                        if npc.home_settlement_id != Some(sid) { continue; }

                        // Score: friend memory count + lower level = better apprentice.
                        let friend_events = npc.memory.events.iter()
                            .filter(|e| matches!(&e.event_type, MemEventType::MadeNewFriend(fid) if *fid == dead_id))
                            .count() as f32;
                        if friend_events > 0.0 {
                            let level_bonus = 1.0 / (1.0 + entity.level as f32 * 0.1);
                            let score = friend_events * level_bonus;
                            if score > best_friend_score {
                                best_friend_score = score;
                                best_friend = Some(ei);
                            }
                        }
                    }

                    // Inherit 30% of behavior tags to best apprentice.
                    if let Some(heir_idx) = best_friend {
                        let heir_name = state.entities[heir_idx].npc.as_ref()
                            .map(|n| n.name.clone()).unwrap_or_default();
                        let npc = state.entities[heir_idx].npc.as_mut().unwrap();

                        for &(tag, value) in &dead_profile {
                            let inherit = value * 0.3;
                            if inherit > 1.0 {
                                let mut action = ActionTags::empty();
                                action.tags[0] = (tag, inherit);
                                action.count = 1;
                                npc.accumulate_tags(&action);
                            }
                        }

                        // Record lineage chain.
                        npc.mentor_lineage.insert(0, dead_id);
                        npc.mentor_lineage.extend_from_slice(&dead_lineage);
                        npc.mentor_lineage.truncate(5); // max 5 deep

                        // Chronicle lineage entry.
                        state.chronicle.push(ChronicleEntry {
                            tick,
                            category: ChronicleCategory::Achievement,
                            text: format!("{} carries on the legacy of {}.", heir_name, name),
                            entity_ids: vec![state.entities[heir_idx].id, dead_id],
                        });
                    }
                }
            }

            // --- 3. Funeral chronicle entry ---
            let settlement_name = state.settlements.iter()
                .find(|s| s.id == sid)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "the wilderness".into());

            if level >= 5 {
                // Generate epitaph from life story.
                let epitaph = generate_epitaph(state, dead_id, &name, level);

                let mourner_text = if mourners > 0 {
                    format!(" {} friends mourned.", mourners)
                } else {
                    String::new()
                };
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Death,
                    text: format!(
                        "A funeral was held in {} for {} (level {}).{} {}",
                        settlement_name, name, level, mourner_text, epitaph
                    ),
                    entity_ids: vec![dead_id],
                });
            }

            // --- 4. Memorial for notable NPCs ---
            if level >= MEMORIAL_LEVEL {
                let mention_count = state.chronicle.iter()
                    .filter(|e| e.entity_ids.contains(&dead_id))
                    .count();

                if mention_count >= 3 {
                    // Class summary for the memorial.
                    let class_summary = state.entities.iter()
                        .find(|e| e.id == dead_id)
                        .and_then(|e| e.npc.as_ref())
                        .map(|npc| {
                            if npc.classes.is_empty() { return "an unclassed soul".to_string(); }
                            npc.classes.iter()
                                .max_by_key(|c| c.level)
                                .map(|c| format!("a {} (L{})", c.display_name, c.level))
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();

                    state.chronicle.push(ChronicleEntry {
                        tick,
                        category: ChronicleCategory::Narrative,
                        text: format!(
                            "A memorial stands in {} for {}, {}, who featured in {} chronicle entries. Their deeds will not be forgotten.",
                            settlement_name, name, class_summary, mention_count
                        ),
                        entity_ids: vec![dead_id],
                    });
                }
            }
        }
    }
}

/// Generate a one-sentence epitaph from an NPC's life story.
fn generate_epitaph(state: &WorldState, entity_id: u32, name: &str, level: u32) -> String {
    let entity = match state.entities.iter().find(|e| e.id == entity_id) {
        Some(e) => e,
        None => return String::new(),
    };
    let npc = match &entity.npc {
        Some(n) => n,
        None => return String::new(),
    };

    let mut parts: Vec<String> = Vec::new();

    // Class identity.
    if let Some(best_class) = npc.classes.iter().max_by_key(|c| c.level) {
        parts.push(format!("{}, {} L{}", name, best_class.display_name, best_class.level));
    }

    // Most dramatic memory.
    let best_memory = npc.memory.events.iter()
        .max_by(|a, b| a.emotional_impact.abs().partial_cmp(&b.emotional_impact.abs())
            .unwrap_or(std::cmp::Ordering::Equal));
    if let Some(mem) = best_memory {
        let memory_text = match &mem.event_type {
            MemEventType::WonFight => "won their greatest battle".into(),
            MemEventType::WasAttacked => "survived terrible wounds".into(),
            MemEventType::FriendDied(fid) => {
                let friend_name = state.entities.iter()
                    .find(|e| e.id == *fid)
                    .and_then(|e| e.npc.as_ref())
                    .map(|n| n.name.clone())
                    .unwrap_or_else(|| format!("a fallen comrade"));
                format!("mourned {}", friend_name)
            }
            MemEventType::CompletedQuest => "completed a legendary quest".into(),
            MemEventType::Starved => "endured famine".into(),
            MemEventType::BuiltSomething => "built something lasting".into(),
            MemEventType::MadeNewFriend(fid) => {
                let friend_name = state.entities.iter()
                    .find(|e| e.id == *fid)
                    .and_then(|e| e.npc.as_ref())
                    .map(|n| n.name.clone())
                    .unwrap_or_else(|| "a dear friend".into());
                format!("cherished {}", friend_name)
            }
            MemEventType::LearnedSkill => "mastered a rare skill".into(),
            _ => String::new(),
        };
        if !memory_text.is_empty() {
            parts.push(memory_text);
        }
    }

    // Family.
    if !npc.children.is_empty() {
        let alive_children = npc.children.iter()
            .filter(|&&cid| state.entities.iter().any(|e| e.id == cid && e.alive))
            .count();
        if alive_children > 0 {
            parts.push(format!("left behind {} {}", alive_children,
                if alive_children == 1 { "child" } else { "children" }));
        }
    }
    if let Some(spouse_id) = npc.spouse_id {
        let spouse_name = state.entities.iter()
            .find(|e| e.id == spouse_id && e.alive)
            .and_then(|e| e.npc.as_ref())
            .map(|n| n.name.clone());
        if let Some(sname) = spouse_name {
            parts.push(format!("beloved of {}", sname));
        }
    }

    // Grudges.
    let grudge_count = npc.memory.beliefs.iter()
        .filter(|b| matches!(b.belief_type, BeliefType::Grudge(_)))
        .count();
    if grudge_count >= 3 {
        parts.push(format!("carried {} unresolved grudges", grudge_count));
    }

    if parts.is_empty() {
        return String::new();
    }

    // Join into a flowing sentence.
    let joined = if parts.len() == 1 {
        format!("They {}.", parts[0])
    } else {
        let last = parts.pop().unwrap();
        format!("They {}, and {}.", parts.join(", "), last)
    };

    joined
}
