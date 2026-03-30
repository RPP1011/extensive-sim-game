//! Heroic legends — NPCs who achieve enough deeds become legendary figures.
//!
//! Legend status is earned, not assigned. An NPC becomes a Legend when they:
//! - Have 5+ chronicle mentions
//! - Have 2+ classes (multiclass hero)
//! - Have survived 3+ friend deaths (hardened by loss)
//!
//! Legends:
//! - Boost settlement morale (+5 for all NPCs at their settlement)
//! - Attract followers (NPCs with low purpose seek to be near them)
//! - Get a "Legend" title in their name
//! - Their death creates a massive mourning event with world-wide effects
//!
//! Cadence: every 500 ticks (legend detection), continuous (legend effects).

use crate::world_sim::state::*;

const LEGEND_CHECK_INTERVAL: u64 = 500;
const MIN_CHRONICLE_MENTIONS: usize = 5;
const MIN_CLASSES: usize = 2;
const MIN_FRIEND_DEATHS: usize = 3;
const LEGEND_MORALE_BOOST: f32 = 3.0;

pub fn advance_legends(state: &mut WorldState) {
    let tick = state.tick;

    // --- Phase 1: Detect new legends (every 500 ticks) ---
    if tick % LEGEND_CHECK_INTERVAL == 0 && tick > 0 {
        let mut new_legends: Vec<(usize, String)> = Vec::new();

        for (i, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };

            // Already a legend? (check for "the Legendary" in name)
            if npc.name.contains("the Legendary") { continue; }

            // Chronicle mention count.
            let mentions = state.chronicle.iter()
                .filter(|e| e.entity_ids.contains(&entity.id))
                .count();
            if mentions < MIN_CHRONICLE_MENTIONS { continue; }

            // Class count.
            if npc.classes.len() < MIN_CLASSES { continue; }

            // Friend death count in memory.
            let friend_deaths = npc.memory.events.iter()
                .filter(|e| matches!(e.event_type, MemEventType::FriendDied(_)))
                .count();
            if friend_deaths < MIN_FRIEND_DEATHS { continue; }

            new_legends.push((i, npc.name.clone()));
        }

        // Elevate new legends.
        for (idx, old_name) in &new_legends {
            let entity = &mut state.entities[*idx];
            if let Some(npc) = &mut entity.npc {
                // Add legendary title.
                npc.name = format!("{} the Legendary", old_name);

                // Emotional impact of becoming legendary.
                npc.emotions.pride = 1.0;
                npc.emotions.joy = (npc.emotions.joy + 0.8).min(1.0);
                npc.needs.esteem = 100.0;
                npc.needs.purpose = 100.0;

                // Leadership tags from legendary status.
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::LEADERSHIP, 10.0);
                    a.add(tags::TEACHING, 5.0);
                    a.add(tags::DIPLOMACY, 5.0);
                    a
                });
            }

            let sid = state.entities[*idx].npc.as_ref()
                .and_then(|n| n.home_settlement_id);
            let settlement_name = sid
                .and_then(|s| state.settlements.iter().find(|st| st.id == s))
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "the wilderness".into());

            let class_summary = state.entities[*idx].npc.as_ref()
                .map(|n| {
                    n.classes.iter()
                        .map(|c| format!("{} L{}", c.display_name, c.level))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Narrative,
                text: format!(
                    "{} of {} has become a LEGEND! Their deeds echo across the realm. Classes: [{}].",
                    old_name, settlement_name, class_summary),
                entity_ids: vec![state.entities[*idx].id],
            });
        }
    }

    // --- Phase 2: Legend effects (every 50 ticks) ---
    if tick % 50 == 0 {
        // Find all living legends and their settlements.
        let legend_settlements: Vec<(u32, u32)> = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc)
            .filter(|e| e.npc.as_ref().map(|n| n.name.contains("the Legendary")).unwrap_or(false))
            .filter_map(|e| {
                e.npc.as_ref()
                    .and_then(|n| n.home_settlement_id)
                    .map(|sid| (e.id, sid))
            })
            .collect();

        // Boost morale at settlements with legends.
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            let sid = match npc.home_settlement_id { Some(s) => s, None => continue };

            if legend_settlements.iter().any(|&(_, ls)| ls == sid) {
                npc.morale = (npc.morale + LEGEND_MORALE_BOOST * 0.02).min(100.0);
                // Legends inspire purpose in others.
                if npc.needs.purpose < 60.0 {
                    npc.needs.purpose += 0.5;
                }
            }
        }
    }

    // --- Phase 3: Legend death (only fire once per legend) ---
    let dead_legend: Option<(u32, String)> = state.entities.iter()
        .filter(|e| !e.alive && e.kind == EntityKind::Npc)
        .filter(|e| e.npc.as_ref().map(|n| n.name.contains("the Legendary")).unwrap_or(false))
        .filter(|e| state.world_events.iter().any(|ev|
            matches!(ev, WorldEvent::EntityDied { entity_id, .. } if *entity_id == e.id)))
        // Only if we haven't already chronicled this legend's death.
        .filter(|e| !state.chronicle.iter().any(|c|
            c.text.contains("world mourns") && c.entity_ids.contains(&e.id)))
        .map(|e| (e.id, e.npc.as_ref().unwrap().name.clone()))
        .next();

    if let Some((legend_id, legend_name)) = dead_legend {
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &mut entity.npc {
                npc.emotions.grief = (npc.emotions.grief + 0.5).min(1.0);
                npc.morale = (npc.morale - 10.0).max(0.0);
            }
        }
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Narrative,
            text: format!(
                "The world mourns. {} has fallen. Songs will be sung of their deeds for generations.",
                legend_name),
            entity_ids: vec![legend_id],
        });
    }
}
