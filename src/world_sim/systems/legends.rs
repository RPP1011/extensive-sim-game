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

        // Single chronicle pass: count mentions per entity_id (was O(E × C)
        // — a filter over the full chronicle for every candidate NPC).
        // Chronicle grows unbounded, so this is now O(C) instead of O(E × C).
        let mut mentions_by_id: std::collections::HashMap<u32, u32, ahash::RandomState> =
            std::collections::HashMap::default();
        for e in &state.chronicle {
            for &id in &e.entity_ids {
                *mentions_by_id.entry(id).or_insert(0) += 1;
            }
        }

        for (i, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };

            // Already a legend? (check for "the Legendary" in name)
            if npc.name.contains("the Legendary") { continue; }

            // Chronicle mention count via prebuilt index.
            let mentions = *mentions_by_id.get(&entity.id).unwrap_or(&0);
            if (mentions as usize) < MIN_CHRONICLE_MENTIONS { continue; }

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
        // Find settlements with at least one living legend. HashSet lookup
        // replaces the O(L) linear scan per NPC (was `iter().any()` inside
        // the outer entity loop — O(E × L)).
        let mut legend_settlements: std::collections::HashSet<u32, ahash::RandomState> =
            std::collections::HashSet::default();
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                if npc.name.contains("the Legendary") {
                    if let Some(sid) = npc.home_settlement_id {
                        legend_settlements.insert(sid);
                    }
                }
            }
        }

        if !legend_settlements.is_empty() {
            // Boost morale at settlements with legends.
            for entity in &mut state.entities {
                if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                let npc = match &mut entity.npc { Some(n) => n, None => continue };
                let sid = match npc.home_settlement_id { Some(s) => s, None => continue };

                if legend_settlements.contains(&sid) {
                    npc.morale = (npc.morale + LEGEND_MORALE_BOOST * 0.02).min(100.0);
                    // Legends inspire purpose in others.
                    if npc.needs.purpose < 60.0 {
                        npc.needs.purpose += 0.5;
                    }
                }
            }
        }
    }

    // --- Phase 3: Legend death (only fire once per legend) ---
    // Previously this ran every tick with nested scans over entities ×
    // world_events × chronicle (chronicle grows unbounded — O(tick)).
    // Restructure: iterate THIS tick's EntityDied world events only,
    // check if any of them reference a legend. Only then do the
    // chronicle "already mourned" check.
    let mut dead_legend: Option<(u32, String)> = None;
    for ev in &state.world_events {
        if let WorldEvent::EntityDied { entity_id, .. } = ev {
            let Some(entity) = state.entity(*entity_id) else { continue };
            if entity.alive { continue; } // already revived
            if entity.kind != EntityKind::Npc { continue; }
            let Some(npc) = &entity.npc else { continue };
            if !npc.name.contains("the Legendary") { continue; }
            // "Already mourned" check — unavoidable chronicle scan but
            // only runs if we've already confirmed a legend died this tick.
            let already = state.chronicle.iter().any(|c|
                c.text.contains("world mourns") && c.entity_ids.contains(entity_id));
            if already { continue; }
            dead_legend = Some((*entity_id, npc.name.clone()));
            break;
        }
    }

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
