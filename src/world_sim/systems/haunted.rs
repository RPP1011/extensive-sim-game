//! Haunted locations — sites of mass death become supernaturally dangerous.
//!
//! When 5+ NPCs die within 30 units of each other, the area becomes haunted.
//! Haunted locations:
//! - Generate persistent fear in nearby NPCs
//! - Attract monsters (increased local density)
//! - Create ghost stories that spread through tavern conversations
//! - Brave adventurers who spend time there gain awareness+faith tags
//! - Eventually "cleansed" by a temple-builder or Oathkeeper at the site
//!
//! Cadence: every 300 ticks.

use crate::world_sim::state::*;

const HAUNT_INTERVAL: u64 = 300;
const DEATH_CLUSTER_RANGE_SQ: f32 = 900.0; // 30 units
const DEATHS_FOR_HAUNTING: usize = 5;
const FEAR_RANGE_SQ: f32 = 2500.0; // 50 units
const MAX_HAUNTED_SITES: usize = 10;

pub fn advance_haunted(state: &mut WorldState) {
    if state.tick % HAUNT_INTERVAL != 0 || state.tick == 0 { return; }
    let tick = state.tick;

    // --- Phase 1: Detect new haunted locations from death clusters ---
    // Collect death positions from recent world events.
    let death_positions: Vec<(f32, f32)> = state.world_events.iter()
        .filter_map(|e| match e {
            WorldEvent::EntityDied { entity_id, .. } => {
                state.entities.iter()
                    .find(|ent| ent.id == *entity_id && ent.kind == EntityKind::Npc)
                    .map(|ent| ent.pos)
            }
            _ => None,
        })
        .collect();

    // Find clusters of deaths.
    let mut haunted_sites: Vec<((f32, f32), usize)> = Vec::new(); // (center, death_count)
    let mut counted: Vec<bool> = vec![false; death_positions.len()];

    for i in 0..death_positions.len() {
        if counted[i] || haunted_sites.len() >= MAX_HAUNTED_SITES { break; }

        let mut cluster = vec![i];
        for j in (i + 1)..death_positions.len() {
            if counted[j] { continue; }
            let dx = death_positions[i].0 - death_positions[j].0;
            let dy = death_positions[i].1 - death_positions[j].1;
            if dx * dx + dy * dy < DEATH_CLUSTER_RANGE_SQ {
                cluster.push(j);
            }
        }

        if cluster.len() >= DEATHS_FOR_HAUNTING {
            for &ci in &cluster { counted[ci] = true; }
            // Center of cluster.
            let cx = cluster.iter().map(|&ci| death_positions[ci].0).sum::<f32>() / cluster.len() as f32;
            let cy = cluster.iter().map(|&ci| death_positions[ci].1).sum::<f32>() / cluster.len() as f32;
            haunted_sites.push(((cx, cy), cluster.len()));
        }
    }

    // --- Phase 2: Apply haunting effects ---
    for &(site_pos, _death_count) in &haunted_sites {
        // Fear effect: NPCs near the haunted site get fear spikes.
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let dx = entity.pos.0 - site_pos.0;
            let dy = entity.pos.1 - site_pos.1;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < FEAR_RANGE_SQ {
                let npc = match &mut entity.npc { Some(n) => n, None => continue };
                let intensity = 1.0 - (dist_sq / FEAR_RANGE_SQ).sqrt();
                npc.emotions.fear = (npc.emotions.fear + intensity * 0.3).min(1.0);
                npc.emotions.anxiety = (npc.emotions.anxiety + intensity * 0.2).min(1.0);

                // Brave NPCs (high resilience) gain awareness from confronting fear.
                let resilience = npc.behavior_value(tags::RESILIENCE);
                if resilience > 50.0 && dist_sq < 400.0 { // within 20 units
                    npc.accumulate_tags(&{
                        let mut a = ActionTags::empty();
                        a.add(tags::AWARENESS, 1.0);
                        a.add(tags::FAITH, 0.5);
                        a
                    });
                }

                // Create ghost story belief.
                let has_ghost_story = npc.memory.beliefs.iter().any(|b| {
                    matches!(b.belief_type, BeliefType::LocationDangerous(_))
                });
                if !has_ghost_story {
                    let location_hash = ((site_pos.0 * 100.0) as u32).wrapping_add((site_pos.1 * 100.0) as u32);
                    npc.memory.beliefs.push(Belief {
                        belief_type: BeliefType::LocationDangerous(location_hash),
                        confidence: 0.8,
                        formed_tick: tick,
                    });
                }
            }
        }
    }

    // --- Phase 3: Chronicle for significant new haunted sites ---
    if !haunted_sites.is_empty() {
        // Only chronicle the most significant site per tick.
        let (site_pos, count) = haunted_sites.iter()
            .max_by_key(|(_, c)| *c)
            .unwrap();

        // Find nearest settlement for context.
        let nearest_name = state.settlements.iter()
            .min_by_key(|s| {
                let dx = s.pos.0 - site_pos.0;
                let dy = s.pos.1 - site_pos.1;
                (dx * dx + dy * dy) as u32
            })
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "the wilderness".into());

        // Deduplicate: don't re-chronicle the same area.
        let already_chronicled = state.chronicle.iter().any(|e|
            e.text.contains("haunted") && e.text.contains(&nearest_name));

        if !already_chronicled {
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Narrative,
                text: format!(
                    "The land near {} has become haunted. {} souls perished there, and the living shudder at its approach.",
                    nearest_name, count),
                entity_ids: vec![],
            });
        }
    }
}
