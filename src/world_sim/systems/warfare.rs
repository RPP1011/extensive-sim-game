//! Warfare — faction wars, declarations, and peace treaties.
//!
//! Factions accumulate grievances from: NPCs killed by rival faction members,
//! settlements lost, betrayals. When grievances exceed a threshold, war is
//! declared. During war:
//! - Adventuring parties target enemy settlements
//! - Trade between warring factions is blocked
//! - Monster mobilization increases near borders
//!
//! Peace is negotiated when both factions are exhausted (low population,
//! depleted treasury). Peace creates mutual oaths and chronicle entries.
//!
//! War state stored on faction pairs in WorldState.relations.
//!
//! Cadence: every 300 ticks.

use crate::world_sim::state::*;

const WAR_CHECK_INTERVAL: u64 = 300;
const GRIEVANCE_WAR_THRESHOLD: f32 = 50.0;
const EXHAUSTION_PEACE_THRESHOLD: f32 = 20.0;
/// RelationKind discriminant for war state (1.0 = at war, 0.0 = peace).
const WAR_RELATION_KIND: u8 = 10;

pub fn advance_warfare(state: &mut WorldState) {
    if state.tick % WAR_CHECK_INTERVAL != 0 || state.tick == 0 { return; }
    if state.factions.len() < 2 { return; }

    let tick = state.tick;

    // --- Phase 1: Calculate inter-faction grievances ---
    // Grievance = sum of grudges held by NPCs of faction A against NPCs of faction B.
    let num_factions = state.factions.len();
    let mut grievances = vec![vec![0.0f32; num_factions]; num_factions];

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        let my_faction = match npc.faction_id { Some(f) => f as usize, None => continue };
        if my_faction >= num_factions { continue; }

        for belief in &npc.memory.beliefs {
            if let BeliefType::Grudge(target_id) = &belief.belief_type {
                // Find target's faction.
                let target_faction = state.entities.iter()
                    .find(|e| e.id == *target_id)
                    .and_then(|e| e.npc.as_ref())
                    .and_then(|n| n.faction_id)
                    .map(|f| f as usize);

                if let Some(tf) = target_faction {
                    if tf < num_factions && tf != my_faction {
                        grievances[my_faction][tf] += belief.confidence * 10.0;
                    }
                }
            }
        }
    }

    // --- Phase 2: War declarations ---
    for a in 0..num_factions {
        for b in (a + 1)..num_factions {
            let mutual = grievances[a][b] + grievances[b][a];
            let fid_a = state.factions[a].id;
            let fid_b = state.factions[b].id;

            // Check if already at war.
            let key = if fid_a < fid_b { (fid_a, fid_b, WAR_RELATION_KIND) }
                else { (fid_b, fid_a, WAR_RELATION_KIND) };
            let at_war = state.relations.get(&key).copied().unwrap_or(0.0) > 0.5;

            if !at_war && mutual > GRIEVANCE_WAR_THRESHOLD {
                // Declare war!
                state.relations.insert(key, 1.0);

                let name_a = state.factions[a].name.clone();
                let name_b = state.factions[b].name.clone();
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("WAR! The {} declares war on the {}! Grievances have boiled over.",
                        name_a, name_b),
                    entity_ids: vec![],
                });

                // War morale: boost morale for aggressive factions, fear for others.
                for entity in &mut state.entities {
                    if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                    if let Some(npc) = &mut entity.npc {
                        if npc.faction_id == Some(fid_a) || npc.faction_id == Some(fid_b) {
                            npc.emotions.anger = (npc.emotions.anger + 0.3).min(1.0);
                            npc.morale = (npc.morale + 5.0).min(100.0); // war fervor
                        }
                    }
                }
            }

            if at_war {
                // --- Phase 3: Check for peace ---
                // Both factions exhausted?
                let pop_a: usize = state.entities.iter()
                    .filter(|e| e.alive && e.kind == EntityKind::Npc
                        && e.npc.as_ref().map(|n| n.faction_id == Some(fid_a)).unwrap_or(false))
                    .count();
                let pop_b: usize = state.entities.iter()
                    .filter(|e| e.alive && e.kind == EntityKind::Npc
                        && e.npc.as_ref().map(|n| n.faction_id == Some(fid_b)).unwrap_or(false))
                    .count();

                let treasury_a: f32 = state.settlements.iter()
                    .filter(|s| s.faction_id == Some(fid_a))
                    .map(|s| s.treasury).sum();
                let treasury_b: f32 = state.settlements.iter()
                    .filter(|s| s.faction_id == Some(fid_b))
                    .map(|s| s.treasury).sum();

                let exhaustion = (pop_a + pop_b) as f32 * 0.1 + (treasury_a + treasury_b) * 0.01;
                let peace_roll = entity_hash_f32(fid_a, tick, fid_b as u64);

                if exhaustion < EXHAUSTION_PEACE_THRESHOLD || peace_roll < 0.02 {
                    // Peace treaty!
                    state.relations.insert(key, 0.0);

                    let name_a = state.factions[a].name.clone();
                    let name_b = state.factions[b].name.clone();
                    state.chronicle.push(ChronicleEntry {
                        tick,
                        category: ChronicleCategory::Narrative,
                        text: format!("PEACE! The {} and {} sign a peace treaty. The war is over.",
                            name_a, name_b),
                        entity_ids: vec![],
                    });

                    // Peace morale boost.
                    for entity in &mut state.entities {
                        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                        if let Some(npc) = &mut entity.npc {
                            if npc.faction_id == Some(fid_a) || npc.faction_id == Some(fid_b) {
                                npc.emotions.joy = (npc.emotions.joy + 0.5).min(1.0);
                                npc.morale = (npc.morale + 10.0).min(100.0);
                                npc.emotions.anger = (npc.emotions.anger - 0.3).max(0.0);
                            }
                        }
                    }

                    // Reduce grievance-related grudges.
                    for entity in &mut state.entities {
                        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                        if let Some(npc) = &mut entity.npc {
                            let my_fid = npc.faction_id;
                            if my_fid == Some(fid_a) || my_fid == Some(fid_b) {
                                // Reduce confidence on grudges against the other faction.
                                for belief in &mut npc.memory.beliefs {
                                    if let BeliefType::Grudge(_) = &belief.belief_type {
                                        belief.confidence *= 0.5; // peace halves grudges
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
