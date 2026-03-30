#![allow(unused)]
//! Family system — marriage, children, and inheritance.
//!
//! NPCs who have socialized frequently (3+ MadeNewFriend events with same NPC)
//! and share a settlement can marry. Married couples share a home building.
//! After 2000 ticks of marriage, they may have a child — a new NPC entity
//! with behavior profile blended 50/50 from parents.
//!
//! Children inherit: blended behavior_profile, averaged personality,
//! parent settlement/faction, and parent lineage chain.
//!
//! Cadence: every 100 ticks (marriage check), every 200 ticks (birth check).

use crate::world_sim::state::*;
use crate::world_sim::naming;

const MARRIAGE_INTERVAL: u64 = 100;
const BIRTH_INTERVAL: u64 = 200;
/// Ticks of marriage before children are possible.
const MARRIAGE_MATURITY: u64 = 2000;
/// Max children per couple.
const MAX_CHILDREN: usize = 3;
/// Max population per settlement before birth suppression.
const POP_CAP_PER_SETTLEMENT: usize = 300;

/// Process marriages and births. Called post-apply from runtime.rs.
pub fn advance_family(state: &mut WorldState) {
    let tick = state.tick;

    // --- Marriage (every 100 ticks) ---
    if tick % MARRIAGE_INTERVAL == 0 && tick > 0 {
        process_marriages(state, tick);
    }

    // --- Births (every 200 ticks) ---
    if tick % BIRTH_INTERVAL == 0 && tick > 0 {
        process_births(state, tick);
    }
}

fn process_marriages(state: &mut WorldState, tick: u64) {
    // Collect unmarried NPC pairs at same settlement who are close friends.
    let mut candidates: Vec<(usize, usize, u32)> = Vec::new(); // (idx_a, idx_b, settlement_id)

    let entity_count = state.entities.len();
    for i in 0..entity_count {
        let entity_a = &state.entities[i];
        if !entity_a.alive || entity_a.kind != EntityKind::Npc { continue; }
        let npc_a = match &entity_a.npc { Some(n) => n, None => continue };
        if npc_a.spouse_id.is_some() { continue; } // already married
        let sid_a = match npc_a.home_settlement_id { Some(s) => s, None => continue };

        // Find marriage candidates at same settlement — NPCs with high social
        // who share a settlement and have compatible personality.
        for j in (i + 1)..entity_count {
            if candidates.len() >= 3 { break; }
            let entity_b = &state.entities[j];
            if !entity_b.alive || entity_b.kind != EntityKind::Npc { continue; }
            let npc_b = match &entity_b.npc { Some(n) => n, None => continue };
            if npc_b.spouse_id.is_some() { continue; }
            if npc_b.home_settlement_id != Some(sid_a) { continue; }

            // Both NPCs must have good social standing (recently socialized).
            if npc_a.needs.social < 40.0 || npc_b.needs.social < 40.0 { continue; }

            // Personality compatibility — similar social drive and compassion.
            let compat = 1.0
                - (npc_a.personality.social_drive - npc_b.personality.social_drive).abs()
                - (npc_a.personality.compassion - npc_b.personality.compassion).abs() * 0.5;
            if compat < 0.3 { continue; }

            // Deterministic marriage check (~3% per eligible pair per check).
            let roll = pair_hash_f32(entity_a.id, entity_b.id, tick, 0xFA01);
            if roll < 0.15 { // 15% per eligible pair per check
                candidates.push((i, j, sid_a));
            }
        }
    }

    // Perform marriages — track who married this tick to prevent polygamy.
    let mut married_this_tick: Vec<u32> = Vec::new();
    for (idx_a, idx_b, sid) in candidates {
        let id_a = state.entities[idx_a].id;
        let id_b = state.entities[idx_b].id;
        if married_this_tick.contains(&id_a) || married_this_tick.contains(&id_b) { continue; }
        married_this_tick.push(id_a);
        married_this_tick.push(id_b);
        let name_a = state.entities[idx_a].npc.as_ref().map(|n| n.name.clone()).unwrap_or_default();
        let name_b = state.entities[idx_b].npc.as_ref().map(|n| n.name.clone()).unwrap_or_default();

        // Set spouse IDs.
        if let Some(npc) = state.entities[idx_a].npc.as_mut() {
            npc.spouse_id = Some(id_b);
            npc.emotions.joy = (npc.emotions.joy + 0.8).min(1.0);
            npc.needs.social = 100.0;
        }
        if let Some(npc) = state.entities[idx_b].npc.as_mut() {
            npc.spouse_id = Some(id_a);
            npc.emotions.joy = (npc.emotions.joy + 0.8).min(1.0);
            npc.needs.social = 100.0;
        }

        // Share home building — B moves to A's home if A has one.
        let home_bid = state.entities[idx_a].npc.as_ref().and_then(|n| n.home_building_id);
        if let Some(bid) = home_bid {
            if let Some(npc) = state.entities[idx_b].npc.as_mut() {
                npc.home_building_id = Some(bid);
            }
        }

        // Chronicle.
        let settlement_name = state.settlements.iter()
            .find(|s| s.id == sid)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Achievement,
            text: format!("{} and {} of {} were wed.", name_a, name_b, settlement_name),
            entity_ids: vec![id_a, id_b],
        });
    }
}

fn process_births(state: &mut WorldState, tick: u64) {
    // Find married couples who've been married long enough and have room for children.
    let mut births: Vec<(usize, usize, u32)> = Vec::new(); // (parent_a_idx, parent_b_idx, settlement_id)

    for i in 0..state.entities.len() {
        if births.len() >= 3 { break; } // max 3 births per tick
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        let spouse_id = match npc.spouse_id { Some(id) => id, None => continue };
        let sid = match npc.home_settlement_id { Some(id) => id, None => continue };
        if npc.children.len() >= MAX_CHILDREN { continue; }

        // Only process one parent per couple (lower ID to avoid duplicates).
        if entity.id > spouse_id { continue; }

        // Simple maturity gate: no births in the first 1000 ticks of the sim.
        if tick < 1000 { continue; }

        // Check population cap at settlement.
        let pop = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc
                && e.npc.as_ref().map(|n| n.home_settlement_id == Some(sid)).unwrap_or(false))
            .count();
        if pop >= POP_CAP_PER_SETTLEMENT { continue; }

        // Find spouse index.
        let spouse_idx = state.entities.iter().position(|e| e.id == spouse_id && e.alive);
        let spouse_idx = match spouse_idx { Some(i) => i, None => continue };

        // Deterministic birth check (~5% per eligible couple per check).
        let roll = entity_hash_f32(entity.id, tick, 0xBA81);
        if roll < 0.05 {
            births.push((i, spouse_idx, sid));
        }
    }

    // Create children.
    for (parent_a_idx, parent_b_idx, sid) in births {
        let parent_a_id = state.entities[parent_a_idx].id;
        let parent_b_id = state.entities[parent_b_idx].id;
        let parent_a_pos = state.entities[parent_a_idx].pos;

        // Blend behavior profiles: 50/50 from each parent, at 30% strength.
        let profile_a = state.entities[parent_a_idx].npc.as_ref()
            .map(|n| n.behavior_profile.clone()).unwrap_or_default();
        let profile_b = state.entities[parent_b_idx].npc.as_ref()
            .map(|n| n.behavior_profile.clone()).unwrap_or_default();

        let mut child_profile: Vec<(u32, f32)> = Vec::new();
        // Merge both profiles.
        for &(tag, val) in &profile_a {
            child_profile.push((tag, val * 0.15)); // 30% of parent A's half
        }
        for &(tag, val) in &profile_b {
            if let Some(existing) = child_profile.iter_mut().find(|(t, _)| *t == tag) {
                existing.1 += val * 0.15;
            } else {
                child_profile.push((tag, val * 0.15));
            }
        }
        child_profile.sort_by_key(|&(t, _)| t);

        // Blend personality.
        let pers_a = state.entities[parent_a_idx].npc.as_ref()
            .map(|n| n.personality).unwrap_or_default();
        let pers_b = state.entities[parent_b_idx].npc.as_ref()
            .map(|n| n.personality).unwrap_or_default();
        let child_pers = Personality {
            risk_tolerance: (pers_a.risk_tolerance + pers_b.risk_tolerance) / 2.0,
            social_drive: (pers_a.social_drive + pers_b.social_drive) / 2.0,
            ambition: (pers_a.ambition + pers_b.ambition) / 2.0,
            compassion: (pers_a.compassion + pers_b.compassion) / 2.0,
            curiosity: (pers_a.curiosity + pers_b.curiosity) / 2.0,
        };

        // Create child entity.
        let child_id = state.next_entity_id();
        let child_name = naming::generate_personal_name(child_id, tick);
        let faction_id = state.entities[parent_a_idx].npc.as_ref().and_then(|n| n.faction_id);
        let home_bid = state.entities[parent_a_idx].npc.as_ref().and_then(|n| n.home_building_id);

        let mut child = Entity::new_npc(child_id, parent_a_pos);
        if let Some(npc) = &mut child.npc {
            npc.name = child_name.clone();
            npc.home_settlement_id = Some(sid);
            npc.home_building_id = home_bid;
            npc.faction_id = faction_id;
            npc.parents = vec![parent_a_id, parent_b_id];
            npc.born_tick = tick;
            npc.behavior_profile = child_profile;
            npc.personality = child_pers;
        }
        state.entities.push(child);

        // Register child on parents.
        if let Some(npc) = state.entities[parent_a_idx].npc.as_mut() {
            npc.children.push(child_id);
            npc.emotions.joy = (npc.emotions.joy + 0.9).min(1.0);
        }
        if let Some(npc) = state.entities[parent_b_idx].npc.as_mut() {
            npc.children.push(child_id);
            npc.emotions.joy = (npc.emotions.joy + 0.9).min(1.0);
        }

        // Chronicle.
        let parent_a_name = state.entities[parent_a_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        let parent_b_name = state.entities[parent_b_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Achievement,
            text: format!("{} was born to {} and {}.", child_name, parent_a_name, parent_b_name),
            entity_ids: vec![child_id, parent_a_id, parent_b_id],
        });
    }
}
