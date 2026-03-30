#![allow(unused)]
//! Titles & honorifics — NPCs earn permanent name titles from deeds.
//!
//! Titles are checked every 500 ticks. Each NPC can earn at most one title
//! (the most prestigious). Titles are prepended or appended to the NPC name.
//!
//! Title hierarchy (highest priority wins):
//! - "the Legendary" — from legends.rs (already exists)
//! - "the Oathkeeper" — 3+ fulfilled oaths
//! - "the Avenger" — killed a grudge target
//! - "the Exile" — redeemed outlaw
//! - "the Bereaved" — lost spouse + 5+ friend deaths
//! - "the Founder" — led a settlement founding expedition
//! - "the Unbroken" — survived with <10% HP + 3+ attacks
//! - "the Merchant Prince/Princess" — gold > 5000
//! - "Dragonslayer" / "Giantslayer" — killed a named monster
//! - "of [Settlement]" — default, shows home
//!
//! Cadence: every 500 ticks.

use crate::world_sim::state::*;

const TITLE_INTERVAL: u64 = 500;

pub fn advance_titles(state: &mut WorldState) {
    if state.tick % TITLE_INTERVAL != 0 || state.tick == 0 { return; }

    // Pre-collect alive entity IDs for spouse checks.
    let alive_ids: Vec<u32> = state.entities.iter()
        .filter(|e| e.alive)
        .map(|e| e.id)
        .collect();

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        if npc.name.contains(" the ") || npc.name.contains("slayer") { continue; }

        let title = determine_title(npc, entity.hp, entity.max_hp, &alive_ids);

        if let Some(title) = title {
            let old_name = npc.name.clone();
            npc.name = format!("{} {}", old_name, title);
        }
    }
}

fn determine_title(npc: &NpcData, hp: f32, max_hp: f32, alive_ids: &[u32]) -> Option<&'static str> {
    // Already legendary? (checked by caller via name contains)

    // Oathkeeper: 3+ fulfilled oaths.
    let fulfilled_oaths = npc.oaths.iter().filter(|o| o.fulfilled).count();
    if fulfilled_oaths >= 3 {
        return Some("the Oathkeeper");
    }

    // Avenger: has a fulfilled vengeance oath.
    let has_vengeance = npc.oaths.iter().any(|o| {
        o.fulfilled && o.kind == crate::world_sim::systems::oaths::OathKind::Vengeance
    });
    if has_vengeance {
        return Some("the Avenger");
    }

    // Bereaved: lost spouse AND 5+ friend deaths.
    if npc.spouse_id.is_some() {
        let spouse_alive = npc.spouse_id
            .map(|sid| alive_ids.contains(&sid))
            .unwrap_or(true);
        let friend_deaths = npc.memory.events.iter()
            .filter(|e| matches!(e.event_type, MemEventType::FriendDied(_)))
            .count();
        if !spouse_alive && friend_deaths >= 5 {
            return Some("the Bereaved");
        }
    }

    // Unbroken: low HP survivor with 3+ attacks.
    let attacks = npc.memory.events.iter()
        .filter(|e| matches!(e.event_type, MemEventType::WasAttacked))
        .count();
    if attacks >= 3 && hp < max_hp * 0.3 && hp > 0.0 {
        return Some("the Unbroken");
    }

    // Merchant Prince: very rare — requires trade tags > 200 AND gold > 5000.
    if npc.gold > 5000.0 && npc.behavior_value(tags::TRADE) > 200.0 {
        return Some("the Merchant Prince");
    }

    // Well-traveled: has survived starvation 3+ times.
    let starved = npc.memory.events.iter()
        .filter(|e| matches!(e.event_type, MemEventType::Starved))
        .count();
    if starved >= 3 {
        return Some("the Enduring");
    }

    // Many children.
    if npc.children.len() >= 3 {
        return Some("the Patriarch");
    }

    None
}
