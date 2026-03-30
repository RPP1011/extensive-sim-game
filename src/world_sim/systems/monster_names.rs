#![allow(unused)]
//! Monster naming — monsters that kill NPCs gain names and notoriety.
//!
//! Tracks kills via last_damage_source in death records. When a monster
//! accumulates 3+ NPC kills, it gains a procedural name and stat boost.
//! Named monsters become bounty targets and create grudges. Killing one
//! is a legendary deed.
//!
//! Uses a lightweight approach: check death events for monster killers,
//! track kill counts on monsters via their level (already buffed by kills).
//!
//! Cadence: every 100 ticks.

use crate::world_sim::state::*;

const NAMING_INTERVAL: u64 = 100;
const KILLS_FOR_NAME: usize = 3;

pub fn advance_monster_naming(state: &mut WorldState) {
    if state.tick % NAMING_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Count NPC kills per monster from death events.
    // Since we can't track persistent kill counts on monsters (no NpcData),
    // we approximate: scan death records for monsters who killed NPCs.
    let mut monster_kills: Vec<(u32, u32)> = Vec::new(); // (monster_id, kill_count)

    for event in &state.world_events {
        if let WorldEvent::EntityDied { entity_id, .. } = event {
            // Find the victim — was it an NPC?
            let was_npc = state.entities.iter()
                .find(|e| e.id == *entity_id)
                .map(|e| e.kind == EntityKind::Npc)
                .unwrap_or(false);
            if !was_npc { continue; }

            // Who killed them? Check if a monster is nearby (proxy for killer).
            let victim_pos = state.entities.iter()
                .find(|e| e.id == *entity_id)
                .map(|e| e.pos)
                .unwrap_or((0.0, 0.0));

            for monster in &state.entities {
                if !monster.alive || monster.kind != EntityKind::Monster { continue; }
                let dx = monster.pos.0 - victim_pos.0;
                let dy = monster.pos.1 - victim_pos.1;
                if dx * dx + dy * dy < 400.0 { // within 20 units
                    if let Some(entry) = monster_kills.iter_mut().find(|(id, _)| *id == monster.id) {
                        entry.1 += 1;
                    } else {
                        monster_kills.push((monster.id, 1));
                    }
                    break; // one killer per death
                }
            }
        }
    }

    // Name monsters with enough kills.
    for (monster_id, kills) in &monster_kills {
        if (*kills as usize) < KILLS_FOR_NAME { continue; }

        let monster = match state.entities.iter_mut().find(|e| e.id == *monster_id && e.alive) {
            Some(m) => m,
            None => continue,
        };

        // Already named? (level > 30 means already boosted significantly)
        if monster.level > 30 { continue; }

        // Generate a fearsome name.
        let h = entity_hash(*monster_id, tick, 0xDE4D);
        let prefix = ["Dread", "Blood", "Shadow", "Iron", "Bone", "Storm", "Death", "Doom"]
            [h as usize % 8];
        let suffix = ["maw", "claw", "fang", "bane", "render", "stalker", "howl", "scourge"]
            [(h as usize / 8) % 8];
        let name = format!("{}{}", prefix, suffix);

        // Stat boost for named monster.
        monster.hp += 100.0;
        monster.max_hp += 100.0;
        monster.attack_damage += 10.0;
        monster.armor += 5.0;
        monster.level += 10;

        // Chronicle.
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Narrative,
            text: format!(
                "A fearsome creature has earned the name {}. It has slain {} warriors and grows ever stronger.",
                name, kills),
            entity_ids: vec![*monster_id],
        });

        // Collect nearby settlement IDs for grudge formation.
        let monster_pos = monster.pos;
        let nearby_sids: Vec<u32> = state.settlements.iter()
            .filter(|s| {
                let dx = s.pos.0 - monster_pos.0;
                let dy = s.pos.1 - monster_pos.1;
                dx * dx + dy * dy < 10000.0
            })
            .map(|s| s.id)
            .collect();

        // Form grudges at nearby settlements.
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            let sid = match npc.home_settlement_id { Some(s) => s, None => continue };
            if !nearby_sids.contains(&sid) { continue; }

            let has = npc.memory.beliefs.iter().any(|b|
                matches!(b.belief_type, BeliefType::Grudge(gid) if gid == *monster_id));
            if !has {
                npc.memory.beliefs.push(Belief {
                    belief_type: BeliefType::Grudge(*monster_id),
                    confidence: 1.0,
                    formed_tick: tick,
                });
            }
        }

        break; // one naming event per tick
    }
}
