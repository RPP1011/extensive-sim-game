#![allow(unused)]
//! Loot drops from combat and quest completion.
//!
//! On quest victory or monster kill, generates item drops as TransferGoods/TransferGold
//! deltas. Quality scales with threat level, item type biased by quest type.
//!
//! Ported from `crates/headless_campaign/src/systems/loot.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, EntityKind, WorldTeam};

// NEEDS STATE: completed_quests_this_tick: Vec<CompletedQuest> on WorldState
//              (or a way to detect quest completion events from other systems)
// NEEDS STATE: CompletedQuest { quest_id, quest_type, threat_level, member_ids, result }
// NEEDS STATE: QuestResult enum { Victory, Defeat, Abandoned }
// NEEDS DELTA: SpawnItem { recipient_id: u32, item_name: String, slot: u8, quality: f32,
//              stat_bonuses: [f32; 4] }

/// Commodity index used for "loot drops" (generic treasure goods).
/// Maps to one of the NUM_COMMODITIES slots.
const LOOT_COMMODITY: usize = 7; // Last commodity slot = treasure/loot

/// Gold drop per unit of threat level on monster kill.
const GOLD_PER_THREAT: f32 = 2.0;

/// Goods drop per unit of threat on monster kill.
const GOODS_PER_THREAT: f32 = 0.5;

pub fn compute_loot(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // --- Monster kill loot ---
    // When a monster dies (hp <= 0, team == Hostile), nearby friendly entities
    // receive gold and goods drops scaled by the monster's level.
    //
    // We check for monsters that are about to die this tick by looking at
    // their current HP. The battles system emits Die deltas, but since
    // all systems read the same frozen snapshot, we detect "near death"
    // monsters (hp <= 0 or hp very low relative to incoming damage).

    for grid in &state.grids {
        // Find dying monsters on this grid
        let dying_monsters: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.kind == EntityKind::Monster && e.alive && e.hp <= 0.0)
            .collect();

        if dying_monsters.is_empty() {
            continue;
        }

        // Find alive friendlies on this grid to receive loot
        let friendlies: Vec<&crate::world_sim::state::Entity> = grid
            .entity_ids
            .iter()
            .filter_map(|&eid| state.entity(eid))
            .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
            .collect();

        if friendlies.is_empty() {
            continue;
        }

        for monster in &dying_monsters {
            let threat = monster.level as f32;

            // Gold reward split among friendlies
            let total_gold = threat * GOLD_PER_THREAT;
            let gold_each = total_gold / friendlies.len() as f32;

            for friendly in &friendlies {
                if gold_each > 0.0 {
                    out.push(WorldDelta::TransferGold {
                        from_id: monster.id,
                        to_id: friendly.id,
                        amount: gold_each,
                    });
                }
            }

            // Goods (treasure) reward to the nearest friendly
            let goods_amount = threat * GOODS_PER_THREAT;
            if goods_amount > 0.0 {
                // Pick closest friendly by position
                let recipient = friendlies
                    .iter()
                    .min_by(|a, b| {
                        let da = dist_sq(a.pos, monster.pos);
                        let db = dist_sq(b.pos, monster.pos);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(recipient) = recipient {
                    out.push(WorldDelta::TransferGoods {
                        from_id: monster.id,
                        to_id: recipient.id,
                        commodity: LOOT_COMMODITY,
                        amount: goods_amount,
                    });
                }
            }
        }
    }

    // --- Quest completion loot ---
    // The original system generates equipment items (weapons, armor, accessories)
    // with grammar-walked names and archetype-biased slot selection.
    //
    // In the delta architecture, equipment items would need a SpawnItem delta
    // variant since they carry structured data (name, slot, quality, stat bonuses)
    // that doesn't map to commodity transfers.
    //
    // Until SpawnItem exists, quest victory loot is represented as gold and
    // commodity transfers above (from monster kills during the quest battle).
    //
    // When the state is extended with quest tracking:
    //
    //   for quest in &state.completed_quests_this_tick {
    //       if quest.result != QuestResult::Victory { continue; }
    //       let drop_chance = quest_type_drop_chance(quest.quest_type);
    //       // RNG roll against drop_chance
    //       // Generate item quality from threat level
    //       // Emit SpawnItem delta to best-matching party member
    //   }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// Drop chance by quest type (mirrors original system).
fn quest_type_drop_chance(quest_type: u8) -> f32 {
    match quest_type {
        0 => 0.70, // Combat
        1 => 0.50, // Exploration
        2 => 0.30, // Rescue
        3 => 0.20, // Gather
        4 => 0.15, // Diplomatic
        5 => 0.40, // Escort
        _ => 0.30,
    }
}
