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
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_loot_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Finds the grid associated with this settlement and processes loot on it.
pub fn compute_loot_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    let grid_id = match state.settlement(settlement_id).and_then(|s| s.grid_id) {
        Some(gid) => gid,
        None => return,
    };
    let grid = match state.grid(grid_id) {
        Some(g) => g,
        None => return,
    };

    if grid.fidelity != crate::world_sim::fidelity::Fidelity::High { return; }

    // Count dying monsters and friendlies without allocating.
    let mut dying_ids = [0u32; 32];
    let mut dying_levels = [0u32; 32];
    let mut dc = 0usize;
    let mut friendly_ids = [0u32; 64];
    let mut fc = 0usize;

    for &eid in &grid.entity_ids {
        if let Some(e) = state.entity(eid) {
            if !e.alive { continue; }
            if e.kind == EntityKind::Monster && e.hp <= 0.0 && dc < 32 {
                dying_ids[dc] = eid;
                dying_levels[dc] = e.level;
                dc += 1;
            } else if e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly && fc < 64 {
                friendly_ids[fc] = eid;
                fc += 1;
            }
        }
    }

    if dc == 0 || fc == 0 { return; }

    for di in 0..dc {
        let monster_id = dying_ids[di];
        let threat = dying_levels[di] as f32;
        let total_gold = threat * GOLD_PER_THREAT;
        let gold_each = total_gold / fc as f32;

        for fi in 0..fc {
            if gold_each > 0.0 {
                out.push(WorldDelta::TransferGold {
                    from_id: monster_id,
                    to_id: friendly_ids[fi],
                    amount: gold_each,
                });
            }
        }

        // Goods (treasure) reward to the first friendly (no alloc nearest search)
        let goods_amount = threat * GOODS_PER_THREAT;
        if goods_amount > 0.0 && fc > 0 {
            out.push(WorldDelta::TransferGoods {
                from_id: monster_id,
                to_id: friendly_ids[0],
                commodity: LOOT_COMMODITY,
                amount: goods_amount,
            });
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
