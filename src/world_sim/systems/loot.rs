//! Loot drops from combat and quest completion.
//!
//! On monster kill, generates gold transfers to nearby friendlies and spawns
//! item entities as loot drops. Item quality scales with monster threat level.
//!
//! **Gold conservation:** Monster kill gold is paid from the settlement treasury
//! as a bounty reward. If the settlement cannot afford it, no gold is paid.
//! Goods drops and item spawns are still awarded (representing physical loot).
//!
//! Items are spawned as unowned entities at the kill location with the monster's
//! settlement affiliation. NPCs pick them up via the equipping system.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{
    Entity, EntityKind, ItemData, ItemRarity, ItemSlot, WorldState, WorldTeam,
    entity_hash,
};

/// Gold drop per unit of threat level on monster kill.
const GOLD_PER_THREAT: f32 = 2.0;

/// Chance that a monster drops an item (0.0–1.0).
const ITEM_DROP_CHANCE: f32 = 0.15;

pub fn compute_loot(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_loot_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_loot_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };
    let grid_id = match settlement.grid_id {
        Some(gid) => gid,
        None => return,
    };
    let grid = match state.fidelity_zone(grid_id) {
        Some(g) => g,
        None => return,
    };

    if grid.fidelity != crate::world_sim::fidelity::Fidelity::High { return; }

    // Count dying monsters and friendlies without allocating.
    let mut dying_ids = [0u32; 32];
    let mut dying_levels = [0u32; 32];
    let mut dying_pos = [(0.0f32, 0.0f32); 32];
    let mut dc = 0usize;
    let mut friendly_ids = [0u32; 64];
    let mut fc = 0usize;

    for &eid in &grid.entity_ids {
        if let Some(e) = state.entity(eid) {
            if !e.alive { continue; }
            if e.kind == EntityKind::Monster && e.hp <= 0.0 && dc < 32 {
                dying_ids[dc] = eid;
                dying_levels[dc] = e.level;
                dying_pos[dc] = e.pos;
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

        // Monster kill gold is paid from settlement treasury as a bounty
        let can_afford = settlement.treasury > total_gold;
        for fi in 0..fc {
            if gold_each > 0.0 && can_afford {
                out.push(WorldDelta::TransferGold {
                    from_entity: settlement_id,
                    to_entity: friendly_ids[fi],
                    amount: gold_each,
                });
            }
        }

        // Item drop chance (deterministic from tick + monster id).
        let drop_h = entity_hash(monster_id, state.tick, 0x100D);
        let roll = (drop_h % 1000) as f32 / 1000.0;
        if roll < ITEM_DROP_CHANCE {
            // Spawn an item entity at the monster's position.
            let slot = match entity_hash(monster_id, state.tick, 0x100E) % 3 {
                0 => ItemSlot::Weapon,
                1 => ItemSlot::Armor,
                _ => ItemSlot::Accessory,
            };

            // Quality and rarity scale with monster level.
            let quality = (1.0 + threat * 0.3).min(20.0);
            let rarity = if threat >= 40.0 { ItemRarity::Epic }
                else if threat >= 20.0 { ItemRarity::Rare }
                else if threat >= 10.0 { ItemRarity::Uncommon }
                else { ItemRarity::Common };

            let name = generate_loot_name(slot, rarity, drop_h as u64);

            out.push(WorldDelta::SpawnItem {
                pos: dying_pos[di],
                item_data: ItemData {
                    slot,
                    rarity,
                    quality,
                    durability: 80.0, // slightly worn from battle
                    max_durability: 100.0,
                    owner_id: None,
                    settlement_id: Some(settlement_id),
                    name,
                    crafter_id: None,
                    crafted_tick: state.tick,
                    history: Vec::new(),
                    is_legendary: false,
                    is_relic: false,
                    relic_bonus: None,
                },
            });
        }
    }
}

fn generate_loot_name(slot: ItemSlot, rarity: ItemRarity, h: u64) -> String {
    let prefix = match rarity {
        ItemRarity::Common => "Rusted ",
        ItemRarity::Uncommon => "Weathered ",
        ItemRarity::Rare => "Ornate ",
        ItemRarity::Epic => "Ancient ",
        ItemRarity::Legendary => "Legendary ",
    };

    let base = match slot {
        ItemSlot::Weapon => {
            const W: [&str; 4] = ["Blade", "War Axe", "Warhammer", "Glaive"];
            W[(h >> 44) as usize % W.len()]
        }
        ItemSlot::Armor => {
            const A: [&str; 4] = ["Cuirass", "Helm", "Shield", "Greaves"];
            A[(h >> 44) as usize % A.len()]
        }
        ItemSlot::Accessory => {
            const C: [&str; 4] = ["Talisman", "Warboots", "Signet Ring", "Ward Amulet"];
            C[(h >> 44) as usize % C.len()]
        }
    };

    format!("{}{}", prefix, base)
}
