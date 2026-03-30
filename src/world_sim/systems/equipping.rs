//! Equipment system — NPCs equip item entities from their settlement.
//!
//! Every EQUIP_INTERVAL ticks, NPCs at settlements check for unowned item entities
//! at their settlement. If an available item is better than their current equipment,
//! they equip it (applying stat bonuses) and unequip the old item.
//!
//! Items are real entities with EntityKind::Item and ItemData. Equipment stat bonuses
//! scale with item quality, rarity, and durability.
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const EQUIP_INTERVAL: u64 = 100;

/// Maximum items to process per settlement per tick (prevent O(n²) blowup).
const MAX_ITEMS_PER_SETTLEMENT: usize = 64;

pub fn compute_equipping(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EQUIP_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_equipping_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_equipping_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % EQUIP_INTERVAL != 0 || state.tick == 0 { return; }

    // Collect unowned items at this settlement.
    let mut available_items: [(u32, ItemSlot, f32); MAX_ITEMS_PER_SETTLEMENT] =
        [(0, ItemSlot::Weapon, 0.0); MAX_ITEMS_PER_SETTLEMENT];
    let mut item_count = 0usize;

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Item { continue; }
        let item = match &entity.item { Some(i) => i, None => continue };
        if item.owner_id.is_some() { continue; } // already owned
        if item.durability <= 0.0 { continue; } // broken
        if item_count >= MAX_ITEMS_PER_SETTLEMENT { break; }

        available_items[item_count] = (entity.id, item.slot, item.effective_quality() * item.durability_fraction());
        item_count += 1;
    }

    if item_count == 0 { return; }

    // For each NPC, find the best available item per slot and equip if upgrade.
    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Check each slot for potential upgrade.
        for slot_idx in 0..3u8 {
            let slot = match slot_idx {
                0 => ItemSlot::Weapon,
                1 => ItemSlot::Armor,
                _ => ItemSlot::Accessory,
            };

            // Current equipped item's effective quality.
            let current_quality = npc.equipped_items.slot_id(slot)
                .and_then(|iid| state.entity(iid))
                .and_then(|e| e.item.as_ref())
                .map(|i| i.effective_quality() * i.durability_fraction())
                .unwrap_or(0.0);

            // Find best available item for this slot.
            let mut best_idx: Option<usize> = None;
            let mut best_quality: f32 = current_quality;
            for ai in 0..item_count {
                let (_, item_slot, quality) = available_items[ai];
                if item_slot == slot && quality > best_quality {
                    best_quality = quality;
                    best_idx = Some(ai);
                }
            }

            if let Some(idx) = best_idx {
                let item_id = available_items[idx].0;

                // Unequip current item if any.
                if let Some(old_id) = npc.equipped_items.slot_id(slot) {
                    out.push(WorldDelta::UnequipItem {
                        npc_id: entity.id,
                        item_id: old_id,
                    });
                }

                // Equip the new item.
                out.push(WorldDelta::EquipItem {
                    npc_id: entity.id,
                    item_id,
                });

                // Mark item as taken (set quality to -1 so no other NPC picks it).
                available_items[idx].2 = -1.0;

                // Behavior tags for equipping.
                let mut action = ActionTags::empty();
                action.add(tags::SMITHING, 0.3);
                action.add(tags::LABOR, 0.2);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: entity.id,
                    tags: action.tags,
                    count: action.count,
                });
            }
        }
    }
}
