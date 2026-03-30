#![allow(unused)]
//! Equipment durability — degrades equipped items every 50 ticks.
//!
//! Items lose durability through use:
//! - NPCs in combat (High-fidelity grids): weapon and armor items degrade.
//! - NPCs traveling/trading/adventuring: accessory items degrade.
//! - When durability reaches 0: item breaks → unequipped, stat bonuses removed.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::fidelity::Fidelity;
use crate::world_sim::state::{EconomicIntent, Entity, EntityKind, ItemSlot, WorldState};

/// Durability check interval (ticks).
const DURABILITY_INTERVAL: u64 = 50;

/// Combat degradation: durability lost per interval for weapon/armor.
const COMBAT_DURABILITY_LOSS: f32 = 1.0;

/// Travel degradation: durability lost per interval for accessories.
const TRAVEL_DURABILITY_LOSS: f32 = 0.5;

/// Don't degrade if the settlement treasury is below this threshold.
const ECONOMY_COLLAPSE_THRESHOLD: f32 = -100.0;

pub fn compute_equipment_durability(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DURABILITY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_equipment_durability_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_equipment_durability_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    // Gate: skip degradation if settlement economy has collapsed.
    let treasury = state.settlement(settlement_id)
        .map(|s| s.treasury)
        .unwrap_or(0.0);
    if treasury <= ECONOMY_COLLAPSE_THRESHOLD {
        return;
    }

    // Collect durability changes: (item_entity_id, durability_loss, npc_id).
    // Applied post-loop to avoid borrow issues.
    let mut degradations: Vec<(u32, f32, u32)> = Vec::new();

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Check if the entity is on a High-fidelity grid (combat).
        let in_combat = entity.grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.fidelity == Fidelity::High)
            .unwrap_or(false);

        if in_combat {
            // Degrade weapon and armor.
            if let Some(wid) = npc.equipped_items.weapon_id {
                degradations.push((wid, COMBAT_DURABILITY_LOSS, entity.id));
            }
            if let Some(aid) = npc.equipped_items.armor_id {
                degradations.push((aid, COMBAT_DURABILITY_LOSS, entity.id));
            }
        }

        match &npc.economic_intent {
            EconomicIntent::Trade { .. }
            | EconomicIntent::Travel { .. }
            | EconomicIntent::Adventuring { .. } => {
                // Degrade accessory.
                if let Some(aid) = npc.equipped_items.accessory_id {
                    degradations.push((aid, TRAVEL_DURABILITY_LOSS, entity.id));
                }
            }
            _ => {}
        }
    }

    // Apply degradation: reduce durability, unequip if broken.
    // We can't emit UnequipItem deltas here (that would need full entity lookup),
    // so durability changes and break detection happen in advance_durability (post-apply).
    // For the delta architecture, we store the degradation info and process in post-apply.
    // However since durability is on item entities we need to handle it directly.
    // We'll use the existing delta system: emit nothing here (durability is handled post-apply).
    // Actually, we emit the degradation amounts as metadata stored for the post-apply phase.

    // Since durability changes need mutable access to item entities, this system
    // doesn't emit deltas. Instead, advance_item_durability runs post-apply.
}

/// Post-apply: degrade item durability and unequip broken items.
/// Called from runtime.rs.
pub fn advance_item_durability(state: &mut WorldState) {
    if state.tick % DURABILITY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Collect items to degrade: (item_id, loss).
    let mut degradations: Vec<(u32, f32)> = Vec::new();

    let entity_count = state.entities.len();
    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Check settlement treasury.
        let treasury_ok = npc.home_settlement_id
            .and_then(|sid| state.settlement(sid))
            .map(|s| s.treasury > ECONOMY_COLLAPSE_THRESHOLD)
            .unwrap_or(true);
        if !treasury_ok { continue; }

        let in_combat = entity.grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.fidelity == Fidelity::High)
            .unwrap_or(false);

        if in_combat {
            if let Some(wid) = npc.equipped_items.weapon_id {
                degradations.push((wid, COMBAT_DURABILITY_LOSS));
            }
            if let Some(aid) = npc.equipped_items.armor_id {
                degradations.push((aid, COMBAT_DURABILITY_LOSS));
            }
        }

        match &npc.economic_intent {
            EconomicIntent::Trade { .. }
            | EconomicIntent::Travel { .. }
            | EconomicIntent::Adventuring { .. } => {
                if let Some(aid) = npc.equipped_items.accessory_id {
                    degradations.push((aid, TRAVEL_DURABILITY_LOSS));
                }
            }
            _ => {}
        }
    }

    // Apply durability loss.
    for (item_id, loss) in &degradations {
        if let Some(item_entity) = state.entities.iter_mut().find(|e| e.id == *item_id) {
            if let Some(item) = &mut item_entity.item {
                item.durability = (item.durability - loss).max(0.0);
            }
        }
    }

    // Find broken items (durability <= 0) and unequip them.
    let mut broken_items: Vec<(u32, u32, ItemSlot)> = Vec::new(); // (item_id, owner_id, slot)
    for entity in &state.entities {
        if entity.kind != EntityKind::Item { continue; }
        let item = match &entity.item { Some(i) => i, None => continue };
        if item.durability > 0.0 { continue; }
        if let Some(owner_id) = item.owner_id {
            broken_items.push((entity.id, owner_id, item.slot));
        }
    }

    // Unequip broken items: remove stat bonuses and clear slot.
    for (item_id, owner_id, slot) in &broken_items {
        // Get stat bonuses to remove (from the item's quality at 0 durability = 0 bonuses,
        // but we need the bonuses that were applied when equipped at full durability).
        // Since durability is now 0, bonuses were already scaling down each tick through
        // the stat system. We just need to clear the slot and remove residual bonuses.
        //
        // Actually, stats are set once on equip and don't track durability changes.
        // We need to recalculate what bonuses the item provided and remove them.
        let item_bonuses = state.entities.iter()
            .find(|e| e.id == *item_id)
            .and_then(|e| e.item.as_ref())
            .map(|i| {
                // Calculate original bonuses (at full durability).
                let eq = i.effective_quality();
                match i.slot {
                    ItemSlot::Weapon => (eq, 0.0, 0.0, 0.0),
                    ItemSlot::Armor => (0.0, eq * 0.5, eq * 3.0, 0.0),
                    ItemSlot::Accessory => (0.0, 0.0, 0.0, eq * 0.02),
                }
            });

        if let Some((attack, armor, hp, speed)) = item_bonuses {
            if let Some(owner) = state.entities.iter_mut().find(|e| e.id == *owner_id) {
                owner.attack_damage = (owner.attack_damage - attack).max(0.0);
                owner.armor = (owner.armor - armor).max(0.0);
                owner.max_hp = (owner.max_hp - hp).max(1.0);
                owner.hp = owner.hp.min(owner.max_hp);
                owner.move_speed = (owner.move_speed - speed).max(0.5);
                if let Some(npc) = &mut owner.npc {
                    npc.equipped_items.set_slot(*slot, None);
                }
            }
        }

        // Clear item owner.
        if let Some(item_entity) = state.entities.iter_mut().find(|e| e.id == *item_id) {
            if let Some(item) = &mut item_entity.item {
                item.owner_id = None;
            }
            // Mark broken items as not alive so they get cleaned up.
            item_entity.alive = false;
        }
    }
}
