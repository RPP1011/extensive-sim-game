#![allow(unused)]
//! Equipment system — NPCs equip gear from settlement stockpile.
//!
//! Every EQUIP_INTERVAL ticks, NPCs at settlements check if better equipment
//! is available. If settlement stockpile has EQUIPMENT commodity, NPC consumes
//! it and upgrades their gear (weapon/armor/accessory rotating).
//!
//! Equipment quality translates to stat bonuses applied via UpdateEntityField.
//! Gear degrades over time (handled by equipment_durability.rs).
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;
use crate::world_sim::commodity;

const EQUIP_INTERVAL: u64 = 100;

/// Equipment commodity consumed per upgrade.
const EQUIP_COST: f32 = 1.0;

/// Quality gained per equipment unit consumed.
const QUALITY_PER_UNIT: f32 = 1.0;

/// Maximum quality per slot.
const MAX_QUALITY: f32 = 20.0;

pub fn compute_equipping(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % EQUIP_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_equipping_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_equipping_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % EQUIP_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // How much equipment is available?
    let equip_available = settlement.stockpile[commodity::EQUIPMENT];
    if equip_available < EQUIP_COST { return; }

    // Count NPCs that need upgrades.
    let mut upgrade_candidates: [(u32, u8); 64] = [(0, 0); 64]; // (entity_id, slot_to_upgrade)
    let mut count = 0usize;

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if count >= 64 { break; }

        // Find the weakest slot to upgrade (rotating priority).
        let slot = if npc.equipment.weapon <= npc.equipment.armor
            && npc.equipment.weapon <= npc.equipment.accessory
        {
            0 // weapon
        } else if npc.equipment.armor <= npc.equipment.accessory {
            1 // armor
        } else {
            2 // accessory
        };

        let current_quality = match slot {
            0 => npc.equipment.weapon,
            1 => npc.equipment.armor,
            _ => npc.equipment.accessory,
        };

        if current_quality >= MAX_QUALITY { continue; }

        upgrade_candidates[count] = (entity.id, slot);
        count += 1;
    }

    if count == 0 { return; }

    // Distribute available equipment among candidates (fair share).
    let units_per_npc = (equip_available / count as f32).min(EQUIP_COST);
    if units_per_npc < EQUIP_COST * 0.5 { return; } // not enough for meaningful upgrades

    let upgrades = count.min((equip_available / EQUIP_COST) as usize);

    for i in 0..upgrades {
        let (entity_id, slot) = upgrade_candidates[i];
        let quality_gain = QUALITY_PER_UNIT;

        // Consume equipment from stockpile.
        out.push(WorldDelta::ConsumeCommodity {
            location_id: settlement_id,
            commodity: commodity::EQUIPMENT,
            amount: EQUIP_COST,
        });

        // Apply stat bonus from the new equipment.
        match slot {
            0 => {
                // Weapon: +attack_damage.
                out.push(WorldDelta::UpdateEntityField {
                    entity_id,
                    field: EntityField::AttackDamage,
                    value: quality_gain, // +1 attack per quality
                });
            }
            1 => {
                // Armor: +armor + max_hp.
                out.push(WorldDelta::UpdateEntityField {
                    entity_id,
                    field: EntityField::Armor,
                    value: quality_gain * 0.5,
                });
                out.push(WorldDelta::UpdateEntityField {
                    entity_id,
                    field: EntityField::MaxHp,
                    value: quality_gain * 3.0,
                });
                out.push(WorldDelta::Heal {
                    target_id: entity_id,
                    amount: quality_gain * 3.0,
                    source_id: entity_id,
                });
            }
            _ => {
                // Accessory: +move_speed.
                out.push(WorldDelta::UpdateEntityField {
                    entity_id,
                    field: EntityField::MoveSpeed,
                    value: quality_gain * 0.02,
                });
            }
        }

        // Behavior tags: equipping is a crafting/labor action.
        let mut action = ActionTags::empty();
        action.add(tags::SMITHING, 0.3);
        action.add(tags::LABOR, 0.2);
        out.push(WorldDelta::AddBehaviorTags {
            entity_id,
            tags: action.tags,
            count: action.count,
        });
    }
}
