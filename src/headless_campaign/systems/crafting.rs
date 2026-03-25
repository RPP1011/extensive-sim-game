//! Resource gathering and crafting system — every 200 ticks (~20s).
//!
//! Resource nodes in guild-controlled regions auto-harvest materials into the
//! guild stockpile. Nodes deplete and slowly regenerate. When sufficient
//! materials are available, the system auto-crafts equipment that would improve
//! an adventurer's gear.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use super::class_system::effective_noncombat_stats;

// ---------------------------------------------------------------------------
// Recipe table
// ---------------------------------------------------------------------------

/// Static recipe list. Crafted items have quality `output_quality` which is
/// higher than loot drops of equivalent level, giving crafting a strategic
/// edge.
const RECIPES: &[CraftingRecipe] = &[
    CraftingRecipe {
        name: "Iron Sword",
        materials: &[(ResourceType::Iron, 30.0), (ResourceType::Wood, 10.0)],
        output_slot: EquipmentSlot::Weapon,
        output_quality: 55.0,
        stat_bonus: 8.0,
    },
    CraftingRecipe {
        name: "Crystal Staff",
        materials: &[(ResourceType::Crystal, 25.0), (ResourceType::Wood, 15.0)],
        output_slot: EquipmentSlot::Weapon,
        output_quality: 60.0,
        stat_bonus: 10.0,
    },
    CraftingRecipe {
        name: "Hide Armor",
        materials: &[(ResourceType::Hide, 30.0), (ResourceType::Herbs, 10.0)],
        output_slot: EquipmentSlot::Chest,
        output_quality: 50.0,
        stat_bonus: 7.0,
    },
    CraftingRecipe {
        name: "Iron Shield",
        materials: &[(ResourceType::Iron, 25.0), (ResourceType::Wood, 15.0)],
        output_slot: EquipmentSlot::Offhand,
        output_quality: 50.0,
        stat_bonus: 6.0,
    },
    CraftingRecipe {
        name: "Obsidian Blade",
        materials: &[(ResourceType::Obsidian, 20.0), (ResourceType::Iron, 15.0)],
        output_slot: EquipmentSlot::Weapon,
        output_quality: 75.0,
        stat_bonus: 14.0,
    },
    CraftingRecipe {
        name: "Crystal Amulet",
        materials: &[(ResourceType::Crystal, 20.0), (ResourceType::Herbs, 10.0)],
        output_slot: EquipmentSlot::Accessory,
        output_quality: 55.0,
        stat_bonus: 6.0,
    },
    CraftingRecipe {
        name: "Herbal Poultice Boots",
        materials: &[(ResourceType::Herbs, 20.0), (ResourceType::Hide, 15.0)],
        output_slot: EquipmentSlot::Boots,
        output_quality: 50.0,
        stat_bonus: 5.0,
    },
    CraftingRecipe {
        name: "Ironwood Buckler",
        materials: &[(ResourceType::Iron, 15.0), (ResourceType::Wood, 20.0)],
        output_slot: EquipmentSlot::Offhand,
        output_quality: 45.0,
        stat_bonus: 5.0,
    },
    CraftingRecipe {
        name: "Obsidian Staff",
        materials: &[(ResourceType::Obsidian, 25.0), (ResourceType::Crystal, 15.0)],
        output_slot: EquipmentSlot::Weapon,
        output_quality: 80.0,
        stat_bonus: 16.0,
    },
    CraftingRecipe {
        name: "Dragonhide Vest",
        materials: &[(ResourceType::Hide, 25.0), (ResourceType::Obsidian, 10.0)],
        output_slot: EquipmentSlot::Chest,
        output_quality: 65.0,
        stat_bonus: 10.0,
    },
];

// ---------------------------------------------------------------------------
// Resource node generation
// ---------------------------------------------------------------------------

/// Generate 2-3 resource nodes per region using deterministic RNG.
pub fn generate_resource_nodes(regions: &[Region], seed: u64) -> Vec<ResourceNode> {
    let mut rng = seed.wrapping_add(0xC4AF_7000_0000_0000u64.wrapping_mul(3));
    let mut nodes = Vec::new();
    let mut next_id = 1u32;

    for region in regions {
        // 2-3 nodes per region
        let node_count = 2 + (lcg_next(&mut rng) % 2) as usize;

        for _ in 0..node_count {
            let type_idx = (lcg_next(&mut rng) % 6) as usize;
            let resource_type = ResourceType::ALL[type_idx];
            let max_amount = 40.0 + (lcg_next(&mut rng) % 61) as f32; // 40-100
            let regen_rate = 0.01 + (lcg_next(&mut rng) % 30) as f32 * 0.001; // 0.01-0.04

            nodes.push(ResourceNode {
                id: next_id,
                region_id: region.id,
                resource_type,
                amount: max_amount, // start full
                regen_rate,
                max_amount,
            });
            next_id += 1;
        }
    }

    nodes
}

// ---------------------------------------------------------------------------
// Tick system
// ---------------------------------------------------------------------------

/// Runs every 200 ticks: harvest from nodes, regenerate, auto-craft.
pub fn tick_crafting(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;

    // --- Decay gathering boosts ---
    state.gathering_boosts.retain(|_, ticks| {
        if *ticks <= 200 {
            false
        } else {
            *ticks -= 200;
            true
        }
    });

    // --- Harvest from resource nodes in guild-controlled regions ---
    // Collect region info we need: (region_id, control, is_guild_owned)
    let region_info: Vec<(usize, f32, bool)> = state
        .overworld
        .regions
        .iter()
        .map(|r| (r.id, r.control, r.owner_faction_id == guild_faction_id))
        .collect();

    let gathering_boosts = state.gathering_boosts.clone();

    for node in &mut state.resource_nodes {
        // Only harvest from guild-controlled regions
        let Some(&(_, control, is_guild)) = region_info.iter().find(|(id, _, _)| *id == node.region_id) else {
            continue;
        };
        if !is_guild {
            // Regenerate nodes in non-guild regions (slower)
            node.amount = (node.amount + node.regen_rate * 0.5).min(node.max_amount);
            continue;
        }

        // Harvest amount scales with region control (0-100 → 0.0-1.0)
        let control_factor = control / 100.0;
        let boost = if gathering_boosts.contains_key(&node.region_id) {
            2.0
        } else {
            1.0
        };
        let harvest = (node.regen_rate * control_factor * boost * 200.0).min(node.amount);

        if harvest > 0.01 {
            node.amount -= harvest;
            *state
                .resources
                .entry(node.resource_type)
                .or_insert(0.0) += harvest;

            events.push(WorldEvent::ResourceGathered {
                resource: format!("{:?}", node.resource_type),
                amount: harvest,
            });
        }

        // Regenerate (always, even guild regions)
        node.amount = (node.amount + node.regen_rate * 200.0 * 0.3).min(node.max_amount);
    }

    // --- Auto-craft: find recipes that would improve an adventurer's gear ---
    // Determine worst equipped slot quality per adventurer.
    // Equipment stores item IDs; look them up in inventory to get quality.
    let mut worst_slot_quality: Option<(EquipmentSlot, f32)> = None;

    // Build a quick lookup of item qualities by ID
    let item_quality: std::collections::HashMap<u32, f32> = state
        .guild
        .inventory
        .iter()
        .map(|i| (i.id, i.quality))
        .collect();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let slots: [(EquipmentSlot, &Option<u32>); 5] = [
            (EquipmentSlot::Weapon, &adv.equipment.weapon),
            (EquipmentSlot::Offhand, &adv.equipment.offhand),
            (EquipmentSlot::Chest, &adv.equipment.chest),
            (EquipmentSlot::Boots, &adv.equipment.boots),
            (EquipmentSlot::Accessory, &adv.equipment.accessory),
        ];
        for (slot, item_id) in &slots {
            let current_quality = item_id
                .and_then(|id| item_quality.get(&id).copied())
                .unwrap_or(0.0);
            if worst_slot_quality
                .as_ref()
                .map_or(true, |(_, q)| current_quality < *q)
            {
                worst_slot_quality = Some((*slot, current_quality));
            }
        }
    }

    // Try to craft one item per tick that targets the worst slot
    if let Some((worst_slot, worst_quality)) = worst_slot_quality {
        for recipe in RECIPES {
            if recipe.output_slot != worst_slot {
                continue;
            }
            if recipe.output_quality <= worst_quality {
                continue;
            }
            // Check materials
            let can_craft = recipe.materials.iter().all(|(rt, needed)| {
                state.resources.get(rt).copied().unwrap_or(0.0) >= *needed
            });
            if !can_craft {
                continue;
            }

            // Consume materials
            for (rt, needed) in recipe.materials {
                *state.resources.entry(*rt).or_insert(0.0) -= needed;
            }

            // Create item
            let item_id = state.next_event_id;
            state.next_event_id += 1;

            // Crafting bonus from adventurer class stats improves quality and stats
            let crafting_bonus: f32 = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead && a.faction_id.is_none())
                .map(|a| effective_noncombat_stats(a).2) // crafting component
                .sum::<f32>()
                .min(30.0); // Cap at +30 to prevent runaway quality
            let boosted_stat = recipe.stat_bonus + crafting_bonus * 0.5;
            let boosted_quality = recipe.output_quality + crafting_bonus;

            let stat_bonuses = match recipe.output_slot {
                EquipmentSlot::Weapon => StatBonuses {
                    attack_bonus: boosted_stat,
                    ..Default::default()
                },
                EquipmentSlot::Offhand | EquipmentSlot::Chest => StatBonuses {
                    defense_bonus: boosted_stat,
                    ..Default::default()
                },
                EquipmentSlot::Boots => StatBonuses {
                    speed_bonus: boosted_stat,
                    ..Default::default()
                },
                EquipmentSlot::Accessory => StatBonuses {
                    hp_bonus: boosted_stat * 5.0,
                    ..Default::default()
                },
            };

            let quality_label = if boosted_quality >= 70.0 {
                "Rare"
            } else if boosted_quality >= 50.0 {
                "Uncommon"
            } else {
                "Common"
            };

            state.guild.inventory.push(InventoryItem {
                id: item_id,
                name: recipe.name.to_string(),
                slot: recipe.output_slot,
                quality: boosted_quality,
                stat_bonuses,
                durability: 100.0,
                appraised: false,
            });

            events.push(WorldEvent::ItemCrafted {
                name: recipe.name.to_string(),
                quality: quality_label.to_string(),
            });

            // Only craft one item per tick
            break;
        }
    }
}
