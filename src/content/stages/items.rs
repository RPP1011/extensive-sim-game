use std::collections::HashMap;

use crate::content::schema::ItemContent;
use crate::content::aot_pipeline::{GenerationStage, Lcg, PipelineError, StageId, WorldContext};
use crate::model_backend::ModelClient;

const ITEM_PREFIXES: &[&str] = &[
    "Rusty", "Enchanted", "Ancient", "Blessed", "Cursed", "Masterwork",
    "Shadowforged", "Ironbound", "Gilded", "Runescribed",
];

const ITEM_TYPES: &[(&str, &[(&str, f32, f32)])] = &[
    (
        "Sword",
        &[("attack_power", 8.0, 25.0), ("attack_speed", 0.8, 1.2)],
    ),
    (
        "Shield",
        &[("defense", 5.0, 20.0), ("block_chance", 0.1, 0.3)],
    ),
    (
        "Staff",
        &[("ability_power", 10.0, 30.0), ("mana_regen", 1.0, 5.0)],
    ),
    (
        "Armor",
        &[("defense", 10.0, 30.0), ("hp_bonus", 20.0, 100.0)],
    ),
    (
        "Ring",
        &[("ability_power", 3.0, 12.0), ("heal_power", 2.0, 10.0)],
    ),
    (
        "Potion",
        &[("heal_amount", 20.0, 80.0)],
    ),
];

const RARITIES: &[&str] = &["common", "uncommon", "rare", "epic", "legendary"];
const RARITY_WEIGHTS: &[f32] = &[0.40, 0.30, 0.18, 0.09, 0.03];

pub struct ItemStage;

impl GenerationStage for ItemStage {
    fn id(&self) -> StageId {
        StageId::Items
    }

    fn run(
        &self,
        ctx: &mut WorldContext,
        model: Option<&ModelClient>,
        rng: &mut Lcg,
    ) -> Result<(), PipelineError> {
        if model.is_some() && model.unwrap().is_available() {
            // Model-backed placeholder.
        }

        let count = rng.next_usize_range(8, 15);

        for _ in 0..count {
            let prefix = rng.choose(ITEM_PREFIXES);
            let (type_name, stat_defs) = rng.choose(ITEM_TYPES);
            let name = format!("{} {}", prefix, type_name);

            // Weighted rarity selection.
            let roll = rng.next_f32();
            let mut cumulative = 0.0;
            let mut rarity = RARITIES[0];
            for (i, &weight) in RARITY_WEIGHTS.iter().enumerate() {
                cumulative += weight;
                if roll < cumulative {
                    rarity = RARITIES[i];
                    break;
                }
            }

            // Rarity multiplier for stat scaling.
            let rarity_mult = match rarity {
                "common" => 1.0,
                "uncommon" => 1.2,
                "rare" => 1.5,
                "epic" => 1.8,
                "legendary" => 2.2,
                _ => 1.0,
            };

            let mut stats = HashMap::new();
            for &(stat_name, lo, hi) in *stat_defs {
                let base = lo + rng.next_f32() * (hi - lo);
                stats.insert(stat_name.to_string(), (base * rarity_mult * 10.0).round() / 10.0);
            }

            let description = format!(
                "A {} {} of {} quality.",
                rarity, type_name.to_lowercase(), prefix.to_lowercase()
            );

            ctx.items.push(ItemContent {
                name,
                description,
                rarity: rarity.to_string(),
                stats,
            });
        }

        Ok(())
    }
}
