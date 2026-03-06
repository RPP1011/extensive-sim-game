//! Training data export: AbilityTrainingRow, extract_training_rows.

use crate::ai::effects::AbilityTargeting;
use crate::ai::core::ability_eval::AbilityCategory;

use super::properties::{ABILITY_PROP_DIM, extract_ability_properties, ability_category_label};

/// Export all abilities from loaded TOML templates as training rows.
/// Each row: (properties: [f32; 80], category_index: usize, targeting_index: usize, hero_name: String, ability_name: String)
pub struct AbilityTrainingRow {
    pub properties: [f32; ABILITY_PROP_DIM],
    pub category: AbilityCategory,
    pub category_index: usize,
    pub targeting_index: usize,
    pub hero_name: String,
    pub ability_name: String,
}

pub fn extract_training_rows(
    templates: &[(String, crate::ai::effects::HeroToml)],
) -> Vec<AbilityTrainingRow> {
    let mut rows = Vec::new();
    for (_path, toml) in templates {
        let hero_name = toml.hero.name.clone();
        for def in &toml.abilities {
            let props = extract_ability_properties(def);
            let cat = ability_category_label(def);
            let cat_idx = cat as usize;
            let tgt_idx = match def.targeting {
                AbilityTargeting::TargetEnemy => 0,
                AbilityTargeting::TargetAlly => 1,
                AbilityTargeting::SelfCast => 2,
                AbilityTargeting::SelfAoe => 3,
                AbilityTargeting::GroundTarget => 4,
                AbilityTargeting::Direction => 5,
                AbilityTargeting::Vector => 6,
                AbilityTargeting::Global => 7,
            };
            rows.push(AbilityTrainingRow {
                properties: props,
                category: cat,
                category_index: cat_idx,
                targeting_index: tgt_idx,
                hero_name: hero_name.clone(),
                ability_name: def.name.clone(),
            });
        }
    }
    rows
}
