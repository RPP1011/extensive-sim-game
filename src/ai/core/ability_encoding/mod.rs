//! Ability encoding — extract raw numeric properties from AbilityDef.
//!
//! Produces a fixed-size feature vector from any ability definition,
//! walking all effects (including delivery sub-effects) to build a
//! comprehensive property summary.

mod properties;
mod effects;

#[cfg(test)]
mod tests;

// Re-export all public items so external code sees the same interface.
pub use properties::{
    ABILITY_PROP_DIM,
    extract_ability_properties, ability_category_label,
};
