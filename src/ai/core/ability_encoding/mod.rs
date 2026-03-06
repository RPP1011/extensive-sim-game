//! Ability encoding — extract raw numeric properties from AbilityDef.
//!
//! Produces a fixed-size feature vector from any ability definition,
//! walking all effects (including delivery sub-effects) to build a
//! comprehensive property summary. Used as input to the ability
//! embedding encoder.

mod properties;
mod autoencoder;
mod effects;
mod training;

#[cfg(test)]
mod tests;

// Re-export all public items so external code sees the same interface.
#[allow(unused_imports)]
pub use properties::{
    ABILITY_PROP_DIM, ABILITY_EMBED_DIM, ABILITY_SLOT_DIM,
    extract_ability_properties, ability_category_label,
};
#[allow(unused_imports)]
pub use autoencoder::{
    TwoLayerWeights, AbilityEncoder, AbilityDecoder,
    load_autoencoder,
};
#[allow(unused_imports)]
pub use training::{
    AbilityTrainingRow, extract_training_rows,
};
