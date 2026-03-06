//! Oracle dataset generator — exports per-unit per-tick training samples.
//!
//! Each sample: per-unit features → action class label (0..9).
//! Action classes abstract away specific target IDs so the model generalizes.

mod actions;
mod features;
mod training;

#[allow(unused_imports)]
pub use actions::{
    ActionClass, CombatActionClass,
    classify_action, classify_combat_action, classify_action_raw,
};
#[allow(unused_imports)]
pub use features::{FEATURE_DIM, extract_unit_features};
#[allow(unused_imports)]
pub use training::{
    TrainingSample, CombatTrainingSample, RawTrainingSample,
    generate_dataset, generate_combat_dataset, generate_raw_dataset,
    write_dataset, write_combat_dataset, write_raw_dataset,
};
