//! Ability transformer inference and training.
//!
//! Two implementations are provided:
//!
//! 1. **`weights`** (always available) — Hand-rolled frozen inference from JSON
//!    weights exported by `training/export_weights.py`. Zero external ML
//!    dependencies, SIMD-friendly, used at game runtime.
//!
//! 2. **`burn_models`** (behind `burn-training` feature) — Full Burn port of
//!    `training/model.py`. Supports autodiff training and can replace the
//!    entire Python training pipeline.
//!
//! Architecture: 2-layer, 4-head, d_model=64 transformer encoder with
//! [CLS] pooling → decision head (urgency + target).

mod weights;
pub mod tokenizer;

#[cfg(feature = "burn-training")]
pub mod burn_models;

pub use weights::AbilityTransformerWeights;
pub use weights::{ActorCriticWeights, ActorCriticWeightsV2, EntityState, AC_NUM_ACTIONS};
pub use weights::{ActorCriticWeightsV3, EntityStateV3, PointerOutput};
