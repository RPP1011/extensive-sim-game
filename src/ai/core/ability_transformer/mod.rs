//! Frozen transformer inference for ability evaluation.
//!
//! Loads JSON weights exported from the Python training pipeline
//! (`training/export_weights.py`) and runs forward inference.
//!
//! Architecture: 2-layer, 4-head, d_model=64 transformer encoder with
//! [CLS] pooling → decision head (urgency + target).

mod weights;
pub mod tokenizer;
pub mod gpu_client;

pub use weights::AbilityTransformerWeights;
pub use weights::{ActorCriticWeights, ActorCriticWeightsV2, EntityState, AC_NUM_ACTIONS};
pub use weights::{ActorCriticWeightsV3, EntityStateV3, PointerOutput};
pub use weights::{ActorCriticWeightsV4, DualHeadOutput};
pub use weights::EmbeddingRegistry;
