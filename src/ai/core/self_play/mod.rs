//! Self-play training pipeline.
//!
//! Per-unit policy with raw features, REINFORCE training.
//! No hand-crafted features — dump raw unit state, L1 regularization prunes.

pub mod features;
pub mod actions;
pub mod policy;
pub mod episode;

use crate::ai::core::ability_encoding::ABILITY_SLOT_DIM;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Features per unit encoding.
const UNIT_FEATURES: usize = 30;
/// Max ability slots encoded.
const MAX_ABILITIES: usize = 8;
/// Features per ability slot (legacy: 4, with encoder: ABILITY_SLOT_DIM).
const ABILITY_FEATURES_LEGACY: usize = 4;
/// Number of nearest enemies encoded.
const NUM_ENEMIES: usize = 3;
/// Number of nearest allies encoded.
const NUM_ALLIES: usize = 3;
/// Global context features.
const GLOBAL_FEATURES: usize = 5;
/// Terrain raycast directions.
const TERRAIN_RAYS: usize = 64;
/// Max raycast distance.
const TERRAIN_RAY_MAX: f32 = 15.0;

/// Legacy feature dimension (no encoder).
pub const FEATURE_DIM: usize =
    UNIT_FEATURES                           // self
    + NUM_ENEMIES * UNIT_FEATURES           // enemies
    + NUM_ALLIES * UNIT_FEATURES            // allies
    + MAX_ABILITIES * ABILITY_FEATURES_LEGACY // ability slots
    + GLOBAL_FEATURES                       // global
    + TERRAIN_RAYS;                         // terrain raycasts

/// Feature dimension with ability encoder.
pub const FEATURE_DIM_ENCODED: usize =
    UNIT_FEATURES
    + NUM_ENEMIES * UNIT_FEATURES
    + NUM_ALLIES * UNIT_FEATURES
    + MAX_ABILITIES * ABILITY_SLOT_DIM       // embedded ability slots
    + GLOBAL_FEATURES
    + TERRAIN_RAYS;

/// Number of discrete actions.
/// 0: attack nearest, 1: attack weakest, 2: attack focus
/// 3-10: use ability 0-7
/// 11: move toward, 12: move away, 13: hold
pub const NUM_ACTIONS: usize = 14;

// ---------------------------------------------------------------------------
// Re-exports — preserve the original public API
// ---------------------------------------------------------------------------

#[allow(unused_imports)]
pub use features::{extract_features, extract_features_encoded};
#[allow(unused_imports)]
pub use actions::{action_mask, action_to_intent, action_to_intent_with_focus, intent_to_action};
#[allow(unused_imports)]
pub use policy::{PolicyWeights, LayerWeights, masked_softmax, lcg_f32, Step, Episode};
#[allow(unused_imports)]
pub use episode::{run_episode, run_episode_greedy, run_episode_greedy_with_focus, write_episodes, load_policy};
