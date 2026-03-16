//! V5 model configuration constants.

/// Entity feature dimension (30 base + 4 spatial).
pub const ENTITY_DIM: usize = 34;
/// Unified zone token dimension.
pub const ZONE_DIM: usize = 12;
/// Aggregate feature dimension.
pub const AGG_DIM: usize = 16;
/// Number of entity types: self=0, enemy=1, ally=2, zone=3, aggregate=4.
pub const NUM_ENTITY_TYPES: usize = 5;
/// Maximum abilities per unit.
pub const MAX_ABILITIES: usize = 8;
/// Number of combat action types.
pub const NUM_COMBAT_TYPES: usize = 10;

/// Default model hyperparameters.
pub const D_MODEL: usize = 128;
pub const N_HEADS: usize = 8;
pub const D_FF: usize = 256;
pub const N_ENCODER_LAYERS: usize = 4;
pub const N_LATENT_TOKENS: usize = 12;
pub const N_LATENT_BLOCKS: usize = 2;
pub const H_DIM: usize = 64;
