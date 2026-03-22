//! Actor-Critic models defined in Burn.
//!
//! Single model definition that works for both:
//! - CPU inference (NdArray backend)
//! - GPU training (LibTorch + Autodiff backend)
//!
//! V5: base model (entity encoder + cross-attention + CfC + decision heads)
//! V6: adds latent interface, spatial cross-attention, value head, h_dim=256

pub mod config;
pub mod entity_encoder;
pub mod cross_attention;
pub mod cfc_cell;
pub mod combat_head;
pub mod actor_critic;
pub mod latent_interface;
pub mod spatial_cross_attn;
pub mod value_head;
pub mod actor_critic_v6;

pub use actor_critic::{
    ActorCriticV5, ActorCriticV5Config,
    ActorCriticOutput, EncodedState,
    AbilityTransformer, AbilityTransformerConfig,
};
pub use entity_encoder::{EntityEncoderV5, EntityEncoderV5Config};
pub use cross_attention::{CrossAttentionBlock, CrossAttentionConfig};
pub use cfc_cell::{CfCCell, CfCCellConfig};
pub use combat_head::{CombatPointerHead, CombatPointerHeadConfig, CombatOutput};
pub use latent_interface::{LatentInterface, LatentInterfaceConfig};
pub use spatial_cross_attn::{SpatialCrossAttention, SpatialCrossAttentionConfig, CORNER_DIM, MAX_CORNERS};
pub use value_head::{ValueHead, ValueHeadConfig, ValueOutput};
pub use actor_critic_v6::{ActorCriticV6, ActorCriticV6Config};

pub mod checkpoint;

#[cfg(feature = "burn-gpu")]
pub mod inference;
#[cfg(feature = "burn-gpu")]
pub mod inference_v6;
pub mod training;

#[cfg(test)]
mod tests;
