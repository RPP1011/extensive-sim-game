#![allow(dead_code, unused_variables, unused_imports)]
//! Headless campaign — retained modules for ability generation and state types.
//!
//! Most of this crate was migrated to `src/world_sim/`. What remains:
//! - `ability_gen` — procedural ability generation (used by class_gen.rs)
//! - `grammar_space` — DSL → vector encoding (used by class_gen.rs)
//! - `ability_quality` — ability scoring (used by class_gen.rs)
//! - `state` — CampaignState types (used by bridge.rs for campaign↔world translation)

pub mod ability_gen;
pub mod ability_quality;
pub mod grammar_space;
pub mod state;

// Internal dependencies of the above:
pub mod class_dsl;
pub mod combat_oracle;
pub mod config;
pub mod vae_slots;

pub use state::{CampaignState, CampaignOutcome};
