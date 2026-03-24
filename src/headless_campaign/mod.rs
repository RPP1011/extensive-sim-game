//! Headless campaign simulator for automated playtesting.
//!
//! Provides a pure-data campaign simulation with no Bevy ECS dependency.
//! Designed for:
//! - MCTS tree search (state is `Clone`, deterministic, fast to step)
//! - Automated playtesting (no GUI required)
//! - Balance probing (MAP-Elites, evolutionary search)
//! - CI regression testing
//!
//! # Usage
//!
//! ```rust,ignore
//! use bevy_game::headless_campaign::{CampaignState, step_campaign, CampaignAction};
//!
//! let mut state = CampaignState::default_test_campaign(42);
//! for _ in 0..1000 {
//!     let result = step_campaign(&mut state, None);
//!     if result.outcome.is_some() { break; }
//! }
//! ```

pub mod actions;
pub mod backstory;
pub mod batch;
pub mod bfs_explore;
pub mod choice_templates;
pub mod class_dsl;
pub mod crisis_templates;
pub mod heuristic_bc;
pub mod combat_oracle;
pub mod content_prompts;
pub mod config;
pub mod fuzz;
pub mod llm;
pub mod mcts;
pub mod quest_hooks;
pub mod state;
pub mod step;
pub mod tokens;
pub mod unit_tiers;
pub mod vae_dataset;
pub mod vae_features;
pub mod vae_gt_dataset;
pub mod vae_inference;
pub mod vae_serialize;
pub mod vae_slots;
pub mod trace;
pub mod trace_viewer;
pub mod world_templates;
mod systems;
#[cfg(test)]
mod tests;

pub use actions::{CampaignAction, CampaignStepResult, WorldEvent, ActionResult, StepDeltas};
pub use state::{CampaignState, CampaignOutcome, CAMPAIGN_TICK_MS};
pub use step::step_campaign;
