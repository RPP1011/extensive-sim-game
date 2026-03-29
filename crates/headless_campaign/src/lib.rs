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

pub mod ability_gen;
pub mod action_meta;
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
pub mod quest_gen;
pub mod quest_hooks;
/// DEPRECATED: Legacy `SkillEffect` dispatch. Use `unified_dispatch` instead.
pub mod skill_effects;
/// DEPRECATED: Legacy `SkillEffect` templates. New skills should be `.ability` files
/// in `dataset/abilities/campaign/` loaded via the unified DSL.
pub mod skill_templates;
pub mod state;
/// Unified campaign effect dispatch using `tactical_sim::effects::Effect`.
/// This is the replacement for `skill_effects.rs`. Migration path:
/// 1. Change `GrantedSkill.skill_effect` from `Option<SkillEffect>` to `Option<Effect>`
/// 2. Update `skill_templates.rs` to construct `Effect` variants (or load from .ability files)
/// 3. Replace `apply_skill_effect()` calls with `campaign_apply_effect()`
/// 4. Remove `skill_effects.rs` and the `SkillEffect` enum
pub mod unified_dispatch;
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
pub mod world_templates;
pub mod world_lore_gen;
mod systems;
#[cfg(test)]
mod tests;

pub use actions::{CampaignAction, CampaignStepResult, WorldEvent, ActionResult, StepDeltas};
pub use state::{CampaignState, CampaignOutcome, CAMPAIGN_TICK_MS, CAMPAIGN_TURN_SECS};
pub use step::step_campaign;
pub use step::step_world;
pub use step::seed_world_population;
