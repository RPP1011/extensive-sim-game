mod math;
mod events;
mod types;
mod helpers;
mod conditions;
mod targeting;
mod apply_effect;
mod apply_effect_ext;
mod triggers;
mod damage;
mod hero;
mod intent;
mod resolve;
mod tick_systems;
mod tick_world;
mod simulation;
mod metrics;
pub mod oracle;
pub mod decision_log;
pub mod dataset;
pub mod ability_eval;
pub mod self_play;
pub mod curriculum;
pub mod ability_encoding;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_stress;

// Re-export the public API so external code sees the same interface as before.
pub use types::*;
pub use events::SimEvent;
pub use simulation::{
    step, run_replay, ReplayResult,
    sample_duel_state, sample_duel_script,
};
pub use helpers::is_alive;
pub use math::{distance, move_towards, move_away, position_at_range};
