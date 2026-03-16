mod math;
mod events;
mod types;
pub mod unit_store;
mod helpers;
mod conditions;
mod targeting;
mod apply_effect;
mod apply_effect_ext;
mod summon_templates;
mod triggers;
mod damage;
mod hero;
mod intent;
mod resolve;
mod tick_systems;
mod tick_world;
mod simulation;
mod replay;
mod determinism;
mod metrics;
pub mod verify;
mod verify_checks;
pub mod oracle;
pub mod decision_log;
pub mod dataset;
pub mod ability_eval;
pub mod self_play;
pub mod curriculum;
pub mod ability_encoding;
pub mod ability_transformer;
#[cfg(any(feature = "burn-gpu", feature = "burn-cpu"))]
pub mod burn_model;
#[cfg(feature = "stream-monitor")]
pub mod monitor;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_stress;

// Re-export the public API so external code sees the same interface as before.
pub use types::*;
pub use events::SimEvent;
pub use simulation::{
    step,
    sample_duel_state, sample_duel_script,
};
pub use replay::{run_replay, ReplayResult};
pub use determinism::{
    verify_determinism, verify_replay_against_hashes,
    DeterminismReport, hash_sim_state,
};
pub use helpers::is_alive;
pub use math::{distance, move_towards, move_away, position_at_range};
pub use verify::{verify_tick, Violation, VerificationReport};
pub use unit_store::UnitStore;
