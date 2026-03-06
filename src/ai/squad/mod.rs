pub mod personality;
pub(crate) mod forces;
pub(crate) mod state;
pub(crate) mod combat;
pub(crate) mod intents;
pub mod replay;

#[cfg(test)]
mod tests;

// Re-export all public API (used by bin crates)
#[allow(unused_imports)]
pub use state::{FormationMode, SquadAiState, SquadBlackboard};
#[allow(unused_imports)]
pub use personality::Personality;
pub use intents::{generate_intents, generate_intents_with_terrain};
pub use replay::{
    run_phase3_sample, sample_phase3_party_state,
};
