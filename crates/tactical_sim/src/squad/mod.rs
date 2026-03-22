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

/// Public wrapper: evaluate hero abilities against a target.
///
/// Used by the student pipeline to integrate ability usage into the hero
/// tactical override without going through the full squad AI intent generation.
pub fn combat_evaluate_hero_ability(
    state: &crate::sim::SimState,
    unit_id: u32,
    target_id: u32,
) -> Option<crate::sim::IntentAction> {
    combat::abilities::evaluate_hero_ability(
        state, unit_id, target_id, FormationMode::Advance,
        &state::TickContext::new(state),
    )
}
