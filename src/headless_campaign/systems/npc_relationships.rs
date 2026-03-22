//! NPC relationship drift — every 300 ticks (~30s).
//!
//! Relationship scores drift toward neutral over time.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::CampaignState;

pub fn tick_npc_relationships(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    _events: &mut Vec<WorldEvent>,
) {
    if state.tick % 300 != 0 || state.tick == 0 {
        return;
    }

    for rel in &mut state.npc_relationships {
        // Drift toward 0 (neutral)
        if rel.relationship_score > 0.0 {
            rel.relationship_score = (rel.relationship_score - 0.5).max(0.0);
        } else if rel.relationship_score < 0.0 {
            rel.relationship_score = (rel.relationship_score + 0.5).min(0.0);
        }

        // Update rescue availability based on relationship
        rel.rescue_available = rel.relationship_score > 50.0;
    }
}
