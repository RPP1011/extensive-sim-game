//! NPC relationship drift — every 300 ticks (~30s).
//!
//! Relationship scores drift toward neutral over time.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::CampaignState;

pub fn tick_npc_relationships(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    _events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.npc_relationships;
    if state.tick % cfg.drift_interval_ticks != 0 || state.tick == 0 {
        return;
    }

    let drift_rate = cfg.drift_rate;
    let rescue_threshold = cfg.rescue_threshold;

    for rel in &mut state.npc_relationships {
        // Drift toward 0 (neutral)
        if rel.relationship_score > 0.0 {
            rel.relationship_score = (rel.relationship_score - drift_rate).max(0.0);
        } else if rel.relationship_score < 0.0 {
            rel.relationship_score = (rel.relationship_score + drift_rate).min(0.0);
        }

        // Update rescue availability based on relationship
        rel.rescue_available = rel.relationship_score > rescue_threshold;
    }
}
