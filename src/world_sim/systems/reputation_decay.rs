#![allow(unused)]
//! Reputation decay system — delta architecture port.
//!
//! Guild/faction reputation decays over time unless actively maintained.
//! Decay is accelerated by inactivity, corruption, losses, and negative
//! rumors, and reduced by recent victories, propaganda, and guild tier.
//!
//! Original: `crates/headless_campaign/src/systems/reputation_decay.rs`
//! Cadence: every 7 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

//   where ReputationHistory { recent_deeds: Vec<ReputationDeed>, trend: ReputationTrend, maintenance_cost: f32 }

/// Cadence: reputation decay every 7 ticks.
const TICK_CADENCE: u64 = 7;

/// Base reputation decay per cycle.
const BASE_DECAY: f32 = 0.1;

/// Reputation floor — won't decay below this.
const REPUTATION_FLOOR: f32 = 10.0;

/// Ticks without quest completion before decay accelerates.
const INACTIVITY_THRESHOLD: u64 = 67;

/// Corruption threshold above which scandal penalty applies.
const CORRUPTION_THRESHOLD: f32 = 50.0;

/// Recent ticks to look back for battle losses.
const RECENT_LOSS_WINDOW: u64 = 33;

/// Recent ticks to look back for quest completions (reducer).
const RECENT_QUEST_WINDOW: u64 = 17;

/// Compute reputation decay deltas.
///
/// Reads guild reputation, quest history, corruption, rumors, propaganda,
/// festivals, and guild tier to calculate net reputation decay. Emits a
/// single delta for the net change (if any).
///
/// Since WorldState lacks guild/reputation fields, this is a structural
/// placeholder. The settlement treasury can serve as a partial proxy:
/// settlements with high treasury lose less reputation (they're prosperous).
pub fn compute_reputation_decay(_state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Stub: guild reputation state not yet tracked. See git history for planned design.
}

// ---------------------------------------------------------------------------
// Reputation trend tracking (pure query)
// ---------------------------------------------------------------------------

/// Reputation trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReputationTrend {
    Rising,
    Stable,
    Falling,
}

/// Thresholds for trend classification.
const TRAJECTORY_RISING_THRESHOLD: f32 = 0.5;
const TRAJECTORY_FALLING_THRESHOLD: f32 = -0.5;

/// Classify reputation trend from a net change over a window.
pub fn classify_trend(net_change: f32) -> ReputationTrend {
    if net_change > TRAJECTORY_RISING_THRESHOLD {
        ReputationTrend::Rising
    } else if net_change < TRAJECTORY_FALLING_THRESHOLD {
        ReputationTrend::Falling
    } else {
        ReputationTrend::Stable
    }
}
