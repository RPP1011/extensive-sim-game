//! Campaign systems ported to the world sim delta architecture.
//!
//! Each system reads the WorldState snapshot and pushes WorldDeltas.
//! Systems are registered here and called from the runtime compute phase.

use super::delta::WorldDelta;
use super::state::WorldState;

/// Run all registered world sim systems, pushing deltas into `out`.
pub fn compute_all_systems(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Systems are called in priority order.
    // Each reads the same frozen snapshot and pushes deltas independently.

    // -- placeholder: systems will be registered here as they're migrated --
}
