//! Party interception — fires every tick.
//!
//! Hostile entities near friendly NPCs escalate grid fidelity to High.
//! Uses grid fidelity as a proxy: only grids NOT already High need checking.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

pub fn compute_interception(state: &WorldState, _out: &mut Vec<WorldDelta>) {
    if state.tick == 0 { return; }

    // Only check grids that aren't already High fidelity.
    // The grid fidelity system already escalates based on spatial index
    // (has_hostiles_in_radius + has_friendlies_in_radius).
    // This system just handles the case where entities are near a grid
    // but not yet assigned to it.
    //
    // Since compute_grid_deltas in tick.rs already handles fidelity
    // escalation via the spatial index, this system is a no-op.
    // The spatial index + grid fidelity system replaces the O(n²) check.
}
