//! Path-following system — NPCs walk tile-by-tile along cached A* paths.
//!
//! CityGrid pathfinding has been removed. NPCs now use world-space move_target
//! directly. The tile system handles movement cost modifiers in movement.rs.
//!
//! Cadence: every tick (movement is per-tick for smooth tile-by-tile walking).

use crate::world_sim::state::*;

/// Advance NPCs along their cached grid paths.
/// Called post-apply from runtime.rs.
/// Currently a no-op — CityGrid A* pathfinding has been removed.
/// NPCs use move_target set directly by the goal/action system.
pub fn advance_pathfinding(_state: &mut WorldState) {
    // CityGrid pathfinding removed — NPCs use world-space move_target directly.
}
