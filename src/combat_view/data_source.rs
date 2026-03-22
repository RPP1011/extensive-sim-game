//! Trait abstraction for combat data sources.
//!
//! Allows `CombatView` to work with both live combat (`MissionSimState`)
//! and replay data (`LastMissionReplay`) without coupling to either.

use crate::ai::core::SimState;
use crate::ai::pathing::GridNav;

/// Provides read-only access to combat state for the `CombatView`.
pub trait CombatDataSource {
    /// Current simulation state snapshot.
    fn sim_state(&self) -> &SimState;

    /// Navigation grid for terrain rendering. `None` if unavailable.
    fn grid_nav(&self) -> Option<&GridNav>;

    /// Whether the simulation is currently paused.
    fn is_paused(&self) -> bool;

    /// Current tick number.
    fn tick(&self) -> u64;

    /// Display name for the mission/scenario.
    fn mission_name(&self) -> &str;

    /// Objective description text.
    fn objective_text(&self) -> &str;

    /// Whether the user can issue commands (live combat only).
    fn can_issue_commands(&self) -> bool;

    /// Optional replay frame info: (current_frame, total_frames).
    fn replay_info(&self) -> Option<(usize, usize)> {
        None
    }
}
