// Items in this lib crate are used by binary targets (xtask, etc.)
// but appear as dead code when the lib is compiled standalone.
#![allow(dead_code)]

pub mod game_core;
pub mod mission;

// ---------------------------------------------------------------------------
// Re-export tactical_sim modules under familiar paths for backward compat.
// All simulation / AI code now lives in the `tactical_sim` crate.
// ---------------------------------------------------------------------------

/// Re-export of `tactical_sim::sim` under the familiar `sim` alias.
pub use tactical_sim::sim;

/// Re-export all AI sub-modules so existing `crate::ai::*` paths keep working.
pub mod ai {
    pub use tactical_sim::sim as core;
    pub use tactical_sim::effects;
    pub use tactical_sim::pathing;
    pub use tactical_sim::squad;
    pub use tactical_sim::goap;
    pub use tactical_sim::control;
    pub use tactical_sim::personality;
    pub use tactical_sim::roles;
    pub use tactical_sim::utility;
    pub use tactical_sim::phase;
    pub use tactical_sim::advanced;
    pub use tactical_sim::student;
    pub use tactical_sim::tooling;
}

pub mod world_sim;
pub mod content;
pub mod model_backend;
pub mod ascii_gen;
pub mod scenario;
pub mod narrative;
pub mod overworld_grid;
pub use tactical_sim::mapgen_voronoi;
