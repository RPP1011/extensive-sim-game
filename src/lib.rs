// Items in this lib crate are used by binary targets (xtask, etc.)
// but appear as dead code when the lib is compiled standalone.
#![allow(dead_code)]

pub mod rendering;
pub mod game_core;
pub mod mission;
pub mod ai;

// Re-export AI sub-modules at crate root so internal paths (crate::sim, etc.)
// continue to resolve after the tactical_sim merge.
pub use ai::core as sim;
pub use ai::effects;
pub use ai::pathing;
pub use ai::squad;
pub use ai::goap;
pub use ai::control;
pub use ai::personality;
pub use ai::roles;
pub use ai::utility;
pub use ai::phase;
pub use ai::advanced;
pub use ai::student;
pub use ai::tooling;

pub mod world_sim;
pub mod content;
pub mod model_backend;
pub mod ascii_gen;
pub mod scenario;
pub mod narrative;
pub mod overworld_grid;
