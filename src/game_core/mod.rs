mod types;
mod overworld_types;
mod roster_types;
mod companion;
mod generation;
mod roster_gen;
mod setup;
mod campaign_systems;
mod overworld_systems;
mod overworld_nav;
mod flashpoint_helpers;
mod flashpoint_spawn;
mod flashpoint_progression;
mod diplomacy_systems;
mod consequence_systems;
mod campaign_outcome;
mod mission_systems;
mod attention_systems;
pub mod faction_ai;
mod save;
mod migrate;
pub mod verify;
mod verify_details;

#[cfg(test)]
mod tests;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use types::*;
pub use overworld_types::*;
pub use roster_types::*;
pub use companion::*;
pub use generation::overworld_region_plot_positions;
pub use generation::overworld_hex_coords;
pub use roster_gen::*;
pub use setup::*;
pub use campaign_systems::*;
pub use overworld_systems::*;
pub use overworld_nav::*;
pub use flashpoint_spawn::*;
pub use flashpoint_progression::*;
pub use diplomacy_systems::*;
pub use consequence_systems::*;
pub use campaign_outcome::*;
pub use mission_systems::*;
pub use attention_systems::*;
pub use save::*;
