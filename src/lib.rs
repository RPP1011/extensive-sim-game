// Items in this lib crate are used by binary targets (xtask, sim_bridge, etc.)
// but appear as dead code when the lib is compiled standalone.
#![allow(dead_code)]

pub mod audio;
pub mod game_core;
pub mod mission;
pub mod progression;

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
pub mod hot_reload;
pub use tactical_sim::mapgen_voronoi;

// ---------------------------------------------------------------------------
// Stub types used by mission::execution that reference the binary crate root.
// When compiled as a library these types need to exist at crate:: to satisfy
// the type-checker; the binary supplies its own full definitions in main.rs.
// ---------------------------------------------------------------------------

/// Screen / navigation state for the hub UI (used by mission::execution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubScreen {
    StartMenu,
    CharacterCreationFaction,
    CharacterCreationBackstory,
    BackstoryCinematic,
    GuildManagement,
    Overworld,
    OverworldMap,
    RegionView,
    LocalEagleEyeIntro,
    MissionExecution,
    ReplayViewer,
}

impl HubScreen {
    /// Whether this screen shows the shared left side panel.
    pub fn shows_side_panel(&self) -> bool {
        !matches!(
            self,
            HubScreen::CharacterCreationFaction
                | HubScreen::CharacterCreationBackstory
                | HubScreen::BackstoryCinematic
        )
    }
}

/// Hub UI resource (used by mission::execution).
#[derive(bevy::prelude::Resource)]
pub struct HubUiState {
    pub screen: HubScreen,
}

/// Camera sub-module stub so `crate::camera::OrbitCameraController` resolves.
pub mod camera {
    /// Orbit camera controller component (stub for library compilation).
    #[derive(bevy::prelude::Component, Default)]
    pub struct OrbitCameraController {
        pub focus: bevy::prelude::Vec3,
        pub radius: f32,
        pub yaw: f32,
        pub pitch: f32,
    }
}
