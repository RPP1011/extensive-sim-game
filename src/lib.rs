// Items in this lib crate are used by binary targets (xtask, sim_bridge, etc.)
// but appear as dead code when the lib is compiled standalone.
#![allow(dead_code)]

pub mod ai;
pub mod audio;
pub mod game_core;
pub mod mapgen_voronoi;
pub mod mission;
pub mod progression;
pub mod scenario;

/// Re-export of `ai::core` under a clearer namespace.
///
/// The simulation engine (`SimState`, `step()`, `UnitState`, `SimEvent`, etc.)
/// is not AI — it's the deterministic physics/rules engine. This alias lets
/// new code use `crate::sim` (or `bevy_game::sim`) instead of `crate::ai::core`.
pub use ai::core as sim;

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
