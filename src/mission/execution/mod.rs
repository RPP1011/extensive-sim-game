pub(crate) mod setup;
mod ui;

// Re-export context
pub use setup::ActiveMissionContext;
pub use setup::ReplayViewerState;

// Re-export systems
#[allow(unused_imports)]
pub use setup::{
    mission_scene_transition_system,
    sync_sim_to_visuals_system,
    replay_viewer_transition_system,
    advance_replay_viewer_system,
    replay_viewer_keyboard_system,
};

#[allow(unused_imports)]
pub use ui::{
    mission_outcome_ui_system,
    ability_hud_system,
};
