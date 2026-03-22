mod debug;
mod events;
mod scenarios;
mod types;
mod viz_template;

pub(crate) mod custom;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use types::{
    CustomScenario, ScenarioObstacle, ScenarioUnit,
};

// Re-export all public functions
pub use debug::build_phase5_debug;
pub use scenarios::{
    analyze_phase4_cc_metrics, reservation_timeline_summary, run_personality_grid_tuning,
    run_scenario_matrix,
};
pub use custom::{
    build_custom_scenario_state_frames, export_custom_scenario_visualization,
    export_horde_chokepoint_hero_favored_visualization, export_horde_chokepoint_visualization,
    export_phase5_event_visualization, export_visualization_index, write_custom_scenario_template,
};
