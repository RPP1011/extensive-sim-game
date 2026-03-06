use bevy::prelude::*;

use crate::ai;

#[derive(Resource, Clone)]
pub struct Scenario3dData(pub ai::tooling::CustomScenario);

#[derive(Resource)]
pub struct ScenarioReplay {
    pub name: String,
    pub frames: Vec<ai::core::SimState>,
    pub frame_index: usize,
    pub tick_seconds: f32,
    pub tick_accumulator: f32,
    pub paused: bool,
    /// Per-frame sim events for animation driving. Same length as `frames`.
    pub events_per_frame: Vec<Vec<ai::core::SimEvent>>,
}

#[derive(Component)]
pub struct ScenarioUnitVisual {
    pub unit_id: u32,
}

#[derive(Resource)]
pub struct ScenarioPlaybackSpeed {
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Component)]
pub struct PlaybackSliderTrack;

#[derive(Component)]
pub struct PlaybackSliderFill;

#[derive(Component)]
pub struct PlaybackSliderLabel;
