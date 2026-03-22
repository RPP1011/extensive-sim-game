use serde::{Deserialize, Serialize};

use crate::sim::IntentAction;
use crate::personality::UnitMode;

#[derive(Debug, Clone)]
pub struct ActionScoreDebug {
    pub action: IntentAction,
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct UnitDecisionDebug {
    pub unit_id: u32,
    pub mode: UnitMode,
    pub chosen: IntentAction,
    pub top_k: Vec<ActionScoreDebug>,
}

#[derive(Debug, Clone)]
pub struct TickDecisionDebug {
    pub tick: u64,
    pub decisions: Vec<UnitDecisionDebug>,
}

#[derive(Debug, Clone)]
pub struct ScenarioSummary {
    pub name: String,
    pub winner: String,
    pub tick_to_first_death: Option<u64>,
    pub team_ttk: Option<u64>,
    pub eliminated_team: Option<String>,
    pub hero_deaths: u32,
    pub enemy_deaths: u32,
    pub event_hash: u64,
    pub state_hash: u64,
    pub deterministic: bool,
    pub casts: u32,
    pub heals: u32,
}

#[derive(Debug, Clone)]
pub struct CcChainMetrics {
    pub primary_target: Option<u32>,
    pub windows: usize,
    pub links: usize,
    pub coverage_ratio: f32,
    pub overlap_ratio: f32,
    pub avg_gap_ticks: f32,
}

#[derive(Debug, Clone)]
pub struct TuningResult {
    pub aggression: f32,
    pub control_bias: f32,
    pub altruism: f32,
    pub score: i32,
    pub event_hash: u64,
    pub winner: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioObstacle {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioUnit {
    pub id: u32,
    pub team: String,
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub elevation: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub move_speed: f32,
    pub attack_damage: i32,
    pub attack_range: f32,
    pub ability_damage: i32,
    pub ability_range: f32,
    pub heal_amount: i32,
    pub heal_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioElevationZone {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub elevation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSlopeZone {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub slope_cost_multiplier: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScenario {
    pub name: String,
    pub seed: u64,
    pub ticks: u32,
    pub world_min_x: f32,
    pub world_max_x: f32,
    pub world_min_y: f32,
    pub world_max_y: f32,
    pub cell_size: f32,
    #[serde(default)]
    pub elevation_zones: Vec<ScenarioElevationZone>,
    #[serde(default)]
    pub slope_zones: Vec<ScenarioSlopeZone>,
    pub obstacles: Vec<ScenarioObstacle>,
    pub units: Vec<ScenarioUnit>,
}
