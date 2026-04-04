//! TOML-driven scenario configuration — no recompilation needed.
//!
//! Scenarios live in `building_scenarios/` as TOML files.
//! The runner loads them at runtime, generates seeds, injects challenges,
//! and feeds observations to the oracle.
//!
//! # File structure
//!
//! ```text
//! building_scenarios/
//!   challenges/          # Challenge definitions (reusable across scenarios)
//!     military.toml
//!     environmental.toml
//!     economic.toml
//!     ...
//!   enemy_profiles/      # Enemy type definitions
//!     orcs.toml
//!     bandits.toml
//!     dragons.toml
//!     ...
//!   seeds/               # Settlement seed templates
//!     hamlet_flat.toml
//!     village_river.toml
//!     town_hill.toml
//!     ...
//!   scenarios/            # Composed scenarios (seed + challenges)
//!     siege_basic.toml
//!     flood_and_raid.toml
//!     winter_prep.toml
//!     ...
//!   oracle_weights/       # Tunable oracle utility weights
//!     default.toml
//!     military_focus.toml
//!     economic_focus.toml
//!     ...
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::types::*;

// ---------------------------------------------------------------------------
// Top-level scenario file (what you edit without recompiling)
// ---------------------------------------------------------------------------

/// A composed scenario: seed + one or more challenges + oracle weight overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioFile {
    pub meta: ScenarioMeta,
    pub seed: SeedConfig,
    pub challenges: Vec<ChallengeConfig>,
    /// Optional: override oracle weights (file path or inline).
    pub oracle_weights: Option<String>,
    /// Optional: override scoring weights.
    pub scoring_weights: Option<ScoringWeightsConfig>,
    /// How many ticks to simulate the challenge forward for scoring.
    #[serde(default = "default_sim_ticks")]
    pub sim_ticks: u64,
    /// Number of random baselines to run.
    #[serde(default = "default_num_baselines")]
    pub num_random_baselines: usize,
}

fn default_sim_ticks() -> u64 {
    500
}
fn default_num_baselines() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Matrix coordinates this scenario covers (for coverage tracking).
    #[serde(default)]
    pub matrix_cells: Vec<MatrixCell>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixCell {
    pub challenge: String,
    pub decision: String,
}

// ---------------------------------------------------------------------------
// Seed configuration — defines initial settlement state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedConfig {
    /// Reference a seed template file, or define inline.
    pub template: Option<String>,

    /// Settlement parameters (override template if both present).
    #[serde(default)]
    pub settlement_level: Option<u8>,
    #[serde(default)]
    pub population: Option<RangeOrFixed>,
    #[serde(default)]
    pub tech_tier: Option<u8>,
    #[serde(default)]
    pub terrain: Option<String>,
    #[serde(default)]
    pub age_ticks: Option<RangeOrFixed>,

    /// Resource overrides: commodity_name -> amount or range.
    #[serde(default)]
    pub stockpiles: HashMap<String, RangeOrFixed>,

    /// Existing buildings (if not using template defaults).
    #[serde(default)]
    pub buildings: Vec<BuildingConfig>,

    /// NPC roster overrides.
    #[serde(default)]
    pub npcs: Vec<NpcConfig>,

    /// High-value NPC specifications.
    #[serde(default)]
    pub high_value_npcs: Vec<HighValueNpcConfig>,

    /// Explicit spatial layout. If present, overrides procedural placement.
    #[serde(default)]
    pub layout: Option<SettlementLayout>,

    /// RNG seed. If absent, randomized per run.
    #[serde(default)]
    pub rng_seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// Settlement layout — explicit spatial structure
// ---------------------------------------------------------------------------

/// Describes the spatial structure of a settlement: building positions,
/// wall circuits, roads, zones, and gates. When present in a seed config,
/// this overrides procedural placement entirely.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SettlementLayout {
    /// Grid dimensions (cols, rows). Required when layout is specified.
    #[serde(default = "default_grid_size")]
    pub grid_size: (u16, u16),

    /// Positioned buildings with footprints.
    #[serde(default)]
    pub buildings: Vec<LayoutBuilding>,

    /// Closed wall circuits (perimeter walls as polygons).
    #[serde(default)]
    pub wall_circuits: Vec<WallCircuit>,

    /// Open wall segments (partial walls, not closed).
    #[serde(default)]
    pub wall_segments: Vec<WallSegment>,

    /// Road paths between waypoints.
    #[serde(default)]
    pub roads: Vec<RoadPath>,

    /// Rectangular zones with a purpose.
    #[serde(default)]
    pub zones: Vec<ZoneRect>,

    /// Gate openings in wall circuits.
    #[serde(default)]
    pub gates: Vec<GateSpec>,
}

fn default_grid_size() -> (u16, u16) {
    (24, 24)
}

/// A building with an explicit grid position and footprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBuilding {
    /// Building type name (must match BuildingType variants).
    #[serde(rename = "type")]
    pub building_type: String,
    /// Top-left corner of the footprint on the grid.
    pub cell: (u16, u16),
    /// Footprint size (cols, rows). Defaults to (1, 1).
    #[serde(default = "default_footprint")]
    pub footprint: (u16, u16),
    /// Material override.
    #[serde(default)]
    pub material: Option<String>,
    /// Which direction the entrance/primary opening faces.
    #[serde(default)]
    pub facing: Option<String>,
    /// Building tier.
    #[serde(default)]
    pub tier: Option<u8>,
}

fn default_footprint() -> (u16, u16) {
    (1, 1)
}

/// A closed wall circuit defined by corner waypoints.
/// Walls are auto-generated between consecutive waypoints,
/// with the last waypoint connecting back to the first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallCircuit {
    /// Corner waypoints forming a closed polygon.
    pub waypoints: Vec<(u16, u16)>,
    /// Wall material.
    #[serde(default = "default_wall_material")]
    pub material: String,
    /// Wall height in tiles.
    #[serde(default = "default_wall_height")]
    pub height: u8,
    /// Wall thickness.
    #[serde(default = "default_wall_thickness")]
    pub thickness: u8,
}

fn default_wall_material() -> String {
    "wood".to_string()
}
fn default_wall_height() -> u8 {
    3
}
fn default_wall_thickness() -> u8 {
    1
}

/// A partial (non-closed) wall segment between two points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallSegment {
    pub from: (u16, u16),
    pub to: (u16, u16),
    #[serde(default = "default_wall_material")]
    pub material: String,
    #[serde(default = "default_wall_height")]
    pub height: u8,
    #[serde(default = "default_wall_thickness")]
    pub thickness: u8,
}

/// A road path defined by waypoints. Cells along the path are marked walkable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadPath {
    pub waypoints: Vec<(u16, u16)>,
}

/// A rectangular zone with a designated purpose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneRect {
    /// Zone type: "residential", "industrial", "military", "commercial", "religious".
    pub zone: String,
    /// Top-left corner.
    pub from: (u16, u16),
    /// Bottom-right corner (inclusive).
    pub to: (u16, u16),
}

/// A gate opening in a wall circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpec {
    /// Grid cell where the gate is placed.
    pub cell: (u16, u16),
    /// Direction the gate faces.
    #[serde(default)]
    pub facing: Option<String>,
    /// Gate width in cells.
    #[serde(default = "default_gate_width")]
    pub width: u8,
}

fn default_gate_width() -> u8 {
    1
}

/// Either a fixed value or a [min, max] range for randomization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RangeOrFixed {
    Fixed(f64),
    Range([f64; 2]),
}

impl RangeOrFixed {
    pub fn resolve(&self, rng: &mut impl FnMut() -> f64) -> f64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Range([lo, hi]) => lo + rng() * (hi - lo),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingConfig {
    pub building_type: String,
    #[serde(default)]
    pub grid_cell: Option<(u16, u16)>,
    #[serde(default)]
    pub tier: Option<u8>,
    #[serde(default)]
    pub material: Option<String>,
    #[serde(default)]
    pub count: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcConfig {
    pub class: String,
    pub level: RangeOrFixed,
    #[serde(default = "default_npc_count")]
    pub count: u16,
    #[serde(default)]
    pub is_garrison: bool,
}

fn default_npc_count() -> u16 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighValueNpcConfig {
    pub role: String,
    pub level: RangeOrFixed,
    #[serde(default)]
    pub class: Option<String>,
    /// 0.0–1.0, how important to protect.
    #[serde(default = "default_protection")]
    pub protection_priority: f32,
}

fn default_protection() -> f32 {
    0.5
}

// ---------------------------------------------------------------------------
// Challenge configuration — layered onto seeds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeConfig {
    /// Reference a challenge template file, or define inline.
    pub template: Option<String>,

    pub category: Option<String>,
    pub sub_type: Option<String>,
    #[serde(default)]
    pub severity: Option<RangeOrFixed>,
    #[serde(default)]
    pub direction: Option<DirectionConfig>,
    /// Ticks after scenario start when this challenge activates.
    #[serde(default)]
    pub delay_ticks: Option<u64>,
    /// Deadline tick (for temporal challenges like "winter in 200 ticks").
    #[serde(default)]
    pub deadline_ticks: Option<u64>,

    /// Enemy profiles for military challenges.
    #[serde(default)]
    pub enemies: Vec<EnemyConfig>,

    /// Environmental parameters.
    #[serde(default)]
    pub env_params: Option<EnvParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DirectionConfig {
    /// Cardinal: "north", "east", "south", "west".
    Named(String),
    /// Exact vector.
    Vector([f32; 2]),
}

impl DirectionConfig {
    pub fn to_vector(&self) -> (f32, f32) {
        match self {
            Self::Named(s) => match s.as_str() {
                "north" => (0.0, -1.0),
                "south" => (0.0, 1.0),
                "east" => (1.0, 0.0),
                "west" => (-1.0, 0.0),
                "northeast" => (0.707, -0.707),
                "northwest" => (-0.707, -0.707),
                "southeast" => (0.707, 0.707),
                "southwest" => (-0.707, 0.707),
                _ => (0.0, -1.0),
            },
            Self::Vector(v) => (v[0], v[1]),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyConfig {
    /// Reference enemy profile file, or define inline.
    pub profile: Option<String>,
    pub type_name: Option<String>,
    pub level_range: Option<[u8; 2]>,
    pub count: Option<RangeOrFixed>,
    #[serde(default)]
    pub can_jump: Option<bool>,
    #[serde(default)]
    pub jump_height: Option<u8>,
    #[serde(default)]
    pub can_climb: Option<bool>,
    #[serde(default)]
    pub can_tunnel: Option<bool>,
    #[serde(default)]
    pub can_fly: Option<bool>,
    #[serde(default)]
    pub has_siege: Option<bool>,
    #[serde(default)]
    pub siege_damage: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvParams {
    #[serde(default)]
    pub flood_severity: Option<RangeOrFixed>,
    #[serde(default)]
    pub fire_spread_rate: Option<f32>,
    #[serde(default)]
    pub earthquake_magnitude: Option<RangeOrFixed>,
    #[serde(default)]
    pub wind_speed: Option<f32>,
}

// ---------------------------------------------------------------------------
// Scoring weight overrides
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeightsConfig {
    #[serde(default)]
    pub defensive: Option<f32>,
    #[serde(default)]
    pub environmental: Option<f32>,
    #[serde(default)]
    pub economic: Option<f32>,
    #[serde(default)]
    pub population: Option<f32>,
    #[serde(default)]
    pub spatial_quality: Option<f32>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a scenario from a TOML file.
pub fn load_scenario(path: &Path) -> Result<ScenarioFile, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
}

/// Load all scenarios from a directory (recursive).
pub fn load_all_scenarios(dir: &Path) -> Result<Vec<(String, ScenarioFile)>, String> {
    let mut results = Vec::new();
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    collect_toml_files(dir, &mut results)?;
    Ok(results)
}

fn collect_toml_files(
    dir: &Path,
    results: &mut Vec<(String, ScenarioFile)>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read dir {}: {}", dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Dir entry error: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_toml_files(&path, results)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            let scenario = load_scenario(&path)?;
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            results.push((name, scenario));
        }
    }
    Ok(())
}
