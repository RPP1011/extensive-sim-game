// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioCfg {
    pub name: String,
    pub seed: u64,
    #[serde(default = "default_hero_count")]
    pub hero_count: usize,
    #[serde(default = "default_enemy_count")]
    pub enemy_count: usize,
    #[serde(default = "default_difficulty")]
    pub difficulty: u32,
    #[serde(default = "default_max_ticks")]
    pub max_ticks: u64,
    #[serde(default = "default_room_type")]
    pub room_type: String,
    #[serde(default)]
    pub hero_templates: Vec<String>,
    /// When set, the enemy team uses hero templates instead of generic enemies.
    #[serde(default)]
    pub enemy_hero_templates: Vec<String>,
    /// Multiply all unit HP/max_hp to increase time-to-kill (default 1.0).
    #[serde(default = "default_hp_multiplier")]
    pub hp_multiplier: f32,
    /// Optional path to an ability manifest TOML for resolving ability_refs in hero templates.
    #[serde(default)]
    pub manifest_path: Option<String>,

    // ── Drill-specific fields ──
    /// Drill type identifier (e.g., "reach_point", "kite_enemy", "kill_healer")
    #[serde(default)]
    pub drill_type: Option<String>,
    /// Target position for movement drills [x, y]
    #[serde(default)]
    pub target_position: Option<[f32; 2]>,
    /// Custom enemy unit definitions (behavior + stats overrides)
    #[serde(default)]
    pub enemy_units: Vec<EnemyUnitDef>,
    /// Static hazard zones
    #[serde(default)]
    pub hazards: Vec<HazardDef>,
    /// Drill objective
    #[serde(default)]
    pub objective: Option<ObjectiveDef>,
    /// Which action heads are enabled ("move_only", "move_attack", "all")
    #[serde(default)]
    pub action_mask: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnemyUnitDef {
    /// Hero template to use for this enemy's abilities/stats
    #[serde(default)]
    pub template: Option<String>,
    /// Behavior DSL file name (assets/behaviors/*.behavior)
    #[serde(default)]
    pub behavior: Option<String>,
    /// Named tag for behavior targeting (e.g., "healer", "tank")
    #[serde(default)]
    pub tag: Option<String>,
    /// Override spawn position
    #[serde(default)]
    pub position: Option<[f32; 2]>,
    /// Override HP
    #[serde(default)]
    pub hp_override: Option<i32>,
    /// Override auto-attack DPS
    #[serde(default)]
    pub dps_override: Option<f32>,
    /// Override attack range
    #[serde(default)]
    pub range_override: Option<f32>,
    /// Override move speed (units/sec)
    #[serde(default)]
    pub move_speed_override: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HazardDef {
    /// "damage_zone", "slow_zone", "heal_zone"
    pub hazard_type: String,
    pub position: [f32; 2],
    pub radius: f32,
    #[serde(default)]
    pub damage_per_tick: f32,
    /// "neutral", "hero", "enemy"
    #[serde(default = "default_hazard_team")]
    pub team: String,
    #[serde(default)]
    pub start_tick: u64,
    /// None = permanent
    #[serde(default)]
    pub duration: Option<u64>,
}

fn default_hazard_team() -> String { "neutral".to_string() }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectiveDef {
    /// "reach_position", "survive", "kill_all", "kill_target", "protect_ally"
    pub objective_type: String,
    #[serde(default)]
    pub position: Option<[f32; 2]>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub duration: Option<u64>,
    #[serde(default)]
    pub target_tag: Option<String>,
    #[serde(default)]
    pub max_damage_taken: Option<f32>,
}

impl Default for ScenarioCfg {
    fn default() -> Self {
        Self {
            name: String::new(),
            seed: 0,
            hero_count: default_hero_count(),
            enemy_count: default_enemy_count(),
            difficulty: default_difficulty(),
            max_ticks: default_max_ticks(),
            room_type: default_room_type(),
            hero_templates: Vec::new(),
            enemy_hero_templates: Vec::new(),
            hp_multiplier: default_hp_multiplier(),
            manifest_path: None,
            drill_type: None,
            target_position: None,
            enemy_units: Vec::new(),
            hazards: Vec::new(),
            objective: None,
            action_mask: None,
        }
    }
}

fn default_hp_multiplier() -> f32 {
    1.0
}
fn default_hero_count() -> usize {
    4
}
fn default_enemy_count() -> usize {
    4
}
fn default_difficulty() -> u32 {
    2
}
fn default_max_ticks() -> u64 {
    3000
}
fn default_room_type() -> String {
    "Entry".to_string()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioAssert {
    pub outcome: Option<String>,
    pub max_ticks_to_win: Option<u64>,
    pub min_heroes_alive: Option<usize>,
    pub max_heroes_dead: Option<usize>,
}

/// Top-level TOML struct for a scenario file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioFile {
    pub scenario: ScenarioCfg,
    pub assert: Option<ScenarioAssert>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssertionResult {
    pub name: String,
    pub passed: bool,
    pub value: String,
    pub expected: String,
}

// ---------------------------------------------------------------------------
// Per-unit combat statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbilityStats {
    pub ability_name: String,
    pub times_used: u32,
    pub damage_dealt: i64,
    pub healing_done: i64,
    pub shield_granted: i64,
    pub cc_applied_count: u32,
    pub cc_duration_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnitStats {
    pub unit_id: u32,
    pub team: String,
    pub template: String,
    pub max_hp: i32,
    pub final_hp: i32,
    pub damage_dealt: i64,
    pub damage_taken: i64,
    pub overkill_dealt: i64,
    pub healing_done: i64,
    pub healing_received: i64,
    pub overhealing: i64,
    pub lifesteal_healing: i64,
    pub shield_received: i64,
    pub shield_absorbed: i64,
    pub cc_applied_count: u32,
    pub cc_received_count: u32,
    pub cc_duration_applied_ms: u64,
    pub abilities_used: u32,
    pub passives_triggered: u32,
    pub attacks_missed: u32,
    pub kills: u32,
    pub deaths: u32,
    pub reflect_damage: i64,
    pub ability_stats: Vec<AbilityStats>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioResult {
    pub scenario_name: String,
    /// "Victory" | "Defeat" | "Timeout"
    pub outcome: String,
    pub tick: u64,
    /// true when all assertions passed (or there were none)
    pub passed: bool,
    pub assertions: Vec<AssertionResult>,
    pub final_hero_count: usize,
    pub final_enemy_count: usize,
    pub events: Vec<String>,
    pub hero_deaths: usize,
    pub unit_stats: Vec<UnitStats>,
}
