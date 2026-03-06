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
