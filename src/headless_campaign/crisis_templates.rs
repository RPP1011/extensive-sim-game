//! Data-driven crisis templates loaded from TOML files.
//!
//! Crisis content (names, champion rosters, escalation rates, etc.) lives in
//! `assets/crises/*.toml`. The Rust code in `systems/crisis.rs` and
//! `systems/threat.rs` is pure mechanics — all content comes from these templates.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// A crisis template loaded from a TOML file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrisisTemplate {
    pub name: String,
    pub description: String,
    /// What type of crisis this is.
    pub crisis_type: String, // "sleeping_king", "breach", "corruption", "unifier", "decline"
    /// World templates this crisis can appear in (empty = any).
    #[serde(default)]
    pub world_templates: Vec<String>,
    /// Campaign progress threshold to trigger (0.0–1.0).
    pub trigger_threshold: f32,
    /// Priority — higher priority crises checked first.
    #[serde(default)]
    pub priority: i32,
    /// Can this crisis co-exist with others?
    #[serde(default = "default_true")]
    pub allow_simultaneous: bool,
    /// Crisis-type-specific configuration.
    pub config: CrisisConfig,
}

fn default_true() -> bool {
    true
}

/// Tagged enum for crisis-specific parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CrisisConfig {
    SleepingKing {
        /// Faction name to match or describe the king's faction.
        king_faction_name: String,
        /// Champion definitions.
        champions: Vec<ChampionDef>,
        /// Ticks before the first champion activates.
        first_activation_ticks: u64,
        /// Ticks between subsequent champion activations.
        activation_interval_ticks: u64,
        /// Power growth formula: "quadratic", "linear", or "exponential".
        power_formula: String,
        /// Number of champions that must arrive before declaring war.
        war_threshold: u32,
    },
    Breach {
        /// Location type to find as the source (e.g. "Dungeon", "Ruin").
        source_location_type: String,
        /// Initial wave strength (multiplied by world threat).
        initial_strength: f32,
        /// Wave strength multiplier per wave.
        strength_multiplier: f32,
        /// Ticks between the first waves.
        initial_wave_interval: u64,
        /// Ticks subtracted from interval each wave.
        wave_acceleration: u64,
        /// Minimum ticks between waves.
        min_wave_interval: u64,
    },
    Corruption {
        /// Ticks between spread events.
        spread_interval_ticks: u64,
        /// Control damage applied per tick to corrupted regions.
        control_damage_per_tick: f32,
        /// Unrest added per tick to corrupted regions.
        unrest_damage_per_tick: f32,
    },
    Unifier {
        /// Ticks between absorption attempts.
        absorb_interval_ticks: u64,
        /// Fraction of absorbed faction's military strength gained.
        strength_absorption_rate: f32,
    },
    Decline {
        /// Gold drained per application (every 100 ticks).
        gold_drain_per_tick: f32,
        /// Supplies drained per application.
        supply_drain_per_tick: f32,
        /// Morale drained per adventurer per application.
        morale_drain_per_tick: f32,
        /// Control drained per region per application.
        control_drain_per_tick: f32,
        /// How fast severity grows over time (severity = 1.0 + elapsed / rate).
        severity_growth_rate: f32,
    },
}

/// A champion definition for the Sleeping King crisis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChampionDef {
    pub name: String,
    pub archetype: String,
    pub level: u32,
    /// Type of buff this champion provides: "military_strength", "attack_multiplier",
    /// "defense_bonus", "gold_income", "recovery_rate", "recruit_bonus", "diplomacy_bonus".
    pub buff_type: String,
    pub buff_value: f32,
    pub starting_position: [f32; 2],
    pub travel_speed: f32,
}

impl CrisisTemplate {
    /// Load a single crisis template from a TOML file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Load all crisis templates from a directory of TOML files.
    ///
    /// Returns them sorted by priority (highest first), then by filename.
    pub fn load_from_dir(dir: &std::path::Path) -> Result<Vec<Self>, String> {
        let mut templates = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read {}: {}", dir.display(), e))?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                let template = Self::load_from_file(&path)?;
                templates.push(template);
            }
        }
        // Sort by priority descending (highest first).
        templates.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(templates)
    }
}

/// Global cache of loaded crisis templates.
static CRISIS_TEMPLATES: OnceLock<Vec<CrisisTemplate>> = OnceLock::new();

/// Load crisis templates from `assets/crises/`, caching the result.
///
/// Returns an empty slice if the directory doesn't exist or has no valid files.
pub fn get_or_load_crises() -> &'static [CrisisTemplate] {
    CRISIS_TEMPLATES.get_or_init(|| {
        let dir = std::path::Path::new("assets/crises");
        if dir.exists() {
            match CrisisTemplate::load_from_dir(dir) {
                Ok(templates) => templates,
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        }
    })
}

/// Convert a `buff_type` string + value into a `LeadershipBuff`.
pub fn parse_leadership_buff(buff_type: &str, value: f32) -> super::state::LeadershipBuff {
    use super::state::LeadershipBuff;
    match buff_type {
        "military_strength" => LeadershipBuff::MilitaryStrength(value),
        "attack_multiplier" => LeadershipBuff::AttackMultiplier(value),
        "defense_bonus" => LeadershipBuff::DefenseBonus(value),
        "gold_income" => LeadershipBuff::GoldIncome(value),
        "recovery_rate" => LeadershipBuff::RecoveryRate(value),
        "reward_multiplier" => LeadershipBuff::RewardMultiplier(value),
        "recruit_bonus" => LeadershipBuff::RecruitBonus(value),
        "diplomacy_bonus" => LeadershipBuff::DiplomacyBonus(value),
        "morale_aura" => LeadershipBuff::MoraleAura(value),
        _ => LeadershipBuff::MilitaryStrength(value), // fallback
    }
}

/// Convert a `source_location_type` string into a `LocationType`.
pub fn parse_location_type(s: &str) -> super::state::LocationType {
    use super::state::LocationType;
    match s {
        "Settlement" => LocationType::Settlement,
        "Wilderness" => LocationType::Wilderness,
        "Dungeon" => LocationType::Dungeon,
        "Ruin" => LocationType::Ruin,
        "Outpost" => LocationType::Outpost,
        _ => LocationType::Dungeon, // fallback
    }
}

/// Compute power boost based on the formula string and champion count.
pub fn compute_power_boost(formula: &str, champions_arrived: u32) -> f32 {
    let n = champions_arrived as f32;
    match formula {
        "quadratic" => 25.0 * n * n,
        "linear" => 50.0 * n,
        "exponential" => 25.0 * 2.0f32.powf(n - 1.0),
        _ => 25.0 * n * n, // default to quadratic
    }
}
