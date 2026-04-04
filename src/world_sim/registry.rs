//! Data-driven registry types for TOML-defined classes, entity templates,
//! terrain templates, and scenario definitions.
//!
//! These types are the serde layer between TOML files on disk and the runtime
//! world simulation. They are plain data — no logic lives here.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::world_sim::state::tag;

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/// Flat stat block used in class definitions and entity templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatBlock {
    #[serde(default)]
    pub hp: f32,
    #[serde(default)]
    pub attack: f32,
    #[serde(default)]
    pub armor: f32,
    #[serde(default)]
    pub speed: f32,
}

// ---------------------------------------------------------------------------
// Class definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassRequirements {
    #[serde(default)]
    pub min_level: u32,
    #[serde(default)]
    pub behavior: HashMap<String, f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassAbilities {
    #[serde(default)]
    pub pool: Vec<String>,
}

/// Full class definition loaded from TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassDefToml {
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub base_stats: StatBlock,
    #[serde(default)]
    pub per_level: StatBlock,
    #[serde(default)]
    pub requirements: ClassRequirements,
    #[serde(default)]
    pub score_weights: HashMap<String, f32>,
    #[serde(default)]
    pub abilities: ClassAbilities,
}

// ---------------------------------------------------------------------------
// Entity templates
// ---------------------------------------------------------------------------

/// Stat block specialised for entity templates (with extra fields and
/// non-zero defaults).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityStats {
    #[serde(default = "EntityStats::default_hp")]
    pub hp: f32,
    #[serde(default = "EntityStats::default_attack")]
    pub attack: f32,
    #[serde(default)]
    pub armor: f32,
    #[serde(default = "EntityStats::default_speed")]
    pub speed: f32,
    #[serde(default = "EntityStats::default_attack_range")]
    pub attack_range: f32,
}

impl EntityStats {
    fn default_hp() -> f32 { 100.0 }
    fn default_attack() -> f32 { 10.0 }
    fn default_speed() -> f32 { 3.0 }
    fn default_attack_range() -> f32 { 1.5 }
}

impl Default for EntityStats {
    fn default() -> Self {
        Self {
            hp: Self::default_hp(),
            attack: Self::default_attack(),
            armor: 0.0,
            speed: Self::default_speed(),
            attack_range: Self::default_attack_range(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StartingClass {
    pub name: String,
    #[serde(default = "StartingClass::default_level")]
    pub level: u16,
}

impl StartingClass {
    fn default_level() -> u16 { 1 }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityClasses {
    #[serde(default)]
    pub starting: Vec<StartingClass>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityAbilities {
    #[serde(default)]
    pub list: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackConfig {
    #[serde(default = "AttackConfig::default_damage")]
    pub damage: i32,
    #[serde(default = "AttackConfig::default_range")]
    pub range: f32,
    #[serde(default = "AttackConfig::default_cooldown")]
    pub cooldown: u32,
    #[serde(default = "AttackConfig::default_cast_time")]
    pub cast_time: u32,
}

impl AttackConfig {
    fn default_damage() -> i32 { 15 }
    fn default_range() -> f32 { 1.5 }
    fn default_cooldown() -> u32 { 1000 }
    fn default_cast_time() -> u32 { 300 }
}

impl Default for AttackConfig {
    fn default() -> Self {
        Self {
            damage: Self::default_damage(),
            range: Self::default_range(),
            cooldown: Self::default_cooldown(),
            cast_time: Self::default_cast_time(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitiesConfig {
    #[serde(default)]
    pub can_jump: bool,
    #[serde(default)]
    pub jump_height: u8,
    #[serde(default)]
    pub can_climb: bool,
    #[serde(default)]
    pub can_tunnel: bool,
    #[serde(default)]
    pub can_fly: bool,
    #[serde(default)]
    pub has_siege: bool,
    #[serde(default)]
    pub siege_damage: f32,
}

/// Full entity template loaded from TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityTemplateToml {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub stats: EntityStats,
    #[serde(default)]
    pub classes: EntityClasses,
    #[serde(default)]
    pub abilities: EntityAbilities,
    pub attack: Option<AttackConfig>,
    pub capabilities: Option<CapabilitiesConfig>,
}

// ---------------------------------------------------------------------------
// Terrain templates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerrainResource {
    #[serde(rename = "type")]
    pub resource_type: String,
    pub amount: f32,
    pub pos: [u32; 2],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerrainFeature {
    #[serde(rename = "type")]
    pub feature_type: String,
    pub from: Option<[u32; 2]>,
    pub to: Option<[u32; 2]>,
    pub pos: Option<[u32; 2]>,
    pub radius: Option<f32>,
}

/// Full terrain template loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainTemplateToml {
    pub name: String,
    #[serde(default = "TerrainTemplateToml::default_size")]
    pub size: [u32; 2],
    pub biome: String,
    #[serde(default)]
    pub resources: Vec<TerrainResource>,
    #[serde(default)]
    pub features: Vec<TerrainFeature>,
}

impl TerrainTemplateToml {
    fn default_size() -> [u32; 2] { [64, 64] }
}

impl Default for TerrainTemplateToml {
    fn default() -> Self {
        Self {
            name: String::new(),
            size: Self::default_size(),
            biome: String::new(),
            resources: Vec::new(),
            features: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NpcStateOverrides {
    pub morale: Option<f32>,
    pub stress: Option<f32>,
    pub fatigue: Option<f32>,
    pub injury: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioNpc {
    pub template: String,
    #[serde(default = "ScenarioNpc::default_count")]
    pub count: u32,
    pub level_range: Option<[u32; 2]>,
    pub state: Option<NpcStateOverrides>,
}

impl ScenarioNpc {
    fn default_count() -> u32 { 1 }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioBuilding {
    #[serde(rename = "type")]
    pub building_type: String,
    pub pos: [u32; 2],
}

/// Consequence variant keyed by the `type` field in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HistoryConsequence {
    Casualties {
        template: String,
        count: u32,
    },
    BuildingDamaged {
        building: String,
        hp_fraction: f32,
    },
    BuildingDestroyed {
        building: String,
    },
    MoraleImpact {
        amount: f32,
    },
    PopulationState {
        stress: Option<f32>,
        fatigue: Option<f32>,
        injury: Option<f32>,
        morale: Option<f32>,
    },
    StockpileDrain {
        resource: String,
        fraction: f32,
    },
    TreasuryDrain {
        fraction: f32,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistoryEvent {
    pub event: String,
    pub ticks_ago: u64,
    pub severity: f32,
    pub summary: String,
    #[serde(default)]
    pub consequences: Vec<HistoryConsequence>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StockpileOverrides {
    pub food: Option<f32>,
    pub wood: Option<f32>,
    pub iron: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InitialStateOverrides {
    pub morale: Option<f32>,
    pub treasury: Option<f32>,
    pub stockpile: Option<StockpileOverrides>,
}

/// A single resolution check with optional params.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResolutionCheck {
    pub check: String,
    #[serde(flatten, default)]
    pub params: HashMap<String, serde_json::Value>,
}

/// How a threat or event resolves.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResolutionConfig {
    /// All conditions must pass.
    All { all: Vec<ResolutionCheck> },
    /// Any condition passing is sufficient.
    Any { any: Vec<ResolutionCheck> },
    /// Single check with optional params flattened in.
    Single {
        check: String,
        #[serde(flatten, default)]
        params: HashMap<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreatEntity {
    pub template: String,
    #[serde(default = "ThreatEntity::default_count")]
    pub count: u32,
    pub level_range: Option<[u32; 2]>,
}

impl ThreatEntity {
    fn default_count() -> u32 { 1 }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreatConfig {
    pub name: String,
    pub approach_direction: Option<[f32; 2]>,
    pub trigger_tick: u64,
    pub duration_ticks: Option<u64>,
    pub resolution: Option<ResolutionConfig>,
    #[serde(default)]
    pub entities: Vec<ThreatEntity>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventConfig {
    #[serde(rename = "type")]
    pub event_type: String,
    pub severity: f32,
    pub trigger_tick: u64,
    pub duration_ticks: Option<u64>,
    pub deadline_ticks: Option<u64>,
    pub template: Option<String>,
    pub resolution: Option<ResolutionConfig>,
}

/// Scenario completion config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    #[serde(default = "CompletionConfig::default_mode")]
    pub mode: String,
    #[serde(default = "CompletionConfig::default_max_ticks")]
    pub max_ticks: u64,
    pub failure: Option<String>,
}

impl CompletionConfig {
    fn default_mode() -> String { "all".to_string() }
    fn default_max_ticks() -> u64 { 2000 }
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            mode: Self::default_mode(),
            max_ticks: Self::default_max_ticks(),
            failure: None,
        }
    }
}

/// Full scenario definition loaded from TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioToml {
    pub name: String,
    pub terrain: Option<String>,
    #[serde(default)]
    pub npcs: Vec<ScenarioNpc>,
    #[serde(default)]
    pub buildings: Vec<ScenarioBuilding>,
    #[serde(default)]
    pub history: Vec<HistoryEvent>,
    pub initial_state: Option<InitialStateOverrides>,
    #[serde(default)]
    pub threats: Vec<ThreatConfig>,
    #[serde(default)]
    pub events: Vec<EventConfig>,
    pub completion: Option<CompletionConfig>,
}

// ---------------------------------------------------------------------------
// Top-level registry
// ---------------------------------------------------------------------------

/// In-memory registry populated from all loaded TOML files.
#[derive(Debug, Default)]
pub struct Registry {
    pub classes: HashMap<u32, ClassDefToml>,
    pub entities: HashMap<u32, EntityTemplateToml>,
    pub terrains: HashMap<u32, TerrainTemplateToml>,
    pub scenarios: HashMap<String, ScenarioToml>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_def_round_trip() {
        let toml_str = r#"
            name = "Warrior"
            tags = ["melee", "combat", "defense"]

            [base_stats]
            hp = 120.0
            attack = 15.0
            armor = 5.0
            speed = 2.5

            [per_level]
            hp = 10.0
            attack = 1.5
            armor = 0.5
            speed = 0.0

            [requirements]
            min_level = 3
            [requirements.behavior]
            aggression = 0.6
            discipline = 0.5

            [score_weights]
            combat = 1.5
            defense = 1.2
            leadership = 0.8

            [abilities]
            pool = ["shield_bash", "battle_cry", "cleave"]
        "#;

        let def: ClassDefToml = toml::from_str(toml_str).expect("parse ClassDefToml");

        assert_eq!(def.name, "Warrior");
        assert_eq!(def.tags, ["melee", "combat", "defense"]);
        assert_eq!(def.base_stats.hp, 120.0);
        assert_eq!(def.base_stats.attack, 15.0);
        assert_eq!(def.per_level.hp, 10.0);
        assert_eq!(def.requirements.min_level, 3);
        assert_eq!(def.requirements.behavior["aggression"], 0.6);
        assert_eq!(def.score_weights["combat"], 1.5);
        assert_eq!(def.abilities.pool, ["shield_bash", "battle_cry", "cleave"]);

        // Round-trip through toml serialization
        let serialized = toml::to_string(&def).expect("serialize ClassDefToml");
        let def2: ClassDefToml = toml::from_str(&serialized).expect("re-parse ClassDefToml");
        assert_eq!(def2.name, def.name);
        assert_eq!(def2.tags.len(), def.tags.len());
    }

    #[test]
    fn entity_template_round_trip() {
        let toml_str = r#"
            name = "Town Guard"
            kind = "humanoid"

            [stats]
            hp = 80.0
            attack = 12.0
            armor = 3.0
            speed = 3.0
            attack_range = 1.5

            [[classes.starting]]
            name = "Warrior"
            level = 2

            [[classes.starting]]
            name = "Guard"
            level = 1

            [abilities]
            list = ["bash", "defend"]

            [attack]
            damage = 20
            range = 1.5
            cooldown = 1200
            cast_time = 250

            [capabilities]
            can_jump = true
            jump_height = 2
            can_climb = true
            can_tunnel = false
            can_fly = false
            has_siege = false
            siege_damage = 0.0
        "#;

        let tmpl: EntityTemplateToml = toml::from_str(toml_str).expect("parse EntityTemplateToml");

        assert_eq!(tmpl.name, "Town Guard");
        assert_eq!(tmpl.kind, "humanoid");
        assert_eq!(tmpl.stats.hp, 80.0);
        assert_eq!(tmpl.stats.armor, 3.0);
        assert_eq!(tmpl.classes.starting.len(), 2);
        assert_eq!(tmpl.classes.starting[0].name, "Warrior");
        assert_eq!(tmpl.classes.starting[0].level, 2);
        assert_eq!(tmpl.classes.starting[1].name, "Guard");
        assert_eq!(tmpl.abilities.list, ["bash", "defend"]);

        let attack = tmpl.attack.as_ref().expect("attack config present");
        assert_eq!(attack.damage, 20);
        assert_eq!(attack.cooldown, 1200);

        let caps = tmpl.capabilities.as_ref().expect("capabilities present");
        assert!(caps.can_jump);
        assert_eq!(caps.jump_height, 2);
        assert!(caps.can_climb);
        assert!(!caps.can_fly);

        // Round-trip
        let serialized = toml::to_string(&tmpl).expect("serialize EntityTemplateToml");
        let tmpl2: EntityTemplateToml = toml::from_str(&serialized).expect("re-parse EntityTemplateToml");
        assert_eq!(tmpl2.name, tmpl.name);
        assert_eq!(tmpl2.classes.starting.len(), tmpl.classes.starting.len());
    }

    #[test]
    fn scenario_round_trip() {
        let toml_str = r#"
            name = "Valley Siege"
            terrain = "highland_valley"

            [[npcs]]
            template = "town_guard"
            count = 4
            level_range = [2, 5]
            [npcs.state]
            morale = 0.8
            stress = 0.2

            [[npcs]]
            template = "blacksmith"
            count = 1

            [[buildings]]
            type = "barracks"
            pos = [10, 12]

            [[buildings]]
            type = "forge"
            pos = [20, 8]

            [[history]]
            event = "orc_raid"
            ticks_ago = 500
            severity = 0.7
            summary = "A band of orcs pillaged the outer farms."
            [[history.consequences]]
            type = "Casualties"
            template = "farmer"
            count = 3
            [[history.consequences]]
            type = "MoraleImpact"
            amount = -0.15

            [initial_state]
            morale = 0.6
            treasury = 250.0
            [initial_state.stockpile]
            food = 80.0
            wood = 120.0

            [[threats]]
            name = "goblin_warband"
            trigger_tick = 100
            duration_ticks = 200
            approach_direction = [1.0, 0.0]
            [[threats.entities]]
            template = "goblin"
            count = 8
            level_range = [1, 3]
            [threats.resolution]
            check = "enemies_dead"

            [[events]]
            type = "plague"
            severity = 0.4
            trigger_tick = 300
            duration_ticks = 150
            deadline_ticks = 500
            [events.resolution]
            check = "healer_present"

            [completion]
            mode = "all"
            max_ticks = 1500
            failure = "settlement_destroyed"
        "#;

        let scenario: ScenarioToml = toml::from_str(toml_str).expect("parse ScenarioToml");

        assert_eq!(scenario.name, "Valley Siege");
        assert_eq!(scenario.terrain.as_deref(), Some("highland_valley"));

        // NPCs
        assert_eq!(scenario.npcs.len(), 2);
        assert_eq!(scenario.npcs[0].template, "town_guard");
        assert_eq!(scenario.npcs[0].count, 4);
        assert_eq!(scenario.npcs[0].level_range, Some([2, 3_u32].map(|_| 0)
            .into_iter().enumerate().map(|(i, _)| [2u32, 5][i]).collect::<Vec<_>>()
            .try_into().unwrap()));
        let npc_state = scenario.npcs[0].state.as_ref().expect("npc state present");
        assert_eq!(npc_state.morale, Some(0.8));

        // Buildings
        assert_eq!(scenario.buildings.len(), 2);
        assert_eq!(scenario.buildings[0].building_type, "barracks");
        assert_eq!(scenario.buildings[0].pos, [10, 12]);

        // History
        assert_eq!(scenario.history.len(), 1);
        assert_eq!(scenario.history[0].event, "orc_raid");
        assert_eq!(scenario.history[0].consequences.len(), 2);
        match &scenario.history[0].consequences[0] {
            HistoryConsequence::Casualties { template, count } => {
                assert_eq!(template, "farmer");
                assert_eq!(*count, 3);
            }
            other => panic!("expected Casualties, got {other:?}"),
        }
        match &scenario.history[0].consequences[1] {
            HistoryConsequence::MoraleImpact { amount } => {
                assert!((*amount - (-0.15)).abs() < 1e-6, "morale amount mismatch");
            }
            other => panic!("expected MoraleImpact, got {other:?}"),
        }

        // Initial state
        let init = scenario.initial_state.as_ref().expect("initial_state present");
        assert_eq!(init.morale, Some(0.6));
        let stockpile = init.stockpile.as_ref().expect("stockpile present");
        assert_eq!(stockpile.food, Some(80.0));

        // Threats
        assert_eq!(scenario.threats.len(), 1);
        assert_eq!(scenario.threats[0].name, "goblin_warband");
        assert_eq!(scenario.threats[0].entities.len(), 1);
        assert_eq!(scenario.threats[0].entities[0].template, "goblin");
        assert_eq!(scenario.threats[0].entities[0].count, 8);
        let threat_res = scenario.threats[0].resolution.as_ref().expect("threat resolution");
        match threat_res {
            ResolutionConfig::Single { check, .. } => assert_eq!(check, "enemies_dead"),
            other => panic!("expected Single resolution, got {other:?}"),
        }

        // Events
        assert_eq!(scenario.events.len(), 1);
        assert_eq!(scenario.events[0].event_type, "plague");
        assert_eq!(scenario.events[0].severity, 0.4);
        assert_eq!(scenario.events[0].deadline_ticks, Some(500));

        // Completion
        let completion = scenario.completion.as_ref().expect("completion present");
        assert_eq!(completion.mode, "all");
        assert_eq!(completion.max_ticks, 1500);
        assert_eq!(completion.failure.as_deref(), Some("settlement_destroyed"));
    }
}
