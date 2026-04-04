//! Data-driven registry for TOML-defined classes, entity templates,
//! terrain templates, and scenario definitions.
//!
//! Defines the serde types for each TOML format and the `Registry` loader
//! that scans `dataset/` subdirectories, parses files, and validates
//! cross-references between them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
#[serde(tag = "type", rename_all = "snake_case")]
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
    pub params: HashMap<String, toml::Value>,
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
        params: HashMap<String, toml::Value>,
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
// Registry loading
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct RegistryError {
    pub file: String,
    pub message: String,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.message)
    }
}

impl std::error::Error for RegistryError {}

impl Registry {
    /// Load registry from a dataset directory.
    ///
    /// Scans `classes/`, `entities/`, `environments/terrains/`, and
    /// `environments/scenarios/` subdirectories. Parses each TOML file and
    /// validates cross-references. Returns collected errors rather than
    /// failing on the first.
    pub fn load(dataset_path: &Path) -> Result<Self, Vec<RegistryError>> {
        let mut registry = Registry::default();
        let mut errors = Vec::new();

        // Load classes from dataset/classes/*.toml
        let classes_dir = dataset_path.join("classes");
        if classes_dir.is_dir() {
            let files = Self::read_dir_files(&classes_dir, "toml", &mut errors);
            for (path, content) in files {
                match toml::from_str::<ClassDefToml>(&content) {
                    Ok(def) => {
                        let hash = tag(def.name.as_bytes());
                        if registry.classes.contains_key(&hash) {
                            errors.push(RegistryError {
                                file: path.display().to_string(),
                                message: format!("duplicate class name '{}'", def.name),
                            });
                        } else {
                            registry.classes.insert(hash, def);
                        }
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            }
        }

        // Load entities from dataset/entities/*.toml (and subdirs)
        let entities_dir = dataset_path.join("entities");
        if entities_dir.is_dir() {
            let files = Self::read_dir_files_recursive(&entities_dir, "toml", &mut errors);
            for (path, content) in files {
                match toml::from_str::<EntityTemplateToml>(&content) {
                    Ok(tmpl) => {
                        let hash = tag(tmpl.name.as_bytes());
                        if registry.entities.contains_key(&hash) {
                            errors.push(RegistryError {
                                file: path.display().to_string(),
                                message: format!("duplicate entity name '{}'", tmpl.name),
                            });
                        } else {
                            registry.entities.insert(hash, tmpl);
                        }
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            }
        }

        // Load terrains from dataset/environments/terrains/*.toml
        let terrains_dir = dataset_path.join("environments").join("terrains");
        if terrains_dir.is_dir() {
            let files = Self::read_dir_files(&terrains_dir, "toml", &mut errors);
            for (path, content) in files {
                match toml::from_str::<TerrainTemplateToml>(&content) {
                    Ok(tmpl) => {
                        let hash = tag(tmpl.name.as_bytes());
                        if registry.terrains.contains_key(&hash) {
                            errors.push(RegistryError {
                                file: path.display().to_string(),
                                message: format!("duplicate terrain name '{}'", tmpl.name),
                            });
                        } else {
                            registry.terrains.insert(hash, tmpl);
                        }
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            }
        }

        // Load scenarios from dataset/environments/scenarios/*.toml
        let scenarios_dir = dataset_path.join("environments").join("scenarios");
        if scenarios_dir.is_dir() {
            let files = Self::read_dir_files(&scenarios_dir, "toml", &mut errors);
            for (path, content) in files {
                match toml::from_str::<ScenarioToml>(&content) {
                    Ok(scenario) => {
                        if registry.scenarios.contains_key(&scenario.name) {
                            errors.push(RegistryError {
                                file: path.display().to_string(),
                                message: format!("duplicate scenario name '{}'", scenario.name),
                            });
                        } else {
                            registry.scenarios.insert(scenario.name.clone(), scenario);
                        }
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            }
        }

        // Validate cross-references.
        Self::validate(&registry, &mut errors);

        if errors.is_empty() {
            Ok(registry)
        } else {
            Err(errors)
        }
    }

    /// Read all files with the given extension from a directory (non-recursive).
    /// Returns `(path, content)` pairs. Pushes IO errors to the error list.
    fn read_dir_files(
        dir: &Path,
        ext: &str,
        errors: &mut Vec<RegistryError>,
    ) -> Vec<(PathBuf, String)> {
        let mut results = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                errors.push(RegistryError {
                    file: dir.display().to_string(),
                    message: format!("cannot read directory: {e}"),
                });
                return results;
            }
        };
        for entry_result in entries {
            let entry = match entry_result {
                Ok(e) => e,
                Err(e) => {
                    errors.push(RegistryError {
                        file: dir.display().to_string(),
                        message: format!("directory entry error: {e}"),
                    });
                    continue;
                }
            };
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(ext) {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(content) => results.push((path, content)),
                Err(e) => errors.push(RegistryError {
                    file: path.display().to_string(),
                    message: format!("read error: {e}"),
                }),
            }
        }
        results
    }

    /// Read all files with the given extension from a directory tree (recursive).
    /// Returns `(path, content)` pairs. Pushes IO errors to the error list.
    fn read_dir_files_recursive(
        dir: &Path,
        ext: &str,
        errors: &mut Vec<RegistryError>,
    ) -> Vec<(PathBuf, String)> {
        let mut results = Vec::new();
        Self::read_dir_files_recursive_inner(dir, ext, errors, &mut results);
        results
    }

    fn read_dir_files_recursive_inner(
        dir: &Path,
        ext: &str,
        errors: &mut Vec<RegistryError>,
        results: &mut Vec<(PathBuf, String)>,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                errors.push(RegistryError {
                    file: dir.display().to_string(),
                    message: format!("cannot read directory: {e}"),
                });
                return;
            }
        };
        for entry_result in entries {
            let entry = match entry_result {
                Ok(e) => e,
                Err(e) => {
                    errors.push(RegistryError {
                        file: dir.display().to_string(),
                        message: format!("directory entry error: {e}"),
                    });
                    continue;
                }
            };
            let path = entry.path();
            if path.is_dir() {
                Self::read_dir_files_recursive_inner(&path, ext, errors, results);
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                match std::fs::read_to_string(&path) {
                    Ok(content) => results.push((path, content)),
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("read error: {e}"),
                    }),
                }
            }
        }
    }

    fn validate(registry: &Registry, errors: &mut Vec<RegistryError>) {
        // Validate entity starting classes reference existing class definitions.
        for (_, entity) in &registry.entities {
            for sc in &entity.classes.starting {
                let class_hash = tag(sc.name.as_bytes());
                if !registry.classes.contains_key(&class_hash) {
                    errors.push(RegistryError {
                        file: format!("entity:{}", entity.name),
                        message: format!(
                            "starting class '{}' not found in class registry",
                            sc.name
                        ),
                    });
                }
            }
        }

        // Validate scenario entity/terrain references.
        for (name, scenario) in &registry.scenarios {
            if let Some(ref terrain_name) = scenario.terrain {
                let terrain_hash = tag(terrain_name.as_bytes());
                if !registry.terrains.contains_key(&terrain_hash) {
                    errors.push(RegistryError {
                        file: format!("scenario:{name}"),
                        message: format!(
                            "terrain '{}' not found in terrain registry",
                            terrain_name
                        ),
                    });
                }
            }
            for npc in &scenario.npcs {
                let tmpl_hash = tag(npc.template.as_bytes());
                if !registry.entities.contains_key(&tmpl_hash) {
                    errors.push(RegistryError {
                        file: format!("scenario:{name}"),
                        message: format!(
                            "NPC template '{}' not found in entity registry",
                            npc.template
                        ),
                    });
                }
            }
            for threat in &scenario.threats {
                for te in &threat.entities {
                    let tmpl_hash = tag(te.template.as_bytes());
                    if !registry.entities.contains_key(&tmpl_hash) {
                        errors.push(RegistryError {
                            file: format!("scenario:{name}"),
                            message: format!(
                                "threat entity template '{}' not found in entity registry",
                                te.template
                            ),
                        });
                    }
                }
            }
        }
    }

    /// Look up a class definition by name.
    pub fn class_by_name(&self, name: &str) -> Option<&ClassDefToml> {
        self.classes.get(&tag(name.as_bytes()))
    }

    /// Look up an entity template by name.
    pub fn entity_by_name(&self, name: &str) -> Option<&EntityTemplateToml> {
        self.entities.get(&tag(name.as_bytes()))
    }

    /// Look up a terrain template by name.
    pub fn terrain_by_name(&self, name: &str) -> Option<&TerrainTemplateToml> {
        self.terrains.get(&tag(name.as_bytes()))
    }

    /// Look up a scenario by name.
    pub fn scenario_by_name(&self, name: &str) -> Option<&ScenarioToml> {
        self.scenarios.get(name)
    }
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
            type = "casualties"
            template = "farmer"
            count = 3
            [[history.consequences]]
            type = "morale_impact"
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
        assert_eq!(scenario.npcs[0].level_range, Some([2, 5]));
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

    #[test]
    fn registry_load_from_temp_dir() {
        let dir = std::env::temp_dir().join("registry_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("classes")).unwrap();
        std::fs::create_dir_all(dir.join("entities")).unwrap();

        std::fs::write(
            dir.join("classes/warrior.toml"),
            r#"
name = "Warrior"
tags = ["melee"]
[base_stats]
hp = 120.0
attack = 15.0
armor = 5.0
speed = 3.0
[per_level]
hp = 10.0
attack = 2.5
armor = 1.7
speed = 0.02
"#,
        )
        .unwrap();

        std::fs::write(
            dir.join("entities/town_guard.toml"),
            r#"
name = "Town Guard"
kind = "npc"
[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0
[[classes.starting]]
name = "Warrior"
level = 2
"#,
        )
        .unwrap();

        let registry = Registry::load(&dir).unwrap();
        assert_eq!(registry.classes.len(), 1);
        assert!(registry.class_by_name("Warrior").is_some());
        assert_eq!(registry.entities.len(), 1);
        assert!(registry.entity_by_name("Town Guard").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn registry_validates_missing_class_ref() {
        let dir = std::env::temp_dir().join("registry_test_validate");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("classes")).unwrap();
        std::fs::create_dir_all(dir.join("entities")).unwrap();

        // Entity references "Warrior" class but no class file exists.
        std::fs::write(
            dir.join("entities/guard.toml"),
            r#"
name = "Guard"
kind = "npc"
[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0
[[classes.starting]]
name = "Warrior"
level = 2
"#,
        )
        .unwrap();

        let result = Registry::load(&dir);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("Warrior")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entity_from_template() {
        use crate::world_sim::state::Entity;

        let class_toml = r#"
name = "Warrior"
tags = ["melee"]
[base_stats]
hp = 10.0
attack = 2.5
armor = 1.7
speed = 0.02
[per_level]
hp = 10.0
attack = 2.5
armor = 1.7
speed = 0.02
"#;
        let entity_toml = r#"
name = "Town Guard"
kind = "npc"
[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0
[[classes.starting]]
name = "Warrior"
level = 3
"#;
        let class_def: ClassDefToml = toml::from_str(class_toml).unwrap();
        let entity_tmpl: EntityTemplateToml = toml::from_str(entity_toml).unwrap();

        let mut registry = Registry::default();
        registry.classes.insert(tag(b"Warrior"), class_def);
        registry.entities.insert(tag(b"Town Guard"), entity_tmpl.clone());

        let entity = Entity::from_template(1, (10.0, 20.0), &entity_tmpl, &registry);
        assert_eq!(entity.hp, 100.0);
        assert_eq!(entity.attack_damage, 10.0);
        assert_eq!(entity.level, 0); // starts at 0 for progression
        let npc = entity.npc.as_ref().unwrap();
        assert_eq!(npc.classes.len(), 1);
        assert_eq!(npc.classes[0].level, 3);
        assert_eq!(npc.classes[0].class_name_hash, tag(b"Warrior"));
    }

    #[test]
    fn entity_from_template_creature() {
        use crate::world_sim::state::Entity;

        let entity_toml = r#"
name = "Wall Jumper"
kind = "creature"
[stats]
hp = 70.0
attack = 10.0
armor = 3.0
speed = 3.5
[capabilities]
can_jump = true
jump_height = 4
"#;
        let entity_tmpl: EntityTemplateToml = toml::from_str(entity_toml).unwrap();
        let registry = Registry::default();

        let entity = Entity::from_template(2, (5.0, 5.0), &entity_tmpl, &registry);
        assert_eq!(entity.kind, crate::world_sim::state::EntityKind::Monster);
        assert_eq!(entity.team, crate::world_sim::state::WorldTeam::Hostile);
        assert_eq!(entity.hp, 70.0);
        let caps = entity.enemy_capabilities.as_ref().unwrap();
        assert!(caps.can_jump);
        assert_eq!(caps.jump_height, 4);
    }

    #[test]
    fn load_real_class_files() {
        let dataset_path = std::path::Path::new("dataset");
        if !dataset_path.join("classes").is_dir() {
            return; // skip if dataset not present
        }
        let classes_dir = dataset_path.join("classes");
        let mut count = 0;
        for entry in std::fs::read_dir(&classes_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            let def: ClassDefToml = toml::from_str(&content)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
            assert!(!def.name.is_empty(), "{}: name is empty", path.display());
            assert!(def.per_level.hp > 0.0, "{}: per_level.hp should be > 0", path.display());
            count += 1;
        }
        assert!(count >= 31, "Expected at least 31 class files, found {count}");
    }
}
