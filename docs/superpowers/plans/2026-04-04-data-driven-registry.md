# Data-Driven Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded class definitions, entity construction, and scenario definitions with a unified data-driven registry loaded from TOML at runtime.

**Architecture:** A `Registry` struct holds parsed class definitions, entity templates, and scenario configs loaded from `dataset/`. Systems that currently use hardcoded data (progression.rs, class_gen.rs, mass_gen.rs) are migrated to read from the registry. The registry is stored on `WorldState` and passed by reference to consuming systems.

**Tech Stack:** Rust, serde + toml crate (already in Cargo.toml), existing `.ability` DSL parser

**Spec:** `docs/superpowers/specs/2026-04-04-data-driven-registry-design.md`

---

## File Structure

### New Files
- `src/world_sim/registry.rs` — Registry struct, TOML types, loading, validation
- `dataset/classes/*.toml` — Class definition files (31 files, one per class from class_gen.rs)
- `dataset/entities/town_guard.toml` — Example NPC entity templates for building AI
- `dataset/entities/farmer.toml`
- `dataset/entities/raider_infantry.toml` — Example hostile entity template
- `dataset/environments/terrains/river_valley.toml` — Example terrain template
- `dataset/environments/scenarios/frontier_outpost.toml` — Example scenario

### Modified Files
- `src/world_sim/mod.rs` — Add `pub mod registry;`
- `src/world_sim/state.rs` — Add `registry: Option<Arc<Registry>>` to WorldState, add `Entity::from_template()`
- `src/world_sim/systems/progression.rs` — Replace `class_bonus()` with registry lookup
- `src/world_sim/class_gen.rs` — Replace `TEMPLATES` array with `RegistryClassGenerator`
- `src/world_sim/runtime.rs` — Use `RegistryClassGenerator` when registry is present
- `src/world_sim/building_ai/mass_gen.rs` — Use `Entity::from_template()` in `generate_npc_roster()`

---

### Task 1: Define Registry TOML Types

**Files:**
- Create: `src/world_sim/registry.rs`
- Modify: `src/world_sim/mod.rs`

- [ ] **Step 1: Write test for ClassDef deserialization**

```rust
// src/world_sim/registry.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::world_sim::state::tag;

// ---------------------------------------------------------------------------
// Stat block — shared between classes and entities
// ---------------------------------------------------------------------------

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
// Class definition (loaded from dataset/classes/*.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDefToml {
    pub name: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub base_stats: StatBlock,
    pub per_level: StatBlock,
    #[serde(default)]
    pub requirements: ClassRequirements,
    #[serde(default)]
    pub score_weights: HashMap<String, f32>,
    #[serde(default)]
    pub abilities: ClassAbilities,
}

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

// ---------------------------------------------------------------------------
// Entity template (loaded from dataset/entities/*.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTemplateToml {
    pub name: String,
    pub kind: String,
    pub stats: EntityStats,
    #[serde(default)]
    pub classes: EntityClasses,
    #[serde(default)]
    pub abilities: EntityAbilities,
    #[serde(default)]
    pub attack: Option<AttackConfig>,
    #[serde(default)]
    pub capabilities: Option<CapabilitiesConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityStats {
    #[serde(default = "default_hp")]
    pub hp: f32,
    #[serde(default = "default_attack")]
    pub attack: f32,
    #[serde(default)]
    pub armor: f32,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_attack_range")]
    pub attack_range: f32,
}

fn default_hp() -> f32 { 100.0 }
fn default_attack() -> f32 { 10.0 }
fn default_speed() -> f32 { 3.0 }
fn default_attack_range() -> f32 { 1.5 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityClasses {
    #[serde(default)]
    pub starting: Vec<StartingClass>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartingClass {
    pub name: String,
    #[serde(default = "default_class_level")]
    pub level: u16,
}

fn default_class_level() -> u16 { 1 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityAbilities {
    #[serde(default)]
    pub list: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackConfig {
    #[serde(default = "default_attack_damage")]
    pub damage: i32,
    #[serde(default = "default_attack_range")]
    pub range: f32,
    #[serde(default = "default_cooldown")]
    pub cooldown: u32,
    #[serde(default = "default_cast_time")]
    pub cast_time: u32,
}

fn default_attack_damage() -> i32 { 15 }
fn default_cooldown() -> u32 { 1000 }
fn default_cast_time() -> u32 { 300 }

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

// ---------------------------------------------------------------------------
// Terrain template (loaded from dataset/environments/terrains/*.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainTemplateToml {
    pub name: String,
    #[serde(default = "default_terrain_size")]
    pub size: [u32; 2],
    #[serde(default)]
    pub biome: String,
    #[serde(default)]
    pub resources: Vec<TerrainResource>,
    #[serde(default)]
    pub features: Vec<TerrainFeature>,
}

fn default_terrain_size() -> [u32; 2] { [64, 64] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainResource {
    #[serde(rename = "type")]
    pub resource_type: String,
    pub amount: f32,
    pub pos: [u32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainFeature {
    #[serde(rename = "type")]
    pub feature_type: String,
    #[serde(default)]
    pub from: Option<[u32; 2]>,
    #[serde(default)]
    pub to: Option<[u32; 2]>,
    #[serde(default)]
    pub pos: Option<[u32; 2]>,
    #[serde(default)]
    pub radius: Option<f32>,
}

// ---------------------------------------------------------------------------
// Scenario definition (loaded from dataset/environments/scenarios/*.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioToml {
    pub name: String,
    #[serde(default)]
    pub terrain: Option<String>,
    #[serde(default)]
    pub npcs: Vec<ScenarioNpc>,
    #[serde(default)]
    pub buildings: Vec<ScenarioBuilding>,
    #[serde(default)]
    pub history: Vec<HistoryEvent>,
    #[serde(default)]
    pub initial_state: Option<InitialStateOverrides>,
    #[serde(default)]
    pub threats: Vec<ThreatConfig>,
    #[serde(default)]
    pub events: Vec<EventConfig>,
    #[serde(default)]
    pub completion: Option<CompletionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioNpc {
    pub template: String,
    #[serde(default = "default_count")]
    pub count: u32,
    #[serde(default)]
    pub level_range: Option<[u32; 2]>,
    #[serde(default)]
    pub state: Option<NpcStateOverrides>,
}

fn default_count() -> u32 { 1 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NpcStateOverrides {
    #[serde(default)]
    pub morale: Option<f32>,
    #[serde(default)]
    pub stress: Option<f32>,
    #[serde(default)]
    pub fatigue: Option<f32>,
    #[serde(default)]
    pub injury: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioBuilding {
    #[serde(rename = "type")]
    pub building_type: String,
    pub pos: [u32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEvent {
    pub event: String,
    #[serde(default)]
    pub ticks_ago: u64,
    #[serde(default)]
    pub severity: f32,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub consequences: Vec<HistoryConsequence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HistoryConsequence {
    #[serde(rename = "casualties")]
    Casualties { template: String, count: u32 },
    #[serde(rename = "building_damaged")]
    BuildingDamaged { building: String, hp_fraction: f32 },
    #[serde(rename = "building_destroyed")]
    BuildingDestroyed { building: String },
    #[serde(rename = "morale_impact")]
    MoraleImpact { amount: f32 },
    #[serde(rename = "population_state")]
    PopulationState {
        #[serde(default)]
        stress: Option<f32>,
        #[serde(default)]
        fatigue: Option<f32>,
        #[serde(default)]
        injury: Option<f32>,
        #[serde(default)]
        morale: Option<f32>,
    },
    #[serde(rename = "stockpile_drain")]
    StockpileDrain { resource: String, fraction: f32 },
    #[serde(rename = "treasury_drain")]
    TreasuryDrain { fraction: f32 },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InitialStateOverrides {
    #[serde(default)]
    pub morale: Option<f32>,
    #[serde(default)]
    pub treasury: Option<f32>,
    #[serde(default)]
    pub stockpile: Option<StockpileOverrides>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StockpileOverrides {
    #[serde(default)]
    pub food: Option<f32>,
    #[serde(default)]
    pub wood: Option<f32>,
    #[serde(default)]
    pub iron: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatConfig {
    pub name: String,
    #[serde(default)]
    pub approach_direction: Option<[f32; 2]>,
    #[serde(default)]
    pub trigger_tick: u64,
    #[serde(default)]
    pub duration_ticks: Option<u64>,
    #[serde(default)]
    pub resolution: Option<ResolutionConfig>,
    #[serde(default)]
    pub entities: Vec<ThreatEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEntity {
    pub template: String,
    #[serde(default = "default_count")]
    pub count: u32,
    #[serde(default)]
    pub level_range: Option<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventConfig {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub severity: f32,
    #[serde(default)]
    pub trigger_tick: u64,
    #[serde(default)]
    pub duration_ticks: Option<u64>,
    #[serde(default)]
    pub deadline_ticks: Option<u64>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub resolution: Option<ResolutionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResolutionConfig {
    Single {
        check: String,
        #[serde(flatten)]
        params: HashMap<String, serde_json::Value>,
    },
    All {
        all: Vec<ResolutionCheck>,
    },
    Any {
        any: Vec<ResolutionCheck>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionCheck {
    pub check: String,
    #[serde(flatten)]
    pub params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    #[serde(default = "default_completion_mode")]
    pub mode: String,
    #[serde(default = "default_max_ticks")]
    pub max_ticks: u64,
    #[serde(default)]
    pub failure: Option<String>,
}

fn default_completion_mode() -> String { "all".to_string() }
fn default_max_ticks() -> u64 { 2000 }

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Registry {
    pub classes: HashMap<u32, ClassDefToml>,
    pub entities: HashMap<u32, EntityTemplateToml>,
    pub terrains: HashMap<u32, TerrainTemplateToml>,
    pub scenarios: HashMap<String, ScenarioToml>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_def_round_trip() {
        let toml_str = r#"
name = "Warrior"
tags = ["melee", "combat", "martial"]

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

[requirements]
min_level = 0

[requirements.behavior]
melee = 100.0
combat = 50.0

[score_weights]
melee = 0.4
defense = 0.3
endurance = 0.2
combat = 0.1

[abilities]
pool = ["ShieldBash", "Charge"]
"#;
        let def: ClassDefToml = toml::from_str(toml_str).unwrap();
        assert_eq!(def.name, "Warrior");
        assert_eq!(def.tags, vec!["melee", "combat", "martial"]);
        assert_eq!(def.base_stats.hp, 120.0);
        assert_eq!(def.per_level.attack, 2.5);
        assert_eq!(def.requirements.min_level, 0);
        assert!((def.requirements.behavior["melee"] - 100.0).abs() < 0.01);
        assert!((def.score_weights["melee"] - 0.4).abs() < 0.01);
        assert_eq!(def.abilities.pool, vec!["ShieldBash", "Charge"]);
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test registry::tests::class_def_round_trip -- --nocapture`
Expected: PASS

- [ ] **Step 3: Add entity template and scenario deserialization tests**

Add to `src/world_sim/registry.rs` tests module:

```rust
    #[test]
    fn entity_template_round_trip() {
        let toml_str = r#"
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

[abilities]
list = ["BasicAttack"]
"#;
        let tmpl: EntityTemplateToml = toml::from_str(toml_str).unwrap();
        assert_eq!(tmpl.name, "Town Guard");
        assert_eq!(tmpl.kind, "npc");
        assert_eq!(tmpl.stats.hp, 100.0);
        assert_eq!(tmpl.classes.starting.len(), 1);
        assert_eq!(tmpl.classes.starting[0].name, "Warrior");
        assert_eq!(tmpl.classes.starting[0].level, 2);
    }

    #[test]
    fn entity_template_with_capabilities() {
        let toml_str = r#"
name = "Wall Jumper"
kind = "creature"

[stats]
hp = 80.0
attack = 12.0
armor = 2.0
speed = 4.0

[capabilities]
can_jump = true
jump_height = 4
"#;
        let tmpl: EntityTemplateToml = toml::from_str(toml_str).unwrap();
        let caps = tmpl.capabilities.unwrap();
        assert!(caps.can_jump);
        assert_eq!(caps.jump_height, 4);
        assert!(!caps.can_fly);
    }

    #[test]
    fn scenario_round_trip() {
        let toml_str = r#"
name = "Test Scenario"
terrain = "river_valley"

[[npcs]]
template = "town_guard"
count = 5
level_range = [2, 4]

[[buildings]]
type = "Barracks"
pos = [30, 30]

[[history]]
event = "raid"
ticks_ago = 100
severity = 0.7
summary = "Raiders attacked"

[[history.consequences]]
type = "casualties"
template = "town_guard"
count = 2

[[threats]]
name = "Raider Party"
approach_direction = [0.0, -1.0]
trigger_tick = 500

[threats.resolution]
check = "hostiles_dead"

[[threats.entities]]
template = "raider_infantry"
count = 6
level_range = [2, 4]

[[events]]
type = "earthquake"
severity = 0.6
trigger_tick = 300
duration_ticks = 50

[events.resolution]
check = "buildings_stable"
hp_threshold = 0.5

[completion]
mode = "all"
max_ticks = 2000
failure = "settlement_wiped"
"#;
        let scenario: ScenarioToml = toml::from_str(toml_str).unwrap();
        assert_eq!(scenario.name, "Test Scenario");
        assert_eq!(scenario.terrain.as_deref(), Some("river_valley"));
        assert_eq!(scenario.npcs.len(), 1);
        assert_eq!(scenario.npcs[0].count, 5);
        assert_eq!(scenario.buildings.len(), 1);
        assert_eq!(scenario.history.len(), 1);
        assert_eq!(scenario.history[0].consequences.len(), 1);
        assert_eq!(scenario.threats.len(), 1);
        assert_eq!(scenario.threats[0].entities.len(), 1);
        assert_eq!(scenario.events.len(), 1);
        assert!(scenario.completion.is_some());
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test registry::tests -- --nocapture`
Expected: 3 tests PASS

- [ ] **Step 5: Register the module**

In `src/world_sim/mod.rs`, add:

```rust
pub mod registry;
```

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/registry.rs src/world_sim/mod.rs
git commit -m "feat: add registry TOML types for classes, entities, terrains, and scenarios"
```

---

### Task 2: Implement Registry Loading

**Files:**
- Modify: `src/world_sim/registry.rs`

- [ ] **Step 1: Write test for Registry::load()**

Add to `src/world_sim/registry.rs`:

```rust
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
            Self::load_dir(&classes_dir, "toml", &mut errors, |path, content| {
                match toml::from_str::<ClassDefToml>(&content) {
                    Ok(def) => {
                        let hash = tag(def.name.as_bytes());
                        registry.classes.insert(hash, def);
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            });
        }

        // Load entities from dataset/entities/*.toml (and subdirs)
        let entities_dir = dataset_path.join("entities");
        if entities_dir.is_dir() {
            Self::load_dir_recursive(&entities_dir, "toml", &mut errors, |path, content| {
                match toml::from_str::<EntityTemplateToml>(&content) {
                    Ok(tmpl) => {
                        let hash = tag(tmpl.name.as_bytes());
                        registry.entities.insert(hash, tmpl);
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            });
        }

        // Load terrains from dataset/environments/terrains/*.toml
        let terrains_dir = dataset_path.join("environments").join("terrains");
        if terrains_dir.is_dir() {
            Self::load_dir(&terrains_dir, "toml", &mut errors, |path, content| {
                match toml::from_str::<TerrainTemplateToml>(&content) {
                    Ok(tmpl) => {
                        let hash = tag(tmpl.name.as_bytes());
                        registry.terrains.insert(hash, tmpl);
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            });
        }

        // Load scenarios from dataset/environments/scenarios/*.toml
        let scenarios_dir = dataset_path.join("environments").join("scenarios");
        if scenarios_dir.is_dir() {
            Self::load_dir(&scenarios_dir, "toml", &mut errors, |path, content| {
                match toml::from_str::<ScenarioToml>(&content) {
                    Ok(scenario) => {
                        registry.scenarios.insert(scenario.name.clone(), scenario);
                    }
                    Err(e) => errors.push(RegistryError {
                        file: path.display().to_string(),
                        message: format!("parse error: {e}"),
                    }),
                }
            });
        }

        // Validate cross-references.
        Self::validate(&registry, &mut errors);

        if errors.is_empty() {
            Ok(registry)
        } else {
            Err(errors)
        }
    }

    fn load_dir(
        dir: &Path,
        ext: &str,
        errors: &mut Vec<RegistryError>,
        mut handler: impl FnMut(&Path, String),
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
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(ext) {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(content) => handler(&path, content),
                Err(e) => errors.push(RegistryError {
                    file: path.display().to_string(),
                    message: format!("read error: {e}"),
                }),
            }
        }
    }

    fn load_dir_recursive(
        dir: &Path,
        ext: &str,
        errors: &mut Vec<RegistryError>,
        handler: &mut impl FnMut(&Path, String),
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
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::load_dir_recursive(&path, ext, errors, handler);
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                match std::fs::read_to_string(&path) {
                    Ok(content) => handler(&path, content),
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
                        message: format!("starting class '{}' not found in class registry", sc.name),
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
                        message: format!("terrain '{}' not found in terrain registry", terrain_name),
                    });
                }
            }
            for npc in &scenario.npcs {
                let tmpl_hash = tag(npc.template.as_bytes());
                if !registry.entities.contains_key(&tmpl_hash) {
                    errors.push(RegistryError {
                        file: format!("scenario:{name}"),
                        message: format!("NPC template '{}' not found in entity registry", npc.template),
                    });
                }
            }
            for threat in &scenario.threats {
                for te in &threat.entities {
                    let tmpl_hash = tag(te.template.as_bytes());
                    if !registry.entities.contains_key(&tmpl_hash) {
                        errors.push(RegistryError {
                            file: format!("scenario:{name}"),
                            message: format!("threat entity template '{}' not found in entity registry", te.template),
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
```

- [ ] **Step 2: Write filesystem-based load test**

Add to tests module:

```rust
    #[test]
    fn registry_load_from_temp_dir() {
        let dir = std::env::temp_dir().join("registry_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("classes")).unwrap();
        std::fs::create_dir_all(dir.join("entities")).unwrap();

        std::fs::write(dir.join("classes/warrior.toml"), r#"
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
"#).unwrap();

        std::fs::write(dir.join("entities/town_guard.toml"), r#"
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
"#).unwrap();

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
        std::fs::write(dir.join("entities/guard.toml"), r#"
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
"#).unwrap();

        let result = Registry::load(&dir);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("Warrior")));

        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test registry::tests -- --nocapture`
Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/registry.rs
git commit -m "feat: implement Registry::load() with filesystem scanning and cross-reference validation"
```

---

### Task 3: Write Class Definition TOML Files

**Files:**
- Create: `dataset/classes/warrior.toml` (and 30 more class files)

Port all 31 classes from `class_gen.rs` TEMPLATES array and `progression.rs` class_bonus(). Each file combines the template's requirements/score_tags with class_bonus's stat tuples. The `per_level` values are `BASE + class_bonus` (baked together as spec requires).

Base stats per level: HP=5.0, ATK=0.5, ARM=0.2, SPD=0.02 (from `progression.rs:18-22`).

The `base_stats` field represents one-time stats granted when the class is first acquired. Use the same values as `per_level` for now (one level's worth of stats on acquisition).

The `score_weights` field preserves the existing `score_tags` weights from `class_gen.rs` so the matching algorithm behavior doesn't change.

The `requirements.behavior` field maps from `ClassTemplate.requirements` in `class_gen.rs`.

- [ ] **Step 1: Create all 31 class TOML files**

Create each file in `dataset/classes/`. Here are the first 10 as examples (the remaining 21 follow the same pattern, porting data from `class_gen.rs:118-311` and `progression.rs:25-58`):

`dataset/classes/warrior.toml`:
```toml
name = "Warrior"
tags = ["melee", "combat", "martial"]

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

[requirements]
min_level = 0

[requirements.behavior]
melee = 100.0

[score_weights]
melee = 0.4
defense = 0.3
endurance = 0.2
combat = 0.1
```

`dataset/classes/ranger.toml`:
```toml
name = "Ranger"
tags = ["ranged", "survival", "awareness"]

[base_stats]
hp = 7.0
attack = 2.0
armor = 0.5
speed = 0.12

[per_level]
hp = 7.0
attack = 2.0
armor = 0.5
speed = 0.12

[requirements]
min_level = 0

[requirements.behavior]
ranged = 100.0

[score_weights]
ranged = 0.4
awareness = 0.2
survival = 0.2
navigation = 0.2
```

`dataset/classes/guardian.toml`:
```toml
name = "Guardian"
tags = ["defense", "endurance", "tank"]

[base_stats]
hp = 13.0
attack = 1.0
armor = 3.2
speed = 0.02

[per_level]
hp = 13.0
attack = 1.0
armor = 3.2
speed = 0.02

[requirements]
min_level = 0

[requirements.behavior]
defense = 100.0
endurance = 50.0

[score_weights]
defense = 0.4
endurance = 0.3
resilience = 0.2
combat = 0.1
```

`dataset/classes/healer.toml`:
```toml
name = "Healer"
tags = ["medicine", "faith", "support"]

[base_stats]
hp = 15.0
attack = 0.5
armor = 0.7
speed = 0.02

[per_level]
hp = 15.0
attack = 0.5
armor = 0.7
speed = 0.02

[requirements]
min_level = 0

[requirements.behavior]
medicine = 50.0

[score_weights]
medicine = 0.4
faith = 0.2
herbalism = 0.2
resilience = 0.2
```

`dataset/classes/merchant.toml`:
```toml
name = "Merchant"
tags = ["trade", "negotiation", "economics"]

[base_stats]
hp = 8.0
attack = 0.5
armor = 0.5
speed = 0.07

[per_level]
hp = 8.0
attack = 0.5
armor = 0.5
speed = 0.07

[requirements]
min_level = 0

[requirements.behavior]
trade = 100.0

[score_weights]
trade = 0.4
negotiation = 0.3
diplomacy = 0.2
navigation = 0.1
```

`dataset/classes/scholar.toml`:
```toml
name = "Scholar"
tags = ["research", "lore", "knowledge"]

[base_stats]
hp = 7.0
attack = 0.5
armor = 0.4
speed = 0.02

[per_level]
hp = 7.0
attack = 0.5
armor = 0.4
speed = 0.02

[requirements]
min_level = 0

[requirements.behavior]
research = 50.0
lore = 30.0

[score_weights]
research = 0.4
lore = 0.3
medicine = 0.15
discipline = 0.15
```

`dataset/classes/rogue.toml`:
```toml
name = "Rogue"
tags = ["stealth", "deception", "agility"]

[base_stats]
hp = 6.0
attack = 1.5
armor = 0.2
speed = 0.17

[per_level]
hp = 6.0
attack = 1.5
armor = 0.2
speed = 0.17

[requirements]
min_level = 0

[requirements.behavior]
stealth = 50.0

[score_weights]
stealth = 0.4
deception = 0.3
awareness = 0.2
survival = 0.1
```

`dataset/classes/artisan.toml`:
```toml
name = "Artisan"
tags = ["crafting", "smithing", "labor"]

[base_stats]
hp = 8.0
attack = 0.8
armor = 0.7
speed = 0.02

[per_level]
hp = 8.0
attack = 0.8
armor = 0.7
speed = 0.02

[requirements]
min_level = 0

[requirements.behavior]
crafting = 100.0

[score_weights]
crafting = 0.3
smithing = 0.3
labor = 0.2
endurance = 0.2
```

`dataset/classes/diplomat.toml`:
```toml
name = "Diplomat"
tags = ["diplomacy", "negotiation", "leadership"]

[base_stats]
hp = 8.0
attack = 0.5
armor = 0.5
speed = 0.05

[per_level]
hp = 8.0
attack = 0.5
armor = 0.5
speed = 0.05

[requirements]
min_level = 0

[requirements.behavior]
diplomacy = 100.0

[score_weights]
diplomacy = 0.4
negotiation = 0.2
leadership = 0.2
trade = 0.2
```

`dataset/classes/commander.toml`:
```toml
name = "Commander"
tags = ["leadership", "tactics", "military"]

[base_stats]
hp = 10.0
attack = 1.5
armor = 1.2
speed = 0.02

[per_level]
hp = 10.0
attack = 1.5
armor = 1.2
speed = 0.02

[requirements]
min_level = 0

[requirements.behavior]
leadership = 50.0
tactics = 30.0

[score_weights]
leadership = 0.3
tactics = 0.3
combat = 0.2
discipline = 0.2
```

Continue for remaining 21 classes: Farmer, Miner, Woodsman, Alchemist, Herbalist, Explorer, Mentor, Builder, Architect, Sentinel, Survivor, Warden, Veteran, Stalwart, Bard, Mariner, Sea Captain, Delver, Dungeon Master, Oathkeeper, Betrayer.

Each follows the pattern: `per_level` values = BASE(5.0, 0.5, 0.2, 0.02) + class_bonus from `progression.rs:25-58`. Requirements from `class_gen.rs:118-311`. Score weights from `class_gen.rs` score_tags.

Reference tables:

| Class | class_bonus (hp,atk,arm,spd) | per_level = BASE+bonus |
|-------|-----|------|
| Farmer | (4.0, 0.3, 0.3, 0.0) | (9.0, 0.8, 0.5, 0.02) |
| Miner | (6.0, 0.5, 1.0, 0.0) | (11.0, 1.0, 1.2, 0.02) |
| Woodsman | (4.0, 0.8, 0.5, 0.05) | (9.0, 1.3, 0.7, 0.07) |
| Alchemist | (2.0, 0.0, 0.2, 0.0) | (7.0, 0.5, 0.4, 0.02) |
| Herbalist | (5.0, 0.0, 0.3, 0.0) | (10.0, 0.5, 0.5, 0.02) |
| Explorer | (3.0, 0.5, 0.3, 0.1) | (8.0, 1.0, 0.5, 0.12) |
| Mentor | (4.0, 0.0, 0.5, 0.0) | (9.0, 0.5, 0.7, 0.02) |
| Builder | default(2.0, 0.3, 0.2, 0.0) | (7.0, 0.8, 0.4, 0.02) |
| Architect | default(2.0, 0.3, 0.2, 0.0) | (7.0, 0.8, 0.4, 0.02) |
| Sentinel | (6.0, 0.5, 2.0, 0.0) | (11.0, 1.0, 2.2, 0.02) |
| Survivor | (8.0, 0.3, 0.5, 0.05) | (13.0, 0.8, 0.7, 0.07) |
| Warden | (5.0, 1.0, 2.0, 0.0) | (10.0, 1.5, 2.2, 0.02) |
| Veteran | (4.0, 2.0, 1.0, 0.0) | (9.0, 2.5, 1.2, 0.02) |
| Stalwart | (10.0, 0.0, 1.5, 0.0) | (15.0, 0.5, 1.7, 0.02) |
| Bard | (3.0, 0.0, 0.2, 0.1) | (8.0, 0.5, 0.4, 0.12) |
| Mariner | (5.0, 0.5, 0.5, 0.1) | (10.0, 1.0, 0.7, 0.12) |
| Sea Captain | (5.0, 1.0, 1.0, 0.05) | (10.0, 1.5, 1.2, 0.07) |
| Delver | (4.0, 0.8, 1.0, 0.0) | (9.0, 1.3, 1.2, 0.02) |
| Dungeon Master | (3.0, 1.5, 1.5, 0.0) | (8.0, 2.0, 1.7, 0.02) |
| Oathkeeper | (8.0, 0.5, 2.0, 0.0) | (13.0, 1.0, 2.2, 0.02) |
| Betrayer | (2.0, 1.5, 0.0, 0.2) | (7.0, 2.0, 0.2, 0.22) |

- [ ] **Step 2: Write a test that loads the real dataset/classes/ directory**

Add to `src/world_sim/registry.rs` tests:

```rust
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
```

- [ ] **Step 3: Run test**

Run: `cargo test registry::tests::load_real_class_files -- --nocapture`
Expected: PASS (31+ class files parsed successfully)

- [ ] **Step 4: Commit**

```bash
git add dataset/classes/
git commit -m "feat: add 31 class definition TOML files ported from class_gen.rs + progression.rs"
```

---

### Task 4: Write Entity Template TOML Files

**Files:**
- Create: `dataset/entities/town_guard.toml`, `dataset/entities/farmer.toml`, `dataset/entities/refugee.toml`, `dataset/entities/raider_infantry.toml`, `dataset/entities/wall_jumper.toml`, `dataset/entities/siege_engine.toml`, `dataset/entities/climber.toml`, `dataset/entities/tunneler.toml`, `dataset/entities/flyer.toml`, `dataset/entities/infiltrator.toml`

These cover the entity types used in `mass_gen.rs` (generate_npc_roster + spawn_enemies). NPC templates use the world sim stat scale (hp=100 base), not the combat sim scale.

- [ ] **Step 1: Create NPC entity templates**

`dataset/entities/town_guard.toml`:
```toml
name = "Town Guard"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Warrior"
level = 1

[abilities]
list = []
```

`dataset/entities/farmer.toml`:
```toml
name = "Farmer"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Farmer"
level = 1

[abilities]
list = []
```

`dataset/entities/archer.toml`:
```toml
name = "Archer"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Ranger"
level = 1

[abilities]
list = []
```

`dataset/entities/defender.toml`:
```toml
name = "Defender"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Guardian"
level = 1

[abilities]
list = []
```

`dataset/entities/builder.toml`:
```toml
name = "Builder"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Builder"
level = 1

[abilities]
list = []
```

`dataset/entities/miner.toml`:
```toml
name = "Miner"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Miner"
level = 1

[abilities]
list = []
```

`dataset/entities/craftsman.toml`:
```toml
name = "Craftsman"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Artisan"
level = 1

[abilities]
list = []
```

`dataset/entities/merchant_npc.toml`:
```toml
name = "Merchant NPC"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Merchant"
level = 1

[abilities]
list = []
```

`dataset/entities/refugee.toml`:
```toml
name = "Refugee"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[abilities]
list = []
```

`dataset/entities/leader.toml`:
```toml
name = "Leader"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Commander"
level = 1

[abilities]
list = []
```

`dataset/entities/master_smith.toml`:
```toml
name = "Master Smith"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Artisan"
level = 1

[abilities]
list = []
```

`dataset/entities/archmage.toml`:
```toml
name = "Archmage"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Scholar"
level = 1

[abilities]
list = []
```

`dataset/entities/high_priest.toml`:
```toml
name = "High Priest"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Healer"
level = 1

[abilities]
list = []
```

`dataset/entities/guild_master.toml`:
```toml
name = "Guild Master"
kind = "npc"

[stats]
hp = 100.0
attack = 10.0
armor = 0.0
speed = 3.0

[[classes.starting]]
name = "Merchant"
level = 1

[abilities]
list = []
```

- [ ] **Step 2: Create hostile entity templates**

`dataset/entities/raider_infantry.toml`:
```toml
name = "Raider Infantry"
kind = "creature"

[stats]
hp = 80.0
attack = 12.0
armor = 5.0
speed = 2.0
```

`dataset/entities/wall_jumper.toml`:
```toml
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
```

`dataset/entities/siege_engine.toml`:
```toml
name = "Siege Engine"
kind = "creature"

[stats]
hp = 200.0
attack = 5.0
armor = 15.0
speed = 1.0

[capabilities]
has_siege = true
siege_damage = 50.0
```

`dataset/entities/climber.toml`:
```toml
name = "Climber"
kind = "creature"

[stats]
hp = 60.0
attack = 10.0
armor = 2.0
speed = 3.0

[capabilities]
can_climb = true
```

`dataset/entities/tunneler.toml`:
```toml
name = "Tunneler"
kind = "creature"

[stats]
hp = 90.0
attack = 8.0
armor = 8.0
speed = 1.5

[capabilities]
can_tunnel = true
```

`dataset/entities/flyer.toml`:
```toml
name = "Flyer"
kind = "creature"

[stats]
hp = 50.0
attack = 12.0
armor = 1.0
speed = 5.0

[capabilities]
can_fly = true
```

`dataset/entities/infiltrator.toml`:
```toml
name = "Infiltrator"
kind = "creature"

[stats]
hp = 60.0
attack = 15.0
armor = 0.0
speed = 4.0
```

- [ ] **Step 3: Commit**

```bash
git add dataset/entities/
git commit -m "feat: add entity template TOML files for NPCs and hostile creatures"
```

---

### Task 5: Wire Registry into WorldState

**Files:**
- Modify: `src/world_sim/state.rs`
- Modify: `src/world_sim/registry.rs` (import Arc)

- [ ] **Step 1: Add registry field to WorldState**

In `src/world_sim/state.rs`, add to imports:

```rust
use std::sync::Arc;
```

Add field to the WorldState struct (after `world_events` field, before `skip_resource_init`):

```rust
    /// Data-driven registry loaded from dataset/ directory.
    /// None if no registry has been loaded (bare WorldState).
    #[serde(skip)]
    pub registry: Option<Arc<super::registry::Registry>>,
```

In `WorldState::new()`, add to the struct initialization:

```rust
        registry: None,
```

- [ ] **Step 2: Add Entity::from_template() constructor**

In `src/world_sim/state.rs`, add after `Entity::new_npc()`:

```rust
    /// Create an entity from a registry template.
    ///
    /// Sets base stats from the template and grants starting ClassSlot entries.
    /// Entity level starts at 0 so the progression system applies class bonuses
    /// on the first cycle.
    pub fn from_template(
        id: u32,
        pos: (f32, f32),
        template: &super::registry::EntityTemplateToml,
        registry: &super::registry::Registry,
    ) -> Self {
        let kind = match template.kind.as_str() {
            "hero" => EntityKind::Npc,
            "npc" => EntityKind::Npc,
            "creature" => EntityKind::Monster,
            _ => EntityKind::Npc,
        };
        let team = match template.kind.as_str() {
            "creature" => WorldTeam::Hostile,
            _ => WorldTeam::Friendly,
        };

        let mut npc_data = NpcData::default();
        if kind == EntityKind::Monster {
            npc_data.creature_type = CreatureType::Territorial;
            npc_data.personality = CreatureType::Territorial.personality();
            npc_data.home_den = Some(pos);
        }

        // Grant starting classes.
        for sc in &template.classes.starting {
            let class_hash = tag(sc.name.as_bytes());
            npc_data.classes.push(ClassSlot {
                class_name_hash: class_hash,
                level: sc.level,
                xp: 0.0,
                display_name: String::new(),
            });
            npc_data.class_tags.push(sc.name.to_lowercase());
        }

        // Apply capabilities if present.
        let enemy_capabilities = template.capabilities.as_ref().map(|c| EnemyCapabilities {
            can_jump: c.can_jump,
            jump_height: c.jump_height,
            can_climb: c.can_climb,
            can_tunnel: c.can_tunnel,
            can_fly: c.can_fly,
            has_siege: c.has_siege,
            siege_damage: c.siege_damage,
        });

        Self {
            id,
            kind,
            team,
            pos,
            grid_id: None,
            local_pos: None,
            alive: true,
            hp: template.stats.hp,
            max_hp: template.stats.hp,
            shield_hp: 0.0,
            armor: template.stats.armor,
            magic_resist: 0.0,
            attack_damage: template.stats.attack,
            attack_range: template.stats.attack_range,
            move_speed: template.stats.speed,
            level: 0, // progression detects class_level_sum > 0 and applies stats
            status_effects: Vec::new(),
            npc: Some(npc_data),
            building: None,
            item: None,
            resource: None,
            inventory: if kind == EntityKind::Npc {
                Some(Inventory::with_capacity(50.0))
            } else {
                None
            },
            move_target: None,
            move_speed_mult: 1.0,
            enemy_capabilities,
        }
    }
```

- [ ] **Step 3: Write test for from_template**

Add to `src/world_sim/registry.rs` tests:

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test registry::tests::entity_from_template -- --nocapture`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/state.rs src/world_sim/registry.rs
git commit -m "feat: add Registry to WorldState and Entity::from_template() constructor"
```

---

### Task 6: Migrate progression.rs to Use Registry

**Files:**
- Modify: `src/world_sim/systems/progression.rs`

Currently `class_bonus()` is a hardcoded match returning stat tuples. Replace it with a registry lookup, falling back to the default `(2.0, 0.3, 0.2, 0.0)` when no registry is loaded or class not found.

- [ ] **Step 1: Change compute_progression signatures to accept optional registry**

Replace the `class_bonus()` function and update `compute_progression_for_settlement()` to accept an optional registry:

In `progression.rs`, replace the `class_bonus` function (lines 25-58) with:

```rust
use crate::world_sim::registry::Registry;

/// Look up per-level stat bonuses for a class from the registry.
/// Falls back to default if no registry or class not found.
/// Returns (hp, attack, armor, speed) per level — includes base gains.
fn class_per_level(registry: Option<&Registry>, class_hash: u32) -> (f32, f32, f32, f32) {
    if let Some(reg) = registry {
        for def in reg.classes.values() {
            if crate::world_sim::state::tag(def.name.as_bytes()) == class_hash {
                return (
                    def.per_level.hp,
                    def.per_level.attack,
                    def.per_level.armor,
                    def.per_level.speed,
                );
            }
        }
    }
    // Fallback: BASE + default class_bonus.
    (BASE_MAX_HP + 2.0, BASE_ATTACK + 0.3, BASE_ARMOR + 0.2, BASE_SPEED + 0.0)
}
```

- [ ] **Step 2: Update compute_progression to pass registry**

Update `compute_progression()`:

```rust
pub fn compute_progression(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % PROGRESSION_INTERVAL != 0 { return; }

    let registry = state.registry.as_deref();

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_progression_for_settlement(state, settlement.id, &state.entities[range], registry, out);
    }
}
```

Update `compute_progression_for_settlement()` signature:

```rust
pub fn compute_progression_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    registry: Option<&Registry>,
    out: &mut Vec<WorldDelta>,
) {
```

- [ ] **Step 3: Replace class_bonus() calls with class_per_level()**

In `compute_progression_for_settlement()`, replace line 101:

Old:
```rust
            let (hp, atk, arm, spd) = class_bonus(class.class_name_hash);
```

New:
```rust
            let per_level = class_per_level(registry, class.class_name_hash);
```

And replace the stat computation (the old code adds BASE + class_bonus; the new per_level already includes BASE):

Old (lines 93-113):
```rust
        let (mut hp_bonus, mut atk_bonus, mut armor_bonus, mut speed_bonus) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if npc.classes.is_empty() {
            // No class — base gains only
        } else {
            let total_weight: f32 = npc.classes.iter().map(|c| c.level as f32).sum();
            for class in &npc.classes {
                let (hp, atk, arm, spd) = class_bonus(class.class_name_hash);
                let w = class.level as f32 / total_weight.max(1.0);
                hp_bonus += hp * w;
                atk_bonus += atk * w;
                armor_bonus += arm * w;
                speed_bonus += spd * w;
            }
        }

        let total_hp = (BASE_MAX_HP + hp_bonus) * new_levels as f32;
        let total_atk = (BASE_ATTACK + atk_bonus) * new_levels as f32;
        let total_armor = (BASE_ARMOR + armor_bonus) * new_levels as f32;
        let total_speed = (BASE_SPEED + speed_bonus) * new_levels as f32;
```

New:
```rust
        let (mut hp_per_level, mut atk_per_level, mut armor_per_level, mut speed_per_level) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if npc.classes.is_empty() {
            // No class — base gains only (shouldn't happen since class_level_sum > 0).
            hp_per_level = BASE_MAX_HP;
            atk_per_level = BASE_ATTACK;
            armor_per_level = BASE_ARMOR;
            speed_per_level = BASE_SPEED;
        } else {
            // Weight by class level: higher-level classes contribute more to stat gains.
            let total_weight: f32 = npc.classes.iter().map(|c| c.level as f32).sum();
            for class in &npc.classes {
                let (hp, atk, arm, spd) = class_per_level(registry, class.class_name_hash);
                let w = class.level as f32 / total_weight.max(1.0);
                hp_per_level += hp * w;
                atk_per_level += atk * w;
                armor_per_level += arm * w;
                speed_per_level += spd * w;
            }
        }

        let total_hp = hp_per_level * new_levels as f32;
        let total_atk = atk_per_level * new_levels as f32;
        let total_armor = armor_per_level * new_levels as f32;
        let total_speed = speed_per_level * new_levels as f32;
```

- [ ] **Step 4: Fix callers of compute_progression_for_settlement**

Search for any direct callers of `compute_progression_for_settlement` outside progression.rs and update their signatures to pass registry. Check `runtime.rs` and test files.

Run: `cargo build 2>&1 | head -40`

Fix any compilation errors from the signature change.

- [ ] **Step 5: Run tests**

Run: `cargo test progression -- --nocapture`
Expected: All existing progression tests pass (fallback path is exercised when no registry is loaded)

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/systems/progression.rs
git commit -m "feat: migrate progression.rs to use registry for class per-level stats with fallback"
```

---

### Task 7: Migrate class_gen.rs to Use Registry

**Files:**
- Modify: `src/world_sim/class_gen.rs`
- Modify: `src/world_sim/runtime.rs`

Replace the hardcoded `TEMPLATES` array with a `RegistryClassGenerator` that reads class definitions from the registry.

- [ ] **Step 1: Add RegistryClassGenerator**

Add after `DefaultClassGenerator` in `class_gen.rs`:

```rust
use std::sync::Arc;
use crate::world_sim::registry::{Registry, ClassDefToml};

/// Class generator that reads templates from the data-driven registry.
pub struct RegistryClassGenerator {
    registry: Arc<Registry>,
}

impl RegistryClassGenerator {
    pub fn new(registry: Arc<Registry>) -> Self {
        Self { registry }
    }
}

impl ClassGenerator for RegistryClassGenerator {
    fn match_classes(&self, behavior_profile: &[(u32, f32)]) -> Vec<ClassMatch> {
        let mut matches = Vec::new();

        for def in self.registry.classes.values() {
            // Check all behavior requirements are met.
            let mut qualified = true;
            for (tag_name, &min_val) in &def.requirements.behavior {
                let tag_hash = tag(tag_name.as_bytes());
                if lookup_tag(behavior_profile, tag_hash) < min_val {
                    qualified = false;
                    break;
                }
            }
            if !qualified {
                continue;
            }

            // Compute weighted dot product using score_weights.
            let mut score = 0.0f32;
            let mut best_tag_hash = 0u32;
            let mut best_weighted = 0.0f32;
            let mut second_tag_hash = 0u32;
            let mut second_weighted = 0.0f32;

            if def.score_weights.is_empty() {
                // No score_weights: use tags with equal weights.
                let w = if def.tags.is_empty() { 1.0 } else { 1.0 / def.tags.len() as f32 };
                for tag_name in &def.tags {
                    let tag_hash = tag(tag_name.as_bytes());
                    let val = lookup_tag(behavior_profile, tag_hash);
                    let weighted = val * w;
                    score += weighted;
                    if weighted > best_weighted {
                        second_tag_hash = best_tag_hash;
                        second_weighted = best_weighted;
                        best_tag_hash = tag_hash;
                        best_weighted = weighted;
                    } else if weighted > second_weighted {
                        second_tag_hash = tag_hash;
                        second_weighted = weighted;
                    }
                }
            } else {
                for (tag_name, &weight) in &def.score_weights {
                    let tag_hash = tag(tag_name.as_bytes());
                    let val = lookup_tag(behavior_profile, tag_hash);
                    let weighted = val * weight;
                    score += weighted;
                    if weighted > best_weighted {
                        second_tag_hash = best_tag_hash;
                        second_weighted = best_weighted;
                        best_tag_hash = tag_hash;
                        best_weighted = weighted;
                    } else if weighted > second_weighted {
                        second_tag_hash = tag_hash;
                        second_weighted = weighted;
                    }
                }
            }

            // Normalize with sigmoid: raw/(raw+100).
            let normalized_score = score / (score + 100.0);
            if normalized_score < SCORE_THRESHOLD {
                continue;
            }

            // Variant naming.
            let display_name = if best_weighted > 0.0
                && second_weighted > best_weighted * 0.8
                && second_tag_hash != 0
            {
                if let Some(suffix) = tag_display_name(second_tag_hash) {
                    format!("{} of {}", def.name, suffix)
                } else {
                    def.name.clone()
                }
            } else {
                def.name.clone()
            };

            matches.push(ClassMatch {
                class_name_hash: tag(def.name.as_bytes()),
                display_name,
                score: normalized_score,
            });
        }

        matches
    }

    fn generate_unique_class(
        &self,
        _behavior_profile: &[(u32, f32)],
        _seed: u64,
    ) -> Option<ClassDef> {
        None
    }
}
```

- [ ] **Step 2: Update runtime.rs to use RegistryClassGenerator when registry is available**

In `src/world_sim/runtime.rs`, find where `WorldSim` is constructed (around line 1335):

```rust
class_gen: Box::new(super::class_gen::DefaultClassGenerator::new()),
```

Change to:

```rust
class_gen: if let Some(ref reg) = state.registry {
    Box::new(super::class_gen::RegistryClassGenerator::new(Arc::clone(reg)))
} else {
    Box::new(super::class_gen::DefaultClassGenerator::new())
},
```

Add `use std::sync::Arc;` to runtime.rs imports if not already present.

- [ ] **Step 3: Run tests**

Run: `cargo test class_gen -- --nocapture`
Expected: All 4 existing class_gen tests pass (they test DefaultClassGenerator which is unchanged)

Run: `cargo test -- --nocapture`
Expected: Full test suite passes

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/class_gen.rs src/world_sim/runtime.rs
git commit -m "feat: add RegistryClassGenerator that reads class templates from registry"
```

---

### Task 8: Migrate mass_gen.rs to Use Entity Templates

**Files:**
- Modify: `src/world_sim/building_ai/mass_gen.rs`

Replace hardcoded entity construction in `generate_npc_roster()` with `Entity::from_template()` calls. Keep fallback to current behavior when no registry is loaded.

- [ ] **Step 1: Update generate_npc_roster to use registry**

Replace the `generate_npc_roster` function body (around lines 589-729). The function now looks up entity templates from the registry when available, falling back to the current hardcoded approach.

```rust
fn generate_npc_roster(
    composition: NpcComposition,
    state: &mut WorldState,
    rng: &mut SimpleRng,
) {
    let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
    let settlement_pos = state
        .settlements
        .first()
        .map(|s| s.pos)
        .unwrap_or((0.0, 0.0));

    let (total, combat_frac, hv_count, level_range) = match composition {
        NpcComposition::MilitaryHeavy => (
            rng.range_u32(20, 40) as usize, 0.6,
            rng.range_u32(1, 2) as usize, (2, 5),
        ),
        NpcComposition::CivilianHeavy => (
            rng.range_u32(25, 50) as usize, 0.1, 1, (1, 3),
        ),
        NpcComposition::Balanced => (
            rng.range_u32(20, 35) as usize, 0.3,
            rng.range_u32(1, 2) as usize, (1, 4),
        ),
        NpcComposition::EliteFew => (
            rng.range_u32(8, 15) as usize, 0.4, 1, (4, 8),
        ),
        NpcComposition::LargeLowLevel => (
            rng.range_u32(40, 70) as usize, 0.2,
            rng.range_u32(0, 1) as usize, (1, 2),
        ),
        NpcComposition::Specialist => (
            rng.range_u32(15, 30) as usize, 0.2,
            rng.range_u32(3, 4) as usize, (2, 5),
        ),
    };

    let combat_count = (total as f32 * combat_frac) as usize;
    let worker_count = total - combat_count;

    let registry = state.registry.clone();

    // Combat NPC template names (cycled).
    let combat_templates = ["Town Guard", "Archer", "Defender"];
    for i in 0..combat_count {
        let eid = state.next_entity_id();
        let class_level = rng.range_u32(level_range.0, level_range.1) as u16;
        let tmpl_name = combat_templates[i % combat_templates.len()];

        let mut entity = if let Some(ref reg) = registry {
            if let Some(tmpl) = reg.entity_by_name(tmpl_name) {
                let mut e = Entity::from_template(eid, settlement_pos, tmpl, reg);
                // Override class level from composition params.
                if let Some(npc) = e.npc.as_mut() {
                    for cs in &mut npc.classes {
                        cs.level = class_level;
                    }
                }
                e
            } else {
                make_combat_npc_fallback(eid, settlement_pos, i, class_level)
            }
        } else {
            make_combat_npc_fallback(eid, settlement_pos, i, class_level)
        };

        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
            npc.archetype = "garrison".into();
        }
        state.entities.push(entity);
    }

    // Worker NPC template names (cycled).
    let worker_templates = ["Farmer", "Builder", "Miner", "Craftsman", "Merchant NPC"];
    for i in 0..worker_count {
        let eid = state.next_entity_id();
        let class_level = rng.range_u32(1, level_range.1.min(3)) as u16;
        let tmpl_name = worker_templates[i % worker_templates.len()];

        let mut entity = if let Some(ref reg) = registry {
            if let Some(tmpl) = reg.entity_by_name(tmpl_name) {
                let mut e = Entity::from_template(eid, settlement_pos, tmpl, reg);
                if let Some(npc) = e.npc.as_mut() {
                    for cs in &mut npc.classes {
                        cs.level = class_level;
                    }
                }
                e
            } else {
                make_worker_npc_fallback(eid, settlement_pos, i, class_level)
            }
        } else {
            make_worker_npc_fallback(eid, settlement_pos, i, class_level)
        };

        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
        }
        state.entities.push(entity);
    }

    // High-value NPC template names (cycled).
    let hv_templates = ["Leader", "Master Smith", "Archmage", "High Priest", "Leader", "Guild Master"];
    for i in 0..hv_count {
        let eid = state.next_entity_id();
        let class_level = rng.range_u32(level_range.1, level_range.1 + 3) as u16;
        let tmpl_name = hv_templates[i % hv_templates.len()];

        let mut entity = if let Some(ref reg) = registry {
            if let Some(tmpl) = reg.entity_by_name(tmpl_name) {
                let mut e = Entity::from_template(eid, settlement_pos, tmpl, reg);
                if let Some(npc) = e.npc.as_mut() {
                    for cs in &mut npc.classes {
                        cs.level = class_level;
                    }
                }
                e
            } else {
                make_hv_npc_fallback(eid, settlement_pos, tmpl_name)
            }
        } else {
            make_hv_npc_fallback(eid, settlement_pos, tmpl_name)
        };

        entity.level = 0;
        if let Some(npc) = entity.npc.as_mut() {
            npc.home_settlement_id = Some(settlement_id);
        }
        state.entities.push(entity);
    }

    if let Some(s) = state.settlements.first_mut() {
        s.population = (combat_count + worker_count + hv_count) as u32;
    }
}

// Fallback constructors when no registry is loaded (preserves existing behavior).
fn make_combat_npc_fallback(eid: u32, pos: (f32, f32), idx: usize, class_level: u16) -> Entity {
    let combat_classes: &[(&str, &[u8])] = &[
        ("warrior", b"Warrior"),
        ("archer", b"Ranger"),
        ("defender", b"Guardian"),
    ];
    let (class_tag, class_key) = combat_classes[idx % combat_classes.len()];
    let mut entity = Entity::new_npc(eid, pos);
    entity.level = 0;
    if let Some(npc) = entity.npc.as_mut() {
        npc.class_tags = vec![class_tag.to_string()];
        npc.classes.push(ClassSlot {
            class_name_hash: tag(class_key),
            level: class_level,
            xp: 0.0,
            display_name: String::new(),
        });
    }
    entity
}

fn make_worker_npc_fallback(eid: u32, pos: (f32, f32), idx: usize, class_level: u16) -> Entity {
    let worker_classes: &[(&str, &[u8])] = &[
        ("farmer", b"Farmer"),
        ("builder", b"Artisan"),
        ("miner", b"Miner"),
        ("craftsman", b"Artisan"),
        ("merchant", b"Merchant"),
    ];
    let (class_tag, class_key) = worker_classes[idx % worker_classes.len()];
    let mut entity = Entity::new_npc(eid, pos);
    entity.level = 0;
    if let Some(npc) = entity.npc.as_mut() {
        npc.class_tags = vec![class_tag.to_string()];
        npc.classes.push(ClassSlot {
            class_name_hash: tag(class_key),
            level: class_level,
            xp: 0.0,
            display_name: String::new(),
        });
    }
    entity
}

fn make_hv_npc_fallback(eid: u32, pos: (f32, f32), role: &str) -> Entity {
    let hv_class_map: &[(&str, &[u8])] = &[
        ("Leader", b"Commander"),
        ("Master Smith", b"Artisan"),
        ("Archmage", b"Scholar"),
        ("High Priest", b"Healer"),
        ("Guild Master", b"Merchant"),
    ];
    let class_key = hv_class_map.iter()
        .find(|(name, _)| *name == role)
        .map(|(_, key)| *key)
        .unwrap_or(b"Commander");
    let mut entity = Entity::new_npc(eid, pos);
    entity.level = 0;
    if let Some(npc) = entity.npc.as_mut() {
        npc.archetype = role.to_lowercase();
        npc.classes.push(ClassSlot {
            class_name_hash: tag(class_key),
            level: 1,
            xp: 0.0,
            display_name: String::new(),
        });
    }
    entity
}
```

- [ ] **Step 2: Build and fix any compilation errors**

Run: `cargo build 2>&1 | head -40`

- [ ] **Step 3: Run tests**

Run: `cargo test mass_gen -- --nocapture`
Run: `cargo test -- --nocapture`
Expected: All tests pass (fallback paths exercised since tests don't load registry)

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/building_ai/mass_gen.rs
git commit -m "feat: migrate mass_gen NPC roster generation to use entity templates from registry"
```

---

### Task 9: Write Example Terrain and Scenario Files

**Files:**
- Create: `dataset/environments/terrains/river_valley.toml`
- Create: `dataset/environments/terrains/mountain_pass.toml`
- Create: `dataset/environments/terrains/coastal_plains.toml`
- Create: `dataset/environments/scenarios/frontier_outpost.toml`
- Create: `dataset/environments/scenarios/mining_colony.toml`
- Create: `dataset/environments/scenarios/post_raid_recovery.toml`

- [ ] **Step 1: Create terrain templates**

`dataset/environments/terrains/river_valley.toml`:
```toml
name = "River Valley"
size = [64, 64]
biome = "temperate"

[[resources]]
type = "wood"
amount = 300.0
pos = [20, 15]

[[resources]]
type = "stone"
amount = 150.0
pos = [40, 30]

[[resources]]
type = "food"
amount = 200.0
pos = [32, 45]

[[features]]
type = "river"
from = [0, 32]
to = [64, 32]
```

`dataset/environments/terrains/mountain_pass.toml`:
```toml
name = "Mountain Pass"
size = [64, 64]
biome = "alpine"

[[resources]]
type = "stone"
amount = 500.0
pos = [30, 30]

[[resources]]
type = "iron"
amount = 300.0
pos = [40, 20]

[[features]]
type = "cliff"
from = [0, 10]
to = [64, 10]

[[features]]
type = "cliff"
from = [0, 54]
to = [64, 54]
```

`dataset/environments/terrains/coastal_plains.toml`:
```toml
name = "Coastal Plains"
size = [64, 64]
biome = "coastal"

[[resources]]
type = "food"
amount = 400.0
pos = [32, 40]

[[resources]]
type = "wood"
amount = 200.0
pos = [15, 25]

[[features]]
type = "coastline"
from = [0, 0]
to = [64, 0]
```

- [ ] **Step 2: Create scenario definitions**

`dataset/environments/scenarios/frontier_outpost.toml`:
```toml
name = "Frontier Outpost"
terrain = "River Valley"

[[npcs]]
template = "Town Guard"
count = 5
level_range = [2, 4]

[[npcs]]
template = "Farmer"
count = 8
level_range = [1, 2]

[[npcs]]
template = "Builder"
count = 3
level_range = [1, 2]

[[buildings]]
type = "Barracks"
pos = [30, 30]

[[buildings]]
type = "Farm"
pos = [25, 35]

[[threats]]
name = "Raider Scouts"
approach_direction = [0.0, -1.0]
trigger_tick = 500
duration_ticks = 300

[threats.resolution]
check = "hostiles_dead"

[[threats.entities]]
template = "Raider Infantry"
count = 6
level_range = [2, 4]

[completion]
mode = "all"
max_ticks = 2000
failure = "settlement_wiped"
```

`dataset/environments/scenarios/mining_colony.toml`:
```toml
name = "Mining Colony"
terrain = "Mountain Pass"

[[npcs]]
template = "Miner"
count = 12
level_range = [1, 3]

[[npcs]]
template = "Town Guard"
count = 3
level_range = [2, 4]

[[npcs]]
template = "Craftsman"
count = 4
level_range = [1, 2]

[[buildings]]
type = "Mine"
pos = [35, 25]

[[events]]
type = "earthquake"
severity = 0.5
trigger_tick = 300
duration_ticks = 50

[events.resolution]
check = "buildings_stable"
hp_threshold = 0.5

[completion]
mode = "all"
max_ticks = 1500
```

`dataset/environments/scenarios/post_raid_recovery.toml`:
```toml
name = "Post-Raid Recovery"
terrain = "River Valley"

[[npcs]]
template = "Town Guard"
count = 3
level_range = [2, 4]
state = { morale = 15.0, stress = 80.0, injury = 30.0 }

[[npcs]]
template = "Farmer"
count = 12
level_range = [1, 2]
state = { morale = 20.0, fatigue = 60.0 }

[[buildings]]
type = "Barracks"
pos = [30, 30]

[[history]]
event = "raid"
ticks_ago = 100
severity = 0.8
summary = "Large infantry raid from the north breached defenses and killed the commander"

[[history.consequences]]
type = "casualties"
template = "Town Guard"
count = 4

[[history.consequences]]
type = "building_destroyed"
building = "Watchtower"

[[history.consequences]]
type = "morale_impact"
amount = -25.0

[initial_state]
morale = 20.0
treasury = 50.0

[initial_state.stockpile]
food = 40.0
wood = 15.0
iron = 5.0

[[threats]]
name = "Follow-up Raid"
approach_direction = [0.0, -1.0]
trigger_tick = 800
duration_ticks = 200

[threats.resolution]
check = "hostiles_dead"

[[threats.entities]]
template = "Raider Infantry"
count = 10
level_range = [3, 5]

[[threats.entities]]
template = "Wall Jumper"
count = 3
level_range = [3, 5]

[[events]]
type = "winter_deadline"
trigger_tick = 0
deadline_ticks = 600

[events.resolution]
check = "deadline"
criteria = "stockpile_above"
threshold = 100.0

[completion]
mode = "all"
max_ticks = 2000
failure = "settlement_wiped"
```

- [ ] **Step 3: Write test that loads the real scenario files**

Add to `src/world_sim/registry.rs` tests:

```rust
    #[test]
    fn load_real_dataset() {
        let dataset_path = std::path::Path::new("dataset");
        if !dataset_path.join("classes").is_dir() {
            return; // skip if dataset not present
        }
        let registry = Registry::load(dataset_path)
            .unwrap_or_else(|errs| {
                for e in &errs {
                    eprintln!("  {e}");
                }
                panic!("{} registry errors", errs.len());
            });
        assert!(registry.classes.len() >= 31, "Expected 31+ classes, got {}", registry.classes.len());
        assert!(registry.entities.len() >= 10, "Expected 10+ entities, got {}", registry.entities.len());
        assert!(registry.terrains.len() >= 3, "Expected 3+ terrains, got {}", registry.terrains.len());
        assert!(registry.scenarios.len() >= 3, "Expected 3+ scenarios, got {}", registry.scenarios.len());
    }
```

- [ ] **Step 4: Run test**

Run: `cargo test registry::tests::load_real_dataset -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dataset/environments/
git commit -m "feat: add terrain templates and prebuilt scenario definitions with history and resolution"
```

---

### Task 10: Integration Test — Full Registry Lifecycle

**Files:**
- Modify: `src/world_sim/registry.rs` (add integration test)

- [ ] **Step 1: Write integration test loading registry and creating entities**

Add to `src/world_sim/registry.rs` tests:

```rust
    #[test]
    fn full_lifecycle_registry_to_entity() {
        use crate::world_sim::state::{Entity, WorldState};

        let dataset_path = std::path::Path::new("dataset");
        if !dataset_path.join("classes").is_dir() {
            return;
        }

        let registry = Registry::load(dataset_path).unwrap();
        let registry = Arc::new(registry);

        // Create a WorldState with the registry.
        let mut state = WorldState::new(42);
        state.registry = Some(Arc::clone(&registry));

        // Create an entity from template.
        let guard_tmpl = registry.entity_by_name("Town Guard").expect("Town Guard template missing");
        let guard = Entity::from_template(1, (10.0, 20.0), guard_tmpl, &registry);
        assert_eq!(guard.hp, 100.0);
        assert_eq!(guard.level, 0);
        let npc = guard.npc.as_ref().unwrap();
        assert!(!npc.classes.is_empty(), "Should have starting class");

        // Create a creature from template.
        let jumper_tmpl = registry.entity_by_name("Wall Jumper").expect("Wall Jumper template missing");
        let jumper = Entity::from_template(2, (5.0, 5.0), jumper_tmpl, &registry);
        assert_eq!(jumper.kind, crate::world_sim::state::EntityKind::Monster);
        let caps = jumper.enemy_capabilities.as_ref().unwrap();
        assert!(caps.can_jump);

        // Verify class definition is accessible via registry.
        let warrior_class = registry.class_by_name("Warrior").expect("Warrior class missing");
        assert!(warrior_class.per_level.hp > 0.0);
        assert!(warrior_class.per_level.attack > 0.0);

        // Verify scenario is loadable.
        let scenario = registry.scenario_by_name("Post-Raid Recovery").expect("scenario missing");
        assert!(!scenario.history.is_empty());
        assert!(!scenario.threats.is_empty());
    }
```

- [ ] **Step 2: Run test**

Run: `cargo test registry::tests::full_lifecycle -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/registry.rs
git commit -m "test: add full lifecycle integration test for registry → entity creation"
```

---

## Self-Review

**1. Spec coverage:**
- [x] Registry struct with classes, entities — Task 1, 2
- [x] Class TOML format with base_stats, per_level, requirements, abilities — Task 1, 3
- [x] Entity template format with stats, starting classes, abilities, capabilities — Task 1, 4
- [x] Registry::load() with filesystem scanning — Task 2
- [x] Cross-reference validation — Task 2
- [x] Wire Registry into WorldState — Task 5
- [x] Replace class_bonus() in progression.rs — Task 6
- [x] Replace TEMPLATES in class_gen.rs — Task 7
- [x] Entity::from_template() — Task 5
- [x] Migrate mass_gen.rs — Task 8
- [x] Terrain templates — Task 1, 9
- [x] Scenario definitions with history, threats, events, resolution — Task 1, 9
- [x] Composable resolution checks (all/any) — Task 1 (types defined)
- [x] History consequence types — Task 1 (HistoryConsequence enum)
- [x] Prebuilt scenarios — Task 9
- [x] score_weights field for class matching — Task 1, 7

**Deferred from this plan (explicitly marked as future in spec):**
- LLM scenario generation from natural language
- hero_templates.rs full absorption into registry (combat sim integration) — complex, involves tactical_sim crate changes
- LoL heroes migration to dataset/entities/lol/ — mechanical file move, can be done separately
- Deleting assets/ directory — already empty per exploration

**2. Placeholder scan:** No TBD/TODO/placeholders found.

**3. Type consistency:** Verified — `ClassDefToml`, `EntityTemplateToml`, `Registry`, `StatBlock`, `RegistryClassGenerator`, `class_per_level()` used consistently across all tasks.
