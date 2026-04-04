# Data-Driven Registry Design

## Goal

Replace hardcoded class definitions, entity construction, and scenario generation with a unified data-driven registry system. All game entity data lives in `dataset/` as TOML and ability DSL files, loaded at runtime. Delete `assets/` entirely.

## Architecture

A single `Registry` struct holds three collections, loaded from filesystem at startup:

```
dataset/
  classes/          # Class definitions (TOML)
  abilities/        # Ability DSL files (.ability)
  entities/         # Entity templates (TOML) — heroes, NPCs, creatures
  environments/
    terrains/       # Terrain/biome templates (TOML)
    scenarios/      # Complete scenario definitions (TOML)
```

The registry is immutable after load. Passed as `&Registry` to systems that need it. Stored as a field on `WorldState`.

### Registry Struct

```rust
pub struct Registry {
    classes: HashMap<u32, ClassDef>,        // keyed by tag(name)
    abilities: HashMap<u32, AbilityDef>,    // keyed by tag(name)
    entities: HashMap<u32, EntityTemplate>, // keyed by tag(name)
}
```

### Loading

`Registry::load(path: &Path)` scans subdirectories, parses each file, and validates cross-references. Returns `Result<Registry, Vec<RegistryError>>` — collects all validation errors rather than failing on the first.

Validation on load:
- Class ability pools reference abilities that exist in the ability registry
- Entity starting classes reference classes that exist in the class registry
- Entity ability lists reference abilities that exist in the ability registry
- No duplicate names within a category
- Required fields present

---

## Class Definitions

**Location:** `dataset/classes/<name>.toml`

Replaces hardcoded `class_bonus()` in `progression.rs` and `ClassTemplate` array in `class_gen.rs`.

### Format

```toml
name = "Warrior"
tags = ["melee", "combat", "martial"]

[base_stats]
hp = 120
attack = 15
armor = 5
speed = 3.0

[per_level]
hp = 10.0
attack = 2.5
armor = 1.7
speed = 0.02

[requirements]
min_level = 0
behavior = { melee = 30.0, combat = 20.0 }

[abilities]
pool = ["ShieldBash", "Charge", "Cleave"]
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Display name, also used for `tag(name)` hash |
| `tags` | string[] | yes | Behavioral/categorical tags |
| `base_stats.hp` | f32 | yes | One-time HP bonus granted when class is first acquired (level 1) |
| `base_stats.attack` | f32 | yes | One-time attack bonus on acquisition |
| `base_stats.armor` | f32 | yes | One-time armor bonus on acquisition |
| `base_stats.speed` | f32 | yes | One-time speed bonus on acquisition |
| `per_level.hp` | f32 | yes | HP gained per class level (applied on each level-up, not on acquisition) |
| `per_level.attack` | f32 | yes | Attack gained per class level-up |
| `per_level.armor` | f32 | yes | Armor gained per class level-up |
| `per_level.speed` | f32 | yes | Speed gained per class level-up |
| `requirements.min_level` | u32 | no | Minimum total entity level to qualify (default 0) |
| `requirements.behavior` | map | no | Behavior tag minimums to qualify |
| `abilities.pool` | string[] | no | Abilities available to this class (registry references) |

### Migration from Hardcoded Classes

The 29 entries in `progression.rs:class_bonus()` and 33 templates in `class_gen.rs` are merged into individual TOML files. The `per_level` field replaces `BASE + class_bonus` — it is the total gain per level (base stats baked in). The `requirements.behavior` field replaces `ClassTemplate.requirements`, and class_gen's `score_tags` become `tags`.

---

## Ability Registry

**Location:** `dataset/abilities/<name>.ability`

The existing `.ability` DSL files, moved from `assets/hero_templates/` into the central registry. The DSL parser is unchanged. Abilities are referenced by name from class definitions and entity templates.

---

## Entity Templates

**Location:** `dataset/entities/<name>.toml`

Replaces both `dataset/hero_templates/*.toml` (combat heroes) and `Entity::new_npc()` (world sim NPCs). One unified format with optional sections.

### Format

```toml
name = "Town Guard"
kind = "npc"                    # npc, hero, creature

[stats]
hp = 100
attack = 10
armor = 0
speed = 3.0
attack_range = 1.5             # optional, default 1.5

[classes]
starting = [{ name = "Warrior", level = 2 }]

[abilities]
list = ["BasicAttack"]

# Optional: combat-specific fields (heroes)
[attack]
damage = 15
range = 1.5
cooldown = 1000
cast_time = 300

# Optional: enemy capabilities (creatures/raiders)
[capabilities]
can_jump = false
jump_height = 0
can_climb = false
can_tunnel = false
can_fly = false
has_siege = false
siege_damage = 0.0
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Display name |
| `kind` | string | yes | `npc`, `hero`, or `creature` |
| `stats.hp` | f32 | yes | Base max HP |
| `stats.attack` | f32 | yes | Base attack damage |
| `stats.armor` | f32 | yes | Base armor |
| `stats.speed` | f32 | yes | Base movement speed |
| `stats.attack_range` | f32 | no | Attack range (default 1.5) |
| `classes.starting` | array | no | Classes granted at spawn, each with `name` and `level` |
| `abilities.list` | string[] | no | Abilities granted at spawn (registry references) |
| `attack` | table | no | Detailed attack config (combat heroes only) |
| `capabilities` | table | no | Movement/siege capabilities (creatures/enemies) |

### Entity Construction

`Entity::new_npc()` remains as a bare-minimum constructor for edge cases. The preferred path is:

```rust
Entity::from_template(template_name: &str, registry: &Registry) -> Entity
```

This looks up the template, sets base stats, grants ClassSlot entries for starting classes, and attaches abilities.

### Migration

- 27 hero templates from `dataset/hero_templates/` move to `dataset/entities/` with the expanded format
- 172 LoL hero imports from `assets/lol_heroes/` move to `dataset/entities/lol/`
- `hero_templates.rs` template loading is absorbed into the registry
- `mass_gen.rs:generate_npc_roster()` calls `Entity::from_template()` instead of hardcoding stats and class slots

---

## Terrain Templates

**Location:** `dataset/environments/terrains/<name>.toml`

Define physical terrain, resource deposits, and geographic features.

### Format

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

[[features]]
type = "river"
from = [0, 32]
to = [64, 32]
```

---

## Scenario Definitions

**Location:** `dataset/environments/scenarios/<name>.toml`

Complete scenario definitions composing terrain, population, buildings, threats, events, history, and resolution conditions.

### Format

```toml
name = "Frontier Outpost Under Siege"
terrain = "river_valley"

# --- Friendly Population ---

[[npcs]]
template = "town_guard"
count = 5
level_range = [2, 4]
state = { morale = 50.0 }          # optional NPC state overrides

[[npcs]]
template = "farmer"
count = 8
level_range = [1, 2]

# --- Starting Buildings ---

[[buildings]]
type = "Barracks"
pos = [30, 30]

[[buildings]]
type = "Farm"
pos = [25, 35]

# --- Recent History ---
# Events applied oldest-first to build up initial state.
# Consequences modify entities, buildings, stockpile, morale.
# Summaries seed the chronicle for oracle memory.

[[history]]
event = "raid"
ticks_ago = 100
severity = 0.7
summary = "Large infantry raid from the north breached outer wall"

[[history.consequences]]
type = "casualties"
template = "town_guard"
count = 4

[[history.consequences]]
type = "building_destroyed"
building = "Watchtower"

[[history.consequences]]
type = "morale_impact"
amount = -20.0

[[history]]
event = "plague"
ticks_ago = 300
severity = 0.5
summary = "Waterborne illness swept through workers"

[[history.consequences]]
type = "casualties"
template = "farmer"
count = 6

[[history.consequences]]
type = "population_state"
stress = 40.0
fatigue = 30.0

[[history]]
event = "trade_collapse"
ticks_ago = 200
severity = 0.4
summary = "Primary trade route cut off by bandits"

[[history.consequences]]
type = "stockpile_drain"
resource = "iron"
fraction = 0.7

[[history.consequences]]
type = "treasury_drain"
fraction = 0.5

# --- Initial State Overrides ---
# For conditions not explained by history events.

[initial_state]
morale = 25.0
treasury = 50.0

[initial_state.stockpile]
food = 30.0
wood = 10.0
iron = 5.0

# --- Military Threats ---

[[threats]]
name = "Northern Raiders"
approach_direction = [0.0, -1.0]
trigger_tick = 500
duration_ticks = 200

[threats.resolution]
check = "hostiles_dead"

[[threats.entities]]
template = "raider_infantry"
count = 8
level_range = [2, 4]

[[threats.entities]]
template = "wall_jumper"
count = 3
level_range = [3, 5]

# --- Non-Entity Events ---

[[events]]
type = "earthquake"
severity = 0.6
trigger_tick = 300
duration_ticks = 50

[events.resolution]
check = "buildings_stable"
hp_threshold = 0.5

[[events]]
type = "refugee_wave"
severity = 0.5
trigger_tick = 200
template = "refugee"

[events.resolution]
check = "housing_covered"

[[events]]
type = "winter_deadline"
trigger_tick = 0
deadline_ticks = 500

[events.resolution]
check = "deadline"
criteria = "stockpile_above"
threshold = 200.0

# --- Scenario Completion ---

[completion]
mode = "all"                        # all challenges resolved = success
max_ticks = 2000                    # hard timeout = failure
failure = "settlement_wiped"        # all NPCs dead = immediate failure
```

### History Consequence Types

| Type | Fields | Effect |
|------|--------|--------|
| `casualties` | `template`, `count` | Mark NPCs as dead (recent losses) |
| `building_damaged` | `building`, `hp_fraction` | Set building HP to fraction of max |
| `building_destroyed` | `building` | Mark building as destroyed |
| `morale_impact` | `amount` | Adjust settlement-wide morale |
| `population_state` | `stress`, `fatigue`, `injury`, `morale` | Set NPC emotional states |
| `stockpile_drain` | `resource`, `fraction` | Reduce specific resource by fraction |
| `treasury_drain` | `fraction` | Reduce treasury by fraction |

### Resolution Checks

Composable conditions for challenge completion. Atomic checks can be combined with `all`/`any` combinators.

**Combat/Entity:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `hostiles_dead` | — | All hostile entities from this challenge eliminated |
| `npc_alive` | `template` or `archetype` | Specific NPC is alive |
| `npc_secured` | `template` or `archetype` | NPC alive AND assigned to a building |

**Buildings:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `building_exists` | `building_type`, `min_count` | Enough buildings of type exist |
| `buildings_stable` | `hp_threshold` | No buildings below HP threshold |

**Economy:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `stockpile_above` | `resource` (or all), `threshold` | Resource stockpile meets threshold |
| `treasury_above` | `threshold` | Treasury meets threshold |

**Population:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `housing_covered` | — | Housing capacity >= population |
| `capability_reached` | `min_level`, `count` | N NPCs at or above level threshold |

**Spatial:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `settlements_connected` | `count` | N settlements linked by trade routes |
| `building_placed_in_zone` | `building_type`, `zone` | Building placed in specific terrain zone |

**Generic:**
| Check | Fields | Condition |
|-------|--------|-----------|
| `score_above` | `threshold` | Utility score meets threshold |
| `tick_survived` | — | Survive until `trigger_tick + duration_ticks` elapsed |

**Combinators:**
```toml
# All conditions must pass
[resolution]
all = [
  { check = "hostiles_dead" },
  { check = "building_exists", building_type = "Watchtower", min_count = 2 },
]

# Any condition passes
[resolution]
any = [
  { check = "tick_survived" },
  { check = "hostiles_dead" },
]
```

---

## Code Migration

### Deleted Entirely
- `assets/` directory (contents moved to `dataset/`)
- `class_bonus()` in `progression.rs` — stat bonuses come from `ClassDef.per_level`
- Hardcoded `ClassTemplate` array in `class_gen.rs` — templates come from registry
- `hero_templates.rs` template loading — absorbed into registry
- `dataset/hero_templates/` — moved to `dataset/entities/`

### Modified
- **`progression.rs`**: `compute_progression()` takes `&Registry`, looks up `per_level` from `ClassDef` instead of calling `class_bonus()`
- **`class_gen.rs`**: `match_classes()` iterates `registry.classes()` instead of hardcoded vec; requirements and score weights come from `ClassDef`
- **`state.rs`**: Add `registry: Registry` field to `WorldState`; add `Entity::from_template(name, &Registry)` constructor
- **`mass_gen.rs`**: `generate_npc_roster()` calls `Entity::from_template()` instead of manually constructing entities and pushing class slots
- **Combat sim**: `hero_toml_to_unit()` reads from entity templates via registry

### Preserved
- `Entity::new_npc()` stays as a bare-minimum constructor for cases needing a blank entity
- `.ability` DSL parser is unchanged — files just move to `dataset/abilities/`
- `EnemyCapabilities` struct is unchanged — populated from entity template `[capabilities]` section

---

## Future Extension: LLM Scenario Generation

A CLI command for generating scenario TOML from natural language descriptions:

```
xtask scenario generate-from-prompt "frontier town recovering from a devastating raid, commander killed, walls breached, winter approaching"
```

The command:
1. Loads registry (available templates, classes, abilities)
2. Sends prompt + registry manifest (available entity names, class names, building types) to LLM
3. LLM produces scenario TOML
4. Validates against registry — rejects hallucinated entity templates or classes
5. Writes to `dataset/environments/scenarios/`

This depends on the registry and scenario format existing first. Not part of the core implementation.
