# Class & Ability System

## Overview

NPCs earn classes by doing work. Work emits behavior tags, tags accumulate on the NPC's profile, and when the profile matches a class template, the class is granted. Classes level up through continued alignment, and abilities are procedurally generated at tier milestones.

```
Work at building → behavior tags → profile accumulates → class match → XP → level up → ability generated
```

## Behavior Tags

Tags are the atomic unit of NPC skill identity. 34 tags across 10 categories:

| Category | Tags |
|----------|------|
| Combat | MELEE, RANGED, COMBAT, DEFENSE, TACTICS |
| Craft | MINING, SMITHING, CRAFTING, ENCHANTMENT, ALCHEMY |
| Social | TRADE, DIPLOMACY, LEADERSHIP, NEGOTIATION, DECEPTION |
| Knowledge | RESEARCH, LORE, MEDICINE, HERBALISM, NAVIGATION |
| Survival | ENDURANCE, RESILIENCE, STEALTH, SURVIVAL, AWARENESS |
| Spiritual | FAITH, RITUAL |
| Labor | LABOR, TEACHING, DISCIPLINE, CONSTRUCTION, MASONRY |
| Resource | FARMING, WOODWORK, EXPLORATION |
| Terrain | SEAFARING, DUNGEONEERING |
| Meta | MASTERY, COMPASSION_TAG, ARCHITECTURE |

### How Tags Are Earned

NPCs working at buildings emit tags every tick via `WorldDelta::AddBehaviorTags`:

| Building Type | Tags Emitted |
|---------------|-------------|
| Farm | FARMING 1.0, LABOR 0.5 |
| Mine | MINING 1.0, LABOR 0.5 |
| Sawmill | WOODWORK 1.0, LABOR 0.5 |
| Forge | SMITHING 1.0, CRAFTING 0.5 |
| Apothecary | ALCHEMY 1.0, MEDICINE 0.5 |
| Temple | FAITH 1.0, RITUAL 0.5 |
| Library | RESEARCH 1.0, LORE 0.5 |
| Barracks | COMBAT 1.0, DISCIPLINE 0.5 |

Tags accumulate on `NpcData.behavior_profile: Vec<(u32, f32)>` — a sorted list of (tag_hash, total_weight) pairs.

## Class Templates

29 hardcoded class templates, each with requirements and scoring weights:

```rust
ClassTemplate {
    name_hash: tag(b"Warrior"),
    display_name: "Warrior",
    requirements: &[(MELEE, 100.0)],           // must have MELEE ≥ 100
    score_tags: &[(MELEE, 0.4), (DEFENSE, 0.3), (ENDURANCE, 0.2), (COMBAT, 0.1)],
}
```

### All 29 Classes

| Class | Primary Requirements | Category |
|-------|---------------------|----------|
| Warrior | MELEE ≥ 100 | Combat |
| Ranger | RANGED ≥ 80, SURVIVAL ≥ 50 | Combat |
| Guardian | DEFENSE ≥ 100, ENDURANCE ≥ 50 | Combat |
| Warden | AWARENESS ≥ 60, COMBAT ≥ 60 | Combat |
| Veteran | COMBAT ≥ 150, ENDURANCE ≥ 80 | Combat |
| Sentinel | DEFENSE ≥ 80, AWARENESS ≥ 80 | Combat |
| Artisan | CRAFTING ≥ 80 | Crafting |
| Miner | MINING ≥ 100 | Crafting |
| Woodsman | WOODWORK ≥ 80 | Crafting |
| Alchemist | ALCHEMY ≥ 80 | Crafting |
| Herbalist | HERBALISM ≥ 60 | Crafting |
| Merchant | TRADE ≥ 80 | Social |
| Diplomat | DIPLOMACY ≥ 80 | Social |
| Bard | DIPLOMACY ≥ 40, LORE ≥ 40 | Social |
| Scholar | RESEARCH ≥ 100 | Knowledge |
| Healer | MEDICINE ≥ 80 | Knowledge |
| Oathkeeper | FAITH ≥ 60, DISCIPLINE ≥ 60 | Knowledge |
| Commander | LEADERSHIP ≥ 100, TACTICS ≥ 50 | Leadership |
| Mentor | TEACHING ≥ 60 | Leadership |
| Builder | CONSTRUCTION ≥ 80 | Construction |
| Architect | ARCHITECTURE ≥ 60, CONSTRUCTION ≥ 60 | Construction |
| Survivor | SURVIVAL ≥ 100, ENDURANCE ≥ 50 | Survival |
| Explorer | EXPLORATION ≥ 80, NAVIGATION ≥ 40 | Survival |
| Betrayer | DECEPTION ≥ 80 | Villainy |
| Sea Captain | SEAFARING ≥ 100, NAVIGATION ≥ 60 | Specialized |
| Mariner | SEAFARING ≥ 60 | Specialized |
| Dungeon Master | DUNGEONEERING ≥ 100 | Specialized |
| Delver | DUNGEONEERING ≥ 60, SURVIVAL ≥ 40 | Specialized |

## Class Matching

Runs every 50 ticks via `run_class_matching()`:

1. Skip NPCs with behavior sum < 10.0
2. For each template: compute `score = weighted_dot_product(profile, score_tags)`
3. Normalize: `normalized = raw_score / (raw_score + 100.0)` (sigmoid)
4. Grant class if `normalized ≥ 0.3` and NPC doesn't already have it
5. Max 5 classes per NPC

### Multi-Classing

NPCs naturally gain multiple classes from diverse work:
- A farmer who also fights gets **Survivor** + **Warrior**
- A miner who studies gets **Miner** + **Scholar**
- A leader who builds gets **Commander** + **Builder**

## Class Progression

### XP Allocation (every 50 ticks)

```
xp_gain = class_match_score × 0.5
```

Higher behavior alignment = faster leveling. A Miner who only mines levels faster than one who splits time.

### Level-Up Cost (exponential)

```
xp_needed = 50.0 × e^(level × 0.1) × global_factor
global_factor = 1.0 + (total_level / 100)²
```

| Level | Base XP | At total_level=50 | At total_level=100 |
|-------|---------|--------------------|--------------------|
| 1 | 55 | 69 | 110 |
| 5 | 82 | 103 | 165 |
| 10 | 135 | 169 | 271 |
| 20 | 369 | 461 | 737 |
| 50 | 7,389 | 9,236 | 14,778 |

### Caps

| Threshold | Effect |
|-----------|--------|
| Total level < 80 | Can acquire new classes |
| Total level 80+ | Consolidates to highest-level class |
| Total level 100 | Hard cap — no more leveling |
| 5+ classes | Prunes lowest, redistributes XP |

## Ability Generation

### Trigger: Tier Milestones

| Class Level | Tier | Max Effects |
|-------------|------|-------------|
| 2-4 | 1 | 1 |
| 5-7 | 2 | 1 |
| 10-12 | 3 | 2 |
| 20-22 | 4 | 2 |
| 35-37 | 5 | 3 |
| 50-52 | 6 | 3 |
| 70-72 | 7 | 4 |

### Generation Pipeline

1. **generate_tiered_ability(archetype, tier, rng, history)** — walks a grammar tree conditioned on archetype and tier to produce a DSL ability text
2. **grammar_space::encode(dsl)** — maps DSL text to a 48-dimensional vector. Every point in [0,1]^48 is a valid ability.
3. **ability_quality::score_ability(vector)** — scores on coherence (0.3), balance (0.15), purpose (0.2), tag consistency (0.15), variety (0.05+)
4. Best candidate kept (if `candidates > 1`, generates multiple and picks highest score)

### Tier Power Scaling

```
TIER_POWER:    [0, 1, 2, 4, 8, 15, 30, 50]
TIER_COOLDOWN: [0, 1, 1.2, 1.5, 2, 2.5, 3, 4]
```

A Tier 7 ability is 50x more powerful than Tier 1 but has 4x longer cooldown.

### Procedural Names

Class display names are generated from behavior profile + class template:
- "Stoneheart Miner" (Miner with high ENDURANCE)
- "Frontier Sentinel" (Sentinel with EXPLORATION)
- "Wild-Born Guardian of Endurance" (Guardian with SURVIVAL + ENDURANCE dominance)

## Files

| File | Purpose |
|------|---------|
| `state.rs` | ClassSlot, behavior_profile, tag constants, ActionTags |
| `class_gen.rs` | ClassGenerator trait, 29 templates, DefaultClassGenerator |
| `ability_gen.rs` | Grammar-tree ability generation, tier scaling |
| `grammar_space.rs` | 48-dim invertible DSL encoding |
| `ability_quality.rs` | Ability scoring heuristics |
| `runtime.rs` | run_class_matching(), XP allocation, level-up, ability trigger |
| `systems/work.rs` | Tag emission from building work |
