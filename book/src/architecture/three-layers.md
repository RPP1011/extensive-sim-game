# Three-Layer Simulation

The game is structured as three nested simulation layers, each operating at a
different timescale and abstraction level.

```
Campaign (turns)
  └─▶ Mission (rooms)
        └─▶ Combat (100ms ticks)
```

## Layer 1: Campaign

**Timescale:** Turn-based (one turn = one strategic action)

The campaign layer manages the overworld — a map of regions controlled by factions.
Each turn, the player chooses actions like moving between regions, initiating
diplomacy, or responding to flashpoints (dynamic crisis events).

**Key types:**
- `CampaignState` — the full state of a campaign in progress
- `HeroState` — a hero's persistent stats across missions
- `OverworldRegion` — a map node with faction ownership and connections

**Module:** `src/game_core/`

The campaign layer invokes the mission layer when the player enters a dungeon or
engages in combat.

## Layer 2: Mission

**Timescale:** Room-by-room progression through a dungeon

A mission is a sequence of procedurally generated rooms with objectives (eliminate
all enemies, protect an NPC, survive for N rounds, etc.). The player's squad moves
through rooms, encountering enemies and environmental hazards.

**Key types:**
- `MissionState` — tracks room sequence, objectives, and squad HP/resources
- `RoomLayout` — procedurally generated floorplan with cover, elevation, obstacles

**Module:** `src/mission/`

Each room's combat is resolved by dropping into the combat layer.

## Layer 3: Combat

**Timescale:** 100ms fixed ticks

This is the heart of the simulation. A deterministic, tick-based combat engine where
units move, attack, cast abilities, and apply effects. The simulation runs until one
side is eliminated or a time limit is reached.

**Key types:**
- `SimState` — the complete state of a combat encounter
- `UnitState` — a single unit's position, HP, cooldowns, abilities, status effects
- `UnitIntent` — what a unit wants to do this tick

**Module:** `src/ai/core/`

The combat layer is where all AI training happens. Its deterministic nature means
training runs are reproducible and debuggable.

## Data Flow Between Layers

```
Campaign                    Mission                     Combat
────────                    ───────                     ──────
HeroState ─────────────▶  squad composition  ──────▶  UnitState[]
faction context ────────▶  enemy templates   ──────▶  UnitState[]
                           room layout       ──────▶  NavGrid
                                                       │
                                              step() loop
                                                       │
                           room outcome  ◀─────────── SimState
campaign consequence ◀──── mission outcome              events log
```

Heroes carry persistent state (XP, items, health) from the campaign into missions.
Mission outcomes propagate back to the campaign as faction reputation changes,
territory control shifts, and narrative progression.
