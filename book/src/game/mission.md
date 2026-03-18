# Mission System

Missions are multi-room dungeon runs that bridge the campaign and combat layers.
The player's squad enters a procedurally generated dungeon, progresses through
rooms, and fights enemies while pursuing objectives.

## Module: `src/mission/`

```
mission/
├── execution/        # Mission state machine
├── room_gen/         # Room generation
├── room_sequence/    # Room ordering
├── sim_bridge/       # Combat simulation integration
├── vfx/              # Visual effects
├── enemy_templates/  # Enemy definitions
├── objectives.rs     # Mission objective types
├── hero_templates.rs # Hero template loading
├── unit_vis.rs       # Unit visualization
└── tag_color.rs      # Effect tag coloring
```

## Mission Execution

The mission state machine (`execution/`) manages the lifecycle:

```
MissionStart
    │
    ▼
RoomEntry ──▶ Combat ──▶ RoomComplete
    │            │              │
    │     (squad wipe)         ▼
    │            │         NextRoom?
    │            ▼         ├── Yes → RoomEntry
    │       MissionFail    └── No  → MissionComplete
    │                               │
    ▼                               ▼
 MissionOutcome ◀───────────────────┘
```

## Room Sequence

The `room_sequence/` module determines room order and composition:
- **Entry room** — initial encounter (lighter)
- **Standard rooms** — main combat encounters
- **Elite rooms** — tougher encounters with better rewards
- **Boss room** — final encounter with unique mechanics

Room difficulty scales with mission level and campaign progression.

## Sim Bridge

The `sim_bridge/` module converts between mission-level state and combat
simulation state:

- Spawns hero `UnitState` from the party's `HeroState`
- Spawns enemy `UnitState` from enemy templates
- Maps room layouts to navigation grids
- Feeds combat outcomes back to mission state
