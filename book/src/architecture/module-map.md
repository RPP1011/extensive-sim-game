# Module Map

The `src/` directory contains approximately 317 Rust source files organized into
a deep module hierarchy. This chapter provides a guided tour.

## Top-Level Modules

From `src/lib.rs`:

```rust
pub mod ai;              // AI systems, simulation engine, effects
pub mod audio;           // Audio playback
pub mod game_core;       // Campaign, overworld, save/load
pub mod mapgen_voronoi;  // Voronoi-based map generation
pub mod mission;         // Multi-room dungeon missions
pub mod progression;     // Narrative progression tracking
pub mod scenario;        // Scenario definitions and runner
```

Additional modules in `src/main.rs` (binary-only):

```
app_systems      — Bevy plugin registration
camera            — Orbit camera controller
cli_args          — CLI argument definitions
hub_ui_draw/      — Hub UI rendering (egui)
game_loop         — Main game loop state machine
character_select  — Character selection screen
```

## The `ai` Module Tree

The `ai` module is the largest in the project (~200 files). Here is its structure:

```
ai/
├── mod.rs              # re-exports spatial, tactics, coordination
├── core/               # ★ Simulation engine (see next chapter)
│   ├── simulation.rs   #   The step() function
│   ├── types.rs        #   SimState, UnitState, Team
│   ├── events.rs       #   SimEvent logging
│   ├── damage.rs       #   Damage calculation
│   ├── targeting.rs    #   Target selection
│   ├── conditions.rs   #   Ability condition checks
│   ├── triggers.rs     #   Effect trigger system
│   ├── helpers.rs      #   is_alive, can_see, etc.
│   ├── math.rs         #   Vector math, movement
│   ├── intent.rs       #   Intent action types
│   ├── resolve.rs      #   Effect resolution pipeline
│   ├── apply_effect.rs #   Effect application
│   ├── determinism.rs  #   RNG, reproducibility
│   ├── metrics.rs      #   Battle statistics
│   ├── replay.rs       #   Replay utilities
│   ├── verify.rs       #   Post-tick verification
│   ├── ability_eval/   #   Neural ability urgency evaluator
│   ├── ability_transformer/  # Grokking transformer
│   ├── burn_model/     #   Burn ML models (feature-gated)
│   ├── self_play/      #   RL training (REINFORCE, PPO)
│   ├── hero/           #   Hero-specific mechanics
│   └── tests/          #   Determinism, mechanics, ability tests
│
├── effects/            # ★ Data-driven ability system
│   ├── defs.rs         #   AbilityDef, PassiveDef
│   ├── types.rs        #   Effect enums, Tags, Areas
│   ├── effect_enum.rs  #   Enum dispatch
│   ├── manifest.rs     #   Ability registry
│   └── dsl/            #   ★ Winnow-based DSL parser
│       ├── parser.rs   #     Top-level parser
│       ├── ast.rs      #     Abstract syntax tree
│       ├── lower.rs    #     AST → AbilityDef lowering
│       ├── emit.rs     #     AbilityDef → DSL text
│       └── ...         #     (17 files total)
│
├── squad/              # Squad-level AI
│   ├── personality.rs  #   Personality profiles
│   ├── intents.rs      #   Intent generation
│   ├── state.rs        #   SquadAiState, FormationMode
│   └── combat/         #   Ability evaluation, targeting
│
├── goap/               # Goal-Oriented Action Planning
│   ├── planner.rs      #   A* planner
│   ├── goal.rs         #   Goal definitions
│   ├── action.rs       #   Action definitions
│   ├── world_state.rs  #   Symbolic world state
│   ├── dsl.rs          #   GOAP file parser
│   └── party.rs        #   Party culture modifiers
│
├── behavior/           # Behavior tree DSL
│   ├── parser.rs       #   .behavior file parser
│   ├── interpreter.rs  #   Tree execution
│   └── types.rs        #   Node types
│
├── pathing/            # Grid navigation
│   └── navigation.rs   #   A* pathfinding, LOS
│
├── advanced/           # Advanced tactics
│   ├── horde.rs        #   Horde mode AI
│   ├── tactics.rs      #   Tactical reasoning
│   └── spatial.rs      #   Spatial awareness
│
├── roles/              # Role system (tank, dps, support)
├── student/            # Learning from oracle policies
├── personality.rs      # Personality types
├── control.rs          # CC timing coordination
├── phase.rs            # Combat phase detection
├── utility.rs          # Utility evaluation
└── tooling/            # Debug & visualization tools
```

## The `game_core` Module

Campaign and overworld systems:

```
game_core/
├── types.rs                  # CampaignState, HeroState, Party
├── overworld_types.rs        # Region, map data
├── roster_types.rs           # Hero roster management
├── setup.rs                  # Campaign initialization
├── save.rs                   # Save/load serialization
├── migrate.rs                # Save version migration
├── campaign_systems.rs       # Turn processing
├── overworld_systems.rs      # Region updates
├── flashpoint_spawn.rs       # Crisis event generation
├── flashpoint_progression.rs # Crisis resolution
├── diplomacy_systems.rs      # Faction relationships
├── companion.rs              # Companion story arcs
└── tests/                    # Integration tests
```

## The `mission` Module

Multi-room dungeon runs:

```
mission/
├── execution/        # Mission state machine
├── room_gen/         # Procedural room generation
│   ├── floorplan.rs  #   Layout algorithms
│   ├── ml_gen.rs     #   ML-assisted generation
│   ├── nav.rs        #   Navigation mesh
│   └── validation.rs #   Layout validation
├── room_sequence/    # Room ordering
├── sim_bridge/       # Combat integration
├── vfx/              # Visual effects
├── enemy_templates/  # Enemy spawn definitions
├── objectives.rs     # Mission objectives
└── hero_templates.rs # Hero loading
```

## The `scenario` Module

```
scenario/
├── types.rs          # ScenarioConfig (TOML)
├── runner.rs         # Scenario execution
├── simulation.rs     # Scenario simulation
└── gen/              # Coverage-driven generation
```
