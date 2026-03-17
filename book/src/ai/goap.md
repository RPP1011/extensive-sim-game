# GOAP Planner

The GOAP (Goal-Oriented Action Planning) system replaces stateless force-based
steering with **persistent, goal-driven plans**. Each unit picks a goal, plans an
action sequence, and executes across ticks. Replanning occurs only when the world
changes meaningfully.

## Module: `src/ai/goap/`

```
goap/
├── planner.rs      # A* planning algorithm
├── goal.rs         # Goal definitions
├── action.rs       # Action definitions and intent templates
├── world_state.rs  # Symbolic world state extraction
├── dsl.rs          # .goap file parser
├── party.rs        # Party culture (modifies costs/priorities)
├── plan_cache.rs   # Per-unit plan execution state
├── spatial.rs      # Spatial reasoning utilities
├── verify.rs       # Plan verification
└── tests.rs
```

## How GOAP Works

1. **World State Extraction** — each tick, the system extracts a symbolic
   `WorldState` for each unit (nearest enemy distance, ally HP, abilities
   available, etc.)

2. **Goal Selection** — the unit's goals are ranked by priority. Each goal has
   preconditions on the world state.

3. **Action Planning** — an A* search finds the cheapest sequence of actions
   that transforms the current world state into one satisfying the chosen goal.

4. **Plan Execution** — the unit follows its plan across ticks, converting
   abstract actions into concrete `IntentAction` values.

5. **Replanning** — when the world state changes significantly (new threat,
   ally died, target killed), the unit replans with hysteresis to prevent
   oscillation.

## World State

```rust
pub struct WorldState {
    pub nearest_enemy_dist: f32,
    pub nearest_ally_dist: f32,
    pub self_hp_pct: f32,
    pub lowest_ally_hp_pct: f32,
    pub target_id: Option<u32>,
    pub has_los_to_target: bool,
    pub ability_ready: [bool; 4],
    // ... more fields
}
```

## Intent Templates

GOAP actions produce `IntentTemplate` values, which are resolved to concrete
`IntentAction` values each tick:

```rust
pub enum IntentTemplate {
    AttackTarget(Target),
    ChaseTarget(Target),
    FleeTarget(Target),
    MaintainDistance(Target, f32),
    CastIfReady(usize, Target),
    Hold,
}
```

The target is resolved dynamically — `NearestEnemy` looks up the actual nearest
enemy each tick, so the action adapts as units move.

## Target Types

```rust
pub enum Target {
    Self_,
    NearestEnemy,
    NearestAlly,
    LowestHpEnemy,
    LowestHpAlly,
    HighestDpsEnemy,
    HighestThreatEnemy,
    CastingEnemy,
    EnemyAttacking(Box<Target>),
    Tagged(String),
    UnitId(u32),
}
```

## GOAP Definition Files

GOAP behaviors are defined in `.goap` files in `assets/behaviors/`:

```
# frontline.goap — aggressive melee fighter

goal KillPriority {
    priority: 90
    precondition: enemy_alive
    satisfaction: enemy_dead
}

goal ProtectHealer {
    priority: 70
    precondition: healer_threatened
    satisfaction: healer_safe
}

action AttackNearest {
    cost: 1.0
    intent: attack nearest_enemy
    effect: enemy_taking_damage
    duration: 3
}

action ChargeTarget {
    cost: 2.0
    precondition: ability_0_ready
    intent: cast 0 nearest_enemy
    effect: enemy_stunned
    duration: 1
}
```

## Party Culture

The `PartyCulture` system modifies GOAP parameters based on team composition
and narrative context:

```rust
pub struct PartyCulture {
    pub replan_hysteresis: f32,           // how much change triggers replan
    pub action_cost_modifiers: HashMap<String, f32>,  // adjust action costs
    pub goal_insistence_modifiers: HashMap<String, f32>,
}
```

This allows the same GOAP definitions to produce different behavior depending
on the campaign context — a battered, retreating party might have higher
flee costs, while a fresh party is more aggressive.

## Verification

In debug builds, `verify::verify_goap()` checks for common issues:

- Units stuck in Hold with no plan for too long
- Plans referencing dead targets
- Infinite replan loops (oscillating between two goals)
