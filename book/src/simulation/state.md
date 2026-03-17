# State & Types

The simulation state is defined in `src/ai/core/types.rs`. These types are the
vocabulary of the entire combat system.

## `SimState`

The top-level simulation state:

```rust
pub struct SimState {
    pub tick: u64,
    pub units: Vec<UnitState>,
    pub rng_state: u64,
    pub projectiles: Vec<Projectile>,
    pub zones: Vec<Zone>,
    pub tethers: Vec<Tether>,
    // ... additional fields
}
```

Key fields:
- `tick` — monotonically increasing tick counter
- `units` — all units (heroes and enemies), alive and dead
- `rng_state` — the single source of randomness (see [Determinism](./determinism.md))
- `projectiles` — in-flight projectiles awaiting collision
- `zones` — active ground-effect zones
- `tethers` — active beam/tether connections between units

## `UnitState`

A unit's complete state. This is a large struct — here are the most important fields:

```rust
pub struct UnitState {
    // Identity
    pub id: u32,
    pub team: Team,

    // Vitals
    pub hp: i32,
    pub max_hp: i32,
    pub shield_hp: i32,
    pub armor: f32,
    pub magic_resist: f32,

    // Position & movement
    pub position: SimVec2,
    pub move_speed_per_sec: f32,

    // Basic attack
    pub attack_damage: i32,
    pub attack_range: f32,
    pub attack_cooldown_ms: u32,
    pub cooldown_remaining_ms: u32,

    // Hero ability engine
    pub abilities: Vec<AbilitySlot>,
    pub passives: Vec<PassiveSlot>,
    pub status_effects: Vec<ActiveStatusEffect>,

    // Resource system
    pub resource: i32,
    pub max_resource: i32,
    pub resource_regen_per_sec: f32,

    // Casting state
    pub casting: Option<CastState>,
    pub channeling: Option<ChannelState>,

    // Summon relationship
    pub owner_id: Option<u32>,
    pub directed: bool,

    // Combat stats (for AI)
    pub total_damage_done: i32,
    pub total_healing_done: i32,

    // Terrain context
    pub cover_bonus: f32,
    pub elevation: f32,
}
```

### Damage Reduction

Armor and magic resist use the formula:

```
reduction = stat / (100 + stat)
```

This means 100 armor gives 50% physical damage reduction, 200 gives 66%, etc.
The formula has diminishing returns, preventing any unit from becoming completely
immune.

## `Team`

```rust
pub enum Team {
    Hero,
    Enemy,
}
```

Simple two-team system. The simulation doesn't need more — missions always pit
the player's hero squad against enemy groups.

## `UnitIntent`

```rust
pub struct UnitIntent {
    pub unit_id: u32,
    pub action: IntentAction,
}

pub enum IntentAction {
    Hold,
    MoveTo { position: SimVec2 },
    Attack { target_id: u32 },
    CastHeal { target_id: u32 },
    CastControl { target_id: u32 },
    UseAbility { ability_index: usize, target: AbilityTarget },
    Dash,
    Retreat,
    Skulk,
    // ...
}
```

Intents are the bridge between AI and simulation. The AI layer produces them;
the simulation layer consumes them.

## `SimVec2`

A simple 2D vector:

```rust
pub struct SimVec2 {
    pub x: f32,
    pub y: f32,
}
```

All combat takes place on a 2D plane. Height/elevation is tracked separately
as a scalar on each unit, not as a 3D position.

## `CastState`

When a unit begins casting an ability, a `CastState` is attached:

```rust
pub struct CastState {
    pub target_id: u32,
    pub target_pos: Option<SimVec2>,
    pub remaining_ms: u32,
    pub kind: CastKind,
    pub area: Option<Area>,
    pub ability_index: Option<usize>,
    pub effect_hint: CastEffectHint,
}
```

The `effect_hint` field tells the AI what category of effect the cast will
produce (damage, heal, CC, utility), enabling spatial awareness without
needing to parse the full ability definition at runtime.
