# Effect System

The effect system is the heart of what makes each hero unique. It defines a
composable vocabulary of **what abilities do** — damage, healing, shields, crowd
control, summons, terrain manipulation, and more — all as plain data.

## Design Principles

1. **Effects are data, not code.** Every effect is a struct with fields, not a
   closure or trait object. This makes effects serializable, inspectable, and
   diffable.

2. **Dispatch via pattern matching.** Effect application uses `match` on enum
   variants, not dynamic dispatch. The compiler can verify exhaustiveness.

3. **Composability over inheritance.** A single ability can combine multiple
   effects, areas, delivery mechanisms, and conditions. There is no class
   hierarchy of ability types.

## Module Layout

```
src/ai/effects/
├── defs.rs           # AbilityDef, PassiveDef, AbilitySlot
├── types.rs          # Effect, Area, Delivery, Tags, ConditionalEffect
├── effect_enum.rs    # Master enum of all effect types
├── manifest.rs       # Ability registry and lookup
└── dsl/              # Parser (see Ability DSL chapter)
```

## The Pipeline

From definition to execution:

```
  .ability file          Parser              Lowering           Runtime
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ ability X {  │    │              │    │              │    │              │
│   damage 50  │───▶│  AST nodes   │───▶│  AbilityDef  │───▶│ apply_effect │
│   stun 2s    │    │              │    │  { effects,  │    │ (pattern     │
│ }            │    │              │    │    delivery } │    │  matching)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. `.ability` DSL files are parsed into an AST by the winnow-based parser
2. The AST is lowered into `AbilityDef` / `PassiveDef` structs
3. At runtime, `apply_effect.rs` dispatches each effect via pattern matching

## AbilityDef

The central type:

```rust
pub struct AbilityDef {
    pub name: String,
    pub targeting: AbilityTargeting,
    pub range: f32,
    pub cooldown_ms: u32,
    pub cast_time_ms: u32,
    pub ai_hint: String,
    pub effects: Vec<ConditionalEffect>,
    pub delivery: Option<Delivery>,
    pub resource_cost: i32,
    // ... advanced fields (charges, toggle, recast, forms)
}
```

### Targeting Modes

```rust
pub enum AbilityTargeting {
    TargetEnemy,      // click an enemy
    TargetAlly,       // click an ally
    SelfCast,         // cast on self
    SelfAoe,          // centered on caster
    GroundTarget,     // click a position
    Direction,        // aim in a direction
    Vector,           // click-drag (start + direction)
    Global,           // hits all enemies on map
}
```

## PassiveDef

Passive abilities trigger automatically:

```rust
pub struct PassiveDef {
    pub name: String,
    pub trigger: Trigger,
    pub cooldown_ms: u32,
    pub effects: Vec<ConditionalEffect>,
    pub range: f32,
}
```

Passives fire when their trigger condition is met (taking damage, HP dropping
below a threshold, using an ability, etc.) and respect their cooldown.
