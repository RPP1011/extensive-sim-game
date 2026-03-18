# Events & Logging

Every meaningful action in the simulation produces a `SimEvent`. These events serve
as the simulation's audit log — used for AI training data extraction, replay
verification, UI animation triggers, and debugging.

## `SimEvent`

The `SimEvent` enum in `src/ai/core/events.rs` covers all observable outcomes:

```rust
pub enum SimEvent {
    // Damage & healing
    DamageDealt { source_id: u32, target_id: u32, amount: i32, ... },
    HealApplied { source_id: u32, target_id: u32, amount: i32 },
    ShieldApplied { source_id: u32, target_id: u32, amount: i32 },
    ShieldAbsorbed { target_id: u32, amount: i32 },

    // Unit lifecycle
    UnitDied { unit_id: u32, killer_id: Option<u32> },
    UnitSummoned { unit_id: u32, owner_id: u32 },

    // Abilities
    CastStarted { unit_id: u32, kind: CastKind, target_id: u32 },
    CastCompleted { unit_id: u32, kind: CastKind },
    AbilityUsed { unit_id: u32, ability_index: usize, name: String },

    // Status effects
    StatusApplied { target_id: u32, kind: StatusKind, duration_ms: u32 },
    StatusExpired { target_id: u32, kind: StatusKind },

    // Movement
    UnitMoved { unit_id: u32, from: SimVec2, to: SimVec2 },
    DashExecuted { unit_id: u32, from: SimVec2, to: SimVec2 },

    // Projectiles & zones
    ProjectileLaunched { source_id: u32, target_id: u32 },
    ProjectileHit { source_id: u32, target_id: u32 },
    ZoneCreated { position: SimVec2, radius: f32 },

    // ...
}
```

## Event-Driven Architecture

Events are **append-only** within a tick. The simulation never reads back its own
events during processing — they are purely output. This keeps the simulation logic
clean:

```rust
// Inside step():
let mut events = Vec::new();

// Systems append events
tick_status_effects(&mut state, tick, dt_ms, &mut events);
advance_projectiles(&mut state, tick, dt_ms, &mut events);
// ...

// Events returned alongside new state
(state, events)
```

## Uses of Events

### AI Training
Events are the raw signal for reward computation. The training pipeline extracts
damage dealt, healing done, kills, and deaths from events to compute per-step
rewards.

### Replay Verification
During replay, events from the original run are compared against events from the
replayed run to detect divergence. This catches determinism regressions.

### Testing
Tests assert on events to verify simulation behavior:

```rust
let (state, events) = step(state, &intents, FIXED_TICK_MS);
assert!(events.iter().any(|e| matches!(e,
    SimEvent::DamageDealt { amount: 55, .. }
)));
```

### UI Animation
In the live game, events trigger visual effects — damage numbers, ability
animations, death sequences, etc. The UI subscribes to the event stream without
coupling to simulation internals.

### Metrics
The `metrics.rs` module aggregates events across ticks to produce battle
statistics: total damage by unit, healing efficiency, CC uptime, etc.
