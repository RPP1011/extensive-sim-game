# Core Simulation

The combat simulation is the foundation of the entire project. It is a deterministic,
tick-based engine that processes unit intents and produces state transitions plus
event logs.

Everything — AI training, replay verification, scenario benchmarking, and live
gameplay — runs through a single entry point:

```rust
pub fn step(state: SimState, intents: &[UnitIntent], dt_ms: u32)
    -> (SimState, Vec<SimEvent>)
```

## Design Philosophy

The simulation follows a **pure functional** design. The `step()` function:

- Takes `SimState` **by value** (not by reference)
- Returns a **new** `SimState` plus a log of `SimEvent`s
- Has **no side effects** — no I/O, no global state, no thread-local storage
- Uses **design-by-contract** annotations via the `contracts` crate

This means the simulation is:

| Property | How It's Achieved |
|----------|------------------|
| Deterministic | All RNG via `SimState.rng_state` |
| Reproducible | Same seed + same intents = same output |
| Replayable | Record intents, replay through `step()` |
| Testable | Create states with `make_state()`, assert on events |
| Trainable | Feed policy-generated intents, observe outcomes |

## Module Layout

```
src/ai/core/
├── simulation.rs       # step() — the main entry point
├── types.rs            # SimState, UnitState, Team, CastState
├── events.rs           # SimEvent enum
├── math.rs             # SimVec2, distance, movement
├── helpers.rs          # is_alive, can_see, can_see_with_nav
├── intent.rs           # Intent collection and validation
├── damage.rs           # Damage formula and application
├── targeting.rs        # Target selection logic
├── conditions.rs       # Ability condition evaluation
├── triggers.rs         # Passive and reactive trigger system
├── resolve.rs          # Cast resolution pipeline
├── apply_effect.rs     # Core effect application
├── apply_effect_ext.rs # Extended effect application
├── tick_systems.rs     # Per-tick subsystems (cooldowns, status, etc.)
├── tick_world.rs       # World-level tick (zones, projectiles)
├── summon_templates.rs # Summon unit creation
├── determinism.rs      # RNG utilities, hash verification
├── replay.rs           # Replay execution
├── metrics.rs          # Battle statistics collection
├── verify.rs           # Post-tick invariant checks
└── unit_store.rs       # Unit collection utilities
```

## The Tick Pipeline

Each call to `step()` processes one 100ms tick in this order:

1. **Increment tick counter**
2. **Tick hero cooldowns** — decrement ability cooldown timers
3. **Tick status effects** — apply/expire buffs, debuffs, DoTs
4. **Advance projectiles** — move in-flight projectiles, check collisions
5. **Tick periodic passives** — fire passives whose timers elapsed
6. **Tick zones** — apply zone (ground AoE) effects to units inside
7. **Tick channels** — progress channeled abilities
8. **Tick tethers** — maintain/break beam connections
9. **Record state history** — push position/HP snapshots for Rewind
10. **Shuffle unit order** — randomize via seeded RNG to prevent first-mover bias
11. **Process unit intents** — for each living unit, resolve their intent
12. **Return** new state + event log

This ordering matters. Projectiles resolve before intents, meaning a projectile
launched last tick can kill a unit before it acts this tick. Status effects tick
before intent processing, so a stun that expires this tick allows the unit to act.
