# Testing

The project uses a multi-layered testing strategy: unit tests, integration tests,
determinism tests, property-based tests, and stress tests.

## Running Tests

```bash
cargo test                          # All tests
cargo test ai::core::tests          # Tests in a specific module
cargo test shield_absorbs           # Single test by name substring
cargo test -- --nocapture           # Show println output
cargo test -- --test-threads=1      # Serial execution
```

## Test Categories

### Unit Tests

Located in `#[cfg(test)] mod tests` blocks within each module. These test
individual functions in isolation:

```rust
#[test]
fn shield_absorbs_damage() {
    let mut units = vec![
        hero_unit(1, Team::Hero, SimVec2::new(0.0, 0.0)),
        hero_unit(2, Team::Enemy, SimVec2::new(1.0, 0.0)),
    ];
    units[0].shield_hp = 50;
    let state = make_state(units, 42);

    let intents = vec![UnitIntent {
        unit_id: 2,
        action: IntentAction::Attack { target_id: 1 },
    }];

    let (state, events) = step(state, &intents, FIXED_TICK_MS);
    assert!(events.iter().any(|e| matches!(e,
        SimEvent::ShieldAbsorbed { .. }
    )));
}
```

### Determinism Tests

**Module:** `src/ai/core/tests/determinism.rs`

Verify that the simulation produces identical results from identical inputs:

```bash
cargo test determinism -- --test-threads=1
```

### Mechanics Tests

**Module:** `src/ai/core/tests/mechanics.rs`

Test specific game mechanics:
- Damage calculation with armor
- Healing cap (can't exceed max HP)
- Status effect application and expiry
- Projectile collision
- Zone ticking
- Summon creation
- Crowd control interrupts

### Ability Tests

**Module:** `src/ai/core/tests/abilities.rs`

Test hero abilities end-to-end:
- Fireball deals expected damage
- Shield absorbs before HP
- Stun prevents actions
- Dash moves the caster
- Heal restores HP

### Property-Based Tests

Using [proptest](https://docs.rs/proptest):

**Module:** `src/ai/effects/dsl/fuzz.rs`

Generate random ability definitions, emit them to DSL text, parse them back,
and verify the round-trip produces identical definitions.

```bash
cargo test fuzz
```

### Stress Tests

**Module:** `src/ai/core/tests_stress.rs`

Run many ticks of simulation with large unit counts to check for:
- Panics from unexpected state combinations
- Performance regressions
- Memory leaks (unbounded growth)

### Integration Tests

**Directory:** `tests/`, `src/game_core/tests/`

Test larger workflows:
- Campaign creation → save → load → verify
- Scenario loading → simulation → assertion checking
- Mission setup → room generation → combat → outcome

## Test Helpers

**Module:** `src/ai/core/tests/mod.rs`

```rust
/// Create a hero unit with default stats at a position.
pub fn hero_unit(id: u32, team: Team, pos: SimVec2) -> UnitState

/// Create a SimState from a list of units with a specific RNG seed.
pub fn make_state(units: Vec<UnitState>, seed: u64) -> SimState
```

These helpers create deterministic test fixtures with sensible defaults,
reducing boilerplate in test code.

## Testing Philosophy

1. **Test behavior, not implementation** — assert on events and final state,
   not internal function calls
2. **Deterministic by default** — all tests use seeded RNG
3. **Small states** — use 2-4 units, not full 4v4 scenarios (unless testing
   scale)
4. **Events as assertions** — the event log is the primary test oracle
5. **Property tests for parsers** — fuzz the DSL parser to catch edge cases
