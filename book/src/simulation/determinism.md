# Determinism Contract

Determinism is the foundation that makes AI training, replay verification, and
automated testing possible. The simulation guarantees that given the same initial
state and the same sequence of intents, it will always produce the same output —
across runs, across platforms, across compiler versions.

## The RNG Contract

All randomness in the simulation flows through a single source:

```rust
pub fn next_rand_u32(state: &mut SimState) -> u32
```

This function advances `state.rng_state` using a deterministic PRNG algorithm.
The seed is part of `SimState`, so it is included in serialization and replay.

### Rules

1. **Never use `thread_rng()`** or any external RNG source in simulation code
2. **Never use `rand::random()`** or any implicit RNG
3. **Always call `next_rand_u32()`** through the `SimState` reference
4. **RNG call order must be deterministic** — this is why unit processing order
   is shuffled via the same RNG (making the shuffle itself deterministic)

## Verification

The determinism contract is verified in `src/ai/core/tests/determinism.rs`:

```rust
#[test]
fn test_determinism() {
    // Run simulation twice with same seed
    let state1 = make_state(units.clone(), 42);
    let state2 = make_state(units.clone(), 42);

    // Execute same intents
    let (result1, _) = step(state1, &intents, FIXED_TICK_MS);
    let (result2, _) = step(state2, &intents, FIXED_TICK_MS);

    // States must be identical
    assert_eq!(hash_sim_state(&result1), hash_sim_state(&result2));
}
```

Additional verification tools:

- `verify_determinism(state, intents, n)` — runs `n` ticks and confirms all
  runs produce identical state hashes
- `verify_replay_against_hashes(replay, expected_hashes)` — checks that a replay
  produces the expected hash at each tick
- `hash_sim_state(state)` — produces a deterministic hash of the full state

## What Can Break Determinism

Common pitfalls that the codebase avoids:

| Trap | How We Avoid It |
|------|----------------|
| `HashMap` iteration order | Use sorted iteration or `Vec` where order matters |
| Floating-point non-determinism | Avoid `f32::sin`/`cos` in simulation; use integer math where possible |
| Thread-local RNG | All RNG through `SimState` |
| Uninitialized memory | Rust's ownership model prevents this |
| Platform-dependent behavior | Stick to `u32`/`i32`/`f32` arithmetic; avoid platform-specific intrinsics |
| Unit processing order | Fisher-Yates shuffle with seeded RNG each tick |

## Running Determinism Tests

```bash
# Run just the determinism tests
cargo test determinism

# Run with serial execution for extra safety
cargo test determinism -- --test-threads=1
```
