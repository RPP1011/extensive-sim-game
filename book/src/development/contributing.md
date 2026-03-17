# Contributing

This chapter covers development practices, conventions, and common workflows
for contributors.

## Getting Started

```bash
git clone <repo-url>
cd super-duper-octo-spork
cargo build
cargo test
```

## Code Organization Rules

### The Determinism Rule

**Never use `thread_rng()` or external RNG in simulation code.** All randomness
must flow through `SimState.rng_state` via `next_rand_u32()`. If you're writing
code that runs during `step()`, use the simulation's RNG. If you're writing code
that runs outside simulation (UI, tooling, generation), you may use `rand`
directly.

### Effects as Data

**Never use closures or trait objects in the effect system.** Effects must be
plain data structs that can be serialized, compared, and pattern-matched. If
you need new behavior, add a variant to the `Effect` enum and a match arm in
`apply_effect.rs`.

### Intent Separation

**AI code must never directly mutate `SimState`.** AI layers produce
`Vec<UnitIntent>`, which the simulation consumes. This separation is fundamental
to the architecture.

## Adding a New Effect

1. Add the variant to the `Effect` enum in `src/ai/effects/types.rs`
2. Add the match arm in `src/ai/core/apply_effect.rs`
3. Add DSL syntax in `src/ai/effects/dsl/parse_effects.rs`
4. Add lowering in `src/ai/effects/dsl/lower_effects.rs`
5. Add emission in `src/ai/effects/dsl/emit_effects.rs`
6. Add tests in `src/ai/core/tests/`
7. Update the fuzz generators in `src/ai/effects/dsl/fuzz_generators.rs`

## Adding a New Hero

1. Create `assets/hero_templates/<name>.toml` with stats
2. Create `assets/hero_templates/<name>.ability` with abilities
3. Test with `cargo test` and `cargo run --bin xtask -- scenario run`

## Adding a New AI Layer

1. Create a module under `src/ai/`
2. Implement intent generation: `fn generate(state: &SimState) -> Vec<UnitIntent>`
3. Integrate into the pipeline in the appropriate position
4. Add tests that verify intent output for known states

## Common Commands

```bash
# Build
cargo build                           # debug
cargo build --release                 # optimized
cargo build --features burn-cpu       # with Burn ML

# Test
cargo test                            # all tests
cargo test -- --nocapture             # with output
cargo test -- --test-threads=1        # serial

# Run
cargo run                             # game
cargo run --bin xtask -- --help       # CLI tools
cargo run --bin sim_bridge            # headless sim

# Python training
uv run --with numpy --with torch python training/train_rl_v5.py ...
```

## Project Structure Quick Reference

| What | Where |
|------|-------|
| Simulation engine | `src/ai/core/simulation.rs` |
| Effect types | `src/ai/effects/types.rs` |
| Ability DSL parser | `src/ai/effects/dsl/parser.rs` |
| Squad AI | `src/ai/squad/intents.rs` |
| GOAP planner | `src/ai/goap/planner.rs` |
| Campaign state | `src/game_core/types.rs` |
| Mission execution | `src/mission/execution/` |
| Hero templates | `assets/hero_templates/` |
| Behavior files | `assets/behaviors/` |
| Training scripts | `training/` |
| CLI tools | `src/bin/xtask/` |
| Tests | `src/ai/core/tests/` |

## Commit Message Convention

Use imperative mood, present tense:
- `add stealth system with LOS occlusion`
- `fix damage calculation with armor stacking`
- `port IMPALA training to rl4burn`

## Debugging Tips

### Simulation Debugging
```bash
# Run with event output
cargo test my_test -- --nocapture

# Check determinism
cargo test determinism -- --test-threads=1

# Run a scenario with verbose output
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml --verbose
```

### DSL Debugging
```bash
# Test parsing a specific ability file
cargo test parse_abilities -- --nocapture

# Run fuzz tests
cargo test fuzz -- --test-threads=1
```

### AI Debugging
```bash
# Run with GOAP verification
cargo test goap -- --nocapture

# Inspect intent generation
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml --trace-intents
```
