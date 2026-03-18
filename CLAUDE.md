# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
cargo test ai::core::tests     # Tests in a specific module
cargo test shield_absorbs      # Single test by name substring
cargo test -- --nocapture      # Show println output
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

### CLI (xtask)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario generate dataset/scenarios/
cargo run --bin xtask -- scenario oracle eval scenarios/
cargo run --bin xtask -- scenario oracle transformer-rl generate scenarios/
cargo run --bin xtask -- train-v6 dataset/scenarios/ --iters 100
cargo run --bin xtask -- roomgen export --output generated/rooms.jsonl
```

## Architecture

### Three-Layer Simulation

```
Campaign (turn-based overworld) → Mission (multi-room dungeons) → Combat (100ms fixed-tick deterministic sim)
```

The combat layer is the core of the AI/ML work. Everything runs through `step(state, intents, dt_ms) → (state, events)` in `src/ai/core/simulation.rs`.

**Namespace alias:** `crate::sim` re-exports `ai::core` — the simulation engine is not AI, it's the deterministic physics/rules engine. New code should prefer `crate::sim::*`.

### Key Module Map

- **`src/ai/core/`** (aliased as `crate::sim`) — Simulation engine: `SimState`, `UnitState`, `step()`, damage calc, effect application
- **`src/ai/effects/`** — Data-driven ability system with DSL parser (`.ability` files). Five composable dimensions: Effect (what), Area (where), Delivery (how), Trigger (when), Tags (power levels)
- **`src/ai/effects/dsl/`** — winnow-based parser for ability DSL: `parser.rs` → `lower.rs` (AST→AbilityDef)
- **`src/ai/core/ability_eval/`** — Neural ability evaluator (urgency interrupt layer, fires when urgency > 0.4)
- **`src/ai/core/ability_transformer/`** — Grokking-based transformer for ability decisions, with cross-attention over entity tokens
- **`src/ai/core/self_play/`** — RL policy learning (REINFORCE + PPO, pointer action space)
- **`src/ai/goap/`** — GOAP (Goal-Oriented Action Planning) AI with DSL parser (`.goap` files), planner, party culture
- **`src/ai/squad/`** — Squad-level AI: personality profiles, formation modes, intent generation
- **`src/ai/pathing/`** — Grid navigation, pathfinding, cover
- **`src/scenario/`** — Scenario config (TOML), runner, coverage-driven generation
- **`src/bin/xtask/`** — CLI task runner (scenarios, oracle, training, roomgen)
- **`src/bin/sim_bridge/`** — Headless sim for external agents via NDJSON protocol

### Determinism Contract

All simulation randomness flows through `SimState.rng_state` via `next_rand_u32()`. Never use `thread_rng()` or any external RNG in simulation code. Unit processing order is shuffled per tick to prevent first-mover bias. Tests in `src/ai/core/tests/determinism.rs` verify reproducibility. CI runs determinism tests in both debug and release modes.

### Effect System

Effects are plain data structs dispatched via pattern matching (no closures). The pipeline:
1. `.ability` DSL file → parser (`winnow`) → AST
2. AST → `lower.rs` → `AbilityDef` (with `Effect`, `Area`, `Delivery`, conditions)
3. At runtime: `apply_effect.rs` / `apply_effect_ext.rs` dispatch effects onto `SimState`

### AI Decision Pipeline

Intent generation flows through layers, each can override:
1. **Squad AI** (`squad/intents.rs`): team-wide personality-driven behavior
2. **Ability Evaluator** (optional): neural urgency interrupt for ability usage
3. **Transformer** (optional): cross-attention decision head over entity + ability tokens
4. **GOAP** (optional): goal-oriented action planning with party culture modifiers
5. **Control AI** (optional): hard CC timing coordination

### Workspace

Two crates: root (`bevy_game`) and `crates/ability_operator` (behavioral embeddings for abilities).

### Hero Templates

Defined in `assets/hero_templates/` as hybrid TOML (stats) + `.ability` DSL (abilities). The `.ability` files are the source of truth for ability definitions. 27 base heroes + 172 LoL hero imports in `assets/lol_heroes/`.

### Test Helpers

In `src/ai/core/tests/mod.rs`: `hero_unit(id, team, pos)`, `make_state(units, seed)` for creating deterministic test fixtures. Tests assert on `SimEvent` logs and unit state after `step()`.
