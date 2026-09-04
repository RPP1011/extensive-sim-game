# Deterministic Agent Simulation Engine

A Rust engine and DSL compiler for large-scale, deterministic, tick-based multi-agent simulation. Rules — what events exist, which physics rules react to them, how views accumulate, how agents score actions — are written in a custom DSL (`.sim` files under `assets/sim/`), not hand-written Rust. A compiler lowers that DSL to a CPU reference implementation and a GPU compute-graph (WGSL), so the same rules run identically on both backends. The hard architectural rule, stated in `docs/game/overview.md`: **the engine itself contains zero game logic.**

This repo is not itself a shipped game. It ships a corpus of ~107 `.sim` demo/regression fixtures spanning many genres (predator/prey, dungeon crawling, duels, boids, diplomacy, crafting, belief/theory-of-mind probes, an Among Us clone, and more) that exercise the DSL and compiler, plus a handful of runtime/viewer crates that drive them. An earlier version of this README described a specific "tactical crisis-management RPG" — hero roster, contested overworld, flashpoint crises, 167 world-sim systems. That description was aspirational; no such campaign/overworld/mission layer is evidenced in the current codebase beyond the fixtures below (`wolves_and_humans`, `dungeon_horde`'s real hero/class system, `squad_skirmish`, `hill_raid`, and similar). What's actually implemented, subsystem by subsystem, is tracked live in `docs/engine/status.md`.

One real game has been built on top of this engine and lives in its own repo, pulling this one in as a pinned `git` dependency: **Mount & Blade: Webband** (github.com/RPP1011/webband, private) — see `docs/superpowers/plans/webband-port.md` for how that split happened and what stayed here as engine fixes.

> **Status:** Active development.

## Nifty diagram illustrating the perils of AI-native development
<img src="spaget.svg">

## Documentation

Start at [`docs/llms.txt`](docs/llms.txt) — an index with reading order into everything else. Highlights:

- [`docs/overview.md`](docs/overview.md) — five-minute architectural intro (rules-as-data, the DSL→engine→GPU pipeline, the deterministic tick).
- [`docs/engine/status.md`](docs/engine/status.md) — **start here** for what's actually built: live per-subsystem ✅/⚠️/❌ status, known weak tests, open verification questions.
- [`docs/spec/`](docs/spec/) — the canonical specification (DSL grammar, engine runtime contract, field catalog, ability DSL, economy). The spec is the contract; other docs cross-reference it rather than restate it.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — comprehensive future-work index.
- [`docs/adr/`](docs/adr/) — locked architecture decisions.
- [`docs/game/`](docs/game/) — the DSL-compiler migration: `compiler_progress.md` is the live milestone tracker (which hand-written legacy code has been replaced by compiler output and which hasn't), `wolves_and_humans.md` is the canonical worked fixture.

Every crate, plus several top-level data/tooling directories (`assets/`, `dataset/`, `stdlib/`, `scripts/`, ...), also carries its own `CLAUDE.md` with AI-agent-oriented depth this README doesn't repeat.

## Build & test

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests (workspace)
cargo test -p engine           # Engine crate tests only
cargo test -p sims             # Fixture pin/determinism tests (crates/sims/tests/*_pin.rs etc.)
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

A few `crates/sims` test files need a larger stack (`RUST_MIN_STACK=33554432`) — see `crates/sims/CLAUDE.md` if you hit a stack-overflow failure there.

`ability_operator`, `ability-vae`, `combat-trainer`, and `world_sim_bench` are excluded from the default workspace build (ML training/benchmarking crates addressed via `--manifest-path`, not `-p`) — see each crate's own `CLAUDE.md` before assuming they build; a couple currently don't (dangling dependency on a deleted crate).

## Running a fixture

Most of the ~107 fixtures aren't wired into any runnable binary and are exercised only through their pin/determinism test under `crates/sims/tests/` — that's the primary way to verify one works (`cargo test -p sims <fixture>_pin`). A handful have a runnable front end:

```bash
# Terminal ASCII visualizer / non-interactive assertion harness
cargo run -p sim_app --bin viz_app --features bin-viz_app -- <sim_name>
cargo run -p sim_app --bin tom_probe_app --features bin-tom_probe_app

# Windowed Vulkan viewers for two specific pilot fixtures
cargo run -p viewer_runtime --bin viewer_app --release [SEED]   # dungeon_horde
cargo run -p viewer_runtime --bin vs_viewer --release [SEED]    # vampire_survivors

# Generic windowed player for any compiled fixture, via voxel_engine + egui
cargo run -p engine_play --bin play -- <fixture> [seed] [agents]
```

See each crate's `CLAUDE.md` for exact fixture names, controls, and gotchas.

## Project Management

Use GitHub issues and milestones for active planning. Implementation plans live in `docs/superpowers/plans/`; per-subsystem status lives in `docs/engine/status.md`.
