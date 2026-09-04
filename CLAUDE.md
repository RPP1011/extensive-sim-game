# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project

Deterministic tactical orchestration RPG built in Rust. Combat sim is a 100ms fixed-tick deterministic engine. The `ComputeBackend` trait in `crates/engine/src/backend.rs` is the CPU/GPU split point; concrete impls land per the compiler-first roadmap (Plan B1' Task 11+). Rules-as-data: a custom DSL under `assets/sim/*.sim` is compiled by `crates/dsl_compiler`. Most fixtures are auto-discovered and compiled into the `crates/sims` mega-crate — its `build.rs` scans `assets/sim/*.sim` and emits `sims::<fixture>::GeneratedRuntime` for every migrated stem (see the match arm list in `crates/sims/build.rs`); adding a fixture is "drop a `.sim` file, add its stem to that list, rebuild" — no new crate. A couple of fixtures still live in their own legacy `crates/*_runtime` crate (each compiling its own `.sim` via its own build.rs into `OUT_DIR`) from before the mega-crate consolidation. The `.ability` corpus lives under `dataset/abilities/lol_heroes/` for parser regression coverage; the `assets/hero_templates/` hero-template layer was retired and is not coming back.

## Constitution

The architectural constitution at `docs/constitution.md` is auto-loaded into agent context on session start (see `.claude/settings.json`). Every plan must include an Architectural Impact Statement preamble per `docs/architecture/plan-template-ais.md` (P8).

## Build & test

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
cargo test -p engine           # Tests in the engine crate only
cargo test -p sims             # Fixture pin/determinism tests (crates/sims/tests/*_pin.rs etc.)
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

### Running a sim fixture

There is no `xtask` umbrella binary today (retired in Phase 7 wolf-sim
wipe, 2026-05-02). Almost every fixture lives inside the `sims` crate
(`sims::<fixture>::GeneratedRuntime`) rather than its own binary — the
main way to exercise one is its pin test under `crates/sims/tests/`.

`crates/sim_app` hosts the remaining runnable binaries, each gated behind
a `bin-<name>` cargo feature so unrelated per-fixture deps aren't pulled
in by default:

```bash
cargo run -p sim_app --bin viz_app --features bin-viz_app -- <sim_name>       # terminal visualizer
cargo run -p sim_app --bin tom_probe_app --features bin-tom_probe_app
```

`viz_app` with no args lists the sims it currently knows how to drive (its
`SIMS` table in `crates/sim_app/src/viz_app.rs`); most fixtures haven't been
wired into that table yet even though they compile and have pin tests.
`crates/tom_probe_runtime` and `crates/viewer_runtime` are the last
crates still following the old one-crate-per-fixture pattern.

## Where to look

- **Reading order:** start with `docs/llms.txt`, fetch the docs you need.
- **What's built:** `docs/engine/status.md` (live per-subsystem implementation status).
- **What's coming:** `docs/ROADMAP.md` (comprehensive future-work index).
- **Contract:** `docs/spec/` (canonical specification; `README.md` is the index/reading order).
- **Active plans:** `docs/superpowers/plans/`.
- **Locked decisions:** `docs/adr/`.

## Conventions

- The spec is the contract. Live status lives in `engine/status.md`. Don't duplicate.
- The constitution states each principle once. Other docs do not paraphrase or redirect.
- Every new plan needs an AIS preamble (P8). Skipping it is a process violation.
- Historical content (executed plans, resolved audits, design rationale) lives in **git history**, not active docs.
- Engine extensions are gated by `.githooks/pre-commit` (cargo check + `// GENERATED` header rule) and the in-tree `crates/engine/tests/schema_hash.rs` test. DSL parse/lower/emit errors surface via each runtime's build.rs during `cargo check` — the standalone `compile-dsl --check` pass was retired with xtask. The Claude-driven critic skills were retired 2026-05-04 — they burned tokens with low signal.
- **One-time setup after clone:** `git config core.hooksPath .githooks` enables the pre-commit hard block. Without it, the hooks don't engage.

## Tooling caveats

- This is a Rust workspace; the root `Cargo.toml` is a virtual manifest (no `[package]`).
- Workspace members: see `Cargo.toml` `[workspace] members` — `dsl_ast`, `dsl_compiler`, `engine`, `engine_data`, `engine_gpu_rules`, `engine_play`, `engine_play_api`, `engine_ui`, `engine_voxel`, `sim_app`, `sims`, `tom_probe_runtime`, `viewer_runtime`. `ability_operator`, `ability-vae`, `combat-trainer` and `world_sim_bench` (ML training / benchmarking crates) are excluded from the default workspace build.
- All simulation randomness MUST flow through `per_agent_u32(seed, agent_id, tick, purpose)` — see P5.
