# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project

Deterministic tactical orchestration RPG built in Rust. Combat sim is a 100ms fixed-tick deterministic engine. The `Backend` trait in `crates/engine/src/backend.rs` is the CPU/GPU split point; concrete impls land per the compiler-first roadmap (Plan B1' Task 11+). Rules-as-data: a custom DSL under `assets/sim/*.sim` is compiled by `crates/dsl_compiler` and consumed by per-runtime build.rs scripts (each `crates/*_runtime` crate compiles its own `.sim` source into `OUT_DIR`). The `.ability` corpus lives under `dataset/abilities/lol_heroes/` for parser regression coverage; the `assets/hero_templates/` hero-template layer was retired and is not coming back.

## Constitution

The architectural constitution at `docs/constitution.md` is auto-loaded into agent context on session start (see `.claude/settings.json`). Every plan must include an Architectural Impact Statement preamble per `docs/architecture/plan-template-ais.md` (P8).

## Build & test

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
cargo test -p engine           # Tests in the engine crate only
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

### Per-sim runtime binaries

There is no `xtask` umbrella binary today (retired in Phase 7 wolf-sim
wipe, 2026-05-02). Each `crates/*_runtime` crate compiles to its own
`*_app` binary; e.g. `cargo run -p boids_runtime --bin boids_app`. List
all of them with `cargo build --bin foo 2>&1 | head` (cargo prints the
available bin targets when you give it an unknown name).

## Where to look

- **Reading order:** start with `docs/llms.txt`, fetch the docs you need.
- **What's built:** `docs/engine/status.md` (live per-subsystem implementation status).
- **What's coming:** `docs/ROADMAP.md` (comprehensive future-work index).
- **Contract:** `docs/spec/` (canonical specification, 6 files: `engine.md`, `dsl.md`, `state.md`, `ability.md`, `economy.md`, `README.md`).
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
- Workspace members: see `Cargo.toml` `[workspace] members` — `crates/dsl_ast`, `crates/dsl_compiler`, `crates/engine`, `crates/engine_data`, `crates/engine_gpu_rules`, plus ~40 `crates/*_runtime` per-sim crates and `crates/sim_app`. `crates/ability_operator` and a few research crates are excluded.
- All simulation randomness MUST flow through `per_agent_u32(seed, agent_id, tick, purpose)` — see P5.
