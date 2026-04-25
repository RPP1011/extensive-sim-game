# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project

Deterministic tactical orchestration RPG built in Rust. Combat sim is a 100ms fixed-tick deterministic engine with both CPU (`SerialBackend`) and GPU (`GpuBackend`) backends. World simulation is a Dwarf-Fortress-style zero-player layer underneath. Rules-as-data: a custom DSL (`assets/sim/*.sim`, `assets/hero_templates/*.ability`) is compiled to both backends.

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

### CLI (xtask)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario generate dataset/scenarios/
```

## Where to look

- **Reading order:** start with `docs/llms.txt`, fetch the docs you need.
- **What's built:** `docs/engine/status.md` (live per-subsystem implementation status).
- **What's coming:** `docs/ROADMAP.md` (comprehensive future-work index).
- **Contract:** `docs/spec/` (canonical specification, 10 files).
- **Active plans:** `docs/superpowers/plans/`.
- **Locked decisions:** `docs/adr/`.

## Conventions

- The spec is the contract. Live status lives in `engine/status.md`. Don't duplicate.
- The constitution states each principle once. Other docs do not paraphrase or redirect.
- Every new plan needs an AIS preamble (P8). Skipping it is a process violation.
- Historical content (executed plans, resolved audits, design rationale) lives in **git history**, not active docs.

## Tooling caveats

- This is a Rust workspace; the root `Cargo.toml` is a virtual manifest.
- Two crates: root (`bevy_game`) and `crates/ability_operator`. Engine + GPU live under `crates/engine*`.
- All simulation randomness MUST flow through `per_agent_u32(seed, agent_id, tick, purpose)` — see P5.
