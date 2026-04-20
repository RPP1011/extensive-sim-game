# Docs — reading order

Start at **`game/overview.md`** to understand the project. Everything else is reference material pulled in on demand.

## Layout

```
docs/
  game/        — what we're building and how the pieces fit (START HERE)
  engine/      — runtime spec: cascade, SoA, spatial, events, determinism
  dsl/         — language spec: declaration kinds, grammar, semantics
  compiler/    — emission spec: DSL → Rust + Python + SPIR-V
  superpowers/ — planning + roadmap (plans, decisions, process)
  project_toc.md — cross-layer plan inventory
  audit_*.md   — periodic audit reports
```

## The split at a glance

- **Engine** = primitives only. Cascade runtime, SoA state containers, spatial index, event ring, RNG, tick orchestration, schema hashing. No game rules.
- **DSL** = all game logic. Entities, events, views, physics cascades, masks, verbs, scoring, invariants, probes, metrics.
- **Compiler** = bridge. DSL source → Rust handlers (engine's `SerialBackend`) + Python dataclasses (external ML training) + SPIR-V kernels (engine's `GpuBackend`).

If a change is about *how the sim runs* (determinism, performance, data layout), it's an engine change. If it's about *what the sim does* (a creature hates another, an action heals, a quest pays gold), it's a DSL change.

## Where the game currently lives

Today the game rules are hand-written Rust in the engine crate — legacy tech debt. The plan is **compiler-first**: grow the DSL compiler incrementally, and retire the legacy code one slice at a time as each compiler milestone lands. No parallel hand-written bootstrap crate.

See `game/compiler_progress.md` for the live milestone tracker and `game/feature_flow.md` for how to add features (and extend the compiler when needed).
