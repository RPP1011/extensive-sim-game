# World Sim Engine Docs

Runtime contract and implementation notes for the world-sim engine crate (`crates/engine/`).

This tree complements `../dsl/` (language reference) and `../compiler/` (DSL → engine lowering). The engine is the Rust library those two target; it owns determinism-load-bearing infrastructure and nothing else.

## What's in this tree

1. **`spec.md`** — the runtime contract. 23 sections covering state, events, spatial index, RNG, actions, cascade, mask, policy, tick pipeline, views, aggregates, trajectory, save/load, invariants, probes, telemetry, schema hash, observation packer, debug & trace runtime, and non-goals. An implementation is correct iff it satisfies §§2–23.

## When to add new docs here

- **`perf.md`** — once benchmarks stabilize and we have a steady-state budget per primitive.
- **`api.md`** — when the public Rust surface is large enough to warrant narrative reference beyond what rustdoc produces.
- **`migration.md`** — when we accumulate enough save-format revisions to need a per-version migration log.

New docs are additive; `spec.md` remains the root contract.

## How to use this alongside the other trees

- Adding a **language feature** (new declaration, new type) — start in `../dsl/spec.md`.
- Adding a **compiler pass** (desugaring, codegen, schema emission) — start in `../compiler/spec.md`.
- Adding a **runtime primitive** (pool, buffer, trait) — start here.

Determinism guarantees cross all three docs. If a proposed change could break replay, the relevant section lives here.
