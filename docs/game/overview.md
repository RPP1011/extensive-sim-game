# Overview

## What we're building

A deterministic, zero-player, Dwarf-Fortress-lineage world simulation: agents with needs, personalities, memberships, and relationships live in a 3D voxel world, form groups, pursue quests, trade, fight, propagate rumors, and leave a traceable narrative. The game is the emergent behavior of the sim — the player observes, interacts at chosen seams, and the sim continues to run whether or not they're watching.

Target scale: **20k–200k agents** on a commodity desktop, running at interactive speed (≥ 30 ticks/sec at 20k, ≥ 2 ticks/sec at 200k), with full deterministic replay from a seed.

## Layer map

```
┌─────────────────────────────────────────────────────────────────┐
│  Game content                                                    │
│  DSL source files describing this specific sim                   │
│  (wolves+humans today, full DF-scale tomorrow)                   │
│  Paths: assets/sim/*.sim, assets/sim/rules/*.sim, ...            │
└─────────────────────────────────────────────────────────────────┘
                             │ compiled by
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Compiler                                                        │
│  Reads DSL source, emits:                                        │
│    • Rust handlers (CascadeHandler impls, view update fns,       │
│       mask predicates, SoA layouts)                              │
│    • Python dataclasses + pytorch Dataset over trace format      │
│    • SPIR-V kernels for GPU backend                              │
│  Spec: compiler.md                                       │
└─────────────────────────────────────────────────────────────────┘
                             │ registers with
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Engine                                                          │
│  Primitives only: cascade runtime, SoA state, spatial hash,      │
│  event ring, RNG, tick orchestration, schema hashing.            │
│  Two backends: SerialBackend (reference), GpuBackend (perf).     │
│  Spec: runtime.md                                         │
└─────────────────────────────────────────────────────────────────┘
```

The hard rule: **engine contains zero game logic**. A balance change, a new creature, a new ability, a new quest type — none of these touch engine code. They are DSL edits. The DSL defines what the sim does; the engine only defines *how it runs*.

## The split — what goes where

| Concern | Layer |
|---|---|
| "Wolves are hostile to humans" | DSL (`entity`, `predator_prey`) |
| "Attack deals 10 damage" | DSL (`scoring` + `physics`) |
| "Stun prevents casting" | DSL (`mask` predicate) |
| "HP hits zero → agent dies" | DSL (`physics` cascade) |
| "Engagement updates every tick-start" | DSL (`physics @phase(pre)`) |
| "Cascade terminates within 8 iterations" | Engine (invariant on runtime) |
| "Agents shuffle in per-tick action order" | Engine (determinism rule) |
| "Spatial queries use 16m uniform grid" | Engine (primitive choice) |
| "Schema hash covers state + events + rules + scoring" | Both (compiler emits, engine validates) |

## Where the game currently lives (2026-04-19)

The DSL compiler does not yet exist. All game rules are hand-written Rust inside `crates/engine/src/ability/*.rs`, `creature.rs`, `policy/utility.rs`, and balance constants in `step.rs` — legacy tech debt from pre-split development.

**Plan: compiler-first.** We grow the DSL compiler incrementally, one milestone at a time. When a milestone lands, the legacy code it replaces is deleted in the same commit. No parallel hand-written emission-target crate; no parity-diff ritual. Every line of game logic that lands in the new codebase arrives through the compiler.

Sequence:

1. **Milestone 0** — compiler scaffold: empty program compiles to empty module.
2. **Milestone 1** — `event` declarations emit Event enum variants.
3. **Milestone 2** — `physics` rules emit `CascadeHandler` impls; legacy handlers retire.
4. **Milestone 3** — `mask` predicates emit validity fns.
5. **Milestone 4** — `scoring` emits utility-table rows; legacy utility backend retires.
6. **Milestone 5** — `entity` emits creature spawn templates; legacy hostility matrix retires.
7. **Milestones 6–10** — `view`, `verb`, `invariant`, `probe`, `metric`.
8. **Milestones 11–12** — Python dataclass emission, SPIR-V kernel emission.

At milestone 4, the wolves+humans scenario should be fully DSL-owned: DSL source in `assets/sim/`, compiler-emitted Rust in `crates/engine_rules/`, legacy engine handlers deleted. See `compiler_progress.md` for the live tracker, `feature_flow.md` for how each milestone lands.

The existing `world-sim` visualization keeps running on legacy code until the relevant milestone retires that slice. The interim is acceptable; the endpoint is not negotiable.

## The first end-to-end: wolves + humans

The scope anchor is the existing `world-sim` visualization binary. It runs a DF-style sim of humans and wolves with the following loop:

- Humans and wolves spawn on voxel terrain
- Wolves hunt humans (predator/prey hostility)
- Humans retreat, fight, communicate threats
- Agents have needs (hunger, thirst, rest) that drive action selection
- Groups form (families, packs); relationships and standings emerge
- Damage, death, decay events propagate through the cascade

We express this whole scenario in DSL and prove parity against the current binary. See `wolves_and_humans.md` for the source-level walkthrough.

## What's out of scope (explicitly)

- **Machine learning in the DSL.** Policy architecture, training algorithms, curriculum, reward shaping, hyperparameters — all external to the DSL. The compiler emits Python dataclasses + a pytorch `Dataset` over the trace format; training scripts live outside the DSL and consume that typed API. The in-engine NPC backend is a utility backend driven by `scoring` declarations (which are also written to traces for external reward shaping).
- **Player UI, rendering, audio.** The sim is headless and deterministic. Visualization consumes the same trace format as training. See `runtime.md` for the observability surface.
- **Mod loading at runtime.** Mods are DSL source files compiled into the artefact at build time. Per-lane handler ordering is specified in `language.md` §9.
