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

## Where the game currently lives

This section originally described a compiler-first migration plan dated 2026-04-19, when the DSL compiler didn't exist yet. It now exists and has substantially replaced the legacy hand-written Rust it targeted: per `compiler_progress.md`'s live milestone tracker, milestones 0–6 (`event`, `physics`, `mask`, `scoring`, `entity`, `view`) and 13 (`config`) are **done** — the wolves+humans scenario runs entirely on DSL-emitted code, not the legacy `ability/*.rs`/`creature.rs`/`policy/utility.rs`/`step.rs` handlers this section used to point at. Milestones 7–12 (`verb`, `invariant`, `probe`, `metric`, Python/SPIR-V emission) are not started.

Two structural facts from that original plan no longer hold and shouldn't be relied on if you're reading old commits or docs:

- **No `engine_rules` crate.** The plan called for compiler output landing in `crates/engine_rules/`; that crate was deleted in the Phase 7 wolf-sim wipe (2026-05-02) and never recreated. Compiler output today is emitted per-fixture, mostly into `crates/sims`' generated `GeneratedRuntime` modules (build-time only, not checked in) or, for the last couple of legacy per-fixture crates, into that crate's own `OUT_DIR`.
- **No `xtask` umbrella binary.** `cargo run --bin xtask -- compile-dsl` and `cargo run --bin xtask -- world-sim` were retired in the same wipe. See the root `CLAUDE.md` for the current build/test/run commands.

See `compiler_progress.md` for the live milestone tracker (including "what's still allowed as hand-written" and why there's no parallel bootstrap crate), `feature_flow.md` for how a milestone lands.

## The first end-to-end: wolves + humans

The scope anchor was the `world-sim` visualization binary described below; that specific binary no longer exists post-wipe, but the wolves+humans scenario itself lives on as a `.sim` fixture in `crates/sims` (parity-pinned against a committed baseline — see `crates/engine/tests/wolves_and_humans_parity.rs`). It runs a DF-style sim of humans and wolves with the following loop:

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
