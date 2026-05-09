# Stress Test — Runtime Ceilings (one-shot characterization)

> Goal: push two runtime dimensions until something breaks (OOM / timeout / divergence / overflow / panic), capture the actual ceiling, document. No CI gate; one-shot only.

## Goal

Find the **current** breaking points along two runtime axes:

1. **Agent count** — sweep 1k → 4k → 16k → 64k → 200k → push past until something breaks. Per-tick wall clock + peak memory + per-tick throughput at each scale. Identify which subsystem fails first (GPU upload, pack/unpack, dispatcher kernel, mask/scoring fusion, snapshot serialization).
2. **Cast rate / AOE density** — every-agent-casts-every-tick at maximum AOE candidate density (tightly-packed agents in a small region). Chronicle ring high-water mark, Spread bitonic-sort wall time, kept-set utilization, ring overflow if any.

Output: a markdown table per dimension at `docs/perf/2026-05-09-stress-ceilings.md` recording the actual numbers + the subsystem that broke first.

## Architectural Impact Statement

- **Existing primitives searched:**
  - Existing perf instrumentation: `cargo build` `emit-stats` warnings via `crates/dsl_compiler/build.rs` — only WGSL kernel byte sizes today, no per-tick wall clock or memory.
  - `crates/spy_network_runtime/src/lib.rs` — closest analog of a "drive a runtime from a test"; behavioral pin only, no metrics.
  - `crates/village_economy_runtime/src/lib.rs::step()` — explicit per-tick step driver; uses every backend subsystem via the apply_ability dispatcher.
  - `SimState::new(agent_cap, seed)` at `crates/engine/src/state/mod.rs:267` — the entry point; `agent_cap` parameter is the lever for the agent-count axis.
  - `SpatialHash::new(agent_cap)` at `crates/engine/src/state/mod.rs:363` — packing density for the AOE axis is set via initial agent positions.
  - `MAX_SPREAD_CANDIDATES = 256` (post PR #39 bitonic sort bump) — the cap that AOE density would saturate.
  - Chronicle ring cap = 65536 slots per dispatch — `if (_slot < 65536u)` gate in `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`. Overflow today silently drops events past the cap.

  Search method: `rg`.

- **Decision:** new — two new fixture+runtime crates (`stress_agent_count_runtime`, `stress_cast_density_runtime`) plus a stand-alone driver bin per crate that runs the stress at increasing scale and writes metrics to stdout. Mirrors the existing `crates/*_runtime/src/lib.rs::tests` pattern but with binary entry points so the harness can be invoked outside cargo test (no test timeout, no parallel-test contention).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `assets/sim/stress_agent_count.sim`, `assets/sim/stress_cast_density.sim`, `assets/ability_test/stress_*/`*.ability` (new fixtures).
  - Generated outputs re-emitted: per-runtime `OUT_DIR` WGSL kernels (build.rs auto-regenerates).

- **Hand-written downstream code:** NONE. The two new runtimes use the same compile-from-`.sim` pattern as every other runtime.

- **Constitution check:**
  - P1 (Compiler-First): PASS — fixtures use the standard `.sim` → `dsl_compiler` → WGSL emit path; no hand-written kernels.
  - P2 (Schema-Hash on Layout): N/A — no SoA / event / mask-predicate changes; engine `.schema_hash` should not move.
  - P3 (Cross-Backend Parity): N/A for this slice. The stress runtimes can run GPU-only; cross-backend determinism is in scope for a separate slice the user explicitly skipped.
  - P4 (`EffectOp` Size Budget): N/A — no new EffectOp variants.
  - P5 (Determinism via Keyed PCG): PASS — fixtures will use seeded PCG, no thread_rng. Determinism only needs to hold within a single GPU run for stress sweeps to be reproducible.
  - P6 (Events Are the Mutation Channel): PASS — all sim mutation through the existing event path.
  - P7 (Replayability Flagged): PASS — events declared with `@replayable` per the existing pattern.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task closes with a commit SHA on the active branch.
  - P10 (No Runtime Panic): PASS — the stress *findings* may include panics (that's the point), but the stress harness itself catches them rather than aborting.
  - P11 (Reduction Determinism): PASS — bitonic sort + AgentId tie-break is the existing convention; stress tests run against it.

- **Runtime gate:** every task that wires up a stress fixture must run the fixture for ≥1 tick and assert tick advance + non-zero event emission. Compile-clean is not runtime-clean (see the 2026-04-28 incident in the AIS template).
  - `crates/stress_agent_count_runtime/src/lib.rs::tests::tick_advances_at_1k_agents` — drive 1 tick at agent_cap=1000, assert `state.tick == 1` and at least one chronicle event emitted.
  - `crates/stress_cast_density_runtime/src/lib.rs::tests::tick_advances_under_max_density` — drive 1 tick with all agents in a single 27-cell region, assert tick advance + Spread sort walked at least once.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

---

## Fixtures

### Fixture A: `stress_agent_count`

Purpose: isolate the agent-count axis. One verb that fires every tick on every alive agent — minimal sim mechanics so per-tick cost is dominated by infrastructure (pack/unpack, dispatcher, fusion, spatial hash, chronicle).

```
verb Pulse(self) =
  action PulseAction
  when self.alive
  apply_ability 1 by self target self
  score 1.0

// Pulse.ability:
ability Pulse {
    target: self
    cooldown: 100ms
    hint: utility
    self_damage 0.0  // no-op effect — just exercises dispatcher path
}

@phase(post)
physics ApplyPulseFromChronicle {
  on EffectSelfDamageApplied { actor: _, target: _, amount: _ } {
    // no-op consumer; forces the chronicle dispatcher emit path to run
  }
}
```

Stress sweep: agent_cap ∈ {1k, 4k, 16k, 64k, 200k}. Run 100 ticks at each scale. Record per-tick wall clock (median + p99), peak GPU memory, peak host memory, total dispatcher kernel WGSL size. If a scale OOMs / panics / blows the WGSL stack, record that as the breakpoint and stop the sweep.

### Fixture B: `stress_cast_density`

Purpose: maximize cast pressure × AOE candidate density. All agents start packed into a single 27-cell hash bin (radius ~1.5 cells). Every agent casts an AOE-Spread ability every tick targeting itself. The Spread shape walks the 27-cell neighborhood and picks `count` closest victims — at maximum density every cell in the spread cap is occupied.

```
ability AoePulse {
    target: self
    area: spread { count: 256 }   // hits the full Spread cap
    cooldown: 100ms
    hint: damage
    damage 0.0
}
```

Stress sweep: agent_cap ∈ {1k, 4k, 16k, 64k} (200k all packed into 27 cells is geometrically impossible — record what cap fits before agents must spread to larger region). Run 100 ticks. Record:
- Chronicle ring high-water mark per tick (overflows = events silently dropped past the 65536 cap).
- Spread bitonic-sort wall time per dispatch (256-element sort × N parallel casts).
- Kept-set fill rate (ratio of kept-set `count` to actual in-radius candidates).
- Ring overflow events (count of dispatches where slot ≥ 65536u).

If the chronicle ring saturates, that's a genuine ceiling — record cast-rate-per-tick at saturation as the cast-rate ceiling.

## Harness

Each fixture's stand-alone driver bin runs the sweep and writes NDJSON to stdout, one record per tick:

```json
{"agent_cap": 1000, "tick": 0, "wall_clock_us": 1234, "peak_mem_mb": 45, "ring_high_water": 1000, "spread_sort_us": 0}
```

A small Python or Rust post-processor aggregates the NDJSON into a markdown table at `docs/perf/2026-05-09-stress-ceilings.md`. Median + p50/p99 per (fixture, agent_cap).

GPU memory measured via `wgpu::Device::poll_for_unmaintained_buffers` then read back from process RSS or `nvidia-smi` (linux/cuda) / `vk_mem_alloc` reporting. Pick the simplest available; this is a one-shot characterization, not a continuous metric.

## Tasks

| # | Task | Files | Description |
|---|---|---|---|
| 1 | Build `stress_agent_count_runtime` foundation | `crates/stress_agent_count_runtime/`, `assets/sim/stress_agent_count.sim`, `assets/ability_test/stress_agent_count/Pulse.ability`, workspace `Cargo.toml` | Mirror `spy_network_runtime`'s shape; agent_cap parameterized via lib fn; behavioral pin tick_advances_at_1k_agents. |
| 2 | Build `stress_cast_density_runtime` foundation | `crates/stress_cast_density_runtime/`, `assets/sim/stress_cast_density.sim`, `assets/ability_test/stress_cast_density/AoePulse.ability` | Same shape as task 1 but seeded with all agents in one 27-cell region; uses `area: spread { count: 256 }`. |
| 3 | Add `stress_agent_count_app` driver bin | `crates/stress_agent_count_runtime/src/bin/stress_agent_count_app.rs` | Reads agent_cap from CLI arg; runs 100 ticks; emits NDJSON metrics per tick. Wraps step in `catch_unwind` so panics are recorded, not lost. |
| 4 | Add `stress_cast_density_app` driver bin | `crates/stress_cast_density_runtime/src/bin/stress_cast_density_app.rs` | Same pattern as task 3 plus chronicle ring high-water + Spread sort wall time per dispatch. |
| 5 | Run sweep — agent count | (no code; produces `docs/perf/2026-05-09-stress-ceilings.md`) | Run task-3 binary at agent_cap ∈ {1k, 4k, 16k, 64k, 200k}; capture stdout to JSONL; aggregate into markdown table; record breakpoint subsystem. |
| 6 | Run sweep — cast density | (no code; appends to `docs/perf/2026-05-09-stress-ceilings.md`) | Same for task-4 binary; record ring saturation cast-rate ceiling. |
| 7 | Write findings + open follow-up tasks | `docs/perf/2026-05-09-stress-ceilings.md` (final summary section); new tasks in TaskWarrior for each ceiling worth raising | Document the actual ceilings; for each "could be raised", file a task with the candidate fix (e.g. "raise chronicle ring cap from 65536 to 1M" or "switch spatial hash from u32 to u64 for >200k agents"). |

Tasks 1+2 can run in parallel (different new crates, different .sim files, different .ability files — no shared file conflicts). Task 3 depends on task 1; task 4 depends on task 2. Tasks 5+6 depend on tasks 3+4 respectively. Task 7 depends on tasks 5+6.

## Out of scope (explicitly)

- Compile-time ceilings (WGSL emit size, EffectOp count, kernel binary size). User skipped this dimension.
- Determinism / cross-backend parity at scale. User skipped this dimension.
- CI gate for any of the above metrics. User picked one-shot characterization, not regression discipline.
- Raising any ceiling. This plan only finds them.
- Fancy charting / web dashboards. Markdown table is the deliverable.
