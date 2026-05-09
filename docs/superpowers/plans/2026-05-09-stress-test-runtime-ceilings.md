# Stress Test Runtime Ceilings (one-shot characterization)

> One-shot per-fixture characterization plan: drive each runtime
> against a sweep of an axis (agent count first; later: events/tick,
> tick budget, .ability corpus size) until the runtime panics or OOMs,
> capture the per-tick wall-clock + GPU memory + emitted kernel size,
> and pin the ceiling subsystem.
>
> Findings live in `docs/perf/2026-05-09-stress-ceilings.md` (table +
> qualitative summary). This plan is the design-time companion that
> lays out the fixture+driver shape and the AIS for adding a new
> runtime crate.

## Goal

Identify the agent-count ceiling on the current dsl_compiler-emitted
mask → score → chronicle → consume per-tick chain. Driver caps the
sweep at 200_000 agents (≥10x megaswarm_10000's working size); a
panic / OOM at any cap stops the sweep and is recorded as the
breakpoint.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `EventRing::new` at `crates/engine/src/gpu/event_ring.rs:80` (constant 65536-slot capacity, used by every fixture)
  - `PackedAbilityRegistryGpu::upload` at `crates/engine/src/ability/registry_gpu.rs` (registry → GPU buffers; reused by every chronicle dispatcher fixture)
  - `dispatch_mask_verb_<name>`, `dispatch_scoring`, `dispatch_physics_<name>` per-kernel helpers in `OUT_DIR/generated.rs`/`dispatch.rs` (build.rs-emitted, one per dsl_compiler kernel)
  - `spy_network_runtime`, `megaswarm_10000_runtime` runtime crate shapes (closest two analogues — multi-verb chronicle + scaling, respectively)
  Search method: `rg` + `Read`.

- **Decision:** new `crates/stress_agent_count_runtime` crate that
  mirrors `spy_network_runtime`'s shape but with one self-target verb
  (Pulse) and a no-op consumer. Justification: characterization fixtures
  need an isolated runtime so per-tick wall-clock isn't muddied by
  multi-verb argmax cost or cascade-driven population shrinkage. Reusing
  `spy_network` would require a second `agent_count` parameter and a
  second cascade-disabled mode — more surface than a fresh per-fixture
  crate.

- **Rule-compiler touchpoints:**
  - DSL inputs added: `assets/sim/stress_agent_count.sim` + `assets/ability_test/stress_agent_count/Pulse.ability`
  - Generated outputs emitted: `crates/stress_agent_count_runtime/{build.rs → OUT_DIR/*.{wgsl,rs}}` — a fresh per-fixture build artifact set; no other runtime crate's generated code changes.

- **Hand-written downstream code:**
  - `crates/stress_agent_count_runtime/src/lib.rs` — runtime state +
    per-tick dispatch chain. Mirror of `spy_network_runtime/src/lib.rs`
    minus the multi-verb / cascade behavioral wiring. The
    `pack_agents` / `unpack_agents` / `kick_snapshot` /
    `upload_sim_cfg` kernels are emitted but not invoked — same as
    every existing fixture (these are scheduler hints, not contracts).
  - `crates/stress_agent_count_runtime/src/bin/stress_agent_count_app.rs` — driver binary. NDJSON output + `catch_unwind` per step.

- **Constitution check:**
  - P1 (Compiler-First): PASS — every WGSL kernel is dsl_compiler emit; no hand-written kernels.
  - P2 (Schema-Hash on Layout): PASS — engine schema unchanged. Verified by `cargo test -p engine --test schema_hash`.
  - P3 (Cross-Backend Parity): N/A — this fixture is GPU-only; CPU oracle exists in `apply_program` but no parity assert here (parity coverage is the apply_ability_*_runtime crates' job).
  - P4 (EffectOp Size Budget): N/A — no new EffectOp variants.
  - P5 (Determinism via Keyed PCG): PASS — agent positions are seeded via `engine::rng::per_agent_u32_pcg(seed_lo, agent_id, 0, purpose)`. No `thread_rng` anywhere.
  - P6 (Events Are the Mutation Channel): PASS — Pulse's only effect is EffectSelfDamageApplied (chronicle event); the consumer rule's `let _ = agents.hp(a)` is a read, not a write.
  - P7 (Replayability Flagged): PASS — every event in the .sim is `@replayable @gpu_amenable`.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — see Tasks section.
  - P10 (No Runtime Panic): PASS — driver wraps every step in `std::panic::catch_unwind` so OOM/panic surfaces as a recorded breakpoint, not an unrecoverable abort.
  - P11 (Reduction Determinism): N/A — no reductions.

- **Runtime gate:**
  - `tick_advances_at_1k_agents` at `crates/stress_agent_count_runtime/src/lib.rs::tests` — drives 1 tick at agent_cap=1000 and asserts `state.tick == 1` plus `event_tail > 0` (chronicle records were emitted).

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design.

## Tasks

1. **Fixture A — agent_count sweep (1k → 200k).** Build the `.sim` +
   `.ability` + runtime crate + driver bin, verify the runtime gate
   test passes, run the sweep, fill in the findings table.
   Status: **DONE** (this plan execution; results in
   `docs/perf/2026-05-09-stress-ceilings.md`).

3. **Driver bin shape.** NDJSON per-tick + summary, `catch_unwind` per
   step. Status: **DONE** (`src/bin/stress_agent_count_app.rs`).

5. **Findings doc.** Markdown table + qualitative summary +
   recommended raise candidates. Status: **DONE**
   (`docs/perf/2026-05-09-stress-ceilings.md`).

(Tasks 2 + 4 reserved for follow-on fixtures: events/tick sweep + tick
budget sweep. Not in scope for this initial plan execution — the
agent_count axis was the critical first signal.)
