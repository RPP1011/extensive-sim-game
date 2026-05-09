# Compiler Debug Mode — Per-Kernel Instrumentation (Phase 1)

> Goal: opt-in compiler flag with **graduated instrumentation depth** that emits GPU timestamp queries (and optionally memory traffic + DSL source mapping) so stress fixtures can attribute total wall-clock time and bandwidth to specific kernels and their DSL origins.

## Goal

Ship a `LowerOpts.debug: DebugDepth` knob with **5 graduated levels (D0-D4)**, deliberately analogous to GCC's `-O0..-O3`. Each level strictly supersets the previous; opt in per-runtime via build.rs. Encoded as a Rust enum that derives `From<u8>` so build.rs can write either `debug: 3.into()` (GCC-style) or `debug: DebugDepth::Kernel` (Rust-style):

```rust
#[repr(u8)]
pub enum DebugDepth {
    /// D0 — no instrumentation. Default. Zero overhead.
    Off         = 0,

    /// D1 — per-stage GPU timestamps. Each engine *phase* (mask, scoring,
    /// chronicle dispatch, fold, consumer) gets one timestamp pair.
    /// Coarse but cheapest — 5-10 timestamps per tick, single readback.
    Stage       = 1,

    /// D2 — stage + memory traffic. Adds host→GPU upload bytes and GPU→host
    /// readback bytes per stage. Catches "we're spending the tick on
    /// snapshot kicks not compute" pattern.
    StageMemory = 2,

    /// D3 — per-WGSL-kernel timestamps. Finer than stage — every emitted
    /// dispatch entry point gets its own timestamp pair (so 'mask'
    /// stage might split into mask_strike + mask_cleave + mask_heal).
    /// Includes everything from D2.
    Kernel      = 3,

    /// D4 — kernel timings annotated with their DSL source. Each timestamp
    /// pair carries the .sim file + line range that produced the
    /// kernel ("physics ApplyDamageFromChronicle at spy_network.sim:241").
    /// Includes everything from D3. Most expensive (string lookups
    /// + source-map table generation at compile time).
    DslMapped   = 4,
}

impl From<u8> for DebugDepth { /* clamp 0..=4, default to highest valid */ }
```

Why levels not flags: granular feature flags (timestamps_per_stage + memory_traffic + per_kernel + dsl_map as 4 separate bools) would let you pick "kernel timings without memory traffic" — but that's a 16-combination matrix nobody actually wants. Levels collapse it to 5 sensible presets in the same `-O0..-O3` shape compiler users already grok. If a real use case for unusual combos shows up, future work can add a `LowerOpts.debug_features: DebugFeatureFlags` bitset alongside the level.

When set above `Off`, the dsl_compiler-emitted Rust dispatch code interleaves `wgpu::CommandEncoder::write_timestamp` calls (Stage / Kernel) + memory traffic counters (StageMemory+) + DSL source-map table (DslMapped) appropriately. Resolves query sets after each tick, exposes the readback as a typed API on the runtime state.

The two stress drivers (`stress_agent_count_app`, `stress_cast_density_app`) extend their NDJSON to include the per-kernel breakdown. Findings doc gets a new section attributing the 16ms cast_density tick.

Phase 2 (worktree-isolated, NOT auto-merged this slice) covers the WGSL-emit-side instrumentation: per-event-kind ring histograms + mask hit rate + scoring candidate count. Those need new atomic counters inside the WGSL itself, not just host-side timestamps.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `LowerOpts.aoe_dispatch: bool` at `crates/dsl_compiler/src/lib.rs` (search) — the precedent flag for opt-in compiler behavior.
  - `wgpu::Features::TIMESTAMP_QUERY` — wgpu feature gate; device must be created with this feature enabled.
  - `wgpu::CommandEncoder::write_timestamp` — host API for marking GPU work boundaries.
  - `wgpu::Queue::get_timestamp_period()` — adapter-specific ns-per-tick conversion.
  - Per-runtime build.rs at `crates/spy_network_runtime/build.rs` — pattern for setting LowerOpts at compile time.
  - Per-tick dispatch chain: `crates/village_economy_runtime/src/lib.rs::step()` shows the ordered kernel list (mask → scoring → chronicle dispatcher → fused fold → seed_indirect).

  Search method: `rg`.

- **Decision:** extend `LowerOpts` with one `debug: DebugDepth` field (5-variant enum, `From<u8>` for GCC-style numeric setting). The compiler conditionally emits write_timestamp calls + query-set creation + readback (D1+), memory traffic counters (D2+), per-kernel granularity (D3+), and DSL source mapping (D4) based on the level. Mirrors the AOE Path B opt-in shape — one knob in build.rs, no global state.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none. (Pure host-side dispatch instrumentation; `.sim` files unchanged.)
  - Generated outputs re-emitted: per-runtime `OUT_DIR/dispatch.rs` for any runtime that opts in. Default-off, so no existing runtime regenerates.

- **Hand-written downstream code:** NONE. The dispatch chain is compiler-emitted; the timestamp instrumentation sits inside the same emitted file.

- **Constitution check:**
  - P1 (Compiler-First): PASS — instrumentation lives in compiler emit, not in hand-written kernels.
  - P2 (Schema-Hash on Layout): PASS — `LowerOpts` is a compile-time configuration knob, not part of the SoA / event / mask-predicate contract. Schema hash unchanged.
  - P3 (Cross-Backend Parity): N/A for this slice. Timestamps are GPU-only by construction; CPU backend is unaffected (stress fixtures are GPU-only anyway).
  - P4 (`EffectOp` Size Budget): N/A — no IR changes.
  - P5 (Determinism via Keyed PCG): PASS — write_timestamp is observation-only, no RNG / state mutation.
  - P6 (Events Are the Mutation Channel): PASS — the timestamp readback is a side-channel observation, not part of the replay fold.
  - P7 (Replayability Flagged): N/A — no events added.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS.
  - P10 (No Runtime Panic): PASS — graceful fallback if the adapter doesn't support TIMESTAMP_QUERY (log + return empty timings vec).
  - P11 (Reduction Determinism): N/A.

- **Runtime gate:**
  - `crates/stress_agent_count_runtime/src/lib.rs::tests::tick_advances_at_1k_agents` — already exists; verify it still passes with `debug: D1` on a follow-up build.
  - New compiler tests:
    - `debug_d0_emits_no_write_timestamp` — assert dispatch.rs string has zero `write_timestamp` calls when `debug: D0`.
    - `debug_d1_emits_per_stage_timestamps` — `>0` write_timestamp calls + matches expected stage count.
    - `debug_d3_emits_per_kernel_timestamps` — strictly more timestamps than D1 (same fixture).
    - `debug_d4_emits_dsl_source_map` — emitted source-map table contains the .sim file paths.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design.

---

## Tasks

| # | Task | Files | Description |
|---|---|---|---|
| 1 | Add `DebugDepth` enum + `LowerOpts.debug` field | `crates/dsl_compiler/src/lib.rs` (or wherever LowerOpts lives — search) | 5-variant enum (D0-D4) + `From<u8>` impl + `LowerOpts.debug: DebugDepth` field defaulting to `D0`. Pass through every place LowerOpts is threaded. |
| 2 | Compiler emits timestamp infra when D1+ | `crates/dsl_compiler/src/cg/emit/` (the dispatch.rs Rust emit, NOT WGSL) | When `debug >= D1`: emit (a) `wgpu::Features::TIMESTAMP_QUERY` request in device init helper, (b) `QuerySet { type: Timestamp }` creation sized for the level (per-stage at D1, per-kernel at D3+), (c) `encoder.write_timestamp(...)` calls bracketing each unit (stage at D1/D2, kernel at D3/D4), (d) `encoder.resolve_query_set(...)` + readback buffer + per-tick readback fn. |
| 3 | Runtime API for read-back timings + memory traffic + DSL map | `crates/dsl_compiler/src/cg/emit/` (same dispatch.rs) | Public methods on runtime state, conditional on level: `kernel_timings() -> &[(String, u64)]` (D1+), `memory_traffic() -> &[(String, MemDelta)]` where MemDelta = {host_to_gpu_bytes, gpu_to_host_bytes} (D2+), `dsl_source_map() -> &[(String, SourceLoc)]` where SourceLoc = {file: &str, line_start: u32, line_end: u32} (D4 only). All three return empty/None when level is below their threshold (compile-time constant — dead-code-eliminated when unused). |
| 4 | Stress agent_count driver integration | `crates/stress_agent_count_runtime/build.rs`, `crates/stress_agent_count_runtime/src/bin/stress_agent_count_app.rs` | Set `LowerOpts { debug: DebugDepth::Kernel, .. }` (D3) in build.rs — captures both stage + per-kernel breakdowns without paying for D4's source-map table on hot stress runs. Driver bin emits a per-kernel breakdown line every 10 ticks: `{"tick": 10, "kernel": "mask_pulse", "wall_ns": 12345, "host_to_gpu_bytes": 4096, "gpu_to_host_bytes": 0}`. |
| 5 | Stress cast_density driver integration | mirror task 4 | Same shape for cast_density. |
| 6 | Re-run sweeps + update findings | `docs/perf/2026-05-09-stress-ceilings.md` | Add a new section "Per-kernel attribution" with the actual numbers from the timestamp-enabled runs. Document where the 16ms cast_density tick goes. |

Tasks 1-3 are sequential (1 unblocks 2 unblocks 3) but tight — single agent could do all three. Tasks 4+5 are parallel (different runtime crates). Task 6 depends on 4+5.

## Phase 2 (separate worktree, NOT auto-merging this slice)

Dispatched in parallel as a worktree-isolated agent. The agent will add WGSL-side instrumentation gated under a NEW level `D5: DslMappedDeep` (or, if cleaner during implementation, a sibling `LowerOpts.debug_features: DebugFeatureFlags` bitset alongside the level). Specifically:

- **Per-event-kind ring histograms** — WGSL emit additions for atomic counter per EventKindId at producer sites; runtime API exposes `event_kind_histogram() -> [u32; N_KINDS]` after each tick.
- **Mask hit rate** — atomic counter incremented per mask kernel pass; ratio of (passed pairs) / (total candidate pairs) per mask.
- **Scoring candidate count** — running count per agent, exposed as a histogram.

Larger surface (~600-1200 LOC across compiler WGSL emit + per-runtime bindings), needs human review before landing — that's why it's worktree-only this slice. After review, lifts the level cap from D4 → D5 (or surfaces the bitset).

## Out of scope (explicitly)

- Per-buffer VRAM accounting. wgpu doesn't expose per-process VRAM; that's an adapter-extension request.
- CPU backend timing. The CPU backend doesn't dispatch; this slice is GPU-only.
- CI gate for perf budgets. User already opted out of regression discipline (per the stress test plan).
- Tracing-style structured timing (spans, parents). The output is a flat `Vec<(label, ns)>` per tick.
