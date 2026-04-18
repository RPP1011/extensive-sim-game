# World Sim Benchmark & SIMD-Targeting Design

**Date:** 2026-04-17
**Status:** Approved
**Scope:** Build a benchmark harness for the world sim that (1) identifies SIMD-optimization candidates and (2) provides a regression harness to validate SIMD changes once implemented. Does not include any SIMD implementations themselves.

## Goal

Answer two questions with data:

1. **Where does time go in the world sim?** At the per-system granularity, across populations of 2K, 10K, and 50K entities.
2. **Did a given SIMD rewrite help?** With statistical confidence, at a per-loop granularity.

The deliverable is the infrastructure to answer these questions — not the answers themselves, and not the SIMD rewrites that the answers enable.

## Non-goals

- Writing SIMD code (every SIMD rewrite is its own follow-up PR, driven by this harness)
- Optimizing the merge phase (acknowledged anti-candidate; benched only for baseline)
- GPU offload
- Multi-machine / CI benchmark integration
- Cross-arch validation (ARM vs x86)

## Architecture

Two cooperating layers, each independently runnable.

### Layer 1 — Diagnostic pass

Answers "where does time go?" by running the real sim and collecting per-system timings.

- New xtask subcommand: `cargo run --bin xtask -- bench world-sim [--scale 2k|10k|50k] [--ticks N] [--flamegraph] [--seed 42]`
- Uses the production `WorldSim::tick()` path
- Collects existing `TickProfile` sub-phase timers **and** new per-system timers
- Artifacts per run:
  - Terminal table (systems sorted by total µs, descending)
  - `generated/world_sim_bench_<scale>_<ts>.json` (structured export for diffing)
  - `generated/world_sim_bench_<scale>_<ts>.svg` (flamegraph, only with `--flamegraph`)

### Layer 2 — Regression harness

Answers "did SIMD help?" via criterion microbenchmarks on serialized fixtures.

- New workspace crate `crates/world_sim_bench/`
- Own `rust-toolchain.toml` pinning nightly (isolates `#![feature(portable_simd)]` from main `bevy_game`, which stays on stable)
- One bench file per targeted hot loop, each comparing scalar baseline vs one or more candidate backends at 2K/10K/50K
- Fixtures (`fixtures/world_{2k,10k,50k}.bin`) are committed bincode-serialized `WorldState` snapshots + merged delta buffers

### Data flow

```
Diagnostic pass ─► ranked per-system report + flamegraph
    │
    ▼
Human picks 3-5 SIMD targets (file:line)
    │
    ▼
Criterion bench added for each target (scalar baseline only)
    │
    ▼
SIMD candidate implemented in main crate behind backend enum
    │
    ▼
Criterion bench confirms/rejects the win
```

## System interface

Prerequisite refactor: introduce a `System` trait and migrate all ~170 existing systems to it before Stage 1 profiling runs.

### The trait

```rust
// src/world_sim/system.rs
pub trait System: Send + Sync {
    fn name(&self) -> &'static str;
    fn stage(&self) -> Stage;

    /// Runs one tick's worth of work for this system.
    /// Returns the count of entities/units touched this call
    /// (used for ns/entity stats — return 0 if unclear).
    fn run(&self, ctx: &mut SystemCtx) -> u32;
}

pub enum Stage {
    ComputeHigh,
    ComputeMedium,
    ComputeLow,
    ComputeGrid,
    ApplyClone,
    ApplyHp,
    ApplyMovement,
    ApplyStatus,
    ApplyEconomy,
    ApplyTransfers,
    ApplyDeaths,
    ApplyGrid,
    ApplyFidelity,
    ApplyPriceReports,
    PostApply,
}

pub struct SystemCtx<'a> {
    pub state: &'a WorldState,
    pub deltas: &'a mut DeltaBuffer,
    pub tick: u64,
    pub rng: &'a mut SimRng,
    // ... plus whatever the existing dispatch passes
}
```

Minimal surface — no `dependencies()`, no `enabled()`. Those are easy to add when needed.

### Registry

Built once at startup in `runtime.rs`:

```rust
pub fn build_registry() -> SystemRegistry {
    let mut r = SystemRegistry::new();
    r.register(ApplyMovement::new(Backend::default_for_cpu()));
    r.register(ApplyEconomy::new(Backend::default_for_cpu()));
    // ...
    r
}
```

Existing dispatch loops in `tick.rs` / `runtime.rs` become trait-dispatch loops keyed by `stage()`.

### Migration strategy

Big-bang: all ~170 systems in `src/world_sim/systems/` and the apply/compute phases refactored to `impl System` before Stage 1 starts. Executed via `/batch` for parallelism across the mechanical rewrite.

### Fidelity vs SIMD — two separate axes

- **Different fidelity = different system, different name.** `CombatHighFidelity` and `CombatLowFidelity` are two structs, both `impl System`, with different `stage()` values. Registry includes both; each dispatches at its own phase.
- **Scalar vs SIMD = same system, same name, swappable backend.** One `impl System`, with a `Backend` enum field.

### Backend dispatch — hybrid enum + generic kernel

Runtime backend selection (enum) for registry cleanliness and CPU-feature dispatch at startup. Monomorphized inner loop (generic kernel) for maximal inlining and auto-vectorization inside the hot path. The match lives **outside** the inner loop, always.

```rust
pub enum Backend { Scalar, Simd }

pub struct ApplyMovement { backend: Backend }

impl ApplyMovement {
    pub fn new(backend: Backend) -> Self { Self { backend } }
}

impl System for ApplyMovement {
    fn name(&self) -> &'static str { "apply_movement" }
    fn stage(&self) -> Stage { Stage::ApplyMovement }
    fn run(&self, ctx: &mut SystemCtx) -> u32 {
        match self.backend {
            Backend::Scalar => run_movement::<ScalarKernel>(ctx),
            Backend::Simd   => run_movement::<SimdKernel>(ctx),
        }
    }
}

#[inline]
fn run_movement<K: MovementKernel>(ctx: &mut SystemCtx) -> u32 {
    let mut touched = 0;
    for chunk in ctx.entities.chunks(K::LANES) {
        K::step(chunk);            // zero dispatch inside the loop
        touched += chunk.len() as u32;
    }
    touched
}

pub trait MovementKernel { const LANES: usize; fn step(chunk: &mut [Entity]); }
pub struct ScalarKernel;
pub struct SimdKernel;
// impls live next to the System, one file per system
```

`Backend::default_for_cpu()` always returns `Backend::Scalar` until the first SIMD implementation lands. Runtime CPU-feature detection is deferred.

## Per-system instrumentation

Gated behind the existing-but-unused `profile-systems` Cargo feature. Default builds have zero overhead.

### Timing wrapper

Single wrapper in `src/world_sim/trace.rs`:

```rust
pub fn run_with_profile(sys: &dyn System, ctx: &mut SystemCtx, prof: &mut TickProfile) -> u32 {
    #[cfg(feature = "profile-systems")] {
        let t = Instant::now();
        let touched = sys.run(ctx);
        let elapsed = t.elapsed().as_nanos() as u64;
        prof.record_system(sys.name(), elapsed, touched);
        touched
    }
    #[cfg(not(feature = "profile-systems"))]
    sys.run(ctx)
}
```

The trait-based design means no macro sprinkled across 170 files and no `stringify!` magic — `sys.name()` is the real identifier.

### Thread-local accumulators

Compute phase runs under rayon. Per-thread `SystemProfileAccumulator` (thread-local) avoids a mutex on the hot path. At end of tick, `TickProfile::fold_thread_locals()` sums them.

### New TickProfile fields

```rust
pub struct TickProfile {
    // ... existing sub-phase timers ...

    /// Populated only under `profile-systems`. Vec of (name, total_ns, call_count, entities_touched).
    pub system_timings: Vec<SystemTiming>,
}

pub struct SystemTiming {
    pub name: &'static str,
    pub stage: Stage,
    pub total_ns: u64,
    pub calls: u32,
    pub entities_touched: u64,
}
```

### Overhead estimate

170 systems × 2 `Instant::now()` calls × 400 ticks/sec ≈ 136K syscalls/sec. Each is ~20 ns on Linux (`clock_gettime(MONOTONIC)`). Total ~3 ms/sec ≈ **1 %** overhead. Acceptable under the feature flag.

## Scale control

Three populations (2K / 10K / 50K) generated reproducibly.

### Generator

New helper in `src/world_sim/runtime.rs`:

```rust
pub fn bench_world(target_entities: usize, seed: u64) -> WorldSim {
    // Scales settlement count + initial NPCs to hit target
    // after a 500-tick warm-up (current sim stabilizes at
    // 700-1300 NPCs per small world, so ~3x for 2K, 15x for
    // 10K, 75x for 50K).
}
```

Warms 500 ticks before the profiling window opens so populations are at equilibrium, not in transient growth.

### Wall-time budget

- 2K × 10K ticks ≈ 25 s (at ~400 tick/sec)
- 10K × 10K ticks ≈ 2 min (assuming ~80 tick/sec)
- 50K × 10K ticks ≈ 3 min (assuming ~50 tick/sec — best-case linear scaling)
- Full sweep + flamegraphs ≈ 10 min

### Reproducibility

Fixed seed default (`--seed 42`). JSON output tags every run with `{scale, seed, tick_window, git_sha, rustc_version, cpu_model}`.

### Fixture capture

`cargo run --bin xtask -- bench regenerate-fixtures` runs the same scaled generator but serializes post-warm-up `WorldState` + merged `DeltaBuffer` to `crates/world_sim_bench/fixtures/world_{2k,10k,50k}.bin` via bincode. A `fixtures.sha` file alongside holds a hash of the serialized `WorldState` schema (field names + types), checked on bench startup. Mismatch aborts with a clear "regenerate fixtures" message.

## Diagnostic output format

### Terminal table

Sorted descending by `total_ns`. Columns:

```
system                  stage         total_ms  calls   entities/call  ns/entity   % of tick
apply_movement          ApplyMovement    842.3    4000        2 150       97.8        18.4%
apply_economy           ApplyEconomy     612.7    4000          180    2 834.0        13.4%
apply_hp_changes        ApplyHp          487.1    4000        1 850      131.6        10.6%
...
```

`ns/entity` is the primary SIMD-target signal. A system at 50 ns/entity doing f32 arithmetic is a fat target. A system at 2 µs/entity doing HashMap lookups is not.

### JSON export

`generated/world_sim_bench_<scale>_<ts>.json`:

```json
{
  "meta": {
    "scale": "10k",
    "seed": 42,
    "tick_window": 10000,
    "warm_ticks": 500,
    "git_sha": "f10c3c64",
    "rustc_version": "1.93.1",
    "cpu_model": "AMD Ryzen ...",
    "timestamp": "2026-04-17T14:23:00Z"
  },
  "sub_phases": { "compute_us": 123, "merge_us": 45, "apply_us": 321, ... },
  "systems": [
    { "name": "apply_movement", "stage": "ApplyMovement", "backend": "Scalar",
      "total_ns": 842_300_000, "calls": 4000, "entities_touched": 8_600_000,
      "ns_per_entity": 97.9, "pct_of_tick": 18.4 },
    ...
  ]
}
```

`scripts/diff_bench.py` (follow-up, not in this plan) compares two JSONs and highlights regressions.

### Flamegraph

`scripts/perf_bench.sh` wraps `cargo flamegraph` around the diagnostic subcommand. Output tagged with scale + timestamp. Run with `--flamegraph` flag on the xtask subcommand (shells out to the script).

**Verification step (Stage 2 exit criterion):** the generated SVG must be opened in Chrome via the browser automation tools and confirmed to render readably — not just exist on disk.

## Criterion microbenchmarks

### Layout

```
crates/world_sim_bench/
├── Cargo.toml              # dev-deps: criterion, bincode; deps: bevy_game
├── rust-toolchain.toml     # channel = "nightly"
├── fixtures/
│   ├── world_2k.bin
│   ├── world_10k.bin
│   ├── world_50k.bin
│   └── fixtures.sha        # schema hash check
├── benches/
│   ├── movement.rs
│   ├── economy.rs
│   ├── hp_changes.rs
│   └── merge.rs            # baseline for anti-candidate
└── src/
    ├── lib.rs              # #![feature(portable_simd)]
    ├── fixtures.rs         # load/regen helpers
    └── candidates/         # per-loop SIMD candidates (nightly-only until they
                            # migrate into the main crate's backend enum)
```

### Per-bench shape

```rust
fn bench_movement(c: &mut Criterion) {
    for scale in ["2k", "10k", "50k"] {
        let (state, deltas) = fixtures::load(scale);
        let mut group = c.benchmark_group(format!("movement/{scale}"));
        for backend in [Backend::Scalar, Backend::Simd] {
            let sys = ApplyMovement::new(backend);
            group.bench_function(format!("{backend:?}"), |b| {
                b.iter(|| sys.run(&mut ctx_from(&state, &deltas)))
            });
        }
    }
}
```

### Summary tool

`scripts/bench_summary.py` parses `target/criterion/**/estimates.json` and emits a markdown table (scalar baseline vs each candidate, all scales, speedup ratio). This is what gets pasted into a SIMD PR description.

### First benches (Stage 3 deliverables, scalar baseline only)

1. `movement.rs` — `apply_movement` force-vector magnitude clamp + position update
2. `economy.rs` — `apply_economy` per-settlement `[f32; 8]` commodity arithmetic
3. `hp_changes.rs` — `apply_hp_changes` damage/heal/shield clamp (mixed: HashMap + arithmetic)
4. `merge.rs` — `delta::merge_deltas` baseline (anti-candidate, benched to prevent accidental pessimization)

## Phasing & deliverables

### Stage 0 — System trait refactor (~3-5 days, parallelizable via `/batch`)

- Define `System` trait, `Stage` enum, `SystemCtx` struct, `SystemRegistry`
- Migrate all ~170 systems in `src/world_sim/systems/` + apply/compute phases to `impl System`
- Refactor `tick.rs` / `runtime.rs` dispatch to iterate trait objects by `stage()`
- **Exit criterion:** `cargo test` passes; sim performance within 5 % of pre-refactor baseline

### Stage 1 — Diagnostic plumbing (~1-2 days)

- Wire `profile-systems` feature: thread-local accumulators, `run_with_profile` wrapper
- `bench_world(target_entities, seed)` generator in `runtime.rs`
- `xtask bench world-sim --scale {2k|10k|50k} --ticks N` subcommand with table + JSON output
- `xtask bench regenerate-fixtures` subcommand (emits bincode snapshots + schema hash)
- **Exit criterion:** diagnostic runs at all three scales produce ranked per-system reports with `ns/entity` for the ~20 hottest systems. Three baseline JSONs committed to `generated/baselines/`.

### Stage 2 — Flamegraph integration (~0.5 day)

- `scripts/perf_bench.sh` wrapping `cargo flamegraph` on the diagnostic subcommand
- `--flamegraph` flag on the xtask that shells out and tags output with scale + timestamp
- **Exit criterion:** SVG generated at each scale, opened in Chrome and visually verified readable. Written target list with file:line references for each SIMD candidate, saved as `docs/superpowers/specs/world_sim_simd_targets.md`.

### Stage 3 — Criterion harness (~1-2 days)

- `crates/world_sim_bench/` created with nightly toolchain pin
- Fixture loading infrastructure + schema-hash safety check
- Bench skeletons for the 3-5 targets identified in Stage 2 (scalar baseline only)
- `scripts/bench_summary.py` parser + markdown emitter
- **Exit criterion:** `cargo bench` in the bench crate produces green output at all three scales. Markdown table generates cleanly and shows reasonable variance (<5 % RSD on scalar baseline).

SIMD rewrites themselves are not part of this plan; each is a follow-up PR driven by a criterion bench.

## Open questions

None blocking. Decisions deferred to first SIMD implementation:
- Which `std::simd` lane width to target (`f32x4` / `f32x8` / `f32x16`) — depends on target CPU
- Runtime CPU-feature detection for `Backend::default_for_cpu()` — stubbed to `Scalar` until first SIMD lands
- Whether to run criterion benches in CI — deferred; criterion is noisy on shared runners

## Risks

1. **System trait refactor breaks determinism** — the sim's determinism tests are the canary. Any mismatch in iteration order or dispatch sequencing will be caught. Mitigation: run full determinism suite after each `/batch` migration slice, not just at the end.
2. **Fixtures rot fast** — `WorldState` is an evolving struct (~187 KB source file). Schema-hash check on load keeps this honest; regeneration is a one-command operation.
3. **50K scale may exceed available RAM on dev machines** — if so, drop to 30K or move the largest scale to a periodic-only run. Monitor during Stage 1.
4. **`profile-systems` overhead exceeds 5 %** — if thread-local bookkeeping is heavier than estimated, fall back to sampling (time every Nth call). Re-measure after Stage 1 wiring.
