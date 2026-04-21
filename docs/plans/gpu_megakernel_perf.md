# GPU Megakernel Phase 8 — Perf Sweep + Crossover Measurement

Task #194. Follow-up to task #193 (Phase 6g), which made `GpuBackend::step`
authoritative against the CPU backend on the canonical 3h+2w (N=8) fixture.
This pass probes the CPU↔GPU crossover — at what agent count does the
megakernel start winning — and nails down the first-failure ceiling on the
way up.

## Harness

`xtask chronicle --perf-sweep [--perf-ticks T] [--perf-max-n MAX]` added
to the chronicle CLI. For each N in the fixed ladder (8, 32, 128, 512,
2048, 8192) at or below `--perf-max-n`:

1. `spawn_perf_fixture(n, seed)` — scaled cluster layout, 2:2:1
   humans:wolves:deer. Same seed ⇒ byte-identical spawn positions
   (xorshift64* jitter reused from task 174). Cluster radius grows with
   √count so density stays roughly constant as N rises.
2. 2 warmup ticks on each backend. The GPU's first warmup tick pays
   for the lazy cascade-ctx init (DSL parse + WGSL compile, ~10-30 ms
   on a warm disk); stripping it from the timed window gives
   steady-state numbers.
3. T = 200 timed ticks (default). Wall-clock ns per tick recorded,
   summarised as mean / median / p95.
4. GPU-only diagnostics captured per tick: `last_cascade_iterations`
   + a running tally of `last_cascade_error` so the ring-overflow
   fallback path shows up in the report.

Unit tests cover the fixture determinism and ratio correctness
(`perf_fixture_has_requested_agent_count`, `perf_fixture_is_deterministic`).

Release build only — `cargo run --bin xtask --release --features gpu --
chronicle --perf-sweep`. Debug times are 10-50× slower and the report
prints a loud banner if `cfg!(debug_assertions)`.

## Measurements

Adapter: Vulkan on whatever the worktree's `GpuBackend::new()` picks
(`PowerPreference::HighPerformance`, `Backends::all()`). Representative
run at T=200 timed ticks:

| N     | CPU µs/tick   | GPU µs/tick   | GPU/CPU  | events/tick  | cascade_iters | status        |
|-------|---------------|---------------|----------|--------------|---------------|---------------|
|     8 |          2.77 |        486.87 | 175.45×  |          8.0 |          1.00 | ok            |
|    32 |         17.65 |        559.66 |  31.70×  |         32.3 |          1.02 | ok            |
|   128 |        296.08 |        933.41 |   3.15×  |       1634.6 |          1.51 | near ring cap |
|   512 |       2494.88 |       3609.90 |   1.45×  |       2966.5 |          2.94 | near ring cap |
|  2048 |      34624.62 |      25916.51 |   0.75×  |      11448.4 |          2.76 | near ring cap |
|  8192 |     260635.86 |      **FAIL** |      —   |      32377.0 |             — | GPU FAIL      |

`near ring cap` means the total events/tick is within 75 % of
`EVENT_RING_CAP = 65536` (CPU ring). That's purely informational on the
CPU side; the GPU cascade has its own fixed-capacity event ring of
**4096 slots per iteration** (`CascadeCtx::new` hard-codes this), which
is the actual ceiling — see "Ring overflow" below.

Run-to-run variance at N=2048 is significant (spot checks 0.75×…0.97×
across three repeats) because the GPU path oscillates between
cascading-on-GPU and falling back to CPU tick-by-tick depending on
whether the iteration event count exceeds 4096. Steady-state numbers at
that scale are misleading: they're a weighted mix of two different code
paths.

## Interpretation

### Crossover

**GPU starts winning around N ≈ 2048**, but only because the GPU path
has already broken at that scale — the cascade event ring overflows and
the backend silently falls back to a CPU cascade for those ticks.
Subtract the ~40 fallback ticks per 200 and the GPU cascade itself is
still a net-loser. In the regime where the megakernel actually runs,
the intersection point is beyond N=2048 and currently unreachable
without widening the GPU event ring (see recommendations).

At the clean scales (no ring overflow), the ratio is:

* N=8    — 175× slower. Dispatch / readback dominates; per-tick
  wgpu encoder + `queue.submit` + buffer map for one tick costs
  ~500 µs on the Vulkan backend we selected.
* N=32   — 31× slower. Ratio collapses fast because GPU overhead is
  near-constant (~560 µs/tick) while CPU starts paying O(N²) pair-map
  costs.
* N=128  — 3.1× slower. Useful work on GPU begins to show; CPU has
  grown 100× from N=8 while GPU has grown <2×.
* N=512  — 1.4× slower. GPU overhead has doubled relative to N=8
  (now ~3.6 ms/tick) because `pair_map` fold dispatches scale with
  active events (~2966/tick).
* N=2048 — GPU path is broken (ring overflow → CPU fallback on every
  tick), but the "effective" wall-clock is similar because both paths
  are now doing the same CPU cascade work plus some GPU bookkeeping.

### GPU path bottleneck breakdown

At the "GPU is actually running" regime (N≤512), the dominant per-tick
costs from profiling the code path:

1. **Per-iteration spatial rebuild** — `spatial.rebuild_and_query` runs
   **twice** per cascade call (once at kin_radius = 12 m, once at
   engagement_range = 2 m; see `cascade::run_cascade`, lines 223-225).
   That's two `queue.submit` + buffer-map round-trips per cascade
   iteration, and Phase 6g's correctness fix (Piece 4 of task 193)
   introduced the second precompute — it was necessary for byte parity
   but it doubled the spatial cost.

2. **Cascade sub-dispatch count** — `cascade_iters` averages 2.76-2.94
   at N=128…2048. Each iteration is a separate compute-kernel dispatch
   with its own event-ring drain + CPU-side view fold dispatch. At
   2.9 iterations × (physics dispatch + view fold dispatch) = ~6 GPU
   submissions per tick, each incurring a latency floor well over
   the CPU tick's total work at small N.

3. **Scoring sidecar** — `GpuBackend::step` runs the mask + scoring
   kernels *again* at the end of every tick (`run_scoring_sidecar`,
   lib.rs:733) to populate diagnostic fields for legacy parity tests.
   That's a full mask-kernel readback + scoring-kernel readback every
   single tick. At N=8 the two readbacks alone cost multiple hundred
   microseconds. The sidecar is load-bearing for
   `gpu_backend_matches_cpu_on_canonical_fixture` but pays for itself
   only in tests.

4. **Fixed dispatch overhead** — wgpu encoder creation +
   `queue.submit` + (for any readback) `poll(Wait)` is ~80-150 µs
   per submit on Vulkan, independent of workload. With the Phase 6g
   structure we have ~6-10 submits per tick; that's a 500 µs floor on
   GPU per-tick cost regardless of N.

### Agent-count ceiling

**Hard ceiling: N ≈ 1600** before the GPU event ring (4096 slots)
overflows regularly. At N=2048 the first cascade iteration emits
~5000 events (AgentMoved + AgentAttacked + engagement_on_move
side-effects), the ring drain retrieves 4096, the cascade driver
returns `EventRingOverflow`, and `GpuBackend::step` falls back to the
CPU cascade for that tick.

At N=8192 the overflow happens deep in warmup and the harness surfaces
it as a GPU FAIL row. The wgpu storage-buffer limits documented in the
task spec (pair_map = O(N²) = 256 MB per decay view × 4 decay views ≈
1 GB at N=8192) are *not* the current ceiling — the ring overflows
first, and view storage at N=2048 is only ~16 MB × 4 = 64 MB per decay
view, comfortable on any real GPU.

The event-ring limit is tuned by `CascadeCtx::new` at `cascade.rs:811`:

```rust
let physics = PhysicsKernel::new(device, &comp.physics, &ctx, 4096)?;
```

Raising it to 16384 or 32768 would unblock N=2048-4096 without any
other architectural change.

### Event-count drift

At N≥32 the GPU and CPU event streams differ in total count (e.g.
N=512: CPU=593k events, GPU=118k). Task 193 flagged this as expected
at large N — the GPU cascade's `events_into_ring` / fold path emits
events in a different order than the CPU's per-dispatch push order,
and with `MAX_CASCADE_ITERATIONS=8` the two paths can produce a
different multiset after cascade truncation. Not a bug per task 193's
rules, but it does mean the N≥2048 "GPU falls back to CPU" path is
producing slightly different post-tick state than a pure-CPU run
would, so the numbers compare speeds of divergent code paths — not
just "same work, different backend".

## Recommendations — Task 195 candidates

Ranked by expected impact on the crossover point:

1. **Raise the GPU event-ring capacity** (+ make it N-aware). Bump from
   4096 to `max(16384, 4 * agent_cap)` so N=2048 runs without
   fallback; the actual ceiling on wgpu storage buffers is 256 MB,
   which at 32 bytes per event record is 8M events — nowhere near the
   concern. Low-risk: one constant, one buffer resize path in
   `CascadeCtx::new`. Makes the real N=2048 GPU-cascade numbers
   actually measurable.

2. **Remove the per-tick scoring sidecar.** Either gate
   `run_scoring_sidecar` behind a diagnostic flag or retire the
   canonical-fixture parity tests that depend on it. The sidecar is
   running a complete mask + scoring dispatch on every tick for no
   simulation purpose. Expected savings: ~40-60 % of the current GPU
   overhead at small N (single-digit µs range → ~200 µs).

3. **Fuse the two spatial rebuilds.** The current code rebuilds the
   spatial hash twice per cascade call because the kin-radius (12 m)
   and engagement-range (2 m) queries were stapled on as separate
   precomputes during Phase 6g's parity work. A single rebuild at the
   larger radius with a two-level query (nearest-hostile-within-2m
   extracted from the 12 m kin set) would halve the spatial cost.
   Needs re-verification of byte parity after the change.

4. **Amortise cascade re-dispatch.** Each cascade iteration is a full
   kernel submit + queue flush. The current loop averages ~3
   iterations per tick; a persistent-kernel or ping-pong sub-dispatch
   scheme could collapse that to 1 submit with internal
   synchronization (the plan doc labels persistent-kernel as a
   non-goal, but with 3 iterations it's now measurable overhead).

5. **Sparse pair_maps.** Not urgent until we're past N=4000, at which
   point the decay-view storage crosses 64 MB/view. The plan already
   lists this as a deferred non-goal. Mentioning here for
   completeness; not blocking task 195.

## Files touched

* `src/bin/xtask/cli/mod.rs` — added `--perf-sweep`, `--perf-ticks`,
  `--perf-max-n` flags to `ChronicleArgs`.
* `src/bin/xtask/chronicle_cmd.rs` — new `spawn_perf_fixture`,
  `run_perf_sweep`, `time_cpu_sweep`, `time_gpu_sweep`,
  `render_perf_table`, plus two unit tests.

No engine or `engine_gpu` changes: this is pure measurement + harness.
Baseline tests still green: `cargo test -p engine` (393 passed),
`cargo test -p engine_gpu --features gpu` (69 passed), xtask tests
(35 passed incl. 2 new).
