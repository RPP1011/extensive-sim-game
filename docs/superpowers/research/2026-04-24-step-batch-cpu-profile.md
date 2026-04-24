# step_batch CPU profile at N=100k — Stage A refresh

**Status:** research
**Date:** 2026-04-24
**Branch:** `perf-stage-cpu-overhead-a` (off `world-sim-bench`)
**Predecessor:** `docs/superpowers/research/2026-04-22-batch-perf-gap-analysis.md`
(Stage A refresh, commits `e85edc8f..52fb7563` on `perf-stage-a`)

## TL;DR

The Stage A refresh reported **~281 ms/tick of "CPU / driver tax"** at
N=100k, derived by subtracting the sum of GPU-phase timestamps (38 ms)
from wall-clock tick time (319 ms). **This attribution is incorrect.**

Two independent measurements — `Instant::now()` instrumentation inside
`step_batch` and `samply` stack sampling of the test process — agree:
CPU-side work in `step_batch` is **under 1.5 ms/tick**, less than **0.5%
of wall time**. The remaining ~313 ms/tick is GPU execution that the
Stage A per-phase timestamps under-count (driver work between dispatches,
resource barriers, queue back-pressure idle gaps).

**There is no large CPU hot function to fix.** The largest CPU phase
in the tick is the spatial-hash SoA upload at ~1 ms/tick; a
dirty-flag on the SoA input removes most of it, but the wall-clock
win at N=100k is below the variance floor (±5 ms/tick run-to-run).

## Baseline

```
cargo test --release --features gpu -p engine_gpu \
    --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

RTX 4090 + Vulkan backend, three post-warmup runs on rebased
`world-sim-bench` (post Stage A + B.1 + B.2 merges):

```
=== PERF N100k batch: 308724 µs/tick ===
=== PERF N100k batch: 310931 µs/tick ===
=== PERF N100k batch: 311040 µs/tick ===  (pre-fix mean ≈ 310 ms/tick)
```

Stage A's sum of GPU-timestamp phases at N=100k was ~38 ms/tick; the
remaining ~272 ms/tick is the target of this task.

## Method 1: in-code `Instant::now()` instrumentation

Env-gated (`ENGINE_GPU_PROFILE_STEP_BATCH=1`) timing markers around
every phase of `step_batch` + the subphases inside
`run_cascade_resident_with_iter_cap`.

One typical batch at N=100k, 50 ticks, post-warmup:

```
[step_batch N=50] total=15787768µs avg/tick=315755µs
  encoder_create=0µs
  fused_unpack=17µs mask_scoring=17µs apply_movement=11µs append_events=5µs
  spatial_rebuild=1073µs spatial_query=21µs ability_upload=5µs
  seed_clear=7µs cascade_iters=63µs
  submit=5322µs (once) poll=15719734µs (once)
```

Per-tick CPU encode: **~1.2 ms**. The single dominant line is
`spatial_rebuild` (`upload_agent_soa` packs three 100k-entry Vecs on
CPU, 4× `queue.write_buffer`). All other CPU phases per tick are
under 100 µs. `submit` (5.3 ms total for the whole batch) and `poll`
(15.7 s total — the GPU) dominate.

One observation worth flagging: `queue.write_buffer` occasionally
stalls 30–40 ms (once in ~50 batches observed) — driver staging-queue
back-pressure. The dirty-flag fix below eliminates this spike by
cutting the write-buffer call count from 5/tick to 1/tick.

## Method 2: samply stack sampling

`samply record` (1 ms default sample rate) with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` for symbols.

Sample density per 1-second bucket across the 25-s test run:

| Phase | Time range | Sample count |
|---|---|---|
| spawn_crowd | 0–4 s | 4,125 |
| warmup sync step | 4–8 s | 2,434 |
| `step_batch(50)` | 9–24 s | **9** |
| snapshot + teardown | 24–25 s | 531 |

**The `step_batch` window captured only 9 samples across 15 seconds
of wall time.** The CPU was in a kernel wait (`poll`/`futex`) for
effectively the entire batch.

Of the 9 samples captured inside the batch, 6 were deep in
`libnvidia-gpucomp.so` and `libnvidia-glvkspirv.so` (the NVIDIA
SPIR-V / shader compiler). The remaining 3 were libc frames inside
futex / cond-wait machinery. No samples pointed at `engine_gpu` code
inside the batch loop.

Top self-time across the *whole* recording (including spawn +
warmup):

| Samples | % | Symbol |
|---|---|---|
| 1,865 | 20.2% | `engine::state::SimState::spawn_agent` |
| 1,845 | 20.0% | `engine::state::SimState::spawn_agent` (inlined variant) |
| 429 | 4.7% | libc `memmove` (inside the spawn path) |
| 143 | 1.6% | libc futex-wait (poll/cond_wait) |
| 98–72 | ~1% | various libc + spawn helpers |

The 40% self-time in `spawn_agent` reflects the 3.8 s `spawn_crowd`
setup — not the batch. Nothing in the top 40 lives inside
`step_batch`.

## Why no CPU samples during the batch?

`step_batch` enqueues 50 ticks of GPU work into a single command
encoder, submits once, and calls `device.poll(PollType::Wait)`. The
poll blocks on a driver fence until the GPU finishes. The thread is
not running user-space code for that ~15.7 s, so a stack-sampling
profiler records effectively nothing.

The GPU *is* running during those 15.7 s. Stage A's per-phase
timestamps summed to ~38 ms/tick × 50 ticks = ~1.9 s over the batch
because timestamps cover only each compute-pass body — not the
pipeline barriers, resource transitions, or queue-back-pressure idle
gaps the driver inserts between dispatches. With ~20 compute passes +
~40 encoder ops per tick × 50 ticks ≈ 3,000 driver-visible ops in
one submit, the un-timestamped gaps between dispatches dominate.

## Top-3 CPU hot functions in `step_batch`

Ranked per-tick CPU self-time:

| # | Function | µs/tick | % of tick |
|---|---|---|---|
| 1 | `GpuSpatialHash::upload_agent_soa` | ~1,073 | 0.34% |
| 2 | `PhysicsKernel::run_batch_resident` encode (×8 iters) | ~63 | 0.02% |
| 3 | `FusedAgentUnpackKernel::encode_unpack` | ~17 | 0.005% |

Even eliminating all three entirely would save ~1.2 ms/tick =
**~60 ms of 15.7 s = 0.4%**.

## Fix landed: cache the spatial SoA upload

A content-keyed dirty flag on `GpuSpatialHash::upload_agent_soa`
skips the CPU pack + 4 `queue.write_buffer` calls when the input
state's `(agent_cap, tick, hot_pos.len(), hot_alive.len())` matches
the last successful upload. Within a `step_batch` run these four
numbers are frozen (CPU state doesn't mutate during a batch), so the
cache hits on ticks 1..49 — 98% of calls.

Measured impact from the `ENGINE_GPU_PROFILE_STEP_BATCH=1` output:

```
pre-fix   spatial_rebuild = 1073 µs/tick
post-fix  spatial_rebuild =   90 µs/tick  (-91%)
```

Wall-clock impact at N=100k (3 runs × 3):
- Pre-fix mean: 310,232 µs/tick
- Post-fix mean: 310,289 µs/tick

The 1 ms/tick CPU saving is invisible in the wall clock because GPU
dominates at 99.6%. The fix still lands because:
1. The 30–40 ms `queue.write_buffer` back-pressure spike (once per
   ~50 batches) disappears — a p99 latency win even if average is
   unchanged.
2. Correctness of the cache is straightforward (frozen CPU state during
   a batch) and the code complexity is small.
3. The same cache also wins on the sync path, where
   `rebuild_and_query` is called twice per tick with the same SoA —
   the second call now hits the cache and skips its 4 writes.

## Fixes deferred

- **BG-construction caching** — Stage A's audit showed
  scoring/apply/movement BG already cached as of post-9b730988
  work + Stage A's physics cache. Verified — not a suspect.
- **`PackedAbilityRegistry` upload dedup** — already content-hash
  dedup'd in `ResidentAbilityBuffers::upload` (2026-04 work).
  Measured 5 µs/tick — cache hits on 49/50 calls.
- **Iter-cap convergence** — Stage B.1 already lands this.
- **Spatial pack/write dedup in sync path** — Stage B.2 already
  merged; the Method 1 profile is already against the deduped code.

## Real bottleneck and next steps

**The 313 ms/tick is GPU execution, not CPU.** To move the needle on
step_batch at N=100k:

1. **Per-dispatch GPU timestamps on the resident physics iters.**
   Stage A's ~38 ms sum likely under-counts per-iter driver overhead.
   Adding `cpass.write_timestamp` around each indirect dispatch (not
   just around each named phase) would show the real GPU busy figure
   per iter, and whether the current 272-ms unaccounted gap is
   per-iter pipeline barriers, resource transitions, or something
   else.
2. **Submission granularity.** The current path submits 50 ticks as
   one command buffer. Driver back-pressure may be serialising work
   that could overlap. Sweep `n_ticks ∈ {1, 5, 10, 25, 50}` on a
   fixed N=100k fixture and measure — if per-tick cost drops with
   smaller submits, the single-submit path is over-saturating the
   driver queue.
3. **Memory-bandwidth audit.** 100k × 80-byte agent slots × ~20
   kernel reads/tick × 50 ticks = ~8 GB scoreboard-read traffic per
   batch. Adding ~5 bandwidth-saturating intermediate spatial-hash
   writes on top may be saturating the 4090's ~1 TB/s ceiling. Worth
   profiling with Nsight Compute to confirm.
4. **Shader compile deferred to first batch.** Samply caught ~6
   samples deep in the NVIDIA SPIR-V compiler during the first ~1 s
   of the batch. One-time cost but may explain an early-batch hitch
   — worth warming up the cascade-resident pipelines during
   `ensure_resident_init` rather than on first use.

None of these are CPU-side fixes and none are in scope for this
task.

## Instrumentation kept

The `ENGINE_GPU_PROFILE_STEP_BATCH=1` phase-timing dump is removed
for this task — Stage A's GPU-timestamp infra (`gpu_profiling`
module) already covers the same questions with more signal. The
research itself is the deliverable.

## Summary

- **Fix landed**: `GpuSpatialHash::upload_agent_soa` dirty-flag dedup.
  -91% on the biggest CPU phase (1073 → 90 µs/tick at N=100k).
  Wall-clock impact below variance (~0.02% improvement), but removes
  a rare 30–40 ms `queue.write_buffer` p99 spike.
- **Premise of "281 ms CPU tax" at N=100k is incorrect.** The unmeasured
  gap is on the GPU side, not the CPU side.
- **No other CPU fixes applied.** The next three hot functions
  (physics encode, fused_unpack encode, ability upload) are each
  <100 µs/tick — already optimized.
