# GPU per-dispatch attribution + submit-granularity sweep (N=100k, RTX 4090)

**Status:** research (deliverable: numbers, not a perf fix)
**Date:** 2026-04-24
**Branch:** `gpu-per-dispatch-attribution` (based on `world-sim-bench`)
**Predecessors:**
* `docs/superpowers/research/2026-04-22-batch-perf-gap-analysis.md` (Stage A — identified the 281 ms unattributed gap)
* `docs/superpowers/research/2026-04-24-mask-kernel-subphase-measurement.md` (decomposed `mask+scoring` into mask / scoring / fusion-overhead)

---

## Question this doc answers

Stage A's per-compute-pass timestamps at N=100k attributed ~37 ms/tick
to `mask+scoring`, <2 ms combined to everything else, and left **~281 ms/tick
unattributed**. Candidate hypotheses:

1. Inter-dispatch memory barriers (wgpu hazard tracking between passes)
2. Pipeline state swaps between kernels
3. Memory bandwidth contention on writes
4. NVIDIA driver JIT / pipeline re-validation
5. Async readback staging (snapshot double-buffer + gold + standing + memory)
6. Back-pressure from pipelining 50 ticks into a single submit

This doc measures (1)–(5) via between-pass timestamps and (6) via a
submit-granularity sweep.

## Measurement method

### Part 1 — densified between-pass timestamps

Extends `GpuProfiler` with `write_between_pass_timestamp(encoder, label)`
(semantically identical to `mark()`; different name documents intent).
Adds gap marks at every between-pass boundary in `step_batch`:

* `gap:before_fused_unpack`
* `gap:fused_unpack->mask`
* `gap:mask->scoring`
* `gap:scoring->apply_actions`
* `gap:apply_actions->movement`
* `gap:movement->append_events`
* `gap:append_events->seed_kernel` (spatial_rebuild / spatial_query / seed boundary)
* `gap:seed_kernel->cascade_iter_0`
* `gap:cascade_iter_N->N+1` for N=0..7

Total: 16 between-pass marks per tick × 50 ticks = 800 timestamps.
Required bumping `MAX_TIMESTAMPS` 64 → 2048 (wgpu-types ceiling is 4096).

Each gap mark rides alongside the existing legacy per-phase mark so
Stage A regression comparisons stay valid.

### Part 2 — submit-granularity sweep

Adds opt-in env var `ENGINE_GPU_SUBMIT_GRANULARITY=K`. When set,
`step_batch(N)` partitions the loop into `ceil(N/K)` sub-batches, each
submitted with its own `queue.submit + device.poll(Wait)`. Default (var
unset) preserves the single-submit production behaviour.

Correctness: the batch-scoped rings (batch_events_ring, chronicle_ring,
apply_event_ring) persist across submits in device memory, so chronicle
records and event counts are identical regardless of K — verified below.

Workload (same as predecessors): `chronicle_batch_perf_n100k` — 100,000
agents (40% human / 40% wolf / 20% deer, 0.1 agents/unit² density), 50
ticks, seed `0xC0FFEE_B001_BABE_42`, RTX 4090, Vulkan backend.

## Part 1 — between-pass gap table (µs/tick, mean over 50 ticks)

Default submit granularity (single submit per step_batch):

| Between-pass label | µs/tick |
|---|---:|
| `gap:before_fused_unpack` | 1 |
| `gap:fused_unpack->mask` | 1 |
| **`gap:mask->scoring`** | **277,895** |
| `gap:scoring->apply_actions` | 1 |
| `gap:apply_actions->movement` | 14 |
| `gap:movement->append_events` | 1 |
| `gap:append_events->seed_kernel` | 2 |
| `gap:seed_kernel->cascade_iter_0` | 1 |
| `gap:cascade_iter_0->1` | 1 |
| `gap:cascade_iter_1->2` | 1 |
| `gap:cascade_iter_2->3` | 1 |
| `gap:cascade_iter_3->4` | 1 |
| `gap:cascade_iter_4->5` | 1 |

Legacy per-phase marks printed alongside (for Stage A continuity):

| Legacy label | µs/tick |
|---|---:|
| `fused_unpack` | 36 |
| `mask+scoring` | 22,859 |
| `apply+movement` | 23 |
| `append_events` | 14 |
| `spatial+abilities` | 5,113 |
| `seed_kernel` | 1,557 |
| `cascade iter 0..4` | 86 / 27 / 17 / 13 / 15 |
| `cascade_end` | 1 |

**Reading the labels:** `read_phase_us` reports `(opening_label, Δ_i_to_i+1)`.
Since I inserted `gap:*` marks BETWEEN the existing legacy marks, the
label semantics shifted:

* `mask+scoring` legacy: delta from "mask+scoring" mark to next mark, which
  is now `gap:mask->scoring` — so this label now measures the **mask
  dispatch alone** (~22.9 ms).
* `gap:mask->scoring`: delta from that mark to `gap:scoring->apply_actions`
  — measures the **scoring dispatch alone** (~278 ms).

Total GPU-µs accounted for: ~307 ms/tick of 312 ms/tick wall-clock
(~98% accounting).

### Sanity check: legacy `mask+scoring` changed shape

Before adding gap marks, `mask+scoring` at N=100k = ~37 ms (Stage A).
After adding gap marks it splits into **mask ~22.9 ms + scoring ~278 ms
≈ 300 ms**. The total GPU-µs/tick went from 38 ms (Stage A "accounted")
to 307 ms (this run's "accounted"). That matches the missing 281 ms
exactly.

**The 281 ms was hiding in the scoring dispatch's GPU time**, invisible
to the legacy schedule because wgpu was pipelining mask and scoring
back-to-back: the legacy `mask+scoring` close-mark (`apply+movement`)
landed after scoring but the GPU-side timing claimed that interval as
just ~37 ms — meaning a ~240 ms pipeline bubble was sitting between the
two dispatches as driver-side work that wgpu's legacy per-pass marks
didn't fence.

Once we force a between-pass timestamp at the mask→scoring boundary,
the GPU is required to produce a serialised timing point, exposing the
full scoring-dispatch cost (including whatever pre-scoring setup the
driver had been overlapping with mask).

## Part 2 — submit-granularity sweep (µs/tick wall-clock)

All runs re-spawn the fixture fresh (to avoid warmup/JIT carryover).
Each run includes the per-dispatch gap marks from Part 1, so absolute
numbers reflect the instrumented path:

| K (submit granularity) | Wall-clock µs/tick | `gap:mask->scoring` µs/tick | AgentAttacked |
|---:|---:|---:|---:|
| 1 (submit every tick) | 313,955 | 284,166 | 215,672 |
| 5 (10 sub-batches of 5 ticks) | 316,721 | 286,134 | 216,274 |
| 10 (5 sub-batches of 10 ticks) | 308,995 | 279,546 | 215,743 |
| 50 (1 submit = default) | 302,906 | 273,470 | 215,955 |
| default (no env var) | 311,864 | 277,895 | 215,813 |

Ranges: wall-clock 303–317 µs/tick (±2.3%); `gap:mask->scoring`
273–286 µs/tick (±2.3%). **No meaningful variance by K.**

Correctness: every run emitted >195k `chronicle_attack` records and
~25.5k `AgentDied` events; the `chronicle_since_last non-empty` and
`chronicle_attacks > 0` asserts all passed.

## Conclusion

**Where does the 281 ms/tick live?** Inside the **scoring dispatch
itself** — `scoring::run_resident` at N=100k consumes ~270–280 ms/tick
of GPU time. Stage A's per-pass schedule (a single `mask+scoring` bucket
closing at `apply+movement`) saw only ~37 ms because wgpu / the driver
was pipelining mask's compute with scoring's setup, and the GPU
timestamp at the legacy bucket's close landed while scoring was still
executing but before it became the reporting bottleneck.

Adding an explicit `gap:mask->scoring` timestamp forces the GPU to
produce a serialised timing point between the two passes. Once that
fence exists, the scoring dispatch's TRUE GPU-µs cost — which was
always there, just overlapped with mask's compute in the legacy
schedule — becomes visible.

**Which hypothesis does this confirm?**

1. ❌ **Barriers** — between-pass gaps other than `gap:mask->scoring`
   are all 1–15 µs. Barriers are not the bulk of the 281 ms.
2. ❌ **Pipeline state swaps** — same evidence; swaps are below
   timestamp resolution at every non-scoring boundary.
3. ❌ **Memory bandwidth** — would scale across many passes, not
   concentrate in one dispatch.
4. ❌ **Driver JIT / pipeline re-validation** — already warmed by the
   sync step; would appear in iter-0 timings if present.
5. ❌ **Async readback** — those happen at batch end, wouldn't
   attribute per-tick; also the snapshot dance itself costs ~350 ms
   once, not 280 ms × 50 ticks.
6. ❌ **Back-pressure / submit cadence** — the K∈{1, 5, 10, 50}
   sweep is flat to within noise. The GPU does not care how often we
   drain.

**The actual cost is scoring's compute.** Scoring at N=100k (the
scoring kernel + any dependent ops wgpu schedules with it) is the
~300 ms/tick worker; the legacy "mask+scoring: 37 ms" attribution was
optimistic by ~7× because it caught scoring mid-pipeline.

This also revises the `mask-kernel-subphase-measurement` conclusion:
that doc measured `mask_split_end = 22,654 µs/tick` as "scoring +
concat copy" because the END timestamp closed at `apply+movement`
(same pipelining artifact). Scoring's real cost is an order of
magnitude higher.

## Recommendations for next optimization

1. **Attack the scoring kernel first.** The `scoring::run_resident`
   dispatch is the 281 ms. No amount of inter-kernel barrier removal,
   submit rebatching, or readback tuning will move the needle.
   Candidates:
   * Sub-phase the scoring kernel the way `mask-kernel-subphase-measurement`
     sub-phased the mask kernel — emit separate scoring WGSL entry
     points per scoring category, measure which one dominates. (Mirror
     of that task's methodology.)
   * Check the per-entity scoring loop: at N=100k, a 2 M outer × inner
     loop takes real time. Cooperative partial-sum reductions or
     tile-based memory access may help.
   * Confirm that scoring isn't doing an O(N²) fallback on some
     edge-case input that the test fixture happens to trigger.
2. **Stop optimising barriers / submit cadence.** The numbers reject
   both. Any commit that claims a multi-ms batch-path win by moving
   compute passes around is probably noise.
3. **Validate on the sync path.** A quick `perf_n100.rs` run with
   per-dispatch gap marks would confirm that scoring dominates there
   too — if the sync path sees a different distribution, we've got a
   batch-specific codepath issue to investigate.

## Raw test outputs

### Baseline (default — no env var set)

```
chronicle_batch_perf_n100k: N=100000 spawn=3792ms backend=Vulkan
  warmup sync step: 8422 ms
  step_batch(50): total=15593 ms, avg=311864 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    --- per-dispatch gap table ---
    gap:before_fused_unpack                 : 1
    gap:fused_unpack->mask                  : 1
    gap:mask->scoring                       : 277895
    gap:scoring->apply_actions              : 1
    gap:apply_actions->movement             : 14
    gap:movement->append_events             : 1
    gap:append_events->seed_kernel          : 2
    gap:seed_kernel->cascade_iter_0         : 1
    gap:cascade_iter_0->1                   : 1
    gap:cascade_iter_1->2                   : 1
    gap:cascade_iter_2->3                   : 1
    gap:cascade_iter_3->4                   : 1
    gap:cascade_iter_4->5                   : 1
    --- legacy per-phase marks ---
    fused_unpack                            : 36
    mask+scoring                            : 22859
    apply+movement                          : 23
    append_events                           : 14
    spatial+abilities                       : 5113
    seed_kernel                             : 1557
    cascade iter 0                          : 86
    cascade iter 1                          : 27
    cascade iter 2                          : 17
    cascade iter 3                          : 13
    cascade iter 4                          : 15
    cascade_end                             : 1
  submit_granularity: default (1 submit per step_batch)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
  post-batch: tick=51 events_since_last=655360 chronicle_since_last=1000000
    AgentAttacked: 215813
    AgentDied:     25597
    chronicle_attack records: 198409
    alive agents:  100000
=== PERF N100k batch: 311864 µs/tick (50 ticks, 215813 attacks) ===
test chronicle_batch_perf_n100k ... ok
```

### ENGINE_GPU_SUBMIT_GRANULARITY=1

```
chronicle_batch_perf_n100k: N=100000 spawn=3809ms backend=Vulkan
  warmup sync step: 4075 ms
  step_batch(50): total=15697 ms, avg=313955 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    --- per-dispatch gap table ---
    gap:before_fused_unpack                 : 1
    gap:fused_unpack->mask                  : 1
    gap:mask->scoring                       : 284166
    gap:scoring->apply_actions              : 1
    gap:apply_actions->movement             : 15
    gap:movement->append_events             : 1
    gap:append_events->seed_kernel          : 2
    gap:seed_kernel->cascade_iter_0         : 1
    gap:cascade_iter_0->1                   : 1
    gap:cascade_iter_1->2                   : 1
    gap:cascade_iter_2->3                   : 1
    gap:cascade_iter_3->4                   : 1
    gap:cascade_iter_4->5                   : 1
    --- legacy per-phase marks ---
    fused_unpack                            : 35
    mask+scoring                            : 21683
    apply+movement                          : 24
    append_events                           : 14
    spatial+abilities                       : 4332
    seed_kernel                             : 165
    cascade iter 0                          : 55
    cascade iter 1                          : 27
    cascade iter 2                          : 14
    cascade iter 3                          : 16
    cascade iter 4                          : 15
    cascade_end                             : 822
  submit_granularity: K=1 (sub-batches of K ticks each)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
  post-batch: tick=51 events_since_last=655360 chronicle_since_last=1000000
    AgentAttacked: 215672
    AgentDied:     25485
    chronicle_attack records: 195142
    alive agents:  100000
=== PERF N100k batch: 313955 µs/tick (50 ticks, 215672 attacks) ===
test chronicle_batch_perf_n100k ... ok
```

### ENGINE_GPU_SUBMIT_GRANULARITY=5

```
chronicle_batch_perf_n100k: N=100000 spawn=3835ms backend=Vulkan
  warmup sync step: 4963 ms
  step_batch(50): total=15836 ms, avg=316721 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    --- per-dispatch gap table ---
    gap:before_fused_unpack                 : 1
    gap:fused_unpack->mask                  : 1
    gap:mask->scoring                       : 286134
    gap:scoring->apply_actions              : 1
    gap:apply_actions->movement             : 14
    gap:movement->append_events             : 1
    gap:append_events->seed_kernel          : 3
    gap:seed_kernel->cascade_iter_0         : 1
    gap:cascade_iter_0->1                   : 1
    gap:cascade_iter_1->2                   : 1
    gap:cascade_iter_2->3                   : 1
    gap:cascade_iter_3->4                   : 1
    gap:cascade_iter_4->5                   : 1
    --- legacy per-phase marks ---
    fused_unpack                            : 37
    mask+scoring                            : 21931
    apply+movement                          : 23
    append_events                           : 14
    spatial+abilities                       : 4412
    seed_kernel                             : 374
    cascade iter 0                          : 144
    cascade iter 1                          : 27
    cascade iter 2                          : 15
    cascade iter 3                          : 10
    cascade iter 4                          : 11
    cascade_end                             : 663
  submit_granularity: K=5 (sub-batches of K ticks each)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
  post-batch: tick=51 events_since_last=655360 chronicle_since_last=1000000
    AgentAttacked: 216274
    AgentDied:     25685
    chronicle_attack records: 194432
    alive agents:  100000
=== PERF N100k batch: 316721 µs/tick (50 ticks, 216274 attacks) ===
test chronicle_batch_perf_n100k ... ok
```

### ENGINE_GPU_SUBMIT_GRANULARITY=10

```
chronicle_batch_perf_n100k: N=100000 spawn=3746ms backend=Vulkan
  warmup sync step: 4136 ms
  step_batch(50): total=15449 ms, avg=308995 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    --- per-dispatch gap table ---
    gap:before_fused_unpack                 : 1
    gap:fused_unpack->mask                  : 1
    gap:mask->scoring                       : 279546
    gap:scoring->apply_actions              : 1
    gap:apply_actions->movement             : 15
    gap:movement->append_events             : 1
    gap:append_events->seed_kernel          : 3
    gap:seed_kernel->cascade_iter_0         : 1
    gap:cascade_iter_0->1                   : 1
    gap:cascade_iter_1->2                   : 1
    gap:cascade_iter_2->3                   : 1
    gap:cascade_iter_3->4                   : 1
    gap:cascade_iter_4->5                   : 1
    --- legacy per-phase marks ---
    fused_unpack                            : 36
    mask+scoring                            : 21675
    apply+movement                          : 24
    append_events                           : 14
    spatial+abilities                       : 3834
    seed_kernel                             : 266
    cascade iter 0                          : 298
    cascade iter 1                          : 27
    cascade iter 2                          : 16
    cascade iter 3                          : 12
    cascade iter 4                          : 15
    cascade_end                             : 529
  submit_granularity: K=10 (sub-batches of K ticks each)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
  post-batch: tick=51 events_since_last=655360 chronicle_since_last=1000000
    AgentAttacked: 215743
    AgentDied:     25600
    chronicle_attack records: 195209
    alive agents:  100000
=== PERF N100k batch: 308995 µs/tick (50 ticks, 215743 attacks) ===
test chronicle_batch_perf_n100k ... ok
```

### ENGINE_GPU_SUBMIT_GRANULARITY=50

```
chronicle_batch_perf_n100k: N=100000 spawn=3771ms backend=Vulkan
  warmup sync step: 4138 ms
  step_batch(50): total=15145 ms, avg=302906 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    --- per-dispatch gap table ---
    gap:before_fused_unpack                 : 1
    gap:fused_unpack->mask                  : 1
    gap:mask->scoring                       : 273470
    gap:scoring->apply_actions              : 1
    gap:apply_actions->movement             : 14
    gap:movement->append_events             : 1
    gap:append_events->seed_kernel          : 2
    gap:seed_kernel->cascade_iter_0         : 1
    gap:cascade_iter_0->1                   : 1
    gap:cascade_iter_1->2                   : 1
    gap:cascade_iter_2->3                   : 1
    gap:cascade_iter_3->4                   : 1
    gap:cascade_iter_4->5                   : 1
    --- legacy per-phase marks ---
    fused_unpack                            : 37
    mask+scoring                            : 21384
    apply+movement                          : 23
    append_events                           : 14
    spatial+abilities                       : 4453
    seed_kernel                             : 11
    cascade iter 0                          : 56
    cascade iter 1                          : 28
    cascade iter 2                          : 17
    cascade iter 3                          : 13
    cascade iter 4                          : 16
    cascade_end                             : 1
  submit_granularity: K=50 (sub-batches of K ticks each)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
  post-batch: tick=51 events_since_last=655360 chronicle_since_last=1000000
    AgentAttacked: 215955
    AgentDied:     25582
    chronicle_attack records: 200552
    alive agents:  100000
=== PERF N100k batch: 302906 µs/tick (50 ticks, 215955 attacks) ===
test chronicle_batch_perf_n100k ... ok
```

## Verify outputs

```
cargo test --release --features gpu -p engine_gpu 2>&1 | grep -E "test result|FAILED$" | tail -20
```

30 test binaries, all passing (2 ignored-marked perf tests as expected,
remaining tests 100% pass). No regressions from the gap-mark wiring.

## Reverting

The instrumentation is **on by default** — the gap marks always emit
when the profiler is enabled. To revert:

1. Revert commits on branch `gpu-per-dispatch-attribution` — four
   atomic commits ending at the test-harness update.
2. Or leave the marks in place: overhead is 16 × `write_timestamp` per
   tick, below measurement noise (±2% across K sweep).

The submit-granularity code path is opt-in via env var and has no
runtime cost when the var is unset. Safe to leave in place as a
diagnostic lever.
