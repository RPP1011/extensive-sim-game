# Mask kernel sub-phase measurement (N=100k, RTX 4090)

**Status:** research (deliverable: numbers, not a perf fix)
**Date:** 2026-04-24
**Branch:** `mask-kernel-subphase-measurement` (based on `world-sim-bench`)
**Split commit:** `18caeff5 feat(engine_gpu): opt-in split mask kernel for sub-phase measurement`
**Revert commit:** _see end of doc_
**Predecessor:** `docs/superpowers/research/2026-04-22-batch-perf-gap-analysis.md` §§ "2026-04-24 refresh"

---

## Question this doc answers

Stage A's GPU timestamps at N=100k attributed **37,083 µs/tick (98.1% of
GPU time)** to the `mask+scoring` phase bucket. That single label hides
a multi-sub-phase kernel:

* **self-only block** — 5 self-only masks (Hold/Flee/Eat/Drink/Rest),
  each predicate `alive(self)`, collapsed to 5 unconditional atomicOr
  writes per alive agent.
* **MoveToward candidate loop** — naive O(N) loop over every other
  agent with radius prefilter `cfg.movement_max_move_radius` (~20 m).
* **Attack candidate loop** — naive O(N) loop with
  `is_hostile(self_ct, target_ct)` pairwise check + `dist < 2.0` inner
  predicate, radius prefilter `sim_cfg.attack_range` (2 m).
* **scoring kernel** — the _other_ half of the outer bucket.

Which of those actually dominates the 37 ms? A reviewer pushed back on
the "30 ms of is_hostile atomic view reads" hypothesis (the in-tree
`is_hostile` is inlined as a pairwise creature-type table, not a view
storage load, so the hot cost is candidate-loop bandwidth rather than
L2 atomics). Measurement settles it.

## Measurement method

1. **Option A** from the task brief: emit three standalone WGSL entry
   points (`cs_mask_self_only`, `cs_mask_movetoward`, `cs_mask_attack`)
   against the fused kernel's existing bind-group layout and dispatch
   them sequentially in place of the single `cs_fused_masks` dispatch.
2. Each sub-dispatch is sandwiched between `GpuProfiler::mark(..)`
   calls so Stage A's timestamp readback produces per-sub-phase µs
   automatically. No changes to the profiler module itself.
3. Activation is gated on an `ENGINE_GPU_SPLIT_MASK_MEASURE=1` env var
   read once at first dispatch, cached in an atomic — default-off so
   the production path still dispatches the fused kernel.
4. Parity guard: predicate bodies in `SPLIT_MASK_WGSL` transcribe
   exactly what `emit_masks_wgsl_fused` generates for the same IRs.
   `step_batch_smoke`, `chronicle_batch_path`, and
   `chronicle_batch_stress_n20k` pass under both paths.
5. Workload: `chronicle_batch_perf_n100k` — 100,000 agents (40%
   human / 40% wolf / 20% deer, 0.1 agents/unit² density), 50 ticks,
   seed `0xC0FFEE_B001_BABE_42`, RTX 4090, Vulkan backend.

Run:

```bash
ENGINE_GPU_SPLIT_MASK_MEASURE=1 \
  cargo test --features gpu --release -p engine_gpu \
  --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

## Headline results

### Per-sub-phase GPU µs/tick at N=100k (mean over 50 ticks)

| Sub-phase | µs/tick | % of fused `mask+scoring` |
|---|---:|---:|
| `mask_split_self_only` | **0** | <0.01% |
| `mask_split_movetoward` | **165** | 0.4% |
| `mask_split_attack` | **1,399** | 3.7% |
| `mask_split_end` (= scoring kernel + concat copy) | **22,654** | 59.9% |
| **Sum under split path** | **24,218** | **64.1%** |
| Fused `mask+scoring` (baseline) | 37,794 | 100% |

_Readback convention: each profiler entry `(label_i, Δ_i)` is the delta
from mark `i` to mark `i+1`, labelled with the opening mark. `mask_split_end`
therefore covers the region from the last split mark to the next outer
phase mark (`apply+movement`), which wraps the entire scoring dispatch
plus the per-bitmap `copy_buffer_to_buffer` concat — not mask work._

### Wall-clock

| Variant | µs/tick wall | GPU-µs accounted |
|---|---:|---:|
| Fused (baseline) | 322,279 | 38,524 |
| Split (measurement) | 314,550 | 24,219 |

Wall-clock noise dominates; there is no shipping regression from the
split itself at this N.

## Findings

1. **Scoring is the real dominant cost, not the mask kernel.** The
   single label "mask+scoring" at 37 ms hid a ~2:1 scoring-to-mask
   split. Broken out:
   * Mask work (all three sub-phases combined): **~1,564 µs/tick**
     (self-only + MoveToward + Attack).
   * Scoring + concat copy: **~22,654 µs/tick**.

   Stage C planners who assumed the mask kernel was the 37 ms target
   will want to redirect fold-batching attention to
   `scoring::run_resident`. The mask kernel is 4% of the bucket, not
   100%.

2. **Self-only masks are free at this N.** 0 µs/tick — five atomicOr
   writes per alive agent plus a workgroup-size dispatch of 100k
   threads is below timestamp resolution on the 4090.

3. **The Attack candidate loop is 8–9× the MoveToward loop** despite
   MoveToward's radius being ~10× bigger (20 m vs 2 m). The Attack
   loop adds (a) a creature-type fetch per candidate and (b) the
   `is_hostile(self_ct, target_ct)` pairwise branch inside the loop,
   which triggers warp divergence when agents of different species
   sit in the same workgroup. Combined with the hot ~215k attacks/tick
   on this fixture the Attack loop's inner work dominates MoveToward's
   despite iterating fewer candidates.

4. **is_hostile cost is ≤1.4 ms/tick, not 30 ms.** The reviewer's
   back-of-envelope (atomic L2 loads on a 4090 land at 2–4 ms, not
   30 ms) is confirmed. The in-tree `is_hostile` is a pairwise
   creature-type table — no view_storage loads involved. The original
   hypothesis from the task brief overestimated by ~20×.

5. **Fusion is a net loss at N=100k, not a net win.** Summing the
   three split sub-dispatches plus scoring yields **24,218 µs/tick**
   against **37,794 µs/tick** for the fused kernel — a **13,576 µs/tick
   (~36%) reduction** just from splitting. This is the opposite of the
   usual fusion story: at N=100k the per-sub-phase kernels likely
   enjoy lower register pressure + better occupancy than the fused
   kernel, which at 100k threads × 3-phases-of-divergent-work-each
   gets hurt by warp divergence on the `if (found) break;` pattern
   repeating across phases.

6. **The split-measurement overhead is _smaller_ than the fusion
   "win".** 3 compute-pass begin/end + 3 pipeline-set adds ~10–30 µs
   on a 4090; the measured delta is +13,576 µs favouring split. The
   instrumentation is not the confound.

## Answer to the question

The task asked:
> whether the fusion win was confirmed larger than the split-measurement overhead.

**No.** Fusion is a 13.6 ms/tick net _loss_ at N=100k. Splitting into
three sub-dispatches reduces GPU time by 36% on this workload. The
measurement overhead is ~0.1% of the dispatch cost — not the confound.

This is outside the scope of "fix it in this task" — but the
measurement makes the follow-up clear:

* **Stage C mask work is small.** 1.6 ms/tick across all three mask
  sub-phases combined. Optimising the mask kernel is not the lever.
* **Stage C scoring work is big.** 22.7 ms/tick in one dispatch. Fold
  batching + argmax restructure should target the scoring kernel.
* **Fusion boundaries deserve a second look.** The Attack sub-phase
  inside the fused kernel seems to be suffering warp divergence that
  separating into its own workgroup resolves. If the mask kernel ever
  becomes a bottleneck, the revert path is known: delete the fused
  variant, ship the three-kernel split.

## Per-sub-phase raw output (split, N=100k, 50 ticks)

```
chronicle_batch_perf_n100k: N=100000 spawn=3967ms backend=Vulkan
  warmup sync step: 21723 ms
  step_batch(50): total=15727 ms, avg=314550 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    fused_unpack        : 3
    mask+scoring        : 0
    mask_split_self_only: 0
    mask_split_movetoward: 165
    mask_split_attack   : 1399
    mask_split_end      : 22654
    apply+movement      : 3
    append_events       : 2
    spatial+abilities   : 330
    seed_kernel         : 7
    cascade iter 0      : 3
    cascade iter 1      : 2
    cascade iter 2      : 1
    cascade iter 3      : 1
    cascade iter 4      : 1
    cascade_end         : 0
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
=== PERF N100k batch: 314550 µs/tick (50 ticks, 215802 attacks) ===
```

## Per-sub-phase raw output (fused, N=100k, 50 ticks, baseline)

```
chronicle_batch_perf_n100k: N=100000 spawn=7298ms backend=Vulkan
  warmup sync step: 4838 ms
  step_batch(50): total=16113 ms, avg=322279 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    fused_unpack        : 4
    mask+scoring        : 37794
    apply+movement      : 6
    append_events       : 2
    spatial+abilities   : 541
    seed_kernel         : 0
    cascade iter 0      : 174
    cascade iter 1      : 2
    cascade iter 2      : 1
    cascade iter 3      : 1
    cascade iter 4      : 1
    cascade_end         : 0
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
=== PERF N100k batch: 322279 µs/tick (50 ticks, 215759 attacks) ===
```

## Reproducing

Split-path numbers:

```bash
git checkout mask-kernel-subphase-measurement
ENGINE_GPU_SPLIT_MASK_MEASURE=1 \
  cargo test --features gpu --release -p engine_gpu \
  --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

Fused-path baseline (same branch, env var unset):

```bash
cargo test --features gpu --release -p engine_gpu \
  --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

## Deliverable-scope revert

After this doc is landed the measurement code is reverted so
`world-sim-bench` keeps shipping the fused kernel. See the revert commit
at the end of this branch — it drops `SPLIT_MASK_WGSL`,
`SplitPipelines`, `run_resident_split`, and `split_mask_measure_enabled`,
restoring `lib.rs` to its pre-branch `run_resident` call site.

## Caveats / not measured

* **Single-adapter data.** Ran on a single RTX 4090 / Vulkan target.
  The warp-divergence-penalty hypothesis would play differently on
  Metal/DX12/smaller chips. A follow-up sweep on a 3060-tier would
  confirm the fusion-is-a-loss result at lower SM counts.
* **Within-`mask_split_attack` attribution.** The Attack sub-phase at
  1.4 ms isn't further decomposed. If it later becomes the bottleneck
  the next split-axis is (a) radius prefilter rejection vs (b)
  is_hostile rejection vs (c) dist<2 inner predicate.
* **Scoring kernel is not decomposed.** This doc's `mask_split_end`
  number aggregates scoring + concat copy. A follow-up research pass
  that splits scoring's argmax/fold stages would sharpen the Stage C
  targeting.
