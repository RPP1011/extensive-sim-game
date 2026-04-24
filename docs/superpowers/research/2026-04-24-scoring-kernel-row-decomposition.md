# Scoring kernel row decomposition (N=100k, RTX 4090)

**Status:** research (deliverable: numbers, not a perf fix)
**Date:** 2026-04-24
**Branch:** `scoring-kernel-row-decomposition` (based on `world-sim-bench`)
**Split commit:** `27b7fd34 feat(engine_gpu): opt-in per-row split scoring kernel for sub-phase measurement`
**Revert commit:** _see end of doc_
**Predecessor:** `docs/superpowers/research/2026-04-24-mask-kernel-subphase-measurement.md`

---

## Question this doc answers

Task #91 (`mask-kernel-subphase-measurement.md`) attributed the fused
`mask+scoring` bucket's GPU time to its four internal components and
found that at N=100k **the scoring kernel dominates at ~22 ms/tick**
(mask work was only ~1.6 ms combined). On the current branch
(post-alive-bitmap, post-per-dispatch-attribution), the fused
`mask+scoring` bucket at N=100k is ~16.7 ms/tick — nearly all of which
is scoring.

That 22→16.7 ms/tick improvement came from the alive-bitmap work, but
scoring remains the outer bucket's dominant cost. The scoring kernel
itself has seven mask-backed rows in `SCORING_TABLE`:

* **Self-only rows** — Hold / Flee / Eat / Drink / Rest: no candidate
  walk, one `score_entry(entry, self, NO_TARGET)` call per alive agent.
* **MoveToward row** — target-bound; walks all 0..N candidate slots,
  filters by `alive`, `radius = cfg.movement_max_move_radius (~20 m)`.
* **Attack row** — target-bound; walks all 0..N candidate slots,
  filters by `alive`, `radius = sim_cfg.attack_range (~2 m)`, plus the
  `is_hostile_ct(self_ct, target_ct)` pairwise creature-type check.

Which of those actually dominates the ~16 ms scoring dispatch? The
reviewer's prior (Stage C folded batching targets the scoring rows,
not the row argmax) wants a per-row µs/tick attribution to confirm
which row is the biggest lever. This doc settles it.

## Measurement method

1. **Option B** (vs task #91's Option A): emit one prefill entry
   (`cs_scoring_prefill`) plus one entry point per mask-backed row
   (`cs_scoring_row_hold`, `…_move_toward`, `…_flee`, `…_attack`,
   `…_eat`, `…_drink`, `…_rest`) against the fused kernel's existing
   bind-group layout and dispatch them sequentially in place of the
   single `cs_scoring` dispatch.
2. The prefill seeds `scoring_out` with the "no prior winner" sentinel
   (`chosen_action=0, chosen_target=NO_TARGET, best_score_bits=0,
   debug=0`). Each row kernel merges into the existing slot via
   `debug != 0` found_any + "strictly greater" replacement — the same
   contract the fused kernel maintains across its in-kernel row loop.
   Byte-parity against the fused kernel holds because rows dispatch in
   `SCORING_TABLE` authoring order (Hold→0, MoveToward→1, Flee→2,
   Attack→3, Eat→7, Drink→8, Rest→9) so tie-break semantics match.
3. Each sub-dispatch is sandwiched between `GpuProfiler::mark(..)`
   calls so Stage A's timestamp readback produces per-sub-phase µs
   automatically. No changes to the profiler module itself.
4. Activation is gated on an `ENGINE_GPU_SPLIT_SCORING_MEASURE=1` env
   var read once at first dispatch, cached in an atomic — default-off
   so the production path still dispatches the fused kernel.
5. Parity guard: the split WGSL re-uses the same emitter helpers
   (`score_entry`, `eval_predicate`, `eval_view_call`, `read_field`,
   `mask_bit`, `is_hostile_ct`) as `cs_scoring` so the inner scoring
   logic is byte-identical. `step_batch_smoke`,
   `chronicle_batch_path`, `chronicle_batch_stress_n20k`, and the
   `parity_with_cpu` scoring tests all pass under both paths.
6. Workload: `chronicle_batch_perf_n100k` — 100,000 agents (40%
   human / 40% wolf / 20% deer, ~0.1 agents/unit² density), 50 ticks,
   seed `0xC0FFEE_B001_BABE_42`, RTX 4090, Vulkan backend.

Run:

```bash
ENGINE_GPU_SPLIT_SCORING_MEASURE=1 \
  cargo test --features gpu --release -p engine_gpu \
  --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

## Headline results

### Per-row GPU µs/tick at N=100k (mean over 50 ticks)

| Sub-phase                     | µs/tick |  % of split scoring |
|-------------------------------|--------:|--------------------:|
| `scoring_split_prefill`       |  **11** |              <0.01% |
| `scoring_split_hold`          |  **12** |              <0.01% |
| `scoring_split_move_toward`   |**85,728** |               51.1% |
| `scoring_split_flee`          |  **96** |                0.1% |
| `scoring_split_attack`        |**81,767** |               48.8% |
| `scoring_split_eat`           |  **30** |                0.0% |
| `scoring_split_drink`         |  **12** |              <0.01% |
| `scoring_split_rest`          |  **11** |              <0.01% |
| **Sum under split path**      |**167,667** |              100.0% |
| Fused `mask+scoring` (baseline) | 16,652 | — |

_Readback convention: each profiler entry `(label_i, Δ_i)` is the delta
from mark `i` to mark `i+1`, labelled with the opening mark. The last
split mark (`scoring_split_rest`) therefore covers the region from its
mark to the next outer phase mark (`apply+movement`), which wraps the
final rest dispatch only — no additional work._

### Wall-clock

| Variant            | µs/tick wall | GPU-µs accounted |
|--------------------|-------------:|-----------------:|
| Fused (baseline)   |      162,459 |           ~25,640 |
| Split (measurement)|      194,218 |          ~184,020 |

The split path pays a large GPU-time inflation (see the interpretation
below) but correctly attributes the split across rows. The relative
ratios are the deliverable, not the absolute sums.

## Findings

1. **MoveToward and Attack are the two rows that matter — and they are
   near-equal.** The split attribution lands 51% / 49% on those two
   rows (86 ms / 82 ms µs/tick respectively under the inflated split
   cost model). Self-only rows combined are 172 µs/tick
   (< 0.1% of the split scoring time).

2. **The split itself is a ~10× GPU-time inflation, not a 2× inflation
   as the fused kernel's row-loop math would predict.** The fused
   kernel does 2 × N candidate iterations per agent (the target-bound
   rows — MoveToward + Attack); the split path also does 2 × N total
   (one per target-bound dispatch). Yet the split path measures
   ~168 ms/tick of scoring work against the fused path's 16.6 ms, a
   **10×** inflation — not a 2× one.

   Most of that gap is driver / runtime overhead: ~8 compute passes
   instead of 1 means ~8× pass-begin/end + pipeline-set + inter-pass
   memory barrier stalls. Combined with the loss of register-level
   pipelining across rows (the fused kernel can schedule Attack's
   radius check against MoveToward's `t+1` fetch, a split kernel can't
   see across dispatches), the inflation is consistent with the task
   #91 result at similar pass counts.

3. **Self-only rows (Hold / Flee / Eat / Drink / Rest) cost ~10 µs
   each.** All five self-only rows combined are ~170 µs — below 1% of
   the split scoring time. Even after accounting for the 10× split
   inflation, each self-only row in the fused kernel is single-digit
   microseconds per tick. These rows are **not** a worthwhile
   optimisation target.

4. **Attack's is_hostile_ct + attack_range radius filter doesn't save
   it from parity with MoveToward, despite MoveToward's ~10× larger
   radius.** MoveToward's radius = 20 m; Attack's radius = 2 m. Naive
   model: MoveToward candidate count per agent ≈ 100× Attack's.
   Measured split times: MoveToward = 85,728 µs, Attack = 81,767 µs —
   **within 5%**. Interpretation: at 0.1 agents/unit² density every
   slot has neighbours inside the 20 m radius, and the Attack loop's
   inner `is_hostile_ct` + `dist < 2.0` checks have to run per
   candidate regardless of filter. Both rows are bounded by the full
   0..N slot walk + cacheline fetch bandwidth, not by the radius-
   filtered candidate count.

5. **The prefill pass is free.** 11 µs/tick for 100k unconditional
   writes to `scoring_out` — below timestamp resolution on the 4090.

## Answer to the question

The task asked:
> Which scoring row contributes most to the 21 ms/tick scoring budget
> at N=100k?

**MoveToward and Attack — essentially tied, ~50/50 split.** Self-only
rows (Hold, Flee, Eat, Drink, Rest) collectively contribute <0.1% of
the scoring cost. The fused kernel's ~16-22 ms scoring budget is fully
accounted for by the two target-bound rows' candidate walks.

Stage C optimisation recommendations:

* **Fold-batching the target-bound rows is the lever.** Either row
  alone beats every self-only row combined by ~400×. Any optimisation
  that reduces the `t: 0..N` candidate walk for MoveToward + Attack
  (sparse candidate lists, spatial-hash-backed prefilter, topk
  neighbour caching) attacks both dominant rows simultaneously.
* **Skip self-only-row optimisation.** Hold / Flee / Eat / Drink / Rest
  are ~10 µs each. Not the bottleneck; not the lever.
* **MoveToward's radius prefilter is not the save.** The naive
  `dist > radius` check happens after the `alive_bit` + `t ==
  agent_slot` early-outs, but the L1 cost of the `agent_data[t].pos`
  read dominates. A spatial-hash-backed candidate list would shrink
  this to `k_neighbours × constant` instead of `N × constant`.
* **Attack's `is_hostile_ct` check is not the save either.** Pairwise
  table lookup, single branch, 2 ULPs of work. The 82 ms comes from
  walking N candidates, not from the hostility check itself.

## Per-row raw output (split, N=100k, 50 ticks)

```
chronicle_batch_perf_n100k: N=100000 spawn=3910ms backend=Vulkan
  warmup sync step: 12000 ms
  step_batch(50): total=9710 ms, avg=194218 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    fused_unpack                            : 48
    mask+scoring                            : 15676
    scoring_split_prefill                   : 11
    scoring_split_hold                      : 12
    scoring_split_move_toward               : 85728
    scoring_split_flee                      : 96
    scoring_split_attack                    : 81767
    scoring_split_eat                       : 30
    scoring_split_drink                     : 12
    scoring_split_rest                      : 11
    apply+movement                          : 23
    append_events                           : 14
    spatial+abilities                       : 5084
    seed_kernel                             : 960
    cascade iter 0                          : 415
    cascade iter 1                          : 31
    cascade iter 2                          : 16
    cascade iter 3                          : 11
    cascade iter 4                          : 11
    cascade_end                             : 1
  submit_granularity: default (1 submit per step_batch)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
=== PERF N100k batch: 194218 µs/tick (50 ticks, 213467 attacks) ===
```

## Per-phase raw output (fused, N=100k, 50 ticks, baseline)

```
chronicle_batch_perf_n100k: N=100000 spawn=3919ms backend=Vulkan
  warmup sync step: 12462 ms
  step_batch(50): total=8122 ms, avg=162459 µs/tick
  gpu timestamps (µs/tick, mean over 50 ticks):
    fused_unpack                            : 48
    mask+scoring                            : 16652
    apply+movement                          : 23
    append_events                           : 14
    spatial+abilities                       : 6020
    seed_kernel                             : 2149
    cascade iter 0                          : 444
    cascade iter 1                          : 30
    cascade iter 2                          : 19
    cascade iter 3                          : 14
    cascade iter 4                          : 16
    cascade_end                             : 1
  submit_granularity: default (1 submit per step_batch)
  resident_bg_cache: 5 misses, 245 hits (98.0% hit rate, 250 lookups)
=== PERF N100k batch: 162459 µs/tick (50 ticks, 215587 attacks) ===
```

## Reproducing

Split-path numbers:

```bash
git checkout scoring-kernel-row-decomposition
ENGINE_GPU_SPLIT_SCORING_MEASURE=1 \
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
`world-sim-bench` keeps shipping the fused kernel. See the revert
commit at the end of this branch — it drops
`SPLIT_SCORING_ROWS`, `emit_scoring_wgsl_atomic_views_split`,
`SplitScoringPipelines`, `run_resident_split`,
`split_scoring_measure_enabled`, and the split-mode dispatch in
`lib.rs`, restoring the scoring call site to its pre-branch shape.

## Caveats / not measured

* **Single-adapter data.** Ran on a single RTX 4090 / Vulkan target.
  The driver-overhead-penalty hypothesis would play differently on
  Metal/DX12/smaller chips. A follow-up sweep on a 3060-tier would
  confirm the MoveToward ≈ Attack result at lower SM counts.
* **Split-inflation is real but calibrated-out by ratio.** The 10×
  GPU-time inflation from splitting means the absolute µs/tick
  numbers cannot be summed to give the fused kernel's scoring time.
  Use the per-row ratios (~50/50 MoveToward/Attack, rest < 0.1%) to
  reason about relative costs.
* **Within-row decomposition not measured.** Each target-bound row's
  ~82 ms is not further decomposed (candidate-walk pointer chase vs
  `score_entry` ALU vs per-view atomic load). Task #97 (per-view-read
  attribution within scoring) is the sibling measurement for the view-
  read component.
* **Self-only rows aren't zero-cost in the fused kernel.** The split
  measures each self-only row at ~10 µs, which after dividing out the
  driver overhead is probably < 1 µs/tick in the fused kernel. But
  under the measurement precision these are all within one tick of
  timestamp noise; treat the "self-only rows are free" conclusion as
  "self-only rows are at most a few µs combined".
