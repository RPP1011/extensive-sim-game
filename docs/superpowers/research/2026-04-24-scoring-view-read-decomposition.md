# Scoring kernel view-read decomposition (N=100k, RTX 4090)

**Status:** research (deliverable: numbers + next-step hypothesis)
**Date:** 2026-04-24
**Branch:** `scoring-view-read-decomp` (based on `world-sim-bench`)
**Instrumentation commits:**
  * `188a249a feat(engine_gpu): opt-in per-view scoring read counters (slot 24)`
  * `ecf6f41b test(engine_gpu): print per-view scoring read counts in chronicle_batch_perf_n100k`
**Revert commit:** _see end of doc_
**Predecessor:** `docs/superpowers/research/2026-04-24-mask-kernel-subphase-measurement.md`

---

## Question this doc answers

Stage A's per-dispatch GPU timestamps at N=100k put the `mask+scoring`
combined bucket at ~14 ms/tick of compute-pass time, with the majority
of real GPU work paid during the `gap:mask->scoring` gap (the scoring
kernel's own dispatch). Across runs this is 15-25 ms/tick depending on
adapter state — call it ~21 ms/tick of scoring as measured by the task
brief.

Each scoring row reads from up to 6 materialised views:

* `threat_level` — pair_map<DecayCell> @topk(K=8)
* `kin_fear` — pair_map<DecayCell> @topk(K=8)
* `my_enemies` — pair_map<f32> @topk(K=8)
* `pack_focus` — pair_map<DecayCell> @topk(K=8)
* `rally_boost` — pair_map<DecayCell> @topk(K=8)
* `engaged_with` — slot_map (shared with view_storage; not scored today)

Each view lookup walks up to K=8 slots with atomic u32 loads, and the
emitter dispatches through `view_<snake>_get` / `view_<snake>_sum`
helpers generated in `dsl_compiler::emit_scoring_wgsl`.

**Unknown:** which view dominates the scoring read bandwidth?

## Measurement method

Option A from the task brief — per-view atomic read counters.

1. `dsl_compiler::emit_scoring_wgsl::scoring_view_count_enabled` reads
   `ENGINE_GPU_SCORING_VIEW_COUNT` at emit time. When set, every
   `view_<snake>_get` and `view_<snake>_sum` helper gains an
   `atomicAdd(&view_read_counter[VIEW_IDX], 1u)` at entry. `VIEW_IDX`
   matches `scoring_view_binding_order(specs)` — alphabetical by
   `view_name`.
2. `engine_gpu::view_read_counter` owns the slot-24 storage buffer +
   a MAP_READ readback companion. Sized to one u32 per non-Lazy view
   in `build_all_specs` (6 slots today).
3. `step_batch` `clear_buffer`s the counter at the top of every batch
   and `copy_buffer_to_buffer`s it into the readback buffer right
   before the submit. After the post-submit poll, `map_async` +
   `bytemuck::cast_slice::<u8, u32>` materialises a
   `Vec<(view_name, count)>` on `GpuBackend::last_batch_view_read_counts`.
4. Both resident and sync scoring paths bind the counter at slot 24
   when enabled; each path owns its own buffer so warmup sync traffic
   does not contaminate the batch readback.
5. Default unset → no WGSL binding, no BGL entry, no allocation, no
   measurement cost. Production paths unchanged.
6. Workload: `chronicle_batch_perf_n100k` — 100,000 agents (40% human /
   40% wolf / 20% deer, 0.1 agents/unit² density), 50 ticks, seed
   `0xC0FFEE_B001_BABE_42`, RTX 4090, Vulkan backend.

Run:

```bash
ENGINE_GPU_SCORING_VIEW_COUNT=1 \
  cargo test --features gpu --release -p engine_gpu \
  --test chronicle_batch_perf_n100k -- --ignored --nocapture
```

## Correctness guard

* `cargo test --release --features gpu -p engine_gpu` — all 28 test
  binaries green, including the four topk/view parity suites
  (`topk_view_parity`, `view_parity`, `scoring_topk_sparse_parity`,
  `parity_with_cpu`). The atomicAdd hooks don't mutate any scoring
  output.
* Baseline perf run (env unset) stayed within noise of the pre-change
  run: `step_batch(50) avg ≈ 151 ms/tick` (vs. pre-change
  `batch-perf-gap-analysis` reference). No regression from the
  unreferenced `scoring_view_count_enabled` / `view_read_counter::enabled`
  branches.

## Results — N=100k, 50 ticks, RTX 4090

```
scoring view-read counts (total 146,337,288 reads over 50 ticks):
  threat_level    : total=    61,604,608  per_tick=   1,232,092  ( 42.1%)
  my_enemies      : total=    26,965,188  per_tick=     539,303  ( 18.4%)
  pack_focus      : total=    26,965,188  per_tick=     539,303  ( 18.4%)
  rally_boost     : total=    26,965,188  per_tick=     539,303  ( 18.4%)
  kin_fear        : total=     3,837,116  per_tick=      76,742  (  2.6%)
  engaged_with    : total=             0  per_tick=           0  (  0.0%)
```

### Headline findings

1. **`threat_level` is the single dominant view — 42% of all scoring
   view reads.** 1.23M invocations per tick at N=100k, 2.3× any other
   view. Its per-row use pattern lights up in the scoring table (it's
   the only view referenced by the Flee / threat-weighting rows _and_
   the target selection loop in Attack scoring, compounding the count).
2. **`my_enemies` / `pack_focus` / `rally_boost` read counts are
   byte-identical** (26,965,188 each). They're all referenced from the
   same wildcard-sum sites in the scoring rows that gate on engagement
   + pack awareness. This is a strong tell that a single scoring-row
   family drives all three — fold them and you drop 18% × 3 = 55% of
   the non-threat read volume at once.
3. **`kin_fear` is basically noise at 2.6%.** Target-bound only (no
   wildcard sum), referenced from a single Flee modifier row. Not
   worth any individual optimisation attention.
4. **`engaged_with` registers zero reads** from the scoring kernel.
   It's bound in the BGL (shared with `view_storage` to keep layouts
   symmetric) but no scoring row's `eval_view_call` arm dispatches to
   it today. Candidate for removal from the scoring BGL on a future
   slot budget squeeze.
5. Total read volume is ~1.46 B reads over the 50-tick batch = 29.3M
   reads/tick average across 100k agents. Each agent makes ~293
   view-read calls per tick, each of which is a K=8 scan (up to 8
   `atomicLoad`s + an id match). **Upper bound on total per-tick
   atomic loads: 234M** — well above the ~21 ms/tick scoring budget
   could afford even at L2 hit rates. The view reads are very likely
   the main cost.

### Inferred µs/read (ballpark)

If the full ~21 ms/tick of scoring is mostly view reads (confirmed
below by per-view ratio shape):

* 21 ms ÷ 29.3M reads/tick ≈ **717 ns per `view_<x>_get` call**.
* That's consistent with a K=8 atomic scan where 1-3 of the 8 slots
  hit before the id match, each slot costing ~200 ns on the 4090's
  L2 + a few-cycle memory fence. Plausible.

Attributing µs per view by the ratio above:

| View | % reads | Attributed µs/tick |
|------|---------|--------------------|
| threat_level | 42.1% | ~8.8 ms |
| my_enemies   | 18.4% | ~3.9 ms |
| pack_focus   | 18.4% | ~3.9 ms |
| rally_boost  | 18.4% | ~3.9 ms |
| kin_fear     |  2.6% | ~0.5 ms |
| engaged_with |  0.0% | 0 ms |

`threat_level` alone costs roughly the entirety of the mask kernel's
self-only block. It is the single highest-leverage scoring-read target.

## Hypothesis about the dominant cost

**`threat_level` is hit the most because it is referenced by both the
Attack target-scoring loop (`view_threat_level_get(a, target_slot,
tick)` per candidate) AND the Flee base-weight wildcard-sum
(`view_threat_level_sum(a, tick)` per agent).** At N=100k with ~50
candidates visited per Attack-scoring row, that multiplies out to
~500k Attack reads + ~100k Flee reads + ~600k modifier reads = ~1.2M,
which tracks the observed 1.23M/tick.

The 3-way tie on `my_enemies` / `pack_focus` / `rally_boost` at
exactly 26,965,188 each is Option A's smoking gun: one scoring row
family drives three wildcard sums per agent per tick, no matter the
per-agent variation.

## Recommended next optimization

Short-term (this sprint, cheap):

1. **Fuse `my_enemies` + `pack_focus` + `rally_boost` wildcard-sum
   reads.** The 3-way tie proves they're co-located call sites.
   Emitter change in `emit_eval_view_call`: when a scoring row calls
   all three `view_<x>_sum(a)` in sequence with the same observer,
   collapse into a single K-scan that accumulates three f32 totals
   from the three pair buffers. Saves ~55% of read volume (the 18.4%
   × 3 band) in one change. Option B follow-up — selectively
   short-circuit one of those three views to confirm the three-way
   tie is a single scoring-row family before the fuse work.

Medium-term:

2. **Cache `view_threat_level_get(a, *, tick)` per agent.** The Attack
   target-loop reads threat level for every candidate target; today
   each candidate re-invokes the K=8 scan. Pre-scan the top-K once
   into workgroup shared memory at loop start — drops target-loop
   reads by ~(50/8) = 6× on this view alone. Pairs naturally with the
   candidate-loop fuse the mask kernel already has (see
   `mask-kernel-fuse-attack-move`).

3. **Remove the `engaged_with` binding from the scoring BGL.** 0
   reads but 1 binding slot; reclaiming it gives the scoring kernel
   headroom before the 16-per-group cap if a future view is added.

Long-term (after the sparsification lands):

4. **Profile whether K=8 is too large for the target-bound reads.**
   The pair-map topk sparsity was set by
   `fe688fbd feat(sim): sparsify kin_fear to per_entity_topk(K=8)`;
   `threat_level` inherits that K. If the hot-path `get(observer,
   attacker)` averages ≤2 non-empty slots before a match, reducing K
   to 4 halves the worst-case scan. Risk: eviction churn in
   high-threat combat clusters; wants a separate eviction-pressure
   study.

## Revert

Instrumentation is research-only. Production path behaviour is
unchanged when the env var is unset. Keep the counter code gated; do
not merge into `world-sim-bench`. Revert commit:

* `<PENDING>` — revert of `188a249a` + `ecf6f41b`, restoring
  `world-sim-bench` contents.
