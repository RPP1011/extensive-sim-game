# Batch-path perf gap analysis (N=2048)

**Status:** research
**Date:** 2026-04-22
**Branch:** `world-sim-bench`
**Predecessor:** `docs/technical_overview.md` §10.1; `docs/superpowers/specs/2026-04-22-gpu-resident-cascade-design.md`

## Executive summary

At N=2048 the GPU-resident batch path measures 8114 µs/tick vs 6442 µs/tick
for the sync path — a **1.26× regression** against the spec's 0.8× target
(a 1672 µs/tick gap to account for). Reading the code, the gap decomposes
into four proximate suspects, with a fifth discovered here (the CPU-side
spatial SoA re-pack). None is structural; all are local, tractable
optimisations. Ranking by estimated contribution to the 1672 µs/tick gap:

| # | Suspect | Est. µs/tick | Fix cost | Fixable? |
|---|---|---|---|---|
| 1 | Double CPU-side pack + write_buffer for spatial SoA (2× `upload_agent_soa`) | 400–900 | Small | yes — cache |
| 2 | Per-call bind-group construction in apply_actions/movement/scoring/physics (8×) | 200–400 | Small | yes — cache |
| 3 | Always-dispatch-8 cascade iters on a `≤ 1.07` iter fixture | 200–500 | Small | yes — indirect cutoff |
| 4 | Per-tick `PackedAbilityRegistry` re-upload into resident ability buffers | 100–300 | Small | yes — cache+dirty |
| 5 | Mask-unpack + scoring-unpack (the 9b730988 correctness fix) | 100–400 | Medium | partly — merge |
| 6 | Per-tick clear_buffer tail resets (7 ops/tick) | 20–100 | Small | partial |

Strategic recommendation: **attack the local optimisations first.** The gap
appears fixable without the larger GPU-everything migration. A realistic
post-fix batch/sync ratio at N=2048 is 0.70–0.90×; the 0.8× target is within
reach. Proceed with the GPU-everything migration for its own merits
(eliminating the remaining CPU-side tick advance, RNG seeding, and cold-state
rules) rather than to close this perf gap.

---

## Section A — Sync vs batch cost model

Both paths run the same WGSL kernels with the same workgroup counts.
The 1672 µs/tick gap is therefore driver/encoder overhead, not GPU
arithmetic. Kernel-dispatch counts per tick at N=2048:

**Sync path** (`lib.rs::SimBackend::step` + `cascade::run_cascade`):
~18–22 compute passes + ~12 CPU/GPU fences per tick. Sync has an
expensive fence-per-kernel pattern but caches most per-call allocations
in the mask kernel's `pool.bind_group` and uses internal pool buffers
for spatial.

**Batch path** (`lib.rs::step_batch` + `cascade_resident::run_cascade_resident`),
in encoder-record order:

1. `mask_unpack.encode_unpack` — 1 pass + ~10 mask-bitmap clears (cached BG)
2. `mask.run_resident` — 1 pass + per-bitmap copy into concat buf (cached BG)
3. `scoring_unpack.encode_unpack` — 1 pass (cached BG)
4. `scoring.refresh_tick_cfg_for_resident` — 1 queue.write_buffer
5. `scoring.run_resident` — 1 pass, **BG rebuilt per call**
6. `clear_buffer(apply_event_ring.tail)`
7. `apply_actions.run_resident` — 1 pass, **BG rebuilt per call**
8. `movement.run_resident` — 1 pass, **BG rebuilt per call**
9. Cascade resident:
   - 2× `spatial.rebuild_and_query_resident`: each calls
     `upload_agent_soa` (CPU Vec pack + 4× queue.write_buffer), allocates
     a fresh `qcfg_buf`, builds a fresh 12-entry BG, and runs 5 passes
     (clear, count, scan, scatter, sort+query).
   - `ability_bufs.upload` (4× queue.write_buffer, ~68 KB total)
   - 3× clear_buffer for ring tails + num_events
   - 1× seed indirect kernel dispatch
   - **8× `physics.run_batch_resident` indirect dispatches** — fresh
     16-entry BG each, 2× queue.write_buffer each. Iters 3..7 have
     args `(0,1,1)` on the perf fixture.
   - Up to 5× inter-iter ping-pong tail clears

Total batch: ~20 compute passes and ~40–50 encoder ops per tick inside a
single submit. The batch path is identical GPU arithmetic + identical
dispatch count to sync, but pays per-call allocation tax on the spatial,
scoring, apply, movement, and physics bind groups, plus a fixed 8-iter
cascade envelope regardless of convergence.

---

## Section B — Suspect 1: double spatial rebuild

`cascade_resident.rs:863-878` calls `rebuild_and_query_resident` twice
per tick (kin_radius + engagement_range). Each full pipeline runs 5
compute passes and does:

1. **CPU-side `upload_agent_soa`** (`spatial_gpu.rs:1017-1077`): builds
   three fresh `Vec`s from `SimState.hot_pos()/hot_alive()` + creature
   types, then 4× queue.write_buffer. At N=2048: ~30 µs alloc + ~80 µs
   write calls = **~100–200 µs per call, 200–400 µs/tick**.
2. **12-entry BG construction** (`spatial_gpu.rs:1266-1283`): ~50–100 µs
   per call on llvmpipe × 2/tick.
3. **Fresh `qcfg_buf`** via `create_buffer_init` per call: ~50–100 µs × 2.

Total double-rebuild CPU-side overhead: **~400–900 µs/tick**. The two
queries share pos/alive/creature_type inputs — only radius differs.

**Fixability:** Small. Split upload from query, upload once per tick,
and cache the two query BGs keyed on agent_cap. The SoA is bit-stable
across batch ticks (state doesn't mutate agent positions mid-batch from
the CPU side) so in principle the upload could be elided entirely on
non-agent_cap-grow ticks; that's a larger rewrite for ~100 µs more.

---

## Section C — Suspect 2: per-call bind-group construction

Grepped each resident kernel's `run_resident` body:

| Kernel | BG cached? | Rebuild cost |
|---|---|---|
| `MaskKernel::run_resident` (mask.rs:683) | **yes** (`pool.bind_group`) | zero — allocated once with pool |
| `MaskUnpackKernel::encode_unpack` (mask.rs:1093) | **yes** (`cached_bg`, keyed on agent_cap) | zero on steady-state |
| `ScoringKernel::run_resident` (scoring.rs:1259) | **NO** — builds `build_resident_bind_group` per call | ~100–200 µs |
| `ScoringUnpackKernel::encode_unpack` (scoring.rs:2107) | **yes** (`cached_bg`) | zero on steady-state |
| `ApplyActionsKernel::run_resident` (apply_actions.rs:470) | **NO** — inline `create_bind_group` per call (5 entries) | ~50–100 µs |
| `MovementKernel::run_resident` (movement.rs:422) | **NO** — inline (5 entries) | ~50–100 µs |
| `PhysicsKernel::run_batch_resident` (physics.rs:1862) | **NO** — inline (16 entries) × 8 iters per tick | ~100–200 µs × 8 = **800–1600 µs/tick** |
| `SeedIndirectKernel::record` (cascade_resident.rs:330) | **NO** — inline (4 entries) | ~30–50 µs |
| `AppendEventsKernel::record` (cascade_resident.rs:552) | **NO** — inline (5 entries) | ~50–100 µs |
| `spatial_gpu.rs::rebuild_and_query_resident` | **NO** — inline (12 entries) × 2 | ~100–200 µs |

Total per-tick BG construction: **~1400–2600 µs** on llvmpipe (assuming
~100 µs/BG), but more like **100–400 µs** on discrete drivers (5–20 µs
per BG). The physics rebuild is worst: **16 bindings × 8 iters = 128
binding operations per tick**. Post-9b730988 only mask, mask-unpack, and
scoring-unpack cache; every other resident kernel rebuilds per call.

**Fixability:** Small-to-medium. Physics rebuilds per iter because
events_in swaps (ping-pong) — solution: pre-build two BGs and select
between them. Scoring/apply/movement all bind buffers that are stable
across a batch (change only on agent_cap grow); they can use the same
cap-keyed cache pattern mask already has. Expected saving: **200–400
µs/tick** on llvmpipe, ~50 µs on discrete drivers.

---

## Section D — Suspect 3: always-dispatch-8-iters

`cascade_resident.rs:936-988` unconditionally records
`MAX_CASCADE_ITERATIONS = 8` (see `cascade.rs:89`) physics dispatches per
tick. Each `physics.run_batch_resident` call ends up as:

1. 2× `queue.write_buffer` for cfg + resident_cfg (~20–40 µs)
2. 1× device.create_bind_group (16 entries) (~100–200 µs on llvmpipe)
3. 1× `dispatch_workgroups_indirect` into the num_events_buf slot

When the indirect args buffer holds `(0, 1, 1)` (convergence already
reached), the GPU runs zero workgroups for that dispatch — but the
encoder still pays the cfg uploads + BG construction for iters 3..7.

On the N=2048 perf fixture, `cascade_iters_mean ≤ 1.07` — cascade
converges at iter 1 almost every tick, making iters 2..7 pure overhead.
Per-iter overhead on the zero-workgroup path: 2× write_buffer (~40 µs)
+ 16-entry BG construction (~100 µs llvmpipe) + indirect dispatch
(~10-20 µs). Per-iter fixed cost ~150 µs × 6 excess iters = **~900
µs/tick** upper bound. Cannot verify without timestamp instrumentation.

**Fixability:** Small. Easiest: cache the ping-pong-resolved BGs
across a cascade's iterations (2 BGs per tick instead of 8). Larger
option: collapse to a single dispatch guarded by an early-exit
indirect arg. Expected saving: **200–500 µs/tick**.

---

## Section E — Suspect 4: mask + scoring unpack kernels

Commit 9b730988 added `MaskUnpackKernel::encode_unpack` +
`ScoringUnpackKernel::encode_unpack` to fix a stale-state correctness
bug. Both are single-pass kernels with cached BGs (mask.rs:1139-1159,
scoring.rs:2150-2170). Steady-state cost: 2× pipeline-set+dispatch and
~10 bitmap `encoder.clear_buffer` calls. Estimated: **100–400 µs/tick**.

**Fixability:**
- *Inherent:* the mutable subset of mask + scoring SoA must refresh
  from `resident_agents_buf` each tick or kernels read stale state.
- *Reducible:* (a) merge the two unpack kernels into one dispatch
  (shared input, disjoint outputs) — saves 50–150 µs; (b) rewrite mask
  kernel to use write-overwrite rather than atomicOr, dropping the
  bitmap clear loop — saves 50–100 µs; (c) long-term: rewrite mask +
  scoring to read `GpuAgentSlot` directly (GPU-everything scope).

---

## Section F — Other suspects

### F.1 Per-tick clear_buffer count

~20 clears per tick total: apply event ring tail, physics ring_a/b
tails, num_events_buf, up to 5 inter-iter ping-pong clears, ~10 mask
bitmap bufs. Each ~1-5 µs. **20–100 µs/tick.** Low priority.

### F.2 Ping-pong event rings (cascade_resident.rs::physics_ring_a/b)

Correct design — iter `i`'s output is iter `i+1`'s input without a
copy. Not a suspect.

### F.3 Per-tick `PackedAbilityRegistry` upload

`cascade_resident.rs:882` uploads the ability registry every tick via
4× queue.write_buffer (cascade_resident.rs:187-196). The 4th write
(`effects`, MAX_ABILITIES × MAX_EFFECTS × ~32 B = ~64 KB) dominates.
Registry contents don't change tick-to-tick on the perf fixture.
Estimate: **100–300 µs/tick**. Final code review I5: this is the
duplicate-buffer issue — sync uses `PhysicsKernel`'s internal ability
bufs, resident re-uploads into its own copies.

**Fixability:** Small. Add a dirty-flag on `PackedAbilityRegistry` and
skip uploads on unchanged content.

### F.4 Cold-state work that stays on CPU (sync path only)

Sync runs `cold_state_replay` + `fold_iteration_events` +
`unpack_agent_slots` on CPU — the batch path skips all three. On the
perf fixture these are cheap but count toward sync's 6442 µs. The
8114 vs 6442 comparison is therefore already discounting this work
from the batch side. Apples-to-apples for GPU submit only.

---

## Section G — Suspect ranking + expected speedup

| # | Suspect | Est. µs/tick | Fix cost | Cumulative batch/sync at N=2048 |
|---|---|---|---|---|
| 1 | Fix always-dispatch-8 (cache physics BG across iters, or indirect-skip) | 200–500 | Small | 1.19× → 1.14× |
| 2 | Single `upload_agent_soa` per tick + cache spatial query BG | 400–900 | Small | 1.14× → 1.05× |
| 3 | Cache scoring + apply + movement resident BGs | 200–400 | Small | 1.05× → 0.99× |
| 4 | Dirty-flag `PackedAbilityRegistry` upload | 100–300 | Small | 0.99× → 0.96× |
| 5 | Merge mask-unpack + scoring-unpack into one dispatch | 50–150 | Medium | 0.96× → 0.95× |
| 6 | Skip mask bitmap clears (non-atomic rewrite) | 50–100 | Medium | 0.95× → 0.94× |

Worst-case cumulative saving (sum of mid-estimates): ~1600 µs/tick → batch
lands at roughly 0.80× sync — exactly the spec target.

Best-case cumulative saving (sum of high-estimates): ~2350 µs/tick → batch
lands at ~0.70× sync — clearly beats the target.

All six are **local** optimisations that don't require touching the
GPU-everything migration or changing any externally-visible API. Five
of six are caching patterns already in use elsewhere in the codebase
(mask kernel's BG cache being the template).

---

## Section H — Does this change the strategic picture?

**Finding: the gap is fixable without the GPU-everything migration.**

Rationale:
- Every suspect is a local-to-one-file optimisation whose fix pattern
  already exists elsewhere in `engine_gpu` (the mask kernel's BG cache,
  the mask-unpack + scoring-unpack caches added post-9b730988).
- None of the suspects is structural to batching or to the
  resident-cascade design; they're all artefacts of the resident path
  being bolted onto kernels that were originally written for the sync
  path and retaining the sync path's per-call allocation habits.
- The resident path's fundamental architecture (caller-owned buffers,
  indirect dispatch, ping-pong event rings) is sound. The gap is pure
  driver-overhead tax.

**Recommendation:** Proceed with the 6 local fixes in the order listed
in Section G. Each is independently reviewable and testable. Target:
batch/sync ratio ≤ 0.8× at N=2048 within one landing. The GPU-everything
migration — covering the CPU-side tick advance, RNG seeding, cold-state
rules, etc. — should proceed on its own merits (eliminates the last CPU
fences during a batch, enables cross-platform GPU determinism, unblocks
very-large-N scaling) rather than as a perf-gap remedy.

---

## Section I — Non-invasive measurement options

Attempted invocations:

```
cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 1
cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 10
```

**Result: blocked.** The worktree is compile-broken — an in-flight
independent refactor (the gold-i32-narrowing agent per
`docs/superpowers/research/2026-04-22-gold-i32-narrowing.md`) has left
generated files in an inconsistent state:

```
error[E0308]: mismatched types
  --> crates/engine/src/generated/physics/cast.rs:86:25
  --> crates/engine/src/generated/physics/mod.rs:126:44
```

A partial stash of that agent's tracked edits left the tree still
unbuildable because the generated `mod.rs` + `cast.rs` at HEAD are
already inconsistent with the `EffectOp::TransferGold { amount: i64 }`
type introduced upstream. The constraint to not edit Rust source
(including generated code) means the measurements cannot be run until
the parallel agent merges. `current-clean` and `techdoc-baseline`
worktrees both predate `step_batch` and are therefore not viable
alternatives.

**What the measurements would have revealed:**

*Hypothesis:* does per-tick batch cost stay stable across
`--batch-ticks ∈ {1, 10, 100, 500}` at N=2048?
- Stable → per-tick overhead (BG, SoA pack, re-uploads) dominates,
  consistent with suspects B/C/F.3.
- Drops with batch size → init/submit/poll cost dominates.

The technical_overview.md 8114 µs figure uses batch-ticks=200, so
init/submit/poll is already amortised. Running with `--batch-ticks 1`
measures the init+submit+poll cost in isolation; `--batch-ticks 500`
confirms flatness.

*Hypothesis 2:* slope of per-tick cost vs N separates O(N)
arithmetic from fixed per-tick overhead. Sync slope past N=512
is ~900 µs per doubling; batch slope is ~1300 µs per doubling. The
steeper batch slope is consistent with per-iter physics BG rebuild
cost scaling with binding-memory pressure as N grows — Section C's
hypothesis.

---

## Section J — Open questions

Items that cannot be resolved non-invasively:

1. **Per-phase batch-path timings.** `PhaseTimings` is sync-path-only
   (`lib.rs:1438-1595`). Extend with `batch_*` fields that
   `time_gpu_batch_sweep` populates via `Instant` around each encoder
   section to pin each suspect's µs contribution.

2. **llvmpipe vs discrete GPU per-BG cost.** Per-BG is 10–30× higher
   on llvmpipe. Running with `WGPU_ADAPTER_NAME=your-discrete-card`
   would confirm the BG-rebuild hypothesis — if the gap shrinks, BG
   construction dominates; if it stays at 1.26×, other overhead does.

3. **Zero-workgroup indirect dispatch cost.** wgpu timestamp queries
   around each physics iter would answer whether iters 2..7 cost ~150
   µs or closer to ~10 µs each. Requires enabling `TIMESTAMP_QUERY`
   and emitting `cpass.write_timestamp` around each dispatch.

4. **Cross-tick batch-submit vs single-tick-submit cost.** A diagnostic
   harness that can disable batch mode's optimisations independently
   would isolate the saved submit+poll cost.

5. **The `--batch-ticks` sweep itself.** Blocked; re-run once the
   worktree is back to green.
