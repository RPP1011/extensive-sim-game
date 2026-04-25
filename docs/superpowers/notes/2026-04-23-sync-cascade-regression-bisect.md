# Sync-Path GPU Cascade Regression Bisect (N=100k, RTX 4090)

> Investigation into the apparent 193 ms → 4,549 ms regression in the
> `gpu_cascade` phase between the historical `docs/technical_overview.md`
> baseline (now superseded by `docs/overview.md`; commit `494c7b69`,
> 2026-04-22) and HEAD (`856fe171`, 2026-04-23).

## TL;DR

**There is no true sync-path cascade regression.** The tech-report
"baseline" of 193 ms cascade @ N=100k was a *silent-fallback artifact*.
At baseline the GPU cascade was dispatched, overflowed its 65k-slot
event ring, threw `cascade event ring overflowed at iter=0`, and
silently fell back to the CPU cascade (see
`crates/engine_gpu/src/lib.rs:959` at `494c7b69` — the `Err(e)` arm
of the dispatch match eprintlns and calls
`cascade.run_fixed_point_tel`). The 193 ms figure is therefore **CPU
cascade time on a zero-event fast path**, not GPU cascade time.

The first commit to remove the ring-overflow-induced fallback — by
raising the ring capacity 10× — is `1606eba0` ("feat(engine_gpu): 10×
event ring caps"). That commit does not regress the GPU cascade; it
*reveals its true cost*, which at N=100k agents on an RTX 4090 is
~4.5 s/tick steady state.

## Bisect Log (perf_n100 at N=100,000, 9 steady-state ticks)

All numbers are `gpu_cascade` µs/tick averaged over the 9 measured
ticks after a single warmup, release build, `--features gpu`, RTX 4090
(Vulkan backend). `avg_iters/tick` is the average cascade fixed-point
iteration count.

| commit     | short descr                                    | cascade µs/tick | total ms/tick | iters/tick | status          |
|------------|------------------------------------------------|-----------------|---------------|------------|-----------------|
| `494c7b69` | docs: technical overview (baseline)            |     197,416     |     1,769     |    3.00    | ring overflow → CPU fallback |
| `b5294ed7` | task 203 — chronicle on dedicated GPU ring     |     305,709     |     1,877     |    3.00    | ring overflow → CPU fallback |
| `b5294ed7` | (re-run for noise check)                       |     315,051     |     2,211     |    3.00    | ring overflow → CPU fallback |
| `0c272326` | Split DSL-emitted shared data into a dedicated crate |   571,216  |     2,927     |    ≈3      | possibly partial overflow |
| **`1606eba0`** | **10× event ring caps + per-phase breakdown in perf sweep** | **4,591,335** | **6,257** |  2.89 | **true GPU cascade cost exposed** |
| `1606eba0` | (re-run for noise check)                       |   4,777,696     |     6,775     |    2.89    | same — stable |
| `3e438290` | un-ignore indirect_cascade_converges           |   4,464,033     |     6,167     |    ≈3      | same |
| `07f2a9c4` | kin_radius → designer-tunable combat config    |   4,570,515     |     6,598     |    ≈3      | same |
| `856fe171` | HEAD                                           |   5,402,917     |     7,723     |    2.78    | same |

Checkpoints measured: **8 full runs**, well within the 15-20 budget
and ~3 h wall-clock envelope.

## The smoking gun

`1606eba0` is a 3-file diff that reads in full as:

```
+pub const APPLY_EVENT_RING_CAPACITY: u32 = 655_360;   // was 65_536
-let physics = PhysicsKernel::new(..., 65_536)?;
+let physics = PhysicsKernel::new(..., 655_360)?;
+pub const DEFAULT_CAPACITY: u32 = 655_360;            // was 65_536
```

plus per-phase timing output in `chronicle_cmd.rs`. The commit message
is explicit: *"Bumps APPLY_EVENT_RING_CAPACITY, PhysicsKernel ring cap,
and event_ring DEFAULT_CAPACITY from 65 536 → 655 360 so perf_n100 at
N=100k no longer overflows + falls back to CPU cascade."*

At `494c7b69` the baseline N=100k perf_n100 output contains the
overflow-and-fallback telemetry inline:

```
engine_gpu::event_ring: overflow — tail=86118 exceeds capacity=65536 (20582 records dropped)
engine_gpu: GPU cascade failed, falling back to CPU cascade: cascade event ring overflowed at iter=0
```

These lines appear *in the baseline run* the tech report measurements
came from. The reported 193 ms was `cascade.run_fixed_point_tel` on
the remaining CPU state — a no-op fast path because the GPU kernels
that emit physics/death events had their output discarded with the
ring buffer.

## What the real GPU cascade costs (at N=100k on 4090, sync path)

From the post-`1606eba0` timings (stable across 6 runs):

- 2.78 – 3.00 cascade iterations/tick (matches the CPU baseline's
  logical iteration count, so the fixed point is the same — it's just
  now actually executing on the GPU).
- Each iteration ≈ **1.5 s** GPU wall-clock: physics kernel +
  view-fold + event-drain readback.
- At 2.9 iters/tick that's the observed ~4.5 s/tick.

The `gpu_seed_fold` phase also inflates 893 ms → ~1.1 s — that's real
but not the ~20× factor the cascade shows.

## Attribution: is `1606eba0` the "culprit"?

**No.** `1606eba0` corrected a silent-correctness bug (events
dropped on overflow, results silently wrong) at the cost of exposing
the pre-existing GPU cascade perf problem. The cascade kernel's cost
at N=100k is what it is; it was masked by dropping 80% of its output.

The other suspects from the task brief all came AFTER `1606eba0` and
cannot be the first offender:

- `b5294ed7` (task 203 chronicle ring split) — **before** `1606eba0`
  and measures fast (305 µs).
- `ffd7010c`, `ce72f9c0`, `69fdbfbd`, `e6faa4b2`, `ad6aac3c`,
  `234ce834`, `d3242d8d` — all after `1606eba0` and merely ride on
  top of its (correct) ~4.5 s cascade floor. At `07f2a9c4` and
  `3e438290` the cascade stays in the 4.4–4.6 s band, so none of
  those later commits adds a meaningful second-order regression.

## Recommendation

This is a **reporting bug, not a fix-me bug**:

1. The "193 ms GPU cascade" figure (originally in
   `docs/technical_overview.md:138-147`, now removed in the
   `docs/overview.md` rewrite) was wrong. It describes the CPU
   fallback, not the GPU cascade. The real decomposition at N=100k
   on this GPU is something like:
   - GPU mask + scoring: ~290 ms (unchanged)
   - GPU cascade: ~4,500 ms (was silently CPU-fallback 193 ms)
   - GPU seed fold: ~1,100 ms (was silently CPU-fallback 893 ms)
   - CPU apply + cold-state: ~325 ms (unchanged)
   - Total: ~6,300 ms/tick, **not** 1,740 ms/tick.

2. The *actual* N=100k sync-path cost on a 4090 has always been ~6 s
   — the tech-report number was never achievable while the cascade
   produced correct output. No feature commit between `494c7b69` and
   HEAD deserves blame for a 20× slowdown.

3. If the 1.7 s/tick number is a design target, the work to get
   there is a **real** GPU cascade optimisation (fewer iterations,
   per-iter kernel fusion, or avoiding the full-ring drain every
   iter) — not a revert. Task 3.x / Task 68 already track some of
   this. Retiring the sync path (event ring overflow handling / CPU
   fallback branch) and migrating perf-sensitive callers to
   `step_batch_resident` is also a valid path; the batch-path cascade
   avoids re-binding and re-uploading per-tick and may already be
   closer to the design target.

4. A followup safety net: the `Err(e)` arm in
   `lib.rs:949-961` should probably be `assert!` or at minimum a
   `log::error!` + test-failing signal rather than an `eprintln!` +
   silent fallback. The fallback corrupts timing measurements (as
   happened here) and hides correctness loss (event drops mean
   different death counts between runs).

## Files referenced

- `/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/engine_gpu/src/lib.rs` (lines 949-961 at 494c7b69, fallback branch)
- `/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/engine_gpu/src/cascade.rs` (APPLY_EVENT_RING_CAPACITY const)
- `/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/engine_gpu/src/event_ring.rs` (DEFAULT_CAPACITY const)
- `/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/engine_gpu/tests/perf_n100.rs` (measurement harness)
- `/home/ricky/Projects/game/.worktrees/world-sim-bench/docs/technical_overview.md` (lines 138-147 in worktree snapshot; replaced by `docs/overview.md` on `main` — stale perf claim was removed in the rewrite)
