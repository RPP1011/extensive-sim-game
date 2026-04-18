# World Sim SIMD Targets — 2026-04-18

Initial diagnostic data captured. This document is meant to be refined as
larger-scale runs land; the current entries are from the 300-tick small-world
smoke run (10 NPCs + 1 settlement).

## Summary of diagnostic setup

- Instrumentation: `profile-systems` feature, 131 systems timed per tick
- Output: `generated/baselines/smoke.json` (300 ticks, small-world preset)
- Method: ranked by `total_ns` across all ticks; `ns/call` reveals per-invocation cost

## Top systems (smoke run, small-world)

| rank | system | total_ms | calls | ns/call | % of wall |
|---|---|---|---|---|---|
| 1 | `scan_all_npc_resources` | 24022 | 300 | 80M | 91% |
| 2 | `structural_tick` | 2430 | 30 | 81M | 9% |
| 3 | `update_agent_inner_states` | 0.21 | 300 | 714 | 0% |
| 4 | `evaluate_and_act` | 0.17 | 300 | 570 | 0% |
| 5 | `advance_movement` | 0.11 | 300 | 372 | 0% |

## Findings

**The dominant cost is voxel/spatial work, not simulation math.** At small-world
scale, `scan_all_npc_resources` (voxel resource discovery) and `structural_tick`
(unsupported voxel collapse) together consume 100% of the wall time. Neither is
a SIMD candidate:

- `scan_all_npc_resources` — HashMap lookups + per-NPC voxel range queries
- `structural_tick` — voxel graph traversal, branchy support analysis

**Simulation-side systems are fast at this scale.** `advance_movement` at 372ns/call
and `evaluate_and_act` at 570ns/call don't register as SIMD candidates because
they process ~10 NPCs. To find SIMD wins in the sim-side code, the bench must
capture data from a larger population — at 10K-50K entities, the f32 arithmetic
systems (movement, HP updates, economy) should dominate and the voxel work
becomes proportionally smaller.

## Runtime-cost finding (not SIMD, but plan-relevant)

During baseline capture, the default-world (2010 entities, 10 settlements) ran
at ~14 seconds per tick under `profile-systems`. Without the feature, prior
measurements suggested ~400 ticks/sec. The feature's measured overhead
dramatically exceeds the "≤1%" estimate in the original spec.

Likely cause: ~580 per-settlement timer pairs per tick × 10 settlements + the
HashMap insertion in `thread_record`. The overhead scales linearly with
settlement count and dispatched-system count, and real-world settlement count
is higher than the "~60 active systems" assumed in the spec.

**Recommendation:** For multi-thousand-tick baselines, either:
- Capture short (50-200 tick) baselines and extrapolate, or
- Swap `thread_record`'s `HashMap` for a `Vec<(name, ns, calls)>` with linear
  append + post-tick fold (amortized O(1) with no hashing during the hot path), or
- Add a `--profile-sample-rate N` flag that records every Nth call, or
- Run without `profile-systems` for aggregate timing + enable only for targeted runs

## Next concrete steps (to add real SIMD targets here)

1. **Lower profile-systems overhead** — flip the thread-local from `HashMap` to
   `Vec<(interned_name_idx, ns, calls)>`. Estimated 5-10× speedup at the
   instrumented path.
2. **Capture baselines at 10K + 50K** — once overhead is fixed, a 200-tick
   baseline at 10K entities completes in minutes. The ~60-system postapply
   block will produce ranking with enough samples to identify real candidates.
3. **Check flamegraphs** — `scripts/perf_bench.sh` is ready; it wasn't
   exercised in this session. Intra-system hot lines are the SIMD-target
   signal that per-system timing can't see.

## Non-targets (already disqualified)

- `scan_all_npc_resources` — pointer-heavy voxel traversal, not SIMD-able
- `structural_tick` — graph traversal, not SIMD-able
- `delta::merge_deltas` — HashMap-based merge, documented anti-candidate
  in the original spec
