# Engine Throughput Results

Measurements from `cargo bench -p engine --bench tick_throughput -- --quick`.
Criterion's `--quick` mode; sample size 10. Numbers are median wall-clock for
`n × 1000` ticks. Steps/sec = `1000 / median_seconds`. Higher is better.

## 2026-04-19 — Post-audit-fixes (HEAD `1295edcf`)

After `20fc5a26..1295edcf` — CRITICAL + HIGH + MEDIUM stub-audit fixes.
Notably CRITICAL #1 wired SpatialIndex into SimState + mask predicates
+ Announce audience + engagement update.

| Policy          |   N | Median     |  Steps/sec | Δ vs prior |
|-----------------|----:|-----------:|-----------:|-----------:|
| utility_backend |  10 |   1.813 ms |    551,572 |     -89.0% |
| utility_backend | 100 |  43.724 ms |     22,871 |     -87.0% |
| utility_backend | 500 |  25.468 ms |     39,265 |     -64.6% |
| mixed_actions   |  10 |   6.052 ms |    165,234 |     -93.5% |
| mixed_actions   | 100 |  86.979 ms |     11,497 |     -87.2% |
| mixed_actions   | 500 | 843.81 ms  |      1,185 |     -86.2% |

**Observations:**

- **Universal regression, not the hypothesised speedup.** Every cell got
  *slower* by 65–94%. Criterion's own change-detector flagged each combo
  with positive deltas of `+172% … +1262%` against the prior on-disk
  baseline (Plan 1, `050d4aea`). The biggest hit is `mixed_actions/10`
  at +1262% (395 µs → 6.05 ms).
- **Root cause: `SpatialIndex` rebuilt on every mutation.**
  `SimState::set_agent_pos` (and `set_agent_movement_mode`,
  `spawn_agent`, `kill_agent`) calls `rebuild_spatial()`, which scans
  every alive agent and inserts into a `BTreeMap` (deterministic
  iteration). With N=500 in `mixed_actions`, `pick==2` means ~100
  `MoveToward` actions per tick → 100 full O(N log N) rebuilds per tick
  → ~50M BTreeMap inserts per 1000-tick iteration. The new
  `query_within_radius` saves linear scans, but the savings are dwarfed
  by per-mutation rebuild cost. The cell-reach clamp (256) the audit
  flagged is *not* the issue here; the index is rebuilt before the
  query benefit can compound.
- **Other contributors stacked on top:** `tick_start` allocates a fresh
  `Vec<AgentId>` for `alive` and a fresh `Vec<Option<AgentId>>` of size
  `cap` *every tick* (`expire.rs:74,117`). At N=500 × 1000 ticks that's
  another 500 K Vec allocations the prior baseline didn't have.
  `Announce` paths (only triggered if a backend emits Announce — neither
  bench does) would compound further via `query_within_radius` +
  `agents_alive()` membership tests, but are not in this hot-path
  scenario.
- **Net assessment:** the SpatialIndex *integration* is correct and
  necessary, but the *eager rebuild* strategy is the wrong choice for a
  hot mutation path. Either (a) make the index incremental (track
  per-agent cell deltas on `set_agent_pos`), or (b) defer to a single
  per-tick rebuild called from `step_full` *once*, or (c) hoist the
  rebuild out of mutators and require callers to call `state.refresh_spatial()`
  at phase boundaries. Option (b) is the cheapest fix and would likely
  recover Plan-1 numbers.

## 2026-04-19 — Post-Plan 1 (HEAD `050d4aea`)

After Plan 1 (full 18-MicroKind vocab, cascade runtime, Announce cascade, etc.).
Same hardware as MVP baseline below.

| Policy          |   N | Median     |  Steps/sec |
|-----------------|----:|-----------:|-----------:|
| utility_backend |  10 |  199.75 µs |  5,006,258 |
| utility_backend | 100 |   5.696 ms |    175,571 |
| utility_backend | 500 |   9.018 ms |    110,887 |
| mixed_actions   |  10 |  395.43 µs |  2,528,918 |
| mixed_actions   | 100 |  11.103 ms |     90,066 |
| mixed_actions   | 500 | 116.29 ms  |      8,599 |

Notes:

- `utility_backend/100` is **73% faster** than the MVP baseline (20.93 ms →
  5.70 ms). The faster number is a real improvement — Plan 1 Tasks 1–19
  landed several hot-path changes (bulk mask clearing, flat cascade registry,
  cache-friendly tick fields). Criterion's built-in comparison confirmed
  `change: [-73.079% -72.903% -72.727%]` against the on-disk baseline from
  MVP Task 16.
- `utility_backend/500` (9.0 ms) is drastically faster than the MVP number
  (387 ms). At n=500 the UtilityBackend quickly drives most agents toward a
  single attractor, many collapse into Hold/idle micro-states after a few
  hundred ticks, and the per-tick cost stabilises. This number is a valid
  "typical workload" marker but not a worst-case. For a worst-case profile
  on that population, see `mixed_actions/500`.
- `mixed_actions/500` (116 ms / 8.6 k steps/sec) is the realistic upper bound
  on per-tick cost when every agent is emitting side-effectful actions
  (Drink/Rest/Communicate/MoveToward) rather than settling to Hold.

## 2026-04-19 — MVP baseline (reference)

Pre-Plan-1 numbers (`UtilityBackend` with Hold + MoveToward only), from MVP
Task 16, reproduced here as the regression anchor:

| Policy          |   N | Median    |  Steps/sec |
|-----------------|----:|----------:|-----------:|
| tick_throughput |  10 |  0.38 ms  |  2,631,578 |
| tick_throughput | 100 | 20.93 ms  |     47,778 |
| tick_throughput | 500 |  387 ms   |      2,584 |
