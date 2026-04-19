# Engine Throughput Results

Measurements from `cargo bench -p engine --bench tick_throughput -- --quick`.
Criterion's `--quick` mode; sample size 10. Numbers are median wall-clock for
`n × 1000` ticks. Steps/sec = `1000 / median_seconds`. Higher is better.

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
