# 2026-05-09 — DSL/runtime stress ceilings

Findings from the stress-test sweep (plan
`docs/superpowers/plans/2026-05-09-stress-test-runtime-ceilings.md`).
Each fixture isolates one suspected ceiling so we can report concrete
numbers (not "it gets slow at scale") to whoever owns the next
optimization slice.

The driver wraps every per-cap run in `std::panic::catch_unwind` (P10).
A failed cap surfaces as `"panic": "<msg>"` in the summary line, not as
an aborted parent harness.

## Fixture B — cast_density sweep

Fixture: `assets/sim/stress_cast_density.sim` —
`damage 0.0 in spread(1.5, 256)` driven from a PerAgent physics rule
(`apply_ability agents.level(self) by self target self`). All agents
seeded inside a single 27-cell hash bin (positions perturbed by P5
PCG inside `[-1.5, 1.5]³` around the world origin).

Per cast: up to 256 `EffectDamageApplied` records (the WGSL bitonic-
sort scratch's 256-slot ceiling, see
`crates/dsl_compiler/src/cg/emit/wgsl_body.rs::area_kind == 4u`
arm). Per tick: agent_cap × ~(in-radius candidate count). The
in-radius hit ratio rises with agent_cap because the per-volume
density rises (constant world-bin volume).

Driver: `crates/stress_cast_density_runtime/src/bin/stress_cast_density_app.rs`.
Median + p99 are over `tick_budget = 100` ticks per cap. Tick 0 is
universally slow because of GPU pipeline-cache misses (first-time
shader compilation) — that hit shows up in the p99 column but is
amortised out of the median.

| agent_cap | ticks_completed | median per-tick (us) | p99 per-tick (us) | max ring high-water | first overflow at tick | spread sort us (median) | breakpoint                                                  |
|-----------|-----------------|----------------------|-------------------|---------------------|------------------------|-------------------------|-------------------------------------------------------------|
| 500       | 100             | 1 250                | 250 069           | 66 298              | 0                      | 1 250                   | RING (cap 65 536, exceeded by 762)                          |
| 1 000     | 100             | 1 211                | 249 304           | 225 841             | 0                      | 1 211                   | RING (3.4× cap)                                             |
| 2 000     | 100             | 1 337                | 250 661           | 510 363             | 0                      | 1 337                   | RING (7.8× cap)                                             |
| 4 000     | 100             | 1 593                | 255 054           | 1 024 000           | 0                      | 1 593                   | RING (15.6× cap)                                            |
| 16 000    | 100             | 3 290                | 253 706           | 4 096 000           | 0                      | 3 290                   | RING (62.5× cap)                                            |
| 64 000    | 100             | 14 661               | 267 627           | 16 384 000          | 0                      | 14 661                  | RING (250× cap; per-cast Spread saturates at max=256)       |

(`max_ring_high_water` is the value the dispatcher's atomicAdd
returned for `event_tail` after every alive agent emitted its in-
radius records. The WGSL append site only stores the record when
`_slot < 65536u`; tail keeps incrementing past the cap, so the
high-water minus 65 536 is the silent-drop count.)

### Saturation point

| agent_cap | ring_high_water | overflowed? |
|-----------|-----------------|-------------|
| 100       | 2 662           | no          |
| 200       | 10 590          | no          |
| 250       | 17 126          | no          |
| 300       | 24 722          | no          |
| 400       | 42 384          | no          |
| 425       | 48 315          | no          |
| 450       | 54 286          | no          |
| 475       | 59 947          | no          |
| 480       | 60 974          | no          |
| 500       | 66 298          | **YES** (first overflow)             |

So the ring first overflows somewhere in `[480, 500]` at this density
seed (`POSITION_SEED = 0x05FE_55BB`, `SEED_HALF_EXTENT = 1.5`). At
agent_cap=500 the dispatch silently dropped 762 chronicle records
(0.7% of the record bursts past the cap). At agent_cap=64 000 it
drops 16 318 464 records per tick — 99.6% of the casts are silently
swallowed.

### Findings

- **Chronicle ring overflow first lands at agent_cap ≈ 480–500** for
  the cast-density fixture. This is the point where `event_tail`'s
  atomicAdd crosses 65 536. Past this point WGSL writes degrade
  silently; only the per-tick tail readback surfaces the overflow.
- **Saturation cast-rate ceiling: ~256 in-radius candidates per
  cast** (the bitonic-sort scratch cap in
  `cg/emit/wgsl_body.rs::area_kind == 4u`). Density above 256 in any
  cast collapses to "the lowest 256 AgentIds in radius" — this is
  the documented spec, not a bug, but the cap is the determinism-
  preserving choice rather than a true throughput limit.
- **Spread bitonic sort scales gracefully across the agent range we
  tested.** At agent_cap=64 000 the per-tick wall clock is ~12× the
  agent_cap=500 baseline (14 661 µs vs 1 250 µs), despite a 128×
  agent-count increase — the per-cast sort is bounded at 256
  elements (= 8 192 compare-swaps), so total per-tick sort work
  scales O(agent_cap) not O(agent_cap²). The GPU's wavefront
  concurrency absorbs the linear growth fine.
- **The chronicle ring is the binding constraint, not the sort.**
  Every cap above 500 trips the cap on tick 0; the cap is
  proportional to the cast-density × agent_cap product. Three
  candidates for raising the 65 536-slot cap:
  1. **Bump to 1 048 576 (1M slots)** — costs ~40 MB of GPU memory
     per fixture. Kicks the saturation point out by 16× to
     agent_cap ≈ 8 000 at this density. Lowest-effort change; the
     constant lives in two literals (`if (_slot < 65536u)` arm
     chain in `wgsl_body.rs` + `EVENT_RING_CAP_SLOTS` in
     `engine/src/gpu/event_ring.rs`).
  2. **Per-event-kind ring fanout** — split the single 65 536 ring
     into N rings keyed by EventKindId so EffectDamageApplied
     casts don't share slots with EffectHealApplied or whatever
     else lands on the same dispatch. Same total memory but
     decouples the saturation point per kind. Higher effort
     (per-kind atomic + per-kind staging buffers).
  3. **Multi-tick spread** — when overflow detection fires, defer
     the late records to the next tick by stashing them in a
     spillover ring. Highest effort (requires the dispatcher to
     persist its iteration state); the only path that actually
     keeps the casts behaviorally lossless.
- **Per-cast cap of 256** is hit when in-radius candidates exceed
  256 — this happened at agent_cap=64 000 where every cast
  collected 256 (= 16 384 000 records / 64 000 casts). The bitonic
  sort still only handles 256 slots; raising the cap requires a
  larger per-thread `array<u32, N>` (N must be a power of 2, doubles
  the sort's `log² N` stages every step). 512 slots ≈ 9 stages × 9
  inner = 81 stages × 256 compare-swaps = 20 736 ops per cast vs
  today's 8 192. Worth measuring before committing to the change.
- **First-tick latency dominates p99** across every cap — pipeline
  compile (~250 ms uniformly across caps) is the same wall cost
  whether agent_cap is 500 or 64 000. p99 reports it; median does
  not. Future stress runs should prepend a "warmup" tick that's
  stripped from the metric set.

### Next stride opportunities (not blocking this slice)

- Add a `--warmup-ticks N` flag to the driver to amortise the GPU
  pipeline-cache cost out of the p99 column.
- Wire a `--seed` flag so the cast-density saturation point can be
  characterised at multiple density profiles (today the fixture pins
  one P5 seed).
- Re-run with `RUST_MIN_STACK=33554432 cargo test -p
  stress_cast_density_runtime --release --lib` whenever the AOE
  Path B emit changes — the `tick_advances_under_max_density` pin
  asserts `kind == 26` (`EffectDamageApplied`) records land in the
  ring with > 0 count.
