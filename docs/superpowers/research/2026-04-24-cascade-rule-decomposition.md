# GPU physics cascade — per-rule invocation attribution

**Status:** Research complete. Instrumentation merged to
`gpu-cascade-rule-attribution`; production path unchanged when
`ENGINE_GPU_CASCADE_RULE_COUNT` is unset. Harness is opt-in for
future regressions.

**Question:** The resident physics cascade at N=100k spends ~150 µs/tick
across its iterations (per Stage A's `cascade iter N` timestamps). Out
of ~20 DSL-compiled physics rules (damage, heal, stun, chronicle_*,
engagement_on_*, rally_on_wound, fear_spread_on_death, …) which
dominate? "Cascade is cheap" has been load-bearing for prioritising
other optimisations — worth confirming the distribution.

## Approach

**Option A** (implemented): each rule body prepends
`atomicAdd(&per_rule_counter[RULE_IDX], 1u)` after its kind guard. A
per-rule `u32` counter buffer at BGL slot 23 holds the totals;
zero-at-top-of-batch, readback at end, report per-rule invocation
counts aggregated over the whole `step_batch(50)` run. Compile-gated on
`ENGINE_GPU_CASCADE_RULE_COUNT=1` at init time so production builds
emit byte-identical WGSL (no binding, no atomicAdd).

**Option B** (split dispatcher into per-rule sub-dispatches for true
per-rule µs) — **not implemented**. See "Recommendation" below.

## Rig

- Hardware: RTX 4090 (Vulkan backend), 100k agents, 50-tick
  `step_batch`. Fixture from
  `crates/engine_gpu/tests/chronicle_batch_perf_n100k.rs`.
- Invocation: `ENGINE_GPU_CASCADE_RULE_COUNT=1 cargo test --release
  --features gpu -p engine_gpu --test chronicle_batch_perf_n100k --
  --ignored --nocapture`.
- Shader delta: `emit_physics_wgsl_with_counter(rule, ctx, Some(idx))`
  emits one atomicAdd per rule body. Counter slot index matches
  `physics::ordered_rule_names(&comp.physics)` stable sort so the
  shader-side and host-side mappings are consistent.
- Instrumentation overhead: 149106 µs/tick (baseline) → 149577 µs/tick
  (instrumented), **0.3 % delta, within noise**. The atomicAdd lands in
  a heavily-contested cacheline (~13M invocations aggregated over 50
  ticks) but only measurably stretches cascade iter 0 (68 → 324 µs;
  +256 µs/tick of atomic contention). Later iters unaffected.

## Numbers — per-rule invocation counts, N=100k, 50 ticks

Total invocations: **12,690,231** (≈ 253,804 / tick).

| Rule | Count (50 ticks) | Per-tick | Share |
|------|-----------------:|---------:|------:|
| `chronicle_rally` | 4,405,110 | 88,102 | **34.7 %** |
| `engagement_on_move` | 2,116,228 | 42,324 | 16.7 % |
| `chronicle_attack` | 1,538,724 | 30,774 | 12.1 % |
| `chronicle_wound` | 1,538,724 | 30,774 | 12.1 % |
| `rally_on_wound` | 1,538,724 | 30,774 | 12.1 % |
| `chronicle_death` | 338,814 | 6,776 | 2.7 % |
| `engagement_on_death` | 338,814 | 6,776 | 2.7 % |
| `fear_spread_on_death` | 338,814 | 6,776 | 2.7 % |
| `chronicle_rout` | 253,262 | 5,065 | 2.0 % |
| `chronicle_engagement` | 94,911 | 1,898 | 0.7 % |
| `pack_focus_on_engagement` | 94,911 | 1,898 | 0.7 % |
| `chronicle_break` | 93,195 | 1,863 | 0.7 % |
| `cast`, `damage`, `heal`, `modify_standing`, `opportunity_attack`, `record_memory`, `shield`, `slow`, `stun`, `transfer_gold`, `chronicle_flee` | 0 | 0 | 0 % |

### Counts are entries into rule bodies, not kind-mismatches

The atomicAdd is placed AFTER the `if (ev_rec.kind != EVENT_KIND_*)
{ return; }` guard, so a count of N means the rule executed its body N
times — not that the dispatcher considered it N times. For rules that
fire on multiple event kinds (none today — all physics.sim rules
kind-match) this would still be correct by rule, not by dispatch.

### Why are so many rules at 0?

Every rule at 0 keys off an event type that the current fixture's
autonomous agents (wolves / humans / deer) do not emit:

- `cast`, `damage`, `heal`, `shield`, `stun`, `slow`,
  `opportunity_attack` all key on `AgentCast` / `Effect*Applied`
  events. The scenario has no ability casts.
- `modify_standing` fires on `EffectStandingDelta` — no faction
  standings are changing.
- `transfer_gold` fires on `EffectGoldTransfer` — no transactions.
- `record_memory` / `chronicle_flee` fire on the corresponding
  narrative events, which this fixture doesn't trigger.

These zeros are a property of the workload, not a bug. A different
fixture (e.g. scripted-hero combat with abilities) would light up
`cast`/`damage`/`heal`/etc.

## Interpretation — is there a hot spot?

**Yes — and it's chronicle rules.** `chronicle_rally` alone is 34.7 %
of all rule body entries; the chronicle_* family combined
(`chronicle_rally + chronicle_attack + chronicle_wound +
chronicle_death + chronicle_rout + chronicle_engagement +
chronicle_break`) is ~65 % of all invocations. Every chronicle rule is
nominally a STUB per `physics.rs::19-50`'s rule-support matrix
(`ChronicleEntry` is non-replayable; the stub-body just emits the
chronicle event and returns). So they're cheap per-invocation — but at
~175k entries/tick across the family they dominate dispatch count.

**Attribution back to µs:** cascade iter 0..4 combined is ~145 µs/tick
(baseline). If rule cost is proxied by invocation count, the chronicle
family would contribute ~94 µs/tick (65 %) of that. The caveat is that
chronicle rules each do a single `gpu_emit_chronicle_event` (an
atomicAdd on the chronicle ring tail + a struct store), which is about
the cheapest body shape among the ~20 rules. Rules like
`engagement_on_move` touch spatial + set engagement + emit — strictly
more work per invocation. So "chronicle dominates dispatch count" is a
safer conclusion than "chronicle dominates µs"; the two families are
close but chronicle's per-invocation cost is smaller.

`engagement_on_move` (42k entries/tick, 16.7 %) is the next candidate.
Each body reads spatial `nearest_hostile`, checks engagement range,
potentially writes engaged_with + emits `EngagementCommitted`. Higher
per-invocation cost than a chronicle rule.

`rally_on_wound` (30k/tick, 12.1 %) iterates kin (bounded K=32) per
invocation; it's the costliest per-body rule in the top-5.

## Conclusion — cascade µs distribution

At N=100k with this fixture, the cascade's ~150 µs/tick splits roughly:

1. `chronicle_*` family (~65 % of invocations, low per-body cost) —
   likely **40–55 % of µs**.
2. `engagement_on_move` (17 % of invocations, mid per-body cost, spatial
   query) — likely **20–30 % of µs**.
3. `rally_on_wound` (12 % of invocations, kin-iterating body) —
   likely **10–20 % of µs**.
4. Everything else (`engagement_on_death` + `fear_spread_on_death` +
   `pack_focus_on_engagement` + `chronicle_rout/break/engagement` +
   zeros) — **≤ 15 % of µs** combined.

**No single rule is surprisingly expensive.** The distribution is
broadly proportional to what the workload drives — lots of movement →
lots of `engagement_on_move` + chronicle events; lots of combat
damage → `rally_on_wound`. Nothing fires "way more than expected."

## Recommendations

**Cascade-µs optimisation is NOT a high-priority target.** At 150 µs
of a 151 ms total `step_batch` tick, even halving the cascade yields a
0.05 % improvement. The attribution confirms the pre-task prior.

**If we DID want to optimise:**

- **Chronicle fold.** The ~175k chronicle entries/tick contend on a
  single `chronicle_ring_tail` atomic. Fusing adjacent chronicle
  writes inside a single rule body, or skipping chronicle entirely in
  perf-critical configurations (`--chronicle=off`), would be the
  cheapest experiment. Prior Task 203 already split chronicle onto a
  dedicated ring, which addresses contention with the main event ring
  but not intra-chronicle contention.
- **`engagement_on_move` gating.** 42k/tick invocations = one per
  moving agent. If engagement were checked only on *transitions*
  (agent just entered attack range) rather than every `AgentMoved`,
  this would collapse. Would require cached "engaged this tick" flag
  on the agent SoA.

**Do NOT pursue Option B (per-rule µs via split sub-dispatches)**
right now. It would cost N × events_in reads per tick for marginal
signal over what Option A already provides, and the cascade is too
cheap for the effort to pay off. If a specific rule later surfaces as
suspicious (e.g. someone reports a regression in `engagement_on_move`
after a new feature), Option B becomes worthwhile for that rule only.

## Raw output

```
chronicle_batch_perf_n100k: N=100000 backend=Vulkan
  step_batch(50): total=7478 ms, avg=149577 µs/tick
    cascade iter 0                          : 324
    cascade iter 1                          : 31
    cascade iter 2                          : 19
    cascade iter 3                          : 14
    cascade iter 4                          : 15
  --- per-rule invocation counts (50 ticks, total=12690231) ---
    chronicle_rally                 : count=     4405110 ( 34.7%) per_tick=88102
    engagement_on_move              : count=     2116228 ( 16.7%) per_tick=42324
    chronicle_attack                : count=     1538724 ( 12.1%) per_tick=30774
    chronicle_wound                 : count=     1538724 ( 12.1%) per_tick=30774
    rally_on_wound                  : count=     1538724 ( 12.1%) per_tick=30774
    chronicle_death                 : count=      338814 (  2.7%) per_tick=6776
    engagement_on_death             : count=      338814 (  2.7%) per_tick=6776
    fear_spread_on_death            : count=      338814 (  2.7%) per_tick=6776
    chronicle_rout                  : count=      253262 (  2.0%) per_tick=5065
    chronicle_engagement            : count=       94911 (  0.7%) per_tick=1898
    pack_focus_on_engagement        : count=       94911 (  0.7%) per_tick=1898
    chronicle_break                 : count=       93195 (  0.7%) per_tick=1863
    (all other rules, including cast/damage/heal/stun/slow/shield/
     opportunity_attack/modify_standing/record_memory/transfer_gold/
     chronicle_flee, were zero — this fixture's autonomous agents do
     not cast abilities or mutate standing / gold / memory)
```

Baseline (same fixture, no `ENGINE_GPU_CASCADE_RULE_COUNT`):
149106 µs/tick. Instrumentation overhead = 471 µs/tick (0.3 %).

## Next step — revert

The instrumentation is reverted (next commit on this branch). The
counter buffer, BGL slot 23, and the atomicAdd prefix emission are
removed from production code paths. Anyone re-running this study flips
the env var and rebuilds — no long-lived instrumentation in the
resident shader.
