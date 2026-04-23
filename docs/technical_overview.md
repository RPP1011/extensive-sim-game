# A Wolves-and-Humans Simulation: Technical Overview

Audience: technical reader, not a games or ECS specialist. Comfortable with asymptotics, basic concurrency, and reading code.

## 1. What the sim is, precisely

A deterministic, tick-based, event-cascading agent simulation. Each tick is a discrete step that advances the world by one unit of simulated time. Each agent is a record with position, health, species, status effects, and a handful of cumulative-memory fields. The world contains several tens to several hundreds of thousands of agents depending on how we're running it. There is no central controller; each agent's next action is determined by a local scoring function over its own state and what it can observe.

The system has been designed so that the same simulation, given the same initial state and the same random seed, produces bit-identical output regardless of whether it runs on a CPU or GPU. This constraint drives most of the interesting architectural choices.

## 2. Architecture: rules as data, compiled twice

The rules of the simulation — what events exist, which physics rules react to which events, how views accumulate, how scoring is computed — live in a domain-specific language (DSL). Here is what a view declaration actually looks like:

```
@materialized(on_event = [AgentAttacked], storage = per_entity_topk(K = 8))
@decay(rate = 0.98, per = tick)
view threat_level(observer: Agent, attacker: Agent) -> f32 {
    initial: 0.0,
    on AgentAttacked { target: observer, actor: attacker } { self += 1.0 }
    clamp: [0.0, 1000.0],
}
```

And a physics rule:

```
physics fear_spread_on_death @phase(event) {
    on AgentDied { agent_id: dead } {
        for kin in query.nearby_kin(dead, 12.0) {
            emit FearSpread { observer: kin, dead_kin: dead, tick: state.tick };
        }
    }
}
```

A compiler reads these files and produces two backends from the same intermediate representation:

1. **CPU backend**: straightforward Rust. Physics handlers become `fn` items, views become `struct`s with explicit fold methods. Used as the deterministic reference.
2. **GPU backend**: WGSL compute shader source, plus the host-side code that manages buffers and dispatches kernels. Used when agent counts make CPU infeasible.

A separate validator statically enforces that every rule body is *GPU-emittable*: no heap allocation, no recursion, no dynamic dispatch, all loops bounded at parse time. The validator runs on every compile; if someone adds a rule that can't lift to GPU, the build breaks.

The practical value of this separation is that the game designer edits DSL text files. The engine authors don't have to rewrite anything when behaviors change. And the 60+ commits' worth of GPU infrastructure didn't require touching a single game rule.

## 3. The tick, formally

`step(state, events, policy, cascade) → state'` is the simulation's atomic unit. Its internal phases:

1. **Mask computation** — for each agent, a bitmap over action kinds (`Attack`, `MoveToward`, `Flee`, `Hold`, `Eat`, `Drink`, `Rest`, `Cast`). Bit set iff the agent is eligible for that action this tick. Eligibility is a rule-expressed predicate — e.g. `Attack` requires a hostile target in range; `Flee` requires a threat within some distance; `Rest` requires no threats within a larger distance.

2. **Scoring** — a function `score(agent, action, target) → f32`. The scoring table has ~20 rows, each row a `(action, modifier)` pair. Modifiers include base scores, threshold predicates over agent fields (`self.hp_pct < 0.3 : +0.6` on `Flee`), and view-call predicates (`view::my_enemies(self, target) > 0.5 : +0.4` on `Attack`). Scoring reduces to argmax over the `(action × target)` space, producing one chosen `(action, target)` per agent.

3. **Apply** — chosen intents become events: `Attack(a, b)` → `AgentAttacked { actor: a, target: b, damage, tick }`, etc. Events are appended to a per-tick event ring.

4. **Cascade** — the cascade is a fixed-point iteration. Each iteration: physics rules dispatch over the current batch of events, possibly emitting new events; the new events fold into views (updating their state); the new events become the next iteration's input. Iteration terminates when no new events are produced, or a hard bound (currently 8) is reached. In practice the canonical wolves-and-humans scenario converges in 2-4 iterations per tick.

5. **Movement** — agents whose chosen action was `MoveToward` or `Flee` update their positions. Movement can emit `AgentMoved` / `AgentFled` events, which trigger the engagement-update physics rule, which can emit more events, which fold further. Movement is thus part of the cascade, not a separate post-pass.

6. **Finalize** — tick counter increments, invariants check, telemetry flushes.

Events are the only observable side channel of the simulation. All derived state (views, chronicle prose, serialization) is reconstructible from the event log, modulo decay timestamps. This gives the simulation a clean replay semantics: replay the event log, recompute everything.

## 4. Views: where memory lives

Agents need memory — what happened in past ticks that should influence current decisions. The simulation represents memory as *views*: derived state computed by folding events into storage. Each view declares (a) the event types it folds on, (b) the storage shape, and (c) a decay rule for how older values attenuate.

For example, `threat_level(observer, attacker)` above is a scalar-valued function of an ordered pair of agents. It folds on `AgentAttacked`: when the event fires, `threat_level[observer=target][attacker=actor] += 1.0`. The value decays exponentially with a rate of 0.98 per tick, so an attack from 100 ticks ago contributes `0.98^100 ≈ 0.13` today. The decay is applied at read time, not write time: each cell stores `(value, anchor_tick)` and the read computes `value * rate^(now - anchor_tick)`. This avoids dispatching a decay pass every tick.

Scoring is allowed to call views as predicates. `view::threat_level(self, target) > 20.0 : +0.3` on the `Attack` row means: if the observer has accumulated threat greater than 20 from this specific target, boost the attack score by 0.3. This is how grudges manifest — memory feeding back into choice.

The simulation currently declares five folding views:

- `my_enemies(observer, attacker)` — clamped indicator of "this attacker has hit me"
- `threat_level(observer, attacker)` — cumulative, decaying
- `kin_fear(observer, dead_kin)` — fear intensity from each specific dead packmate
- `pack_focus(observer, target)` — signal that allies are attacking this target
- `rally_boost(observer, wounded_kin)` — signal that an ally has been wounded

Plus a non-decaying pair-slot view `engaged_with(agent) → Option<Agent>` and three lazily-evaluated views (`is_hostile`, `is_stunned`, `slow_factor`) that are pure expressions, not folded.

## 5. Spatial queries

Many rules need "agents near this point" or "nearest hostile within radius R." The simulation maintains a uniform-grid spatial hash: the world is partitioned into 16m × 16m cells; an insertion records an agent's cell membership by position; a `within_radius(center, R)` query scans the `O((R/cell)²)` cells covering the query disk and returns all agents in those cells. On GPU, the hash is rebuilt each tick from scratch because incremental updates would require atomics and cross-workgroup synchronization. Rebuild is three kernel dispatches (count, prefix-sum, scatter) and is cheap relative to everything else.

Query results are capped at K=32 per call. Beyond that, results truncate in id-ascending order. In the worst case (one densely-populated cell with 1000+ agents in 16m²) this truncation affects correctness, but in the range we actually operate (at most ~30 agents per cell) it's noise-level.

## 6. The GPU architecture

Modern GPUs are structured as a grid of Single-Instruction Multiple-Thread (SIMT) processors. Work is submitted as a compute *kernel*, a function that is instantiated once per thread in a grid typically of thousands. Threads within a *workgroup* (~32-256 threads) share on-chip memory and can synchronize; threads in different workgroups cannot synchronize within a single kernel launch.

That last constraint is important. In the simulation's tick, the cascade is a fixed-point loop. Within one cascade iteration, physics kernels can process thousands of events in parallel — each thread one event. But between iterations, we need to know whether any new events were produced, and we need those events to be visible to the next kernel. That requires a full GPU→CPU synchronization (a "readback") or careful use of atomic counters on GPU-resident state.

The engine currently uses the second approach: all cascade state lives in GPU buffers, the event ring's tail is an `atomic<u32>` incremented by emit sites, and the host driver issues cascade-iteration kernels in sequence without intermediate readback. A single host-side readback at tick end drains the accumulated events back to CPU for chronicle rendering and cold-state rules that don't run on GPU (gold transfers, standing adjustments, memory-ring writes — currently kept on CPU because the GPU agent record doesn't carry those fields).

Each kernel binds its inputs and outputs through a *bind group* — roughly, a tuple of GPU buffers mapped to numbered slots. WGSL (the shader language the host compiles to via the `naga` library and dispatches via `wgpu`) enforces a per-binding size cap of ~2 GB on common adapters; total bindings per bind group are capped at 16. Much of the scaling work was staying under these caps.

## 7. The O(N²) wall and its solution

Naive storage for a view like `threat_level(observer, attacker)` is a dense 2D grid: one cell per ordered pair. At agent count N, this is O(N²) cells. At N=20,000: 400 million cells × 4 bytes = 1.6 GB — already over the 2 GB per-binding cap once you include the anchor-tick buffer. At N=200,000: 40 billion cells per view, 160 GB per view, 640 GB across the four pair-structured views. Physically impossible on any GPU.

The insight that breaks the wall: most cells are always zero or uniform-initial. An agent has been attacked by maybe 3-5 distinct other agents over the window the decay keeps alive; an agent has kin-died nearby maybe 2-4 times in the relevant window; etc. Dense pair-maps pay full N² storage to represent very sparse actual content.

Replace: per-observer top-K storage. For each observer, keep an array of at most K = 8 entries, each `(id, value, anchor_tick)`. Reads look up target id by scanning K slots (linear but K is tiny). Wildcard sums (the semantic "sum over all attackers" used by some scoring predicates) walk the K slots. Writes insert into the first empty slot; if all K slots are full, evict the slot with the smallest value if the new entry's value would exceed it, else drop the write. Because decay attenuates unused slots rapidly, the eviction mechanism almost never fires in practice — slots expire on their own.

Storage drops from O(N²) to O(N · K). For N=200,000 and K=8: 1.6 million cells per view × ~12 bytes each ≈ 19 MB per view. All four views fit in under 100 MB.

The semantic change: a wildcard-sum is now a sum over the top-K ever-seen, not a sum over all pairs. For sparse structured behavior — which is what all the scoring predicates actually care about — the difference is below measurement noise. The small-scale byte-for-byte parity baseline still holds because at N ≤ K + 1 the two representations are equivalent.

## 8. Cascade determinism on GPU

Fold operations like `threat_level[obs][attacker] += 1.0` are associative-and-commutative in abstract math but not in IEEE 754 floating-point: `(a + b) + c ≠ a + (b + c)` in general, because rounding after each operation introduces bits of error that depend on operand magnitudes.

On GPU, multiple threads may attempt to fold events that target the same cell. If two workgroups fold concurrently using `atomicCompareExchange` loops, the order in which they win the CAS race depends on hardware scheduling — non-deterministic. The simulation avoids this by serializing fold dispatches per event (one kernel launch per event in the fold phase). Within a single event, there's no concurrent write — the ordering constraint becomes the ordering across events, which is enforced by the host issuing events in deterministic sequence.

This is correct but has a cost: at N=100,000 with heavy combat (~20,000 events per tick), the per-event dispatch overhead dominates. Batching the folds while preserving determinism requires either (a) sort events by target cell and use a segmented-reduction kernel, or (b) restrict folds to commutative-enough deltas (the current `self += 1.0` pattern would qualify, but the framework doesn't yet prove it generally). Both are straightforward; neither is implemented yet.

The event log itself also needs a deterministic total order when it's drained for chronicle rendering. Events emit into the GPU ring in nondeterministic order; the host-side drain sorts by `(tick, kind, payload[0])` before appending to the replayable log. This is the only total-ordering discipline the simulation relies on.

## 9. Emergent behaviors

These are not scripted; they fall out of scoring interactions.

**Rout cascade.** Wolf A dies. The `fear_spread_on_death` physics rule emits `FearSpread` events to each alive wolf within 12m. Those events fold into `kin_fear(observer, dead_kin)`. Now each surviving nearby wolf reads, via its `Flee` scoring row, `view::kin_fear(self, *) sum > 0.5 : +0.4`. Combined with a lower `Attack` score (because the target wolf may now be less optimal), the scoring argmax flips from `Attack` to `Flee`. The wolf retreats. If that retreat exposes another wolf, and it dies, more FearSpread fires — the fear propagates as a wave.

**Grudge persistence.** Wolf W is attacked by Human H1. `my_enemies(W, H1)` sets to 1.0. Next tick W's scoring has a +0.4 bonus on `Attack(W, H1)`. Even if H2 is closer or weaker, H1 wins the argmax until the my_enemies value decays or H1 dies.

**Pack hunting.** Wolf W1 engages Human H. `EngagementCommitted` fires. The `pack_focus_on_engagement` physics rule emits PackAssist events to each wolf within radius. `pack_focus(observer, target)` folds. Now every nearby wolf has a +0.4 bonus on `Attack(self, H)`. Multiple wolves converge on H.

**Deer herding.** Deer have a species flag `herds_when_fleeing = true`. When a deer's chosen action is `Flee`, the movement kernel computes the flee direction as `(away_from_threat + kin_bias · toward_kin_centroid).normalize()`. Under pressure, deer therefore bunch together; in calm they spread out.

None of these rules are special-cased. They are generic scoring predicates and generic physics rules whose composition happens to reproduce the recognizable patterns.

## 10. Performance

Measured wall-clock on a discrete GPU (RTX 4090 at release build):

| N (agents) | CPU ms/tick | GPU ms/tick | GPU verdict |
|---|---|---|---|
| 8 | 0.003 | 0.5 | 175× slower — fixed dispatch overhead dominates |
| 32 | 0.018 | 0.56 | 32× slower |
| 128 | 0.30 | 0.93 | 3× slower |
| 512 | 2.5 | 3.6 | 1.5× slower |
| 1,000 | 15 | 5 | 3× faster — crossover |
| 10,000 | 502 | 23 | 22× faster |
| 100,000 (calm) | ≈ hours | 259 | ∞ |
| 100,000 (combat) | ≈ hours | 1,740 | ∞ |

The crossover at N≈1000 is where per-agent GPU compute starts amortizing the fixed per-dispatch latency. Beyond that, GPU scales linear-ish in N while CPU hits its O(N²) wall.

At N=100,000 with active combat (~20,000 deaths across 10 ticks), per-tick cost decomposes as:

- GPU mask + scoring: 277 ms
- GPU cascade (physics dispatch, fold, drain): 193 ms
- GPU per-event view folds: 893 ms  ← dominant
- CPU apply + cold-state replay: 322 ms
- everything else: <1 ms

The per-event fold dispatch is the next bottleneck. Fixing it (segmented reduction over events grouped by target cell) is the next architectural step.

### 10.1 Batch API — GPU-resident cascade (2026-04-22)

Alongside the per-tick `SimBackend::step()` path, `engine_gpu::GpuBackend` exposes an additive batch API:

- `step_batch(n)` — records N ticks into a single command encoder. One `queue.submit`, one `device.poll(Wait)` at end. Eliminates the ~12 per-tick CPU/GPU fences of the sync path.
- `snapshot()` — double-buffered staging; returns agent state + events accumulated since the previous snapshot. Non-blocking GPU copy; one-frame lag.

The batch path uses indirect dispatch for cascade iterations (physics kernel writes next-iter workgroup count to a GPU-resident args buffer; zero workgroups means the dispatch GPU-no-ops), caller-owned spatial-output buffers so two queries per tick don't alias, and GPU unpack kernels that derive mask and scoring's kernel-specific layouts from `resident_agents_buf` each tick (no per-tick CPU→GPU uploads).

Parity tests and deterministic chronicle rendering stay on the sync path. The batch path is explicitly non-deterministic in event fold order — acceptable because it's intended for rendering and headless observation, not replay.

Perf envelope (llvmpipe fallback, N-ladder 8..2048, 200 timed ticks):

| N | Sync GPU µs/tick | Batch GPU µs/tick | Batch/Sync |
|---|---|---|---|
| 8 | 1679 | 835 | 0.50× |
| 32 | 1722 | 738 | 0.43× |
| 128 | 1958 | 1362 | 0.70× |
| 512 | 2814 | 2981 | 1.06× |
| 2048 | 6442 | 8114 | 1.26× |

At N ≤ 128 the batch path halves per-tick wall clock by amortising dispatch overhead across the batch. At N = 2048 the cascade work itself dominates, and batch-mode-specific overhead (double spatial rebuild per tick for kin + engagement radii, per-call bind-group construction, always-dispatch-8-iters pattern) pushes batch ~25% slower than sync. The batch path is a correctness-preserving observation primitive today; driving N=2048 below the sync curve requires cascade-level optimisations that are out of scope for the batch API itself.

Design + plan: `docs/superpowers/specs/2026-04-22-gpu-resident-cascade-design.md`, `docs/superpowers/plans/2026-04-22-gpu-resident-cascade.md`.

## 11. What isn't yet right

- Three full-tick byte-exact parity tests at the 50-agent canonical fixture are passing, but the GPU intentionally omits three CPU-only side effects — opportunity attacks, engagement-slow decel on movement, and flee direction with kin-bias are either absent on GPU or implemented as pure-away — so any test exercising those won't reproduce CPU byte-for-byte. Statistical parity (alive counts, event multisets) holds at all tested scales.
- Generated code currently lives mixed in with hand-written engine primitives. A cleaner architecture would isolate it in its own crate so compile-dsl rebuilds don't invalidate the core engine's incremental cache.
- At very small N (≤32) GPU dispatch overhead is a real loss. Running GPU is only worth it past ~500 agents.
- The cold-state rules (transfer_gold, standing updates, memory ring writes, chronicle prose) stay on CPU because the GPU agent record doesn't carry those fields. They run in a bulk pass after the GPU tick, driven off the drained event log. This works but adds CPU time at scale; porting them is possible but has not been done.
