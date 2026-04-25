# Project Overview

Audience: technical reader, comfortable with asymptotics and basic concurrency, but not necessarily a games or ECS specialist. This is the five-minute "what is this" doc. It cross-references the canonical specs under `docs/spec/` rather than duplicating them.

## 1. What this project is

A deterministic, tick-based, event-cascading agent simulation. Each agent is a record with position, health, species, status effects, and a handful of cumulative-memory fields; tens to hundreds of thousands of agents run with no central controller. Each agent's next action is the argmax of a local scoring function over its own state and what it can observe.

The system is bit-deterministic: the same initial state and seed produces byte-identical output across runs and across the CPU and GPU backends. That single constraint drives most of the architecture.

Target scale: 20k–200k agents on a commodity desktop, interactive speed (≥30 ticks/sec at 20k, ≥2 ticks/sec at 200k), full replay from a seed.

## 2. Architecture: rules as data, compiled twice

The flagship insight. The rules of the simulation — what events exist, which physics rules react to them, how views accumulate, how scoring is computed — live in a domain-specific language. Designers edit `.sim` text files. Engineers maintain the compiler and the runtime. A view declaration looks like:

```
@materialized(on_event = [AgentAttacked], storage = per_entity_topk(K = 8))
@decay(rate = 0.98, per = tick)
view threat_level(observer: Agent, attacker: Agent) -> f32 {
    initial: 0.0,
    on AgentAttacked { target: observer, actor: attacker } { self += 1.0 }
    clamp: [0.0, 1000.0],
}
```

A single compiler IR lowers to two backends:

1. **CPU (`SerialBackend`)** — straightforward Rust. Physics handlers are `fn` items, views are `struct`s with explicit fold methods. The deterministic reference.
2. **GPU (`GpuBackend`)** — WGSL/SPIR-V compute shader source plus host-side dispatch code. The performance path.

A static validator rejects rule bodies that can't lift to GPU: no heap allocation, no recursion, no dynamic dispatch, all loops bounded at parse time. If a rule can't compile to GPU, the build breaks. As a consequence, dozens of commits' worth of GPU infrastructure have landed without touching a single game rule.

Spec: `spec/language.md` (DSL), `spec/compiler.md` (lowering), `spec/runtime.md` (engine contract).

## 3. The DSL → engine → GPU pipeline

```
.sim source  ─►  compiler IR  ─►  Rust handlers   (SerialBackend, reference)
                              └►  SPIR-V kernels  (GpuBackend, performance)
                              └►  Python classes  (training / external ML)
```

The compiler owns text-to-engine lowering and emits all three targets from the same IR. Both backends consume the same `Engine` runtime: `Pool<T>` IDs, the event ring, the spatial index, the RNG streams, the mask buffer, the policy backend, and the materialized/lazy/topk view traits. The contract between compiler output and runtime is fixed by `spec/scoring_fields.md` (the `field_id` ABI) and the schema hash (`crates/engine/.schema_hash`); CI catches drift.

Cross-refs:

- DSL grammar and semantics: `spec/language.md`.
- Field catalog (every SoA field, who reads, who writes): `spec/state.md`.
- Built-in functions and namespaces: `spec/stdlib.md`.
- Codegen and lowering passes: `spec/compiler.md`.
- Ability DSL (a sub-language for ability definitions): `spec/ability.md`.
- Economic system layered on top: `spec/economy.md`.

## 4. The deterministic tick

`step(state, events, policy, cascade) → state'` is the atomic unit. Six phases, in order: **mask** (per-agent bitmap of eligible action kinds), **scoring** (argmax of `score(agent, action, target)` over the action × target space), **apply** (chosen intents become events on the per-tick ring), **cascade** (fixed-point iteration: physics rules dispatch over events, fold into views, may emit new events; bounded at 8 iterations, typically converges in 2–4), **movement** (positions update, can re-emit events that fold further — movement is part of the cascade, not a post-pass), and **finalize** (tick counter, invariants, telemetry). Events are the only observable side channel; all derived state is reconstructible from the event log.

Authoritative description: `spec/runtime.md §14` (tick pipeline), with §11 (cascade), §12 (mask), §13 (policy), §15 (views), §22 (schema hash), §2 (determinism contract).

Determinism on GPU is enforced by serializing fold dispatches per event so concurrent atomicCompareExchange races can't reorder floating-point folds; the host issues events in deterministic sequence. The drained event log is sorted by `(tick, kind, payload[0])` before chronicle replay. See `spec/runtime.md §2` and §8.

## 5. A worked example: rout cascade

To see why the rules-as-data architecture buys real expressive power, watch how one wolf's death propagates without any of it being scripted:

1. Wolf A dies. The CPU/GPU runtime emits an `AgentDied { agent_id: A }` event onto the per-tick event ring.
2. The physics rule `fear_spread_on_death` matches `AgentDied`. Its body queries the spatial index for kin within 12m and emits a `FearSpread { observer: kin, dead_kin: A }` event for each.
3. Each `FearSpread` event folds into the materialized view `kin_fear(observer, dead_kin)`, incrementing the per-observer top-K slot keyed by `dead_kin = A`.
4. Next tick (or later in this tick's cascade, if movement re-triggers anything), each surviving nearby wolf scores its actions. The `Flee` row of the scoring table includes a predicate `view::kin_fear(self, *) sum > 0.5 : +0.4`. That predicate now fires for every wolf whose kin_fear top-K sums above the threshold.
5. Combined with a lower `Attack` score (the original target may now be less optimal, or out of range), the argmax flips from `Attack` to `Flee` for many of those wolves.
6. If the retreat exposes another wolf, and it dies, more `FearSpread` events fire — fear propagates as a wave, the rout cascades.

Nothing in steps 1–6 is special-cased. The view is a generic top-K fold; the physics rule is a generic spatial query plus emit; the score row is a generic predicate. The compiler emitted it all from text. Pack-hunting (`pack_focus_on_engagement`), grudges (`my_enemies`), and herding-under-pressure all factor the same way — different rules and views, same plumbing.

The full set of currently-declared views: `my_enemies`, `threat_level`, `kin_fear`, `pack_focus`, `rally_boost` (folded), plus `engaged_with` (pair-slot) and pure expressions `is_hostile`, `is_stunned`, `slow_factor`. See `spec/state.md` for the field-level catalog.

## 6. Where to go next

If you want the contract — read the spec:

- `spec/README.md` — index and reading order for the canonical specification.
- `spec/runtime.md` — the engine contract (state, events, mask, policy, cascade, tick pipeline, schema hash).
- `spec/language.md` — the DSL grammar and semantics.
- `spec/compiler.md` — DSL → Rust + SPIR-V + Python lowering.
- `spec/ability.md`, `spec/economy.md` — sub-languages and layered systems.

If you want to know what's built right now — read live status:

- `engine/status.md` — per-subsystem ✅/⚠️/❌ truth, currently-known issues, what would falsify each claim.
- `spec/gpu.md` — GPU backend contract (resident cascade, sim-state, cold-state replay, ability eval, kernel reference).

If you want to know what's coming — read planning:

- `ROADMAP.md` — comprehensive index of future work (active / drafted / deferred).
- `superpowers/plans/` — written plans for in-flight work.
- `superpowers/research/` and `superpowers/notes/` — design exploration and bisects.

A reasonable five-file path for a new contributor: this doc → `engine/status.md` → `spec/README.md` → `spec/runtime.md` → `ROADMAP.md`.
