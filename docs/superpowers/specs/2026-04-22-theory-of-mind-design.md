# Theory of Mind for the Combat Engine — Design

> Spec: wire belief-based decision-making into the combat sim. Scoring
> currently reads ground truth (`target.hp_pct`, `view::threat_level(self,
> _)`); this spec adds a first-order belief layer so scoring can read
> what each agent *believes* about the world — a state that updates on
> observed events and decays with time.
>
> Cross-refs: `docs/dsl/spec.md` §9 "Source-tagged information with
> theory-of-mind", `docs/dsl/state.md` (believed_knowledge + MemoryEvent
> sections), `docs/superpowers/notes/2026-04-22-engine-expressiveness-gaps.md`
> §8 (line-of-sight), `docs/superpowers/notes/2026-04-22-terrain-integration-gap.md`.
>
> **Migration note (2026-04-25, post-Spec-B'):** This spec was authored
> against the pre-Spec-B' architecture. After Spec B' landed, engine has
> zero rule-aware code: chronicle, engagement, step body, mark_*_allowed,
> ViewRegistry field, with_engine_builtins helper, and `engine/src/generated/`
> all moved out (some deleted, some emitted into engine_rules from the DSL).
> Reading this spec for an implementation plan, account for these mappings:
>
> - **§3.1 `crates/engine/src/belief/mod.rs`** — adding to engine triggers
>   the `engine/build.rs` allowlist gate (Spec B' D11). Two paths:
>   (a) `belief/` is genuinely a primitive (just `BoundedMap` storage with no
>   per-tick rule logic) → allowlist edit + critic gate; (b) it's rule-aware
>   (the update cascade lives there) → it goes to `engine_rules` as emitted.
>   Recommendation: split. `BoundedMap<T, K>` lives in engine as a primitive
>   (under `pool/` or new `bounded_map/`); the `BeliefState` shape lives in
>   engine_data (emitted from DSL); the update cascade handler is emitted
>   from `assets/sim/physics.sim` into `engine_rules/src/physics/`.
>
> - **§3.2 update cascade "initially hand-written in
>   `engine/src/cascade/update_beliefs.rs`; migrate to DSL-generated once
>   Phase 1 stabilizes"** — skip the hand-written stage. Engine refuses
>   hand-written cascade handlers (`__sealed::Sealed` + ast-grep CI rule).
>   Author the cascade rule directly in `assets/sim/physics.sim`; the DSL
>   may need a small grammar extension if "update belief slot K of agent A
>   given event E" isn't expressible.
>
> - **§3.5 grammar extension `beliefs(self).about(target).<field>`** — fits
>   the existing scoring grammar pattern; resolver/IR work goes in
>   `crates/dsl_compiler/`. Plan accounts for this as Phase-1 grammar work.
>
> - **`SimState.cold_beliefs` SoA field** — needs to be added via DSL agent
>   declaration (engine_data emits SimState fields per Plan B1' D14
>   approach). NOT hand-edited in `engine/src/state/mod.rs`.
>
> - **`CascadeRegistry::with_engine_builtins`** — moved to
>   `engine_rules/src/cascade_reg.rs` (compiler-emitted from
>   `dsl_compiler::emit_cascade_register`). The "wire into builtins" step
>   becomes "emit `reg.register(BeliefUpdateHandler)` from
>   `emit_cascade_register`."
>
> The companion plan (`plans/2026-04-25-theory-of-mind-impl.md`) translates
> the §9 build sequence to the new architecture.

---

## 1. Problem

The combat sim's decision layer reads ground truth. Every agent scores
actions against the physical state of the world this tick. Consequence:

- No ambush (an unseen wolf still shows up in `view::threat_level`).
- No stale-data pursuit (predator always knows where prey IS, not
  where they last KNEW it to be).
- No fog of war, no deception, no bluffs, no emergent coordination via
  mutual inference, no betrayal-with-stakes.
- The information surface is identical for every agent.

Meanwhile the DSL spec and the engine SoA already *anticipate* ToM. The
docs commit to "Source-tagged information with theory-of-mind" and
"per-bit volatility (Short / Medium / Long half-lives)" for
`believed_knowledge`. `MemoryEvent` already carries `source: Source` +
`confidence: f32`. The infrastructure is half-built — what's missing is
the **scoring layer actually consuming the belief state** rather than
ground truth.

This spec closes that gap for first-order beliefs. After it lands,
scoring rows can read `beliefs(self).about(target).<field>` and the
engine produces genuinely information-asymmetric decisions.

---

## 2. Scope

### 2.1 In scope (Phase 1)

- `BeliefState` struct + `cold_beliefs: BoundedMap<AgentId, BeliefState, K=8>`
  per-agent SoA field.
- Belief-update cascade handler that fires on observable events and
  updates the observer's belief of the actor.
- `observation_range` config (smaller than `aggro_range`), distance-based
  visibility gate — no terrain LOS in Phase 1.
- Per-tick belief decay: `confidence *= decay_rate`.
- DSL grammar extension: `beliefs(self).about(target).<field>` and
  `beliefs(self).<view_name>(_)` accessors.
- One reference scenario: **The Silent Wolf** — a stationary wolf
  outside observation_range that ground-truth scoring would have the
  human flee from, belief-based scoring lets the human walk past.

### 2.2 Out of scope (Phase 2)

- Second-order beliefs (`beliefs(self).beliefs_of(other).about(target)`).
- Terrain-based line-of-sight (prerequisite: the terrain integration
  gap doc). Phase 1 uses pure distance.
- Lying as a `Communicate` action that writes into the receiver's
  beliefs.
- Trust state (`beliefs.about(B).reliability`) and betrayal cascades.
- Rewriting existing wolves+humans scoring to use `beliefs(.)` —
  canonical fixture stays on ground truth. Belief-reads are opt-in per
  row.

### 2.3 Non-goals (will not build)

- **No full-state mirrors.** `BeliefState` is a narrow snapshot of a
  target's observable properties, not a full state copy.
- **No cognitive realism.** Confidence decays are linear/exponential,
  not bayesian.
- **No belief-based hidden state beyond visibility.** Stealth
  (invisibility regardless of distance), fog-of-war as persistent map
  obscuration, etc. require the terrain layer + more primitives.
- **No determinism compromise.** Every belief update is a pure function
  of the event stream filtered by observer position; replay holds byte-
  exact. Parity tests still pass.

---

## 3. Architecture

### 3.1 Data: BeliefState + BoundedMap

New type in a new module `crates/engine/src/belief/mod.rs`:

```rust
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct BeliefState {
    /// Last-observed position. Stale data — not the target's current pos.
    pub last_known_pos: Vec3,
    /// Last-observed HP. Decays only on re-observation, not on time.
    pub last_known_hp: f32,
    pub last_known_max_hp: f32,
    pub last_known_creature_type: CreatureType,
    /// Tick of last observation. Decay reads this.
    pub last_updated_tick: u32,
    /// 1.0 = fresh (same tick). Decays per tick toward 0.
    /// At confidence < EVICTION_THRESHOLD the slot may be reused.
    pub confidence: f32,
}
```

New SoA field on `SimState` (cold — accessed only during scoring and
cascade update, not on every physics path):

```rust
pub cold_beliefs: Vec<BoundedMap<AgentId, BeliefState, BELIEFS_PER_AGENT>>,
```

`BELIEFS_PER_AGENT = 8` (Phase 1 constant). Bounded to keep the SoA
fixed-size. Eviction policy: LRU by `last_updated_tick`, with a floor on
`confidence` — high-confidence beliefs (recent observations) don't get
evicted even if many other agents become relevant.

`BoundedMap<K, V, N>` is a small fixed-cap associative array over
`SmallVec<[(K, V); N]>` with linear probe. Already usable pattern in
the engine (`SortedVec<Membership, 8>` precedent in state-port).

**SoA footprint:** 8 BeliefStates × ~40 bytes each = ~320 bytes per
agent. At 200K agents = 64 MB cold. Acceptable per the spec's
hot/cold split — beliefs are cold, read only during scoring + cascade
application.

### 3.2 Update physics: the belief-update cascade rule

New rule in `assets/sim/physics.sim`:

```
physics update_beliefs {
    on AgentMoved { actor, location } {
        for observer in query.agents_within(location, config.belief.observation_range) {
            if observer == actor { continue; }
            beliefs(observer).observe(actor) with {
                last_known_pos:  location,
                last_known_hp:   agents.hp(actor),
                last_known_max_hp: agents.max_hp(actor),
                last_known_creature_type: agents.creature_type(actor),
                last_updated_tick: world.tick,
                confidence: 1.0,
            }
        }
    }
    on AgentAttacked { actor, target } {
        // Attacking is louder than moving — observers farther away detect it.
        for observer in query.agents_within(agents.pos(actor), config.belief.loud_observation_range) {
            if observer == actor { continue; }
            beliefs(observer).observe(actor) ...  // (same as above)
            beliefs(observer).observe(target) ...
        }
    }
    on AgentDied { agent_id } { /* observers update belief + evict stale kin */ }
    on AgentFled { agent_id, to } { /* observers update last_known_pos */ }
}
```

The `beliefs(observer).observe(target) with { ... }` form is a new DSL
primitive — it mutates the observer's cold_beliefs BoundedMap entry for
target. Lowered by the compiler to:

```rust
state.belief_mut(observer).upsert(target, BeliefState { ... });
```

### 3.3 Update physics: per-tick decay

A new tick-phase step (`phase::belief_decay`) — runs once per tick, after
the cascade fixed-point converges:

```rust
fn belief_decay_phase(state: &mut SimState, config: &Config) {
    let rate = config.belief.decay_rate;          // e.g., 0.98
    let floor = config.belief.eviction_threshold; // e.g., 0.05
    for observer in state.agents_alive() {
        let mut beliefs = state.belief_mut(observer);
        beliefs.retain(|_target, bs| {
            bs.confidence *= rate;
            bs.confidence >= floor
        });
    }
}
```

Small phase — linear in alive agents × K, runs once per tick, cold-memory
access.

### 3.4 Visibility: Phase 1 heuristic

Phase 1 visibility is pure distance, driven by two new config values:

```
config belief {
    observation_range:       f32 = 10.0,  // passive observation
    loud_observation_range:  f32 = 25.0,  // for Attack events
    decay_rate:              f32 = 0.98,  // per-tick confidence decay
    eviction_threshold:      f32 = 0.05,  // evict beliefs below this
}
```

`observation_range` < `aggro_range` (50m). Today aggro_range leaks
hostility detection out to 50m for every agent regardless of whether
anything was observed. After this spec, aggro_range remains the gate
for "who's a candidate for hostility reasoning" but the ACTUAL
detection now requires an observed event within observation_range.

Phase 2 replaces distance with real terrain LOS via the terrain query
(gap doc §8).

### 3.5 Scoring grammar: the belief accessor

New syntax in `assets/sim/scoring.sim`:

```
Flee = 0.0
    + (if beliefs(self).about(nearest_hostile).hp_pct < 0.3 { 0.4 } else { 0.0 })
    + (beliefs(self).threat_level(_) per_unit 0.01)
    + (if beliefs(self).confidence(nearest_hostile) < 0.1 { -0.2 } else { 0.0 })
```

Three forms:

| Form | Lowering |
|------|----------|
| `beliefs(self).about(target).<field>` | read `cold_beliefs[self][target].<field>`; fall back to default if absent |
| `beliefs(self).<view_name>(_)` | aggregate across all believed agents (mirrors today's ground-truth views but filters by `cold_beliefs[self]` contents) |
| `beliefs(self).confidence(target)` | direct confidence read; `0.0` if no belief exists |

When no belief exists for `target`:
- `.about(target).hp_pct` → treated as `UNKNOWN` — the modifier skips
  (emitted as `0.0` contribution by the compiler).
- `.confidence(target)` → `0.0`.
- View aggregations (`beliefs(self).threat_level(_)`) skip believed-
  absent agents — they aren't summed.

Compile-time: the IR emitter gains `BeliefsAccessor(AgentRef, Field)`
and `BeliefsConfidence(AgentRef)` nodes. `emit_scoring` lowers them to
runtime Rust reads against `state.cold_beliefs`. Interpreter
(`dsl_ast::eval::scoring`) mirrors via a new `ReadContext::belief_about`
method.

### 3.6 Mask extensions

A natural consequence: masks gain a `visible(target)` primitive.

```
mask Attack(target: AgentId) from candidates
  ...
  and beliefs(self).about(target).is_fresh(max_age_ticks = 20)
```

This prevents an agent from attacking a target it hasn't recently
seen — you can't swing at a ghost.

Phase 1 ships this as a mask primitive reading `world.tick -
belief.last_updated_tick`.

---

## 4. Reference scenario: The Silent Wolf

New test: `crates/engine/tests/silent_wolf_belief.rs`.

**Setup:**
- Seed: `0x5117_E17_D0E5_CA7E`
- Observation range: 5m (override config for this test)
- Agents:
  - Human at `(0, 0, 0)`, hp=100, moving toward east.
  - Wolf at `(8, 0, 0)` — between 5m and 50m from human. Inside
    aggro_range (50m), OUTSIDE observation_range (5m).
- Ticks: 40.

**Ground-truth behavior** (current engine):
- Tick 0: human's `view::threat_level` sees the wolf because it's within
  aggro_range. Flee row fires (threat present). Human flees west.

**Belief-based behavior** (with this spec):
- Tick 0: human's `cold_beliefs` is empty. No observed events yet.
  `beliefs(self).threat_level(_)` sums over zero entries = 0. Flee score
  = 0.1 (Hold baseline). Human picks MoveToward (eastward).
- Ticks 0–K: human moves east. Eventually enters observation_range of
  wolf → belief-update cascade fires → human's belief of wolf populates.
- Tick K+: human's Flee score now > 0. Human flees.

**Assertions:**
- `first_flee_tick >= 5` — human didn't flee immediately.
- `human_observed_wolf_tick` exists in the event log — there's a
  moment where the human transitioned from "didn't know" to "knew."
- Under ground-truth scoring (compile without
  `--features theory-of-mind`), `first_flee_tick == 0` — confirms the
  canonical behavior is preserved without the feature.

**Why this test matters:** it's the smallest demonstration that the
belief layer is actually driving decisions. If `first_flee_tick` = 0
under the feature, the belief accessor wasn't wired correctly. If
`first_flee_tick` ≠ 0 under default, the belief layer is leaking into
the canonical path.

---

## 5. Feature flag + compatibility

New Cargo feature: `theory-of-mind`.

- `OFF` (default): the new SoA field is not allocated (or allocated
  zero-cap). The scoring rows using `beliefs(.)` don't compile into
  the emitted Rust — conditional emission in `emit_scoring` when the
  scoring IR contains `BeliefsAccessor` nodes. The DSL still parses;
  belief-using rows are stripped or compile-errored.
- `ON`: full pipeline active.

Wolves+humans canonical parity test:
- Default build: byte-identical to committed baseline. The canonical
  scoring uses no `beliefs(.)` accessor, so outputs are unchanged.
- With feature on: also byte-identical, because the canonical scoring
  rows still use ground-truth accessors. The belief layer is populated
  as a side-effect of the cascade rule, but nothing reads it.

Gate: introducing `beliefs(.)` into wolves+humans scoring would change
the baseline. That's an explicit, reviewable operation — regen the
baseline with `WOLVES_AND_HUMANS_REGEN=1` and commit.

---

## 6. Phasing

| Phase | Content |
|---|---|
| **Phase 1 (this spec)** | BeliefState, cold_beliefs SoA, update cascade, decay phase, belief accessor in scoring grammar, distance-based visibility, Silent Wolf reference test. |
| **Phase 2** | Second-order beliefs: `beliefs(self).beliefs_of(other).about(target)`. SoA doubles to `BoundedMap<AgentId, BoundedMap<AgentId, BeliefState, K2>, K1>`. Limited K2 (e.g., 3) to cap memory. |
| **Phase 3** | Terrain LOS replaces distance visibility (requires the terrain layer). |
| **Phase 4** | Lying: `Communicate { payload }` with payload = fact; cascade rule writes into receiver's `cold_beliefs` with the payload values + `confidence < 1.0`. Trust tracking: `beliefs.about(B).reliability` updates when later observation contradicts B's prior communication. |

Phases 2–4 are sketched here for context. This spec scopes Phase 1 only.

---

## 7. Cost analysis

### 7.1 Memory

- Per agent: 8 BeliefStates × ~40 bytes = 320 bytes cold.
- 200K agents: ~64 MB cold.
- Compared to existing cold fields (`cold_memory_events`: 64 × 16 = 1 KB,
  `cold_relationships`, etc.), belief storage is modest.

### 7.2 CPU: update cascade

- Per event that triggers `update_beliefs`: O(observers_in_range).
  Observation_range << aggro_range, so typical observer count is small.
- Per tick: ~5 events/tick × ~3 observers/event = ~15 belief writes.
  Trivial.

### 7.3 CPU: decay phase

- O(alive_agents × K=8) per tick. At N=200K, K=8: 1.6M ops per tick.
  One pass, cold memory, vectorizable. Budget: ~2ms per tick on host.

### 7.4 CPU: scoring with beliefs

- Belief read adds one map-lookup per `beliefs(...)` accessor per score
  evaluation. K=8 bounded map = max 8 compares. Each scoring row may
  trigger a few such reads.
- Worst case: +~10ns per scoring row per agent. At N=200K, per tick,
  with ~5 scoring rows: ~10ms. Not great, need a hash or sorted-keys
  variant. Tune via BoundedMap impl.

### 7.5 Determinism

- Belief updates are pure functions of (event, observer_pos,
  target_state, config). No RNG, no side channels.
- Decay is pure f32 arithmetic.
- Scoring reads of beliefs are pure.
- `replayable_sha256()` byte-exact across runs at same seed.
- Cross-backend parity (Serial vs GPU) holds only if the GPU path
  implements the same cascade + decay phases. Add to the per-kernel
  parity contract in `docs/compiler/spec.md`.

---

## 8. Acceptance criteria

1. `cargo check -p engine` clean.
2. `cargo test -p engine` — canonical wolves+humans baseline unchanged.
3. `cargo test -p engine --features theory-of-mind` — both canonical
   baseline (unchanged because canonical scoring uses ground truth) AND
   the new `silent_wolf_belief` test pass.
4. `silent_wolf_belief::human_delays_flee` — `first_flee_tick >= 5`.
5. `silent_wolf_belief::human_observes_at_contact` — event log contains
   a belief-update for the wolf with `observer_id = human_id`.
6. Under `--features theory-of-mind`, replay determinism holds: run the
   same seed twice, identical `replayable_sha256`.
7. DSL: `assets/sim/scoring.sim` with a `beliefs(...)` row compiles
   without error. An out-of-scope accessor (e.g., `beliefs(self).about(target).stun_remaining`
   for a field that's not in `BeliefState`) gives a clear resolver
   error.

---

## 9. Build sequence

1. `belief/mod.rs`: BeliefState + BoundedMap + constants.
2. `SimState.cold_beliefs` SoA + ctor + accessor methods.
3. Feature gate: `theory-of-mind` in `engine/Cargo.toml`, feature-gate
   the new SoA allocation to zero-cap when off.
4. Belief-update cascade handler (initially hand-written in
   `engine/src/cascade/update_beliefs.rs`; migrate to DSL-generated
   once Phase 1 stabilizes). Wire into `CascadeRegistry::with_engine_builtins`.
5. Per-tick decay phase: new `phase::belief_decay` in `step::step_full`.
6. DSL grammar: `dsl_ast` parser accepts `beliefs(agent).about(target).<field>`
   and `beliefs(agent).<view>(_)`. Resolver produces `IrBeliefsAccessor` /
   `IrBeliefsView` / `IrBeliefsConfidence` nodes.
7. Compiler emission: `emit_scoring` lowers the new IR to Rust reads
   against `state.cold_beliefs`.
8. Interpreter: `dsl_ast::eval::scoring` handles the new IR nodes via a
   new `ReadContext::belief_about` method. Engine `EngineReadCtx` impl
   delegates to `SimState::cold_beliefs`.
9. Mask primitive: `beliefs(self).about(target).is_fresh(max_age)`
   (optional for Phase 1 — can defer to Phase 1.5).
10. Reference test: `crates/engine/tests/silent_wolf_belief.rs`.
11. Doc update: `docs/dsl/scoring_fields.md` — add belief accessors.
12. Parity sweep: `cargo test -p engine` (default) + `cargo test -p
    engine --features theory-of-mind`.

Estimated task count for an implementation plan: 12–15 tasks, most
small. The shape mirrors P1b (the IR interpreter) in pattern —
incremental DSL grammar extension + runtime wiring + one reference
test.

---

## 10. Open design questions

None marked `[OPEN]` — each decision has a rationale in the relevant
section. If any surprise arrives during implementation (e.g., the
BoundedMap impl is slower than budgeted; the interpreter path blows up
on second-order belief shapes in Phase 2), that's an amendment back to
this doc, not an in-flight deviation.
