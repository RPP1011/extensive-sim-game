# World Sim DSL Specification

Canonical specification for the World Sim DSL. Merges the former `language.md` (grammar, types, runtime semantics), `stdlib.md` (standard library reference), `compiler.md` (lowering architecture), and `scoring_fields.md` (scoring field-id mapping) into one canonical reference.

For runtime contract (backends, tick pipeline, cascade, determinism), see `engine.md`. For field catalog, see `state.md`.

---

## 1. Overview

The DSL declares a closed-world simulation: typed state, typed events, declarative derivations, cascade rules, action masks, scoring weights, invariants, and regression probes. Ten top-level declaration kinds compose into a single compiled artefact (Rust runtime + SPIR-V kernels + Python dataclass mirrors + trace-format schema).

Machine learning — policy architecture, training algorithm, curriculum, reward shaping, observation packing — is **not** in the DSL. The compiler emits Python dataclasses + a pytorch `Dataset` over the trace format so that external training scripts consume a typed API. The in-engine NPC backend is a utility backend (ships permanently as bootstrap + regression baseline); `scoring` declarations drive utility scoring and are also written to traces so Python training jobs can reshape them into rewards externally.

### 1.1 Compiler chain at a glance

The compiler emits **four artefact classes per DSL program**:

| Emission | What gets emitted | Consumed by |
|---|---|---|
| **Scalar Rust** | Rust `fn` bodies + cascade closures for `SerialBackend` | `SerialBackend` (engine) |
| **GPU dispatch code** | Rust code calling engine kernel-dispatch API with FieldHandles + SPIR-V kernel names | `GpuBackend` (engine) |
| **SPIR-V kernels** | Compiled shader bytecode for DSL-specific mask predicates / cascade handlers / view reductions | `GpuBackend` (loaded via `GpuHarness::load_kernel`) |
| **Python dataclasses + `Dataset`** | Python module with one `@dataclass` per `entity`, `event`, and `@traced` view, plus a pytorch `Dataset` wrapping the trace ring-buffer format | External pytorch training scripts |

For a complete DSL program, all four artefacts are produced from the same DSL source. The Python emission is strictly a trace-consumer API: it has no runtime role inside the engine. The engine selects at init which backend to use — the compiler doesn't choose.

### 1.2 Declaration kinds

- **`entity`** — parameterization of one of the three predefined root kinds (Agent, Item, Group).
- **`event`** — typed, append-only records. The universal state-mutation channel.
- **`view`** — pure or event-folded derivations. Eager (`@materialized`) or lazy.
- **`physics`** cascade rule — phase-tagged transforms from events to events.
- **`mask`** — per-action validity predicates.
- **`verb`** — composition sugar that bundles mask + cascade + scoring into a named gameplay action.
- **`scoring`** — per-action personality-weighted utility table.
- **`invariant`** — static, runtime, or debug-only predicates over state.
- **`probe`** — named scenario + behavioral assertion evaluated against seeded trajectories.
- **`metric`** — runtime observability declaration.
- **`spatial_query`** — named per-pair candidate filter referenced by mask `from`-clauses and physics fold-iter sources via `spatial.<name>(...)` (§2.12).

---

## 2. Top-level declarations

### 2.1 `entity` declaration

`entity` parameterizes the three predefined root kinds. There is no grammar for declaring a fourth root kind.

```
entity <Name> : Agent {
  creature_type:        CreatureType,
  capabilities:         Capabilities { ... },
  default_personality:  Personality,
  default_needs:        Needs,
  hunger_drives:        [ HungerDriveKind, ... ],
  default_memberships:  [ GroupSpec, ... ],
  predator_prey:        { prey_of: [CreatureType], preys_on: [CreatureType] },
}

struct Capabilities {
  channels:       SortedVec<CommunicationChannel, 4>,   // §9 D30 — first-class communication modalities
  languages:      SortedVec<LanguageId, 4>,
  can_fly:        bool,
  can_build:      bool,
  can_trade:      bool,
  can_climb:      bool,
  can_tunnel:     bool,
  can_marry:      bool,
  max_spouses:    u8,                                   // §9 D17 polygamy cap
}

enum CommunicationChannel {                             // §9 D30
  Speech,             // linguistic — humans, elves, dragons; range set by hearing
  PackSignal,         // body + scent + short vocal — wolves, dogs, canids
  Pheromone,          // chemical gradient — insects, some reptiles
  Song,               // vocal non-linguistic long-range — birds, whales
  Telepathy,          // fantasy — faction-wide, range-less
  Testimony,          // written/symbolic — propagates via ItemKind::Document transfer
}

entity <Name> : Item {
  kind:           ItemKind,
  rarity:         Rarity,
  base_stats:     { ... },
  slots:          [EquipSlot, ...],
}

entity <Name> : Group {
  kind:                    GroupKind,
  eligibility_predicate:   <predicate>,
  recruitment_open:        bool,
  governance:              GovernanceKind,
  default_standings:       [ (GroupKind, Standing), ... ],
  required_fields:         [ FieldName, ... ],
}
```

Layout hints:

- `@materialized` — the compiler generates an update path (see §2.3).
- `@spatial` — the entity carries a `pos` field participating in spatial indices.
- `@capped(K)` — on collection fields; bounds the fixed storage layout.
- `@inline` — embed the child struct in the parent's SoA row rather than reference it.

The compiler emits a Rust struct, an enum-variant extension for the corresponding discriminator, a spawn-template, and a schema-hash contribution for the one-hot width of that discriminator (§4).

### 2.2 `event` declaration

Typed records. Every state mutation reaches primary state through an event.

```
event <Name> {
  <field>: <type>,
  ...
}

@replayable                  // default; event replays deterministically
@non_replayable              // text-gen events, LLM prose, side-channel
@high_volume                 // routes to a separate ring-buffer storage class
@traced                      // emitted to the trace ring buffer for external consumption
@gpu_amenable                // scalar fields only; triggers GPU event-fold codegen
```

`MemoryEvent` carries a `source: Source` field alongside the usual payload:

```
enum Source {
    Witnessed,              // confidence = 1.0
    TalkedWith(AgentId),    // confidence = 0.8
    Overheard(AgentId),     // confidence = 0.6
    Rumor { hops: u8 },     // confidence = 0.8^hops
    Announced(GroupId),     // confidence = 0.8
    Testimony(ItemId),      // confidence = item.trust_score
}
```

String payloads are permitted only on `@non_replayable` events; the compiler rejects `String` fields on replayable events. All load-bearing references are `AgentId`, `GroupId`, `ItemId`, `QuestId`, or `AuctionId` — no text.

The compiler emits:
- A variant on the runtime event enum for that kind-bucket.
- A fixed-capacity ring buffer, sized by `@high_volume` (larger) or the default.
- Serialisation hooks for replay (if `@replayable`).
- An entry in the trace-format event-vocabulary table (if `@traced`), plus an emitted Python `@dataclass`.
- A pattern-match kernel for cascade, scoring, and probe blocks.

### 2.3 `view` declaration

> ⚠️ **Audit 2026-04-26:** `@spatial query` declarations parse (`Decl::Query(QueryDecl)`) but `Compilation` has no `queries` field — silently dropped after parsing. Annotations `@indexed_on(field)`, `@top_k(K)`, `@backend(cpu|gpu)`, `@fidelity(>= Medium)` are accepted by the generic annotation parser but the resolver never interprets them. Fold body operator-set restriction (`ResolveError::UdfInViewFoldBody`) is **not implemented** — authors can write forbidden constructs in fold bodies without a compile-time error.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

Views are pure over their inputs. The compiler chooses between lazy evaluation and eager event-fold materialization per declaration.

```
view <name>(<args>) -> <type> {
  <expression>
}

@materialized(on_event = [EventName, ...])
view mood(a: Agent) -> f32 {
  initial: 0.0,
  on DamageReceived{target: a} { self -= 0.1 * e.damage / a.max_hp }
  on FriendDied{observer: a}   { self -= 0.3 }
  on NeedsSatisfied{agent: a}  { self += 0.05 * e.delta }
  clamp: [-1.0, 1.0],
}

@lazy                           // default; pure function evaluated at read
view is_hostile(a: Agent, b: Agent) -> bool {
  relationship(a, b).valence < HOSTILE_THRESH
  || groups_at_war(a, b)
  || predator_prey(a.creature_type, b.creature_type)
}

@spatial(radius, kind = [Agent|Resource|Structure])
query nearby_agents(self: Agent, radius: f32) -> [Agent]
sort_by distance(self, _) limit K { ... }

@indexed_on(field)             // forces a sorted or hashed index on `field`
@top_k(K)                      // bounded output size
@backend(cpu | gpu)            // override auto-selection
@fidelity(>= Medium)           // skip evaluation at lower fidelity
```

##### Sibling view-shape annotations

`@materialized` views may carry one of the following storage-shape annotations as a sibling (mutually exclusive with each other and with an explicit `storage = ...` argument inside `@materialized(...)`):

- `@symmetric_pair_topk(K = N)` — pair-keyed storage with top-K bounded retention. Lowers to `StorageHint::SymmetricPairTopK { k: N }`. Equivalent to `@materialized(storage = symmetric_pair_topk(K = N))`.
- `@per_entity_ring(K = N)` — per-entity ring buffer with capacity `N`. Lowers to `StorageHint::PerEntityRing { k: N }`. Equivalent to `@materialized(storage = per_entity_ring(K = N))`.

Both forms require an integer literal `K` argument; `K` is clamped into `u16`. Verified: `crates/dsl_ast/src/resolve.rs::lower_view_kind`.

The compiler emits:
- For `@materialized`: a field on the corresponding entity + an event-dispatch table mapping each `on_event` to the update body. GPU-amenable materializations sort events by target before commutative reduction.
- For `@lazy`: an inline function referenced from mask predicates, scoring expressions, and cascade bodies.
- For `@spatial`: routing to the appropriate spatial index.
- For `@top_k`: a fixed-cap partial-sort into a `SimScratch` buffer.

#### Fold body operator set

`@materialized` fold handler bodies are restricted to a closed operator set so the event-fold path compiles to commutative, GPU-friendly updates.

Allowed: compound self-assignment (`self +=`, `self -=`, `self *=`, `self /=`), plain self-assignment, conditionals, arithmetic, comparison, logical, bounded folds (`count`, `sum`, `min`, `max`), built-in math (`abs`, `floor`, `ceil`, `pow`, `ln`, `sqrt`, `clamp`), stdlib 1-hop accessors, `let x = <closed expr>` intermediates.

Forbidden: recursion, unbounded loops, user-declared helper functions called inside the fold body, calls to other views (including views that reference other views).

#### `@decay(rate = R, per = tick)`

Sugar layered on `@materialized` views. Lowers to the anchor pattern: stored value is `(base_at_anchor, anchor_tick)` and the observable at tick `t` is `base * rate^(t - anchor)`, clamped.

```
@materialized(on_event = [AgentAttacked, EffectDamageApplied],
              storage = pair_map)
@decay(rate = 0.98, per = tick)
view threat_level(a: Agent, b: Agent) -> f32 {
  initial: 0.0,
  on AgentAttacked { actor: b, target: a } { self += 1.0 }
  on EffectDamageApplied { actor: b, target: a } { self += 1.0 }
  clamp: [0.0, 1000.0],
}
```

Constraints: requires a sibling `@materialized` annotation; `rate` is a compile-time float literal in `(0.0, 1.0)`; `per` is the identifier `tick`.

### 2.4 `physics` cascade rule

> ⚠️ **Audit 2026-04-26:** `@before(Other)` / `@after(Other)` annotations are **silently accepted but never enforced** — no ordering-constraint graph built. Race detection (two unordered same-phase rules writing same field via non-commutative updates) is **not implemented**.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

Cascade rules respond to events and emit further events. They never mutate state directly — all writes go through `emit`. Phase-tagged ordering prevents races.

```
physics <name> @phase(pre | event | post | per_agent) {
  on <EventPattern> where <predicate> {
    emit <EventName> { <field>: <expr>, ... }
    ...
  }
}

physics collapse_chain @phase(event) {
  on VoxelDamaged{pos: p, new_integrity: i} where i <= 0.0 {
    emit FragmentCollapse { pos: p, cause: NpcHarvest }
    for q in voxel_neighbors_above(p) {
      emit VoxelDamaged { pos: q, amount: COLLAPSE_DAMAGE }
    }
  }
}
```

Annotations:
- `@phase(pre | event | post)` — fixed three-phase ordering. Each rule fires once per matching event in the cascade.
- `@phase(per_agent)` — extends the phase set: dispatch is `PerAgent` (one body invocation per alive agent per tick), and the handler's `on <Pattern>` clause is satisfied for every alive agent regardless of event-ring contents. Used by every per-tick movement / steering rule (`assets/sim/boids.sim`, `predator_prey_min.sim`, `target_chaser.sim`, `crowd_navigation_min.sim`, `particle_collision_min.sim`, `swarm_event_storm.sim`, `spatial_probe.sim`). Within the `per_agent` body, `self` is bound to the iterating agent slot and the lowering ignores `on_event`. Verified: `crates/dsl_compiler/src/cg/lower/physics.rs::is_per_agent_phase`.
- `@cpu_only` — the rule is dispatched only on the host (scalar Rust) backend; the GPU lowering skips it entirely. Used for handlers whose bodies touch host-only surfaces (chronicle `String` payloads, dev-time logging) — example: `assets/sim/predator_prey_min.sim::ChronicleDeath`. Verified: `crates/dsl_compiler/src/cg/lower/driver.rs::lower_all_physics` (`if rule.cpu_only { continue; }`).
- `@before(OtherRule)` / `@after(OtherRule)` — explicit ordering between rules in the same phase (parser accepted; enforcement pending — see audit callout above).
- `@terminating_in(N)` — asserts the cascade converges within N hops when self-emission is possible.

Compile-time validation (implemented):
1. Event-type existence on every `on` pattern.
2. Pattern-field mismatch against the event's declared fields.
3. Cycle detection in the event-type emission DAG; self-loops require `@terminating_in`.
4. Schema-hash drift on event or field references.
5. All referenced views, events, and fields resolve.

### 2.5 `mask` predicates

Per-action validity predicates. Cross-entity references are allowed; the compiler routes them through CPU when they require entity-table walks.

```
mask <ActionHead>(<bindings>) when <predicate>

mask Attack(t) when t.alive
                ∧ is_hostile(self, t)
                ∧ distance(self, t) < AGGRO_RANGE

mask PostQuest{type: Conquest, party: Group(g), target: Group(t)}
  when g in self.leader_groups
     ∧ g.kind in {Faction, Pack}
     ∧ g.military_strength > 0
     ∧ g.standings[t] != AtWar
     ∧ eligible_conquest_target(self, t)

mask Bid(auction_id, payment, _) when auction(auction_id).visibility contains self
                                    ∧ self.gold >= auction(auction_id).reserve_price.gold_amount()
                                    ∧ auction(auction_id).deadline > now
```

Supported operators: set membership (`contains`, `in`), quantifiers (`forall`, `exists`), bounded folds (`count`, `sum`, `max`, `min`), arithmetic comparison, and view calls. No user-defined functions inside masks — they cannot be safely compiled to GPU boolean kernels.

Compilation:
- Per-action validity buffers `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`, consumed by the utility backend.
- Mask predicates that reference only intrinsic scalar fields compile to SPIR-V; cross-entity predicates are CPU-patched into the same boolean buffer before scoring evaluation.
- Every predicate node has a stable AST ID; an explanation kernel reruns the predicate against a captured state snapshot for `trace_mask(agent, action, tick)`.

### 2.6 `verb` (composition sugar)

> ⚠️ **Audit 2026-04-26 — partial close (2026-05-03, Slice A):** the silent-drop is closed for the **mask + scoring**
> stages of verb expansion. The compiler now expands every `VerbIR` at lower time
> (`crates/dsl_compiler/src/cg/lower/verb_expand.rs`) into a synthetic `MaskIR` named `verb_<name>` (whose predicate is
> the verb's `when` clause) and — when the verb declares a `score` clause — a synthetic `ScoringEntryIR` named
> `verb_<name>` appended to the first `ScoringIR` block (or a synthesised block when none exist). The existing
> mask / scoring lowering and emit passes pick up the synthesised entries automatically; no new emit file is
> introduced. **Cascade injection is still deferred (SKIP):** the verb's `emit` clause needs an "action selected"
> event source the current event taxonomy doesn't expose, so verbs with non-empty `emits` surface a
> `LoweringError::VerbExpansionSkipped { reason: CascadeNeedsActionEvent }` diagnostic — the gap is visible at
> lower time rather than silently absent. Stress test: `crates/dsl_compiler/tests/verb_emit.rs`. The §2.9 `probe`
> callout below remains accurate; §2.8 `invariant` and §2.11 `metric` were partially closed 2026-05-03 (see those
> callouts).
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

`verb` declares a named gameplay action that composes an existing micro primitive with additional mask predicates, cascades, and scoring entries. It does NOT add to the closed categorical action vocabulary.

```
verb Pray(self, shrine: Structure) =
  action Converse(target: shrine.patron_agent_id)
  when  self.memberships contains Group{kind: Religion}
      ∧ distance(self, shrine) < REACH
  emit  PrayCompleted { prayer: self, shrine: shrine, faith_delta: 1.0 }
  score 0.5 * self.personality.piety
```

The compiler expands a `verb` into: (1) a mask entry narrowing an existing primitive, (2) a cascade handler that emits the declared event on successful action application, (3) a scoring entry appended to the scoring table.

Adding a `verb` does not bump the schema hash. Adding a new micro primitive does (see §4).

### 2.8 `invariant` declaration

> ⚠️ **Audit 2026-04-26 — partial close:** the silent-drop is closed for the **per-agent scalar-view-bounded shape** —
> `invariant <name>(a: Agent) @<mode> { <view>(a) <op> <literal_f32> }` over a `@materialized` `f32` view. The
> compiler synthesises a `pub fn check_<name>(view_storage: &[f32]) -> Vec<Violation>` into `invariants.rs` (added to
> `EmittedArtifacts.rust_files`); per-fixture runtimes consume it via `state.check_invariants()`. Unsupported
> shapes (multi-arg scope, quantifiers, field access, vec3 views) emit a `// SKIP <reason>` comment so the gap is
> visible at build time rather than silent. Implementation: `crates/dsl_compiler/src/cg/emit/invariants.rs`,
> stress test: `crates/predator_prey_runtime/tests/invariant_violation.rs`. The §2.6 `verb` callout above was
> partially closed 2026-05-03 by Slice A of the verb/probe/metric emit plan (mask + scoring; cascade still SKIP).
> §2.9 `probe` remains silent-drop; §2.11 `metric` was partially closed 2026-05-03 by Slice C.

```
invariant <name>(<scope>) @<mode> { <predicate> }

@static         // enforced at compile time via type system / dataflow analysis
@runtime        // checked after every cascade phase that writes the predicate's support
@debug_only     // runtime check, panic only in debug builds

invariant no_bigamy(a: Agent) @runtime {
  count(g in a.memberships where g.kind == Family && g.is_marriage) <= 1
}

invariant append_only_trace_schema() @static {
  forall (old_offset, new_offset) in schema_diff:
    new_offset >= old_offset || is_field_internal_append(old_offset, new_offset)
}
```

Cascades that write a field in the invariant's support must annotate `@must_preserve(<invariant>)`.

### 2.9 `probe` declaration

Named CI regression assertions: fixture scenario → seeded event trajectory → behavioural check against the utility backend. Probes live in `probes/` alongside their seed scenarios.

```
probe <Name> {
  scenario   "probes/<name>.toml"
  seed       <u64>
  ticks      <u32>
  tolerance  <f32>

  assert {
    <assert_expr>
  }
}

assert_expr := count_expr | prob_expr | mean_expr

count_expr  := "count" "[" filter "]" <comparator> <scalar>
prob_expr   := "pr"    "[" <action_filter> "|" <obs_filter> "]" <comparator> <prob>
mean_expr   := "mean"  "[" <scalar_expr> "|" <filter> "]" <comparator> <scalar>
```

Example:

```
probe LowHpFlees {
  scenario "probes/low_hp_1v1.toml"
  seed 42
  ticks 200
  tolerance 0.02

  assert {
    pr[ action.micro_kind in {Flee, MoveToward_away_from_threat}
      | self.hp_pct < 0.3 ]
    >= 0.80
  }
}
```

Probes compile to trajectory queries over the trace ring-buffer format. Schema-hash mismatch between a probe's reference fields and the current DSL is a hard error.

### 2.11 `metric` declaration

> ⚠️ **Audit 2026-04-26 — partial close (2026-05-03, Slice C):** the silent-drop is closed for the
> **wrapped-scalar shape** — `metric <name> = <kind>(<inner>) [emit_every N]` where `<kind>` is one of
> `gauge` / `counter` / `histogram` and `<inner>` is a numeric literal or `world.tick`. The compiler
> synthesises a `pub struct MetricsSink` with one field per metric (`Counter` / `Gauge` / `Histogram`)
> plus `pub fn record_tick(&mut self, world_tick: u64)` into `metrics.rs` (added to
> `EmittedArtifacts.rust_files`); per-fixture runtimes consume it via `state.metrics()` after each
> step. Per-metric `record_<name>(value: f32)` setters cover hand-driven cases. Unsupported value
> shapes (aggregates like `count(... events.this_tick where ...)` / `mean(... for ... in agents)`,
> view calls, quantifiers, the `conditioned_on` / `alert when` predicates) emit a `// SKIP <reason>`
> comment so the gap is visible at build time rather than silent. Implementation:
> `crates/dsl_compiler/src/cg/emit/metrics.rs`, stress test:
> `crates/swarm_storm_runtime/tests/metric_records.rs`. The `MetricKind` lift from stdlib-call to
> enum (audit at §7.3 below) is deliberately deferred — the emitter does the recognition today,
> per the verb/probe/metric plan's open question §C resolution.

```
metric {
  metric <name> = <expr>
    [ window <ticks> ]
    [ emit_every <ticks> ]
    [ conditioned_on <expr> ]
    [ alert when <value_comparator> ]
}

metric {
  metric cascade_iters_per_tick = histogram(cascade.iterations)
    window 1000 emit_every 100 alert when max_bin > 0.90

  metric alive_agents = gauge(count(agent where agent.alive))
    emit_every 10 alert when value < 1
}
```

Alerts emit structured log records to the engine telemetry sink. Metrics are sim-observability only — no coupling to training.

### 2.12 `spatial_query` declaration

A `spatial_query` declares a named per-pair candidate filter that mask `from`-clauses and physics fold-iter sources reference via the `spatial.<name>(...)` namespace (§7.2).

```
spatial_query <name>(self: AgentId, candidate: AgentId [, <value_arg>: <type> ...])
  = <filter_expr>
```

The body is a single expression returning `bool`. `self` is the iterating agent; `candidate` is the per-pair binder the spatial walk yields. Additional `value_arg` parameters are call-site arguments substituted into the filter at lowering time.

```
// assets/sim/boids.sim
spatial_query nearby_other(self: AgentId, candidate: AgentId) =
  candidate != self

// assets/sim/spatial_probe.sim
spatial_query nearby_probe(self: AgentId, candidate: AgentId) =
  candidate != self

// assets/sim/predator_prey_min.sim
spatial_query nearby_other(self: AgentId, candidate: AgentId) = candidate != self
```

The compiler emits the filter into the spatial walker (today: full per-cell scan; radius-bounded walks pending — see `cg/emit/kernel.rs::spatial_filtered_walk_body`). At reference sites, the call shape `spatial.<name>(self [, value_args...])` is the only authoring surface — the resolver routes it through `NamespaceCall { ns: Spatial, method: <name>, ... }` and rejects unknown names with `ResolveError::UnknownSpatialQuery`.

Verified: `crates/dsl_ast/src/parser.rs::spatial_query_decl`, `crates/dsl_compiler/src/cg/lower/driver.rs::mask_spatial_kind`.

---

## 3. Action vocabulary

### 3.1 Action heads

Closed action vocabulary. Adding to this vocabulary bumps the schema hash; adding a `verb` does not.

```
action {
  head categorical macro_kind: enum {
    NoOp, PostQuest, AcceptQuest, Bid, Announce, InviteToGroup, AcceptInvite,
    WithdrawQuest,
    SetStanding
  }

  head categorical micro_kind: enum {
    Hold,
    MoveToward, Flee,
    Attack, Cast, UseItem,
    Harvest, Eat, Drink, Rest,
    PlaceTile, PlaceVoxel, HarvestVoxel,
    Converse, ShareStory,
    Communicate, Ask, Read,
    Remember
  }

  head categorical channel: enum CommunicationChannel  // §9 D30
  head pointer target: select_from
    nearby_actors ∪ nearby_resources ∪ nearby_structures
    ∪ known_actors ∪ known_groups ∪ active_quests
    ∪ active_auctions ∪ incoming_invites ∪ memberships
    ∪ recent_memory_events
  head pointer fact_ref: select_from recent_memory_events
  head continuous pos_delta: vec3 ∈ [-1, 1]³
  head continuous magnitude: f32  ∈ [0, 1]

  // Structured parameter heads for macros
  head categorical quest_type:        enum QuestType
  head categorical party_scope:       enum PartyScope
  head categorical quest_target:      enum QuestTarget
  head categorical reward_kind:       enum RewardKind
  head categorical payment_kind:      enum PaymentKind
  head categorical group_kind:        enum GroupKind
  head categorical announce_audience: enum AnnounceAudience
  head categorical standing_kind:     enum StandingKind
  head categorical resolution:        enum Resolution
}
```

`macro_kind` and `micro_kind` are decoupled. When the model emits `macro_kind != NoOp`, the macro parameter heads drive the structured action. When `macro_kind == NoOp`, the `micro_kind` + its heads drive a per-tick physical/social primitive.

### 3.2 Action vocabulary (consolidated)

Four macro mechanisms + nineteen micro primitives (eighteen at runtime after `Read → Ask` lowering):

| Head | Variants |
|---|---|
| `macro_kind` | `NoOp`, `PostQuest{...}`, `AcceptQuest{quest_id, role_in_party}`, `WithdrawQuest{quest_id}`, `Bid{auction_id, payment, conditions}`, `Announce{audience, fact_ref}`, `InviteToGroup{kind, target, terms}`, `AcceptInvite{invite_id}`, `SetStanding{target_group, kind}` |
| `micro_kind` | `Hold`, `MoveToward(pos)`, `Flee(from)`, `Attack(t)`, `Cast(ability, target)`, `UseItem(slot, target)`, `Harvest(node)`, `Eat(food)`, `Drink(water)`, `Rest(loc)`, `PlaceTile(pos, type)`, `PlaceVoxel(pos, mat)`, `HarvestVoxel(pos)`, `Converse(t)`, `ShareStory(audience, topic)`, `Communicate(recipient, fact_ref)`, `Ask(target, query)`, `Read(doc)` *(sugar — compiler lowers to `Ask(doc, QueryKind::AboutAll)`)*, `Remember(entity, valence, kind)` |

Enums carried as parameter heads:
- `QuestType` — Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Custom, Conquest, MutualDefense, Submit, Found, Charter, Diplomacy, Marriage, Pilgrimage, Service, Heist, Trade, FulfillProphecy, Claim, Peace, Raid, HaveChild.
- `PartyScope` — `Individual(AgentId)`, `Group(GroupId)`, `Settlement(SettlementId)`, `Anyone`, `Role(RoleTag)`.
- `QuestTarget` — `Agent`, `Group`, `Location`, `Structure`, `CreatureType`, `Pair(Target, Target)`, `Predicate`, `Item`, `Region`, `Role(RoleTag)`.
- `RewardKind` — Gold, Xp, Items, Reputation, Faith, Spoils, Charter, Union, Reciprocal, Protection, Glory, Promise, Combination.
- `PaymentKind` — Gold, Commodity, Item, Service, Reputation, Combination.
- `GroupKind` — Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Alliance, Coven, Other.

For the detailed treatment of the four universal macro mechanisms (PostQuest, AcceptQuest, Bid, Announce) with gameplay scenario tables, see Appendix A.

### 3.3 `scoring`

Per-action personality-weighted utility values consumed by the utility backend.

```
scoring {
  Attack(t)       = 0.6 * self.personality.aggression
                    + 0.2 * (1.0 - self.hp_frac)
                    + 0.2 * is_hostile(self, t)
  Flee(from)      = 0.8 * (1.0 - self.hp_frac)
                    + 0.2 * threat_level(from)
  Eat             = 0.7 * hunger_pressure(self)
  Converse(t)     = 0.3 * self.personality.social
                    + 0.2 * relationship(self, t).valence

  PostQuest{type=Conquest, target=Group(g)}
                  = 0.9 * self.personality.ambition
                    * (strength(self.leader_groups[0]) / strength(g))
}
```

The compiler emits:
- A per-tick scoring kernel evaluating each masked candidate's utility expression, writing results into `utility_field[N × NUM_ACTIONS]` for argmax.
- Trace-emission hooks: `(tick, agent_id, action, score)` rows streamed into the trace ring buffer.

#### Gradient scoring modifiers

A scoring entry's right-hand side may include **gradient modifier** terms of the form `<expr> per_unit <delta>`:

```
scoring {
  Attack(t) = 0.5                                         // base
            + (if self.hp_pct < 0.3 { 0.4 } else { 0.0 }) // boolean modifier
            + threat_level(self, t) per_unit 0.02          // gradient modifier
            + (1.0 - t.hp_pct) per_unit 0.6
}
```

Constraints:
- `<delta>` must be a float literal at v1.
- `<expr>` must evaluate to `f32`.
- `per_unit` binds between `+`/`-` and `*`/`/`.
- `a per_unit b per_unit c` is a grammar error.

Adding a `scoring` entry does not bump the schema hash's state / event / rules components — only the `scoring_hash` sub-hash (§4).

---

## 4. Schema versioning

The schema hash is a content-addressed fingerprint over:

1. **`state_hash`** — entity field layouts (field order, offsets, types, bounded-collection capacities).
2. **`event_hash`** — declared event names and field shapes.
3. **`rules_hash`** — physics cascades, mask predicates, and verb declarations.
4. **`scoring_hash`** — scoring-table expressions and their input dependencies.

The combined schema hash is `sha256(state_hash || event_hash || rules_hash || scoring_hash)`. Loading a trace whose combined hash differs from the current DSL is a hard error.

There are no `@since` annotations, no padded-zero migration tables, no v1/v2/v3 schemas in the codebase — git holds history.

#### Schema emission mechanism

The compiler emits four sub-hashes and one combined hash:

```
schema.state_hash    = sha256(canonicalize(entity_field_layouts))
schema.event_hash    = sha256(canonicalize(event_taxonomy))
schema.rules_hash    = sha256(canonicalize(physics_cascades + masks + verbs))
schema.scoring_hash  = sha256(canonicalize(scoring_tables))
schema.combined_hash = sha256(state_hash || event_hash || rules_hash || scoring_hash)
```

On mismatch the error prints a diff of the four sub-hashes plus a git-remediation hint. CI guard: a commit that modifies entity fields, events, cascades/masks/verbs, or scoring declarations computes pre- and post-change hashes; non-append changes (remove, reorder, type change) block merge without an explicit schema bump.

---

## 5. Type system

### 5.1 Primitive and structural types

```
Scalar:    f32, i32, u32, u64, bool
Vector:    vec2, vec3, vec4
Time:      Tick (u64)
ID:        AgentId, ItemId, GroupId, QuestId, AuctionId, InviteId, StructureRef,
           EventId, PredicateId, TagId, ArchetypeId, RoleTag
FactRef:   { owner: AgentId, event_id: u32 }
FactPayload: { tick, kind, params: [u32; 4], author_id: AgentId }
Bounded:   Bitset<N>,
           SortedVec<T, K>,
           RingBuffer<T, K>,
           Map<K, V, Cap>,
           OneOf<K, V>
Struct:    Membership, Relationship, MemoryEvent, AgentData, Group, Item,
           Quest, Auction, Invite, Document, ChronicleEntry
Enum:      Source, AnnounceAudience, CommunicationChannel, CreatureType,
           ItemKind (incl. Document), GroupKind, AuctionKind, AuctionResolution,
           GovernanceKind, RoleTag, ChronicleCategory, DropReason, ...
```

`@spatial` types expose `pos: vec3` and participate in 3D spatial indices. Every bounded collection declares its capacity; capacity is part of the schema hash.

### 5.2 Forbidden types

> ❌ **Audit 2026-04-26:** Neither restriction is enforced. The resolver accepts `String` on replayable events and unbounded `Vec<T>` on entity/event fields silently.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

`String` is forbidden on `@replayable` events, on `@primary` state fields, and on any field referenced by a mask or reward predicate. Permitted only on `@non_replayable` events, the `chronicle_prose` side channel, and at display time.

Unbounded `Vec<T>` is forbidden on `entity`, `event`, and in-world struct declarations. Only `SimScratch` pools and world-level ring buffers may hold `Vec`.

### 5.3 Structural type relationships

- `Agent.pos: vec3` — world-space 3D position. Ground-locked creature_types have `pos.z` snapped by the post-phase cascade.
- `Agent.movement_mode: enum { Walk, Climb, Fly, Swim, Fall }` — cascade-updated; drives spatial-index sidecar (§9 #25). Slopes are Walk.
- `Agent.spouse_ids: SortedVec<AgentId, 4>` — polygamous-capable; cross-species gated by `can_marry(a, b)` view (§9 #17).
- `Relationship.believed_knowledge: Bitset<32>` — theory-of-mind projection.
- `Relationship.believed_knowledge_refreshed: [u32; 32]` — per-bit last-reinforced tick; decay computed at read time.
- `KnowledgeDomain.volatility: enum { Short=500, Medium=20_000, Long=1_000_000 }` — per-variant half-life in ticks.
- `Agent.memberships: SortedVec<Membership, 8>` — capped at K=8 simultaneous groups.
- `Agent.known_actors: SortedVec<Relationship, 32>` — LRU by `last_known_age`; pinned (spouse, mentor) entries survive eviction.
- `Agent.memory_events: RingBuffer<MemoryEvent, 64>` — with up to 5 "indelible" slots reserved for `emotional_impact > 0.8`.
- `Agent.behavior_profile: SortedVec<(TagId, f32), 16>` — capped; lowest-weight eviction.
- `Document: Item{ kind: Document, author_id: AgentId, tick_written: Tick, seal: Option<SealId>, facts: SortedVec<FactPayload, 16> }` — reader derives confidence from `(relationship_to_author, seal_validity, known_author_biases)` (§9 #27).
- `Quest.fact_refs: SortedVec<FactPayload, 8>` — materialised at `QuestPosted`.
- `Quest.child_quest_ids: SortedVec<QuestId, 8>` — sub-quest tracking; drives recursive cancellation and spoils distribution (§9 #15).
- `Group.members` — derived view over agent `memberships`, not stored.
- `Group.standings: SortedVec<(GroupId, StandingKind, f32), 16>` — updated by `SetStanding` macro.
- `ChronicleEntry` — `{ tick, category: ChronicleCategory, entity_ids: [AgentId; 4], template_id: u32 }` with text via side-channel.

---

## 6. Runtime semantics

### 6.1 Tick pipeline

```
pre phase    (@phase(pre) rules; read-only state access)
       ↓
event emission (agent actions + world events emitted this tick)
       ↓
materialized-view updates (event-fold per @materialized view, sorted-key reduction)
       ↓
mask evaluation (GPU + CPU patch for cross-entity)
       ↓
scoring evaluation (per-action utility expression per masked candidate)
       ↓
action selection (utility backend argmax; shuffled-but-seeded order via rng_state)
       ↓
cascade fixed-point (@phase(event) rules; up to terminating_in(N) hops)
       ↓
post phase   (@phase(post) rules; ground-snap, overhear scan,
              chronicle / metric emission, @traced event flush)
       ↓
metric sinks updated; alerts emitted
```

**Ground-snap phase** — a post-MovementApplied cascade rule for ground-locked `creature_type`s:

```
physics ground_snap @phase(post) {
  on MovementApplied{agent: a} where creature_type(a).ground_locked {
    let h = match a.inside_building_id {
      Some(b) => floor_height(a.pos, b),
      None    => surface_height(a.pos.xy),
    };
    emit SetPos { agent: a, pos: vec3(a.pos.x, a.pos.y, h + creature_height(a) / 2) }
  }
}
```

**Announce propagation** — a cascade rule triggered by `Announce`:

```
physics announce_broadcast @phase(event) {
  on AnnounceAction{speaker: s, audience: aud, channel: ch, fact_ref: f} {
    let range = view::channel_range(ch, s);
    let shares = |r| r.capabilities.channels.contains(ch);
    let recipients = match aud {
      Group(g)    => members_of(g).filter(|r| shares(r) && hearing_eligible(s, r, ch))
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
      Area(c, r)  => query::nearby_agents_3d(c, r.min(range))
                                   .filter(|r| r != s && shares(r))
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
      Anyone      => query::nearby_agents_3d(s.pos, range)
                                   .filter(|r| r != s && shares(r))
                                   .take(MAX_ANNOUNCE_RECIPIENTS),
    };
    for r in recipients {
      emit RecordMemory { observer: r, payload: copy_fact(f),
                          source: match aud { Group(g) => Announced(g), _ => Overheard(s) },
                          confidence: 0.8 }
    }
    // Overhear scan
    let overhear_range = view::channel_range(ch, s) * OVERHEAR_RANGE_FRACTION;
    for b in query::nearby_agents_3d(s.pos, overhear_range)
                 .filter(|b| b != s && !in_audience(b, aud)
                          && b.capabilities.channels.contains(ch)
                          && overhear_eligible(s, b, ch)) {
      let cat = overhear_category(s, b, ch);
      let base = channel_overhear_base(ch, cat);
      let conf = base * exp(-planar_distance(s, b) / overhear_range);
      emit RecordMemory { observer: b, payload: copy_fact(f),
                          source: Overheard(s), confidence: conf }
    }
  }
}
```

Runtime constants: `MAX_ANNOUNCE_RECIPIENTS` (default 64); `OVERHEAR_RANGE_FRACTION` (default 0.2). Per-channel defaults (`channel_range(ch, sender)`): `Speech` → `SPEECH_RANGE * sender.vocal_strength` (default 30m × modifier); `PackSignal` → `PACK_RANGE` (default 20m); `Pheromone` → `PHEROMONE_RANGE * wind_factor()` (default 40m × wind); `Song` → `LONG_RANGE_VOCAL` (default 200m); `Telepathy` → `f32::INFINITY`; `Testimony` → `0.0` (propagates via item transfer). These values live in `assets/sim/config.sim` and `assets/config/default.toml`.

### 6.2 Determinism contract

- All randomness flows through `rng_state: u64` (PCG-style) plus per-agent derived streams. Per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose_tag)`.
- Agent processing order is deterministically shuffled each tick via an RNG-seeded permutation.
- Events emitted within a phase are collected in append order; handlers process them deterministically.
- Text generation (`ChronicleEntry.text`, LLM prose) is `@non_replayable` (§9 #21): templates render eagerly at event emission; an async LLM pass may rewrite entries in flagged categories.

### 6.3 Replay scope

Practical replay window is the bug-report scope (~1000 ticks, ~2–5 GB compressed via zstd). Full-run replay at 200K agents is not a goal.

Replay artefacts per recorded segment:
1. Initial snapshot (zstd-compressed safetensors).
2. Event log with `{ tick, kind: u16, params: [u32; 4], source_agent: u32 }` — zstd-framed per 500-tick segment.
3. Tick-boundary RNG checkpoints every 100 ticks.
4. Schema combined-hash.
5. (Bug-report artefacts only) Chronicle template library snapshot.

### 6.4 Save/load

Snapshot contents:

- `tick`, `rng_state`, `next_id`, `max_entity_id`.
- Primary fields of all entities (hot + cold).
- Group definitions (memberships, standings, treasuries, active_quests).
- Quest / auction / invite state.
- `tiles`, `build_seeds`, `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan.seed`.
- Bounded rings: chronicle, world_events, per-agent memory_events.
- Materialized views (serialized) with schema-hash guard. Mismatch forces rebuild from baseline.
- Event log ring (fixed-cap).
- GPU-resident buffers downloaded and serialized; re-uploaded on load.

Format: safetensors-compatible, length-prefixed, deserialize-in-place. Zero-malloc load uses pre-allocated agent slot pool.

---

## 7. Stdlib reference

The stdlib is the set of Rust-backed primitives the compiler recognises without requiring a DSL declaration. It contains **no game-level views** — no `is_hostile`, no `at_war`, no `married`, no `threat_level`, no `hunger_pressure`, no `reputation_log`. Those are game-authored and live in the game's own `.sim` sources as `view` declarations.

Every identifier listed here is reserved by the compiler and resolves without DSL source.

### 7.1 Primitive functions

Primitives are `Builtin` variants in `crates/dsl_compiler/src/ir.rs`. They are resolved in `crates/dsl_compiler/src/resolve.rs::stdlib::seed()`.

**Spatial:**

| Name | Signature | Notes |
|---|---|---|
| `distance` | `(Vec3, Vec3) -> f32` | 3D Euclidean distance. |
| `planar_distance` | `(Vec3, Vec3) -> f32` | Distance in the XY plane; ignores Z. |
| `z_separation` | `(Vec3, Vec3) -> f32` | `abs(a.z - b.z)`. |

**Numeric:**

| Name | Signature | Notes |
|---|---|---|
| `abs` | `(f32) -> f32` / `(i32) -> i32` / `(i64) -> i64` | Absolute value. |
| `min` | `(T, T) -> T` where `T ∈ {i32, i64, u32, u64, f32}` | Pairwise minimum; also aggregation: `min(x in xs, x.field)`. |
| `max` | `(T, T) -> T` where `T ∈ {i32, i64, u32, u64, f32}` | Pairwise maximum. Same aggregation overload. |
| `clamp` | `(T, T, T) -> T` | `(value, lo, hi)`. |
| `floor` | `(f32) -> f32` | Round toward `-∞`. |
| `ceil` | `(f32) -> f32` | Round toward `+∞`. |
| `round` | `(f32) -> f32` | Banker's rounding (ties to even). |
| `ln` | `(f32) -> f32` | Natural log. |
| `log2` | `(f32) -> f32` | |
| `log10` | `(f32) -> f32` | |
| `sqrt` | `(f32) -> f32` | |

**ID dereference:**

| Name | Signature | Notes |
|---|---|---|
| `entity` | `(AgentId) -> EntityRow` / `(ItemId) -> EntityRow` / `(GroupId) -> EntityRow` | Resolves an ID handle to its table row. |

**Aggregations and quantifiers** (parsed as dedicated AST nodes `Fold` / `Quantifier`, not calls):

| Name | Syntax | Notes |
|---|---|---|
| `count` | `count(b in iter where pred)` / `count[pred]` | Fold. |
| `sum` | `sum(b in iter, expr)` | Fold. |
| `forall` | `forall x in set: body` | Quantifier; returns `bool`. |
| `exists` | `exists x in set: body` | Quantifier; returns `bool`. |

All primitive functions are pure over their arguments. The only stdlib surface that draws randomness is the `rng` namespace below.

### 7.2 Namespaces

Each namespace is a pre-seeded identifier. Field access (`world.tick`) resolves to `NamespaceField { ns, field, ty }`. Method calls (`rng.uniform(...)`) resolve to `NamespaceCall { ns, method, args }`.

Six **legacy collection namespaces** — `agents`, `items`, `groups`, `quests`, `auctions`, `tick` — are used as iteration sources in aggregations and will be replaced with first-class query surfaces in a later milestone.

#### `world`

| Field | Type |
|---|---|
| `world.tick` | `u64` |
| `world.seed` | `u64` |
| `world.n_agents_alive` | `u32` |

#### `cascade`

Valid only inside metric / probe bodies and post-phase handlers.

| Field | Type |
|---|---|
| `cascade.iterations` | `u32` |
| `cascade.phase` | `enum CascadePhase { Pre, Event, Post }` |

#### `event`

Valid inside a physics or fold handler.

| Field | Type |
|---|---|
| `event.kind` | `EventKindId` |
| `event.tick` | `u64` |

#### `mask`

Primarily used in metric bodies.

| Field | Type |
|---|---|
| `mask.rejections` | `u64` |

#### `action`

Valid inside a scoring body.

| Field | Type |
|---|---|
| `action.head` | `ActionHeadKind` |
| `action.target` | `Option<AnyId>` |

#### `rng`

Deterministic random sampling backed by `SimState.rng_state`.

| Method | Signature |
|---|---|
| `rng.uniform(lo, hi)` | `(f32, f32) -> f32` |
| `rng.gauss(mu, sigma)` | `(f32, f32) -> f32` |
| `rng.coin()` | `() -> bool` |
| `rng.uniform_int(lo, hi)` | `(i32, i32) -> i32` |

#### `query`

> ⚠️ **Audit 2026-04-26:** `query.nearby_agents`, `query.within_planar`, `query.nearby_items` resolve in `stdlib::method_sig` but **emit only in mask `from`-clauses**. They have **no lowering in `emit_physics.rs`, `emit_scoring.rs`, or `emit_view.rs`** — they fall through to `EmitError::Unsupported`. Highest-impact gap blocking neighbour-iterating physics rules. `query.agents_within` is a non-spec alias that does not exist; use `query.nearby_agents`.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

| Method | Signature |
|---|---|
| `query.nearby_agents(pos, radius)` | `(Vec3, f32) -> [AgentId]` |
| `query.within_planar(pos, radius)` | `(Vec3, f32) -> [AgentId]` — planar (XY) distance |
| `query.nearby_items(pos, radius)` | `(Vec3, f32) -> [ItemId]` |
| `query.nearest_hostile_to(actor, radius)` | `(AgentId, f32) -> Option<AgentId>` — nearest hostile (species predicate) within `radius`. Argmin on distance; ties broken on raw `AgentId`. CPU lowering: `crate::spatial::nearest_hostile_to`. |
| `query.nearest_hostile_to_or(actor, radius, fallback)` | `(AgentId, f32, AgentId) -> AgentId` — sentinel-returning sibling of `nearest_hostile_to`. Returns `fallback` when no hostile is found, so the physics rule body can stay in the GPU-emittable subset (no `if let Some` narrowing). Verified: registered in `crates/dsl_compiler/src/cg/lower/driver.rs::populate_namespace_registry` and exercised by the CG-level spatial lowering. |
| `query.nearby_kin(actor, radius)` | `(AgentId, f32) -> [AgentId]` — same-species spatial scan, returns every alive same-species neighbour within `radius`. Used by the `fear_spread_on_death` pattern to emit a `FearSpread` event per kin. Bounded by `SpatialHash::within_radius`'s cell-reach cap. |
| `query.nearest_k(center, k, max_radius)` | `(AgentId, u32 literal, f32) -> [AgentId]` — top-K topological neighbour query. Returns up to `k` same-species neighbours sorted ascending by distance (ties broken on `AgentId`). The `k` arg MUST be a non-negative integer literal so the GPU emitter can bake the heap size into a compile-time `array<u32, K>`. CPU lowering: `crate::spatial::nearest_k`. |

> Status: every entry above resolves through `stdlib::method_sig`; `nearest_hostile_to_or` additionally has a registered WGSL stub in `populate_namespace_registry`. The other three (`nearest_hostile_to`, `nearby_kin`, `nearest_k`) are CPU-only at present. Treat them as part of the spec contract — they back the engagement / kin-broadcast physics rules called out in the registry comments.

#### `spatial`

References to `spatial_query <name>(...)` declarations (§2.12). The method set is open-ended: every declared `spatial_query` adds one method to this namespace.

| Method | Signature |
|---|---|
| `spatial.<name>(self [, value_args...])` | `(AgentId [, ...]) -> [AgentId]` — yields per-pair `candidate` bindings filtered by the named `spatial_query` body. Used as a `from`-clause source for masks and as an iter source inside fold / `for` bodies. |

Examples (all from `assets/sim/`):

```
let n = sum(other in spatial.nearby_other(self) where ...) // boids.sim
for prey in spatial.closest_prey(self) { ... }              // predator_prey.sim
for other in spatial.nearby_particles(self, R) { ... }      // particle_collision.sim
```

Resolution: the symbol-table seeds `spatial` as `NamespaceId::Spatial`; method dispatch consults `symbols.spatial_queries`. Unknown names surface as `ResolveError::UnknownSpatialQuery`. The `query.<name>` namespace is NOT an alias for `spatial.<name>` — `query.*` is the older flat-call surface (the table above), and `spatial.*` is the user-declared filter surface introduced by Phase 7. Verified: `crates/dsl_ast/src/resolve.rs:104` (namespace seed), `crates/dsl_compiler/src/cg/lower/driver.rs::mask_spatial_kind` (lowering routes only `NamespaceId::Spatial`).

#### `voxel`

> ⚠️ **Audit 2026-04-26:** All three `voxel.*` methods emit only in `emit_physics.rs`. View bodies and scoring bodies that call `voxel.surface_height` etc. fail emission with `Unsupported`.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for detail.

| Method | Signature |
|---|---|
| `voxel.neighbors_above(pos)` | `(Vec3) -> [Vec3]` |
| `voxel.neighbors_below(pos)` | `(Vec3) -> [Vec3]` |
| `voxel.surface_height(x, y)` | `(f32, f32) -> i32` |

#### `agents`

> 🤔 **Audit 2026-04-26 — stale name (4 methods):** Spec names below are wrong. Implementation uses **absolute expiry tick**, not remaining ticks:
> - `agents.stun_remaining_ticks(a)` → actual: `agents.stun_expires_at_tick(a)`
> - `agents.set_stun_remaining_ticks(a, v)` → actual: `agents.set_stun_expires_at_tick(a, v)`
> - `agents.slow_remaining_ticks(a)` → actual: `agents.slow_expires_at_tick(a)`
> - `agents.set_slow_remaining_ticks(a, v)` → actual: `agents.set_slow_expires_at_tick(a, v)`
>
> Active stun semantic: `world.tick < agents.stun_expires_at_tick(a)`. Task 143 changed the storage but did not update this spec.
>
> Audit's "implementation-extra" inventory (`agents.is_hostile_to`, `agents.engaged_with*`, `agents.record_memory`, `agents.cooldown_next_ready*`, `query.nearest_hostile_to*`, `query.nearby_kin`, `query.nearest_k`, full `abilities.*` and `terrain.*` namespaces) is now catalogued below in the Engagement / hostility / memory / cooldown extensions table and the new `abilities` / `terrain` namespace sections.
> See `docs/superpowers/notes/2026-04-26-audit-language-stdlib.md` for full inventory (57 stdlib functions: 30 ✅ / 16 ⚠️ / 7 ❌ / 4 🤔).

Per-agent slot accessors used by `physics` cascades.

| Method | Signature |
|---|---|
| `agents.alive(a)` | `(AgentId) -> bool` |
| `agents.pos(a)` | `(AgentId) -> Vec3` |
| `agents.vel(a)` | `(AgentId) -> Vec3` — per-agent velocity. Companion to `pos`; written by `set_vel`. |
| `agents.set_pos(a, v)` | `(AgentId, Vec3) -> ()` — write-back of the per-agent position slot. The canonical `@phase(per_agent)` per-tick movement step (`set_vel` + `set_pos`) is exercised by every shipped fixture (boids, predator-prey, particle-collision, target-chaser, swarm-event-storm, crowd-navigation, spatial-probe). |
| `agents.set_vel(a, v)` | `(AgentId, Vec3) -> ()` |
| `agents.hp(a)` | `(AgentId) -> f32` |
| `agents.max_hp(a)` | `(AgentId) -> f32` |
| `agents.shield_hp(a)` | `(AgentId) -> f32` |
| `agents.attack_damage(a)` | `(AgentId) -> f32` — falls back to `config.combat.attack_damage` (default 10.0) |
| `agents.set_hp(a, v)` | `(AgentId, f32) -> ()` |
| `agents.set_shield_hp(a, v)` | `(AgentId, f32) -> ()` |
| `agents.kill(a)` | `(AgentId) -> ()` — flips alive bit and tears agent out of spatial index. Idempotent. |
| `agents.stun_remaining_ticks(a)` | `(AgentId) -> u32` — **spec name stale; see audit callout** |
| `agents.set_stun_remaining_ticks(a, v)` | `(AgentId, u32) -> ()` — **spec name stale; see audit callout** |
| `agents.slow_remaining_ticks(a)` | `(AgentId) -> u32` — **spec name stale; see audit callout** |
| `agents.set_slow_remaining_ticks(a, v)` | `(AgentId, u32) -> ()` — **spec name stale; see audit callout** |
| `agents.slow_factor_q8(a)` | `(AgentId) -> i16` — q8 fixed-point speed multiplier under slow |
| `agents.set_slow_factor_q8(a, v)` | `(AgentId, i16) -> ()` |
| `agents.gold(a)` | `(AgentId) -> i64` — signed: debt is representable |
| `agents.set_gold(a, v)` | `(AgentId, i64) -> ()` |
| `agents.add_gold(a, delta)` | `(AgentId, i64) -> ()` — `gold += delta` via `i64::wrapping_add` |
| `agents.sub_gold(a, delta)` | `(AgentId, i64) -> ()` — `gold -= delta` via `i64::wrapping_sub` |
| `agents.adjust_standing(a, b, delta)` | `(AgentId, AgentId, i16) -> ()` — symmetric pair-standing, clamped to `[-1000, 1000]` |
| `agents.hunger(a)` | `(AgentId) -> f32` — 0.0 sated … 1.0 starving |
| `agents.thirst(a)` | `(AgentId) -> f32` — 0.0 sated … 1.0 parched |
| `agents.rest_timer(a)` | `(AgentId) -> f32` — 0.0 rested … 1.0 exhausted |

##### Engagement / hostility / memory / cooldown extensions

These rows extend the `agents` namespace with the engagement-lock, hostility-predicate, memory-emit, and cooldown surfaces called out in the audit. All resolve through `crates/dsl_ast/src/resolve.rs::stdlib::method_sig`; the engagement and hostility predicates additionally appear in the CG namespace registry (`populate_namespace_registry`).

| Method | Signature |
|---|---|
| `agents.is_hostile_to(a, b)` | `(AgentId, AgentId) -> bool` — species-level hostility predicate. Returns `false` when either slot is empty. The DSL-declared `view is_hostile(a, b)` body forwards here so the hostility matrix stays on `CreatureType::is_hostile_to` without a hand-written shim. |
| `agents.engaged_with(a)` | `(AgentId) -> Option<AgentId>` — wraps `state.agent_engaged_with(a)`. Returns the engagement partner if any, else `None`. Used by mask predicates (the engagement-lock clause in `mask Cast`). |
| `agents.engaged_with_or(a, default)` | `(AgentId, AgentId) -> AgentId` — unwrap-or-default sibling of `engaged_with`. Returns the partner if any, else `default`. Lets the rule body sentinel on the agent itself when no partner is set, keeping the body GPU-emittable. |
| `agents.set_engaged_with(a, partner)` | `(AgentId, AgentId) -> ()` — eagerly writes the SoA `hot_engaged_with` slot to `Some(partner)` so same-tick cascade handlers observe the new partner before the view-fold rebuild. |
| `agents.clear_engaged_with(a)` | `(AgentId) -> ()` — paired with `set_engaged_with`; clears the slot. Split form avoids needing an `Option` ctor on the DSL surface for the two-arg setter. |
| `agents.record_memory(observer, source, payload, confidence, tick)` | `(AgentId, AgentId, _, f32, u32) -> ()` — quantises `confidence` to q8, constructs a `MemoryEvent`, and pushes it onto the observer's cold memory ring. Primitive used by the `record_memory` physics rule. |
| `agents.cooldown_next_ready(a)` | `(AgentId) -> u32` — read of the per-agent global-cooldown cursor. |
| `agents.set_cooldown_next_ready(a, tick)` | `(AgentId, u32) -> ()` — direct write of the same cursor (legacy single-cursor form). |
| `agents.record_cast_cooldowns(caster, ability, now)` | `(AgentId, AbilityId, u32) -> ()` — split-primitive form (2026-04-22 ability-cooldowns subsystem): writes BOTH the per-agent global cursor (with `config.combat.global_cooldown_ticks`) and the per-(agent, slot) local cursor (with the ability's own `gate.cooldown_ticks`). Replaces `set_cooldown_next_ready` in the `physics cast` rule; fixes the shared-cursor bug. |

> `agents.set_move_target(member, dest)` appears in `assets/sim/crowd_navigation.sim:268` (the full fixture; the runtime crate compiles `crowd_navigation_min.sim` instead). Parser accepts; lowering: TODO — verify before relying on this surface. Not present in `stdlib::method_sig` today.

These map one-to-one to `SimState::agent_alive`, `agent_pos`, `agent_hp`, `agent_max_hp`, `agent_shield_hp`, `agent_attack_damage`, `set_agent_hp`, `set_agent_shield_hp`, `kill_agent`, etc. The compiler emits namespace calls as direct method invocations on the `&mut SimState` parameter passed to a cascade handler.

#### `abilities`

Ability-registry accessor used by the `cast` physics rule and by mask gates. The singular alias `ability::` shares the same method schema (added 2026-04-22 so designers can write `ability::on_cooldown(<slot>)` in mask / physics predicates with the natural singular form).

| Method | Signature |
|---|---|
| `abilities.is_known(id)` | `(AbilityId) -> bool` — registry-membership predicate; lets the cast handler bail silently on an unregistered ability id. |
| `abilities.cooldown_ticks(id)` | `(AbilityId) -> u32` — returns the program's `gate.cooldown_ticks`. |
| `abilities.effects(id)` | `(AbilityId) -> [EffectOp]` — yields the program's ordered `EffectOp` list for the dispatch for-loop to iterate. |
| `abilities.known(agent, ability)` | `(AgentId, AbilityId) -> bool` — 2-arg mask-side sibling of `is_known`. The emitter lowers it to a registry `get(...).is_some()`, ignoring the agent argument (mask-gate does not yet key on per-agent spellbooks). |
| `abilities.cooldown_ready(agent, ability)` | `(AgentId, AbilityId) -> bool` — folds `state.tick >= agent_cooldown_next_ready` into a single boolean the mask predicate can `&&`-chain. |
| `abilities.on_cooldown(slot)` | `(u8 literal) -> bool` — designer-facing inverted form of `cooldown_ready`. Returns `true` while the slot is still on cooldown (the natural "gate blocks" reading). The implicit subject is the rule's `self`; the slot arg coerces to `u8` via the argument lowering in the emitter. Use the `ability::on_cooldown(s)` singular form in mask / physics predicates. |
| `abilities.hostile_only(ability)` | `(AbilityId) -> bool` — exposes the program's `Gate.hostile_only` field for target-side filters. |
| `abilities.range(ability)` | `(AbilityId) -> f32` — exposes the program's `Area::SingleTarget.range` field for target-side filters. |

#### `terrain`

Terrain backend accessor (MVP Task 81). Routes through `SimState.terrain: Arc<dyn TerrainQuery>` at emit time. The flat-plane default keeps every legacy scoring / physics path unchanged; examples and future mask rows opt in by reading through this namespace.

| Method | Signature |
|---|---|
| `terrain.line_of_sight(from, to)` | `(Vec3, Vec3) -> bool` |

The `height_at` / `walkable` surface on `TerrainQuery` is intentionally not exposed at this slice — the smallest possible method set the height-bonus scoring gate needs.

#### Roadmap-stub namespaces

The following namespaces resolve in `stdlib::method_sig` (so calls type-check) but their backing runtime is not yet wired. Predicates evaluate against placeholder state; emitters return `Unsupported`. Each is keyed to a Roadmap subsystem; the grammar stub keeps the surface stable until the subsystem lands.

| Namespace | Methods | Roadmap |
|---|---|---|
| `membership` | `is_group_member(agent, kind) -> bool`, `is_group_leader(agent) -> bool`, `can_join_group(agent, group) -> bool`, `is_outcast(agent, group) -> bool` | §1 (Memberships) |
| `relationship` | `is_hostile(a, b) -> bool`, `is_friendly(a, b) -> bool`, `knows_well(a, b) -> bool` | §3 (Relationships) — replaces Combat Foundation's stub `is_hostile_to` once the relationship runtime lands |
| `theory_of_mind` | `believes_knows(observer, subject, domain) -> bool`, `can_deceive(observer, subject, fact) -> bool`, `is_surprised_by(observer, subject, domain) -> bool` | §6 (Theory-of-mind) — bit reads against the 32-bit `Relationship.believed_knowledge` domain bitset |
| `group` | `exists(id) -> bool`, `is_active(id) -> bool`, `has_leader(id) -> bool`, `can_afford_from_treasury(g, cost) -> bool` | §7 (Groups) — singular `group` is distinct from the legacy collection accessor `groups` |
| `quest` | `can_accept(agent, q) -> bool`, `is_target(entity, q) -> bool`, `party_near_destination(party, q) -> bool` | §12 (Quests) — singular `quest` is distinct from the legacy collection accessor `quests` |

Status: Parser accepts; resolver gives each call its declared return type. Lowering: TODO — verify the backing runtime before relying on these surfaces. See `docs/superpowers/roadmap.md` for the per-subsystem cutover plan.

### 7.3 Metric wrappers are NOT stdlib functions

`histogram(expr)`, `gauge(expr)`, `counter(expr)` appear in the DSL but are part of the **metric-declaration grammar**, not callable stdlib functions. They belong alongside `window` / `emit_every` / `alert when` on `MetricDecl`, as a `MetricKind` enum. **TODO:** lift them into `MetricKind` in the AST / parser / IR.

### 7.4 What is NOT in stdlib

The following names are **game-specific derivations**, not engine primitives. A game that uses them must declare them in its own DSL (`view <name>(...)` for scalar derivations):

- `is_hostile`, `at_war`, `groups_at_war`
- `married`
- `threat_level`
- `hunger_pressure`, `hunger_urgency`
- `reputation`, `reputation_log`
- `relationship`
- `eligibility` / `eligible_*` predicates
- `can_marry`, `can_trade`, any `can_*` capability derivation
- `predator_prey`, `prey_of`

### 7.5 Adding to stdlib

Requires all three of:
1. An entry in this document, with a pinned signature.
2. A `Builtin` variant or `NamespaceId` / `stdlib::field_type` / `stdlib::method_sig` extension in `crates/dsl_compiler/src/`, plus a fixture in `crates/dsl_compiler/tests/fixtures/` exercising it and a blessed IR golden.
3. An engine-side implementation at `crates/engine/src/stdlib/<name>.rs`.

A stdlib addition bumps the compiler vocabulary, the schema hash, and requires buy-in from downstream training / replay consumers.

---

## 8. Scoring grammar specifics — field-id mapping

The compiler-emitted scoring table (`engine_rules::scoring`) encodes every predicate as a `PredicateDescriptor` with a `field_id: u16`. This section pins the mapping from DSL field reference to the integer id the scorer's `read_field` function dispatches on.

Changing this table breaks the committed `SCORING_TABLE` constants and bumps `SCORING_HASH`. Treat every row as a stable contract.

### 8.1 Agent-local fields (`self.*`)

| `field_id` | DSL reference | Engine accessor | Notes |
|-----------:|---|---|---|
| 0 | `self.hp` | `state.agent_hp(agent).unwrap_or(0.0)` | raw hit points |
| 1 | `self.max_hp` | `state.agent_max_hp(agent).unwrap_or(1.0)` | defaults to 1.0 so `hp_pct` is well-defined |
| 2 | `self.hp_pct` | `hp / max_hp` (derived) | 0.0..=1.0 normalised |
| 3 | `self.shield_hp` | `state.agent_shield_hp(agent).unwrap_or(0.0)` | |
| 4 | `self.attack_range` | `state.agent_attack_range(agent).unwrap_or(2.0)` | |
| 5 | `self.hunger` | `state.agent_hunger(agent).unwrap_or(0.0)` | 0.0 sated … 1.0 starving |
| 6 | `self.thirst` | `state.agent_thirst(agent).unwrap_or(0.0)` | 0.0 sated … 1.0 parched |
| 7 | `self.fatigue` | `state.agent_rest_timer(agent).unwrap_or(0.0)` | Engine slot is `rest_timer`; DSL surface calls it `fatigue`. 0.0 rested … 1.0 exhausted. |
| 8 | `self.personality.aggression` | placeholder `0.0` (task 141) | Personality SoA not yet wired. |
| 9 | `self.personality.social_drive` | placeholder `0.0` (task 141) | |
| 10 | `self.personality.ambition` | placeholder `0.0` (task 141) | |
| 11 | `self.personality.altruism` | placeholder `0.0` (task 141) | |
| 12 | `self.personality.curiosity` | placeholder `0.0` (task 141) | |

### 8.2 Target-side fields (`target.*`)

Reserved range `field_id ∈ [0x4000, 0x8000)`. A `target.*` read on a self-only row (where `target == None`) surfaces as `f32::NAN`.

| `field_id` | DSL reference | Engine accessor |
|-----------:|---|---|
| 0x4000 | `target.hp` | `state.agent_hp(target).unwrap_or(0.0)` |
| 0x4001 | `target.max_hp` | `state.agent_max_hp(target).unwrap_or(1.0)` |
| 0x4002 | `target.hp_pct` | `target.hp / target.max_hp` (derived) |
| 0x4003 | `target.shield_hp` | `state.agent_shield_hp(target).unwrap_or(0.0)` |

The compiler-side field-id resolver (`emit_scoring.rs`'s `scoring_field_id`) does not yet emit target-side ids — the scoring DSL grammar needs `target.<field>` parsing first.

### 8.3 Pair fields (PairField predicates)

Reserved range `field_id >= 0x8000`. Not emitted at milestone 5 — pair predicates land with the verb / mask expansion.

`field_id == u16::MAX` is reserved for the "invalid field" sentinel. The scorer returns `f32::NAN` for it.

---

## 9. Compiler architecture

### 9.1 Emission modes

See §1.1 for the four artefact classes. The compiler does not choose which engine backend to use; it emits all four artefact classes unconditionally.

### 9.2 Scalar Rust emission details

- SoA buffers per entity kind, with `@hot` / `@cold` field partitioning (§9.4).
- Per-agent kernels as `fn` with `#[inline(never)]` in profiling builds.
- `SimScratch` pools carry all per-tick scratch — zero steady-state allocation.
- **Spatial index is 2D-grid + per-column sorted z-list + movement-mode sidecar** (§9 #25). Primary structure keys `(cx, cy) → SortedVec<(z, AgentId)>` with 16m cells.
- RNG: a single `rng_state: u64` per world; per-agent streams seeded from `hash(world_seed, agent_id, tick, purpose)`.

### 9.3 GPU dispatch + SPIR-V kernel emission

GPU emission covers the deterministic sim's rules layer — mask predicates, cascade handlers, event-folded views, spatial-hash queries. ML forward passes are NOT compiled here; ML is out of DSL scope.

GPU-amenable kernels:
- Mask evaluation for intrinsic scalar predicates, including `distance` / `planar_distance` / `z_separation`.
- Cascade handlers that touch GPU-resident `AggregatePool<T>` — per-event match + emit loops with fixed-size iteration bounds.
- Event-fold materialization for commutative scalar views (sort events by target before reduction).
- 3D spatial hash (voxel-chunk-keyed) for `query::nearby_agents`.

Always host-side (regardless of engine backend):
- Chronicle prose rendering.
- Metric sink dispatch.
- Save/load serialization.
- Trace-format emission.

**GPU determinism constraints:**
- Reductions feeding scoring decisions use integer fixed-point or sorted-key accumulation.
- Materialized views sort events by `target_id` before atomic accumulation.
- Reduction shader workgroup size is pinned via specialization constants.
- Utility-backend tiebreak RNG seeds from `hash(world_seed, agent_id, tick, "scoring")`.

### 9.4 Hot/cold storage split

Mandatory at 200K scale. Authors annotate Agent fields with `@hot` or `@cold`:

```
entity Agent {
  // Hot — resident, read every tick by masks, scoring, and traced views
  @hot pos:              vec3,
  @hot hp:               f32,
  @hot max_hp:           f32,
  @hot shield_hp:        f32,
  @hot needs:            [f32; 6],
  @hot emotions:         [f32; 6],
  @hot personality:      [f32; 5],
  @hot memberships:      SortedVec<Membership, 8>,

  // Cold — paged, loaded on scoring-tick for High fidelity only
  @cold memory_events:   RingBuffer<MemoryEvent, 64>,
  @cold behavior_profile: SortedVec<(TagId, f32), 16>,
  @cold class_definitions: [ClassSlot; 4],
  @cold creditor_ledger: [Creditor; 16],
  @cold mentor_lineage:  [AgentId; 8],
}
```

Fidelity gating: `@fidelity(>= Medium)` on a view or cascade skips evaluation for Background-fidelity agents. Target: hot ≤ 4 KB/agent, 200K × 4 KB = 800 MB; cold paged to SSD with LRU.

### 9.5 Lowering passes

Anticipated passes (currently sketched only):

- **Verb desugaring.** `verb` declarations lower into mask predicate, cascade rule, and scoring-table entries.
- **Read → Ask lowering.** `Read(doc)` is sugar for `Ask(doc, QueryKind::AboutAll)`. The compiler rewrites every `Read(x)` into the document-target branch of `Ask`; the runtime MicroKind enum carries `Ask` only.
- **View storage-hint selection.** `@materialized(on_event=[...], storage=<hint>)` authors pick `pair_map` / `per_entity_topk(K, keyed_on=<arg>)` / `lazy_cached`. Compiler rejects infeasible combinations. GPU/CPU routing follows from storage hint.
- **Cascade dispatch codegen.** `physics` rules lower to phase-tagged handlers with compile-time cycle detection, race detection, and schema-drift guards.
- **Python emission.** Every `entity`, `event`, and `@traced` view gets a matching `@dataclass`. The Dataset class wraps the trace ring-buffer format.

### 9.6 Compiler decisions

16. **Mod event-handler conflict resolution** — **C (named lanes)**: handlers declare a lane `on_event(EventKind) in lane(Validation | Effect | Reaction | Audit)`. Lanes run in order; within a lane, handlers run in lexicographic mod-id. Multiple handlers per lane coexist (additive). Destructive overrides happen via forking the DSL source, not via a replace keyword.

24. **Utility backend is the production NPC backend.** Permanent. ML training is external; the compiler emits Python dataclasses + a pytorch `Dataset` for trace-format consumption. `scoring` declarations drive utility-backend scoring AND are written to traces so external pytorch scripts can reshape them into rewards.

---

## 10. Errors and diagnostics

The compiler raises hard errors on:
- Event-type non-existence on `on` patterns.
- Pattern-field mismatch against event's declared fields.
- Cycle detection in the event-type emission DAG without `@terminating_in`.
- Schema-hash drift on event or field references.
- Unresolved views, events, or fields.
- `String` fields on `@replayable` events (not yet enforced — see §5.2 audit callout).
- Unbounded `Vec<T>` on entity/event/struct declarations (not yet enforced).

Trace-format mismatch on load:

```
error: trace format mismatch
  trace: traces/run_2026-04-10.bin
  trace schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended entity field: Agent.war_exhaustion (offset 240, size 4, f32)
    + appended action variant: macro_kind::InviteToGroup (slot 4)
    + appended event: InvitePosted
  action: run the older engine version that matches the trace schema, or
          re-emit traces from the current DSL.
```

---

## 11. What's NOT in scope

This DSL does not handle:

- **Machine learning**: policy architecture, training algorithms, curriculum, reward shaping, observation packing. Compiler emits Python dataclasses + a pytorch `Dataset` for trace-format consumption; training code lives in external pytorch scripts.
- Text generation, LLM prose, dialogue authoring — `@non_replayable` side channel.
- Asset pipeline — mesh, texture, shader assets live elsewhere.
- Rendering — voxel meshing, marching-cubes, SDF rendering are voxel-engine concerns.
- Networking / multiplayer synchronization.
- Audio, particle effects, UI — display layers outside the sim loop.
- Build-system integration beyond `shaderc` SPIR-V compilation.
- Online learning / federated training.
- Human-in-the-loop labelling.

---

## 12. Worked example

Three agents A, B, C in a settlement. A proposes marriage to B; cascade forms a Family group; B's membership updates; C witnesses and gains a positive memory.

**State before:**

```
Agent A { id=1, pos=vec3(10, 5, 42.1), creature_type=Human, can_marry=true,
          married=false, personality.social_drive=0.7 }
Agent B { id=2, pos=vec3(11, 5, 42.1), creature_type=Human, can_marry=true,
          married=false }
Agent C { id=3, pos=vec3(12, 6, 42.0) }
view relationship(1, 2) = { valence=0.72, familiarity=0.60 }
```

**Tick T — A acts.** The `InviteToGroup{kind=Family, target=B}` candidate wins the utility argmax. A emits:

```
action { macro_kind = InviteToGroup, group_kind = Family, target = known_actors[0] }
```

**Mask evaluation (A):** `A.can_marry ∧ ¬married(A) ∧ ¬married(B) ∧ relationship(A, 2).valence > MIN_TRUST (0.72 > 0.5) ∧ ¬A.in_active_invite_to(2, Family)` → mask passes.

**Cascade fires** (`@phase(event)`): emits `InvitePosted { invite_id: 42, inviter: 1, target: 2, kind: Family, expires_tick: T+500 }`.

**Tick T+1 — B accepts.** `AcceptInvite(42)` cascade fires, emitting `MarriageFormed { a: 1, b: 2 }`, which triggers `FoundGroup { kind: Family, founder_ids: [1, 2] }` and `RecordMemory` for both spouses.

**`witness_cascade` (@phase(post)):** C is nearby (within `WITNESS_RADIUS`, same altitude), receives `Witnessed { observer: 3, subject: 1, kind: MadeNewFriend, source: Witnessed, confidence: 1.0 }`.

**State after (tick T+2, after A announces to settlement):**

```
Agent A { memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent B { memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent C { memory_events.push(Witnessed{MadeNewFriend, [A, B], 0.15}) }
Group  99 { kind=Family, founder_ids=[1, 2] }
```

The entire cascade is 4 events, one mask per action, two scoring-driven action selections, and one post-phase spatial query. No special "marriage system" exists; the outcome falls out of cascade rules, views, and scoring.

---

## 13. Settled decisions

### 13.1 Action / quest mechanics

1. **Auction state machine** — `Resolution` enum = `{HighestBid, FirstAcceptable, MutualAgreement, Coalition{min_parties: u8}, Majority}`. `PostAuction` is an alias of `PostQuest{kind: Diplomacy|Charter|Service}`.
2. **Macro head firing rate** — Macro head runs **every tick**. Most emissions are `NoOp`; scoring evaluates both heads each tick.
3. **Macro credit assignment** — Out of DSL scope. Lives in external pytorch training scripts.
4. **Quest discovery push/pull hybrid** — Posting emits `Announce(fact_ref=quest_id, audience)` cascade → recipients get the quest into their `known_quests` slot. Physical proximity produces pull-style discovery for quests outside the announce radius.
5. **Slot K tuning** — K=12 across the board for spatial slots; leaders get larger `known_actors`, `known_groups`. Option-A ("agent must actively seek information") — no auto-populated global views.
6. **Cross-entity mask index design** — Eager index materialization for `standing(group_a, group_b)`, `quest.eligible_acceptors`, `same_building`; lazy scan for low-cardinality predicates.
7. **Concurrent quest membership** — Multi-quest list (up to K=4 active).
8. **Reward delivery on long quests** — Compute at completion from current state.
9. **Cancellation / amendment** — `WithdrawQuest` macro head for taker only.
10. **Bid currency parity** — `Payment::Combination{material, immaterial}` with per-agent valuation.
15. **Nested quest cancellation** — Parent Quest has `child_quest_ids: SortedVec<QuestId, 8>`. Parent cancel emits a cascade that cancels children.
17. **Polygamous / cross-species / multi-parent family** — `Agent.spouse_ids: SortedVec<AgentId, 4>`, `can_marry(a, b)` view, `ChildBorn{parents: [AgentId; 4]}`.
18. **Mercenary / service payment direction** — `AuctionKind::Service` inverts roles via `AuctionParty::{Buyer=Patron, Seller=Labourer}`.
19. **Alliance obligation enforcement** — "alliance" is not a first-class concept; it is an emergent standing derived from `standing(group_a, group_b)`. `SetStanding(target, kind)` is the universal macro.
20. **Group-level invites vs agent-level invites** — Agent-level only. Group mergers use `PostQuest{kind: Diplomacy, resolution: Coalition{min_parties: K}}`.
30. **Communication channels (D30)** — `Capabilities.channels: SortedVec<CommunicationChannel, 4>` replaces `can_speak` / `can_hear` booleans. `channel: CommunicationChannel` is a parameter head on relevant macro and micro primitives.
31. **Materialized-view storage hint (D31)** — `@materialized(on_event=[...], storage=<hint>)` picks `pair_map`, `per_entity_topk(K, keyed_on=<arg>)`, or `lazy_cached`. GPU/CPU routing follows from storage: intrinsic scalars + per-entity-slot materializations compile to GPU; lazy + unbounded-pair predicates stay CPU.

### 13.2 Runtime / infrastructure

11. *(retired — LlmBackend is no longer a DSL concept; ML is out of DSL scope.)*
12. **Per-agent RNG streams** — Per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose)`. Enables fully-parallel sampling without stored per-agent RNG state.
13. **Materialized-view restoration on load** — Views serialize with schema-hash guard; on mismatch, rebuild from event log if available, otherwise refuse load.
14. **Event log storage compression** — Event-type filtering to replayable subset, fixed snapshot cadence N=500 ticks, zstd compression. ~2–5 GB per ~1000-tick bug report window.
21. **Chronicle prose side-channel lifecycle** — Eager template rendering at event emission; async LLM rewrite for flagged categories; saved prose is canonical; replay artefacts bundle the template library.
22. **Probe default episode count** — Config-definable per world + per-probe override via `seeds [42, 43, ...]` syntax.
23. **Training is out of DSL scope** — DSL emits Python dataclasses + a pytorch `Dataset`. All training-script concerns live in external pytorch code.
24. **Utility backend is the production NPC backend** — permanent. See §9.6.
25. **3D spatial hash structure** — 2D grid (cell=16m, voxel-chunk edge) with per-column sorted z-list, plus a `movement_mode ≠ Walk` sidecar. `movement_mode` is a primary field updated by the cascade.
26. **Overhear confidence decay** — Category-based base + exponential distance decay. `base[SameFloor/DiffFloor/Outdoor] = {0.75, 0.55, 0.50}`; `confidence = base * exp(-planar_distance / OVERHEAR_RANGE)`.
28. **`believed_knowledge` decay rate** — 3-tier volatility model: `KnowledgeDomain` carries `volatility: {Short=500, Medium=20_000, Long=1_000_000}` ticks half-life.

### 13.3 Schema / memory

27. **Document trust_score authoring** — `trust_score` field removed. Reader computes confidence from `(relationship_to_author, seal_validity, known_author_biases)`.
29. **FactRef ownership after memory eviction** — Materialize at creation. Quests and Documents store `FactPayload` inline.

---

## Appendix A. Universal Action Mechanisms (detailed reference)

The ~110 distinct gameplay action verbs (DeclareWar, ProposeAlliance, Vassalize, ...) collapse to **four universal mechanisms** plus micro primitives.

### A.1 `PostQuest(type, party_scope, target, reward, terms, deadline)`

"I (or a group I lead) want this thing done. Here's payment."

| Scenario | type | party_scope | target | reward |
|---|---|---|---|---|
| War on rival faction | `Conquest` | `Group(my_faction)` | `Group(rival_faction)` | `Spoils(territory)` |
| Form mutual-defense alliance | `MutualDefense` | `Group(my_faction)` | `Group(other_faction)` | `Reciprocal` |
| Submit weak group as vassal | `Submit` | `Group(weak_faction)` | `Group(my_faction)` | `Protection` |
| Found a new settlement | `Found` | `Group(my_faction)` | `Location(loc)` | `Charter` |
| Propose marriage | `Marriage` | `Individual(self)` | `Agent(other)` | `Union` |
| Post bounty on outlaw | `Assassinate` | `Anyone` | `Agent(outlaw)` | `Gold(amount)` |
| Hire mercenary | `Service` | `Individual(self)` | `Anyone` | `Gold(stipend)` |
| Rescue captured agent | `Rescue` | `Anyone` | `Agent(captive)` | `Gold + reputation` |
| Escort agent to destination | `Escort` | `Anyone` | `Pair(agent, location)` | `Gold` |
| Hunt creatures of a kind | `Hunt` | `Anyone` | `CreatureType(k)` | `Gold per kill` |

### A.2 `AcceptQuest(quest_id)` / `JoinParty(party_id)`

"I'll do that." Universal acceptance.

| Scenario | Action |
|---|---|
| Accept a marriage proposal | `AcceptQuest{type=Marriage}` by the targeted agent |
| Join the war on one's group's side | `AcceptQuest{type=Conquest/Defend}` by group members |
| Swear vassalage | `AcceptQuest{type=Submit}` by the weaker group |
| Take a mercenary contract | `AcceptQuest{type=Service}` |

`JoinParty` also handles defection: leaving party A and joining party B.

### A.3 `Bid(auction_id, payment, conditions)`

"I want that resource at that price."

| Scenario | Action |
|---|---|
| Offer a private trade | private 2-party `Bid` in an ad-hoc auction |
| Buy from the local market | `Bid` in the standing commodity-market auction |
| Hire a worker / mercenary | `Bid` in a Service auction |
| Lobby for a settlement charter | `Bid` in a political auction with reputation as currency |
| Ransom a captive | `Bid` against captor for release |

### A.4 `Announce(audience, fact_ref)`

"I'm telling a group of people a thing I know." One emission reaches many recipients instead of one.

| Scenario | Action |
|---|---|
| Faction leader announces war | `Announce{audience=Group(faction), fact_ref=WarDeclared}` |
| Town crier shouts news | `Announce{audience=Area(town_center, 30), fact_ref=...}` |
| Prophet proclaims a vision | `Announce{audience=Anyone, fact_ref=ProphecyHeard}` |
| Captain issues orders to patrol | `Announce{audience=Group(patrol_party), fact_ref=Order}` |

Cascade: enumerate eligible recipients, emit `MemoryEvent` per recipient. Bounded by `MAX_ANNOUNCE_RECIPIENTS`.

### A.5 Micro primitives

These can't be reduced because they ARE the per-tick physical/social acts:

- **Movement**: `MoveToward(pos)`, `Flee(from)`, `Hold`
- **Combat**: `Attack(target)`, `Cast(ability_idx, target)`, `UseItem(slot, target)`
- **Resource**: `Harvest(node)`, `Eat(food_source)`, `Drink(water_source)`, `Rest(loc)`
- **Construction**: `PlaceTile(pos, type)`, `PlaceVoxel(pos, mat)`, `HarvestVoxel(pos)`
- **Social atomic**: `Converse(target)`, `ShareStory(audience, topic)`
- **Info atomic (push)**: `Communicate(recipient, fact_ref)` — point-to-point sharing of one memory event; cascade inserts a `MemoryEvent` into `recipient.memory` with `source = TalkedWith(self)`, confidence `min(self.memory[fact_ref].confidence, 0.8)`.
- **Info atomic (pull)**: `Ask(target, query)` — request facts from an entity. If target is an `Agent`, emits `InformationRequested`; the target's next-tick scoring may respond. If target is a `Document` item, cascade resolves immediately — extract `target.facts` matching `query`, insert into `self.memory` with `source = Testimony(target.id)`.
- **Info atomic (sugar)**: `Read(doc)` — language-surface shorthand for `Ask(doc, QueryKind::AboutAll)`. The compiler lowers `Read` to the document-target branch of `Ask`; runtime vocabulary does not carry a separate `Read` micro.
- **Memory atomic**: `Remember(entity, valence, kind)`

### A.6 Information system

Every `MemoryEvent` carries a `source` tag recording its provenance:

```rust
enum Source {
    Witnessed,              // confidence = 1.0
    TalkedWith(AgentId),    // confidence = 0.8
    Overheard(AgentId),     // confidence = 0.6
    Rumor { hops: u8 },     // confidence = 0.8^hops
    Announced(GroupId),     // confidence = 0.8
    Testimony(ItemId),      // confidence per §13.3 #27
}
```

Confidence propagation rules:
- `Communicate`: `conf_r = min(self.memory[fact_ref].confidence, 0.8)`.
- `Announce{Group | Area | Anyone}`: 0.8 for direct audience; 0.6 for overhear bystanders.
- `Read(doc)`: per `(relationship_to_author, seal_validity, known_author_biases)`.
- `Rumor{hops}`: each subsequent `Communicate` increments `hops`; confidence multiplies by 0.8.

Mask predicates gating information-sensitive actions:
- `knows_event(self, event_kind, target_pattern)` — memory contains a matching event.
- `knows_agent(self, agent_id)` — memory references `agent_id` OR `agent_id ∈ self.relationships`.
- `confident_about(self, fact, threshold)` — memory contains `fact` with `confidence ≥ threshold`.
- `recent(self, event_kind, max_age_ticks)` — memory contains matching event with `now − tick ≤ max_age_ticks`.

### A.7 Auction state machine

```rust
struct Auction {
    id:            u32,
    kind:          AuctionKind,
    item:          AuctionItem,
    seller:        AuctionParty,
    bids:          Vec<Bid>,
    open_tick:     u64,
    deadline_tick: u64,
    resolution:    AuctionResolution,
    visibility:    Visibility,
    reserve_price: Option<Payment>,
}

enum AuctionKind {
    Item, Commodity, Service, Charter, Diplomatic, Marriage, Ransom
}

enum AuctionResolution {
    HighestBid, FirstAcceptable, MutualAgreement, Allocation
}
```

Lifecycle events:

```
AuctionPosted    { auction_id, kind, seller, item, deadline, visibility }
BidPlaced        { auction_id, bidder, payment, conditions }
BidWithdrawn     { auction_id, bidder }
AuctionResolved  { auction_id, winner: Option<EntityId>, payment }
AuctionExpired   { auction_id }
```

Cascade rules:
- `AuctionResolved{kind=Item, winner=W, payment=P}` → `TransferGold(W → seller, P.amount)` + `TransferItem(seller → W, item)`
- `AuctionResolved{kind=Marriage, winner=W}` → `MarriageFormed(seller, W)`
- `AuctionResolved{kind=Charter, winner=W}` → `CharterGranted(settlement, W, terms)`

### A.8 Total action vocabulary

```
4 macro mechanisms:     PostQuest, AcceptQuest|JoinParty, Bid, Announce
19 micro (surface):     movement(3) + combat(3) + resource(4) + construction(3)
                      + social(2) + info(3: Communicate push, Ask pull, Read)
                      + memory(1)
18 micro (runtime):     same, minus Read — lowered to Ask(doc, QueryKind::AboutAll)
─────────────────────────
23 categorical actions at the language surface;
22 at the runtime (MicroKind enum) after Read → Ask lowering.
```

The richness comes from the parameter heads — `QuestType`, `PartyScope`, `QuestTarget`, `Reward`, `BidConditions`, `Payment`, `AnnounceAudience`, `FactRef` enums each have many variants.
