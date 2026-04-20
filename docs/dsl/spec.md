# World Sim DSL Specification

Canonical specification for the ECS DSL. Supersedes `spec.md`, `spec.md`, and `spec.md` (all folded into this doc). §9 lists the 29 settled design decisions; per-decision rationale is extracted into `decisions.md`.

Appendix A contains the detailed universal-mechanisms reference (PostQuest/AcceptQuest/Bid/Announce).

For compiler-layer concerns (codegen, lowering, schema emission), see `docs/compiler/spec.md`.

---

## 1. Language overview

The DSL declares a closed-world simulation: typed state, typed events, declarative derivations, cascade rules, action masks, scoring weights, invariants, and regression probes. Ten top-level declaration kinds compose into a single compiled artefact (Rust runtime + SPIR-V kernels + Python dataclass mirrors + trace-format schema).

Machine learning — policy architecture, training algorithm, curriculum, reward shaping, observation packing — is **not** in the DSL. The compiler emits Python dataclasses + a pytorch `Dataset` over the trace format so that external training scripts consume a typed API. The in-engine NPC backend is a utility backend (ships permanently as bootstrap + regression baseline); `scoring` declarations drive utility scoring and are also written into traces so Python training jobs can reshape them into rewards externally.

- **`entity`** — parameterization of one of the three predefined root kinds (Agent, Item, Group). Authors cannot introduce new root kinds; they declare `creature_type`, `ItemKind`, or `GroupKind` variants with default stats, capabilities, eligibility, and starting memberships. (`stories.md` §1, §10, §11)
- **`event`** — typed, append-only records. The universal state-mutation channel. Annotations mark replayability, high-volume classification, trace visibility. (`stories.md` §2; `stories.md` §15; `stories.md` §60)
- **`view`** — pure or event-folded derivations. Eager (`@materialized`) or lazy; first-class spatial and non-spatial queries. (`stories.md` §3, §5; `stories.md` §26)
- **`physics`** cascade rule — phase-tagged transforms from events to events with compile-time cycle detection, race detection, and schema-drift guards. (`stories.md` §4; `stories.md` §27)
- **`mask`** — per-action validity predicates. Emits both a Rust predicate fn and a SPIR-V kernel so the legality contract is identical on both backends. (`stories.md` §8)
- **`verb`** — composition sugar that bundles mask + cascade + scoring into a named gameplay action without extending the categorical action vocabulary. (`stories.md` §7)
- **`scoring`** — per-action personality-weighted utility table consumed by the utility backend. Also written to traces so external ML training scripts can reshape them into rewards.
- **`invariant`** — static, runtime, or debug-only predicates over state, checked at the phase boundary they describe. (`stories.md` §6)
- **`probe`** — named scenario + behavioral assertion, evaluated against seeded trajectories. CI regression surface.
- **`metric`** — runtime observability declaration: per-tick counters, histograms, gauges with optional alert thresholds. No coupling to training.

At runtime the tick pipeline is fixed: pre-phase rules → event emission → materialized-view updates → mask build → utility-backend scoring → action selection → action application → cascade fixed-point → post-phase rules → metric emission. Determinism flows through a single per-world RNG; text generation sits outside the deterministic fold. (`stories.md` §27, §29; `stories.md` §60)

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
  languages:      SortedVec<LanguageId, 4>,             // vocabulary within Speech/Testimony channels
  can_fly:        bool,
  can_build:      bool,
  can_trade:      bool,
  can_climb:      bool,
  can_tunnel:     bool,
  can_marry:      bool,
  max_spouses:    u8,                                   // §9 D17 polygamy cap
  // Ability-scaled attributes live on the Agent, not here.
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
  base_stats:     { ... },           // fixed-shape struct per ItemKind
  slots:          [EquipSlot, ...],
}

entity <Name> : Group {
  kind:                    GroupKind,
  eligibility_predicate:   <predicate>,    // compiles to EligibilityRef
  recruitment_open:        bool,
  governance:              GovernanceKind,  // Hereditary | Elective | Council | Theocratic | AlphaContest | AdHoc
  default_standings:       [ (GroupKind, Standing), ... ],
  required_fields:         [ FieldName, ... ],
}
```

Layout hints:

- `@materialized` — the compiler generates an update path (see §2.3).
- `@spatial` — the entity carries a `pos` field participating in spatial indices (KD-tree, uniform grid, voxel chunk grid).
- `@capped(K)` — on collection fields; bounds the fixed storage layout (§5.1 bitset / sorted-Vec / ring buffer / Map types).
- `@inline` — embed the child struct in the parent's SoA row rather than reference it (cache-line tuning).

The compiler emits:

- A Rust struct populated with defaulted fields per declaration.
- An enum-variant extension for the corresponding discriminator (`CreatureType`, `ItemKind`, or `GroupKind`).
- A spawn-template that initialises all required fields.
- A schema-hash contribution for the one-hot width of that discriminator (§4).

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
                             // (Python Dataset, probe evaluation, debug scrubbing)
@gpu_amenable                // scalar fields only; triggers GPU event-fold codegen
```

`MemoryEvent` (declared in `state.md`) carries a `source: Source` field alongside the usual payload:

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

`Source` is a standard type carried inside `MemoryEvent` payloads.

The compiler emits:

- A variant on the runtime event enum for that kind-bucket (`WorldEvent` / `MemoryEvent` / `StructuralEvent` — the bucket is chosen by the compiler from the event's payload references, not by the author).
- A fixed-capacity ring buffer, sized by `@high_volume` (larger) or the default.
- Serialisation hooks for replay (if `@replayable`).
- An entry in the trace-format event-vocabulary table (if `@traced`), plus an emitted Python `@dataclass` mirroring the event's fields.
- A pattern-match kernel for use in cascade, scoring, and probe blocks.

String payloads are permitted only on `@non_replayable` events; the compiler rejects `String` fields on replayable events (`stories.md` §60). All load-bearing references are `AgentId`, `GroupId`, `ItemId`, `QuestId`, or `AuctionId` — no text. This forces `class_tags: Vec<String>` and `archetype: String` on legacy agent fields to migrate to `TagId` / `ArchetypeId` (`stories.md` §60 cross-cutting).

### 2.3 `view` declaration

Views are pure over their inputs. The compiler chooses between lazy evaluation at read time and eager event-fold materialization per declaration.

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
@fidelity(>= Medium)           // skip evaluation at lower fidelity (§2.3 of this doc; stories.md §E.31)
```

The compiler emits:

- For `@materialized`: a field on the corresponding entity + an event-dispatch table mapping each `on_event` to the update body. GPU-amenable materializations sort events by target before commutative reduction to preserve determinism (`stories.md` §29 GPU-determinism traps).
- For `@lazy`: an inline function referenced from mask predicates, scoring expressions, and cascade bodies.
- For `@spatial`: routing to the appropriate spatial index — `voxel_engine::ai::spatial` on GPU, or a CPU uniform-grid fallback.
- For `@top_k`: a fixed-cap partial-sort that writes into a `SimScratch` buffer.

Queries are first-class and carry their output cap. `query::nearby_agents(self, r)` returns at most K records; `query::known_actors(self)` returns the union `{spouse, mentor, apprentice, group_leaders(self), top_grudges, top_friendships}` deduped (`spec.md` §2.1.5).

#### Fold body operator set

`@materialized` fold handler bodies are restricted to a closed operator set so the event-fold path compiles to commutative, GPU-friendly updates. This mirrors §2.5's mask-predicate restriction. The compiler rejects anything outside the list below with `ResolveError::UdfInViewFoldBody` pointing at the offending construct.

Allowed:

- Compound self-assignment: `self +=`, `self -=`, `self *=`, `self /=`.
- Plain self-assignment: `self = <closed expr>`.
- Conditional: `if <closed expr> { <closed stmts> } else { <closed stmts> }`.
- Arithmetic: `+`, `-`, `*`, `/`, `%` on primitives.
- Comparison: `<`, `<=`, `==`, `>=`, `>`, `!=`.
- Logical: `&&`, `||`, `!`.
- Bounded folds: `count`, `sum`, `min`, `max` over `SortedVec`/`SmallVec`.
- Built-in math: `abs`, `floor`, `ceil`, `pow`, `ln`, `sqrt`, `clamp`.
- Stdlib 1-hop accessors: `agents.hp(a)`, `distance(p1, p2)`, etc.
- `let x = <closed expr>` intermediates.

Forbidden:

- Recursion.
- Loops (`while`, unbounded `for`).
- User-declared helper functions called inside the fold body.
- Calls to other views — including views that reference other views (no cross-view composition in fold bodies).

#### `@decay(rate = R, per = tick)`

Sugar layered on `@materialized` views. Lowers to the anchor pattern: the stored value is `(base_at_anchor, anchor_tick)` and the observable at tick `t` is `base * rate^(t - anchor)`, clamped. Event handlers fold the event's delta into a new base and advance the anchor to the current tick.

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

Constraints (enforced by the resolver):

- Requires a sibling `@materialized(...)` annotation; a bare `@decay` on a `@lazy` or plain view is a hard error.
- The host view body must be a `Fold` (the anchor pattern needs a base value + event handlers).
- `rate` is a compile-time float literal in the open interval `(0.0, 1.0)`. Variable decay rates are not supported in v1.
- `per` is the identifier `tick`. Other time bases are reserved for later milestones.

### 2.4 `physics` cascade rule

Cascade rules respond to events and emit further events. They never mutate state directly — all writes go through `emit`. Phase-tagged ordering prevents races.

```
physics <name> @phase(pre | event | post) {
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

physics auction_item_resolved @phase(event) {
  on AuctionResolved{kind: Item, winner: W, payment: P, auction_id: id} {
    let a = auction(id);
    emit TransferGold { from: W, to: a.seller, amount: P.gold_amount() }
    emit TransferItem { from: a.seller, to: W, item: a.item.item_id() }
  }
}
```

Annotations:

- `@phase(pre | event | post)` — fixed three-phase ordering (`stories.md` §27). `pre` runs before event emission (read-only state access). `event` runs during the fixed-point event fold. `post` runs after all events quiesce for the tick.
- `@before(OtherRule)` / `@after(OtherRule)` — explicit ordering between rules in the same phase.
- `@terminating_in(N)` — asserts the cascade converges within N hops when self-emission is possible (`stories.md` §4).

Compile-time validation:

1. Event-type existence on every `on` pattern.
2. Pattern-field mismatch against the event's declared fields.
3. Field-write race: two unordered rules in the same phase both emitting events that materialize into the same entity field fail unless one uses a commutative update (`+=`, `min=`, `max=`).
4. Cycle detection in the event-type emission DAG; self-loops require `@terminating_in`.
5. Schema-hash drift on event or field references.
6. All referenced views, events, and fields resolve.

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

mask InviteToGroup{kind: Family, target: Agent(t)}
  when self.creature_type.can_marry
     ∧ ¬married(self)
     ∧ ¬married(t)
     ∧ relationship(self, t).valence > MIN_TRUST
     ∧ ¬self.in_active_invite_to(t, Family)
```

Supported operators: set membership (`contains`, `in`), quantifiers (`forall`, `exists`), bounded folds (`count`, `sum`, `max`, `min`), arithmetic comparison, and view calls. No user-defined functions inside masks (they cannot be safely compiled to GPU boolean kernels).

Compilation:

- Per-action validity buffers `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`, etc., consumed by the utility backend when filtering scoring candidates.
- Mask predicates that reference only intrinsic scalar fields compile to SPIR-V compute shaders; cross-entity predicates (`t in quest.eligible_acceptors`, `at_war(self.faction, f)`) are CPU-patched into the same boolean buffer before scoring evaluation.
- Every predicate node has a stable AST ID; an explanation kernel reruns the predicate against a captured state snapshot for `trace_mask(agent, action, tick)` (`stories.md` §34).

### 2.6 `verb` (composition sugar)

`verb` declares a named gameplay action that composes an existing micro primitive with additional mask predicates, cascades, and scoring entries. It does NOT add to the closed categorical action vocabulary. (`stories.md` §7)

```
verb Pray(self, shrine: Structure) =
  action Converse(target: shrine.patron_agent_id)
  when  self.memberships contains Group{kind: Religion}
      ∧ distance(self, shrine) < REACH
  emit  PrayCompleted { prayer: self, shrine: shrine, faith_delta: 1.0 }
  score 0.5 * self.personality.piety
```

The compiler expands a `verb` into:

1. A mask entry narrowing an existing `Converse` / `Attack` / other micro primitive.
2. A cascade handler that emits the declared event on successful action application.
3. A scoring entry appended to the scoring table (§3.4).

Adding a `verb` does not bump the schema hash. Adding a new micro primitive does (see §4).

### 2.7 `policy` declaration — REMOVED: ML out of DSL scope, see compiler trace-emission

### 2.8 `invariant` declaration

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

invariant party_member_agents_alive(q: Quest) @debug_only {
  forall m in q.party_member_ids: entity(m).alive
}
```

Cascades that write a field in the invariant's support must annotate `@must_preserve(<invariant>)`; the compiler rejects cascades that do not declare it.

Schema invariants are separate from state invariants: the append-only-trace-schema invariant is checked by the compiler at CI time, not at runtime.

### 2.9 `probe` declaration

Named CI regression assertions: fixture scenario → seeded event trajectory → behavioural check against the utility backend. Probes live in `probes/` alongside their seed scenarios. (`stories.md` §18)

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

Probes always run against the utility backend (the production NPC backend). They are behavioural regression assertions, not ML evaluation harnesses; external ML evaluation consumes the trace format separately.

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

probe NoSpouseAttacks {
  scenario "probes/random_family_5agents.toml"
  assert {
    pr[ action.micro_kind == Attack | action.target.is_spouse == 1 ] == 0.0
  }
}
```

Probes compile to trajectory queries over the trace ring-buffer format. Schema-hash mismatch between a probe's reference fields and the current DSL is a hard error (fail-loud, §4).

### 2.10 `curriculum` declaration — REMOVED: ML out of DSL scope, see compiler trace-emission

### 2.11 `metric` declaration

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

  metric events_emitted_per_kind = histogram(event.kind)
    window 1000

  metric alive_agents = gauge(count(agent where agent.alive))
    emit_every 10 alert when value < 1

  metric masks_gate_rejections = counter(mask.rejection)
    window 1000 alert when value > 10_000
}
```

Alerts emit structured log records to the engine telemetry sink. Metrics are sim-observability only — no coupling to training.

---

## 3. Action vocabulary

### 3.1 Observation declaration — REMOVED: ML out of DSL scope, see compiler trace-emission

### 3.2 Action heads

Closed action vocabulary. Adding to this vocabulary bumps the schema hash; adding a `verb` does not.

```
action {
  head categorical macro_kind: enum {
    NoOp, PostQuest, AcceptQuest, Bid, Announce, InviteToGroup, AcceptInvite,
    WithdrawQuest,                  // taker-only: retract acceptance (§9 #9)
    SetStanding                     // unify DeclareWar / MarkHostile / MarkFriendly (§9 #19)
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

  head categorical channel: enum CommunicationChannel
                                 // §9 D30 — required for Communicate / Ask / Converse /
                                 // ShareStory micro primitives and for PostQuest / Announce /
                                 // InviteToGroup macro emissions. Ignored for non-communicating
                                 // actions.

  head pointer target: select_from
    nearby_actors ∪ nearby_resources ∪ nearby_structures
    ∪ known_actors ∪ known_groups ∪ active_quests
    ∪ active_auctions ∪ incoming_invites ∪ memberships
    ∪ recent_memory_events

  head pointer fact_ref: select_from recent_memory_events
                         // FactRef = local memory event_id; used by
                         // Communicate and Announce

  head continuous pos_delta: vec3 ∈ [-1, 1]³
  head continuous magnitude: f32  ∈ [0, 1]

  // Structured parameter heads for macros
  head categorical quest_type:        enum QuestType
  head categorical party_scope:       enum PartyScope
  head categorical quest_target:      enum QuestTarget
  head categorical reward_kind:       enum RewardKind
  head categorical payment_kind:      enum PaymentKind
  head categorical group_kind:        enum GroupKind            // for InviteToGroup
  head categorical announce_audience: enum AnnounceAudience     // Group / Area / Anyone
  head categorical standing_kind:     enum StandingKind         // for SetStanding (Hostile | Neutral | Friendly | Vassal | Suzerain)
  head categorical resolution:        enum Resolution           // auction resolution
                                                                // HighestBid | FirstAcceptable | MutualAgreement
                                                                // | Coalition{min_parties: u8} | Majority (§9 #1)
}
```

`macro_kind` and `micro_kind` are decoupled. When the model emits `macro_kind != NoOp`, the macro parameter heads drive the structured action and `micro_kind` is ignored. When `macro_kind == NoOp`, the `micro_kind` + its heads drive a per-tick physical/social primitive.

### 3.3 Action vocabulary (consolidated)

Four macro mechanisms + thirteen micro primitives.

| Head           | Variants                                                                                                                  |
|----------------|---------------------------------------------------------------------------------------------------------------------------|
| `macro_kind`   | `NoOp`, `PostQuest{...}`, `AcceptQuest{quest_id, role_in_party}`, `WithdrawQuest{quest_id}` (taker retracts), `Bid{auction_id, payment, conditions}`, `Announce{audience, fact_ref}`, `InviteToGroup{kind, target, terms}`, `AcceptInvite{invite_id}`, `SetStanding{target_group, kind}` (Hostile/Friendly/Vassal/Neutral/Suzerain — universal backbone for war/peace/alliance, §9 #19) |
| `micro_kind`   | `Hold`, `MoveToward(pos)`, `Flee(from)`, `Attack(t)`, `Cast(ability, target)`, `UseItem(slot, target)`, `Harvest(node)`, `Eat(food)`, `Drink(water)`, `Rest(loc)`, `PlaceTile(pos, type)`, `PlaceVoxel(pos, mat)`, `HarvestVoxel(pos)`, `Converse(t)`, `ShareStory(audience, topic)`, `Communicate(recipient, fact_ref)`, `Ask(target, query)`, `Read(doc)` *(sugar for `Ask(doc, QueryKind::AboutAll)`)*, `Remember(entity, valence, kind)` |

Enums carried as parameter heads:

- `QuestType` — Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Custom, Conquest, MutualDefense, Submit, Found, Charter, Diplomacy, Marriage, Pilgrimage, Service, Heist, Trade, FulfillProphecy, **Claim**, **Peace**, **Raid**, **HaveChild**. (Claim/Peace/Raid extensions from `stories.md` §46, §51; HaveChild from `stories.md` §52.)
- `PartyScope` — `Individual(AgentId)`, `Group(GroupId)`, `Settlement(SettlementId)`, `Anyone`, `Role(RoleTag)`.
- `QuestTarget` — `Agent`, `Group`, `Location`, `Structure`, `CreatureType`, `Pair(Target, Target)`, `Predicate`, `Item`, `Region`, **`Role(RoleTag)`** (for Claim quests).
- `RewardKind` — `Gold`, `Xp`, `Items`, `Reputation`, `Faith`, `Spoils`, `Charter`, `Union`, `Reciprocal`, `Protection`, `Glory`, `Promise`, `Combination`.
- `PaymentKind` — `Gold`, `Commodity`, `Item`, `Service`, `Reputation`, `Combination`.
- `GroupKind` — for `InviteToGroup`: Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Alliance, Coven, Other.

Invite and auction coverage (`stories.md` §63, `stories.md` §54):

| Invite / auction usage       | `macro_kind`        | Parameters                                                         |
|------------------------------|---------------------|--------------------------------------------------------------------|
| Marriage proposal            | `InviteToGroup`     | `kind=Family` (spouse role), `target=AgentId`                     |
| Alliance petition            | `PostQuest`+auction | `AuctionKind::Diplomatic`, `AuctionItem::AllianceTerms`            |
| Vassal petition              | `InviteToGroup`     | `kind=Vassal` (hierarchical standing)                              |
| Party recruitment (ad-hoc)   | `InviteToGroup`     | `kind=Party`                                                       |
| Apprenticeship               | `InviteToGroup`     | `kind=Family` (mentor/apprentice sub-role)                         |
| Job offer                    | `InviteToGroup`     | `kind=Guild`                                                       |
| Guild admission              | `InviteToGroup`     | `kind=Guild`                                                       |
| Peace treaty                 | `PostQuest`         | `type=Peace`, target=rival Group                                   |
| Mercenary hiring             | `PostQuest`+auction | `AuctionKind::Service`                                             |
| Market commodity auction     | emergent            | agents emit `Bid` on posted auctions                               |
| Charter petition             | `PostQuest`+auction | `AuctionKind::Charter`, `Payment::Reputation` permitted            |
| Coup                         | `PostQuest`         | `type=Conquest`, party=cabal of loyalists, target=own Group        |

Gameplay verbs like Pray, DefendDen, Raid-by-opportunity are `verb` declarations over these primitives.

### 3.4 `scoring`

Per-action personality-weighted utility values consumed by the utility backend. A `scoring` table maps each categorical action (micro or macro head variant) to a scalar utility expression; the utility backend argmaxes over masked candidates each tick. Scoring values are also written to traces so external pytorch training scripts can reshape them into rewards without reading engine internals.

```
scoring {
  // Micro utilities — expressions reference self fields, views, and per-tick deltas.
  Attack(t)       = 0.6 * self.personality.aggression
                    + 0.2 * (1.0 - self.hp_frac)
                    + 0.2 * is_hostile(self, t)
  Flee(from)      = 0.8 * (1.0 - self.hp_frac)
                    + 0.2 * threat_level(from)
  Eat             = 0.7 * hunger_pressure(self)
  Converse(t)     = 0.3 * self.personality.social
                    + 0.2 * relationship(self, t).valence

  // Macro utilities — personality-weighted so utility backend can fire rare actions.
  PostQuest{type=Conquest, target=Group(g)}
                  = 0.9 * self.personality.ambition
                    * (strength(self.leader_groups[0]) / strength(g))
  PostQuest{type=Marriage, target=Agent(t)}
                  = 0.7 * self.personality.compassion
                    * relationship(self, t).valence
  InviteToGroup{kind=Family, target=Agent(t)}
                  = PostQuest{type=Marriage, target=Agent(t)}
  AcceptInvite(i) = 0.5 + 0.5 * invite(i).source_standing
}
```

Compiler emits:

- A per-tick scoring kernel that evaluates each masked candidate's utility expression and writes the result into `utility_field[N × NUM_ACTIONS]` for the utility backend to argmax over.
- Trace-emission hooks: `(tick, agent_id, action, score)` rows streamed into the trace ring buffer so external Python training can consume per-action scores as reward-shaping inputs.
- Validation: scoring expressions must reference declared views, entity fields, or personality traits.

#### Gradient scoring modifiers

In addition to the expression-sum grammar above, a scoring entry's right-hand side may include **gradient modifier** terms of the form `<expr> per_unit <delta>`. Semantically, this contributes `expr * delta` to the final score; the compiler lifts each such term into a dedicated `KIND_GRADIENT` modifier row so the engine-side decoder can evaluate the expression once per tick and multiply by the baked-in delta.

```
scoring {
  Attack(t) = 0.5                                         // base
            + (if self.hp_pct < 0.3 { 0.4 } else { 0.0 }) // boolean modifier
            + threat_level(self, t) per_unit 0.02          // gradient modifier
            + (1.0 - t.hp_pct) per_unit 0.6                // arithmetic-expr gradient
}
```

Constraints:

- `<delta>` must be a float literal at v1. Variable deltas are a later milestone.
- `<expr>` must evaluate to `f32`. Type-checking is deferred to milestone 1b.
- `per_unit` binds between `+`/`-` and `*`/`/`, so `foo * k per_unit d` parses as `(foo * k) per_unit d` — the gradient expression can include a scalar multiplier. `a per_unit b per_unit c` is a grammar error.

Adding a `scoring` entry does not bump the schema hash's state / event / rules components — only the `scoring_hash` sub-hash (§4). External training code that depends on a specific scoring shape is responsible for recomputing from the current DSL.

### 3.5 Backend — REMOVED: ML out of DSL scope, see compiler trace-emission. The NPC backend is the utility backend (permanent); see `docs/compiler/spec.md` §Decisions #24.

---

## 4. Schema versioning

The schema hash is a content-addressed fingerprint over:

1. **`state_hash`** — entity field layouts (field order, offsets, types, bounded-collection capacities).
2. **`event_hash`** — declared event names and field shapes.
3. **`rules_hash`** — physics cascades, mask predicates, and verb declarations.
4. **`scoring_hash`** — scoring-table expressions and their input dependencies.

The combined schema hash is `sha256(state_hash || event_hash || rules_hash || scoring_hash)`. Loading a trace whose combined hash differs from the current DSL is a hard error (`stories.md` §15, `stories.md` §23, `stories.md` §64). Traces emitted under a different schema must be consumed by an engine version that matches, or re-emitted from the current DSL.

There are no `@since` annotations. There are no padded-zero migration tables. There are no v1/v2/v3 schemas in the codebase — git holds history. Two branches with different schemas produce mutually incompatible traces, which is correct.

For the emission mechanism (sub-hash layout, error-message format, CI guard) see `docs/compiler/spec.md` §2 Schema emission.

This rewrites `spec.md` §4.

---

## 5. Type system

### 5.1 Primitive and structural types

```
Scalar:    f32, i32, u32, u64, bool
Vector:    vec2, vec3, vec4
Time:      Tick (u64)
ID:        AgentId, ItemId, GroupId, QuestId, AuctionId, InviteId, StructureRef,
           EventId, PredicateId, TagId, ArchetypeId, RoleTag
FactRef:   { owner: AgentId, event_id: u32 }  // ephemeral handle into live memory
FactPayload: { tick, kind, params: [u32; 4], author_id: AgentId }
                                                 // durable materialisation used by
                                                 // Quest.fact_refs and Document.facts (§9 #29)
Bounded:   Bitset<N>,              // fixed-size bit vector
           SortedVec<T, K>,        // sorted, cap K, partial-sort eviction
           RingBuffer<T, K>,       // fixed ring, oldest-evict
           Map<K, V, Cap>,         // fixed-capacity ordered map
           OneOf<K, V>             // sum-type union with fixed discriminant
Struct:    Membership, Relationship, MemoryEvent, AgentData, Group, Item,
           Quest, Auction, Invite, Document, ChronicleEntry (structural-only)
Enum:      Source (Witnessed / TalkedWith / Overheard / Rumor / Announced /
                   Testimony),
           AnnounceAudience (Group / Area / Anyone),
           ...plus §3.3 parameter heads + CreatureType, ItemKind (incl.
           Document), GroupKind, AuctionKind, AuctionResolution,
           GovernanceKind, RoleTag, ChronicleCategory, DropReason, ...
```

`@spatial` types expose `pos: vec3` and participate in 3D spatial indices. Every bounded collection declares its capacity; capacity is part of the schema hash.

### 5.2 Forbidden types

`String` is forbidden on `@replayable` events, on `@primary` state fields, and on any field referenced by a mask or reward predicate. Strings are permitted only on `@non_replayable` events, the `chronicle_prose` side channel, and at display time. All load-bearing references are IDs (`stories.md` §60). `class_tags` and `archetype` agent fields carry `TagId` / `ArchetypeId` with a compile-time string table.

Unbounded `Vec<T>` is forbidden on `entity`, `event`, and in-world struct declarations. Only `SimScratch` pools and world-level ring buffers may hold `Vec`. The compiler rejects:

```
entity BadAgent : Agent { history: Vec<Event> }    // ERROR: unbounded collection
```

### 5.3 Structural type relationships

- `Agent.pos: vec3` — world-space 3D position. Ground-locked creature_types have `pos.z` snapped by the post-phase cascade in §7.
- `Agent.movement_mode: enum { Walk, Climb, Fly, Swim, Fall }` — cascade-updated; drives spatial-index sidecar (§9 #25). Slopes are `Walk`.
- `Agent.spouse_ids: SortedVec<AgentId, 4>` — polygamous-capable; cross-species gated by `can_marry(a, b)` view (§9 #17).
- `Relationship.believed_knowledge: Bitset<32>` — theory-of-mind projection of `knowledge_domain_bits`.
- `Relationship.believed_knowledge_refreshed: [u32; 32]` — per-bit last-reinforced tick; decay computed at read time using per-bit volatility (§9 #28).
- `KnowledgeDomain.volatility: enum { Short=500, Medium=20_000, Long=1_000_000 }` — compile-time per-variant tag; sets half-life in ticks.
- `Agent.memberships: SortedVec<Membership, 8>` — capped at K=8 simultaneous groups.
- `Agent.known_actors: SortedVec<Relationship, 32>` — LRU by `last_known_age`; pinned (spouse, mentor) entries survive eviction.
- `Agent.memory_events: RingBuffer<MemoryEvent, 64>` — with up to 5 "indelible" slots reserved for `emotional_impact > 0.8`. `MemoryEvent` carries `source: Source` and `confidence: f32`.
- `Agent.behavior_profile: SortedVec<(TagId, f32), 16>` — capped; lowest-weight eviction.
- `Document: Item{ kind: Document, author_id: AgentId, tick_written: Tick, seal: Option<SealId>, facts: SortedVec<FactPayload, 16> }` — no stored `trust_score`; reader derives confidence on `Read(doc)` from `(relationship_to_author, seal_validity, known_author_biases)` (§9 #27). Forgery is an emergent skill-gated quest.
- `Quest.fact_refs: SortedVec<FactPayload, 8>` — materialised at `QuestPosted` from the poster's memory (§9 #29).
- `Quest.child_quest_ids: SortedVec<QuestId, 8>` — sub-quest tracking; populated by `QuestPosted{parent_id: Some(_)}` cascade; drives recursive cancellation and spoils distribution (§9 #15).
- `Group.members` — derived view over agent `memberships`, not stored.
- `Group.standings: SortedVec<(GroupId, StandingKind, f32), 16>` — updated by `SetStanding` macro; "alliance" / "war" are emergent thresholds over `standing_value` (§9 #19).
- `Quest.party_member_ids` — resolved at read time from `party_scope` against current group membership; no stored list.
- `Quest.reward_evaluation: enum { CommitAtPosting, ComputeAtCompletion }` — every reward variant evaluates at completion (§9 #8). `CommitAtPosting` reserved for rare designer-opt-in cases; not the default.
- `ChronicleEntry` — `{ tick, category: ChronicleCategory, entity_ids: [AgentId; 4], template_id: u32 }` with text via side-channel `chronicle_prose: Map<ChronicleEntryId, String>`.

---

## 6. Compilation targets → moved to `docs/compiler/spec.md` §1

Native Rust (CPU) + GPU (`voxel_engine::compute::GpuHarness`) + hot/cold storage split now live in `docs/compiler/spec.md` §1. Section numbering below is preserved; §7 onwards retains its original number.

---

## 7. Runtime semantics

### 7.1 Tick pipeline

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

Volumetric `creature_type`s (Dragon, Fish, Bat) have no matching rule; their `pos.z` persists as emitted.

**Announce propagation** — a cascade rule triggered by `Announce`:

```
physics announce_broadcast @phase(event) {
  on AnnounceAction{speaker: s, audience: aud, channel: ch, fact_ref: f} {
    let range = view::channel_range(ch, s);            // §9 D30 — per-channel reach
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
    // Overhear scan — bystanders not in the audience (§9 #26, §9 D30 channel-filtered)
    let overhear_range = view::channel_range(ch, s) * OVERHEAR_RANGE_FRACTION;
    for b in query::nearby_agents_3d(s.pos, overhear_range)
                 .filter(|b| b != s && !in_audience(b, aud)
                          && b.capabilities.channels.contains(ch)
                          && overhear_eligible(s, b, ch)) {
      let cat = overhear_category(s, b, ch);  // per-channel; Speech uses building/floor,
                                              // PackSignal uses scent propagation, etc.
      let base = channel_overhear_base(ch, cat);
      let conf = base * exp(-planar_distance(s, b) / overhear_range);
      emit RecordMemory { observer: b, payload: copy_fact(f),
                          source: Overheard(s), confidence: conf }
    }
  }
}
```

Runtime constants: `MAX_ANNOUNCE_RECIPIENTS` (default 64) bounds the per-emission cascade; `OVERHEAR_RANGE_FRACTION` (default 0.2) sets overhear reach relative to the channel's primary range. Per-channel defaults (`channel_range(ch, sender)`): `Speech` → `SPEECH_RANGE * sender.vocal_strength` (default 30m × modifier); `PackSignal` → `PACK_RANGE` (default 20m scent + short vocal); `Pheromone` → `PHEROMONE_RANGE * wind_factor()` (default 40m × wind); `Song` → `LONG_RANGE_VOCAL` (default 200m); `Telepathy` → `f32::INFINITY`; `Testimony` → `0.0` (propagates via item transfer, not space). Overhear eligibility per channel: `Speech` uses `same_building ∨ (planar_distance < range ∧ z_separation < 3.0)`; `PackSignal` uses `distance < range` (scent doesn't respect walls); `Pheromone` uses downwind-only gradient; `Telepathy` uses faction-membership instead of space.

### 7.2 Determinism contract

- All randomness flows through `rng_state: u64` (PCG-style) plus per-agent derived streams (§9 #12). Per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose_tag)`; no stored per-agent RNG state. Never `thread_rng` or external RNG inside the simulation.
- Agent processing order is deterministically shuffled each tick via an RNG-seeded permutation.
- Events emitted within a phase are collected in append order; handlers process them deterministically.
- HashMap iteration seeded from `hash(world_seed ^ tick)` at world init; never re-seeded.
- Wall-clock reads (`SystemTime`, `Instant`) are forbidden in DSL code. Only the tick counter is available.
- GPU reductions over floats use sorted-key or fixed-point accumulation.
- Text generation (`ChronicleEntry.text`, LLM prose) is `@non_replayable` (§9 #21): templates render eagerly at event emission, an async LLM pass may rewrite entries in flagged categories (`Legendary`, `Founding`, `Death`, `Prophecy`). Saved prose is canonical — template-library changes only affect future entries. Bug-report replay artefacts bundle the template library; casual replay uses the current library.

Tests in `src/ai/core/tests/determinism.rs` (runtime) and the compiler's invariant checker verify that:

- `sim(seed).step(N).save() + load() + step(M) ≡ sim(seed).step(N + M)` for all primary state.
- Any rule, scoring, or mask code reading `event.text` fields is rejected at compile time.

### 7.3 Replay scope

Practical replay window is the bug-report scope (~1000 ticks, ~25 GB uncompressed, 2–5 GB compressed via zstd) (`stories.md` §33). Full-run replay at 200K agents (~5 TB/hour raw event volume) is not a goal.

Event-log storage policy (§9 #14): (a) filter to the replayable event-type subset only, (c) snapshot cadence **N=500 ticks** for rollback granularity, (d) zstd compression on both snapshot and event-log segments. Dev-only per-save rollback lets time-travel debugging jump to any 500-tick boundary inside the current bug-report window.

Replay artefacts per recorded segment:

1. Initial snapshot (zstd-compressed safetensors).
2. Event log with `{ tick, kind: u16, params: [u32; 4], source_agent: u32 }` — numeric IDs only, zstd-framed per 500-tick segment.
3. Tick-boundary RNG checkpoints every 100 ticks.
4. Schema combined-hash (forces trace-format-compatible replay; see §4).
5. (Bug-report artefacts only, §9 #21) Chronicle template library snapshot for prose-exact reproduction.

Non-replayable text is regenerated from templates on playback.

### 7.4 Save/load

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

Snapshot format: safetensors-compatible, length-prefixed, deserialize-in-place. Zero-malloc load uses pre-allocated agent slot pool.

---

## 8. Worked example

Three agents A, B, C in a settlement. A proposes marriage to B; cascade forms a Family group; B's membership updates; C witnesses and gains a positive memory.

**State before** (relevant fields only):

```
Agent A { id=1, pos=vec3(10, 5, 42.1), creature_type=Human, can_marry=true,
          married=false, personality.social_drive=0.7,
          memberships=[Settlement{s=1, role=Member}] }
Agent B { id=2, pos=vec3(11, 5, 42.1), creature_type=Human, can_marry=true,
          married=false, memberships=[Settlement{s=1, role=Member}] }
Agent C { id=3, pos=vec3(12, 6, 42.0), creature_type=Human,
          memberships=[Settlement{s=1, role=Member}] }

// surface_height((10, 5)) = 41.3; creature_height(Human) = 1.6; ground-snap keeps
// pos.z = 42.1 for all three after prior movement.

view relationship(1, 2) = { valence=0.72, familiarity=0.60, last_interaction=tick-5 }
```

**Tick T — A acts.** Scoring evaluates over A's masked candidates; the `InviteToGroup{kind=Family, target=B}` candidate wins the utility argmax because B sits in A's `known_actors` view with high valence. A emits:

```
action {
  macro_kind  = InviteToGroup
  group_kind  = Family
  target      = known_actors[0]      // resolves to agent_id = 2
}
```

**Mask evaluation (A):**

```
mask InviteToGroup{kind: Family, target: Agent(2)}
  when A.can_marry            = true
     ∧ ¬married(A)            = true
     ∧ ¬married(B)            = true
     ∧ relationship(A, 2).valence > MIN_TRUST (= 0.5)   = 0.72 > 0.5  ✓
     ∧ ¬A.in_active_invite_to(2, Family)              = true
  → mask passes
```

**Cascade fires (@phase(event)):**

```
physics invite_posted @phase(event) {
  on InviteToGroup{inviter: u, target: t, kind: k, terms: tm} {
    emit InvitePosted { invite_id: next_invite_id(), inviter: u, target: t,
                        kind: k, terms: tm, expires_tick: now + 500 }
  }
}

→ emit InvitePosted { invite_id: 42, inviter: 1, target: 2,
                      kind: Family, expires_tick: T+500 }
```

**Tick T+1 — B sees the invite.** B's `incoming_invites[0]` surfaces invite 42. B's scoring evaluates the `AcceptInvite` candidate against relationship valence (0.72) and invite kind (Family); the utility argmax picks it. B emits:

```
action {
  macro_kind = AcceptInvite
  target     = incoming_invites[0]    // resolves to invite_id = 42
}
```

**Mask evaluation (B):**

```
mask AcceptInvite{invite_id: 42}
  when 42 in B.incoming_invites             = true
     ∧ invite(42).kind_acceptable_to(B)      = true  (B is eligible for Family)
     ∧ B.eligible_for(invite(42).kind)       = true
     ∧ invite(42).expires_tick > now         = true
  → mask passes
```

**Cascade fires (@phase(event)):**

```
physics invite_accepted @phase(event) {
  on AcceptInvite{invite_id: id} {
    let inv = invite(id);
    match inv.kind {
      Family => {
        emit MarriageFormed { a: inv.inviter, b: inv.target, tick: now }
      }
      Alliance => { ... }
      ...
    }
    emit InviteResolved { invite_id: id, accepted: true }
  }
}

physics marriage_formed @phase(event) {
  on MarriageFormed{a: x, b: y, tick: t} {
    emit FoundGroup { kind: Family, founder_ids: [x, y], name_seed: (x ^ y) }
    emit RecordMemory { observer: x, event_type: MarriedB, entity_ids: [y],
                        emotional_impact: 0.7 }
    emit RecordMemory { observer: y, event_type: MarriedA, entity_ids: [x],
                        emotional_impact: 0.7 }
  }
}

physics witness_cascade @phase(post) {
  on MarriageFormed{a: x, b: y} {
    for w in query::nearby_agents_3d(pos(x), WITNESS_RADIUS)
             where w != x && w != y
                && z_separation(w, x) < 3.0 {
      emit Witnessed { observer: w, subject: x, kind: MadeNewFriend,
                       location: pos(x), entity_ids: [x, y],
                       emotional_impact: 0.15, source: Witnessed,
                       confidence: 1.0 }
    }
  }
}
```

**Agent C (nearby, same altitude) witnesses.** Cascade emits a `Witnessed` event; C's `memory_events` ring ingests it with `source=Witnessed, confidence=1.0`; C's `joy` view bumps on the next read.

**Tick T+2 — A announces to the settlement.** A's scoring elevates the `Announce` candidate because the just-completed marriage sits at the top of A's memory ring as a fresh, high-valence event. A emits:

```
action {
  macro_kind         = Announce
  announce_audience  = Group(settlement_s1)
  fact_ref           = recent_memory_events[0]   // resolves to A's MarriageFormed memory
}
```

Cascade enumerates up to `MAX_ANNOUNCE_RECIPIENTS` settlement members within hearing range, emits `RecordMemory { observer: r, source: Announced(s1), confidence: 0.8 }` for each. Bystanders outside the settlement group but within `OVERHEAR_RANGE` of A receive `source: Overheard(A), confidence: 0.6`. This seeds gossip propagation: a recipient's later `Communicate` rebroadcasts the fact as `source: TalkedWith`, decrementing confidence per the rules in `spec.md`.

**State after (tick T+2):**

```
Agent A { id=1, memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent B { id=2, memberships=[Settlement{s=1}, Family{g=99, role=Founder}] }
Agent C { id=3, memory_events.push(Witnessed{MadeNewFriend, [A, B], 0.15}) }
Group  99 { kind=Family, founder_ids=[1, 2], members=[1, 2], founded_tick=T+1 }

view mood(A) += 0.05 (from NeedsSatisfied via social_drive fulfillment)
view relationship(C, A).familiarity += 0.02
```

The entire cascade is 4 events (`InvitePosted`, `AcceptInvite` trigger, `MarriageFormed`, `FoundGroup` + memory + witness), one mask per action, two scoring-driven action selections (A's emit + B's emit), and one post-phase spatial query for witness cascade. No special "marriage system" exists; the outcome falls out of `verb` declarations, cascade rules, and views.

---

## 9. Settled decisions

All 29 open questions from the prior revision have been resolved through design interviews. Entries record the decision and one-line rationale; authoritative detail lives in the cross-referenced schema sections.

### 9.1 Action / quest mechanics

1. **Auction state machine** — `Resolution` enum = `{HighestBid, FirstAcceptable, MutualAgreement, Coalition{min_parties: u8}, Majority}`. `Coalition` for multi-party diplomatic pacts (stories_H §54); `Majority` for contested succession (story 46). `PostAuction` is an alias of `PostQuest{kind: Diplomacy|Charter|Service}` — no separate macro head. Cadence is per-world config, not compiled-in.
2. **Macro head firing rate** — Macro head runs **every tick**. Most emissions are `NoOp`; scoring evaluates both heads each tick so the utility backend can act on fresh state. Avoids stale-context issues from gated macro inference.
3. **Macro credit assignment** — Out of DSL scope. Credit-assignment algorithms for long-horizon macro decisions (Conquest, Marriage, Found) live in external pytorch training scripts that consume the trace format; the DSL only emits per-tick scoring values and the event stream that training code can reshape.
4. **Quest discovery push/pull hybrid** — **Hybrid**: posting emits `Announce(fact_ref=quest_id, audience)` cascade → recipients get the quest into their `known_quests` slot. Physical proximity + `GatherInformation` intent produces pull-style discovery for quests outside the announce radius. No always-on public broadcast; quest visibility is behaviour-driven.
5. **Slot K tuning** — **K=12 across the board** for spatial slots; K for non-spatial slots scales with role (leaders get larger `known_actors`, `known_groups`). Emergent behaviour and reputation dynamics are option-A ("agent must actively seek information") — no auto-populated global views.
6. **Cross-entity mask index design** — **Hybrid**: eager index materialization for `standing(group_a, group_b)`, `quest.eligible_acceptors`, `same_building`; lazy scan for low-cardinality predicates. Indices rebuild on the owning event (`GroupStandingChanged`, `QuestPosted`, `ChunkEnteredBuilding`).
7. **Concurrent quest membership** — **Multi-quest list** (up to K=4 active). Information primitives support multi-step plans (scout → report → assault chain). Mutual exclusion is scoring-level against internal state, not schema-enforced; the utility backend can rank a conflicting second quest below acceptance.
8. **Reward delivery on long quests** — **Compute at completion from current state**. No escrow. Poster reliability becomes emergent reputation: `QuestDefault` event fires if poster can't pay; accepters price the risk via `believed_intrinsic_value(poster_reputation, reward)`. Behavior-driven quest economics.
9. **Cancellation / amendment** — **`WithdrawQuest` macro head for taker only** (accepter changes mind, pays reputation cost). Poster amendment is forbidden; a poster who wants different terms must post a new quest. Simpler semantics; forces honest initial terms.
10. **Bid currency parity** — **`Payment::Combination{material, immaterial}` with per-agent valuation**. `intrinsic_value(observer, payment)` view computes subjective worth; `believed_intrinsic_value(self, other, payment)` gives theory-of-mind for bid crafting. Material and immaterial contributions are both required; ratio is per-buyer preference.
15. **Nested quest cancellation** — **C with B as acceleration structure**: parent Quest has `child_quest_ids: SortedVec<QuestId, 8>` materialised by the `QuestPosted{parent_id: Some(...)}` cascade. Parent cancel emits a cascade that cancels children; reward cascades traverse child_quest_ids for spoils distribution. Child tracking is the acceleration index; cancellation is first-class.
17. **Polygamous / cross-species / multi-parent family** — **C for all three**: `Agent.spouse_ids: SortedVec<AgentId, 4>` (polygamy), `can_marry(a, b) → bool` view folds in `creature_type` compatibility (cross-species at designer opt-in), `ChildBorn{parents: [AgentId; 4], inheritance_blend}` supports multi-parent lineages.
18. **Mercenary / service payment direction** — **C**: `AuctionKind::Service` inverts roles via `AuctionParty::{Buyer=Patron, Seller=Labourer}` with `Payment` flowing Buyer→Seller on completion. Service commitments use `Payment::ServicePledge{duration_ticks, scope}` evaluated against `intrinsic_value` for comparison.
19. **Alliance obligation enforcement** — **C / C**: "alliance" is not a first-class concept; it is an emergent standing derived from `standing(group_a, group_b)`. `SetStanding(target, kind)` is the universal macro; "declare war" is `SetStanding(target, kind=Hostile)`; "alliance" is `standing ≥ Friendly` ∧ reciprocal. Default response to ally-under-attack is scoring-governed, not mechanism-forced.
20. **Group-level invites vs agent-level invites** — **Agent-level only**. All invitations are agent-to-agent. Group mergers / coalitions use `PostQuest{kind: Diplomacy, resolution: Coalition{min_parties: K}}`. Removes the edge case where party-to-party agreement would tear factions apart.
30. **Communication channels (D30)** — `Capabilities.channels: SortedVec<CommunicationChannel, 4>` replaces `can_speak` / `can_hear` / `hearing_range` booleans. Enum variants: `Speech`, `PackSignal`, `Pheromone`, `Song`, `Telepathy`, `Testimony`. `channel: CommunicationChannel` is a parameter head on `PostQuest` / `Announce` / `InviteToGroup` / `Communicate` / `Converse` / `ShareStory`. Ranges, overhear eligibility, and recipient filtering are all per-channel. Cross-species communication requires a shared channel; wolves coordinate via `PackSignal` without language. Humans/dragons use `Speech`. Telepathic factions use unbounded-range `Telepathy`. Documents propagate via `Testimony` (item transfer, not spatial).
31. **Materialized-view storage hint (D31)** — `@materialized(on_event=[...], storage=<hint>)` lets the author pick a storage layout: `pair_map` (dense small-N × small-N, e.g. `Group × Group` standings), `per_entity_topk(K, keyed_on=<arg>)` (bounded per-entity slots, e.g. per-agent-per-membership Claim eligibility), `lazy_cached` (compute-on-demand + per-tick cache, e.g. low-cardinality derivations). Compiler rejects infeasible combinations (e.g., `pair_map` on `(AgentId, AgentId)` at N=200K). GPU/CPU routing (see `docs/compiler/spec.md` §1.2) follows from storage: intrinsic scalars + per-entity-slot materializations compile to GPU; lazy + unbounded-pair predicates stay CPU.

### 9.2 Runtime / infrastructure

11. *(retired — LlmBackend is no longer a DSL concept; ML is out of DSL scope.)*
12. **Per-agent RNG streams** — **C**: per-agent RNG seeded from `hash(world_seed, agent_id, tick, purpose)`. Enables fully-parallel sampling without save/load complexity (no extra stored state; seeds are derived).
13. **Materialized-view restoration on load** — **A primary / C rollback**: views serialize with schema-hash guard; on mismatch, rebuild from event log if available, otherwise refuse load. Rollback (dev-only, per-save) enables time-travel debugging from snapshot boundaries.
14. **Event log storage compression** — **(a) + (c) + (d) combined**: event-type filtering to replayable subset, fixed snapshot cadence **N=500 ticks**, zstd compression codec. ~2–5 GB per ~1000-tick bug report window.
21. **Chronicle prose side-channel lifecycle** — **C / X / P**: eager template rendering at event emission; async LLM rewrite pass for flagged categories (`Legendary`, `Founding`, `Death`, `Prophecy`); saved prose is canonical across template changes (player-facing history doesn't retcon); replay artefacts bundle the template library for exact reproduction in bug reports.
22. **Probe default episode count** — **Config-definable** per world + per-probe override via `seeds [42, 43, ...]` syntax. No hard-coded default.
23. **Training is out of DSL scope** — DSL emits Python dataclasses mirroring entities/events/`@traced` views plus a pytorch `Dataset` over the trace ring-buffer format. All training-script concerns (algorithm dispatch, importance sampling, curriculum, reward shaping, observation packing) live in external pytorch code, not DSL.
24. **Utility backend is the production NPC backend** — the utility backend is permanent. ML training is external to the DSL; scoring tables feed utility scoring and are written to traces for external reward reshaping. See `docs/compiler/spec.md` §Decisions.
25. **3D spatial hash structure** — **D**: 2D grid (cell=16m, voxel-chunk edge) with per-column sorted z-list, plus a `movement_mode ≠ Walk` sidecar (`Climb`, `Fly`, `Swim`, `Fall`). Slopes are Walk (they don't violate column-clustering). `movement_mode` is a primary field updated by the cascade.
26. **Overhear confidence decay** — **B + D hybrid**: category-based base + exponential distance decay. `base[SameFloor/DiffFloor/Outdoor] = {0.75, 0.55, 0.50}`; `confidence = base * exp(-planar_distance / OVERHEAR_RANGE)`. Walls are not raycast; wall structure is captured by category.
28. **`believed_knowledge` decay rate** — **3-tier volatility model**: `KnowledgeDomain` enum carries `volatility: {Short=500, Medium=20_000, Long=1_000_000}` ticks half-life. Reinforcement via observation-of-use, Communicate, and negative-evidence clearing. `Relationship.believed_knowledge_refreshed: [u32; 32]` stores per-bit last-refresh tick (~1 GB at 200k agents, acceptable).

### 9.3 Schema / memory

27. **Document trust_score authoring** — **C**: `trust_score` field removed. `Document: Item{kind=Document, author_id: AgentId, tick_written: u32, seal: Option<SealId>, facts: SortedVec<FactPayload, 16>}`. Reader computes confidence on `Read(doc)` from `(relationship_to_author, seal_validity, known_author_biases)`. Forgery is an emergent skill-gated quest.
29. **FactRef ownership after memory eviction** — **A**: materialize at creation. Quests and Documents store `FactPayload { tick, kind, params, author_id }` inline (16 × 20 B ≈ 320 B per doc). Memory rings evict freely; no refcount machinery; stale numeric IDs are an identity-layer concern, not a fact-layer concern.

### 9.4 Modding

16. **Mod event-handler conflict resolution** — → see `docs/compiler/spec.md` §Decisions.

---

A standing decision log (`docs/dsl/decisions.md`) carries rationale and reversal criteria in more detail; §9 summarises the current-state view.

---

## 10. What's NOT in scope

This DSL does not handle:

- **Machine learning**: policy architecture, training algorithms, curriculum, reward shaping, observation packing. Compiler emits Python dataclasses + a pytorch `Dataset` over the trace format; training code lives in external pytorch scripts.
- Text generation, LLM prose, dialogue authoring — `@non_replayable` side channel; out of deterministic sim.
- Asset pipeline — mesh, texture, shader assets for the voxel renderer live elsewhere.
- Rendering — voxel meshing, marching-cubes, SDF rendering are voxel-engine concerns.
- Networking / multiplayer synchronization — snapshot format exists but replication protocol is not specified.
- Save-file format with asset references — snapshots carry simulation state only; asset binding is a separate layer.
- Audio, particle effects, UI — display layers outside the sim loop.
- Build-system integration → see `docs/compiler/spec.md` §Non-goals.
- Online learning / federated training — all training is offline over serialized replay buffers.
- Human-in-the-loop labelling — probes are the authored-assertion surface; no runtime labelling.
- Procedural content generation beyond `region_plan` (terrain) and cascade-driven naming — narrative generation, dungeon layout, quest chains procedurally derived from cascades are not DSL-authored.

---

## Appendix A. Universal Action Mechanisms (detailed reference)

Detailed treatment of the four macro mechanisms — `PostQuest`, `AcceptQuest`, `Bid`, `Announce` — with enumerated gameplay scenarios mapped to each. Referenced by §3.3 of the main spec.

The v2 system reframings produced ~110 distinct action verbs (DeclareWar, ProposeAlliance, Vassalize, ProposeMarriage, FoundSettlement, PostBounty, Court, SwearOath, ...). The attacker correctly noted this list is fictional — it was generated by treating each gameplay outcome as a separate verb. In fact those outcomes collapse to **four universal mechanisms** plus a small set of micro primitives.

This doc specifies the universal mechanisms. The downstream action-vocabulary chapter (`spec.md` §3) consumes it.

### The four universal mechanisms

#### 1. `PostQuest(type, party_scope, target, reward, terms, deadline)`

"I (or a group I lead) want this thing done. Here's payment." A leader of a strong group with a weak rival emits `PostQuest{type=Conquest, party=Group(my_faction), target=Group(rival_faction), reward=Spoils(territory)}`. A merchant emits `PostQuest{type=Escort, party=Individual(self), target=Location(destination), reward=Gold(10)}`. A would-be spouse emits `PostQuest{type=Marriage, party=Individual(self), target=Agent(other), reward=Union}`.

```rust
struct PostQuestAction {
    type:        QuestType,
    party_scope: PartyScope,        // who's the proposer
    target:      QuestTarget,       // what/whom the quest concerns
    reward:      Reward,
    terms:       QuestTerms,        // duration, exclusivity, side conditions
    deadline:    Option<u64>,
}
```

##### Examples — gameplay scenarios expressed as `PostQuest{type=X}`:
| Scenario                          | type                | party_scope             | target                  | reward                  |
|-----------------------------------|---------------------|-------------------------|-------------------------|-------------------------|
| War on rival faction              | `Conquest`          | `Group(my_faction)`     | `Group(rival_faction)`  | `Spoils(territory)`     |
| Form mutual-defense alliance      | `MutualDefense`     | `Group(my_faction)`     | `Group(other_faction)`  | `Reciprocal`            |
| Submit weak group as vassal       | `Submit`            | `Group(weak_faction)`   | `Group(my_faction)`     | `Protection`            |
| Found a new settlement            | `Found`             | `Group(my_faction)`     | `Location(loc)`         | `Charter`               |
| Propose marriage                  | `Marriage`          | `Individual(self)`      | `Agent(other)`          | `Union`                 |
| Post bounty on outlaw             | `Assassinate`       | `Anyone`                | `Agent(outlaw)`         | `Gold(amount)`          |
| Issue prophecy                    | `FulfillProphecy`   | `Anyone`                | `Predicate(p)`          | `Glory`                 |
| Hire mercenary                    | `Service`           | `Individual(self)`      | `Anyone`                | `Gold(stipend)`         |
| Rescue captured agent             | `Rescue`            | `Anyone`                | `Agent(captive)`        | `Gold + reputation`     |
| Escort agent to destination       | `Escort`            | `Anyone`                | `Pair(agent, location)` | `Gold`                  |
| Hunt creatures of a kind          | `Hunt`              | `Anyone`                | `CreatureType(k)`       | `Gold per kill`         |
| Defend a settlement               | `Defend`            | `Group(my_faction)`     | `Location(loc)`         | `Reputation`            |
| Pilgrimage to a shrine            | `Pilgrimage`        | `Individual(self)`      | `Location(shrine)`      | `Faith`                 |
| Plan a heist                      | `Heist`             | `Group(my_party)`       | `Structure(target)`     | `Loot`                  |

#### 2. `AcceptQuest(quest_id)` / `JoinParty(party_id)`

"I'll do that." Universal acceptance.

```rust
struct AcceptQuestAction { quest_id: u32, role_in_party: PartyRole }
struct JoinPartyAction   { party_id: u32, role_in_party: PartyRole }
```

##### Examples:
| Scenario                              | Action                                                       |
|---------------------------------------|--------------------------------------------------------------|
| Accept a marriage proposal            | `AcceptQuest{type=Marriage}` by the targeted agent           |
| Join the war on one's group's side    | `AcceptQuest{type=Conquest/Defend}` by group members         |
| Swear vassalage                       | `AcceptQuest{type=Submit}` by the weaker group               |
| Sign onto an alliance                 | `AcceptQuest{type=MutualDefense}` by target group            |
| Join a settlement-founding expedition | `AcceptQuest{type=Found}` by would-be colonists              |
| Take a mercenary contract             | `AcceptQuest{type=Service}`                                  |
| Enlist in an active war               | `JoinParty(faction_war_party)`                               |

`JoinParty` also handles defection: leaving party A (with possible reputation/grudge cascade events) and joining party B.

#### 3. `Bid(auction_id, payment, conditions)`

"I want that resource at that price." Auction-style competition for limited goods or contested outcomes.

```rust
struct BidAction {
    auction_id: u32,
    payment:    Payment,        // gold, items, services, future-promises
    conditions: BidConditions,  // min-quantity, must-win, etc.
}
```

##### Examples:
| Scenario                              | Action                                                       |
|---------------------------------------|--------------------------------------------------------------|
| Offer a private trade                 | private 2-party `Bid` in an ad-hoc auction                   |
| Buy from the local market             | `Bid` in the standing commodity-market auction               |
| Hire a worker / mercenary             | `Bid` in a Service auction                                   |
| Compete for an artifact               | `Bid` in an artifact auction                                 |
| Lobby for a settlement charter        | `Bid` in a political auction with reputation as currency     |
| Ransom a captive                      | `Bid` against captor for release                             |

#### 4. `Announce(audience, fact_ref)`

"I'm telling a group of people a thing I know." The macro counterpart to the `Communicate` micro primitive: one emission reaches many recipients instead of one. This is the group-level information-sharing mechanism — an order broadcast to a faction, a rumor seeded into a settlement, a prophecy proclaimed to pilgrims.

```rust
struct AnnounceAction {
    audience:  AnnounceAudience,
    channel:   CommunicationChannel,    // §9 D30 — modality of transmission
    fact_ref:  FactRef,                 // local memory event_id the speaker wants to broadcast
}

enum AnnounceAudience {
    Group(GroupId),       // all members of a group
    Area(vec3, f32),      // all agents within radius of a point
    Anyone,               // public broadcast, radius MAX_ANNOUNCE_RADIUS from speaker
}
```

Cascade: enumerate eligible recipients (group members in speaker's hearing range for `Group`; agents within the sphere for `Area`; agents within `MAX_ANNOUNCE_RADIUS` for `Anyone`). For each, insert a `MemoryEvent` with `source = Announced(g)` (for `Group`) or `source = Overheard(speaker)` (for `Area` / `Anyone` bystanders), confidence 0.8. Bounded by `MAX_ANNOUNCE_RECIPIENTS` per cascade to prevent runaway propagation.

##### Examples:
| Scenario                              | Action                                                       |
|---------------------------------------|--------------------------------------------------------------|
| Faction leader announces war          | `Announce{audience=Group(faction), fact_ref=WarDeclared}`    |
| Town crier shouts news                | `Announce{audience=Area(town_center, 30), fact_ref=...}`     |
| Prophet proclaims a vision            | `Announce{audience=Anyone, fact_ref=ProphecyHeard}`          |
| Captain issues orders to patrol       | `Announce{audience=Group(patrol_party), fact_ref=Order}`     |
| Outlaw shouts a threat                | `Announce{audience=Area(self.pos, 20), fact_ref=Threat}`     |

### Micro primitives (still needed as direct verbs)

These can't be reduced because they ARE the per-tick physical/social acts:

- **Movement**: `MoveToward(pos)`, `Flee(from)`, `Hold`
- **Combat**: `Attack(target)`, `Cast(ability_idx, target)`, `UseItem(slot, target)`
- **Resource**: `Harvest(node)`, `Eat(food_source)`, `Drink(water_source)`, `Rest(loc)`
- **Construction**: `PlaceTile(pos, type)`, `PlaceVoxel(pos, mat)`, `HarvestVoxel(pos)`
- **Social atomic**: `Converse(target)`, `ShareStory(audience, topic)`
- **Info atomic (push)**: `Communicate(recipient, fact_ref)` — point-to-point sharing of one memory event; cascade validates `fact_ref ∈ self.memory`, inserts a `MemoryEvent` into `recipient.memory` with `source = TalkedWith(self)`, confidence `min(self.memory[fact_ref].confidence, 0.8)`. Mask: `r.can_hear ∧ planar_distance(self, r) < CONVERSE_RANGE`. `FactRef` is a bounded handle — a local event_id drawn from the speaker's memory ring.
- **Info atomic (pull)**: `Ask(target, query)` — request facts from an entity. The target type disambiguates the cascade:
  - If `target` is an `Agent`: emits `InformationRequested { asker: self, target, query }`. The target's next-tick scoring sees the pending request in its `incoming_requests` view; the utility backend may pick `Communicate(asker, fact_ref)` matching the query, or leave it unanswered. Mask: `target.can_hear ∧ planar_distance(self, target) < CONVERSE_RANGE`.
  - If `target` is a `Document` item: cascade resolves immediately — extract `target.facts` matching `query`, insert into `self.memory` with `source = Testimony(target.id)`, confidence derived per §9 D27 from `(relationship_to_author, seal_validity, known_author_biases)`. Mask: `target ∈ self.inventory.documents`.
- **Info atomic (sugar)**: `Read(doc)` — language-surface shorthand for `Ask(doc, QueryKind::AboutAll)`. The compiler lowers `Read` to the document-target branch of `Ask`; runtime vocabulary does not carry a separate `Read` micro. See `docs/compiler/spec.md` §3.
- **Memory atomic**: `Remember(entity, valence, kind)` (typically internal, but expose as an action so policies can choose what to record)

### Total action vocabulary

```
4 macro mechanisms:     PostQuest, AcceptQuest|JoinParty, Bid, Announce
19 micro (surface):     movement(3) + combat(3) + resource(4) + construction(3)
                      + social(2) + info(3: Communicate push, Ask pull, Read)
                      + memory(1)
18 micro (runtime):     same, minus Read — which the compiler lowers to
                        Ask(doc, QueryKind::AboutAll).
─────────────────────────
23 categorical actions at the language surface;
22 at the runtime (MicroKind enum) after Read → Ask lowering.
```

The richness comes from the parameter heads — `QuestType`, `PartyScope`, `QuestTarget`, `Reward`, `BidConditions`, `Payment`, `AnnounceAudience`, `FactRef` enums each have many variants. The categorical action head is small (~19); the parameter heads carry the actual semantic complexity.

### Required types

#### `QuestType` enum

```rust
enum QuestType {
    // physical / errand
    Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Custom,
    // political
    Conquest,        // war for territory
    MutualDefense,   // alliance
    Submit,          // vassalage
    Found,           // settlement founding
    Charter,         // settlement charter petition
    Diplomacy,
    // personal
    Marriage,        // proposal
    Pilgrimage,      // religious journey
    // economic
    Service,         // mercenary contract / employment
    Heist,           // organized theft
    Trade,           // long-haul caravan run
    // narrative
    FulfillProphecy,
}
```

#### `PartyScope` enum (new)

Who or what is acting as a party / who can be assigned?

```rust
enum PartyScope {
    Individual(AgentId),       // just one agent
    Group(GroupId),            // all members of a group (faction, family, guild,
                               //   pack, religion, party — Group is universal)
    Settlement(SettlementId),  // all residents (= members of the settlement-Group)
    Anyone,                    // open call
    Role(RoleTag),             // anyone matching a role/tag
}
```

`Group` is the universal social-collective primitive — same machinery handles factions, families, guilds, religions, hunting packs, criminal cabals, adventuring parties, settlements. The `kind` discriminator differentiates them.

`Quest.party_member_ids: Vec<AgentId>` materializes the scope at the moment of acceptance:
- `Group(g)` → at quest creation, all current group members are eligible; updates if membership changes during quest
- `Anyone` → list grows as agents `AcceptQuest`
- `Individual(e)` → fixed at one agent

#### `QuestTarget` enum — what the quest is about

```rust
enum QuestTarget {
    Agent(AgentId),
    Group(GroupId),
    Location(vec3),               // 3D world-space point
    Structure(StructureRef),      // a derived view over a tile/voxel region
    CreatureType(CreatureType),   // e.g. "any wolves" for hunt quests
    Pair(Box<QuestTarget>, Box<QuestTarget>),  // escort: (Agent, Location)
    Predicate(PredicateId),       // prophecy: a derived condition
    Item(ItemId),
    Region(RegionId),
}
```

#### `Reward` enum

```rust
enum Reward {
    Gold(f32),
    Xp(u32),
    Items(Vec<ItemId>),
    Reputation(f32),
    Faith(f32),
    Spoils(SpoilsKind),       // territory, loot from conquest
    Charter(CharterTerms),    // for Found / Charter quests
    Union,                    // marriage union
    Reciprocal,               // mutual defense
    Protection,               // vassalage
    Glory,                    // narrative reward
    Promise(Box<Reward>, u64),// reward delivered later (futures)
    Combination(Vec<Reward>),
}
```

#### `Payment` enum (for Bid / Trade)

```rust
enum Payment {
    Gold(f32),
    Commodity(CommodityKind, f32),
    Item(ItemId),
    Service(ServicePromise),       // future labor
    Reputation(f32),
    Combination(Vec<Payment>),
}
```

### Information as an action class

Information is a first-class action channel, parallel to quests / auctions / invites. Agents produce, share, and consume memory events with explicit provenance; masks gate actions on what an agent knows.

#### Source

Every `MemoryEvent` carries a `source` tag that records its provenance and seeds its confidence:

```rust
enum Source {
    Witnessed,              // saw it directly; default confidence = 1.0
    TalkedWith(AgentId),    // told by another agent; 0.8
    Overheard(AgentId),     // caught someone else's Communicate/Announce; 0.6
    Rumor { hops: u8 },     // N-hop propagation; 0.8^hops
    Announced(GroupId),     // received from a group broadcast; 0.8
    Testimony(ItemId),      // read from a Document; per doc.trust_score
}
```

`MemoryEvent.confidence: f32 ∈ [0, 1]` is derived from `source` by default; cascades may override when they have semantic reason to.

#### FactRef

`FactRef` is a bounded handle referring to a specific memory event in the speaker's ring:

```rust
struct FactRef {
    owner:     AgentId,     // whose memory ring
    event_id:  u32,         // local id within that ring
}
```

Cross-agent `FactRef` is valid only at the moment of `Communicate` / `Announce`; recipients receive a *copy* of the referenced event into their own ring (with an updated `source`). A recipient's subsequent `Communicate` generates a new `FactRef` pointing into their own memory.

#### AnnounceAudience

```rust
enum AnnounceAudience {
    Group(GroupId),
    Area(vec3, f32),
    Anyone,
}
```

Runtime constants bound the cascade: `config.communication.max_announce_recipients` caps recipient count per emission; `config.communication.max_announce_radius` caps the `Anyone` / `Area` radius. Overhear scans agents within `config.communication.overhear_range` of the speaker for both `Communicate` and `Announce`, admitting them as bystanders with `source = Overheard(speaker)` subject to the mask eligibility in §Overhear. These values live in `assets/sim/config.sim` and are tuneable at runtime through `assets/config/default.toml`; previous drafts of this spec referred to them as the `MAX_*` / `OVERHEAR_*` `pub const`s that have since been migrated into `config` blocks.

#### Confidence propagation

Propagation rules on insertion into a recipient's memory:
- `Communicate`: `conf_r = min(self.memory[fact_ref].confidence, 0.8)`.
- `Announce{Group | Area | Anyone}`: 0.8 for direct audience; 0.6 for overhear bystanders.
- `Read(doc)`: `conf_r = doc.trust_score` per fact.
- `Rumor{hops}`: each subsequent `Communicate` that the recipient emits (under the `Rumor` variant, used when the speaker is themselves relaying a `TalkedWith` source) increments `hops`; confidence multiplies by 0.8.

#### Mask predicates

New mask operators gate information-sensitive actions on memory state:

- `knows_event(self, event_kind, target_pattern)` — memory contains a matching event (any source).
- `knows_agent(self, agent_id)` — memory references `agent_id` OR `agent_id ∈ self.relationships`.
- `confident_about(self, fact, threshold)` — memory contains `fact` with `confidence ≥ threshold`.
- `recent(self, event_kind, max_age_ticks)` — memory contains matching event with `now − tick ≤ max_age_ticks`.

Example mask rules:
- `PostBounty(t) when knows_agent(self, t) ∧ confident_about(self, Fact::CommittedCrime(t), 0.5)`.
- `Deceive(t, fact) when ¬believed_knowledge(t, fact.domain)` (reads the theory-of-mind bitset on `Relationship`).

#### Document as Item kind

A new `ItemKind::Document` carries a bounded list of facts:

```rust
struct Document {
    id:         ItemId,
    author:     AgentId,
    facts:      SortedVec<FactRef, 16>,
    trust_score: f32,        // set at creation by author; drives Read confidence
}
```

The `Read(doc_item)` micro primitive cascades each fact in `doc_item.facts` into `self.memory` with `source = Testimony(doc_item.id)` and confidence `doc_item.trust_score`. Enables scrolls, charters, wills, and oaths-on-paper as first-class information vectors.

#### Overhear semantics

When `Communicate(recipient)` or `Announce(audience)` fires, the cascade also scans agents within `OVERHEAR_RANGE` of the speaker. Each such bystander receives a `MemoryEvent` copy with `source = Overheard(speaker)`.

Overhear eligibility:
- **Same interior** — `tile.building_id(bystander.pos) == tile.building_id(speaker.pos)`, any floor within the same building.
- **Outdoor** — `planar_distance(bystander, speaker) < OVERHEAR_RANGE ∧ z_separation(bystander, speaker) < 3.0` (roughly same altitude layer).

[OPEN] whether overhearing degrades confidence further beyond the 0.6 default (e.g. partial hearing reducing it to 0.4).

#### Chronicle is not agentic

The `chronicle` stream on `WorldState` is dev-facing only: narrative UI, post-mortem analysis, player-facing history. Agents never read it. Any agent-facing knowledge of a chronicled event is carried by the explicit propagation channels above (`Witnessed` / `TalkedWith` / `Overheard` / `Announced` / `Rumor` / `Testimony`). This preserves the determinism contract: `ChronicleEntry.text` can be regenerated from templates across replays without affecting agent decisions.

### Auction state machine

Sketch:

```rust
struct Auction {
    id:            u32,
    kind:          AuctionKind,        // Item, Service, Charter, Diplomatic, Commodity, ...
    item:          AuctionItem,        // what's being sold
    seller:        AuctionParty,       // who's offering (Individual/Group/World)
    bids:          Vec<Bid>,           // accepted bids
    open_tick:     u64,
    deadline_tick: u64,
    resolution:    AuctionResolution,  // HighestBid, FirstAcceptable, MutualAgreement
    visibility:    Visibility,         // who can see and bid
    reserve_price: Option<Payment>,    // minimum acceptable
}

enum AuctionKind {
    Item,                 // standard goods auction
    Commodity,            // standing market (food, wood, iron...)
    Service,              // hire NPCs (mercenaries, advisors, builders)
    Charter,              // settlement charter petitions
    Diplomatic,           // alliance brokering, vassal petitions
    Marriage,             // dowry-style marriage market
    Ransom,               // captive release
}

enum AuctionResolution {
    HighestBid,           // best payment wins
    FirstAcceptable,      // first bid above reserve wins
    MutualAgreement,      // both parties must accept (used for diplomatic / marriage)
    Allocation,           // multi-winner: distribute item across top bidders
}
```

#### Auction lifecycle as events

```
AuctionPosted    { auction_id, kind, seller, item, deadline, visibility }
BidPlaced        { auction_id, bidder, payment, conditions }
BidWithdrawn     { auction_id, bidder }
AuctionResolved  { auction_id, winner: Option<EntityId>, payment }
AuctionExpired   { auction_id }            // no acceptable bids
```

Cascade rules:
- `AuctionResolved{kind=Item, winner=W, payment=P}` → `TransferGold(W → seller, P.amount)` + `TransferItem(seller → W, item)`
- `AuctionResolved{kind=Marriage, winner=W}` → `MarriageFormed(seller, W)` event (which updates spouse_id on both)
- `AuctionResolved{kind=Charter, winner=W}` → `CharterGranted(settlement, W, terms)`
- etc.

The auction system becomes the universal mediator for any "who gets this contested resource" decision.

### How "war is a quest" plays out

Concrete walkthrough:

1. **A leader of group G_self reads** `G_self.military_strength`, `known_groups[K]` with their strength + standings, recent grievance events, settlement values — all via declared views.
2. **Leader's scoring picks** `PostQuest{type=Conquest, party_scope=Group(G_self), target=Group(G_rival), reward=Spoils(rival_settlements), deadline=tick+5000}`.
3. **Cascade emits** `QuestPosted` event with party_member_ids = current G_self members at that tick. The leader's next decision is `Announce{audience=Group(G_self), fact_ref=<QuestPosted memory event>}`, broadcasting the quest to faction members. Each member receives a `MemoryEvent{source=Announced(G_self), confidence=0.8}` referencing the new quest.
4. **G_self members see** the quest in their `active_quests` view AND in `recent_memory_events` with `source=Announced`. Their next-tick scoring now treats "I'm in a Conquest quest against G_rival" as an active context — closing the information path rather than relying on magic faction-wide awareness.
5. **Each member's scoring decides** how to act on the war: the utility backend picks per-tick micro primitives (`MoveToward(rival_settlement)`, `Attack(rival_agent)`, etc.) gated by the quest context.
6. **Quest progress tracked** as a derived view: `progress(quest) = f(G_rival.military_strength, G_rival.settlements_remaining, casualties)`.
7. **Quest completes** when G_rival is conquered → `QuestCompleted{quest_id, winners=party}` event → cascade emits `SettlementConquered`, `Spoils` distribution, reputation/legend updates.
8. **Quest fails** if deadline reached without victory or military_strength reduced to zero → `QuestExpired` → war-exhaustion effects on participants, possible defection cascade.

War is a long-duration group-scoped Conquest quest. The runtime needs:
- Quest with `PartyScope::Group(_)` semantics
- Per-tick progress evaluation as derived view
- Cascade rules from quest completion to gameplay outcomes (spoils, reputation, etc.)

### How "marriage is a quest" plays out

1. Agent A's scoring selects `PostQuest{type=Marriage, party_scope=Individual(A), target=Agent(B), reward=Union, terms=ExclusiveAcceptance}`.
2. Cascade emits `QuestPosted` with `party_member_ids = [B]` (target as sole eligible acceptor).
3. Agent B sees the quest in `active_quests`. B's scoring evaluates accept/reject based on `relationship(B, A).trust`, personality traits, own marital state.
4. If B emits `AcceptQuest(quest_id)` → cascade emits `QuestCompleted` → `MarriageFormed(A, B)` event → both A and B's `memberships` get a `Group{kind=Family}` entry via event handler.
5. If quest deadline passes without acceptance → `QuestExpired` → A's emotion.grief bump (rejection); no marriage.

### Open questions

1. **How rich does the auction system need to be at MVP?** Minimum viable: HighestBid resolution, fixed deadline, no withdrawal, single-winner. Everything else (allocations, withdrawal, side conditions) is later. Probably ~200-300 LOC for the minimum.

2. **Quest deadline + group-scoped party interaction.** If a group war takes 2000 ticks but member agents die/migrate in/out, does party_member_ids stay frozen or update? Probably update — `members(quest) = current_group_membership(group_id)` as a view, not a stored list.

3. **Reward delivery on long quests.** Conquest reward = "spoils from target faction" can't be specified upfront. The cascade rule for `QuestCompleted{type=Conquest}` computes spoils at completion from current state. This means reward type carries logic, not just data — needs careful DSL design.

4. **Quest visibility / discovery.** How do NPCs learn about quests they should accept? Two models:
   - Push: `QuestPosted` event triggers re-scoring for nearby/eligible NPCs (cheap if parties are spatially clustered)
   - Pull: NPCs query `available_quests_for(self)` as part of scoring evaluation (always works but more expensive)
   Probably hybrid: faction-scoped quests pushed to members, public bounties pulled.

5. **Concurrent quest membership.** Can an NPC be in multiple quests at once (in a faction war AND on a personal pilgrimage AND running a caravan)? Current `party_id: Option<u32>` is single-slot. Need to extend or split: `party_id: Option<u32>` for adventuring + `active_quests: Vec<u32>` for everything else.

6. **Bid currency for non-economic auctions.** What does it mean to "bid reputation" in a Charter auction? Need to decide: are bids in the auction's native currency, or any compatible Payment type?

7. **Cancellation / amendment.** Can a leader retract a war declaration after posting? An NPC retract a marriage proposal? Probably yes via `WithdrawQuest(quest_id)` action — but it's another verb.

8. **Auction frequency.** Standing commodity markets resolve continuously. Item auctions on a fixed schedule. Diplomatic auctions only when posted. Need cadence per `AuctionKind`.

### Implementation order (if we ship this)

1. Extend `QuestType` enum with new variants
2. Add `PartyScope`, `QuestTarget`, `Reward`, `Payment` enums
3. Implement minimal Auction state machine (HighestBid resolution only)
4. Add cascade rules for new QuestType completions (Conquest → Spoils, Marriage → MarriageFormed, etc.)
5. Add quest-eligibility view for mask and scoring construction
6. Migrate existing quest-like systems (warfare, family, settlement_founding, vassalage, alliance_blocs) to emit PostQuest instead of direct mutations
7. Migrate existing trade/contract systems to use Auction
8. Delete the now-redundant systems

This is several weeks of work but collapses ~30 systems into ~5 mechanisms. Worth it if the unification holds.

### Risks

- **Quest abstraction may be too uniform.** A 5-tick marriage proposal and a 5000-tick faction war are different beasts; treating them with one machinery may force compromises in either direction.
- **Auction overhead for small trades.** A trivial commodity buy might end up as auction post → bid → resolve over 1+ ticks when current direct-trade is one tick. May need fast-path for "simple immediate trade."
- **NPC scoring complexity grows.** "Should I PostQuest? Of what type? With what party_scope and target and reward?" is a much richer decision than "should I attack?" Utility-backend scoring tables have to cover the full structured-action space.
- **Quest discovery mechanism affects gameplay.** Push-based means NPCs only see quests they're notified about; pull-based means per-tick view evaluation cost balloons. Wrong choice creates gameplay artifacts.

### What's settled vs what's open

**Settled:** the four universal mechanisms (PostQuest / AcceptQuest / Bid / Announce) collapse the action vocabulary cleanly. Information is a first-class action class, carried by `Communicate` / `Announce` / `Read` + source-tagged `MemoryEvent`s.

**Open:** auction implementation details, quest semantics for long-duration faction quests, reward computation timing, cancellation/amendment, multi-quest membership, currency cross-compatibility in bids.

This doc captures the design intent. The DSL surface (`spec.md`) and the synthesis pass (`synthesis.md` — pending) consume this vocabulary.
