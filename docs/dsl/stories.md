# User-Story Investigations

Per-batch investigations of the 64 user stories. Batches group by domain. Story numbers carry a batch prefix so cross-references from other docs can disambiguate (e.g., §E.30 = story 30 in batch E).

## Contents
- [Batch A-B — DSL & Action/Mask Authoring](#batch-ab--dsl--actionmask-authoring) — (formerly `stories.md`) — stories 1-12
- [Batch C — Observation Schema (ML Engineer)](#batch-c--observation-schema-ml-engineer) — (formerly `stories.md`) — stories 13-18
- [Batch D — Training Pipeline (Trainer)](#batch-d--training-pipeline-trainer) — (formerly `stories.md`) — stories 19-24
- [Batch E — Runtime / Sim Engineer](#batch-e--runtime--sim-engineer) — (formerly `stories.md`) — stories 25-33
- [Batch F/G/I/J — Observability, Modding, Auctions, Adversarial](#batch-fgij--observability-modding-auctions-adversarial) — (formerly `stories.md`)
- [Batch H — Player-experienced Behaviors (Acceptance Tests)](#batch-h--player-experienced-behaviors-acceptance-tests) — (formerly `stories.md`) — stories 45-64

---

## Batch A-B — DSL & Action/Mask Authoring

Per-story analysis of the user stories proposed in *(archived)* under category **A (DSL authoring)** and **B (Action and mask authoring)**. Each entry cites specific sections of the settled design and flags gaps concretely.

---

#### Story AB.1: Declare a new entity type with baseline fields
**Verdict:** GAP (as written) → REFRAMED as SUPPORTED
**User's framing note:** Raw entity-type declaration is inherently dangerous. We should only support this as abstractions *over* the existing entity types (Agent / Item / Group) — never as new root kinds that force codegen changes.

**How the DSL supports this (reframed):**

The settled design already fixes the entity taxonomy at three kinds (plus an optional Projectile): Agent, Item, Group. See `README.md` — "Settled" list, item "Three entity types: Agent + Item + Group (+ optional Projectile)". The discriminators that differentiate instances within a kind are:

- `Agent.creature_type` (`state.md` — "Identity & Lifecycle" row `creature_type`): Human, Elf, Dwarf, Wolf, Dragon, Goblin, ...
- `Group.kind` (`state.md` § "Group (universal)" — `kind: GroupKind`): Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other
- `Item` is a single shape with owner / durability / history; differentiation by `ItemKind` (not yet enumerated in these docs — see story 41 which is the modder version)

So "a new entity type" almost always decomposes to one of:

1. **A new `creature_type`** — story 11 handles this.
2. **A new `GroupKind`** — story 10 handles this.
3. **A new `ItemKind`** — story 41 (modder).
4. **A genuinely new top-level kind** — not supported and per the user's note, intentionally so.

The question then becomes: what does "abstraction over existing entity types" mean in DSL surface terms? Three plausible readings:

- **Alias / newtype** — `entity Centaur = Agent { creature_type: CreatureType::Centaur, default_personality: ..., default_memberships: [Group{kind=Pack}] }`. Pure parameterization; compiler expands to Agent + defaults.
- **View-only** — `view Merchant(a: Agent) = a.memberships contains Group{kind=Guild, specialty=Trade} ∧ a.data.class_tags contains "trader"`. Gives authors a named predicate, produces no storage.
- **Parameterized role** — a Template with default stats/personality/starting-memberships used by a spawner, compiled down to Agent + config. Matches the existing `Template.kind` machinery (`state.md` AgentData row `creature_type`).

**Implementation walkthrough (reframed: add a Centaur-like "entity"):**

Files a developer would touch:
- Add `CreatureType::Centaur` variant (spec.md §2.1.1, creature_type_one_hot) — story 11.
- Author an `entity Centaur = Agent { ... }` alias block in the DSL source; compiler expands to Agent struct with defaults (`creature_type`, default `capabilities`, default `personality`, default `memberships`).
- Add `hero_template`-equivalent data file for spawn parameters.
- Observation schema gets one new bit in the one-hot (`state.md` — "creature/role" row); schema hash bumps (spec.md §4).

**Gaps / open questions:**

- No DSL block name is currently proposed for the alias/template form. The policy schema's DSL pseudocode (spec.md §3) declares observations, action, mask, reward, backend — but has no top-level `entity` block. The docs need a grammar clarification: are templates/aliases DSL declarations or external data files?
- **Proposed extension:** add a top-level `template <Name> : Agent { ... }` declaration whose only role is parameter defaults and implicit group assignments. Forbid any other top-level entity declaration. This nails down "abstraction over existing entity types" without opening the door to new root kinds.

**Related stories:** 10 (GroupKind), 11 (creature_type), 41 (ItemType).

---

#### Story AB.2: Declare typed events
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Events are the universal state-mutation channel. From `README.md` — "Settled" list item "Strict event-sourcing rubric. State mutations are events; current state is a fold over events + entity baseline."

Concrete event taxonomy already catalogued:

- **Low-level WorldEvent enum** in `state.md` § "WorldEvent Enum (13 variants)": `Generic`, `EntityDied`, `QuestChanged`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.
- **Auction/quest events** in `spec.md` § "Auction lifecycle as events": `AuctionPosted`, `BidPlaced`, `BidWithdrawn`, `AuctionResolved`, `AuctionExpired`.
- **Per-agent MemoryEvent** in `state.md` § "MemoryEvent" with typed `event_type: MemEventType` (WasAttacked, FriendDied, CompletedQuest, etc.).
- **ChronicleEntry** in `state.md` § "ChronicleEntry" with typed `category: ChronicleCategory`.
- **Per-tick StructuralEvent** in `state.md` § "StructuralEvent" (`FragmentCollapse`, `StressFracture`).

The reward block (`spec.md` §2.5) already treats event references as first-class: `+1.0 on event(EntityDied{killer=self ∧ target.team ≠ self.team})`. That's only sound if events are typed and matchable in the DSL.

**Implementation walkthrough (add a new `Harvest` event):**

1. Declare in DSL: `event Harvest { harvester: AgentId, node: ResourceRef, commodity: CommodityKind, amount: f32 }`
2. Compiler emits:
   - A variant on the runtime event enum (plus matching append-only buffer entry)
   - Serialization / replay handlers (the ring is append-only per spec.md §1 "Event-sourced runtime")
   - A mask for policy-side event pattern matching (`event(Harvest{harvester=self})`)
3. Add cascade rule (see story 4) that reads the event and mutates state (e.g. `Harvest → inventory[harvester][commodity] += amount → if tile.remaining ≤ 0: emit ResourceDepleted`).
4. Reward block can reference it via `on event(Harvest{harvester=self})`.
5. Bump schema hash if event is observation-visible (spec.md §4 + Open question #17 "Schema hash scope — should hash cover observation only, or also action vocabulary + reward?").

**Gaps / open questions:**

- The `event` declaration block is not yet fleshed out as formal grammar (see README.md Open question #8: "DSL surface details. The pseudocode... isn't a formal grammar yet").
- Event categories split across three buckets (WorldEvent, MemoryEvent, StructuralEvent). Is the DSL single-namespace, or does the author have to pick a bucket? Needs decision in the synthesis doc.
- Several events today carry `String` (`Generic{category,text}`, `EntityDied{cause}`, `ChronicleEntry.text`) which are flagged GPU-hostile in `state.md` § Summary. Story 60 ("Determinism bound on text generation") points at the resolution: text fields should be marked non-replayable / non-observation.
- **Proposed extension:** add `event` annotations — `@replayable` (default), `@non_replayable` (text fields), `@observation_visible` (drives schema hash), `@gpu_amenable` (drives backend-compile decision).

**Related stories:** 4 (cascade rules), 15 (versioning), 27 (add rule without breaking), 60 (non-replayable text gen), 64 (ActionKind bumps schema hash).

---

#### Story AB.3: Declare a derived view
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Views are the second pillar after events. `README.md` — "Settled" items "Buildings and resources are derived views over the world's spatial data" and "`is_hostile(a, b)` is a derived view over relationship valence + group war standings."

Concrete view examples already written:

- `spec.md` §2.3 mask block: `is_hostile(self, t) = relationship.valence < HOSTILE_THRESH ∨ groups_at_war(self, t) ∨ predator_prey(self.creature_type, t.creature_type)`.
- `spec.md` §3 observation DSL: `self.mood = view::mood(self)`, `view::agent_features_for_observation(other, viewer=self)`, `view::relationship(self, other).valence`, `view::shared_group_count`, `view::known_actors(self)`.
- `state.md` § "What's NOT on Group (derived views)": `population`, `is_at_war(other)`, `is_allied(other)`, `wealth_per_member`, `cultural_descriptor`, `reputation_among(other_group)`.
- `state.md` § "Derivation Graph" splits fields into PRIMARY INPUTS, SECONDARY (Computed/Emergent), INFRASTRUCTURE. The SECONDARY block is the view surface.

**Implementation walkthrough (declare `view::is_legendary(a: Agent)`):**

1. Author the DSL: `view is_legendary(a: Agent) = a.data.deeds.len() ≥ 3 ∨ a.data.fame_log > LEGENDARY_THRESH`.
2. Compiler inspects the body for `@materialize` hint (spec.md §1 settled item "GPU compilation is scoped") — if lazy (default), the view compiles to a function called per-read; if eager, it emits event-fold logic that materializes the value whenever inputs change (see story 26).
3. Mask and observation blocks reference `view::is_legendary(self)` directly.
4. For materialized views, the compiler wires subscription to input events (e.g. `ChronicleEntry{entity_ids ∋ a.id}` triggers recompute).
5. Story 40 ("Flamegraph attribution") implies views need per-declaration timing instrumentation — the compiler tags each view body with a timing ID.

**Gaps / open questions:**

- Materialization hint syntax not specified. README.md "Settled" row mentions `view::mood(self)` as hot path but not the annotation that marks it eager.
- Views with cross-entity aggregation (e.g. `query::shared_group_count`) need indexing strategy — called out in README.md Open question #7: "Cross-entity mask predicates. `t ∈ quest.eligible_acceptors`, `at_war(self.faction, f)`, etc. need careful index design to avoid O(N²)."
- **Proposed extension:** add two view annotations:
  - `@materialize(on_event: [<event_list>])` — view body becomes an event-fold updater on a stored field.
  - `@lazy` (default) — view body is a pure function evaluated at read time.
  - `@gpu_kernel` — eligible for GPU compilation (per spec.md §5 "GPU-amenable" list).

**Related stories:** 4 (cascade rules feed views), 26 (eagerly materialize hot views), 40 (flamegraph per-declaration).

---

#### Story AB.4: Declare a physics cascade rule
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Physics cascade rules are the third pillar. Mutations from events → state changes → emitted events. Cited directly in the user story: `Damage(d) → hp[d.target] -= d.amount → if hp ≤ 0: emit EntityDied`.

Concrete examples already written:

- `spec.md` § "Auction state machine" "Cascade rules":
  - `AuctionResolved{kind=Item, winner=W, payment=P}` → `TransferGold(W → seller, P.amount)` + `TransferItem(seller → W, item)`
  - `AuctionResolved{kind=Marriage, winner=W}` → `MarriageFormed(seller, W)` (updates spouse_id on both)
  - `AuctionResolved{kind=Charter, winner=W}` → `CharterGranted(settlement, W, terms)`
- `spec.md` § "How 'war is a quest' plays out" / "How 'marriage is a quest' plays out" — step-by-step cascade walkthroughs.
- `state.md` § "Membership" — `JoinGroup` / `LeaveGroup` events update the `memberships` Vec.
- `state.md` § "StructuralEvent" — voxel collapse emits events that cascade into more events (cascade_collapse in `damage_voxel`).

**Implementation walkthrough (add `PrayCompleted` cascade):**

1. Declare event (story 2): `event PrayCompleted { prayer: AgentId, shrine: StructureRef, faith_delta: f32 }`
2. Declare cascade rule:
   ```
   cascade PrayCompleted(e):
     agent_mut(e.prayer).data.needs.purpose += 5.0
     agent_mut(e.prayer).data.emotions.joy   += 0.2
     group_mut(religion_group_of(e.prayer)).treasury += DONATION
     emit FaithGained { agent: e.prayer, amount: e.faith_delta }
   ```
3. Compiler validates:
   - All read references are declared fields or `view::` calls (spec.md §3 "Validation: every observation field must reference declared entity field, view, or event source" — same rule for cascades).
   - No cycles (the user story explicitly calls out "cycle detection").
   - Ordering: cascade dependencies form a DAG; topological sort gives stable execution order (per README.md "Open" item #27 "Add a physics rule without breaking others" — wants compile-time validation).
4. Register with event dispatcher: compiler emits a switch on event kind → vector of cascade handlers.

**Gaps / open questions:**

- Cycle-detection algorithm not specified. Two events emitting each other is an easy mistake.
- Rule precedence for overrides (story 43 — modders want `Damage halves on undead`) — user's instinct is "just make the whole system modifiable, no need for layering." That pushes rule definition back to source-level rather than a layering mechanism. Implies cascades are DSL source declarations, not runtime-registered.
- **Proposed extension:** `cascade <EventName>(e: EventType) { ... }` block with compile-time validation of:
  - Acyclic event emission graph (error if `A emits B emits A` without explicit `@terminating_in(N)` bound)
  - All state reads/writes reference declared fields
  - Ordering constraints between cascade handlers for the same event (`@before(OtherHandler)`, `@after(OtherHandler)`)

**Related stories:** 2 (events), 6 (invariants after cascade), 27 (compose without breaking), 35 (debug cascade fan-out), 43 (override/layering — user deprioritized).

---

#### Story AB.5: Declare a spatial / non-spatial query
**Verdict:** PARTIAL (design intent settled, codegen complexity acknowledged)
**User's framing note:** "Good, but code gen for this has to be very smart to do it well."

**How the DSL supports this:**

Queries are first-class in the observation DSL. `spec.md` §3:

```
slots nearby_actors[K=12] from query::nearby_agents(self, radius=50)
                          sort_by distance(self, _) { ... }
slots known_actors[K=10]  from query::known_actors(self)
                          sort_by relevance(self, _) { ... }
slots known_groups[K=6]   from query::known_groups(self)
                          sort_by relevance(self, _) { ... }
```

Plus queries referenced by mask and cascade rules: `query::structures_at(pos, radius)`, `query::resources_in(region)`, `query::nearby_agents`, `query::known_actors`. Cited in README.md "Settled" — "Buildings and resources are derived views over the world's spatial data. `query::structures_at(pos, radius)` walks tiles + voxels to enumerate buildings; `query::resources_in(region)` walks voxel materials + harvest event log."

Two query classes already implicit in the design:

1. **Spatial** — parameterized by position + radius. Indexes: uniform grid, chunk grid, nav grid. See `state.md` for the voxel/tile/navgrid stack.
2. **Non-spatial / named** — `query::known_actors(self) = union(spouse_id, mentor_id, apprentice_id, group_leader_ids of my groups, top-K-grudges, top-K-friendships)` (per spec.md §2.1.5).

**Implementation walkthrough (add `query::fellow_religionists(self)`):**

1. Declare in DSL:
   ```
   query fellow_religionists(self: Agent) -> Vec<AgentId> {
     for g in self.memberships where g.kind == Religion:
       for m in g.members where m != self.id:
         yield m
   }
   ```
2. Compiler chooses execution strategy based on complexity hints:
   - If all membership groups are indexed by `kind` (settled GroupIndex from state.md), the compiler can O(k) scan the agent's religion groups and iterate their `members: Vec<AgentId>`.
   - If the resulting set is observation-visible, compiler allocates top-K storage per agent.
3. Annotations needed for smart codegen:
   - `@indexed_by(kind)` on Group — tells compiler Groups are indexed by `kind` so filtering is O(1) rather than scanning all Groups.
   - `@spatial` on a query that takes `(pos, radius)` — tells compiler to route to spatial index (uniform grid or chunk index, not HashMap linear scan).
   - `@top_k(K)` — tells compiler to allocate fixed-cap buffer and use partial sort, no Vec alloc (per story 30 "Zero per-tick allocations").

**Codegen complexity (user-flagged):**

This is the hardest compiler piece. Queries combine:

- Spatial indexing — multiple backends (KD-tree, uniform grid, chunk+axis-aligned voxel grid from `state.md`).
- Cross-entity joins — "nearby_agents" ⋈ "memberships" to find hostile-faction members in range.
- Top-K ranking with custom sort keys.
- GPU vs CPU dispatch — spatial queries might be GPU-amenable (spec.md §5 "Hybrid" list: "Spatial queries (nearby_actors, threats) — could be GPU with hash-grid texture; first implementation likely CPU with good cache layout").

The simplest viable strategy: hand-author a fixed library of queries (~10–15) implemented as handwritten Rust, then have the DSL surface reference them as opaque names. That aligns with README.md "Open" question #7 "Cross-entity mask predicates... need careful index design."

**Gaps / open questions:**

- No grammar for user-authored queries. Spec implies a fixed library.
- Indexing declarations aren't in the design — need them before queries can be compiled.
- README.md Open questions #5 ("Quest discovery mechanism") and #7 (cross-entity predicates) are directly downstream of this.
- **Proposed extension:** `query` block with annotations:
  - `@spatial(radius, kind=[Agent|Resource|Structure])` — routes to spatial index.
  - `@indexed_on(field)` — requires a sorted/hashed index on that field.
  - `@top_k(K)` — bounded output.
  - `@backend(cpu|gpu)` — override auto-selection.

**Related stories:** 3 (views are often queries), 8 (masks use queries), 15 (quest discovery), 26 (materialize hot views), 31 (scale to 200K — queries are the cost center).

---

#### Story AB.6: Declare an invariant
**Verdict:** PARTIAL (design acknowledges them; grammar/surface not yet specified)
**User's framing note:** "Great."

**How the DSL supports this:**

Invariants appear implicitly across the design. Several declarations IMPLY invariants without a formal block:

- **Single-spouse** — `state.md` AgentData row `spouse_id: Option<u32>` — type system expresses "at most one". The story asks for a richer version: "no agent has multiple Marriage memberships."
- **Bounded collections** — `state.md` § "Relationship" "Capped at 20 relationships per NPC" — invariant + eviction rule.
- **Append-only observation schema** — spec.md §4 "Append-only schema growth — when adding features, append; never reorder existing slots." This is an invariant on the schema, caught at compile time via CI guard.
- **Memberships must reference alive groups** — implicit from `state.md` `dissolved_tick: Option<u64>`. Story 61 ("Group dissolution with active quests") surfaces the same concern.
- **`party_member_ids` must reference alive agents** — story 62.

**Implementation walkthrough (add "at most one active Marriage membership"):**

1. Author in DSL:
   ```
   invariant no_bigamy(a: Agent) {
     count(g in a.memberships where g.kind == Family ∧ g.is_marriage) ≤ 1
   }
   ```
2. Compiler emits:
   - **Compile-time check where decidable:** e.g. if `spouse_id: Option<u32>` is the only path to Family-marriage membership, the type system enforces it.
   - **Runtime check elsewhere:** on every `JoinGroup` cascade touching a marriage-Family group, re-evaluate the invariant. Violation triggers a panic in debug, drop + warn in release (matching story 59 — "malformed action drops safely").
3. Proof-of-obligation: cascade handlers that could violate the invariant get annotated `@must_preserve(no_bigamy)`; compiler rejects cascades that write affected fields without the annotation.

**Gaps / open questions:**

- No `invariant` block in current DSL pseudocode (spec.md §3 covers observation/action/mask/reward/backend but not invariants).
- Compile-time vs runtime split is unclear — the user story explicitly says "compile time where possible, runtime otherwise."
- Invariants over event sequences (e.g. "no agent AcceptQuest-s after they're dead") need temporal expression, not just current-state.
- **Proposed extension:** add a top-level `invariant <name>(<scope>) { <predicate> }` block with compiler modes:
  - `@static` — enforced at compile time via type system / dataflow analysis. Compiler errors otherwise.
  - `@runtime` (default) — checked after every cascade that touches a field in the predicate's support.
  - `@debug_only` — runtime check, panic only in debug builds (too expensive for release).

**Related stories:** 15 (versioning — the schema is an invariant), 27 (compose without breaking), 59 (malformed action drop), 61 (group dissolution dangling refs), 62 (agent death mid-quest), 63 (simultaneous marriage proposal race).

---

#### Story AB.7: Add a new agentic action verb
**Verdict:** GAP (as written) → REFRAMED as SUPPORTED-in-principle, grammar gaps
**User's framing note:** Raw verb addition is dangerous — same concern as new entity types. Prefers action primitives that compose verbs.

**How the DSL supports this (reframed):**

The settled design already reduces ~110 proposed verbs to ~16 categorical actions. `spec.md` § "Total action vocabulary":

```
3 macro mechanisms:    PostQuest, AcceptQuest|JoinParty, Bid
~13 micro primitives:  movement(3) + combat(3) + resource(4) + construction(3) + social(2) + memory(1)
─────────────────────────
~16 categorical actions
```

The micro primitives (spec.md § "Micro primitives"):

- Movement: `MoveToward`, `Flee`, `Hold`
- Combat: `Attack`, `Cast`, `UseItem`
- Resource: `Harvest`, `Eat`, `Drink`, `Rest`
- Construction: `PlaceTile`, `PlaceVoxel`, `HarvestVoxel`
- Social atomic: `Converse`, `ShareStory`
- Memory atomic: `Remember`

The macro mechanisms carry all the high-level decision richness via parameter heads: `QuestType`, `PartyScope`, `QuestTarget`, `Reward`, `Payment` (spec.md §2.2).

**So "adding a new verb" collapses to one of:**

1. **A new mask predicate over existing primitives** — e.g. "Pray" = `Converse(target=shrine_agent) when self.memberships ∋ Group{kind=Religion}`. No runtime change, pure DSL.
2. **A new `QuestType`** — if the verb is a long-duration structured action (Heist, Conquest). Story 9 handles this. "Pray" as a multi-day pilgrimage is `QuestType::Pilgrimage` (already in spec.md's QuestType enum).
3. **A new `AuctionKind`** — for competitive/bidding semantics. Already has 7 kinds + extensible.
4. **A new `PartyScope`/`Reward`/`Payment` variant** — parameters that compose with existing verbs.
5. **A new micro primitive** — only when a genuinely new per-tick physical/social atom is needed. This is the only truly-dangerous path, and the set has already been reduced to 13. Adding one requires justification (an atomic act not expressible by existing micros).

**Implementation walkthrough (add "Pray" as a new action):**

Option 1 (preferred): compose from existing primitives.

1. Add mask:
   ```
   mask Pray(self) when self.memberships ∋ Group{kind=Religion}
                     ∧ distance(nearest(Structure{kind=Shrine})) < REACH
   ```
   Wait — but `Pray` isn't a `micro_kind` in spec.md §2.2. Two sub-options:
   - **1a. Treat as a parameterized `Converse`:** `Converse(target=nearest_shrine_agent)` gated by religion membership. Reward block handles the faith gain.
   - **1b. Add `Pray` as a 14th micro primitive** — requires extending `micro_kind` enum, bumping schema hash, retraining (see story 64). This is the "raw verb" path the user wants to avoid.

Option 2: promote to a `QuestType::Pilgrimage` if it's long-form (already in the enum).

The reframed takeaway: most new "verbs" are new **mask predicates + reward hooks** on existing primitives — zero schema impact.

**Gaps / open questions:**

- The DSL doesn't yet have syntax for "I want `Converse(target=shrine, intent=Pray)` to trigger a `PrayCompleted` event." Needs a mask-plus-handler pairing.
- Proposed extension: introduce `verb` as syntactic sugar:
  ```
  verb Pray(self, shrine: Structure) =
    action Converse(target=shrine.patron_agent_id)
    when  self.memberships ∋ Group{kind=Religion} ∧ distance(shrine) < REACH
    emit  PrayCompleted { prayer: self, shrine: shrine, faith_delta: 1.0 }
    reward +0.5
  ```
  This compiles to: (a) a mask entry on `Converse`, (b) a cascade from `Converse` emitting `PrayCompleted`, (c) a reward hook. No new `micro_kind` enum variant — same action tensor shape.
- When a genuinely-new primitive is needed, the schema hash (spec.md §4, story 64) bumps. Keep this path, but gate it behind an explicit "new action primitive" compiler flag so authors must acknowledge the ML consequence.

**Related stories:** 8 (mask predicates), 9 (QuestType), 20 (reward declaration), 64 (new ActionKind bumps schema hash).

---

#### Story AB.8: Write a mask predicate
**Verdict:** SUPPORTED
**User's framing note:** "Great."

**How the DSL supports this:**

Masks are load-bearing in the current design. `README.md` — "Settled" — "Role power = mask + cascade, not a smarter policy." `spec.md` §2.3 is the mask block:

```
mask {
  Attack(t)         when t.alive ∧ is_hostile(self, t) ∧ distance(t) < AGGRO_RANGE
  Eat               when self.inventory.food > MEAL_COST
  PostQuest{type=Conquest, party=Group(g), ...}
                    when g ∈ self.leader_groups
                        ∧ g.kind ∈ {Faction, Pack}
                        ∧ g.military_strength > 0
  ...
}
```

Compilation target (spec.md §2.3 tail):

- Per-tick boolean tensor `[N × NUM_ACTIONS]`
- Per-head for sampling efficiency: `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`
- GPU-amenable (spec.md §5 "GPU-amenable" list: "Per-head mask evaluation — boolean tensor computation per NPC").

**Implementation walkthrough (add `mask Pray(self) when self.memberships ∋ Group{kind=Religion}`):**

1. Author predicate in mask block:
   ```
   mask Pray(self) when self.memberships ∋ Group{kind=Religion}
                     ∧ distance(nearest(Structure{kind=Shrine})) < REACH
   ```
2. Compiler decomposes:
   - `self.memberships ∋ Group{kind=Religion}` — looks up `memberships[i].group_id` for each membership, reads that group's `kind`, checks equality. Cross-entity reference → CPU (spec.md §5 "GPU-hostile" list).
   - `distance(nearest(Structure{kind=Shrine}))` — spatial query on the tile/voxel derived-structures view. Spatial → hybrid (GPU with hash-grid texture, CPU first).
3. Per-head mask compilation:
   - `categorical_mask[i][Pray]` = `self[i].memberships ∋ Group{kind=Religion} ∧ distance(...) < REACH`
   - Emitted as an event-driven re-eval when either `memberships` or `pos` changes (or once per tick if queries are conservatively recomputed).

**Gaps / open questions:**

- Set-membership operator `∋` not formally in the grammar. Needs a decision: Unicode glyph (pretty but hard to type) or ASCII `contains`/`in`?
- Cross-entity mask compilation (README.md Open question #7) — the index strategy question is still open.
- Story 12 (hot-reload tuning) asks for constants (AGGRO_RANGE, REACH, MEAL_COST) to be externalized — the mask DSL should reference them by named constant, not inline literals.
- **Proposed extension:** formalize mask grammar as:
  ```
  mask <VerbName>(<bound_vars>) when <predicate> [emit <event>]
  ```
  with supported predicate ops: set membership (`contains`), quantifiers (`∀`/`∃`/`forall`/`exists`), bounded folds (`count`, `sum`, `max`, `min`), arithmetic comparison, view calls. No user-defined functions (too hard to compile to GPU boolean kernels).

**Related stories:** 7 (verbs are mostly new masks), 12 (tune constants), 34 (trace why mask false), 40 (flamegraph).

---

#### Story AB.9: Add a new QuestType
**Verdict:** SUPPORTED (reframed — QuestType is a label on data, not a primitive)
**User's framing note:** Same danger concern as new verbs.

**How the DSL supports this (reframed):**

`QuestType` is structured data, not code. From `spec.md` § "Required types":

```rust
enum QuestType {
    // physical / errand
    Hunt, Escort, Deliver, Explore, Defend, Gather, Rescue, Assassinate, Custom,
    // political
    Conquest, MutualDefense, Submit, Found, Charter, Diplomacy,
    // personal
    Marriage, Pilgrimage,
    // economic
    Service, Heist, Trade,
    // narrative
    FulfillProphecy,
}
```

The ONLY place `QuestType` is special-cased is the **cascade rule on completion**. Everything else — party scope, target, reward, deadline, acceptance, quest lifecycle — is polymorphic over the type. `spec.md` § "How 'war is a quest' plays out" step 7: "Quest completes → cascade emits `SettlementConquered`, `Spoils` distribution, reputation/legend updates." That cascade is the only per-type logic; the quest state machine itself is universal.

So "new QuestType" = "new label on the `QuestType` enum + one new cascade rule on `QuestCompleted{type=<New>}`."

The user's reframing is correct: QuestType is a compositional vocabulary (parameter to PostQuest), not a hardcoded primitive. The cascade rule is the only place the type is differentiated.

**Implementation walkthrough (add `QuestType::BuildMonument`):**

1. Extend the `QuestType` enum — one line (spec.md § "Required types").
2. Extend relevant `Reward` variants if the monument reward isn't already expressible (e.g. `Reward::Monument(LocationRef)` — or reuse `Reward::Spoils` with a Location target).
3. Add cascade rule:
   ```
   cascade QuestCompleted(q) when q.quest_type == BuildMonument:
     voxel_world.place_monument(q.target.location)
     chronicle.append("Monument {name} erected by {party}", ...)
     for agent in q.party_member_ids:
       agent.data.esteem += 20.0
       agent.data.fame_log += log(q.scale)
     emit MonumentBuilt { at: q.target.location, builders: q.party_member_ids }
   ```
4. Add mask predicate (if there are specific gating conditions):
   ```
   mask PostQuest{type=BuildMonument, party=Group(g), target=Location(loc), ...}
     when g ∈ self.leader_groups
         ∧ g.treasury > MONUMENT_COST
         ∧ tile_at(loc).is_constructable
   ```
5. Schema hash bumps if `QuestType` is observation-visible. It IS (spec.md §2.1.6 `active_quests[K=4]` observation slots → story 21 mentions `DeclareWar` as a rare macro action needing up-weighting → QuestType features thread into observation).

**Why this isn't dangerous:**

Adding a QuestType variant requires:
- One enum variant extension (structured, reviewable)
- One cascade rule (pure additive, doesn't touch existing rules — story 27 guarantee)
- One mask block (additive)
- No model retraining unless the enum bit width changes

Contrast with a new `micro_kind` (story 7's dangerous path) which requires retraining because the categorical action head's output dim changes.

**Gaps / open questions:**

- README.md "Settled" lists "Quest extensions" as Open (#2): `QuestType needs ~10 new variants (Conquest, MutualDefense, Submit, Found, Marriage, Service, Heist, Trade, FulfillProphecy). PartyScope enum needs adding. Reward and Payment enums need adding.` Author needs to land these before story 9 lands.
- Reward computation at completion (README.md Open #15: "Reward delivery on long quests — compute spoils at completion from current state vs commit at posting?") is a general QuestType design question, not specific to BuildMonument.
- **Proposed extension:** `quest_type` block in DSL:
  ```
  quest_type BuildMonument {
    target: Location,
    reward: Monument(LocationRef) | Gold(..),
    party: Group | Anyone,
    completion: cascade QuestCompleted -> ...,
    mask: ...,
  }
  ```
  compiler expands to: enum variant + cascade + mask entry + schema update.

**Related stories:** 2 (typed events — QuestCompleted), 4 (cascade), 7 (verbs via parameters), 20 (reward), 21 (rare actions oversampling), 27 (compose without breaking), 64 (schema hash on new ActionKind).

---

#### Story AB.10: Define a new GroupKind
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

`Group` is the universal social-collective primitive (README.md "Settled" — "Group — universal social-collective primitive (faction, family, guild, religion, party, hunting pack, settlement, court, cabal)"). `state.md` § "Group (universal)":

```
kind: GroupKind | Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other
```

And the per-kind shapes table (state.md § "Per-kind shapes"):

| Kind | Typical fields populated |
|---|---|
| `Faction` | `military_strength`, `standings`, `governance`, `tech_level`, recruitment_open |
| `Settlement` | `facilities`, `charter`, `stockpile`, `treasury`, ... |
| `Guild` | `treasury`, `recruitment_open`, `dues`, `eligibility_predicate`, ... |
| `Family` | `leader_id` (head of household), `members` derived from kin events |
| `Party` | `leader_id`, `founded_tick`, `dissolved_tick` ... |
| `Religion` | `charter` holding scripture, `eligibility_predicate`, leadership chain |
| `Pack` | `leader_id` is the alpha, `eligibility_predicate` keyed on `creature_type` |
| `Cabal` | `recruitment_open=false`, restrictive eligibility, secret standings |
| ... |

The `Group` struct itself carries the superset: `treasury`, `stockpile: Option<_>`, `facilities: Option<_>`, `military_strength: Option<_>`, `standings`, `leader_id: Option<_>`, etc. A new kind just populates a specific subset.

**Implementation walkthrough (add `GroupKind::Coven`):**

1. Extend `GroupKind` enum. One line.
2. Declare the Coven's typical-fields shape (for authoring clarity) and default values:
   ```
   group_kind Coven {
     eligibility_predicate: must_have_class("witch") ∧ ¬self.memberships.contains(Religion),
     recruitment_open: false,
     governance: Council,
     standings: default_to(Tense with Religion),
   }
   ```
3. Extend observation one-hot (spec.md §2.1.3 `group_kind_one_hot(8)` → must go to 9+, bump schema hash).
4. Extend the primary_group_kind and n_religious_groups-style summary atoms if Covens aren't religions per se.
5. No cascade rules needed — Group dynamics (JoinGroup, LeaveGroup, standings updates) are kind-agnostic.

**Gaps / open questions:**

- `EligibilityRef` is mentioned (state.md § Group "Recruitment & Eligibility" `eligibility_predicate: EligibilityRef`) but the format is open. Needs a predicate language.
- Multi-kind intersection: what if an agent could belong to both a Religion and a Coven and the rules conflict? Invariant machinery (story 6) should express this.
- Standings-default per kind-pair (e.g. Covens default-Tense with Religions) — is this data or DSL?
- **Proposed extension:** `group_kind` block:
  ```
  group_kind <Name> {
    eligibility_predicate: <predicate>,
    recruitment_open: <bool>,
    governance: <GovernanceKind>,
    default_standings: [ (<OtherKind>, <Standing>), ... ],
    required_fields: [treasury, military_strength, ...],
  }
  ```
  compiler validates: only required fields are populated; eligibility predicate is pure (no cascades).

**Related stories:** 5 (queries for "members of kind X"), 6 (invariants over multi-membership), 11 (creature_type eligibility), 41 (modder extensibility).

---

#### Story AB.11: Add a new creature_type
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

`creature_type` is the per-Agent discriminator. `state.md` § "Identity & Lifecycle" — `creature_type: CreatureType`. Per README.md "Settled" — "Agent — any agentic actor: humans, wolves, dragons, goblins. Same struct, distinguished by creature_type tag + personality/needs config + group memberships."

The hooks already in place:

- **Capabilities** — `state.md` AgentData row `capabilities: Capabilities` "derived from `creature_type`" (jump/climb/tunnel/fly/siege + can_speak / can_build / can_trade flags).
- **Predator/prey** — `spec.md` §2.1.4 "Hostility is derived per-pair via `... ∨ predator_prey(self.creature_type, other.creature_type)`".
- **Observation one-hot** — `spec.md` §2.1.1 "Creature/role (10): creature_type_one_hot(~8), ...".
- **Mask gates** — `spec.md` §2.3 `Talk(t) when t.creature_type.can_speak ∧ distance(t) < SOCIAL_RANGE`, and `PostQuest{type=Marriage, ...} when self.creature_type.can_marry`.
- **Spawn / template** — `state.md` AgentData row `creature_type: CreatureType ... | constructor (template.kind), monster spawn`.
- **Per-creature personality / hunger drives** — `state.md` § "Personality" "Set at spawn, drifts via events" — defaults can vary by `creature_type`.

**Implementation walkthrough (add `CreatureType::Centaur`):**

1. Extend `CreatureType` enum.
2. Declare capability defaults:
   ```
   creature_type Centaur {
     capabilities: { can_speak: true, can_build: false, can_trade: true, fly: false, jump: 2.0, ... },
     default_personality: { risk: 0.6, social: 0.7, ambition: 0.4, compassion: 0.3, curiosity: 0.5 },
     default_needs: { hunger_decay: 1.2, ... },
     hunger_drives: [ GrazeGrass, HuntSmallAnimal ],
     default_memberships: [ Group{kind=Pack} ],
     predator_prey: { prey_of: [], preys_on: [Rabbit, Deer] },
   }
   ```
3. Extend observation `creature_type_one_hot(~8)` → add bit, bump schema hash.
4. Spawn templates (`assets/hero_templates/` or a creature equivalent) reference the new type.

**Gaps / open questions:**

- `Capabilities` struct fields not fully listed in state.md (mentions "Jump/climb/tunnel/fly/siege flags + can_speak / can_build / can_trade"). Needs formalizing.
- Hunger drives are a gameplay dimension not formally listed in state.md § Needs — they're per-creature. Probably belongs on `creature_type` declaration block.
- Predator/prey table — the `predator_prey(self, other)` view needs a lookup table. Either authored per-type or globally as a matrix.
- **Proposed extension:** `creature_type <Name> { capabilities, defaults, drives, predator_prey }` block. Compiler validates that the one-hot width bumps are handled (schema hash flag).

**Related stories:** 1 (entity abstraction — this is the primary alias path), 8 (mask uses creature_type), 10 (GroupKind eligibility keyed on creature_type e.g. Pack), 48 (monster defend dens — creature_type defaults).

---

#### Story AB.12: Tune a mask without code change
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

The design already externalises tunables. Mask predicates in spec.md §2.3 reference named constants: `AGGRO_RANGE`, `MEAL_COST`, `REACH`, `HARVEST_RANGE`, `SOCIAL_RANGE`, `HOSTILE_THRESH`, `MIN_TRUST`, `THRESH` (population).

Those constants should live in a config file, not in DSL source. The settled design has two precedents for this pattern:

- **Backend weights** — `backend Neural { weights: "npc_v3.bin", h_dim: 256 }` (spec.md §3 — bottom). Weights are an external artifact hot-swappable (story 23).
- **Schema versioning** — `@since(v=1.1)` annotations (spec.md §4) separate declaration from metadata, allowing migration.

**Implementation walkthrough (change AGGRO_RANGE from 50 to 60):**

1. Author mask:
   ```
   mask Attack(t) when t.alive ∧ is_hostile(self, t) ∧ distance(t) < AGGRO_RANGE
   ```
2. Declare constant:
   ```
   const AGGRO_RANGE: f32 = 50.0 @tunable
   ```
3. Compiler emits a runtime lookup (not a compile-time literal) for constants tagged `@tunable`, so the value is read from a config table at mask evaluation time.
4. Hot-reload: a file watcher on `config/tunables.toml` triggers a `reload_tunables()` call that rewrites the table. Next tick sees new value. No retraining needed — constants don't affect tensor shapes.

**For story 44 (hot-reload mod changes) generality:**

Hot-reload works cleanly for:
- `@tunable` constants (scalar / small fixed-shape)
- Reward coefficients (spec.md §2.5 — `× 0.1`, `× 5`, `+1.0`)
- Mask predicate TREE STRUCTURE? Only if no new action verbs or slots appear — otherwise model doesn't know what to do with the new output dim.

Hot-reload does NOT work for:
- Observation schema changes (breaks model — schema hash check hard-rejects — spec.md §4)
- Action vocabulary changes (same)
- Reward reference changes that add new event subscriptions without the event declaration
- Views with `@materialize` (need to re-run the event fold)

**Gaps / open questions:**

- No `const @tunable` syntax in the current DSL pseudocode — this is a small additive.
- Config format (TOML? JSON? the mod-overrides case needs to interact with story 42 civilizations which are data).
- Validation — what happens if a tunable is set out of a sensible range (AGGRO_RANGE = -5)? Probably clamp or error in loader.
- **Proposed extension:**
  ```
  const <Name>: <Type> = <default> @tunable [@range(lo, hi)]
  ```
  compiler emits a runtime-backed table lookup and a reload hook that validates against `@range` constraints.

**Related stories:** 20 (reward coefficients are tunables), 23 (hot-swap), 42 (civilization data), 44 (hot-reload mods).

---

### Cross-cutting observations

**The three-primitive taxonomy holds up.** Entity = Agent|Item|Group. Event-sourced state mutation. Mask + cascade = role differentiation. The entire "dangerous new primitive" axis (stories 1, 7, 9) reduces to additive vocabulary extensions (creature_type, GroupKind, QuestType) that compose with the existing primitives via enum variants + cascade rules. New `micro_kind` is the only path that forces schema-hash churn.

**Schema hash (spec.md §4) is the contract boundary.** Changes that bump the hash: new micro_kind variant, new observation field, new macro-action parameter width, new creature_type (one-hot bit), new GroupKind (one-hot bit). Changes that DON'T bump: mask predicate changes, new QuestType (if reward width unchanged), new cascade rules, new tunable values, new views (unless materialized and observation-visible).

**The biggest unresolved DSL surface gaps for stories A/B:**

1. **`template` / alias block** (story 1 reframe) — no grammar.
2. **`event` declaration block + annotations** (story 2) — mentioned, not formalized.
3. **`view` materialization annotations** (story 3) — `@materialize`, `@lazy`, `@gpu_kernel` not specified.
4. **`cascade` block with cycle detection + ordering** (story 4) — implied, not formal.
5. **`query` block with indexing hints** (story 5) — the compiler complexity sink.
6. **`invariant` block with static/runtime split** (story 6) — missing.
7. **`verb` as composition-sugar** (story 7 reframe) — not proposed yet; closes the "dangerous" gap.
8. **`group_kind` / `creature_type` / `quest_type` declarative blocks** (stories 9, 10, 11) — implicit; elevating them to first-class declarations gives the "new kind = data" framing teeth.
9. **`const @tunable` with `@range`** (story 12) — small additive.

These are the next batch of additions for `spec.md` (README.md open iteration #3).

---

## Batch C — Observation Schema (ML Engineer)

Stories 13–18 from *(archived)*. Each verdict is relative to the
current state of `spec.md` (§2.1 observation, §3 DSL surface,
§4 versioning). Where the proposal as written conflicts with user intent on
story 15 (no backwards compat), this document overrides the proposal.

---

#### Story C.13: Inspect schema as JSON

**Verdict:** PARTIAL

**User's framing note:** "Essential." ML pipelines need a machine-readable
description of the observation tensor to build correctly-shaped models.

**How the DSL supports this:**
- `spec.md` §3 lists "JSON schema dump for ML pipeline tooling"
  as a compiler output, alongside "Packed struct with named field offsets
  (debug tools decode bytes back to fields)." That bullet is the hook story 13
  depends on.
- §2.1 decomposes the ~1655-float tensor into nameable groups (self atomic,
  self contextual, group memberships, spatial slots, non-spatial slots, context
  blocks). Those groupings are what the JSON schema must expose.
- §3's DSL pseudocode gives every field a name (`self.hp_pct`,
  `self.psychological`, `nearby_actors[i].relationship_valence`, ...). The
  named form maps 1:1 onto JSON schema entries.

**Implementation walkthrough:**
A workable JSON schema shape — not specified in the proposal but consistent
with the pseudocode — is:

```json
{
  "schema_hash": "sha256:...",
  "total_floats": 1655,
  "groups": [
    {
      "name": "self.atomic",
      "offset": 0,
      "size": 55,
      "fields": [
        { "name": "hp_pct",        "offset": 0,  "dtype": "f32",
          "norm": { "kind": "identity", "range": [0,1] } },
        { "name": "max_hp_log",    "offset": 1,  "dtype": "f32",
          "norm": { "kind": "log1p", "scale": 1.0 } },
        ...
      ]
    },
    {
      "name": "spatial.nearby_actors",
      "offset": 295,
      "size": 360,
      "slot_count": 12,
      "slot_size": 30,
      "slot_fields": [
        { "name": "relative_pos_x", "offset": 0, "dtype": "f32",
          "norm": { "kind": "scale", "denom": 50 } },
        { "name": "creature_type_one_hot", "offset": 2, "dtype": "f32[8]",
          "norm": { "kind": "one_hot", "vocab_size": 8 } },
        ...
      ]
    }
  ],
  "one_hot_vocabularies": {
    "creature_type":  ["Human","Elf","Dwarf","Wolf","Goblin","Dragon",...],
    "group_kind":     ["Faction","Family","Guild","Religion","Party","Pack",
                       "Settlement","Other"],
    "relationship_kind": ["Spouse","Kin","Mentor","Apprentice","Friend",
                          "Rival","SwornEnemy","Stranger"],
    "event_type":     [...]
  },
  "slot_repeats": {
    "nearby_actors": { "k": 12, "sort_key": "distance" },
    "known_actors":  { "k": 10, "sort_key": "relevance" }
  },
  "action_vocabulary": {
    "macro_kind": ["NoOp","PostQuest","AcceptQuest","Bid"],
    "micro_kind": ["Hold","MoveToward","Flee","Attack","Cast","UseItem",
                   "Harvest","Eat","Drink","Rest","PlaceTile","PlaceVoxel",
                   "HarvestVoxel","Converse","ShareStory"]
  }
}
```

Tools consume this by:
1. Loading `schema.json` at PyTorch model build time.
2. Declaring input shape `[batch, total_floats]` and stashing `schema_hash`
   into the checkpoint's metadata.
3. Using `groups[i].offset/size` to slice the tensor for grouped encoders
   (e.g., one sub-MLP for self atomic, a transformer over spatial slots).
4. Exposing `one_hot_vocabularies` to the replay-buffer decoder (story 17).

The stability contract: once `schema_hash` is emitted, it is a
**content-addressed fingerprint** of the observation layout. Changing *any*
field's offset, dtype, or normalization changes the hash. Tools must refuse
to load a checkpoint whose hash disagrees with the current DSL build
(story 15).

**Gaps / open questions:**
- The proposal does not specify where the normalization constants live. The
  pseudocode hints at `log1p`, `/50`, `/100` scaling in individual atoms, but
  never commits to a canonical set of normalization kinds. A `norm` taxonomy
  must be fixed early (identity / log1p / scale / clamp / one_hot / bitset)
  because all three of story 13, 14, and 17 consume it.
- Action space, mask shape, and reward declaration are also part of the
  training contract. §6 open question 17 already flags this: "Should the
  schema hash cover the action vocabulary?" — if yes, the JSON schema has to
  emit action + reward sections too. Recommend yes; checkpoint compatibility
  is all-or-nothing.
- The DSL compiler that produces this JSON does not yet exist; §3 only
  sketches the surface. The schema emitter is the first concrete output the
  compiler owes the ML pipeline.

**Related stories:** 14 (append-only growth must preserve the emitted shape),
15 (hash is the mismatch detector), 17 (decoder CLI reads this file).

---

#### Story C.14: Add a new observation feature

**Verdict:** SUPPORTED

**User's framing note:** "Essential." Day-to-day ML work. The engineer edits
the DSL to add `self.fame_log`, rebuilds, and the model gets the new signal.

**How the DSL supports this:**
- §4.3 "Append-only schema growth": "when adding features, append; never
  reorder existing slots. Removing a feature is a breaking change requiring
  model retraining."
- `self.fame_log` is already in §2.1.2's Self contextual list — the block
  exists. Adding a brand-new scalar outside the existing blocks is the
  interesting case.
- §3 pseudocode shows how an atom is declared:
  ```
  self.fame_log = log1p(view::fame(self))
  ```
  Or inside a block:
  ```
  block self.social_standing {
    from view::fame(self)       as f32 via log1p
    from view::reputation(self) as f32
  }
  ```

**Implementation walkthrough:**

The append-only workflow, assuming `view::fame` already exists (if not, it
gets declared in a `view` block and its source events/fields must already be
in `state.md`):

1. **Edit DSL.** Add `atom fame_log = log1p(view::fame(self))` to the relevant
   observation block. Source references must resolve to declared entity
   fields, views, or events — §3 explicitly states this validation rule.
2. **Rebuild.** The DSL compiler:
   - Re-orders feature offsets so the new atom is appended after all existing
     ones in the same block. "Appended" means *within the block*; the
     aggregate tensor has per-block contiguous regions, so a new atom in
     `self.contextual` lives at the end of that region, not at position 1655.
   - Recomputes the schema hash. Any append changes the hash.
   - Re-emits `schema.json` with the new field and its `offset/size/norm`.
   - Regenerates the Rust packing kernel (new line that writes the value).
   - Regenerates the GPU packing kernel if the block is `@gpu_kernel`.
3. **Bump the checkpoint.** §4.4 "CI guard — observation schema changes that
   touch existing slots fail CI unless accompanied by a model checkpoint
   bump." An append-only change must still bump the hash, but does not edit
   any existing offset. CI check: old offsets + types unchanged under the new
   schema's field list.
4. **Train.** The ML engineer re-initializes the input layer (or widens it to
   match new total_floats) and trains. The append-only rule means old
   trajectories from pre-change runs can be re-padded with zeros at the new
   tail positions *if and only if the engineer opts in* — but per story 15,
   we are not building automatic padded-zero migration (see below).

**What changes for the model:**
- `total_floats` increases by 1 (or however many floats the new feature
  contributes).
- The input layer widens. For a linear input projection this is a trivial
  re-init; for a grouped encoder where the new atom joins an existing block,
  the block's sub-MLP widens.
- All downstream shapes are stable because the new feature is appended.

**Gaps / open questions:**
- Per-slot append is trickier than per-atom append. If we add a field to
  `nearby_actors` slots (currently 30 floats × 12 slots), every slot grows.
  That's still append-only (the new slot field goes at offset 30 within the
  slot), but the total footprint change is `K × new_bytes`, not
  `1 × new_bytes`. Compiler must handle slot-internal appends identically.
- There is no story for **removing** a feature. §4.3 says removal is a
  breaking change. We should explicitly fail compilation on an attempted
  removal unless a `@deprecated` path was declared (and even then, story 15's
  no-backcompat stance means we just retrain from scratch).
- View/event references introduce an indirection: adding `self.fame_log`
  requires `view::fame` to be backed by a concrete `state.md` field
  (currently `reputation_summary` is listed there but `fame_log` is not
  explicitly — it needs a source field or a view over `legendary_deeds`
  events).

**Related stories:** 13 (new field appears in JSON schema), 15 (the schema
hash bump is what makes the old checkpoint refuse to load), 19 (bootstrap
trajectories recorded before the append cannot be replayed verbatim against
the new model).

---

#### Story C.15: Schema versioning

**Verdict:** GAP (the proposal's answer is wrong for this project)

**User's framing note:** **Re-scoped.** The proposal's §4 prescribes three
versioning tools (schema hash, `@since` annotations, padded-zero migration
tables). The user explicitly rejects two of those: "We do not want to have v1
and v2 and v3 in the same codebase. Git can be used to store versions, it is
nothing but waste to support backwards compatibility in a solo project like
this."

The correct design is **fail loud; no coexisting versions; no padded-zero
migration.** The schema hash stays — as a fingerprint, not as a migration
hinge.

**How the DSL supports this (after re-scoping):**

Drop from the proposal:
- `@since(v=1.1)` field annotations — implies two schemas coexist. Delete.
- "Migration tables" in §4.2 — implies automatic padded-zero fill. Delete.
- "rejected-with-explanation when the model would need fields that no longer
  exist" — only half wrong: the rejection is right, the prose implying the
  other branch (zero-pad when fields were *added*) is wrong. Replace with
  unconditional rejection.

Keep from the proposal:
- **Schema hash.** SHA256 over the canonical (sorted/normalized) observation
  layout + normalization constants + action vocabulary + reward declaration.
  Burned into every checkpoint at save time.
- **Append-only growth.** Same mechanical rule as story 14.
- **CI guard.** Breaking changes (reorder, remove, type change, norm change)
  must not merge without a corresponding checkpoint bump.

**Implementation walkthrough:**

The error UX when a mismatched checkpoint meets a rebuilt DSL:

```
error: policy checkpoint schema mismatch
  checkpoint: generated/npc_v3.bin (trained 2026-04-10)
  checkpoint schema_hash: sha256:a1b2c3...7890
  current DSL schema_hash: sha256:e4f5g6...2345
  diff:
    + appended: self.fame_log  (offset 1655, size 1, norm log1p)
    + appended: nearby_actors[].has_quest_conflict
                (slot-internal offset 30, size 1, norm identity)
  action: retrain from current DSL, or git-checkout the commit whose
          schema_hash matches the checkpoint.
```

Key UX properties:
1. **Hard fail on mismatch.** No auto-padding, no opt-in flag, no warning
   mode. The loader refuses and exits with a nonzero code.
2. **Explain the drift.** The error prints a diff of what changed (requires
   keeping the checkpoint's schema JSON alongside its weights, which is
   cheap — a few KB per checkpoint).
3. **Suggest git.** The canonical remediation is either retrain or checkout
   an older commit. Git is the version control; the codebase is not.
4. **Single live schema.** The compiler only emits one schema at a time.
   There is no "v1" or "v2" path in the Rust code. If two developers (or
   two branches) disagree, their checkpoints are mutually incompatible and
   that is fine — they live on different branches.

**Is the schema hash still valuable as a fingerprint?** Yes, unambiguously.
Without it we have no way to detect silent corruption. A checkpoint without a
hash cannot be safely loaded — if someone trains on schema A, commits schema
B locally without noticing, and loads the checkpoint, the tensor layout is
garbage but the model happily produces actions. The hash turns this
silent-corruption case into the loud-error case.

The hash also covers:
- **Action vocabulary.** Per §6 open question 17 — yes, it should. Adding a
  new `ActionKind` enum variant changes the output head's shape; the model is
  incompatible even if the observation didn't move.
- **Reward DSL (optional).** Reward changes don't break the trained model
  mechanically, but they break dataset-level comparability (replay buffer
  rewards were computed under a different rule). Recommend hashing reward
  too, but exposing it as a separate `reward_hash` so observation-only
  compatibility checks can ignore reward drift.

**What FAIL LOUD prevents that migration tables would permit:**
- A field gets renamed. Under padded-zero migration the old slot reads as
  "this used to be the value, now it's zero" and the model wastes capacity
  on a dead feature. Under fail-loud, we retrain with the renamed field
  present from the start.
- A one-hot vocabulary grows a category. Padded-zero would leave the new
  category permanently unselectable in old checkpoints. Fail-loud forces us
  to notice.
- A normalization constant changes (someone tightens the clamp range). Under
  migration the model's learned weights assume the old scale. Fail-loud
  catches it.

**Gaps / open questions:**
- Where does the schema hash physically live in a checkpoint? Proposal does
  not specify. Minimum viable: a `meta` dict in the checkpoint (`.bin` or
  `.json` sibling file) with `schema_hash`, `action_vocab_hash`, commit SHA,
  and training date.
- Do we hash the normalization constants or just the structure? **Must hash
  the constants.** Changing `log1p` to `scale(1/100)` on the same field is a
  silent model-breaker otherwise.
- How does the hash interact with the `one_hot_vocabularies` section of
  story 13's JSON schema? Vocabularies are load-bearing; any reorder or
  insert changes the hash.
- `[OPEN]` from proposal §4: action-vocabulary hashing scope. Recommend
  resolve to "yes, one combined hash; observation + action + reward as
  separate component hashes too for more granular diffing."

**Related stories:** 13 (JSON schema is what the hash is computed over), 14
(every append bumps it), 19 (bootstrap trajectories are now hash-tagged and
rejected against newer schemas).

---

#### Story C.16: Per-tick training data emission

**Verdict:** SUPPORTED

**User's framing note:** "Great." Standard RL/IL plumbing; the only question
is tuple format and buffer file shape.

**How the DSL supports this:**
- §2.5 names the mechanism: "Logging hooks for (observation, action, reward)
  tuples → training dataset emission" as a reward-compiler output.
- §2.1 observation packing and §2.2 action emission already run every tick;
  adding a log sink is a tee, not new computation.
- `SimState` tick loop (existing `step()` harness in `src/ai/core/simulation.rs`
  per CLAUDE.md) already produces events each tick; the terminal flag is
  derivable from existing `SimEvent::EntityDied` (agent-terminal) or the
  scenario-end event (episode-terminal).

**Implementation walkthrough:**

**Per-tick tuple, per agent:**

```
TrainingTuple {
  tick:          u64
  agent_id:      u32
  episode_id:    u64        // for GAE / return computation
  schema_hash:   [u8; 32]   // fingerprint (story 15)

  observation:   [f32; OBS_DIM]    // packed, exactly what the policy saw
  action:        PackedAction       // multi-head: macro_kind, micro_kind,
                                    // target_slot_idx, pos_delta, magnitude,
                                    // quest_type, party_scope, reward_type,
                                    // payment_type
  mask_snapshot: [bool; NUM_HEADS × MAX_CHOICES]   // optional, expensive

  log_prob:      f32         // log π(a | s), summed across heads
  value_est:     f32         // V(s) from critic (if actor-critic)
  reward:        f32         // from the reward DSL this tick
  terminal_flag: bool        // agent died or episode ended this tick
}
```

Sizing: `1655 × 4 = 6620` observation bytes + ~80 bytes other fields ≈ 6.7 KB
per agent per tick. At 20K agents × 10 Hz sim = 1.3 GB/s raw, unworkable as
flat JSONL. See volume estimates below.

**Buffer file format:**

Recommend **length-prefixed flatbuffers or msgpack frames** written to a
rolling file (`buffer/ep_00042.bin`):

- Fixed header per file: schema_hash, action_vocab_hash, episode_id,
  tick_start, sim version.
- Body: concatenated frames, one per tick, each frame = length-prefixed
  batch of tuples for all live agents that tick.
- Optional float16 compression of the observation tensor for 2× savings;
  keep action/log_prob/reward in float32.

Do **not** use JSONL for the observation (O(7 KB) stringified is absurd).
Reserve JSONL for debug single-agent dumps (story 17).

**Replay buffer integration:**

The file format already matches what a standard PyTorch `IterableDataset`
wants: open file, seek to frame, yield a batch of tuples. Two buffer-shape
options:

- **Episode buffer** (one file per episode): trivial for on-policy
  algorithms (REINFORCE, PPO). Re-read to compute GAE with known
  terminals. What we'd start with.
- **Prioritized replay buffer**: sharded by `episode_id`, with a priority
  index over macro actions (rare events under §2.2 and the "prioritized
  replay buffer with rare-event up-weighting" mitigation in §7). Needed
  later for macro-action credit assignment; defer.

**Per-tick volume estimates:**

| Scale                   | Agents | Tuple bytes | Per tick | Per second (10 Hz) |
|-------------------------|--------|-------------|----------|--------------------|
| Combat scenario         | 20     | 6700        | 134 KB   | 1.3 MB/s           |
| Small world sim         | 500    | 6700        | 3.3 MB   | 33 MB/s            |
| Full world sim target   | 20000  | 6700        | 134 MB   | 1.3 GB/s           |

The full-world case cannot write flat uncompressed tuples in real time. Need:
- Downsample (log every Nth tick, or only on non-NoOp action).
- Compress observations to float16 → 670 MB/s, still a lot.
- Separate the observation from the action; the action stream is O(100
  bytes) per agent per tick. Log actions always, observations only on
  ticks flagged "interesting" (non-NoOp macro, or reward magnitude >
  threshold, or 1-in-N uniform sample).

**Gaps / open questions:**
- "Terminal flag" has two meanings: agent death vs. episode end. Proposal
  doesn't distinguish. Buffer tuple should carry both as separate booleans.
- Mask snapshot is expensive; the number of heads and slots means ~100s of
  bools per tuple. Recommend omit by default, add a `--log-masks` flag for
  debugging why a policy picked what it picked.
- Value estimate only exists if the backend is actor-critic; needs to be
  `Option<f32>` or sentinel NaN.
- Log-prob must be the sum across all heads the action actually used (macro,
  micro, target, continuous) — compiler must emit a helper that returns the
  summed log-prob given the PackedAction.

**Related stories:** 13 (tuple shape is derived from the observation schema),
19 (bootstrap trajectories reuse this exact format), 20 (reward DSL is the
tuple's reward producer).

---

#### Story C.17: Single agent observation decoder CLI

**Verdict:** SUPPORTED

**User's framing note:** "Great." Debug tool for "what did the model see?"
Essential when the model does something surprising.

**How the DSL supports this:**
- §3 explicitly outputs "Packed struct with named field offsets (debug tools
  decode bytes back to fields)." That is exactly story 17's dependency.
- Story 13's JSON schema contains the offset table — it *is* the named-offset
  table story 17 consumes.
- Story 16's buffer format stores the packed observation tensor alongside
  `tick` and `agent_id`. The decoder just correlates the two.

**Implementation walkthrough:**

CLI shape, consistent with existing `xtask` subcommands:

```
cargo run --bin xtask -- obs decode \
    --buffer generated/buffer/ep_00042.bin \
    --agent 17 \
    --tick 1200 \
    [--schema generated/schema.json]   # default: current DSL schema
    [--format pretty | json | flat]
    [--slice self.atomic]              # restrict to a group
    [--show-zeros]                     # default: suppress zero slot rows
```

Example output (`pretty`):

```
tick=1200 agent=17 episode=42 schema_hash=sha256:e4f5g6...2345

self.atomic (55 floats, offset 0)
  hp_pct              = 0.73
  max_hp_log          = 5.30   (raw max_hp ≈ 200)
  shield_pct          = 0.00
  ...

self.contextual (120 floats, offset 55)
  aspiration.need_vector = [hunger=0.2 safety=0.8 ...]
  ...

spatial.nearby_actors[12, 30 each, offset 295]
  slot 0  exists=1 dist_rank=0
    relative_pos       = (-3.2, +1.1)
    creature_type      = Wolf [one_hot_arg=3]
    hp_pct             = 0.45
    relationship_valence = -0.8  (hostile)
    n_shared_groups    = 0
  slot 1  exists=1 dist_rank=1
    ...
  slot 2  exists=0 (suppressed; rerun with --show-zeros to see)

action taken
  macro_kind = NoOp
  micro_kind = Attack
  target     = spatial.nearby_actors[0]  (the Wolf above)
  log_prob   = -0.34
  reward     = -0.50
```

Mechanics:
1. Load `schema.json` (from flag or by deriving from the DSL's current
   schema_hash).
2. Compare `schema_hash` in the buffer header to the schema's hash. **On
   mismatch, fail loud** (story 15) — print both hashes and the
   git-remediation hint; exit nonzero. Do not attempt to decode.
3. Seek to `tick` frame, find tuple with matching `agent_id`.
4. For each `group` in the schema, slice `observation[offset .. offset+size]`,
   then apply the group's field layout. For slot arrays, iterate `slot_count`
   times at stride `slot_size`.
5. Invert the `norm` spec to recover raw values when possible (log1p →
   expm1, scale → multiply; identity and one_hot pass through).
6. Render one-hot fields as their winning category name using
   `one_hot_vocabularies`.

The tool is ~300 lines of Rust; the payoff is every future "why did the
model do X?" investigation uses it.

**Gaps / open questions:**
- Need a companion `obs diff --tick A --tick B --agent 17` to see what
  changed frame-to-frame. Trivial once the single-tick decoder works.
- Need a companion `obs grep --predicate "hp_pct < 0.2"` to find all
  low-HP snapshots. Story 18's probes idea (below) generalizes this.
- The buffer file format from story 16 must keep tick + agent_id indexable
  without a full scan. Minimum viable: a sidecar `buffer/ep_00042.idx` with
  `(tick, agent_id) → byte_offset`.
- Handling of `known_actors[K]` slots that backref `nearby_actors` (§2.1.5):
  the decoder should resolve `in_nearby_actors_slot_idx` and print the
  linked slot's contents inline.

**Related stories:** 13 (schema source), 15 (hash check), 16 (buffer source),
18 (probes are a batched generalization of this tool).

---

#### Story C.18: Compare two policy backends + probes on a known dataset

**Verdict:** PARTIAL (comparison: supported; probes: genuine GAP, needs new
DSL surface)

**User's framing note:** "Great." Plus the extension: "I would like a way of
evaluating probes on a known dataset as well." The probes piece is the more
interesting half and is not in the current proposal.

**How the DSL supports this:**

The comparison half is already expressible:
- §2.4 defines `PolicyBackend` as a trait with one method
  (`evaluate_batch(obs, mask) -> ActionBatch`). The trait makes Utility vs
  Neural vs LLM structurally interchangeable.
- Determinism contract (CLAUDE.md: "All simulation randomness flows through
  `SimState.rng_state`") guarantees that a seeded scenario with fixed agent
  spawns produces identical observations across backend swaps. Any decision
  divergence is purely from the policies.
- Story 17's decoder handles the single-agent rendering; an A/B diff is the
  same tool called twice then diffed.

CLI shape (comparison):

```
cargo run --bin xtask -- policy compare \
    --scenario scenarios/basic_4v4.toml \
    --seed 42 \
    --backend-a utility \
    --backend-b neural:generated/npc_v3.bin \
    --output generated/compare_v3.jsonl
```

Per-tick output row: `{tick, agent_id, action_a, action_b, agree: bool,
rationale_a, rationale_b}`. Summary: action-agreement rate, distribution of
disagreements by macro/micro head, cases where one backend moved while the
other held, etc.

**Probes — the real work:**

A **probe** is a small, hand-authored scenario plus an expected behavioral
assertion. Not "did the model pick action X at tick Y" (brittle), but "does
the distribution of decisions on this scenario satisfy the stated
property?" Probes live in-repo as regression fixtures and run every
checkpoint eval.

Proposed probe DSL surface:

```
probe LowHpFlees {
  scenario "probes/low_hp_1v1.toml"
  seed 42
  ticks 200
  backend neural:generated/npc_v3.bin

  // A behavioral claim the policy must satisfy.
  assert {
    // Over all ticks where agent 0 has hp_pct < 0.3,
    // the chosen action is Flee or MoveAway > 80% of the time.
    pr[ action.micro_kind in {Flee, MoveToward_away_from_threat}
      | self.hp_pct < 0.3 ]
      >= 0.80
  }

  tolerance 0.02  // allow 2% absolute slack for stochastic policies
}

probe LeaderPostsQuestsUnderThreat {
  scenario "probes/threatened_settlement.toml"
  backend neural:generated/npc_v3.bin

  assert {
    count[ action.macro_kind == PostQuest
           ∧ action.quest_type == Defense
         | self.is_leader_anywhere == 1
           ∧ settlement.threat_level > 0.6 ]
    >= 1  // at least one defense quest gets posted
  }
}

probe NoSpouseAttacks {
  scenario "probes/random_family_5agents.toml"
  backend neural:generated/npc_v3.bin

  assert {
    pr[ action.micro_kind == Attack
      | action.target.is_spouse == 1 ]
    == 0.0
  }
}
```

**Where probes live:** `probes/` at repo root (sibling to `scenarios/`),
with probe `.probe` files referencing seed scenarios in `probes/scenarios/`.
Keep them small (1–10 agents, short tick budgets) so the full suite runs in
seconds.

**How probes integrate with the training pipeline:**

1. **As regression tests.** Every `generated/npc_v*.bin` checkpoint runs the
   probe suite via `xtask policy probe generated/npc_v3.bin probes/`.
   Checkpoint fails to be "released" if any essential probe fails.
2. **As eval metrics.** The probe pass rate is a scalar per checkpoint,
   tracked alongside episode return. "94.2% probe pass, 0.73 avg return."
3. **As training-time signal.** Probes can gate curriculum steps ("don't
   advance curriculum until AttackSpouse probe passes"). Longer-term, a
   probe's `assert` expression can be converted to an auxiliary loss term
   (soft version of the pass condition).
4. **As a comparison target.** For story 18's core compare: run both
   backends against the same probe suite, report which probes each passes.
   This is the quantitative version of "side-by-side decision diff."

**Probe assertion grammar (minimum viable):**

```
assert_expr := count_expr | prob_expr | mean_expr
count_expr  := "count" "[" filter_expr "]" comparator scalar
prob_expr   := "pr"    "[" action_expr "|" filter_expr "]" comparator prob
mean_expr   := "mean"  "[" scalar_expr "|" filter_expr "]" comparator scalar
filter_expr := boolean over observation fields + action fields + derived
               facts (settlement.*, relationship(self, target).*, ...)
```

All names on the left side of `|` are **action fields** from story 16's
tuple; all names on the right side are **observation fields** resolved via
story 13's schema plus a few cross-reference helpers (target_of_action,
relationship, settlement_of).

**Why this is the right extension of story 18:**

- A raw tick-by-tick diff between two backends is high-noise; 50% disagreement
  doesn't tell you if either is wrong.
- Named probes ("Does low-HP agent flee?" "Does leader post defense
  quests?") are *interpretable* and *stable* — they stay valid as long as
  the observation schema contains `hp_pct` and `micro_kind` has `Flee`.
- They double as sanity tests *and* as comparison oracles. A new backend
  that passes fewer probes is a regression regardless of its episode return.
- The assert syntax is intentionally close to SQL-over-trajectories; no new
  ML framework needed, just a trajectory query engine.

**Gaps / open questions:**
- The probe grammar is new DSL surface; not written yet. It overlaps the
  reward DSL (§2.5) in shape — both filter over (observation, action,
  event) tuples. Consider unifying the two grammars; a probe is a reward
  that asserts instead of accumulates.
- Who authors probes? The ML engineer for behavioral regressions; the game
  designer for "don't attack spouses" style cultural constraints. Need a
  clear convention for which probes are essential (CI-gating) vs. advisory.
- Running probes against stochastic policies requires enough seeded episodes
  to get stable probability estimates. Decide on a default N (e.g., 32
  episodes per probe) and an explicit `seeds [42, 43, ...]` override.
- How do probes interact with schema versioning? A probe references
  `self.hp_pct` by name; if the observation drops `hp_pct` (breaking
  change), the probe fails to compile. Fail-loud, same as everywhere else.
  A probe that references a field that was appended under story 14 works
  automatically once the probe's target schema_hash matches the checkpoint.
- Probes for macro actions require long scenarios (Conquest resolves 2000+
  ticks after PostQuest). Budget accordingly, or split probes by horizon
  (short / medium / long).

**Related stories:** 13 (probes reference schema field names), 14 (a new
observation feature enables new probe predicates), 15 (probes run against a
specific schema_hash; mismatch fails loud), 16 (probes are queries over the
stored replay tuples — probes can be run post-hoc on saved buffers, not just
live), 17 (probes that fail should emit failing ticks to the decoder for
inspection), 19–20 (training on bootstrap trajectories: probes are the
"did bootstrap give us sensible behavior?" check).

---

### Cross-story summary

| Story | Verdict  | Blocking gap                                           |
|-------|----------|--------------------------------------------------------|
| 13    | PARTIAL  | Canonical normalization taxonomy unspecified; compiler does not yet exist |
| 14    | SUPPORTED| Slot-internal append semantics need spelling out       |
| 15    | GAP      | Proposal's migration-table design is wrong for this project; re-scope to hash + fail-loud |
| 16    | SUPPORTED| Volume at 20K agents needs downsampling strategy       |
| 17    | SUPPORTED| Buffer file format needs an index sidecar              |
| 18    | PARTIAL  | Probe DSL and runner are new surface, not in proposal  |

The single biggest piece of work across category C is **story 18's probe
DSL + runner**. It unblocks sanity-checking every checkpoint, serves as the
comparison surface for backend swaps, and doubles as a cultural-constraint
test bed ("no spouse-killing"). Stories 13/17 fall out of the DSL
compiler's JSON emitter (already listed as a compiler output in §3). Story
15's correction (drop `@since`, drop migration tables) is pure
deletion/simplification, not new work. Story 16 is plumbing with known
volume tradeoffs.

---

## Batch D — Training Pipeline (Trainer)

Per-story analysis of stories 19-24 from *(archived)*. Cites `spec.md` (policy/obs/action/reward/backend schema) and `spec.md` (action vocabulary) as design anchors.

Overall posture: the DSL is **very strong** on observation + mask + single-backend interchange, **moderately strong** on reward declaration, and **under-specified** on the trainer-side plumbing — replay buffer format, curriculum stages, telemetry hooks. The interchangeable `PolicyBackend` trait (§2.4) and declarative `reward { ... }` block (§2.5) give us the right primitives for IL bootstrap and on-policy RL; actor-critic, curriculum, and rare-event upweighting are additive extensions that need to be written into the DSL surface explicitly.

---

#### Story D.19: Bootstrap from utility trajectories
**Verdict:** SUPPORTED

**User's framing note:** "Great"

**How the DSL supports this:**
The single-backend commitment (README "Settled" bullet 3 / proposal §2.4) already makes the Utility backend a first-class `PolicyBackend` implementation — same observation packing, same masks, same action space. Running the sim with `UtilityBackend` emits the same `(observation, action)` tuples the Neural backend expects to consume. Story 16's per-tick training-data emission (marked "Great") fills in the logging side: `(observation, action, log_prob, reward, terminal_flag)` per agent per tick to a buffer file. Put those together and trajectory bootstrap is a single pipeline — run sim under utility, drain the emitted replay buffer, warm-start the neural weights on it.

**Implementation walkthrough:**
1. Trainer authors a policy block with `backend Utility { rules: "npc_utility.rules" }`. Compiler emits the utility argmax-over-masked-candidates implementation (proposal §2.4 comment: "declarative scoring rules from DSL").
2. Trainer enables per-tick trajectory logging (story 16 hook, likely a `@emit_training_data` annotation on the policy block or a CLI flag).
3. Run N ticks × M agents. Each agent's (observation, mask, action, reward, terminal) tuple is flushed to an append-only file. Utility backend's `log_prob` is degenerate (argmax → 1.0; softmax-with-temperature over utility scores if the trainer wants calibrated behavior-cloning targets) — this is an open detail.
4. Behavior cloning loop (offline, outside the sim): load replay buffer → train neural weights to minimize cross-entropy against the utility's categorical choices per head (macro_kind, micro_kind, each parameter head), MSE against the continuous heads (pos_delta, magnitude). Masks are part of the observation record, so the BC loss excludes masked-off actions automatically.
5. Swap `backend Utility` → `backend Neural { weights: "npc_v0_bc.bin", h_dim: 256 }` in the policy block. Same DSL file, different backend line. Story 23 handles the hot-swap.

Because observation packing, masks, action space, and reward are declared once in the DSL and consumed by both backends, there is no schema drift risk between bootstrap and training — the schema hash is computed once per policy block.

**Gaps / open questions:**
- Utility `log_prob` semantics. BC is cleanest when utility emits calibrated probabilities (softmax over utility scores with a temperature). Needs specification — is `softmax_temperature` a property of the utility backend or a per-policy annotation? The DSL should declare it so that trajectories carry correct log-probs for later off-policy correction (importance sampling if BC later transitions to off-policy RL).
- Trajectory format is open (README "low-priority / defer" item 20: "Trainer integration — replay buffer format, episode boundary semantics"). Minimum schema: `{ schema_hash, tick, agent_id, obs_bytes, mask_bytes, action, log_prob, reward, done, terminal }`. Should be emitted as columnar parquet or NDJSON with a sidecar manifest declaring schema_hash for loud failure on mismatch.
- Episode boundary semantics for zero-player worldgen. Agents live indefinitely — when does an "episode" end for reward discounting? Proposals: death-as-terminal + rolling windows, or fixed N-tick horizons.
- Utility → Neural swap path is actually Utility → BC-warm Neural → RL fine-tune. Story 20's reward block governs the second half.

**Related stories:** 16 (per-tick trajectory emission), 18 (backend comparison), 20 (reward block for RL fine-tune), 22 (curriculum), 23 (checkpoint deploy).

---

#### Story D.20: Declare reward in DSL (with actor-critic extension)
**Verdict:** PARTIAL

**User's framing note:** "Great, but we will also want to support actor critic"

**How the DSL supports this:**
Proposal §2.5 gives us a declarative reward block, shown below verbatim from the spec:

```
reward {
  delta(self.needs.satisfaction_avg)         × 0.1
  delta(self.hp_frac)                        × 5
  +1.0  on event(EntityDied{killer=self ∧ target.team ≠ self.team})
  -1.0  on event(EntityDied{target ∈ self.close_friends})
  +0.05 per behavior_tag accumulated this tick
  +2.0  on event(QuestCompleted{quest.party_member_ids ∋ self})
  ...
}
```

The compiler emits a per-tick reward kernel that diffs pre/post-tick observations and scans events involving `self`, plus logging hooks for (observation, action, reward) tuples, plus validation that every term references a declared view or event (§2.5 "Compiler emits"). This covers vanilla REINFORCE cleanly: per-step reward → discounted return → policy gradient.

**Actor-critic is where the current schema stops short.** Proposal §2.5 only specifies a scalar reward stream. Nothing in the current DSL declares a **value function head**, an **advantage estimator**, or **PPO clip parameters**. These need to be added, and they're natural additions because the policy block already has a `backend Neural { weights: ..., h_dim: ... }` declaration — the value head is just another output projection off the same trunk.

**Implementation walkthrough (proposed actor-critic extension):**

Proposed extension to §2.5 and §2.4 — the policy block grows `value` and `training` sub-blocks:

```
policy NpcDecision {
  observation { ... }
  action { ... }          // heads: macro_kind, micro_kind, target, pos_delta, magnitude, quest_type, ...
  mask { ... }
  reward { ... }          // scalar reward stream (unchanged from §2.5)

  value {
    // Separate head off the shared trunk. Scalar V(s).
    head scalar v_pred                  // shared trunk → Linear(h_dim, 1)
    // Or: separate trunk (decoupled A/C). Default shared; override per-block.
    trunk shared                        // shared | separate
    loss mse                            // mse | huber
    clip_range 10.0                     // optional reward clipping
  }

  advantage {
    kind gae                            // gae | nstep | montecarlo
    gamma   0.99
    lambda  0.95                        // GAE-λ; ignored for nstep/MC
    normalize per_batch                 // per_batch | per_agent | none
  }

  training {
    algorithm ppo {
      clip_epsilon       0.2
      vf_coef            0.5
      entropy_coef       0.01
      n_epochs           4
      minibatch_size     4096
      target_kl          0.02           // optional early stop
    }
    // or: algorithm reinforce { baseline: v_pred }
    // or: algorithm bc { loss: cross_entropy }
    optimizer adamw { lr: 3e-4, beta2: 0.98, weight_decay: 1.0 }
    grokfast ema { alpha: 0.98, lamb: 2.0 }   // optional (MEMORY.md pattern)
  }

  backend Neural { weights: "npc_v3.bin", h_dim: 256 }
}
```

Concrete answers to the prompt's actor-critic questions:

- **Does the policy block need a `value` head separate from action heads?** Yes. It's declared in a `value` sub-block that sits alongside `action`. Default trunk is shared with the action heads (one encoder, multi-head output); `trunk separate` produces a decoupled critic when divergence hurts training (see MEMORY.md note: "PPO collapses — bad value head" — that issue came from a shared-trunk bootstrap that the action gradients destabilized; having `trunk shared|separate` as a declared toggle lets the trainer recover without code changes).
- **Where does the critic train?** Same place as the policy — the compiler emits a per-tick dataset row `(obs, action, log_prob, reward, v_pred)` where `v_pred = model.value_head(obs)`. The offline trainer computes GAE advantages (δ_t = r_t + γ V(s_{t+1}) − V(s_t), Â_t = δ_t + γλ Â_{t+1}) per the declared `advantage { kind=gae, gamma=0.99, lambda=0.95 }` block. Critic loss is MSE(V(s), R_t) where R_t is the discounted return or TD(λ)-target. Both policy and value losses backprop through the shared trunk (unless `trunk separate`).
- **What's the reward block syntax that supports both REINFORCE and PPO/AC training?** The `reward` block is identical — it always declares the per-step scalar reward. What **differs** is the `advantage` and `training` blocks. REINFORCE uses `training { algorithm reinforce { baseline: v_pred | none } }`; PPO uses `training { algorithm ppo { clip_epsilon: 0.2, ... } }`. Behavior cloning uses `training { algorithm bc }` (value + advantage blocks ignored). This separation of concerns (reward = environment semantics, advantage = credit assignment, training = optimizer recipe) keeps the reward block the same across all algorithms while making the switch declarative.
- **PPO clip parameters?** Explicit: `clip_epsilon` (policy ratio clip, default 0.2), `vf_coef` (value loss weight, default 0.5), `entropy_coef` (exploration bonus, default 0.01), `n_epochs` (PPO epochs per rollout, default 4), `minibatch_size`, optional `target_kl` for early stopping. All declared, all validated at compile time against the chosen `algorithm` (REINFORCE rejects `clip_epsilon`).

The hierarchical action heads (§2.2 macro_kind × micro_kind × pointer × continuous) each need their own `log_prob` and `entropy` computed and summed — this is a codegen concern but the DSL already declares the heads.

**Gaps / open questions:**
- **Credit assignment for macro actions.** Proposal §2.5 flags this as `[OPEN]`: "Reward shaping for rare macro actions (Conquest reward arrives 2000+ ticks after the PostQuest decision)." GAE with γ=0.99 has an effective horizon of ~100 ticks — far too short. Options: per-head γ (macro head uses γ_macro=0.999 ≈ 1000-tick horizon; micro head uses γ_micro=0.99), n-step returns with explicit goal-conditioned bootstrap, or separate trainers per head. The DSL should allow `advantage { macro: {gamma: 0.999}, micro: {gamma: 0.99} }` overrides.
- **Shared vs separate trunk default.** MEMORY.md records that PPO collapsed on shared-trunk. Need an empirical rule for when to decouple — probably "decouple if entropy collapse observed within N minibatches" — but that's a training-time heuristic not a DSL concern.
- **Value clipping (PPO's vf_clip trick).** Not in the current proposal. Should probably be a `value { clip_range: 0.2 }` knob.
- **Off-policy vs on-policy dispatch.** PPO/REINFORCE are on-policy; BC is off-policy. When the utility backend emits trajectories for BC, log_probs are mis-calibrated for importance correction. Need either a strict "BC only for utility trajectories" rule or importance-sampling machinery (V-trace / Retrace). Defer.
- **Reward shaping per-role.** A leader's reward shouldn't be the same as a commoner's (cascade-impact rewards skew everything). Proposal doesn't distinguish. Could declare reward overrides per role via observation-conditional terms in the reward block — but this is speculative.

**Related stories:** 16 (training data emission — must log `v_pred` and `log_prob` alongside action), 19 (BC as the `algorithm bc` training variant), 21 (rare-action upweighting interacts with advantage normalization), 22 (curriculum stages swap the `training` block over time).

---

#### Story D.21: Up-weight rare actions in training
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The hierarchical macro/micro action decomposition (proposal §2.2) is half of the answer — it already separates rare (macro: PostQuest/AcceptQuest/Bid) from common (micro: Hold/Move/Attack/...) at the head level. Each head can, in principle, get its own learning rate or loss weight. Proposal §7 "Risks" acknowledges rare-action training is hard and lists "prioritized replay buffer with rare-event up-weighting" as a mitigation. But the DSL surface for declaring which actions are rare, and at what weight, isn't specified.

**Implementation walkthrough (proposed):**

Add a `training_weights` annotation to action heads, or a per-action tag in the action block, that the replay sampler honors:

```
action {
  head categorical macro_kind: enum {
    NoOp,                              @training_weight(1.0)
    PostQuest { @training_weight(50.0) @rare },
    AcceptQuest { @training_weight(20.0) },
    Bid { @training_weight(10.0) },
  }

  head categorical micro_kind: enum {
    Hold,                              @training_weight(0.1)  // very common, downweight
    MoveToward, Flee,                  @training_weight(1.0)
    Attack, Cast, UseItem,             @training_weight(2.0)
    Harvest, Eat, Drink, Rest,         @training_weight(1.0)
    PlaceTile, PlaceVoxel, HarvestVoxel, @training_weight(5.0)  // rare
    Converse, ShareStory,              @training_weight(3.0)
  }

  head categorical quest_type: enum QuestType {
    Hunt, Escort, ...,                 // defaults to 1.0
    Conquest @training_weight(100.0),   // only a handful per campaign
    Marriage @training_weight(50.0),
    Found    @training_weight(100.0),
  }
}
```

Two consumers of these weights:

1. **Prioritized replay buffer**: when offline training samples minibatches, rows get probability proportional to `max(training_weight[head_k] for each head k with nonzero log_prob)`. Rows where the agent chose `Conquest` are ~100× more likely to appear in a minibatch than rows where it chose `Hold`. Standard prioritized-experience-replay math (Schaul et al.) with importance-sampling correction `w_i = (N × P_i)^{-β}` to debias the policy gradient.
2. **Per-head loss scaling** at the training loss level: the cross-entropy loss for the `macro_kind` head is multiplied by `5.0` if the sampled macro was rare, so even in a uniformly sampled batch the rare macros get gradient signal.

The sampler also needs **action-frequency telemetry** to detect when weights need tuning — running counts per `(head, action)` tuple, exposed via the same telemetry hooks story 24 uses. If `count(PostQuest) / count(NoOp) < 10^-4` after N steps, emit a warning.

**Gaps / open questions:**
- Whether to annotate weights at the action definition site (shown above) or in a separate `training { ... }` block (cleaner separation, but loses locality). Probably the latter to keep the action vocab language-only.
- Replay buffer format is unspecified (README defer-list item 20). Minimum: a priority-tree or sum-tree index over the trajectory file. `sample_weighted(batch_size)` returns rows with IS-correction weights.
- Pointer-head rare-target upweighting is harder — pointer targets are slot indices, not fixed verbs. Rare-target emphasis is an advantage-estimator concern (e.g. give higher advantage to selecting the rarely-selected slot 9 when rewarded), not a replay-priority concern.
- Interaction with curriculum (story 22): in stage 1 `Conquest` is masked out, so `training_weight(100)` is wasted. Weights should be **stage-scoped**: stage 1 declares `Conquest @training_weight(0)`, stage 5 declares `Conquest @training_weight(100)`. See story 22.
- MEMORY.md note: V3 pointer action space has "BC alone: 2.9% (model collapses to hold without engagement heuristic)" — the `Hold` downweight above is the DSL-level prophylactic for exactly that collapse.

**Related stories:** 20 (training block where weights live), 22 (stage-scoped weight overrides), 24 (action-frequency telemetry feeds back to weight tuning).

---

#### Story D.22: Curriculum / staged training
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The mask language (§2.3) is the right primitive — masking out unreached verbs is already the declarative mechanism for role gating, and curriculum is just training-time role gating. The README "Settled" bullet 4 explicitly ties role power to mask: "Role power = mask + cascade, not a smarter policy." Stage N's curriculum is literally a training-time mask override: stage 1's mask zeroes every action except `Hunt` and `Eat`; stage 5's mask zeroes nothing. Because the mask is per-head and compiled to a boolean tensor, overriding it per stage is cheap.

What's missing: a **stage declaration surface** and **transition criteria**.

**Implementation walkthrough (proposed):**

Add a `curriculum` block to the policy:

```
policy NpcDecision {
  ...

  curriculum {
    stage Foraging {
      mask_override {
        // Allow only these micro_kinds; everything else training-time-masked to 0
        micro_kind allow [Hunt, Eat, Drink, Rest, MoveToward, Hold]
        macro_kind allow [NoOp]
      }
      training_weights {
        // Stage-specific upweights
        micro_kind { Hunt: 5.0, Eat: 3.0 }
      }
      transition_when {
        metric action_entropy(micro_kind)  >= 1.2
        metric mean_episode_reward         >= 2.0
        min_steps 50_000
      }
      reward_override {
        // Emphasize survival during foraging stage
        +5.0 × delta(self.hunger_frac)   // was 0.1 × delta(needs.satisfaction)
      }
    }

    stage Combat {
      inherits Foraging
      mask_override {
        micro_kind allow_additional [Attack, Cast, Flee]
      }
      transition_when {
        metric win_rate_vs_baseline >= 0.4
        min_steps 100_000
      }
    }

    stage Social {
      inherits Combat
      mask_override {
        micro_kind allow_additional [Converse, ShareStory]
      }
      ...
    }

    stage Macro {
      inherits Social
      mask_override {
        macro_kind allow_additional [PostQuest, AcceptQuest, Bid]
        quest_type allow [Hunt, Escort]   // start narrow, expand
      }
      ...
    }

    stage Full {
      inherits Macro
      mask_override { macro_kind allow_all; micro_kind allow_all; quest_type allow_all }
    }
  }
}
```

Mechanics:

1. **`mask_override` composes with runtime mask.** The runtime mask (§2.3 predicates: `Attack(t) when is_hostile ∧ distance < AGGRO_RANGE`, etc.) is AND-ed with the stage mask. Runtime rules still hold ("can't attack a non-hostile target") but the stage further restricts ("can't attack at all, even hostile, during Foraging"). Implementation: compiler emits `stage_mask_i[N × NUM_ACTIONS]` static tensor per stage; final mask = `runtime_mask & stage_mask_i`.
2. **`transition_when` block** evaluates against emitted training telemetry (story 24). Criteria are AND-ed: entropy ≥ threshold AND mean episode reward ≥ threshold AND min steps elapsed. When all true, move to next stage. No auto-regression — once advanced, stays advanced (unless the trainer manually rolls back by editing the stage pointer in the checkpoint).
3. **`inherits`** lets later stages additively expand masks without redeclaring base.
4. **`reward_override` and `training_weights`** are stage-scoped, overriding the base `reward` and per-action weights.
5. The compiler validates that every action mentioned in a stage's mask_override actually exists in the action block, and that `transition_when` metrics are declared in the telemetry surface.

Transition criteria could also be **action-distribution-driven** — e.g. transition out of Combat once the model reliably selects `Attack` when hostile targets are in range (precondition coverage ≥ 90%). This needs a declared metric hook; see story 24.

**Gaps / open questions:**
- **Stage state.** Which stage are we in? Lives in the model checkpoint metadata alongside weights, schema hash, training step count. Loading a checkpoint resumes at the stored stage.
- **Stage pointer vs stage weights.** Do we keep one set of weights per stage (stage 1 weights used once, frozen) or continuously finetune the same weights across stages? Default: continuous — the mask widening is what drives behavior change, not a reset. The Grokfast EMA / AdamW state also persists.
- **Curriculum and BC bootstrap interact.** If utility trajectories were generated under Stage 5 (full action space), BC on stage 1 is awkward — we'd be dropping all the rare-macro data. Solution: filter BC data by the stage mask, or generate stage-specific utility trajectories.
- **Reward scheduling (`reward_override` above)** is an actor-critic correction. Changing the reward function mid-training is a non-stationary-environment problem; the critic V(s) becomes stale at every stage transition. Mitigation: re-warm the critic by freezing the policy for M steps after each transition and letting V(s) catch up.
- **Stage transitions and early stopping.** If `target_kl` (from PPO training block) fires before `transition_when` is satisfied, stage is stuck. Need explicit fail-forward or abort semantics.

**Related stories:** 20 (training block's reward/algorithm are what curriculum overrides), 21 (stage-scoped action weights), 24 (entropy + action-distribution metrics feed transition criteria).

---

#### Story D.23: Deploy a model checkpoint
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Proposal §2.4 declares `backend Neural { weights: "npc_v3.bin", h_dim: 256 }` as a file path on the policy block. The schema-hash guard (§4) is specified: "Every model checkpoint stores its training-time hash. Loading a model whose hash mismatches the current DSL is a hard error." Story 15's user annotation is explicit: "WE DO NOT WANT TO HAVE V1 AND V2 AND V3 in the same codebase. Git can be used to store versions." So schema mismatch **FAILS LOUD** — no migration tables, no pad-zero fallback, no auto-upgrade path. The §4 proposal's migration-table language is superseded by the story 15 constraint.

**Implementation walkthrough:**

Checkpoint file layout:
```
npc_v3.bin
├── header (fixed offset, 128 bytes):
│   ├── magic:            "NPCPOL\0\0"          (8 bytes)
│   ├── format_version:   u32                   (4 bytes)
│   ├── schema_hash:      [u8; 32]              (32-byte SHA256 of obs+action+mask schema)
│   ├── policy_name:      [u8; 32]              ("NpcDecision\0...")
│   ├── training_step:    u64                   (for provenance)
│   ├── stage_name:       [u8; 16]              (curriculum stage from story 22)
│   ├── reserved:         [u8; 12]
│   └── weights_offset:   u64                   (offset to weights section)
├── weights section (contiguous tensor blobs with named offsets)
└── footer: CRC32 over weights section
```

Hot-swap semantics:

1. Trainer drops `npc_v3.bin` into the configured weights directory. Either the path is hot-polled (watcher) or explicitly signaled via an admin command.
2. Runtime reads the header **before** loading weights.
3. **Schema hash check (FAILS LOUD):** `if header.schema_hash != compiled_dsl.schema_hash { abort_with_error("...") }`. Per story 15, there is no fallback, no migration table, no pad-zero behavior. The error message shows the expected hash, the loaded hash, and points to the DSL commit that produced each. The current tick's decisions continue under the old weights; the new weights are rejected entirely.
4. **Magic + format_version check (FAILS LOUD):** reject non-matching magic or unrecognized format_version with a clear error.
5. **CRC check (FAILS LOUD):** detect truncated/corrupt file.
6. **Shape check (FAILS LOUD):** every tensor declared by the neural backend's architecture (`h_dim: 256`, head dimensions derived from action block) is matched against the weights file. Missing tensors, wrong shapes → abort load.
7. **Atomic swap:** weights load into a staging buffer; on success, a single pointer swap (RCU or mutex) makes the next tick use the new weights. Tick N uses old, tick N+1 uses new — no partial state. In-flight forward passes finish against the weights they started on.
8. **Graceful fallback when no weights at all:** if `backend Neural` is declared but no file exists, falls back to `backend Utility` (if declared as the bootstrap in the same policy) or refuses to start. This is the one graceful path; all others fail loud.

What fails gracefully vs catastrophically:

| Failure mode                          | Behavior                                    |
|---------------------------------------|---------------------------------------------|
| File doesn't exist at startup         | Graceful: fall back to Utility if declared  |
| File doesn't exist at hot-swap        | Graceful: keep current weights              |
| Magic mismatch                        | **LOUD ABORT**: wrong file type             |
| Format version mismatch               | **LOUD ABORT**: codebase too old/new        |
| Schema hash mismatch                  | **LOUD ABORT** (per story 15): no migration |
| Tensor shape mismatch                 | **LOUD ABORT**: reject swap, keep old       |
| CRC mismatch                          | **LOUD ABORT**: file corrupt                |
| Stage name in checkpoint unknown      | **LOUD ABORT**: curriculum mismatch         |
| Out-of-memory during stage load       | **LOUD ABORT**: reject swap, keep old       |

Story 15 constraint is load-bearing here: "Git can be used to store versions, it is nothing but waste to support backwards compatibility." Every code path that might silently pad-zero, upcast, or remap channels is **explicitly rejected**. The only "smart" behavior is the atomic pointer swap + rollback-on-error; everything else is a bright-line check with a clear error.

**Gaps / open questions:**
- Whether the schema hash covers action vocabulary + reward, or just observation. Proposal §4 [OPEN] item 17. Recommendation given story 15's strictness: hash **all three** (observation + action + reward) — changing any of them changes what the model learned to do, and silent success is the failure mode we're preventing.
- Does the curriculum `stage_name` participate in the schema hash? Probably not — stage is training provenance, not input/output layout. But the runtime should warn if loaded stage doesn't match the DSL-declared expected production stage.
- How to handle rolling deploys / canarying. Not in the current proposal. Could add `backend Neural { weights: "npc_v3.bin", canary_weights: "npc_v4.bin", canary_frac: 0.05 }` — 5% of agents use v4 for A/B.
- Weight file format: raw packed tensors vs safetensors vs custom. Recommendation: **safetensors-style** with named tensor offsets so the header can be inspected without loading the whole file.
- Hot-swap during an in-flight training step is UB unless training and inference use separate weight buffers. Trainer writes to a staging path; atomic rename triggers the runtime watcher.

**Related stories:** 14 (schema hash bump on obs change), 15 (hash mismatch FAILS LOUD — constraint-defining), 22 (stage metadata in checkpoint).

---

#### Story D.24: Detect mode collapse
**Verdict:** PARTIAL

**User's framing note:** "Essential"

**How the DSL supports this:**
The runtime already has everything needed to compute action-distribution telemetry — the `PolicyBackend::evaluate_batch` call (§2.4) produces `ActionBatch`, which is a typed per-agent action containing head choices. Counting those choices per tick gives raw distributions. What the DSL doesn't yet declare is **which metrics to emit, how to emit them, and alert thresholds**. That's a `telemetry { ... }` block that needs to exist.

**Implementation walkthrough (proposed):**

Add a `telemetry` block to the policy:

```
policy NpcDecision {
  ...

  telemetry {
    // Per-head entropy — primary mode-collapse detector.
    metric entropy_macro_kind  = entropy_of(action.macro_kind)
           window 1000                    // rolling window size in ticks
           emit_every 100                 // tick cadence for emission
           alert when value < 0.3         // log2(1.35) — model picked one verb >70% of the time

    metric entropy_micro_kind  = entropy_of(action.micro_kind)
           window 1000
           alert when value < 0.5         // tighter — more options, less collapse allowed

    metric entropy_quest_type  = entropy_of(action.quest_type)
           conditioned_on action.macro_kind = PostQuest
           window 10000                   // rare — need longer window
           alert when value < 0.8

    // Per-head action frequencies — individual-action detection (not whole-head).
    metric freq_micro          = histogram(action.micro_kind) window 1000 emit_every 100
           alert when max_bin > 0.85      // any one action > 85% of tick choices

    // Parameter-head coverage.
    metric pointer_slot_coverage = coverage(action.target, n_slots=42) window 5000
           alert when value < 0.3         // using <30% of available slot types

    metric continuous_pos_delta_stddev = stddev(action.pos_delta) window 1000
           alert when value < 0.05        // collapsed to near-constant movement

    // Reward + value head diagnostics (feeds back to story 20 actor-critic tuning).
    metric mean_reward         = mean(reward) window 1000
    metric value_error         = mse(value.v_pred, montecarlo_return) window 1000
           alert when value > 10.0        // critic has exploded

    // Mask-coverage: which masked-in actions does the model actually use?
    metric mask_util           = mean(argmax_action_in_mask_top_k(k=3)) window 1000
           alert when value < 0.7         // model picking low-probability (per its own logits) actions
  }
}
```

Mechanics:

1. **Runtime computes metrics in a tick-end hook.** Because `evaluate_batch` returns the full per-agent `ActionBatch`, histogram updates are a single parallel scatter per head. Cost is O(N agents × N heads) per tick, ~trivial.
2. **Rolling windows** implemented as ring buffers per metric, sized by `window`. Entropy and stddev computed from running sums (Welford for numerically stable variance).
3. **Emission** writes to a structured log (JSON lines, one record per emit cadence) with `{ metric_name, value, tick, policy_name }`. Training infrastructure consumes this for dashboards and alert routing.
4. **Alert criterion** is declared inline. When `value` hits the `alert when` predicate, the runtime emits a `METRIC_ALERT` event (loud: stderr + log + optional webhook). It does not halt training — the trainer decides.
5. **Conditional metrics** (e.g. `entropy_quest_type conditioned_on macro_kind=PostQuest`) only sample on tick when the condition fires. Essential for rare-macro-action telemetry: measuring `quest_type` entropy over all ticks is meaningless when >99% of ticks emit NoOp for macro_kind.
6. **Curriculum integration:** `transition_when` criteria in the curriculum block (story 22) can reference declared metrics directly: `metric entropy_micro_kind >= 1.2`.

Alert criteria — specific thresholds for mode-collapse detection:

- **Entropy-based (primary):** `H(action) < ε` where `ε ≈ log(K) × 0.3` for a K-way categorical head. A K=12 micro_kind head has `H_max = log2(12) ≈ 3.58`; alert at `< 0.5`. For K=4 macro_kind, `H_max = 2`; alert at `< 0.3`. These are starting points; trainer tunes.
- **Max-bin frequency:** `max_k P(action = k) > 0.85` — single-action dominance.
- **Zero-frequency floor:** `min_k P(action = k) < 0.001` in a stage where action k is unmasked — model ignoring a valid verb.
- **Pointer-slot coverage:** fraction of `[0, NUM_SLOTS)` ever selected over the window — catches collapse to a single target.
- **Continuous-head stddev:** too-low stddev on pos_delta / magnitude indicates collapsed continuous policy.
- **Critic-specific (story 20):** value-error explosion (`mse(v, R) > 10`) or value clip saturation — catches bad-critic issues MEMORY.md already documented ("PPO collapses — bad value head").

Reference point from MEMORY.md: V2 actor-critic "12% win rate (HvH-specialized, doesn't transfer)" and BC-alone "model collapses to hold" (2.9% win rate) — these are exactly the patterns entropy + max-bin alerts should have caught in real time.

**Gaps / open questions:**
- **Where does the alert go?** Runtime stderr + log is cheap; webhook/email is trainer-specific. Probably keep to structured log and let external infra decide.
- **Cross-agent vs per-agent distributions.** Declared metrics above are aggregated across all agents per tick. Per-role / per-stage distributions might also be useful — needs a `group_by role` clause. Defer unless empirically needed.
- **Historical baseline.** "Entropy is low" is absolute; "entropy is 3σ below the training-window mean" is relative. Current proposal is absolute thresholds; extending to z-score-style thresholds is a later refinement.
- **Interaction with curriculum.** When the curriculum widens the mask (Stage 1 → Stage 2), entropy should *spike* (more options) — an entropy-too-low alert firing right before transition is expected, not a bug. Alerts should be suppressed for N ticks after stage transition.
- **Action-sequence telemetry.** Mode collapse sometimes manifests as cycles (`Hold, Move, Hold, Move, ...`) rather than a single action dominating. Detecting this needs n-gram distributions, more expensive. Defer.

**Related stories:** 20 (value head diagnostics feed into the critic loss tuning), 21 (action-frequency histograms are the feedback signal for weight tuning), 22 (metrics drive `transition_when` curriculum criteria), 23 (checkpoint metadata could include last-known telemetry snapshot for provenance).

---

### Cross-story summary

| Story | Verdict  | Key gap                                             |
|-------|----------|-----------------------------------------------------|
| 19    | SUPPORTED | utility `log_prob` calibration; trajectory format  |
| 20    | PARTIAL   | needs `value` + `advantage` + `training` sub-blocks (proposed) |
| 21    | PARTIAL   | needs per-action `training_weight` annotations + replay sampler hook |
| 22    | PARTIAL   | needs `curriculum { stage { ... } }` block with `mask_override` + `transition_when` |
| 23    | SUPPORTED | hash mismatch FAILS LOUD per story 15; atomic swap; graceful only for "file missing" |
| 24    | PARTIAL   | needs `telemetry { metric ... alert when ... }` block |

**Cross-cutting observation:** the proposal is strongest where ML contracts meet the runtime (obs packing, masks, action heads, schema hashing, backend trait). It's thinnest where trainers meet the DSL (reward-to-algorithm pipeline, curriculum staging, telemetry emission, replay buffer format). README "low-priority / defer" item 20 ("Trainer integration — replay buffer format, episode boundary semantics") is no longer low-priority if stories 19-24 are essential; the next proposal revision should promote it and spec out `training { ... }`, `curriculum { ... }`, `telemetry { ... }` blocks as first-class policy sub-blocks alongside `observation / action / mask / reward / backend`.

The actor-critic extension (story 20) is the most consequential addition: it introduces the `value` head, `advantage` block, and `training { algorithm ppo { ... } }` surface, all of which then become the substrate stories 21-24 extend (stage-scoped training weights, per-stage algorithm overrides, value-error telemetry).

---

## Batch E — Runtime / Sim Engineer

Analysis of the runtime-engineering user stories in *(archived)* against
the current DSL design (`spec.md`, `spec.md`,
`state.md`) and the voxel-engine GPU backend at `/home/ricky/Projects/voxel_engine/`.

Each story is scored SUPPORTED / PARTIAL / GAP against the schema as it exists today.
"Supported" means the proposal already describes a concrete mechanism; "partial" means
the mechanism exists but the story surfaces a concrete extension to spec; "gap" means
new design work is required.

---

#### Story E.25: One batched neural forward per tick
**Verdict:** SUPPORTED

**User's framing note:** "Good"

**How the DSL supports this:**
`spec.md` §2.4 defines exactly one call point:
```
fn evaluate_batch(observations: &PackedObservationBatch,  // [N, OBS_DIM]
                  masks:        &PackedMaskBatch)
                  -> ActionBatch;                         // [N] of typed Action
```
One forward per tick for every alive agent is the committed shape. §2.1.8 sizes
the batch: `1655 × 4 bytes × 20K agents = 132 MB`. The proposal states this is "on-GPU
is free; CPU pack <5ms" (§2.1.8). §5 splits compilation: observation packing → GPU,
mask evaluation → GPU, neural forward → GPU (already what Burn does).

**Implementation walkthrough:**
1. `iter_alive_agents()` produces a stable agent index layout for the tick. The
   `GroupIndex` in `state.md` already clusters entities by
   `(settlement_id, party_id)`, which gives contiguous ranges suitable for
   `[N, OBS_DIM]` packing without per-agent scatter.
2. The DSL's `observation { ... }` block compiles to a packing kernel (§3). Each
   atom/block/slot array maps to an offset within the packed row. Compiler
   emits "packed struct with named field offsets" so debug tools can decode
   bytes back to named fields.
3. For the voxel-engine backend (story 28), packing compiles to a single
   compute shader bound to (a) an agent-ID buffer, (b) the source field buffers
   (hp, treasury, etc.), (c) the output packed observation buffer. One dispatch
   with workgroup-per-agent. `GpuHarness::dispatch` in
   `voxel_engine/src/compute/harness.rs` already supports exactly this shape —
   named kernels + storage-buffer fields + `[groups_x, 1, 1]` dispatch.
4. Mask evaluation is a second dispatch into a per-head boolean tensor
   `[N × NUM_KINDS]`, `[N × NUM_SLOTS]`, etc. (§2.3). Cross-entity
   predicates that are GPU-hostile (§5 — `t ∈ quest.eligible_acceptors`) fall
   back to CPU into the same boolean tensor.
5. Burn (or an equivalent tensor library over the harness-managed buffer) runs
   the forward pass. Sampling with mask is one fused kernel that consumes
   logits + mask and writes typed action rows.

Memory is pinned on GPU for the whole run. `GpuHarness::create_field` returns
`FieldHandle`s that stay allocated until explicit release — no per-tick Vulkan
allocations. The packed observation buffer, the mask buffer, the logits buffer,
and the action buffer are all created once at startup and reused per tick.

**Gaps / open questions:**
- **Batch layout for dead/asleep agents.** The proposal says "all alive NPCs go
  through one forward pass" but fidelity zones (`state.md`,
  `FidelityZone.fidelity ∈ {High, Medium, Low, Background}`) imply background
  agents may skip inference. Two options: (a) pack a compacted `[N_active, OBS_DIM]`
  and maintain an active-ID side list, or (b) pack `[N_max, OBS_DIM]` and use an
  active mask; compute cost is N_active either way but (b) avoids a compaction
  pass. Needs specification.
- **Macro-head cadence (open question #12 in schema).** If macro inference runs
  only every 10 ticks, the batch on those ticks is larger (more heads) than on
  pure-micro ticks. Either accept two batch shapes or always emit all heads
  and ignore the macro output 9/10 ticks.
- **Observation feature pipeline.** Some atoms depend on *derived views*
  (`view::mood`, `is_hostile(a, b)`, group-war standings). If those views are
  lazy (story 26 PARTIAL case), the pack kernel stalls on view computation.
  Story 26 is the fix.

**Related stories:** 26 (view materialization feeds the pack kernel), 28 (the
compilation target that *runs* the batched forward), 31 (determines batch size).

---

#### Story E.26: Materialize hot views eagerly
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
`spec.md` §5 calls out "Event-fold view materialization —
reductions over event rings (sum damage events for hp, etc.)" as GPU-amenable.
§1.1 commits to event sourcing: "current state is a fold over events + entity
baseline." This is exactly the materialization-vs-lazy-fold axis. The
`state.md` derivation graph already distinguishes "Updated by" columns
(event-driven fields) from "Updater computes" (derived views). `behavior_profile`
(state.md line 276) is the canonical "updated incrementally on each event"
example — not re-folded every tick.

**Implementation walkthrough:**
Proposed DSL annotation:
```
view mood(self) @materialized {
  initial: 0.0
  on event(DamageReceived{target=self}) { self.mood -= 0.1 * e.damage / self.max_hp }
  on event(FriendDied{observer=self})   { self.mood -= 0.3 }
  on event(NeedsSatisfied{agent=self})  { self.mood += 0.05 * e.delta }
  clamp: [-1, 1]
}
```
`@materialized` compiles to:
1. A field on the agent struct (append-only addition, §4 "schema versioning").
2. An event-handler dispatch table — for each matching event type, a CPU or GPU
   update call. Since the event stream is append-only per tick, this is a
   fused reduction: group events by target agent, accumulate the delta, write
   once.
3. The pack kernel (story 25) reads `agent.mood` directly with no fold.

Lazy / non-materialized views compile to inline folds inside the pack kernel.
The DSL distinguishes them by the `@materialized` annotation. Cost trade-off:
eager pays cost per event, lazy pays cost per observation pack. For
frequently-read views (hp, treasury, mood, morale, `is_hostile`, relationship
valence), eager wins decisively because packing runs every tick whereas some
events are rare.

**Event handler codegen shape:**
```
// compiled from the DSL
for evt in world.events_this_tick.iter() {
    match evt {
        Event::DamageReceived { target, damage } => {
            agent_mut(target).mood -= 0.1 * damage / agent(target).max_hp;
        }
        Event::FriendDied { observer, .. } => {
            agent_mut(observer).mood -= 0.3;
        }
        ...
    }
}
```
For GPU targets, the same handler is a compute kernel dispatched over the
event ring where each thread handles one event; it uses atomic fetch-add into
the target agent's materialized field when multiple events target the same
agent on the same tick.

**Gaps / open questions:**
- **Fold order vs replay determinism (story 33).** When handlers use atomics
  across events targeting the same agent, fold order is non-deterministic on
  GPU. Mitigation: sort events by (target_id, source_tick_index) before the
  reduction, or run the reduction on CPU. §5 classifies the event-fold
  materializer as GPU-amenable but only for scalar reductions with commutative
  operators; for non-commutative chains (e.g. "max of an ordered sequence"),
  CPU.
- **Cache invalidation on save/load (story 32).** If materialized views live in
  the serialized snapshot, the invariant is "fold-from-baseline ≡ materialized".
  A CI check should re-derive materialized views from the event log on a
  sampled snapshot and diff against stored values.
- **Which views need `@materialized`?** The proposal doesn't list them. Working
  list from the observation schema: `hp`, `mood`, `morale_pct`, `focus`,
  `gold`, `n_close_friends`, `fame_log`, `group.treasury`, `group.military_strength`,
  `relationship.valence`. All appear in §2.1 and are read every tick per alive
  agent.

**Related stories:** 25 (pack kernel reads from materialized views), 27
(new event types need corresponding event handler bindings), 30 (materialized
views live in fixed-cap agent struct fields, no per-tick alloc), 33 (fold
determinism bounds replay).

---

#### Story E.27: Add a physics rule without breaking others
**Verdict:** PARTIAL (core composition supported; compile-time validation underspecified)

**User's framing note:** "Essential, should have 'compile' time validation"

**How the DSL supports this:**
The DSL is event-sourced (§1.1). Rules are of the form
`on event(<Pattern>) { <cascade or mutation> }`. New rules don't edit existing
ones — they register additional handlers. The composition semantics are
"every handler matching the event fires."

`spec.md` §5 already separates "cascade rules (CPU)" from
"GPU packers" and lists cascade rules as stay-CPU, keeping the dispatch surface
flexible for new rules.

**Implementation walkthrough:**

*Rule registration model.*
```
rule collapse_chain {
  on event(VoxelDamaged{pos=p, new_integrity=i}) where i <= 0.0 {
    emit StructuralEvent::FragmentCollapse { pos: p, cause: NpcHarvest }
    for q in voxel_neighbors_above(p) {
      damage_voxel(q, COLLAPSE_DAMAGE)
    }
  }
}

rule collapse_chronicle {  // independent, added later
  on event(StructuralEvent::FragmentCollapse{pos=p, cause}) {
    emit ChronicleEntry {
      category: ChronicleCategory::Crisis,
      text: format!("A section of ground collapsed at {:?}", p),
      entity_ids: entities_near(p, 20.0),
    }
  }
}
```
Both rules fire on the same underlying cascade chain. Adding
`collapse_chronicle` does not touch `collapse_chain`. The compiler collects all
rules with matching event patterns into a dispatch table.

*Ordering semantics.* The proposal needs to commit to one of:
  (a) **Unordered** — handlers must be commutative; compiler rejects rules
      that mutate fields other rules also write. Simple but restrictive.
  (b) **Priority-tagged** — `rule foo @priority(50)`; default 100. Compiler
      emits a topological warning if cycles would exist after priority resolution.
  (c) **Phase-based** — `on_pre_event`, `on_event`, `on_post_event` phases,
      strict within-phase commutativity.
  Recommendation: (c). Matches the existing determinism contract (from `CLAUDE.md`
  combat-sim: "unit processing order is shuffled per tick to prevent first-mover
  bias", but phase ordering is deterministic).

*Compile-time validation — what CAN be statically caught.* This is the user's
explicit extension:

1. **Event-type existence.** `on event(UnknownFoo{})` fails at compile time.
2. **Pattern-field mismatch.** `on event(EntityDied{weapon=w})` when the
   event struct has no `weapon` field.
3. **Field-write races in the same phase.** Two unordered rules both writing
   `agent.hp` → compiler error unless one of them uses an explicit
   commutative operator (`+=`, `min=`, `max=`).
4. **Cycle detection in cascades.** Rule A emits event type X, rule B handles
   X and emits Y, rule C handles Y and emits X, with no quiescence predicate
   → compiler warning. Detectable as a cycle in the
   event-type → emitter → event-type graph.
5. **Invariant violations (partial).** Rules annotated with pre/post-invariants
   (`@requires agent.hp >= 0`, `@ensures agent.hp >= 0`) can be checked with
   abstract interpretation over the mutation body. Full invariant checking is
   a research problem; first pass: syntactic checks (any write to a clamped
   field must go through the clamp helper).
6. **Mask-rule-observation coherence.** The proposal's mask predicates
   reference views; if a rule deletes a view's source event type, the mask
   still compiles but silently always-false. Compiler should warn.
7. **Schema-hash drift (§4).** Any change to event struct shape bumps the
   schema hash; CI blocks unless a migration is declared.

*What CANNOT be statically caught.*
- Semantic consistency (two rules that both decrement `hp` "correctly" but
  together overshoot). Requires runtime invariant checks.
- Rule interaction with RNG (`rng_state` is global serialized; two rules
  consuming randomness in different orders on different platforms are the
  determinism bug story 29 is about).
- Fixed-point convergence of cascade rules — a rule that re-emits on its
  own output may loop fewer/more times depending on agent count.

**Gaps / open questions:**
- Phase ordering not yet written into the proposal. Needs adding to §5 or a
  new §9 "rule composition".
- Compile-time checks 3, 5, 6 require a type-flow analysis pass in the
  compiler. Not specified.
- `emit` vs `apply` distinction — does the DSL allow direct mutation inside
  rules, or only event emission? Event-only is cleaner for replay (story 33)
  and for static analysis (all writes reduced to the event-dispatch handler),
  but is more verbose. The current proposal is ambiguous; §2.5 reward shows
  direct mutation (`delta(self.hp_frac)` reads), but §5 cascade rules say
  "event-fold". Pick one: **event-only**, with `apply` reserved for the
  compiler-generated fold itself.

**Related stories:** 25 (new rules add new mask predicates → observation packer
must be regenerated), 29 (rule ordering is a determinism knob), 33
(event-type additions require replay-log schema migration), 38 (new-rule
"dry-run" = self-contained tick reproduction).

---

#### Story E.28: Compile to either Rust or CUDA (use voxel-engine)
**Verdict:** PARTIAL (scoping is clear; voxel-engine integration shape needs specification)

**User's framing note:** "Essential, use voxel-engine though, that is what that
project is for"

**How the DSL supports this:**
`spec.md` §5 explicitly scopes GPU compilation — not "compile
the whole DSL to CUDA". Concrete GPU kernels listed:
- Observation packing
- Per-head mask evaluation
- Neural forward pass
- Event-fold view materialization

GPU-hostile (stay CPU):
- Cross-entity mask predicates
- LLM backend
- Chronicle text generation
- Cascade rules with heterogeneous events
- World event emission

Hybrid:
- Spatial queries (nearby_actors, threats)

**Implementation walkthrough via voxel-engine:**

voxel-engine at `/home/ricky/Projects/voxel_engine/` already provides the
relevant Vulkan/ash GPU infrastructure:

- `src/compute/harness.rs` — `GpuHarness` with `create_field`,
  `load_kernel` (SPIR-V bytes + binding count), `dispatch`, `upload`,
  `download`. Host-visible, host-coherent memory with persistent mappings.
  Exactly the "taichi-style CPU↔GPU harness" shape the DSL compiler needs.
- `src/terrain_compute.rs` — reference for a real pipeline (materialization
  with LRU slot pool, descriptor sets, fences). Useful precedent for the
  DSL's chunked-field management model.
- `src/vulkan/instance.rs` + `src/vulkan/allocator.rs` — already-working
  Vulkan context + gpu-allocator.
- `shaders/` directory — 30+ precompiled compute shaders for
  physics/fluid/SDF work. Precedent for the shader build pipeline
  (`build.rs` + `shaderc`).

Integration shape the DSL compiler should produce:

```
// CPU-side Rust stub emitted by the DSL compiler
pub struct PolicyRuntime {
    harness: voxel_engine::compute::GpuHarness,
    obs_field: FieldHandle,     // [N, OBS_DIM] f32
    mask_field: FieldHandle,    // per-head boolean buffers
    logits_field: FieldHandle,  // [N, NUM_LOGITS] f32
    action_field: FieldHandle,  // [N] packed action rows
    weights_field: FieldHandle, // model weights
    ...
}

impl PolicyRuntime {
    pub fn tick(&mut self, world: &WorldState) -> &[Action] {
        // 1. Observation pack kernel
        upload_event_ring_delta(&mut self.harness, world);
        self.harness.dispatch("pack_observations", &[...], [N_groups, 1, 1])?;

        // 2. Mask evaluation (GPU-amenable predicates)
        self.harness.dispatch("eval_mask_micro", &[...], [N_groups, 1, 1])?;
        // CPU patch for cross-entity predicates
        cpu_patch_mask_for_quest_eligibility(&mut self.harness, world);

        // 3. Neural forward (Burn over the field buffers, or hand-written
        //    shader if Burn interop is a problem)
        self.harness.dispatch("mlp_forward", &[...], ...)?;

        // 4. Sample + writeback
        self.harness.dispatch("sample_with_mask", &[...], ...)?;
        self.harness.download(&ctx, &self.action_field)
    }
}
```

**Which DSL constructs map cleanly to voxel-engine kernels:**

| DSL construct | Maps cleanly? | Kernel |
|---|---|---|
| `observation { self.hp_pct = self.hp / self.max_hp }` | Yes | one line in pack shader |
| `slots nearby_actors[K=12] from query::nearby_agents(self, radius=50) sort_by distance` | Hybrid | Spatial query on CPU (or via `src/ai/spatial.rs`), gather on GPU |
| `block self.psychological { ... }` | Yes | struct-of-arrays gather in pack shader |
| `summary recent_chronicle[...] { group_by e.category output count_log }` | Partial | If the event ring is GPU-resident, yes; currently it contains `String` (`state.md` "chronicle, `Vec<ChronicleEntry>` with String") so it's CPU-only |
| `mask { Attack(t) when t.alive ∧ is_hostile(self, t) ∧ distance(t) < AGGRO }` | Yes for intrinsic predicates | `is_hostile` decomposes to relationship_valence < T ∨ groups_at_war bitset ∨ predator_prey table — all GPU-friendly scalars |
| `mask { AcceptQuest(qid) when self ∈ quest.eligible_acceptors }` | No | Stay CPU; write into shared mask buffer |
| `view mood @materialized { on event(X) ... }` | Yes | compute shader dispatched per event |
| `rule collapse_chain { on event(VoxelDamaged) ... }` | No | Stay CPU (heterogeneous events + emit chain) |
| `reward { +1.0 on event(EntityDied{killer=self ...}) }` | Yes | reduction over filtered event stream |

**Which constructs DON'T map cleanly:**
- **String-bearing events** (`ChronicleEntry.text`, `WorldEvent::Generic{text}`).
  These must stay CPU or be represented by event-ID + parameter-bag. Per story
  60 ("text gen should not be load bearing. Use numeric IDs for everything
  important"), chronicles carry a `category: u8 + params: [u32; 4]` tuple and
  text is rendered post-hoc on the CPU for display only.
- **Quest/auction eligibility** — quest membership is a `Vec<AgentId>` that
  grows dynamically; GPU-resident only if we commit to a fixed-cap
  per-quest member table. Story 30 pushes us toward this anyway.
- **Room-growth automaton** (construction.rs flood fill) — not a per-tick
  inference kernel; keep as-is, CPU.

**voxel-engine-specific integration notes:**
- voxel-engine is Vulkan/ash + gpu-allocator. **Not wgpu and not CUDA.**
  `spec.md` §5 mentions "wgpu/CUDA"; this should be updated
  to **ash/Vulkan via voxel-engine's `GpuHarness`**. The DSL compiler emits
  SPIR-V (via `shaderc`, which is already in `voxel_engine/Cargo.toml`
  `build-dependencies`) and loads kernels via `harness.load_kernel`.
- Burn-on-Vulkan is technically supported (Burn's `wgpu` backend could bind
  to voxel-engine's Vulkan context via external memory, but that's a research
  integration). Simpler: hand-emit GEMM + activation shaders for the policy
  network, since observations and action-head shapes are declared in the DSL
  and the compiler can generate a fused forward shader specific to the
  network topology. This is the same technique Grokking transformer uses
  (our existing transformer is small: d=32, 4 layers — easily fits one
  compute shader).
- The CPU target ("native Rust, rayon-parallel") is the unchanged path —
  `evaluate_batch` with `par_iter` over the `[N, OBS_DIM]` rows. Used for
  debug builds and small-scale tests.

**Gaps / open questions:**
- **Weight upload cadence.** Neural weights change on each training
  checkpoint swap, not every tick. A separate `update_weights(checkpoint)` API
  on `PolicyRuntime` that calls `harness.upload` once, and a hot-swap fence
  to avoid tearing. Not currently specified.
- **Mixed CPU/GPU mask.** §5 says cross-entity predicates run on CPU and "that
  boolean tensor is what GPU consumes." Concretely, the flow is: GPU mask
  kernel writes initial mask → CPU patches the bits that require cross-entity
  walks → GPU sampler reads final mask. Needs two fences or a single
  read-back/write-back round trip per tick. Cost bound unclear.
- **Readback cost.** `harness.download` of `[N] × action_row` per tick. At N=200K
  and action_row = ~32 bytes, that's 6.4 MB/tick over PCIe. PCIe 4.0 x16
  handles this in ~70 µs; PCIe 3.0 ~140 µs. Fine, but worth tracking.
- **Shader codegen.** The compiler needs a SPIR-V emitter. Options:
  (a) emit GLSL and call `shaderc` at compile time; (b) emit SPIR-V directly
  via the `rspirv` crate; (c) emit Rust code that calls `shaderc` at runtime
  for specialized shaders. (a) is the simplest — matches voxel-engine's
  existing shader build pipeline.

**Related stories:** 25 (this is the deployment target), 29 (GPU determinism
constraints — sort event reductions, no float non-associativity for critical
reductions), 30 (GPU field buffers stay allocated for the run = zero-malloc),
31 (GPU is the only way to reach 200K).

---

#### Story E.29: Deterministic sim given seed
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Determinism is an existing first-class contract. From `state.md`:
```
rng_state | u64 | PCG-style RNG state; sole randomness source
next_rand_u32 / next_rand | all stochastic systems
```
and from project `CLAUDE.md`:
> All simulation randomness flows through `SimState.rng_state` via
> `next_rand_u32()`. Never use `thread_rng()` or any external RNG in simulation
> code. Unit processing order is shuffled per tick to prevent first-mover bias.
> Tests in `src/ai/core/tests/determinism.rs` verify reproducibility. CI runs
> determinism tests in both debug and release modes.

The DSL inherits this contract. Every `next_rand` call in generated code
advances the single `rng_state`.

**Implementation walkthrough:**

*RNG semantics.* Per-world, single state. Draws serialized by call order.
Inside a tick, the order is:
  1. Pre-tick events (fidelity zone membership rebuild — no RNG)
  2. Event handler dispatch (materialized-view updates — no RNG)
  3. Policy inference (deterministic given weights + obs + mask; sampling is
     temperature-softmax with RNG draws in a fixed agent order)
  4. Action application / cascade rules (RNG for dice rolls, damage variance)
  5. Construction / terrain updates (RNG for procedural gen if triggered)

*Agent ordering.* The existing rule ("unit processing order shuffled per tick
to prevent first-mover bias") translates to: the policy runs on all agents in
parallel (no order dependency because observations are packed from a frozen
pre-tick snapshot), but action APPLICATION is a serialized fold over agents
in a shuffled-but-seeded order. The shuffle itself consumes RNG draws from
`rng_state`, making the order reproducible.

*Per-agent RNG vs world RNG.* Proposal: stick with world RNG for now. Per-agent
RNG streams (seeded from world RNG at agent spawn) are a future optimization
for parallel policy sampling but they complicate save/load (story 32) because
each agent needs to carry its stream state. Deferred.

*Event ordering.* Events emitted within a single phase are collected into a
per-tick buffer (`world_events`, `structural_events`). The order of emission
follows agent processing order (shuffled with RNG). Handlers process the buffer
in append order. This gives deterministic cascade order.

*GPU determinism traps.* The GPU path (story 28) introduces non-determinism
risks:
- **Float associativity.** Sum reductions over large arrays on GPU use
  tree reductions with non-fixed ordering across warps → bit-different results.
  Mitigation: for reductions that feed into policy decisions, use integer
  fixed-point accumulation (scale floats to i32, sum, rescale), or sort by
  key before reduction.
- **Atomic fetch-add ordering.** If two events on the same tick both do
  `agent.mood -= X` via atomics, the final value is commutative-correct but
  the intermediate may not be observable. For materialized views this is fine
  because only the final value is read. For views that accumulate into a
  history ring, sort first.
- **Reduction shader warp counts.** Vendor-specific. Pin the reduction shader
  to a fixed workgroup size (specialization constants).

*What breaks determinism — and how the proposal handles it.*
- **Text generation** (per story 60): `ChronicleEntry.text` is allowed to
  diverge across replays. Story 60 marks text-gen events as non-replayable.
  The DSL uses numeric event IDs + parameter bags (agent IDs, quantities) as
  the replay-safe representation; text is a CPU-side render of those
  parameters and may use non-deterministic font/language models without
  breaking sim replay. Per user's framing: "text gen should not be load
  bearing. Use numeric IDs for everything important."
- **Wall-clock dependence.** Compiler forbids reading `SystemTime` or
  `Instant` in DSL code; only the tick counter is available.
- **HashMap iteration order.** `ahash::RandomState` is already seeded per
  process. If serializable, can seed deterministically from `rng_state`.
  The proposal should commit: hash_seed = (world_seed ^ tick) at state init,
  never re-seeded.
- **LLM backend** (§2.4) is explicitly non-deterministic and off the per-tick
  path.

**Gaps / open questions:**
- The GPU-determinism details above aren't in the current proposal. Needs a
  §6 "determinism under GPU compilation" addition.
- HashMap seed policy not specified.
- Policy-sampling determinism: temperature softmax + categorical sample needs
  a specified RNG stream. Current combat transformer uses a derived stream
  (hash of `(world_rng, agent_id, tick)`) per agent to parallelize safely.
  The DSL should document this pattern.

**Related stories:** 27 (rule ordering is a determinism knob), 28 (GPU
compilation introduces new determinism hazards), 32 (snapshot includes
`rng_state`), 33 (replay from seed + event log + initial state), 60
(text-gen bound on determinism).

---

#### Story E.30: Zero per-tick allocations in steady state
**Verdict:** PARTIAL (existing `SimScratch` pattern extends; three concrete malloc sites remain)

**User's framing note:** "Essential, ideally we can avoid memory allocation past
startup altogether."

This is a hard constraint. No malloc past startup. Let me inventory where the
current design still allocates.

**How the DSL supports this:**
`state.md` already documents `SimScratch` — pooled scratch buffers
reused across tick systems, cleared+refilled within each call, `Clone` returns
`Default` (doesn't duplicate scratch). Pre-pooling baseline: "~55 page faults
per tick, 220KB/tick allocator churn." This is the existing precedent. The DSL
extends it.

The per-agent struct is already fixed-size (atomic features, contextual blocks
are fixed-width from §2.1.1–2.1.7). Slot arrays are fixed K:
`nearby_actors[K=12]`, `known_actors[K=10]`, `known_groups[K=6]`,
`memberships[K=8]`, `behavior_profile` top-K=8, etc. Everything the observation
packer reads has a static upper bound.

**Implementation walkthrough — allocation sites and elimination strategies:**

**Site 1: `behavior_profile: Vec<(u32, f32)>` (state.md line 276).**
Currently unbounded sorted tag_hash → weight pairs. Grows as agent accumulates
new action tags.
- *Fix:* cap at K=16 (matches the top-K=8 observation + headroom). When a 17th
  tag would be added, evict the lowest-weight tag. Store as
  `[(u32, f32); 16]` fixed array. Weight accumulation still uses binary search.

**Site 2: Relationship records.**
`Relationship` is per-pair; an agent may know N others. Currently implied `Vec`.
- *Fix:* `known_actors[K=10]` is already the observation-visible cap; extend
  this to the storage representation. The agent's "knows about" set is
  `[Relationship; K_MAX=32]` with LRU eviction by `last_known_age`.
  Full-resolution per-pair matrix is not stored; only the top-K known
  per agent. Distant rivals are not "forgotten" — they're evicted from the
  per-agent knowledge slot when a closer relationship takes precedence, just
  as real NPCs forget unimportant others. Relationships the agent should
  never forget (spouse, mentor) are pinned.

**Site 3: `memberships: Vec<Membership>` (state.md, referenced in §2.1.3).**
- *Fix:* `K=8` cap, already matches the observation slot array. Agent can be
  in at most 8 groups simultaneously. Design constraint, enforced by the DSL.

**Site 4: Memory event ring.**
`memory_events` per agent — ring buffer of recent events agent observed.
- *Fix:* fixed-size ring buffer `[MemoryEvent; 64]` with write-index. Already
  a ring in the existing code per CLAUDE memory; just make it fixed-size at
  the type level.

**Site 5: Per-tick event buffers (`world_events`, `structural_events`).**
`state.md` documents `structural_events: Vec<StructuralEvent>` "Cleared
at tick start." That's pool-like but still a `Vec`.
- *Fix:* `SmallVec<[StructuralEvent; 256]>` with a documented overflow
  behavior (log warning + drop oldest, or pre-size with a worst-case bound).
  Since events are bounded-per-tick (per agent, each can emit at most
  K actions), worst case is `N_agents * K_actions`. Statically sizeable.

**Site 6: Chronicle.**
`chronicle: Vec<ChronicleEntry>` (bounded ring per `state.md`).
- Partly fixed — ring-bounded — but entries contain `String`. Per story 60's
  "use numeric IDs for everything important", convert to
  `ChronicleEntry { category: u8, tick: u64, entities: [u32; 4], param_ids: [u32; 4] }`.
  Text rendering is a separate CPU-side display layer with its own non-sim
  allocations (which don't count — they're out of the sim loop).

**Site 7: Quest/auction lists.**
`Quest.party_member_ids: Vec<AgentId>`. Conquest quests can have hundreds of
members.
- *Fix:* quests store an `ArcWindow` into a pre-allocated pool of agent-ID
  slots OR resolve `PartyScope::Group(g)` as a pointer to the group rather
  than materializing the member list. The latter is what
  `spec.md` Open Question 2 already asks. Recommend:
  `PartyScope::Group(g) → members-view-at-tick(g)`, computed from the
  agent's `memberships` on demand; no stored list.

**Site 8: Spatial query results.**
`query::nearby_agents(self, radius=50)` returns a slice. Currently implied Vec.
- *Fix:* use `SimScratch.snaps` — already an existing pooled buffer. DSL
  codegen routes all `query::` calls through scratch.

**Site 9: GPU buffers (story 28).**
- *Non-issue.* `GpuHarness::create_field` allocates once at startup; the
  observation / mask / logits / action buffers stay resident for the run.

**Site 10: HashMap.**
`tiles: HashMap<TilePos, Tile>`, `chunks: HashMap<ChunkPos, Chunk>`,
`surface_cache`, `cell_census` — all allocate as they populate.
- Tiles and chunks legitimately grow as the world is explored (load-time
  allocations for new chunks). This is NOT per-tick steady-state allocation
  as long as chunk loading is gated to "when NPC enters new area". For a
  truly bounded world, pre-allocate all chunks at init (small-world mode
  already does this per existing commit `fix: skip CPU chunk pre-gen in
  --world small`).
- `surface_cache` and `cell_census` are lazy caches — per
  `state.md` they're "populated lazily on HashMap miss." Fix: switch
  to the flat-grid versions already mentioned in the doc ("Candidates for
  flat-grid conversion").

**Remaining malloc sites that are HARD to eliminate:**
- New agent spawn (e.g. birth event) — has to allocate an agent slot. Fix:
  pool of `MAX_AGENTS` slots allocated at init with a free list; spawning
  pops a slot. Despawn returns the slot. This is the "fixed-cap world"
  commitment — `MAX_AGENTS=200_000` (story 31). ~200KB/agent × 200K =
  40 GB, too big; need to tune per-agent size down or use tiered storage
  (hot/cold separation already exists per `state.md`
  `entity_index`/`hot`/`cold` pattern).
- New chunk load — real allocation. Either pre-allocate all chunks (small
  world, bounded) or accept that chunk-load triggers allocation (not
  "per-tick" in steady state, because an agent rarely crosses a chunk boundary
  every tick — mostly a bulk-load event at session start).
- Save-file I/O (story 32) — serialization uses a scratch buffer, can be
  pooled.

**Concrete zero-malloc DSL constraints:**
```
agent NpcAgent {
  // All fields fixed-size
  vitals: Vitals,
  needs: [f32; 6],
  emotions: [f32; 6],
  personality: [f32; 5],
  behavior_profile: [(u32, f32); 16],      // fixed-cap
  memberships: [Membership; 8],            // fixed-cap
  known_actors: [KnownActor; 32],          // fixed-cap, LRU-evicted
  memory_events: RingBuffer<MemEvent, 64>, // fixed-cap
  ...
}

// Compile error: unbounded collection in agent struct
agent Bad { history: Vec<Event> }  // ERROR: Vec fields forbidden
```

The DSL can statically forbid unbounded fields in agent / group / quest
structs. `Vec<T>` is only allowed in scratch pools (SimScratch) or in the
world-level event ring (itself bounded).

**Gaps / open questions:**
- Event buffer worst-case sizing is a hard capacity-planning problem. If
  drops are observable (determinism break), we need a static cap proof.
  Proposal: log a tick-level event-rate metric; CI test fails if any scenario
  exceeds 80% of the static cap.
- Chunk allocation on exploration genuinely is a malloc; either accept it
  (small-world mode eliminates) or use a fixed chunk slab pool with LRU
  eviction (mirrors voxel-engine's `terrain_compute.rs` 1024-slot pool).
- Save buffer — serialization of worldstate to a `Vec<u8>` — can be pooled
  (reuse buffer across saves) or streamed to disk with a fixed chunk size.
  Needs story 32 to specify.

**Related stories:** 25 (pack kernel writes into pre-allocated GPU buffer), 28
(GPU field buffers pre-allocated), 31 (200K agents means fixed-cap slot pool
is the only viable storage), 32 (save buffer pooling), 33 (event log must be
ring-bounded or streamed).

---

#### Story E.31: Scale 2K → 200K agents
**Verdict:** PARTIAL (architectural approach is right; specific bottlenecks need quantification)

**User's framing note:** "Essential, the purpose behind this entire thing"

This is the north-star constraint. Everything above rolls up into whether the
design scales linearly.

**How the DSL supports this:**
The design decisions that make 100× scale possible:
1. **Single neural backend** (§2.4) — no per-role dispatch overhead.
2. **Batched forward** (story 25) — inference cost is one shader dispatch
   regardless of N.
3. **Fixed-cap per-agent state** (story 30) — O(N) memory, no hidden O(N²).
4. **Top-K slot arrays** (§2.1.4/5) — observation is O(1) per agent, not
   O(N_nearby).
5. **Per-pair relationships bounded by K_known=32** (story 30) — no O(N²)
   relationship matrix.
6. **GroupIndex contiguous ranges** (`state.md`) — per-settlement
   iteration is a slice, not a scan.
7. **Fidelity zones** (`state.md`) — agents far from player run at
   Background fidelity, skipping policy inference.

**Per-agent and per-tick cost breakdown at 200K:**

*Observation packing.* 1655 f32 per agent × 200K agents = 330M f32 = 1.3 GB
observation tensor per tick. On a GPU with 8 GB VRAM: fits, but leaves 6.7 GB
for everything else (weights, mask, logits, voxel chunks). **First pressure
point.** Options:
- Pack as fp16 instead of fp32 (→ 660 MB). Most observation features are
  bounded in [-1, 1] after normalization; fp16 has enough precision.
- Use sparse observation for Background-fidelity agents (subset of features).
  Currently not specified.
- Per-fidelity observation dimension (High=1655, Medium=400, Low=80,
  Background=skip). The proposal hints at this with "per-role K" but doesn't
  commit.

*Mask evaluation.* `NUM_LOGITS` ≈ 16 macro × 16 quest_type × 6 party × 10 reward
× 6 payment = ~92K combinations per agent. Per-head mask is much smaller:
`[16] + [16] + [6] + [10] + [6] + [50 target slots]` ≈ 104 bytes per agent =
21 MB for 200K agents. Fine.

*Neural forward.* Assume a modest network: input 1655, hidden 256, hidden 256,
heads summing to ~256 outputs. Parameters: 1655×256 + 256×256 + 256×256 ≈
555K weights. FLOPs per agent: ≈ 1.1M. Total: 2.2e11 FLOPs/tick at N=200K.
An RTX 3080 does ~30 TFLOP/s fp32, so one forward is ~7 ms. Fine. fp16 cuts
to ~2 ms. **Not a bottleneck.**

*Event volume.* Per agent per tick: 1 action + cascade (~3 events) ≈ 4
events/agent/tick = 800K events/tick. At 32 bytes/event, 25 MB/tick event
buffer. At 60 ticks/sec, 1.5 GB/sec event traffic. **Second pressure point.**
Options:
- Only the deltas relevant to materialized views go through the event handler
  kernel (per story 26). Rare events (QuestPosted, MarriageFormed) are CPU.
- Event buffer is ring-bounded; historical events dropped after view updates
  fold them in.
- Chronicle ring: per story 60, already bounded.

*Cascade rule cost.* CPU rules running over 800K events/tick. If each rule
is O(1) per event and there are ~50 rules, that's 40M rule-invocations/tick.
At 10 ns per invocation (branch + mutation), 400 ms CPU time. **Third pressure
point.** Options:
- Rules grouped by event-type dispatch table; only rules matching the event
  type run. Typical event matches 2-3 rules, not 50.
- SIMD-friendly rule bodies for hot rules (damage application, hp clamp).
- Fidelity-gated rules: Background-fidelity agents skip cascade entirely
  except for death/spawn events.

*Memory footprint.* At 200KB/agent × 200K = 40 GB. Too much. Options:
- **Hot/cold split** already exists (`state.md`: "`entity_index`
  sentinel sizing", `hot_entity`/`cold_entity`). Hot = packed observation
  fields (~7KB at 1655×f32) kept resident. Cold = infrequent fields (full
  memberships list, behavior_profile, memory_events, RNG stream, class
  definitions) loaded on policy-tick for High fidelity, evicted otherwise.
- With hot alone at 7KB × 200K = 1.4 GB — fits. Cold paged to SSD with LRU.
  Only High-fidelity agents need cold resident.
- Per-agent size target: hot ≤ 4KB (pack booleans, use u8 enums, drop unused
  slots). 200K × 4KB = 800 MB.

*Cross-entity queries (spatial).* `query::nearby_agents(self, radius=50)` at
N=200K is O(N) naive, O(log N) with a spatial hash. Per `state.md`,
`chunk_census` / `surface_grid` already exist as spatial indices.
voxel-engine's `src/ai/spatial.rs` provides GPU-side spatial indexing (named
in the module listing). Use that for slot-array gather on GPU.

**Scaling table (targets to validate):**

| Component | 2K cost | 200K cost | bottleneck? |
|---|---|---|---|
| Observation pack | 0.05 ms | 5 ms | No (linear, GPU) |
| Observation tensor size | 13 MB | 1.3 GB | Yes — fp16 + tiered fidelity |
| Mask eval | 0.01 ms | 1 ms | No |
| Neural forward | 0.07 ms | 7 ms | No |
| Event volume/tick | 8K events | 800K events | Partial — ring-bound, fold on GPU |
| Cascade rules (CPU) | 4 ms | 400 ms | **Yes — fidelity-gate or SIMD** |
| Cold state paging | 400 MB RAM | 40 GB | **Yes — hot/cold split required** |
| Spatial queries | 0.2 ms | 20 ms | No (log N with voxel-engine `ai/spatial.rs`) |

**Bottlenecks ranked:** (1) cold state memory, (2) cascade rule CPU cost,
(3) observation tensor VRAM, (4) event volume.

**Gaps / open questions:**
- Hot/cold split not yet specified in the DSL surface. Should be an
  annotation: `@hot field treasury: f32`, `@cold field creditor_ledger: [Creditor; 32]`.
- Fidelity-gated rules not specified. Extend §5: rules may annotate
  `@fidelity(>=Medium)` to skip at lower fidelity.
- Scale validation: no concrete plan to benchmark 200K in the proposal. Needs a
  milestone in `prototype_plan.md` (referenced but unwritten).
- Network size (d_model, layers) is implicit. For 200K, a small network
  (~500K params) is ideal; training may want larger. Specify in DSL as
  `backend Neural { h_dim: 256 }` (already shown in §3). The compiler enforces
  static size compatibility with the packed obs shape.

**Related stories:** 25, 26, 28, 30, 33.

---

#### Story E.32: Save and reload mid-run
**Verdict:** PARTIAL (serialization strategy choice needs committing; one hard decision outstanding)

**User's framing note:** "Essential"

**How the DSL supports this:**
`state.md` already distinguishes primary vs derived state:
> **Primary state** (irreplaceable): `tick`, `rng_state`, `next_id`, `tiles`,
> `voxel_world.chunks`, `voxel_world.sea_level`, `region_plan` (regenerable
> from seed), `build_seeds`, `chronicle`, `world_events`, `fidelity_zones`
> (zone definitions), `structural_events` (per-tick buffer).
>
> **Derived state** (rebuildable from primary): `entity_index`, `group_index`,
> `settlement_index`, `surface_cache`, `surface_grid`, `cell_census`,
> `nav_grids`, `max_entity_id`, `fidelity_zones[].entity_ids`.

Everything derived has `#[serde(skip)]` annotations and a rebuild path.
The DSL inherits this distinction via field annotations: `@primary`,
`@derived(rebuild=rebuild_nav_grids)`, `@scratch`.

**Implementation walkthrough:**

*Snapshot contents.* A full snapshot must contain:
1. `tick`, `rng_state`, `next_id`, `max_entity_id` — scalars.
2. All primary per-agent fields (`@primary` annotated).
3. Group definitions (memberships, standings, treasuries).
4. Quests/auctions in flight.
5. `tiles`, `build_seeds`, `voxel_world.chunks`, `voxel_world.sea_level`.
6. `region_plan.seed` only — the plan itself regenerates from seed.
7. `chronicle` ring (bounded — capped memory).
8. `world_events` ring (bounded).
9. **Materialized views (decision point below).**

*Decision: serialize materialized views, or recompute from event log on
load?*
- **Option A — serialize views.** Pro: instant load. Con: invariant
  risk — if the view derivation changes between save and load, the snapshot
  is wrong. Mitigation: schema hash covers view definitions; hash mismatch
  forces option B.
- **Option B — recompute from event log on load.** Pro: view derivation
  changes are safe. Con: the event log must extend back to the point where
  the materialized state was initialized. That's the whole simulation
  history, which is not ring-bounded. Infeasible for long-running sims.
- **Option C — hybrid with event-log horizon.** Serialize both the
  materialized-view value AND the tick at which it was last fully
  reconstituted. On load, replay events from that tick forward. Horizon
  bounded by the event log ring size. Works if views stabilize over short
  horizons.
- **Recommendation: Option A with schema-hash guard.** Rebuild from
  baseline (not event log) when hash mismatches, treating it as a migration.

*Event log in the snapshot.* If the event log is ring-bounded (say 10K
events), snapshot stores the ring + head-index. On load, derived views fold
from that ring forward — all earlier events are already baked into the
saved materialized view values. This bounds snapshot size.

*Snapshot size at 200K agents.* Assuming 4KB hot + 8KB cold per agent, 12KB
per agent × 200K = 2.4 GB per snapshot. Plus chunks (300KB each × ~5000 loaded
chunks = 1.5 GB). Plus event ring (10K × 32B = 320KB). Plus chronicle
(bounded, ~1 MB). Total: ~4 GB. Disk I/O at 1 GB/s SSD = 4 s save, 4 s load.
Acceptable for "save every N ticks".

*Incremental snapshots.* Save only changed chunks + agent deltas since last
snapshot. Reduces to ~100 MB for a typical save. More complex; defer to v2.

*Reload correctness.* The contract: `sim(seed).step(N).save()` + `load()` +
`step(M)` ≡ `sim(seed).step(N+M)`. Determinism tests
(`src/ai/core/tests/determinism.rs` per CLAUDE.md) already check this for the
combat sim; extend to world sim.

**Zero-malloc save/load (story 30 crossover).** Save writes into a pre-allocated
scratch buffer (reuse across saves). Load reads into the same
pre-allocated agent pool slots — no new allocations, just refill. Snapshot
format must be length-prefixed + deserialize-in-place. Bincode 2 supports
this with `decode_from_std_read_borrowed`.

**Gaps / open questions:**
- Commit on Option A vs C for view restoration.
- Event log size bound (story 33 dependency).
- Snapshot format — custom binary or bincode/postcard. Performance
  implications at 4 GB/save.
- GPU-resident buffers (story 28). On save, they must be downloaded.
  `GpuHarness::download` gives us this. On load, upload. Cost: ~1 GB GPU
  buffer traffic at PCIe bandwidth ≈ 100 ms. Add to save/load timing.

**Related stories:** 26 (materialized view persistence), 29 (save must
include full RNG state), 30 (snapshot format must be reallocation-free
on load), 33 (replay-from-snapshot is the same machinery as replay-from-tick-0).

---

#### Story E.33: Replay any range of ticks
**Verdict:** SUPPORTED

**User's framing note:** "Essential"

**How the DSL supports this:**
Event sourcing (§1.1) is the foundation: "current state is a fold over events
+ entity baseline." Replay is literally running the fold.

Story 60's ruling — "text gen should not be load bearing. Use numeric IDs for
everything important" — explicitly bounds replay determinism: text-bearing
events are allowed to diverge on replay.

**Implementation walkthrough:**

*Replay model: event log + RNG + baseline.*
Recorded artifacts per run:
1. Initial snapshot (story 32 save format) at tick T0.
2. Event log from T0 onward. Each event entry:
   `{ tick: u64, kind: u16, params: [u32; 4], source_agent: u32 }`.
   Numeric IDs only. No strings.
3. Tick-boundary RNG checkpoints (every 100 ticks). Bounds drift on replay
   divergence: any mismatch is caught within 100 ticks.

To replay from tick A to tick B:
1. Load snapshot at tick ≤ A (nearest).
2. Replay events from snapshot-tick to A, advancing sim step-by-step.
3. Optionally halt and emit a full snapshot at A.
4. Continue step-by-step from A to B, allowing inspection hooks.

*Event types that MUST be preserved.*
- All events that drive materialized view updates (story 26).
- All events that drive policy-observable state (positions, HP, relationships).
- All action events that trigger cascades.
- RNG state at each tick boundary checkpoint.

*Event types that can be skipped on replay.*
- `ChronicleEntry` emissions (story 60 — text is post-hoc). The tick at
  which a chronicle was emitted matters for age-based observations
  (`recent_chronicle_event_counts_by_category` in §2.1.7). So record the
  category + tick but not the text. Text regenerates on display.
- `WorldEvent::Generic{text}` — same treatment, carry category ID.

*Deterministic re-execution.* Story 29's contract makes this work: same seed +
same DSL + same input → same output. Input to a replay segment is:
  - snapshot state
  - events (numeric ID form) from the recorded log
  - RNG state at segment start
If the replay diverges, the tick-checkpoint RNG mismatches and we fail fast
with "replay divergence at tick T".

*Text-gen interaction.* User-facing display of a replay re-runs the chronicle
renderer on the numeric events; LLM-generated names for new NPCs born during
replay may differ from the original run. This is explicitly OK per story 60.
The cast of agent IDs is identical — only display strings differ.

*Storage cost.* 200K agents × 4 events/tick × 32 bytes/event = 25 MB/tick.
At 60 tps, 1.5 GB/s. Per hour of sim: 5.4 TB. **Too much to store raw.**
Compression options:
- Batched delta encoding (many events per tick are ambient decay with same
  shape) → 10-50× reduction.
- Log only the "policy-input-affecting" events, fold derivation locally.
  Any event used solely to update a materialized view can be reconstructed
  from its inputs (the agent state pre-event + the rule). Hard to formalize.
- Snapshot every 10 minutes + event log per 10-minute segment. 10-min segment
  = 900 GB uncompressed, 50 GB compressed. Still large.
- **Practical replay scope:** not "replay an entire hours-long run", but
  "replay a bug-report window of 1000 ticks (≈16 sec)". 1000 ticks × 25 MB
  = 25 GB uncompressed, 2-5 GB compressed. Feasible.

*Replay of GPU-sim.* Replay uses the same GPU path; nothing about replay
requires CPU. As long as GPU determinism (story 29 GPU-determinism traps) is
enforced, replay produces bit-identical output.

*Policy weight versioning.* Replaying a run requires the exact policy
checkpoint used originally. Snapshot metadata records `policy_weights_hash`
from §4. Loading a snapshot with mismatched weight hash → hard error.

**Gaps / open questions:**
- Storage compression strategy not specified. Needs design.
- Whether replay is guaranteed only for a bug-report window or for whole
  runs. Practical scope recommendation is bug-report window (1000 ticks).
- Partial-replay for debugging — can we replay only a *subset* of agents? No:
  agent interactions cross-cut; a partial replay gives a partial sim, which
  is the "A/B compare" story 36, not 33.
- Text-gen events that *feed back* into sim (agent speech influencing
  listener NPC). Per user: "Use numeric IDs for everything important" — so
  the text itself is not load-bearing; only the act of speech (event
  `SpeechEmitted{topic_id, speaker, audience}`) matters. The actual text is
  display-only. Compiler should forbid policy code from reading
  `event.text` fields.

**Related stories:** 26 (materialized views depend on deterministic
re-execution), 28 (GPU determinism is a replay precondition), 29 (replay is
determinism validated over time), 32 (snapshot = replay starting point),
38 (self-contained tick reproduction is replay-of-range N=1), 60 (text-gen
boundary rules).

---

### Cross-cutting themes

1. **Hot/cold split is a recurring requirement.** Stories 30, 31, 32 all
   depend on it. The DSL should specify `@hot` vs `@cold` annotations on
   agent fields as a first-class concern, not bolt-on.

2. **Event-type IDs instead of strings.** Story 60's "numeric IDs for
   everything important" rules out strings from the event log (story 33),
   materialized-view handlers (story 26), and replay artifacts. Needs to be
   elevated to a schema-level rule: **no `String` fields in `@primary`
   event or state structs.** Strings only allowed in display-time render
   code.

3. **voxel-engine is the GPU backend, not wgpu.** Update
   `spec.md` §5 from "wgpu/CUDA" to "ash/Vulkan via
   `voxel_engine::compute::GpuHarness`". Shader compilation via `shaderc`
   (already in voxel-engine's build-dependencies). SPIR-V bytes loaded via
   `GpuHarness::load_kernel`.

4. **Compile-time validation is the DSL's differentiation.** Story 27 asks
   for it explicitly. The compiler needs a static-analysis pass covering:
   event-type existence, field-write race detection, cascade cycle detection,
   schema-hash drift, and optionally invariant checking. This is a
   multi-week piece of work but it's the payoff justifying the DSL over
   hand-written Rust.

5. **Fixed-cap everything.** Zero-malloc (story 30) and 200K scale (story 31)
   both require this. The DSL surface should statically reject unbounded
   collections in agent/group/quest/event structs. Only `SimScratch` and
   world-level ring buffers may contain `Vec`. This constraint propagates
   through observation design (top-K slots already do this) and quest
   semantics (`party_member_ids` resolved from groups lazily, not stored).

6. **Determinism under GPU compilation is under-specified.** Story 29's
   core contract is supported, but story 28's GPU-compilation target
   introduces new hazards (float associativity, atomic ordering, warp-count
   dependence). Needs a dedicated §6 in the policy schema proposal.

### Suggested proposal-schema deltas

- **§5 GPU compilation** — rename "wgpu/CUDA" to
  "Vulkan via voxel-engine `GpuHarness` + SPIR-V via shaderc".
  Add "GPU determinism constraints" subsection covering float
  associativity, atomic ordering, workgroup-size pinning.

- **New §6 Rule composition** — phase-based ordering
  (pre_event / event / post_event), commutativity within phase, compile-time
  race detection, cycle detection in cascade-rule graph.

- **New §7 Static analysis / compile-time checks** — enumerate the
  compile-time validations (story 27): event-type existence, field-write
  races, schema-hash drift, unbounded-collection rejection in agent structs.

- **New §8 Hot/cold storage model** — `@hot`/`@cold` field annotations,
  cold-eviction policy, fidelity-zone integration with cold paging.

- **Extension to §4 versioning** — schema hash scope includes: observation
  shape, action vocabulary, event-type registry, rule set identity (hash
  of sorted rule bodies).

- **Extension to §2.4 backend** — explicit commitment that the backend
  trait is called with GPU-resident `PackedObservationBatch` handles
  (not owned buffers); weight loading is a separate lifecycle op, not
  per-tick.

---

## Batch F/G/I/J — Observability, Modding, Auctions, Adversarial

Scope: debugging & observability (F), modding / extensibility (G), auction / unified mechanics (I), adversarial / failure modes (J). Format per story: verdict, user's framing note, how the DSL supports the story, implementation walkthrough, gaps, related stories.

Cross-reference docs: `spec.md` (§ 2.1 observation, § 2.2 action heads, § 2.3 mask, § 4 schema versioning, § 5 GPU-scoped compilation), `spec.md` ("Auction state machine", "Auction lifecycle as events", "Open questions §§ 1, 4, 7, 8"), `state.md` (Quest, Group, ChronicleEntry), `state.md` (MemoryEvent ring buffer), `state.md` (chronicle / world_events taxonomy).

---

### F. Debugging and observability

#### Story FGIJ.34: Trace why a mask evaluated false
**Verdict:** PARTIAL
**User's framing note:** "Great."

**How the DSL supports this:**
The mask is declarative in § 2.3 — each candidate action is paired with a predicate expression built from declared views (`is_hostile`, `distance`, `groups_at_war`, `g ∈ self.leader_groups`, `relationship.valence > MIN_TRUST`, ...). Because predicates are DSL expressions rather than opaque Rust closures, the compiler has enough structure to retain a per-predicate AST: each sub-clause is an addressable node with a stable ID.

**Implementation walkthrough:**
1. The mask compiler emits two artefacts per predicate: (a) a fast boolean kernel that evaluates the conjunction to a single bit, and (b) an "explanation kernel" — the same AST with per-node result capture — that is *not* run in production but is used on demand. The bit goes into `per_head_mask_tensor[agent_idx, action_idx]`.
2. A `trace_mask(agent_id, action_idx, tick)` query reruns the explanation kernel against the captured observation snapshot for `(agent_id, tick)`. Output: an AST flattened to `[ (node_id, expr_text, inputs[{name, value}], result) ]`. The first `result=false` node in a conjunction is the "failing clause," and its inputs tell you why (`distance(self, t)=84 > AGGRO_RANGE=50`, `groups_at_war(self.faction, t.faction)=false`, ...).
3. Because every input to the mask is itself a declared view or observation field, values can be decoded back through the observation schema — there is no opaque integer that isn't named.

**Gaps / open questions:**
- Observation snapshots are needed per `(agent_id, tick)`. The ~1655 float observation is cheap per agent, but retaining every observation for every tick is expensive. Policy options: (a) keep last N ticks rolling (matches story 39), (b) re-pack on demand from event log if the agent state at tick T can be reconstructed (requires full event-sourcing discipline to hold).
- Cross-entity predicates (`t ∈ quest.eligible_acceptors`) — § 5 lists these as CPU-resident. Ensure the explanation kernel respects that boundary (policy proposal open question #15).
- "Why didn't agent X pick action Y" at tick T also requires the sampled logits, not just the mask. The mask tells you *eligibility*; the sampled action tells you *choice*. Story 34 as written addresses only eligibility — logit-level attribution is out of scope here and lives in story 39 territory.

**Related stories:** 35 (cascade), 37 (step), 38 (repro), 39 (decision history).

---

#### Story FGIJ.35: Inspect cascade fan-out
**Verdict:** SUPPORTED
**User's framing note:** "Great."

**How the DSL supports this:**
The runtime is event-sourced (README "Settled" bullet 1). Every state mutation is a typed event; cascade rules are declarative `event → emit` transforms, e.g. `AuctionResolved{kind=Item, winner=W, payment=P} → TransferGold(W → seller, P.amount) + TransferItem(seller → W, item)` (universal mechanics "Auction lifecycle as events"). Because each cascade rule names its trigger and its emitted events, the DSL compiler can assign each event instance a **(tick, sequence_idx, parent_event_id)** triple.

**Implementation walkthrough:**
1. The runtime tags every emitted event with `cause: Option<EventId>` — the event that triggered its cascade rule. Root causes (raw agent actions, scheduled physics) have `cause=None`.
2. "Fan-out" = DAG edge-traversal from a root event. `causal_tree(root_event_id, tick)` returns the transitive closure: `ChosenAction → Damage → EntityDied → SuccessionTriggered → QuestExpired → ...`.
3. Chronicle entries are emitted by cascade rules. They therefore sit at leaf positions in the causal tree and share the tree's parent chain. Filtering chronicle to "events caused by agent X's action at tick T" is a DAG subtree query.
4. Rendering is uniform: each node prints `kind + key_fields + triggered_by(parent)`, and the debug UI draws the tree.

**Gaps / open questions:**
- Event retention. Chronicle is a "bounded ring buffer" (state.md line 432); `world_events` is bounded too. A cascade whose root was flushed out of the buffer loses its parent edge. Policy: either raise retention for debug builds, or spill to an on-disk event log that's not ring-buffered.
- Cross-tick cascades. Most cascades complete within a tick, but quest completion can fire a cascade 2000+ ticks after the `PostQuest` action. If `cause` points to an event outside retention, the chain is truncated. This is annoying but acceptable for long-horizon narratives; tools should show "truncated — root was at tick T, outside retention."
- Non-replayable events (see story 60) still participate in the causal DAG — they just don't reproduce under replay. Their structural presence is what matters for debugging.

**Related stories:** 34 (why mask false), 37 (step), 38 (repro), 60 (text gen / replayability).

---

#### Story FGIJ.37: Step through a tick
**Verdict:** SUPPORTED
**User's framing note:** "Essential."

**How the DSL supports this:**
The tick pipeline is a fixed sequence of stages derivable from the policy/schema DSL. From the proposal:
1. **Observation packing** (§ 2.1): gather fields + views into the packed buffer per agent.
2. **Mask evaluation** (§ 2.3): per-head boolean tensors.
3. **Backend forward** (§ 2.4): single `evaluate_batch` call over all alive NPCs.
4. **Action sampling**: apply mask, sample from logits.
5. **Cascade**: events cascade through declarative rules.
6. **State fold**: events applied to baseline state → new tick state.

Each stage has a named boundary in the compiled runtime, so a debug driver can `step(tick, until=Stage)` and halt between stages.

**Implementation walkthrough:**
1. The runtime exposes `tick_stepper` with hooks at each stage boundary. Every hook receives a `TickDebugHandle` from which you can read: the agent observation batch (decoded back to named fields), the mask batch (per head), the raw logits per head (pre-mask), the sampled action per agent, the proposed cascade events, the applied state deltas.
2. A REPL / UI can step across stages, drill into one agent, compare logits across a mask boundary (`what did this head want before mask zeroed it?`), and watch cascades resolve.
3. Re-running a stage is permitted because each stage is a pure function of the prior stage's output. Re-running the backend with a different temperature, for example, is a debug primitive.

**Gaps / open questions:**
- Non-determinism in backend sampling. Stepping with `seed=S` and re-running the backend must produce the same logits *and* the same sampled action. The RNG stream has to be stage-boundary-addressable (seed the sampler with `hash(tick, agent_id, "sample")`), otherwise re-running a stage shifts the RNG and invalidates downstream stages. This is the same constraint combat sim already upholds (CLAUDE.md "Determinism Contract" — `SimState.rng_state` + `next_rand_u32()`).
- GPU-resident observation/mask tensors need a CPU readback at each hook for the debug path. Cheap (~132 MB/tick per the proposal). Do not try to inspect on-GPU — add a `@debug_readback` annotation to force CPU copies.

**Related stories:** 34 (mask trace), 35 (cascade), 38 (repro), 39 (decision history), 40 (flamegraph).

---

#### Story FGIJ.38: Self-contained tick reproduction
**Verdict:** SUPPORTED
**User's framing note:** "Essential."

**How the DSL supports this:**
Because the runtime is event-sourced and the schema is hashed (§ 4), a tick is fully captured by: (state snapshot at T, events emitted during T, decisions made during T, schema hash). This is smaller than a full world dump — you only need the agents and world regions the tick actually touched.

**Implementation walkthrough:**
1. `capture_tick(T)` emits a bundle:
   ```
   tick:             T
   schema_hash:      sha256 of (observation schema || action vocab || event taxonomy)
   rng_state_pre:    u64
   world_slice:      compact snapshot of touched agents, groups, quests, tiles, voxels
                     (derivable via: event_log ⟶ "which entities were read/written at T"
                      ⟶ include only those + their views' inputs)
   observation_pack: [N, OBS_DIM] packed (or elide if deterministically reconstructible
                     from world_slice + tick via obs pack kernel)
   mask_pack:        per-head masks
   logits:           pre-sample raw logits from backend
   actions:          sampled actions per agent + per-head choices
   cascade_events:   ordered list of events emitted during the tick
   ```
2. A reproduction harness loads the bundle, reapplies `rng_state_pre`, runs the tick via the normal runtime, and diffs against the captured `cascade_events` + resulting state. Divergence means non-determinism.
3. The bundle format is compressed; most fields are reconstructible from `world_slice + rng_state + schema_hash + actions`. Observations + mask are redundant in principle but useful for debug UIs that don't want to rerun the pack.

**Gaps / open questions:**
- Bundle size. 20K agents × 1655 floats × 4 bytes = 132 MB observation alone. Real bundles for a *bug report* are usually single-agent slices — we need a "narrow bundle" that includes only the agent(s) in question and their referenced entities (spatial slot neighbours, known_actors, quest party members).
- Inclusion closure for "world_slice": transitive. Agent → its group → group's standings → other groups → their leaders (via `known_actors` back-references). Needs a predefined closure function or risk missing inputs.
- Replay of non-replayable events (story 60): the bundle records them as *outputs*, not as inputs to reproduce. Replay reproduces the structural event (numeric IDs) but not the text — text gen is expected to differ.

**Related stories:** 35 (cascade), 37 (step), 39 (history), 60 (text gen / replay scope), 64 (schema hash mismatch).

---

#### Story FGIJ.39: Per-agent decision history
**Verdict:** SUPPORTED (with retention policy TBD)
**User's framing note:** "Essential."

**How the DSL supports this:**
Observation, mask, logits, and action are all named and typed artefacts per stage boundary. The DSL compiler knows the schema; retention is a policy wrapped around the tick pipeline, not a per-system hack.

**Implementation walkthrough:**
1. Per agent, per tick, emit a `DecisionRecord { tick, agent_id, observation_hash, mask_summary, chosen_action, chosen_per_head_logits_top3, sampling_temp }`.
2. Two retention layers:
   - **Hot ring** (last ~500 ticks): full records with observation payload → debug UI can replay any agent's recent decisions.
   - **Cold log** (append-only, compressed): headers only (`observation_hash + action`) for historical replay. Rehydration of cold observations requires re-running the observation pack kernel against a reconstructed state (= full event-sourced replay), which the schema hash makes safe.
3. Query primitives: `agent_history(id, [t_from, t_to])`, `actions_by_kind(id, ActionKind)`, `never_chose(id, ActionKind)` (useful for "why does this agent never Attack?"), `pattern_search(id, sequence)` (for weird patterns like `Move → Move → Move → Idle`).

**Gaps / open questions:**
- 20K agents × 500 ticks × ~1655 floats = 66 GB hot ring — too much. Either (a) only retain "interesting" decisions (heuristic: agent changed action kind, or mask was near-tie), (b) retain full for a subset (debug focus list), (c) retain only header + observation_hash and accept that full replay is needed for rehydration. Option (c) is the cheapest and composes with the event-sourced design.
- "Weird patterns" detection is a debug-UI feature; DSL just needs to expose the history query. Policy open question #15 (cross-entity predicates) also applies: filtering history by "agent was in quest X at the time" requires a secondary index.

**Related stories:** 34 (mask trace), 37 (step), 38 (repro), 60 (what data is replayable).

---

#### Story FGIJ.40: Flamegraph attribution to DSL declarations
**Verdict:** SUPPORTED
**User's framing note:** "Essential."

**How the DSL supports this:**
The compiler-generated code is the attribution unit. Every named DSL declaration — `view::mood`, `mask Attack`, `observation.slots nearby_actors`, `cascade rule on AuctionResolved` — compiles to a named function (or named kernel). Timing is simply instrumentation at the declaration boundary.

**Implementation walkthrough:**
1. Compiler emits each declaration as a function with a stable symbol derived from the DSL path (`view::mood::eval`, `mask::attack::eval`, `slot::nearby_actors::pack`, `cascade::auction_resolved::apply`).
2. A `tick_profile` harness wraps each call with `scope_begin(sym) / scope_end(sym)`; ticks emit a flat trace of `(sym, start_ns, dur_ns, tick)` records. These feed a standard flamegraph renderer.
3. Per-declaration aggregates: percentile durations, invocation count, cost-per-agent. The per-tick total is the sum of all scope durations (modulo parallelism).
4. GPU kernels have their own timing channel (`@gpu_kernel` annotation in § 5 — observation pack, mask eval, neural forward, event-fold views). These show up as "GPU stage" scopes on the timeline with async dispatch + readback markers.

**Gaps / open questions:**
- Parallel GPU dispatch vs CPU cascade bookkeeping — flamegraph has to show both lanes distinctly. Standard: parent scope "Tick T" has children "CPU cascade" and "GPU observation pack" (overlap implied).
- Inlining by the Rust compiler can erase declaration boundaries. Need `#[inline(never)]` on the generated declaration functions in profiling builds, and a release-with-profiling build configuration.
- "view::mood: 4% of tick" requires us to know which agents' observations triggered `mood` this tick. Because observation pack walks all alive agents, attribution is per-kernel not per-agent — fine for optimisation, not fine for narrative debugging. If we need per-agent attribution, add a `@agent_scope` counter rather than wrapping each agent in a scope (which would balloon the profile).

**Related stories:** 37 (step), 38 (repro). Orthogonal to the others — this is about perf, not correctness.

---

### G. Modding / extensibility

#### Story FGIJ.41: Add a new ItemType
**Verdict:** PARTIAL
**User's framing note:** "Essential."

**How the DSL supports this:**
`Item` is one of the three first-class entity types (README "Settled" bullet 9; `systems.md` treats items as full entities with IDs emitted by `ItemCrafted{crafter, item_entity_id, quality, slot}` and despawned by `ItemBroken`). An item composes along orthogonal dimensions:

1. **Identity** — type/kind tag, rarity, base stats, slot.
2. **Ownership / path history** (README: "Item — path-dependent owner / durability / history").
3. **Event reactions** — ESSENTIAL physics: durability integration per tick, `ItemBroken` at `durability <= 0`.
4. **Derived view participation** — `effective_quality(item) = base_quality × durability_frac`, and `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac` (systems.md line 551).
5. **Mask interactions** — `ClaimItem(item_id, slot)`, `EquipItem`, `UnequipItem` actions are masked by ownership, settlement presence, slot compatibility, and upgrade opportunity.
6. **Cascade participation** — `ItemCrafted` → spawn entity. `ItemBroken` → despawn + `ItemUnequipped` cascade. `ItemTransferred` → ownership update + reputation cascade (depending on transfer context: trade/gift/loot).

**Implementation walkthrough — adding "SpellScroll":**

A modder adds a `.item` declaration (or a section of an item DSL file):

```
item SpellScroll {
  kind:      Consumable
  rarity:    {Common, Rare, Legendary}
  base_stats { spell_power: f32, durability: f32 = 1.0 /* single-use */ }
  slots:     [hand, inventory]
  events {
    on ItemCrafted { emit NamedEntity { id, name: gen_name("scroll", rng) } }
    on ItemUsed(by: AgentId) {
      emit CastSpell { caster: by, spell_id: self.spell_id, power: self.spell_power }
      emit ItemBroken { item: self.id }    // single-use cascade
    }
  }
  mask {
    UseItem(self)   when self in by.inventory
                         ∧ by.creature_type.can_speak
  }
  derived {
    display_text: "Scroll of " + spell_name(self.spell_id)
  }
}
```

What this touches in the DSL surface:
- **Adds a variant to `ItemKind`** — the enum is open-ended (append-only per § 4) so adding one doesn't renumber existing variants. Schema hash bumps (story 64).
- **Registers an event handler** on the existing `ItemUsed` event — `ItemUsed` isn't a new event, it's a new *reaction*. New events (like `CastSpell` if it's not already defined) would also need registration and schema-hash bump.
- **Adds a mask entry for `UseItem` when target is SpellScroll** — because the mask predicate is an expression and `UseItem(t)` already exists in the micro action vocabulary, the modder doesn't need to add an ActionKind. They narrow an existing predicate with an item-type branch.
- **Adds a derived view** `display_text` — purely cosmetic (story 60: text is not load-bearing).
- **Observation impact** — none direct. Items appear in `self.inventory commodity counts` and `equipped_items` summaries. A new ItemKind widens the `item_kind_one_hot` vector in item-slot features; either expand the one-hot (schema hash bump) or leave unused slots reserved (graceful extension).

**Gaps / open questions:**
- **One-hot vector widening.** `spec.md` § 2.1.1 includes `creature_type_one_hot(~8)`; item kinds would have an analogous vector. Adding a new kind changes its width — this is a schema-hash-breaking change per § 4. Mitigation: reserve slack slots in one-hot vectors (`creature_type_one_hot(16)` with some unused), or accept that mods bump the schema and require retraining/migration.
- **Event handler conflicts.** If two mods add handlers for `ItemUsed`, what order do they fire? Either (a) declarative `on ItemUsed { priority: N }` with priority lanes, or (b) rejectandrequire mods to compose handlers explicitly. User's note on story 43 ("make the entire system modifiable, they can add or delete rules themselves") suggests (b): mods own their rules, and conflicts are a load-time error.
- **Effects system (combat) already has a `.ability` DSL** (CLAUDE.md "Effect System"). If items reference abilities, the world-sim item DSL and the combat ability DSL need a shared identifier space.
- **Stability of entity IDs across reloads** — covered in story 44.

**Related stories:** 43 (override physics rule — user rejected as syntax sugar only), 44 (hot reload), 64 (schema hash on vocab extension).

---

#### Story FGIJ.44: Hot-reload mod changes
**Verdict:** PARTIAL
**User's framing note:** "Good."

**How the DSL supports this:**
Because the DSL is the source of truth for observation schema, masks, cascade rules, and views, a reload regenerates the compiled artefacts. The event-sourced runtime means the *state* is separable from the *rules* — state lives in the world store + entity baselines + event log; rules live in compiled DSL.

**What survives a reload:**
- All primary state: tick, rng_state, entities, groups, quests, chronicle, world_events, tiles, voxels (everything listed under "Primary state" in state.md "Summary").
- Active entity references — if `ItemKind` gets a new variant but all existing items' kinds still exist, every existing item remains valid.
- Ongoing quests (their `quest_type` enum members either still exist or were append-only extended).
- Agent memberships (`Group` kinds still exist).
- Event log (existing events still decode against the new event taxonomy if the taxonomy is append-only).

**What doesn't survive:**
- **Removed variants.** If a mod deletes an `ItemKind`, existing items with that kind are orphaned. Policy: reloads that remove enum variants fail unless a migration function is provided (move all instances to a fallback kind or despawn them).
- **Renumbered variants.** Must not happen. Schema enforces append-only variant indexing (§ 4 "Append-only schema growth").
- **Model checkpoints trained against an older schema hash.** Either the new schema is strictly append-compatible (old model still loads via zero-padding new fields per § 4.2), or the model needs retraining. A reload that changes observation width beyond pad-compatibility blocks hot reload of the model.
- **In-flight cascade state mid-tick.** Reload must happen between ticks, never during. The cascade queue is ephemeral per tick — no reload can happen with events pending.
- **GPU-resident compiled kernels.** Observation pack, mask eval, neural forward (§ 5). These recompile; kernel recompilation latency is the dominant hot-reload cost.

**Implementation walkthrough:**
1. Modder saves a `.item` or `.ability` or `.policy` file. File watcher triggers `hot_reload` at the next tick boundary.
2. **Validate the new schema.** Compute new `schema_hash`. Diff against current: (a) observation fields added? Append-only ⟶ ok. (b) Events added? Append-only ⟶ ok. (c) Enum variants added? Append-only ⟶ ok. (d) Anything removed or renumbered? Fail the reload, log, keep running old rules.
3. **Recompile affected kernels.** Observation pack kernel if schema widened; mask kernels if mask predicates touched; cascade tables if new event reactions.
4. **Reconcile existing state.** If new observation fields are added, they read as zero for pre-existing entities (per § 4.2 zero-padding). If new cascade rules match past events, they do *not* fire retroactively — cascade rules apply to future events only. This is the principled non-retroactive invariant.
5. **Model layer.** If the neural backend's `schema_hash` matches the new compiled schema (zero-padding compatible), keep it. Otherwise warn loudly — policy goes to a utility fallback until a compatible model is loaded.

**Gaps / open questions:**
- **Cascade rule additions that could have fired earlier.** "My new mod says ItemBroken emits a Dust particle cascade. Should broken items from before the reload retroactively spawn dust?" Principled answer: no. Rule changes are forward-only to preserve deterministic replay.
- **What if the reload changes a derived view's formula?** E.g. `is_hostile` predicate broadens. Derived views are pure functions of state — no retro-fire needed; the new formula applies from the next read. But any cached observation tensors need re-packing, so effectively the next tick naturally picks up the change.
- **Event handler priority / ordering.** See story 41. Hot reload must produce a stable ordering so that the same events fire in the same order before and after the reload (for a given tick and set of rules).
- **Chronicle text reformatting.** Old chronicle entries retain their original text; new entries use the new text template. No retroactive edit. This aligns with story 60 (text is not load-bearing).

**Related stories:** 41 (ItemType add), 43 (override rule — rejected), 64 (schema hash / vocab versioning).

---

### I. Auction / unified mechanics validation

#### Story FGIJ.55: Settlement commodity auction (REFRAMED)
**Verdict:** PARTIAL — reframed per user instruction
**User's framing note:** "I am not aligned with this so much as I think the entire economy should operate on a bidding system. I don't mind if some emergent behavior causes community auctions, but I don't want a physics rule saying every 100 ticks a settlement will put all its merchants in the town square and they will dump inventory."

**Reframe:** No scheduled physics rule. Auctions emerge from agent decisions.

**How the DSL supports this (emergent pathway):**

Auction state exists as a first-class primitive (`spec.md` "Auction state machine" + "Auction lifecycle as events"). There is no top-level periodic kernel that posts auctions. Auctions form when an agent chooses `PostQuest{type=Trade, ...}` or when an agent emits a direct `PostAuction` action (listed in `systems.md` line 124 as `PostAuction(item_id, reserve_price, deadline)`). In both cases the initiating act is an **agent policy decision**, not a scheduled system tick.

Supply-side surplus (e.g. a miner agent with too much iron) drives the `PostAuction` or `PostQuest{type=Trade}` decision via observation features — `self.inventory.iron > surplus_threshold`, `settlement.demand_for(iron) > 0` (derived view). The mask passes `PostAuction` when the agent has inventory + visibility into a market. Demand-side agents observe the auction in their `known_quests` / `observable_auctions` slot and emit `Bid(auction_id, ...)` if the good matches their need and their price belief.

**Implementation walkthrough (emergent auction):**
1. Agent A's observation includes `self.inventory.commodities`, `price_beliefs` (8 commodities), `settlement.stockpile`, `known_groups` (potential buyers). A's policy scores `PostQuest{type=Trade, target=CommodityKind::Iron, reward=Gold(price)}` highly because A has surplus iron.
2. A emits the action. Cascade emits `AuctionPosted{auction_id, kind=Commodity, seller=A, item=Iron, quantity, reserve_price, deadline=tick+20, visibility=settlement}`.
3. Every agent in `visibility` sees the auction next tick in an observation slot (`observable_auctions[K=8]`). Their mask passes `Bid(auction_id, payment, ...)` if `self.gold >= reserve ∧ auction.deadline > now` (§ 2.3).
4. Demand-side agents with price_beliefs lower than the reserve ignore it (logits low); agents needing iron and believing the price fair emit `Bid`. Cascade emits `BidPlaced{auction_id, bidder, payment}`.
5. At `deadline`, auction resolves via `HighestBid`. Cascade: `AuctionResolved{auction_id, winner=W, payment=P}` → `TransferGold(W → A, P)` + `TransferItem(A → W, Iron, qty)`.
6. No physics rule scheduled the auction. Agent A chose to sell; that choice composed into an auction because the `Trade`-quest cascade rule turns it into one.

**Why community auctions still emerge naturally:**
- Many agents with surplus of the same commodity post simultaneously; the market clears across them.
- If agents learn that Tuesday-morning auctions are common (from belief formation over observed history), they show up. The "marketplace" is behavioural, not scheduled.
- Festivals / seasonal events can trigger many agents to post (`season=harvest` → many `PostQuest{type=Trade, item=grain}`), producing the feel of a community auction without a scheduler.

**Gaps / open questions:**
- **Auction visibility.** `visibility=settlement` requires a cheap check (`self.home_settlement_id == auction.settlement_id`) or a spatial radius. See `spec.md` "Open questions" § 4 (quest discovery: push vs pull).
- **Too many tiny auctions.** If every 0.5-iron surplus becomes an auction, the auction buffer explodes. Mitigation: `reserve_price` threshold in the mask (`self.gold_value_of_good >= MIN_AUCTION_VALUE`), plus a policy-learned bias against trivial auctions.
- **Fast-path for "simple immediate trade."** See spec.md "Risks" — a trivial 2-party trade going through a full auction is overkill. Candidate: private 2-party `Bid` in an ad-hoc auction (already in the universal mechanics examples, line 83 of that doc).
- **Price belief convergence.** Settlement-level prices arise from the clearing prices of auctions. Agents update `price_beliefs` from observed `AuctionResolved` events (memory.record_event → belief_formation). This is the *emergent* price view the original story wanted — but the mechanism is "agents watch auctions" not "settlement runs market every tick."

**Related stories:** 56/57 (user rejected as too abstract / too implementation-specific), 58 (Service auction variant), 54 (Diplomatic auction).

---

#### Story FGIJ.58: Mercenary hiring as auction
**Verdict:** SUPPORTED
**User's framing note:** "Great."

**How the DSL supports this:**
`AuctionKind::Service` is defined in `spec.md` line 231. A `PostQuest{type=Service, ...}` action composes with the auction state machine to produce a hiring workflow. The `Bid → AcceptQuest` cascade is the closing move.

**Implementation walkthrough (hiring 5 fighters for a Conquest quest):**
1. **Settlement leader** observes weak settlement defences + an outstanding Conquest quest requiring more manpower. Leader's policy emits `PostQuest{type=Service, party_scope=Group(my_faction), target=Pair(Agent(my_faction.leader), Location(battlefield)), reward=Gold(500), terms=ExclusiveAcceptance=false, deadline=tick+100}` with a role spec `role_in_party: Fighter, slots_remaining: 5`.
2. Cascade: `QuestPosted{quest_id, ...}` + `AuctionPosted{auction_id, kind=Service, seller=leader, item=ServiceSlot{quest_id, role=Fighter, slots=5}, reserve=50, deadline=tick+100}`.
3. **Idle adventurer NPCs** — agents with `self.work_state == Idle` (or near-idle), `self.inventory.gold < target`, and appropriate skill — see `observable_auctions[K]` in their observation. Their mask passes `Bid(auction_id, payment=Service(Promise{contract_fulfillment}))` when:
   - `self.creature_type.can_fight` ∧ `self.skill_level >= role.min_skill` ∧ `¬self.in_active_quest_excluding(Service)` (or the quest culture permits dual-tracking).
4. Qualified agents `Bid` with an asking price — payment is a **future service** at the leader's `reward` rate. Their policy factors in personal gold need, reputation with the leader, distance to battlefield, perceived risk of the Conquest quest.
5. At `deadline` (or once 5 winning bids accumulate under `FirstAcceptable` resolution), cascade emits `AuctionResolved{winner=W, payment=P}` for each slot. Per-winner cascade: `BidAccepted → AcceptQuest(quest_id, role=Fighter)` on W's behalf.
6. Once `AcceptQuest` fires, W is added to `quest.party_member_ids`. W's next-tick observation now shows `active_quests` including the Service contract. Their per-tick decisions (move toward battlefield, attack enemy) are gated by the quest context (§ 2 "How war is a quest plays out" walkthrough).
7. On quest completion: cascade `QuestCompleted → TransferGold(leader → W, reward)` for each winner.

**Mask for "currently looking for work":**
```
mask {
  Bid(auction, ...) when auction.kind == Service
                     ∧ self.work_state ∈ {Idle, SeekingWork}
                     ∧ self.gold < ambition_threshold
                     ∧ self.creature_type ∈ auction.eligible_roles
                     ∧ auction.deadline > now
                     ∧ ¬self.in_active_quest_type(Service)   // optional: exclusive
}
```

**Gaps / open questions:**
- **Multi-slot auctions** (`AuctionResolution::Allocation` from line 241 of universal_mechanics). `FirstAcceptable` with 5 slots is more natural for hiring than `HighestBid` (we don't want the 5 richest mercenaries — we want 5 fighters willing to work below `reserve`). Implementation: N winners chosen in order of bid arrival past reserve threshold until slots fill.
- **Payment semantics.** `Payment::Service(ServicePromise)` from line 202 — the mercenary "pays" with a service promise and receives gold from the leader. This inverts the usual auction: buyer has gold, sellers have labour. Need clarity in the auction state machine which side is `seller` vs `bidder`. For a Service auction: the poster is buyer, bidders are sellers (of labour).
- **Withdrawal**. A mercenary who changed their mind mid-contract — does `WithdrawQuest` apply (spec.md "Open questions" § 7)? Likely yes, with reputation / grudge cascade.
- **Discovery cadence**. Auction frequency (`Open questions § 8`): Service auctions post when an NPC decides; no fixed cadence. Discovery is push + pull — faction members get pushed, open-market discovered via spatial/observable slot.

**Related stories:** 55 (commodity auction reframe), 54 (Diplomatic auction), 49 (betray for personal gain — mercenary defection is the same machinery), 41 (items, if service includes "and take this loot").

---

### J. Adversarial / failure modes

#### Story FGIJ.59: Malformed action drops safely
**Verdict:** SUPPORTED
**User's framing note:** "Essential, keeps the runtime validation of the existing system."

**How the DSL supports this:**
The action pipeline has two validation points: **sample-time** (mask zeroes invalid logits) and **execute-time** (the action is re-validated against the *current* state when it fires, in case the world changed between sample and execute within a cascade). Both points are declarative — the mask is DSL (§ 2.3), and the execute-time re-validation reuses the same mask predicate.

**Implementation walkthrough:**
1. **Sample-time.** Backend emits logits; sampler applies the per-head mask; masked-out actions have probability zero. An agent *cannot* sample `Attack(dead_entity)` because the mask on `Attack(t)` requires `t.alive`.
2. **Stale-target case.** Between sampling (start of tick) and execution (later in tick, after cascade rules have fired), `t` may have died or left the arena. The action pipeline re-validates at execute time:
   ```
   execute(action, agent, state):
     if not mask_still_valid(action, agent, state):
       log_warn("action dropped stale: action=<Action> reason=<predicate>")
       emit ActionDropped { agent, action, reason, tick }
       return                   // fall through; next action in queue or NoOp
     apply_action(action, state)
   ```
3. **Invalid pointer.** The action has `target: PointerIndex` indexing into observation slots (§ 2.2). If the slot references an ID that no longer exists in the entity table, re-validation fails at the `t.alive` / `t ∈ entity_table` check. Same fall-through path.
4. **Out-of-schema action.** If a model produces a categorical action index outside the declared `ActionKind` range (e.g. a model trained on an older schema hash reaching for a removed kind), the sampler clamps or drops (see story 64 — schema hash mismatch should fail loud at model load, not silently at runtime).
5. **ActionDropped** is a first-class event so it appears in cascade logs, decision history (story 39), and chronicle if the drop was game-relevant.

**Error semantics:**
- **Drop, don't crash.** Log at warn level. Emit structured event for downstream tooling.
- **Determinism preserved.** Two runs with the same seed drop the same actions at the same ticks. The drop decision is deterministic because it's based on `state + predicate`, both deterministic.
- **Replay-safe.** `ActionDropped` is a replayable event; it reproduces under story 38 repro bundles.
- **Rate-limit logging.** A broken model that spams invalid actions would blow out logs. Emit one `ActionDropped` with a count per agent per tick rather than per drop; keep the event log sparse.

**Gaps / open questions:**
- **What if the action-that-would-have-been-valid is now forbidden by the mask but has no alternative?** The agent just falls through to NoOp for the tick. Policy impact: under RL training, the reward signal reflects this (no op = no progress); agent learns to not sample stale targets. This is the right outcome.
- **Non-pointer argument validation.** `pos_delta: vec2 ∈ [-1, 1]²` — a model emitting a value outside the range should be clamped, not dropped. Continuous heads have different validation semantics than categorical ones.
- **Adversarial-model diagnostic.** A model producing ≥5% drop rate is pathologically broken; the runtime should surface this to the ML pipeline. Add a `drop_rate_per_agent` metric next to the existing decision-history stream (story 39).

**Related stories:** 37 (step — drops visible per stage), 39 (decision history includes drops), 60 (drops are replayable; text is not), 64 (load-time schema check prevents in-flight drops).

---

#### Story FGIJ.60: Determinism bound on text generation
**Verdict:** SUPPORTED — with strict ID discipline
**User's framing note:** "Essential, text gen should not be load bearing. Use numeric IDs for everything important."

**How the DSL supports this:**
The spec.md § 4 names schema hash + observation versioning. The universal mechanics doc treats events as typed structures. Text lives at the chronicle boundary (state.md "ChronicleEntry", state.md "chronicle: Vec<ChronicleEntry>"), which is an *observer* artefact, not a simulation input. The rule: **anything the simulation reads must be a numeric ID; text is cosmetic output only.**

**Data carrying text vs IDs — inventory:**

| Field | Load-bearing? | Type |
|---|---|---|
| `Agent.name` (state.md line 230) | **No** — cosmetic. Numeric `AgentId` drives all sim logic. | `String` for UI; regenerable from `(entity_id, seed)` |
| `Agent.class_tags` (line 274) | **Yes** — ability unlock, behavior bias. | Lowercase `Vec<String>` acting as tags — these should become `Vec<TagId>` (FNV-1a hash at compile time per state.md "Tag hashing" note) |
| `Agent.archetype` (line 318) | **Yes** — ability unlock, stat scaling. | Should be `ArchetypeId` not string |
| `Group.name` (state.md line 579) | **No** — cosmetic. `GroupId` is the referent. | String |
| `Quest.name` (line 151) | **No** — UI title. `quest_id` is the referent. | String |
| `QuestPosting.name` (line 179) | **No** — UI title. | String |
| `Building.name` (state.md line 355) | **No** — cosmetic. `BuildingId` + `building_type` drive mechanics. | String |
| `ChronicleEntry.text` (state.md line 343) | **No** — purely narrative output. | String, derived from `(category, entity_ids, template_id, tick)` |
| `WorldEvent::Generic { category, text }` (line 350) | **No** for `text`; **Yes** for `category` | `ChronicleCategory` enum (numeric), text is display-only |
| `DungeonSite.name` (line 265) | **No** — cosmetic | String |

**The rule in one sentence:** events encode **IDs and typed enums**; chronicle entries encode **IDs plus a template reference**; display code renders templates to user-visible strings outside the sim loop.

**Segregation of chronicle text:**
- `ChronicleEntry.text` is *generated* at event-capture time, but its generation is explicitly marked non-replayable. The chronicle entry's replayable content is `{ tick, category, entity_ids, template_id }`; the `text` field is recomputed from those four on demand for UI. Replay produces identical `template_id` + IDs; the rendered string may differ if the template library changes but the sim state doesn't.
- For LLM-generated prose: stored separately in a `chronicle_prose: HashMap<ChronicleEntryId, String>` side channel. This channel is not part of the deterministic fold. Two runs with the same seed produce identical `ChronicleEntry` records and may produce different prose strings — that's fine because nothing in the simulation reads the prose.

**Replay semantics:**
1. **Replayable events.** Everything in `WorldEvent`, cascade events, action events, memory events. All fields are IDs + typed enums + scalars. Full determinism guaranteed by `rng_state` + event log.
2. **Non-replayable events.** Events whose emitted `String` content is LLM-generated or non-deterministic. Specifically the *text field* is non-replayable; the *structural event* is replayable. So `ChronicleEntry { tick, category, entity_ids, text }` replays the first three; `text` is regenerated deterministically from a template, or regenerated non-deterministically by an LLM (if the chronicle uses LLM prose).
3. **Bundle format** (story 38) stores structural content only. Text is regenerated at playback.
4. **Tests.** Determinism tests compare event logs by `(tick, kind, structural_fields)` and ignore `text` fields.

**Implementation walkthrough:**
1. The DSL marks chronicle text + any LLM-emitted strings with `@non_replayable` annotation.
2. The runtime's event log stores `(EventId, event_kind, structural_payload, optional_text_handle)`. The text handle points to a side table that's excluded from equality checks and repro bundles.
3. When the story 38 repro bundle is captured, the text side-table is omitted. Replay produces structurally identical output; text may differ and that's expected.
4. Story 59 (action dropped) emits a structured event, never a string reason. The "reason" is an enum: `DropReason { TargetDead, OutOfRange, StaleMask, MaskNowFalse{predicate_id} }`. The human-readable text is rendered by tooling from `predicate_id`.
5. **Memory beliefs** (state.md line 152 `MemEventType` variants) are enums, not strings — good, already compliant.

**Gaps / open questions:**
- **Existing `String` fields that need demotion.** `class_tags: Vec<String>` and `archetype: String` on agents should convert to ID-based representations with a separate string table for display. This is a pre-existing tech-debt item called out in the state doc ("tag hashing" note).
- **User-authored content (mods, quest names).** A mod defines `item SpellScroll { name: "Scroll of Fire" }`. At load, `"Scroll of Fire"` interns into a `StringId`. The sim sees `StringId`; display renders the string. Hot reload can replace the string table without replay invalidation because IDs remain stable.
- **Chat / dialogue systems** (if the sim grows to include NPC dialogue). These are inherently text-heavy and MUST be in the non-replayable side channel; the *decisions* driving dialogue must be structural (what `topic_id`, what `sentiment_enum`, what `target_id`).

**Related stories:** 35 (cascade — chronicle is a cascade leaf), 38 (repro bundle excludes non-replayable text), 59 (drop events structural, not textual), 64 (schema hash covers structural fields only).

---

#### Story FGIJ.61: Group dissolution with active quests
**Verdict:** SUPPORTED — with explicit per-role cascade
**User's framing note:** "ESSENTIAL."

**How the DSL supports this:**
`DissolveGroup` is a first-class event (state.md line 404: "JoinGroup/LeaveGroup, FoundGroup, DissolveGroup"). Groups own active quests (line 617: `active_quests: Vec<QuestId>`). Quests reference parties via `party_scope` and `party_member_ids` (universal_mechanics.md line 156). The cascade on dissolution is declarative.

**Principled policy on outstanding quests scoped to a dissolved group:**

The dissolution cascade differentiates by **quest role** (is the dissolved group the *poster* or the *party*?) and by **quest type**.

```
cascade on DissolveGroup(g, reason):
  for q in g.active_quests:
    match (q.poster_scope, q.party_scope, q.type):
      // Group is the poster (the quest-giver)
      (Group(g), _, _) where q.status == InProgress:
        emit QuestCancelled { quest_id: q.id, reason: PosterDissolved }
        // Party members who accepted lose reward claims; may get partial payout
        // from q.escrow if terms permit.

      // Group is the sole party and quest type admits replacement
      (_, Group(g), type ∈ {Hunt, Escort, Deliver, Gather, Rescue}):
        // Cancel — the contract was with "this group."
        emit QuestCancelled { quest_id: q.id, reason: PartyDissolved }
        // Poster may re-post if still interested.

      // Group is the sole party and quest IS the group's raison d'être
      (_, Group(g), Conquest):
        // War-party dissolves → war ends unfinished.
        emit QuestAbandoned { quest_id: q.id, war_outcome: Stalemate }
        // Cascade: rival-group standings update to Tense (not AtWar),
        //         participants get war-exhaustion emotion bumps.

      (_, Group(g), MutualDefense):
        // Treaty dissolves with the group.
        emit TreatyDissolved { treaty_id: q.id, between: q.target_group }
        // Standing returns to prior value.

      // Individual-scoped quests with membership of the dissolving group
      // are untouched — individuals' quest participation isn't group-scoped.
      _:
        pass
  // Finally:
  emit GroupDissolved { group_id: g, tick }
  for member in g.members:
    emit LeaveGroup { agent: member, group: g, reason: DissolvedUnderThem }
```

**Why this is the principled choice:**
- **Transfer is rarely the right default.** If a Conquest quest's party dissolves, handing the war to a random successor group is a narrative non-sequitur. Dissolution is an act of finality; quests scoped to the dissolved entity end with it.
- **Cancellation preserves cascade structure.** `QuestCancelled` is already part of the cascade vocabulary. No new event type needed.
- **Individual quests survive.** An agent with a personal pilgrimage who happens to be in a dissolved guild still has their pilgrimage. Multi-membership (per README "Settled" bullet 10) means an agent's identity isn't bound to one group.
- **Narrative richness.** War-party dissolution → `QuestAbandoned{war_outcome: Stalemate}` → chronicle entry "The war between A and B ended inconclusively when A's warband fractured" — emergent, not authored.
- **Reputation accounting.** Cancellation from poster-side triggers reputation hits on the poster (broke contract); cancellation from party-side triggers reputation hits on accepted-but-unfulfilled party members. Both are cascade leaves.

**Implementation walkthrough:**
1. Leader of group G (or governance system) emits `DissolveGroup(G, reason)` action.
2. Cascade fires the match block above over G.active_quests in order.
3. For each `QuestCancelled` / `QuestAbandoned` / `TreatyDissolved`: further cascades run (reputation updates, emotion bumps, chronicle entries).
4. `GroupDissolved` emitted last. `LeaveGroup` fires for each member in deterministic order (sorted by agent id).
5. Historical: `group.dissolved_tick` set, group moves to the "dissolved but referenced" pool (chronicle entries still reference it by id).

**Gaps / open questions:**
- **Nested quests.** A quest that spawned sub-quests (a Conquest producing multiple Battle sub-quests). Dissolution of the parent → cancel children? Probably yes, recursively. Needs explicit child tracking which the current schema doesn't carry — open design point.
- **Escrow / held rewards.** `AuctionPosted` may have held `reserve_price` in escrow; if the group owning the auction dissolves, who gets the escrow? Candidate: return to the poster's individual wallet if posted by Group-but-via-member; forfeit to world treasury if truly group-owned.
- **Dissolution triggered by conquest.** If G is dissolved because it *lost* a war, `DissolveGroup(G)` is itself a cascade consequence. The cancellation of G's pending quests is then secondary cascade — this works fine; events form a DAG.

**Related stories:** 35 (cascade visibility on dissolution), 46 (succession on leader death — related but distinct mechanism), 62 (agent death mid-quest), 49 (defection as LeaveGroup).

---

#### Story FGIJ.62: Agent death mid-quest
**Verdict:** SUPPORTED — with extension for "agent is quest criteria"
**User's framing note:** "ESSENTIAL, unless agent is part of quest criteria."

**How the DSL supports this:**
`EntityDied` is a typed cascade event; `Quest.party_member_ids` is a derived view over membership (per universal_mechanics.md "Open questions" § 2: "`members(quest) = current_group_membership(group_id)` as a view, not a stored list"). The cascade on death differentiates by the agent's role in the quest.

**Principled differentiation — "party member died" vs "quest target died":**

```
cascade on EntityDied(a, cause):
  // Clean up party membership
  for q in active_quests_referencing(a):
    match a in q:
      // Case A: a was in the party (fighter, escort, porter...)
      ParticipantRole(_):
        emit PartyMemberRemoved { quest_id: q.id, agent: a, cause }
        // Quest continues if party still viable
        if quest_still_viable(q):
          // Optional: posted replacement request
          if q.terms.auto_replace:
            emit PostQuest { type: Service, role: <vacated_role>,
                             party_scope: q.party_scope, reward: ... }
        else:
          emit QuestFailed { quest_id: q.id, reason: PartyNonViable }

      // Case B: a was the quest target (marriage partner, escort subject,
      //         assassinate target, rescue subject)
      TargetCriteria:
        match q.type:
          Marriage:
            // Target of a marriage proposal died → quest fails
            emit QuestExpired { quest_id: q.id, reason: TargetDeceased }
            // Proposer gets grief spike on next tick

          Escort | Rescue:
            // Can't escort/rescue a corpse; mission fails
            emit QuestFailed { quest_id: q.id, reason: TargetDeceased }

          Assassinate:
            // Target died → quest succeeds (regardless of killer)
            emit QuestCompleted { quest_id: q.id, winners: [ cause.killer_id? ] }
            // Cascade: bounty paid (if anyone claims), reputation to killer

          Hunt:
            // Target was a specific creature id; hunt satisfied.
            emit QuestCompleted { ... }

          _:
            // Default: target-death → fail
            emit QuestFailed { quest_id: q.id, reason: TargetDeceased }

      // Case C: a was the poster of a still-open quest
      PosterRole:
        match q.status:
          Posted (not yet accepted):
            emit QuestExpired { quest_id: q.id, reason: PosterDeceased }
          InProgress (already accepted):
            // Accepted contracts continue; reward escrow pays from poster's estate.
            // q.poster becomes the estate executor (family head / heir).
            emit QuestPosterTransferred { quest_id: q.id, to: heir_of(a) }
```

**Why this is the principled choice:**
- **Membership is a derived view, not stored state.** Removing a dead agent from `party_member_ids` happens automatically on the next read — no corruption possible.
- **Role-based dispatch** handles the user's extension: "unless agent is part of quest criteria." Party membership vs target criteria vs poster are three distinct roles with three distinct outcomes, named explicitly.
- **Emergent narrative.** Target-death on Assassinate = success; target-death on Marriage = grief. Same event, different cascade outcomes because the quest type knows what role the agent played.
- **Poster death ≠ instant cancellation for in-progress quests.** The contract has been signed; estate law fulfills. This is necessary for long-horizon quests to survive the churn of individual mortality.

**Implementation walkthrough:**
1. Agent A dies (combat, starvation, old age). Runtime emits `EntityDied{entity_id: A, cause: ...}`.
2. A reverse index `agent_to_quests: AgentId → Vec<(QuestId, Role)>` — built from quest records — gives the affected quests. This is the cross-entity index the proposal's policy open question #15 flags.
3. For each (quest, role), cascade block above matches and emits role-appropriate follow-up events.
4. `Quest.party_member_ids` view naturally excludes `A` next read (A.alive == false).
5. Chronicle entries emitted for quest state transitions.

**Gaps / open questions:**
- **Grief cascade for quest failure.** The marriage example: proposer's observation at tick T+1 includes "my proposed spouse died"; next tick their emotion.grief bumps. Is this cascaded by quest lifecycle or by memory.events? Probably both — memory.events records the personal impact, and quest lifecycle records the quest state. Proposer's observation gets both signals.
- **Party viability predicate.** `quest_still_viable(q)` — hand-coded per QuestType or declarative? Declarative preferred: `quest_still_viable { Conquest: party.military_strength >= threshold; Hunt: party.members.len() >= 1; ... }`.
- **Heir-of function.** Requires family group membership + succession discovery. Already required by story 46 (succession dynamics); shared mechanism.

**Related stories:** 35 (cascade visibility), 46 (leader succession), 47 (marriage formation mechanics), 61 (group dissolution — similar differentiated cascade).

---

#### Story FGIJ.63: Marriage race condition (REFRAMED to invite-style)
**Verdict:** SUPPORTED — reframed per user instruction
**User's framing note:** "I think Marriage should be more like a social group invite. Also, the per frame timing means asymmetric relationships would never result in marriages?"

**Reframe:** Two-party agreement is **not a race**. It's an **invite + accept pattern** generalising to all actions requiring mutual consent. Replace `PostQuest{type=Marriage}` by A and by B racing within one tick with **invite-pending** semantics.

**The general invite-style pattern:**

```
action InviteToGroup {
  kind:          GroupKind,      // Family, Faction, Guild, Alliance, Party, Vassal, ...
  inviter:       AgentId,        // implicit = self
  target:        AgentId,        // the other party
  terms:         GroupTerms,     // dowry, charter, oath, rank, ...
  expires_tick:  u64,
}
// emits: InvitePosted { invite_id, kind, inviter, target, terms, expires_tick }
//   stored in a per-agent slot in the TARGET's observation

action AcceptInvite { invite_id: u32 }
// emits: InviteAccepted { invite_id }
//   → cascade on GroupKind-specific formation event

action DeclineInvite { invite_id: u32 }
// emits: InviteDeclined { invite_id }
//   → minor reputation / emotion cascade
```

**How it eliminates the race:**
- A's policy emits `InviteToGroup{kind=Family, target=B}` at tick T.
- Cascade: `InvitePosted{invite_id=I, inviter=A, target=B}` — stored in B's pending-invites inbox.
- At tick T+1 (or any later tick before `expires_tick`), B's observation includes the pending invite in an `incoming_invites[K=4]` slot. B's policy sees it and can choose `AcceptInvite(I)` or `DeclineInvite(I)` or ignore.
- If B simultaneously emitted `InviteToGroup{kind=Family, target=A}` at tick T: cascade emits two separate `InvitePosted` events (deterministic order by agent_id). B's invite is in A's inbox, A's invite is in B's inbox. At tick T+1 each sees the other's invite and can accept. If both accept: the *first* accepted invite resolves (deterministic tie-break by invite_id), the second is auto-cancelled as `InviteMoot{invite_id, reason: GoalAlreadyMet}`. Net result: exactly one `MarriageFormed` event.
- **Asymmetric timing solved.** B doesn't have to respond same tick. B can ignore the invite for 50 ticks, reflect on the relationship, then accept or decline.

**Actions that fit this pattern:**

| Action | Invite kind | Accepter | Formation event |
|---|---|---|---|
| Marriage proposal | `Family` (spouse role) | target agent | `MarriageFormed{a, b}` |
| Alliance proposal | `Alliance` (MutualDefense quest wrapped) | target group leader | `AllianceFormed{g1, g2}` |
| Vassal petition | `Vassal` (hierarchical standing) | prospective suzerain | `VassalSworn{g_weak, g_strong}` |
| Party recruitment (ad-hoc) | `Party` (adventuring party role) | target agent | `JoinedParty{agent, party}` |
| Apprenticeship | `Family` subtype (mentor/apprentice) | target agent | `ApprenticeshipFormed{mentor, apprentice}` |
| Job offer (employer-driven) | `Guild` (or contract) | target agent | `EmploymentAccepted{employer, worker}` |
| Guild admission | `Guild` | target agent or guild board | `JoinedGuild{agent, guild}` |
| Religious conversion invitation | `Religion` | target agent | `JoinedReligion{agent, religion}` |
| Adoption | `Family` (child role) | target agent (consenting if adult) or guardian | `Adopted{child, family}` |
| Peace treaty | `Alliance` (with standing=Neutral terms) | target group | `TreatySigned{g1, g2}` |

**The pattern generalises wherever two parties must both consent.** A single-party action (Attack, Harvest, PostQuest{type=Assassinate}) doesn't need invite semantics. A public-auction action (Bid) doesn't either — the auction is the mediation; the poster already consented by posting.

**Implementation walkthrough (marriage):**
1. Agent A observes B in `known_actors`. A's mask passes `InviteToGroup{kind=Family, target=B}` when `A.can_marry ∧ ¬married(A) ∧ ¬married(B) ∧ relationship(A,B).valence > MIN_TRUST`.
2. A's policy emits the action. Cascade: `InvitePosted{invite_id=I, inviter=A, target=B, kind=Family, expires_tick=now+200}`.
3. B's next-tick observation includes `incoming_invites[K]`. Each entry: `invite_id, kind_one_hot, inviter_id, terms_summary, time_remaining, relationship_summary_with_inviter`.
4. B's policy either accepts, declines, or ignores (no action taken). Mask for `AcceptInvite(I)`: `I ∈ self.incoming_invites ∧ I.kind_acceptable_to(self) ∧ self.eligible(I.kind)`.
5. If B accepts: cascade emits `InviteAccepted{I}` → `MarriageFormed{A, B}` (Family-kind formation) → both A and B gain `Membership{Group: new_family_group, role: Spouse}`.
6. If B ignores until `expires_tick`: cascade emits `InviteExpired{I}` → A gets minor grief (same as current QuestExpired behaviour).
7. **Concurrent-proposal case.** A and B both emit invites at tick T. Two `InvitePosted` events fire (deterministic order). At tick T+1 both agents see the other's invite. If both `AcceptInvite` at tick T+1: cascade processes in agent-id order; first `InviteAccepted` fires `MarriageFormed`; the second fires against a now-married pair, which fails the re-validation mask (`¬married(t)`), and falls through to `InviteMoot{invite_id, reason: AlreadySatisfied}` (a structured drop per story 59).

**Gaps / open questions:**
- **Invite TTL.** What's the default expiration? Marriage probably long (~500 ticks); alliance probably short (~50 ticks). Parameterise per kind.
- **Invite inbox capacity.** Per-agent K=4 invites observable. Beyond K, invites queue but aren't visible to the policy — they age out silently. Is that ok? Probably; a popular agent with 40 suitors doesn't need to see all 40, just top-K by observation-relevance ranking.
- **Invite withdrawal.** Inviter changes their mind before target responds: `WithdrawInvite(I)` action. Cascade cancels. Analogous to `WithdrawQuest` (open question § 7 in universal_mechanics).
- **Observation features for invites.** A new `incoming_invites[K=4]` slot array needs adding to § 2.1 (policy schema). Schema hash bump (story 64) — append-only ok. Model must learn to attend to this slot to accept invites at all.
- **Relationship to existing PostQuest{type=Marriage}.** The invite is essentially a special-cased quest with `PartyScope::Individual(target)` and a built-in accept/decline UI. Could be implemented *as* a quest with `exclusive_acceptance=true`, or as a dedicated `Invite` primitive. Invite as primitive is cleaner for the observation schema (dedicated slot); quest-based is cheaper (no new mechanism). Recommendation: invite as a narrow specialization of quest, with the `incoming_invites` slot being a filtered view over `active_quests` where `self ∈ eligible_acceptors ∧ quest.type ∈ {Marriage, Alliance, Vassal, ...}`.
- **Group-level invites.** `InviteToGroup{kind=Alliance, target=OtherGroupLeader}` — accepted by a group via its leader, but other group members have opinions. This becomes closer to a quest-with-acceptance than a pure 2-party invite. Stay with quest machinery for group-level consent; reserve invite primitive for agent-to-agent.

**Related stories:** 47 (marriages form by trust + compatibility), 49 (LeaveGroup — the symmetric exit), 54 (alliance via Diplomatic auction — alternative mechanism for group-level two-party agreement), 61 (group dissolution — family-group dissolves on divorce).

---

#### Story FGIJ.64: Adding a new ActionKind breaks model
**Verdict:** SUPPORTED — FAIL LOUD at model load
**User's framing note:** "ESSENTIAL."

**How the DSL supports this:**
§ 4 of the policy schema already covers schema hashing for observation. Per open question #17 ("Schema hash scope — observation only, or include action vocab + reward?"), this story is the resolution: **the hash covers action vocabulary too.**

**Implementation walkthrough:**
1. **Schema hash definition.**
   ```
   schema_hash = sha256(
     canonicalize(observation_schema) ||
     canonicalize(action_vocab) ||       // ← NEW: included
     canonicalize(event_taxonomy) ||     // for cascade compat
     canonicalize(reward_vocab)          // optional; affects training, not inference
   )
   ```
   Canonicalization is sorted + normalized so cosmetic reorderings don't change the hash.

2. **Action vocab canonicalization.**
   ```
   for each head in (macro_kind, micro_kind, quest_type, party_scope, ...):
     emit head.name, sorted variant list, each variant's name + @since tag
   ```
   Append-only growth per § 4: adding a new variant appends to the list; removing or renumbering is a *different* hash change class (breaking vs extending).

3. **At model load.**
   ```
   loaded_hash = read from model checkpoint
   current_hash = compute from running DSL
   if loaded_hash == current_hash:
     ok
   else:
     diff = canonical_diff(loaded_schema, current_schema)
     if diff.is_append_only_in_obs_and_vocab:
       // Zero-pad new observation fields; mask new action variants as always-invalid
       // (model was never trained to produce them, so never sample them).
       warn("loaded model is {diff.n_fields_behind} fields behind; running with zero-pad migration")
       apply_migration(model, diff)
     else:
       // Removed / renumbered / incompatible → HARD FAIL
       FAIL_LOUD("model schema mismatch: model expects {loaded_schema}, runtime is {current_schema}")
   ```

4. **CI guard.**
   - Changes to observation fields, action variants, event taxonomy go through a CI check that computes the schema hash pre/post.
   - An append-only hash change emits a migration-needed marker. A non-append change (remove/reorder) blocks CI unless accompanied by a model-checkpoint version bump and explicit migration doc.
   - Unit test: load checkpoint, compare hashes, assert the load path the model took (exact match, append-migration, hard fail).

5. **Story 15 constraint — "FAIL LOUD."**
   Silent zero-filling of a new action-vocab slot is *not* acceptable if the modeller thought they were loading a compatible model. The load-time check emits:
   - `ok` with matching hash → proceed.
   - `warn` with append-only diff → requires `--accept-migration` flag (explicit opt-in) — not silent.
   - `fail` with incompatible diff → error out with a clear message pointing at the conflicting variant names.

**Why this is the principled choice:**
- **Observation schema and action vocab are both load-bearing ML contracts.** Observation widens → model sees new features it wasn't trained on (handle via zero-pad). Action vocab widens → model's categorical head needs a new logit it wasn't trained on (zero-pad is also an option — new variant gets probability 0 until retraining); vocab removes → model has logits pointing at nonexistent kinds (hard fail; old model *cannot* safely run).
- **Event taxonomy** matters for cascade reconstruction (replay fidelity, story 38 repro bundle compat). Including it in the hash means a bundle captured under hash H replays identically on runtimes with hash H and fails loud on anything else.
- **Reward vocab** matters only for training, not inference. Include in hash for training-time safety; production inference can mismatch reward without harm (but training-data + model would mismatch, so practically the whole hash is coupled).

**Gaps / open questions:**
- **Migration tooling.** When the hash diff is "append-only new action variant," what does `apply_migration(model, diff)` actually do? Options:
  - (a) Expand the categorical head's output dimension; initialize new logit weights to `-inf` (so mask would drop to zero anyway). Mask the new variant as always-invalid until retrained.
  - (b) Soft-expand and allow the new variant to be sampled with a tiny prior (epsilon exploration). Only useful if downstream training is imminent.
  Default: (a) for production, (b) opt-in for exploration.
- **How does this interact with hot reload (story 44)?** Hot reload computes new schema hash; if the change is append-only and the running model is append-compatible, reload succeeds. Otherwise reload fails or requires model swap.
- **Monte-carlo-style mixed models.** If different agents run different model checkpoints (A/B testing), each checkpoint has its own hash check. Already covered by the per-checkpoint validation.
- **"Silent extension" loophole.** Be careful that `@since` annotations on observation fields don't drift the hash in a way that makes two semantically-identical schemas produce different hashes. Canonicalization must be `@since`-agnostic (annotations are metadata, not schema identity).

**Related stories:** 34 (mask — new action variant contributes to mask; existing masks unaffected), 38 (repro bundle carries hash), 41 (ItemType add — doesn't change action vocab, only observation if one-hot widened), 44 (hot reload intersects with schema hash), 59 (action drops — an out-of-vocab sampled action is an adversarial-model event that the runtime drops gracefully *if* loaded, but ideally is caught at load time by the hash check).

---

### Cross-cutting observations

1. **F (debugging) is well-served by the event-sourced + declarative-DSL design.** Story 34 (mask trace), 35 (cascade fan-out), 37 (tick step), 38 (repro), 39 (decision history), 40 (flamegraph) all fall out naturally because every stage boundary is named and every mutation is a typed event. The hard work is retention policy, not expressiveness.

2. **G (modding) is PARTIAL because of one-hot vector widening.** Adding content that contributes to categorical vectors (ItemKind one-hot, creature_type one-hot) is a schema-hash-breaking change. Options: reserve slack slots (wasteful but future-proof), or commit that mod authorship implies schema/model update. User's story-43 stance ("make the whole system modifiable, they can add or delete rules themselves") leans toward explicit ownership of schema by mods.

3. **I (auction) is correctly reframed as emergent-not-scheduled.** Story 55's commodity auction emerges from individual `PostQuest{type=Trade}` decisions; story 58's mercenary hiring is a canonical `AuctionKind::Service` case. Story 54 (alliance via Diplomatic auction) is a close neighbour that the DSL supports symmetrically. The rejected stories (56 charter auction, 57 GPU state) are out of scope for valid design-taste reasons, not gaps in the proposal.

4. **J (adversarial) forces tight discipline on two invariants:**
   - **ID-over-text.** Story 60 mandates numeric IDs for any load-bearing state; text is cosmetic. Practical impact: convert existing `class_tags: Vec<String>` and `archetype: String` to ID-based representations; segregate chronicle text from chronicle structure.
   - **Schema hash over action vocab.** Story 64 extends § 4 to cover action vocabulary and resolves open question #17 in favour of "yes, hash covers vocab." FAIL LOUD at model load for incompatible changes; explicit opt-in migration for append-only changes.

5. **Story 63's invite pattern is more important than it looks.** It generalises to at least 10 actions (marriage, alliance, vassal petition, party recruitment, apprenticeship, adoption, guild admission, religious conversion, job offer, peace treaty) and replaces a class of race conditions with a pending-invite queue. Recommendation: formalise `Invite` as a specialisation of `PostQuest{exclusive_acceptance=true}` with an explicit `incoming_invites[K=4]` observation slot. This costs ~40 observation floats and buys deterministic two-party consent semantics across the whole social / political vocabulary.

6. **Story 61 (group dissolution) + story 62 (agent death) share a pattern:** a referent disappears; cascade must differentiate by the referent's role in each referencing structure. The "role-based dispatch" template used in both stories is likely the principled template for all "referent disappeared" cascades (settlement destroyed, item broken, location conquered, ...). Worth codifying once rather than re-inventing per entity type.

---

## Batch H — Player-experienced Behaviors (Acceptance Tests)

This batch validates that the DSL supports **the gameplay the player will experience**. Each story is an end-to-end acceptance test: observation → policy decision → action → cascade → emergent outcome. Stories that cannot be traced end-to-end surface real gaps in the architecture.

Schemas referenced:
- `spec.md` — observation, action space, mask, reward
- `spec.md` — PostQuest / AcceptQuest / Bid vocabulary
- `state.md`, `state.md`, `state.md` — field-level state

---

#### Story H.45: Wars are motivated
**Verdict:** PARTIAL (grievance-driven path supported; cold-start + expansion paths need work)

**User's framing note:** "Who would start the first war? What about wars for expansion?" The grievance-threshold mask predicate doesn't bootstrap on an empty chronicle, and doesn't explain wars that aren't reactive at all.

**End-to-end trace (grievance case — the easy one):**

1. **Observation (leader of G_self):**
   - `self.creature_type`, `self.ambition` (personality, 5-d)
   - `memberships[K=8]` — includes G_self with `my_role_in_group = Leader`
   - `known_groups[K=6]` — G_rival with `standing_with_me = Tense`, `group_strength_log`
   - `nearby_actors` / `known_actors` — members of G_rival I recognise as sources of past grievances
   - `recent_memory_events[K=12]` — past `WasAttacked / FriendDied / BetrayalWitnessed` events whose `entity_ids` resolve back to G_rival members
   - **Derived view** `grievance_matrix(G_self, G_rival)` — sum of negative memory events of G_self members toward G_rival members with recency decay. Feeds observation as a scalar feature on the `known_groups[G_rival]` slot. The policy schema alludes to this at §2.1.5 (`n_opposed_groups`) and in `systems.md:131` (`grievance_matrix(a, b) = Σ over npcs_of(a) Grudge beliefs toward npcs_of(b)`).
   - World context: `tick_normalized`, season, treasury, military_strength.

2. **Policy:** emits
   ```
   PostQuest{
     type=Conquest,
     party_scope=Group(G_self),
     target=Group(G_rival),
     reward=Spoils(territory),
     deadline=tick+5000
   }
   ```

3. **Mask (gate):**
   ```
   PostQuest{type=Conquest, party=Group(g), target=Group(t)}
     when g ∈ self.leader_groups               // I lead G_self
        ∧ g.kind ∈ {Faction, Pack}              // only factions/packs wage Conquest
        ∧ g.military_strength > 0               // can actually fight
        ∧ g.standings[t] ≠ AtWar                // not already at war
        ∧ eligible_conquest_target(self, t)     // OPEN — see below
   ```

4. **Cascade:**
   - `QuestPosted{quest_id, type=Conquest, party_scope=Group(G_self), target=Group(G_rival)}` fires.
   - **Cascade handler** updates `G_self.standings[G_rival] = AtWar` and reciprocally `G_rival.standings[G_self] = AtWar`. (This replaces the old `faction.at_war_with.push()` side effect — see `systems.md:131`.)
   - `G_self.active_quests.push(quest_id)`.
   - Members of G_self see the new `active_quests[K=4]` slot on next tick. Their `at_war` view flips. Per-tick policy emits micro primitives (`MoveToward(rival_settlement)`, `Attack(...)`) gated by the war-context.
   - `known_groups[G_rival]` standing flips in the observation of every member.

5. **Emergent outcome (player-visible):**
   - Raids, battles, settlement conquest events land in the chronicle.
   - Treasury drains (units + upkeep). Casualty events accumulate.
   - Eventually either `QuestCompleted{type=Conquest}` → Spoils cascade, or `QuestExpired` → war-exhaustion view fires → leader emits `PostQuest{type=Peace}` / `Bid` in a diplomatic auction.

**Cold-start case (fresh world, no chronicle):**

The grievance path breaks because `grievance_matrix(a, b) ≈ 0` for every (a, b) pair. Something has to break the symmetry or the world is frozen in peace forever.

Three complementary mechanisms that *together* bootstrap first-ever wars from the observation/mask already in the schema:

- **(a) Personality-driven bellicosity.** `self.personality.ambition` is a primary input, drifted by events. Seeded at spawn. A high-ambition leader with high `known_groups[G_other].group_strength` contrast and adjacent territory chooses `PostQuest{Conquest, target=G_other}` with nonzero probability even at `grievance=0`. The reward function biases this: `+Δ(fame_log)`, `+Δ(self.esteem)`, and `+Spoils` on `QuestCompleted`. Without any grievance, ambition + a weaker rival is enough for a ratty leader to open hostilities.

- **(b) Territory contestability feature.** Add an observation feature surfacing contestable-territory signal so the policy can see "expansion is profitable" without depending on a chronicle. Proposed: on `known_groups[K=6]` slot, add
  - `controls_resource_i_lack` (boolean/soft): whether the group controls a biome/commodity our home_settlement is short on
  - `is_adjacent_territory` (boolean)
  - `military_strength_ratio` (signed log)
  - `standing_with_me_one_hot` already exists

  These let a leader observe "there's a weaker group sitting on food I need" as raw features, not as an accumulated grievance.

- **(c) Ambition as a reward-shaping signal.** The reward block already includes `+Δ(self.fame_log)` and `+2.0 on QuestCompleted`. That's enough to create a learned gradient toward "I gain by expanding my group." A learned ambition-biased leader converges on expansion wars during training even when grievances are nil.

**Expansion-for-expansion's-sake case:**

Same observation, different triggering features:
- `G_self.military_strength_ratio_vs[G_other]` high
- `territory_size(G_self)` small
- `G_other.controls_resource_i_lack` true
- `self.personality.ambition` high
- `self.needs.esteem` low (ambition unmet)

Policy emits `PostQuest{Conquest, target=G_other}` purely because the observation + reward gradient points to it. The mask passes because G_other isn't already at war with us. No grievance required.

**How the DSL supports this:**

- `PostQuest{type=Conquest, party_scope=Group(g)}` — universal action (proposal doc §1)
- Mask predicate for Conquest on leader groups (proposal doc §2.3)
- Rich observation includes `known_groups[K=6]` with standing/strength/size, agent personality, self emotions/needs (§2.1)
- Reward block sums `Δ(fame_log)`, `QuestCompleted+2`, need-recovery — sufficient to shape ambition-driven wars if data is balanced.

**Implementation walkthrough:**

1. Extend `known_groups[K=6]` slot features with `military_strength_ratio`, `controls_scarce_resource`, `is_adjacent`. These are cheap derived views over `Group.military_strength`, `Group.stockpile`, spatial query over territory cells.
2. Add the `eligible_conquest_target(self, other)` mask predicate. Minimum gate: `other.kind ∈ {Faction, Pack}` ∧ `other.alive` ∧ `standings[other] ≠ Vassal` ∧ (optional) `distance_to_territory(other) < CAMPAIGN_RANGE`.
3. Add cascade rule `QuestPosted{type=Conquest}` → set both groups' `standings = AtWar`.
4. Confirm reward includes `+Δ(self.personality.ambition × fame_gain)` or leave as pure fame — during training the ambition dimension becomes correlated with expansion behaviour.
5. During bootstrap (utility backend), hand-score `Conquest` with a small base `ambition × military_ratio × need_esteem_gap` term so the initial trajectories have some expansion wars, then distill into the neural backend.

**Gaps / open questions:**

- **No "war declaration event" distinct from `QuestPosted{Conquest}`.** Consumer systems (UI chronicle, diplomatic cascade) currently hunt for `WarDeclared`. Proposal: `QuestPosted{type=Conquest}` cascade rule emits a derived `WarDeclared{a, b, casus_belli}` chronicle entry. Casus belli is read from `QuestPosted.terms` (new field), defaulted to `Grievance(count)` when `grievance_matrix > 0` else `Expansion`.
- **No opposite-side auto-acceptance.** When G_self posts Conquest against G_rival, G_rival's leader doesn't need to `AcceptQuest` — they get drafted into a war as defender. Proposal: cascade fires a *separate* auto-generated `Defend` quest on G_rival's side (`PostQuest{type=Defend, party=Group(G_rival), target=Location(G_rival.home)}`), implicitly accepted by leader. Members see it in `active_quests[K=4]`.
- **`territory_size` + `controls_scarce_resource` aren't in the current observation spec.** Need to append to `known_groups` slot schema. Append-only per the versioning rule (§4 of schema).
- **Casus belli enum.** `Grievance`, `Expansion`, `Revenge`, `ReligiousDifference`, `Succession`, `Reclaim`. Each slot on `QuestPosted` becomes narrative fuel.
- **Credit assignment.** Conquest reward lands 2000+ ticks after the decision. Flagged as open in schema §6 (Q14). Not unique to this story but biting here.
- **Minimum-viable extension:** schema additions for `known_groups` (`military_strength_ratio`, `is_adjacent_territory`, `controls_scarce_resource`), mask predicate `eligible_conquest_target`, cascade rule `QuestPosted{Conquest} → WarDeclared + Defend counter-quest`, and a hand-scored `ambition_expansion_utility` in the utility backend for bootstrap. Total: ~1 week.

**Related stories:** 46 (succession — leader chosen for ambition reshuffles war chances), 49 (betrayal — war stress amplifies defections), 51 (long war → economy), 54 (alliance bidding — wars trigger mutual-defense auctions).

---

#### Story H.46: Succession dynamics on leader death
**Verdict:** PARTIAL → SUPPORTED once the observation exposes a "leader vacancy" feature. User is right that succession should be emergent; no special cascade rule is needed.

**User's framing note:** "Should be emergent." No hand-coded succession system. Agents must observe "leader vacancy" and compete via normal actions.

**End-to-end trace:**

1. **Trigger event.** `EntityDied{entity_id=leader, cause=...}` fires (combat, assassination, disease — all existing events). Cascade handler updates `G.leader_id = None` and emits `LeaderVacant{group_id, predecessor_id, predecessor_role_tenure}` chronicle entry.

2. **Observation update (all senior members of G):**
   - `memberships[k]` for the group `G` now has `group.leader_id_absent = true` (new slot feature, boolean).
   - `known_actors[k]` slots for potential rivals (fellow officers) update with `is_competing_for_leadership = true` if they've already acted.
   - `recent_memory_events[K=12]` includes the `EntityDied{leader}` event, so every member who witnessed the death has it in their ring.
   - `self.standing_in_group` (already present in `memberships` slot) surfaces my own claim strength.
   - `self.personality.ambition` + `self.esteem_need` drive willingness to compete.

3. **Policy (claimant path):** A senior member with `role ∈ {Officer, Founder}`, high `standing_in_group`, high `ambition` emits:
   ```
   PostQuest{
     type=Submit,                    // invert the relation: "all members submit to me"
     party_scope=Group(G),
     target=Agent(self),             // I am the successor candidate
     reward=Reciprocal,              // "I lead, you benefit"
     terms=SuccessionClaim{predecessor=X}
   }
   ```
   or equivalently a new `Claim` quest type (`type=Claim, party=Group(G), target=Role(Leader)`). Both work; `Claim` is cleaner because `Submit` currently implies vassalage.

4. **Policy (voter path):** Other members observe `active_quests[K=4]` filled with one or more Claim quests. They emit:
   ```
   AcceptQuest{quest_id=claim_of_rival_A}
   ```
   as a vote. Mask passes `AcceptQuest` only if `self ∈ quest.eligible_acceptors` — here the eligibility predicate is `self ∈ G.members ∧ self.role ∉ {Outcast}`.

5. **Resolution (cascade):** When a Claim quest collects acceptance majority (either by `MutualAgreement` auction resolution over current members, or by hitting `party_member_ids.len() / G.members.len() > 0.5`), cascade fires `LeaderSucceeded{group_id, new_leader, predecessor, mechanism=Election}`. Losing claimants see their own quest `QuestExpired`, which hits their emotion.grief + grief_penalty reward. High-ambition losers may then emit `LaunchCoup` (→ `PostQuest{type=Coup, target=new_leader}`) or `LeaveGroup(G)` + `JoinGroup(G_rival)`.

6. **Mask (gate):**
   ```
   PostQuest{type=Claim, party=Group(g), target=Role(Leader)}
     when g ∈ self.memberships
        ∧ g.leader_id = None            // vacancy exists
        ∧ self.role_in(g) ∈ {Officer, Founder}
        ∧ tick - self.data.joined(g) > MIN_TENURE
   AcceptQuest{q}
     when self ∈ q.eligible_acceptors
        ∧ q.type = Claim
        ∧ q.group = g ∈ self.memberships
        ∧ ¬conflicts_with_my_active_quests(q)    // can't back two claimants
   ```

7. **Emergent outcomes player observes:**
   - **Uncontested succession:** only one senior member claims, others auto-accept → smooth transition. Chronicle: "After Lord X died, his nephew Y was crowned."
   - **Contested succession:** two officers claim simultaneously → partial acceptance on each → neither reaches majority → `QuestExpired` on both → frustrated claimants try again with better diplomacy (bribes via `Bid` in a private Political auction) or `LaunchCoup`.
   - **Coup:** a claimant who lost the vote PostQuest's Conquest against their own group (`party=cabal of my_faction_loyalists, target=Group(G)`). Runs as any civil war.
   - **Collapse:** no one claims (tenure predicate fails for all; group decapitated). `G.dissolved_tick = now` cascade rule when `G.leader_id = None` for > DECAPITATION_GRACE ticks and no claimants remain.

**How the DSL supports this:**

- `Group.leader_id: Option<AgentId>` field already exists (`state.md:588`). Vacancy = `None`.
- `memberships[K=8]` slot exposes `my_role_in_group`, `my_standing_in_group` — enough for self-claim scoring (§2.1.3).
- `PostQuest{Claim}` + `AcceptQuest` + `AuctionResolution::MutualAgreement` = the full contested-vote machinery (spec.md §3).
- `Group.governance: GovernanceKind` already differentiates `Hereditary / Elective / Council / Theocratic` (`state.md:590`). The eligibility predicate on Claim quest reads governance to decide who can claim: Hereditary groups permit only `self ∈ G.leadership_chain`; Elective permits all officers; etc. This is data-driven — NO "succession system" — just richer mask predicates.

**Implementation walkthrough:**

1. Extend `QuestType` with `Claim` variant (as already listed in OPEN). Target type `Role(RoleTag)` — add to `QuestTarget`.
2. Add cascade rule: when `EntityDied{id}` fires and `id` appears as `leader_id` on any Group, set that Group's `leader_id = None` and emit `LeaderVacant{group_id}` chronicle entry.
3. Add `auto-dissolve` cascade: `G.leader_id = None ∧ tick - G.leader_vacated_tick > DECAPITATION_GRACE ∧ no_pending_claim(G)` → `DissolveGroup{G}`.
4. Add mask predicates for `PostQuest{Claim}` / `AcceptQuest{Claim}` keyed on `governance`.
5. Extend `memberships[k]` with two floats: `group_leader_vacant` (0/1) and `my_tenure_relative_to_other_seniors`. These are the features that cause the policy to "notice" the vacancy.
6. Reward component for successful Claim: `+Δ(self.esteem)`, `+Δ(fame_log)`, `+Δ(my_standing_in_group)`. For failed Claim: `+grief emotion`, `-self.esteem`. Shapes ambition-contested succession.

**Gaps / open questions:**

- **Need a `Claim` quest type** — currently not in `QuestType` enum (`spec.md:115`). Using `Submit` with inverted target is semantically wrong; add `Claim`.
- **`Role(RoleTag)` target** not yet in `QuestTarget` enum (§"QuestTarget" in mechanics doc). Small addition.
- **No "senior officer" feature** — `role_in_group_one_hot` has `{Member, Officer, Leader, Founder, Apprentice, Outcast}`. Officer and Founder are the senior claim roles — sufficient.
- **Auction resolution for election.** `MutualAgreement` resolves when all targeted acceptors say yes — wrong semantics for majority vote. Need a new `AuctionResolution::Majority` or let Claim quests resolve via `FirstAcceptable` after N ticks with `winner = quest with most acceptances`. Auction state machine is already OPEN (schema §5, mechanics §"Auction state machine").
- **Governance-sensitive masks.** Hereditary → only next in `leadership_chain` can claim; Elective → all officers; Theocratic → charter-designated pool. These are ~5 predicate rules each gated by `governance` enum.
- **Minimum-viable extension:** add `Claim` QuestType + `Role` target + vacancy cascade + `Majority` auction resolution + 2 memberships slot features. ~2 weeks including the auction state machine work.

**Related stories:** 45 (a war-weary leader dies mid-conquest → succession contest during war), 49 (disgruntled loser defects), 50 (reputation determines who accepts whose claim), 52 (inherited traits bias claimant ambition).

---

#### Story H.47: Marriages by trust + compatibility
**Verdict:** SUPPORTED

**User's framing note:** "Sure."

**End-to-end trace:**

1. **Observation (proposer, Agent A):**
   - `self.spouse_id = None` (implicit in `has_spouse = false` on §2.1.2 "Social ties")
   - `self.creature_type.can_marry` (derived from species)
   - `known_actors[K=10]` — up to 10 known agents including potential mates. Per-slot features:
     - `relationship.trust`, `relationship.familiarity`
     - `relationship_kind_one_hot(8)` (Friend, Rival, Spouse, Kin, Mentor, Apprentice, Stranger, Sworn Enemy)
     - `perceived_personality` (embedded via the cross-block psychological view — proposal mentions `perceived_personality` at `state.md:175`). Compatibility is a dot-product between `self.personality` and `perceived_personality.traits`.
     - `n_shared_groups` — cultural eligibility (same faction? same religion?)
     - `has_spouse` of other (so A doesn't propose to already-married agent)
   - `self.personality.social_drive` + `self.personality.compassion` — marriage propensity
   - `self.needs.social` low, `self.needs.esteem` depending on cultural status — drive
   - `memberships[k].group.kind = Family` of self — am I already in a Family?

2. **Policy:** emits
   ```
   PostQuest{
     type=Marriage,
     party_scope=Individual(A),
     target=Agent(B),
     reward=Union,
     terms=ExclusiveAcceptance,
     deadline=tick+500
   }
   ```

3. **Mask:**
   ```
   PostQuest{type=Marriage, party=Individual(self), target=Agent(t)}
     when self.creature_type.can_marry
        ∧ ¬has_spouse(self)
        ∧ ¬has_spouse(t)
        ∧ t.creature_type.can_marry
        ∧ t.alive
        ∧ relationship(self, t).trust > MIN_TRUST    // familiarity must exist
        ∧ marriage_eligibility(self, t)               // cultural rule (view)
                                                     // distance NOT required
   ```
   `marriage_eligibility` is a derived view over `memberships` + group eligibility predicates. Example: a Stone Clan religion only permits in-clan marriage → predicate rejects cross-clan pairings.

4. **Cascade (post):** `QuestPosted{quest_id, type=Marriage, party_member_ids=[B]}` — B is the sole eligible acceptor. Quest lands in B's `active_quests[K=4]` on next tick with `type=Marriage` and `posted_by=A`.

5. **Observation (acceptor, Agent B):**
   - `active_quests[k]` now surfaces a Marriage quest targeting me from A.
   - Same compatibility features in `known_actors[B's view of A]`: `trust`, `familiarity`, `perceived_personality`, shared groups.
   - B's own `personality`, `needs`, `emotions`.

6. **Policy:** emits either
   ```
   AcceptQuest{quest_id}
   ```
   or doesn't emit (passive refusal → quest expires at deadline).

7. **Cascade (accept):**
   - `QuestCompleted{quest_id}` fires.
   - Cascade handler for `Marriage` emits:
     - `MarriageFormed{a=A, b=B, tick}` (`systems.md:197`)
     - `FoundGroup{kind=Family, founder_ids=[A, B]}` (or append both to existing family — if one spouse has kin group).
     - A.spouse_id = B.id; B.spouse_id = A.id; both get `memberships.push(Membership{group_id=new_family, role=Founder})`.
     - A.personality.social_drive, B.personality.social_drive += 0.02 (drift).
     - A.emotions.joy += 0.3, B.emotions.joy += 0.3.

8. **Observation (everyone else):** `known_actors` slots for A/B update `is_spouse`; Family group becomes known to both spouses' `known_groups[k]`.

**Emergent outcomes player observes:**
- Proposals cluster around NPCs with high familiarity + shared groups (work coworkers, adventuring party members).
- High-compassion NPCs marry high-compassion NPCs (trait homophily emerges from perceived_personality-weighted choice).
- Cross-faction marriages form political alliances (see story 54).
- Failed proposals cause grief on A (+0.15 emotion.grief on QuestExpired). Multiple rejections shift A's personality (curiosity drift, social drive drift down).

**How the DSL supports this:**

- `PostQuest{Marriage}` → `AcceptQuest` cascade fully specified in `spec.md` §"How 'marriage is a quest' plays out" (:282).
- Observation features in `known_actors[K=10]` — `relationship_valence`, `relationship_familiarity`, `relationship_kind_one_hot`, `n_shared_groups` (§2.1.5).
- `self.personality` (5d) + `known_actors[K].perceived_personality` → compatibility dot-product computable as a derived feature within the `known_actors` slot.
- Mask predicate in §2.3 is explicit about marriage.
- `Group{kind=Family}` as the post-marriage container (`state.md:638`).

**Implementation walkthrough:**

1. Add the `FoundGroup{Family}` cascade rule on `QuestCompleted{type=Marriage}` — already specified in mechanics doc.
2. Add `marriage_eligibility(a, b)` derived view reading both agents' membership predicates (religious/faction exclusivity). Data-driven from Group.eligibility_predicate.
3. Extend `known_actors[k]` slot with `compatibility_score` = cosine(self.personality, other.perceived_personality.traits × confidence). Purely derived; cheap.
4. On `QuestExpired{type=Marriage}`: emit `MarriageRejected{proposer, refused_by}` → proposer.emotion.grief += 0.15, proposer reward -1.0.
5. Cap: one active Marriage PostQuest per NPC (avoid spamming proposals). Mask predicate `¬any_active_marriage_quest(self)`.

**Gaps / open questions:**

- **Who proposes?** Either side — mask is symmetric (`can_marry` ∧ `¬has_spouse` ∧ `relationship.trust > MIN`). Training will produce both. OK.
- **Multi-spouse cultures.** Current `spouse_id: Option<u32>` is single-slot. A polygamous culture needs `spouses: Vec<u32>`. Out of scope for v1; make single-spouse assumption.
- **Arranged marriage (by parent).** A's parent posts on behalf of A. That's `PostQuest{Marriage, party_scope=Individual(parent_of_A), target=Agent(B)}` with `terms=ArrangedFor(A)`. Cascade wires A into the resulting Family. Not hard; just another cascade rule on `terms`.
- **User's hint about story 63 (invite).** If "Invite" becomes a first-class quest type distinct from PostQuest (e.g. `Invite{target=B, kind=Marriage}`), the same machinery applies — mask + cascade. I treat this as a naming nit: Marriage-as-PostQuest already reads as an invite to B.
- **Minimum-viable extension:** `marriage_eligibility` view, `FoundGroup{Family}` cascade, `compatibility_score` feature on `known_actors` slot, `MarriageRejected` reward hook. Small; 2–3 days.

**Related stories:** 52 (children inherit parental traits, requires marriage to pair parents), 49 (marriage is NOT a betrayal-immune bond; high-ambition low-loyalty profile can LeaveGroup(Family) via `AnnulMarriage`).

---

#### Story H.48: Monsters defend dens, hunt prey
**Verdict:** SUPPORTED (with small observation-feature additions)

**User's framing note:** "Sure."

**End-to-end trace (hunting case):**

1. **Observation (wolf alpha, Agent W):**
   - `self.creature_type = Wolf` → one_hot(~8) in self.atomic (§2.1.1, `creature_type_one_hot`). Derivation: wolf has `can_speak=false`, `has_inventory=false`, `can_build=false`, `can_trade=false`; `is_predator=true`.
   - `self.needs.hunger` high
   - `self.personality.risk` (reused for prey-boldness)
   - `memberships[k]` includes `Group{kind=Pack, leader_id=self or pack_leader_id, eligibility_predicate=creature_type==Wolf}`
   - `nearby_actors[K=12]` sorted by distance — each slot has `creature_type_one_hot(~8)`, `hp_pct`. `is_hostile(self, other)` is the derivation:
     ```
     is_hostile(self, other) = relationship_valence < HOSTILE
                             ∨ groups_at_war(self, other)
                             ∨ predator_prey(self.creature_type, other.creature_type)
     ```
     (Proposal §2.1.4 footnote.) A deer in `nearby_actors` is hostile-to-hunt to a wolf via the `predator_prey(Wolf, Deer)` branch.
   - `home_den: Option<(f32, f32)>` (`state.md:255`) — surfaces as `distance_to_den` feature in self contextual.

2. **Policy:** emits
   ```
   micro_kind=Attack
   target=nearby_actors[deer_slot_idx]
   ```
   Micro primitive, not a quest. `Attack(t)` passes mask because `is_hostile(W, t)` via predator_prey.

3. **Cascade:** standard combat. `Kill` event → wolf's inventory not used (no `can_trade`), but `Eat` action consumes the corpse next tick (`Eat` mask: `corpse_nearby ∧ self.needs.hunger > EAT_THRESH`).

4. **Pack coordination:** alpha wolf's kill event fires `BehaviorObserved{witness=pack_member, subject=alpha, kind=HuntSuccess}` for every pack member in `nearby_actors`. This populates each member's memory ring, subtly increasing their hunting action-outcomes EMA. Over time the pack learns to hunt together. No special pack system.

**Den defense case:**

1. **Trigger:** a Human enters `nearby_actors[k]` of wolf W, with `is_hostile(W, Human) = predator_prey(Wolf, Human) = false` (wolves aren't prey-predators on humans unless hungry — predicate returns false by default), but the intruder's `distance_to_den < DEN_RADIUS`.
2. **Observation:** new slot feature `near_my_den[K=12]` (boolean on each nearby_actors slot). Alternatively, a derived `is_trespasser(self, other)` view:
   ```
   is_trespasser(self, other) = self.home_den.is_some()
                              ∧ distance(other.pos, self.home_den) < DEN_RADIUS
                              ∧ other ∉ self.memberships[Pack].members
   ```
   Surfaces as a feature on `nearby_actors[k]`.
3. **Mask:** `Attack(t)` is already gated by hostility. Extend `is_hostile` to include `is_trespasser(self, t) ∧ t.creature_type ∈ threatening_types`. Or add a new mask path: `DefendDen(t) when is_trespasser(self, t) ∧ self.needs.safety < SAFE_THRESH` that compiles to the same Attack action.
4. **Policy:** with `need.safety` dipping (aggressor near den → fear increase), action_eval picks Attack. Pack members nearby also see the intruder; `memberships[Pack]` + proximity triggers them to join via normal Attack actions (no special defense cascade).

**How the DSL supports this:**

- `creature_type` tag differentiates the wolf from a human using the same struct (`state.md:11`).
- `home_den: Option<(f32, f32)>` already in per-agent data (`state.md:255`), listed for `Territorial/PackPredator`.
- `predator_prey(ct1, ct2)` — derived view keyed on the two types. Data-driven table in `Registry`.
- `Group{kind=Pack, eligibility_predicate=creature_type==Wolf}` — same group primitive (`state.md:639`).
- Pack members see each other's kills via `recent_memory_events[K=12]` — `BehaviorObserved` or `HuntSuccess` event.

**Implementation walkthrough:**

1. Populate the `predator_prey` table in Registry: `{(Wolf, Deer), (Wolf, Rabbit), (Dragon, Cow), ...}`. Symmetric-false; explicit pairs.
2. Spawn wolves with `home_den = Some(pos)`, `memberships=[Pack{leader=alpha}]`.
3. Add `is_trespasser(self, other)` as a derived view; surface on `nearby_actors[k]` as a boolean feature.
4. Verify mask for `Attack` uses `is_hostile(self, t)` that includes both predator_prey and trespasser branches.
5. Confirm the `Eat` action has a `Corpse` target (an Item with `durability=low` and `food_value`) accessible via `nearby_resources` — derived view over dropped-item events.

**Gaps / open questions:**

- **`is_trespasser` feature** — needs adding to the `nearby_actors[k]` slot spec. Small: +1 boolean per slot. Append-only.
- **Pack morale / loss of alpha cascade.** When alpha dies, pack succession runs via story 46's Claim machinery — need to confirm Claim quest mask permits Pack. Today's mask gates Claim on `governance ∈ {Hereditary, Elective, ...}`. Packs should use `governance = AlphaContest` (fight-winner succeeds) — another governance enum variant.
- **Monsters can't speak or use quests.** `creature_type.can_speak = false` → mask rejects `PostQuest` / `AcceptQuest` / `Bid` / `Converse` / `ShareStory`. Monsters are constrained to micro primitives. Good; prevents nonsensical "wolf posts a trade auction."
- **Prey awareness.** Deer sees wolf in `nearby_actors`; `is_hostile(Deer, Wolf) = predator_prey(Deer, Wolf) = false` but `predator_prey(Wolf, Deer) = true` creates asymmetry. Deer's mask for Flee: `threats[k]` slot sorted by `time_to_impact`. Add predator_prey-reversed threats slot population: any nearby actor for which `predator_prey(other, self) = true` appears in threats. This is straightforward.
- **Minimum-viable extension:** Registry `predator_prey` table, `is_trespasser` slot feature, `AlphaContest` governance variant, reversed-predator_prey threats inclusion. ~1 week.

**Related stories:** 46 (alpha succession by fight — cleaner Claim variant), 49 (defection — a lone wolf LeaveGroup(Pack) joins another pack; multi-membership not relevant for packs but action verbs are).

---

#### Story H.49: Some NPCs betray for personal gain
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Observation (agent N, member of factions G_home and considering G_other):**
   - `self.personality.ambition` high
   - `self.loyalty` (legacy float, `state.md:315`; maps into observation via `self contextual` block)
   - `memberships[K=8]`: slot for G_home with `my_standing_in_group`, `my_tenure_log`, `group_activity_log`. Low standing + long tenure + low activity = "I'm undervalued here."
   - `known_groups[K=6]`: slot for G_other with `standing_with_me_one_hot`, `group_strength_log`, `my_membership ∈ {none, exiled, member, leader}`.
   - `known_actors[K=10]`: G_other's leader or officer — if my `relationship(me, them).trust > 0`, there's a welcome channel.
   - `recent_memory_events`: grievances against G_home leadership (unpaid wages, missed promotion, betrayal by ally).

2. **Policy path (step 1 — Leave):** emits
   ```
   LeaveGroup{group_id=G_home}
   ```

3. **Mask for LeaveGroup:**
   ```
   LeaveGroup(g) when g ∈ self.memberships
                    ∧ ¬self.binding_oaths_with(g)
                    ∧ ¬self.role_in(g) ∈ {Founder ∧ only_founder_alive(g)}
   ```
   The `binding_oaths_with` predicate reads `active_oaths(npc)` view from the memory (`systems.md:254`). A sworn oath blocks defection unless `BreakOath` is emitted first (explicit betrayal).

4. **Cascade (LeaveGroup):**
   - `NPCLeftGroup{npc=N, group=G_home, reason=...}` event.
   - `G_home.members` set recomputed (or if cached, N removed).
   - N's memberships mutated to drop G_home.
   - **Reputation cascade:** every existing member of G_home with `relationship(member, N).familiarity > 0` gets a memory event `WasBetrayedBy{betrayer=N}` (if they had active loyalty stake) or `AllyLeft{ex_ally=N}` (neutral). Their relationship.trust toward N drops by 0.3 (WasBetrayedBy) or 0.05 (AllyLeft). These are event-cascade reward hooks, not background scans.
   - Chronicle entry `NPCDefected{npc, old_group, new_group=None_yet}` if leaving with intent to join.

5. **Policy path (step 2 — Join):** same tick or later tick, N emits
   ```
   JoinGroup{group_id=G_other, role=Member}
   ```
   or equivalently `AcceptQuest{type=Service, poster=G_other_leader}` if G_other posted a recruitment bounty.

6. **Mask for JoinGroup:**
   ```
   JoinGroup(g) when g.recruitment_open
                   ∧ self.eligible_for(g)      // eligibility_predicate
                   ∧ g.kind ≠ Family           // you can't just "join" a family
                   ∧ ¬g ∈ self.memberships     // not already in
   ```

7. **Cascade (JoinGroup):** `G_other.members` grows, N's memberships gain G_other. Relationship trust with G_other's existing members starts at baseline + small welcome bump.

8. **Multi-membership conflict (the load-bearing case):**
   If `self.binding_oaths_with(G_home)` rejects LeaveGroup, agent can still emit `JoinGroup(G_other)` WITHOUT leaving G_home. Then self belongs to both. The observation slots now carry G_home and G_other in `memberships[K=8]` with possibly-conflicting standings (`G_home.standings[G_other] = AtWar`). Derived feature `n_conflicting_group_pairs(self) = count of (g1, g2) ∈ memberships² where g1.standings[g2] = AtWar`.
   - When G_home ∧ G_other are AtWar, mask predicates for `Attack(t)` read group-war standings and fire ambiguously: "attack member-of-G_other is `is_hostile(G_home view)` but `¬is_hostile(G_other view)`." The reward hook resolves it: attacking a friend is `-1.0` (via `target ∈ self.close_friends`) and attacking an enemy is `+1.0` — agent learns to not attack own-side members. A sufficiently dramatic conflict triggers a forced choice via cascade: after N ticks of being in warring groups, either group's standings-cascade auto-emits `NPCLeftGroup` for one side based on... (OPEN — see gaps).

**Emergent outcomes player observes:**
- A high-ambition merchant in the merchant guild watches gold flow away because guild leader is corrupt. Eventually LeaveGroup, takes trade secrets, JoinGroup(rival guild). Chronicle narrates. Former guildmates' relationship trust drops — they harbor grudges → revenge memory events stored → later betrayers face assassination bounties.
- A soldier on the losing side of a war defects to the winning faction. `JoinGroup(winning)` because their memberships slot shows high-war-risk + low-standing.
- A monk who swore oaths cannot defect cleanly — must `BreakOath` first (public betrayal) with a much harsher reputation cascade.

**How the DSL supports this:**

- `LeaveGroup` + `JoinGroup` universal actions (`spec.md` §2.3).
- `Group.standings[other]` replaces `faction.at_war_with` — multi-group standings queryable (`state.md:604`).
- Per-pair `relationship` records carry the reputation cascade (trust drop for ex-allies) (`state.md:172`).
- `memberships[K=8]` observation slot supports multi-membership with conflict-observable features (`n_shared_groups`, `n_opposed_groups`).
- `active_oaths(npc)` view gates betrayals (`systems.md:255`).

**Implementation walkthrough:**

1. Cascade rules on `LeaveGroup`:
   - Remove from Group.members.
   - For each pre-existing member with familiarity > 0, emit `WasBetrayedBy{betrayer=self}` memory event weighted by prior alliance.
   - Chronicle entry.
2. Cascade rule on `JoinGroup`:
   - Add to Group.members.
   - For each existing member, emit `AllyJoined` memory event (mild positive valence).
3. `binding_oaths_with(g)` view — reads active_oaths from memory.beliefs where `oath.kind ∈ {Loyalty, Service}` and `oath.target = g`.
4. `n_conflicting_group_pairs(self)` feature on self contextual block.
5. Reward: `-0.5 on LeaveGroup(g) where self.role_in(g) = Leader` (leaders carry a higher cost for leaving) — incentivises Founder-attrition story arcs.
6. `BreakOath(oath_idx)` as a distinct action (`systems.md:254`). Mask permits only when the oath exists. Triggers `OathBroken` chronicle + reputation drop across the target Group.

**Gaps / open questions:**

- **Multi-membership conflict resolution.** If I'm in G_home and G_other and they go to war, what forces me to pick? Options:
  1. **Passive — agent's own reward gradient.** Attacking own-group-members is -1.0; avoiding creates tension; eventually agent LeaveGroup one side. Emergent but slow.
  2. **Cascade — automatic expulsion.** When `G_home.standings[G_other] = AtWar` is declared and `self ∈ G_home.members ∩ G_other.members`, emit `GroupExpelled{group, agent, reason=CompetingLoyalty}` on one side (which? lowest standing? lowest tenure?).
  3. **Mask pressure.** Add a mask predicate that forces a "must emit LeaveGroup(one_of_them)" action within N ticks when conflict detected. Violates single-tick-action semantics.
  Cleanest: (1) with the optional cascade for extreme cases — `tick - conflict_detected > 500 ∧ agent_took_no_side` → cascade picks lowest `standing × tenure` group and emits `GroupExpelled`.
- **Atomicity of "defect."** LeaveGroup + JoinGroup are two actions. Between them the agent is factionless. Is that OK gameplay? Probably yes — feels realistic. A failed defection (got kicked out of home, never accepted by rival) produces the interesting "wandering exile" state.
- **`JoinParty` variant.** Current doc mentions `JoinParty` for party/quest contexts (`spec.md:64`). Unify under `JoinGroup` — a party is just `Group{kind=Party}`.
- **Minimum-viable extension:** `LeaveGroup`/`JoinGroup` cascade handlers, `WasBetrayedBy` memory-event emission on leave, `binding_oaths_with` view, `n_conflicting_group_pairs` feature. ~1 week.

**Related stories:** 45 (war pressure motivates defection), 46 (losing claimant defects), 50 (low reputation increases defection attractiveness), 54 (diplomatic auction may include "vassal commitments" that function as enforced membership).

---

#### Story H.50: Renowned characters emerge from reputation
**Verdict:** SUPPORTED (after the user's reframe — drop `is_legendary` boolean; reputation is a continuous signal that abilities read)

**User's framing note:** "Legendariness is fluffy. There should be a reputation system, and abilities should be capable of being impacted by reputation, so that way a legendary hero can emerge organically without making a new system for it."

**End-to-end trace:**

1. **Reputation is a continuous derived view, not a stored field.**
   ```
   reputation(npc) = Σ_witnessed_events_in_settlement weighted_by (impact × recency_decay)
   ```
   (Per `systems.md:532` — already in the derivable-view list.) Features the view pulls from:
   - `chronicle_mentions(npc)` — count of chronicle entries mentioning npc
   - `witnessed_positive_events(npc)` — sum over all `memory.events` of all other npcs where npc is a subject with positive `emotional_impact`
   - `witnessed_negative_events(npc)`
   - Both decayed by `recency_decay(tick - event.tick)`.

   Output: a scalar per (npc, observer) pair, OR a global scalar per npc. Both are useful; the global view is cheap (one float) and drives world-level phenomena; the per-pair view is richer (the same deed is famous in hometown, unknown abroad).

2. **Reputation in observation:**
   - `self.fame_log` already in self contextual (§2.1.2) — this becomes the primary reputation atom. Rename to `self.reputation_log` to clarify.
   - `self.reputation_summary` already listed as a feature.
   - `known_actors[K=10]` slots: add `other_reputation_known` float per slot. Derived.
   - `nearby_actors[K=12]` slots: add `other_reputation_visible` if nearby actor is famous enough. Derived.

3. **Reputation drives action emission (policy):**
   - High self-reputation + high ambition → higher probability of emitting ambitious PostQuests (Conquest, Charter, FulfillProphecy). Policy learns this from reward correlation (`+Δ(fame_log)` on important deeds).
   - Observing a high-reputation rival raises anxiety / fear (observation features: `known_actors[k].other_reputation_visible` → self.anxiety via emotion kernel).

4. **Reputation drives mask (eligibility):**
   - Charter petitions: `PostQuest{type=Charter, target=Settlement(s)}` mask gates on `self.reputation_log > CHARTER_FAME_THRESH ∨ self ∈ settlement.charter_eligible_by_peerage`. High-reputation commoners can petition; low-reputation nobles can by membership.
   - `Claim` quests in high-reputation groups: `self.reputation_log > MIN_CLAIM_REP` to prevent commoners from claiming a kingdom.
   - Auction visibility: some Diplomatic auctions only open to `visibility ⊇ reputation > THRESH`.

5. **Reputation drives ability effects (user's explicit reframe):**
   Abilities in `.ability` DSL files already carry effect + tag + scaling definitions. Extend them to read reputation as a scaling input:
   ```
   ability Intimidate {
     effect Fear {
       target: Agent
       magnitude: base + 0.1 × self.reputation_log
       duration: 3 ticks
     }
     conditions: relationship(self, t).trust < 0  ∨  t.creature_type.can_speak
   }
   ```
   Combat damage scaling:
   ```
   ability Attack {
     effect Damage {
       target: Agent
       magnitude: self.attack_damage × (1 + 0.05 × self.reputation_log)
     }
   }
   ```
   Persuasion checks: `Bid` payment in a Diplomatic auction can include `reputation_bid` as a Payment variant. Reputation spent decreases; reputation earned from successful Bid increases.

6. **Cascade emits reputation-affecting events:**
   - Every meaningful event already has `chronicle_category` + `entity_ids`. The reputation view sums over chronicle mentions by default.
   - `BehaviorThresholdCrossed{npc, tag, tier}` event (`systems.md:460`) fires when an NPC crosses cumulative behavior thresholds (e.g. "killed 10 monsters" or "saved 3 settlements"). This events reads like a reputation spike emission; cheap to derive from behavior_profile.
   - `LegendAscended` is NO LONGER EMITTED AS A SPECIAL EVENT. Instead, "legendary" is a predicate: `reputation_log(npc) > LEGENDARY_THRESH`. At UI render time we can label npcs that cross the threshold as "the Legendary," but that's cosmetic — the underlying data is the continuous reputation.

**Emergent outcomes player observes:**
- An unknown commoner grinding small deeds slowly raises reputation. Eventually their combat damage is measurably higher, they can petition charters, and nearby NPCs react with respect/fear.
- A "legendary hero" is just the top-K reputation NPC(s) at any tick; no special flag. Death of a legend is `EntityDied` of a high-reputation npc → cascade increases grief in everyone who witnessed their deeds, via memory.events lookup.
- Infamous outlaws have negative reputation (NO separate infamy dimension — just signed). Their Intimidate scales from negative-rep = fear; their charter petitions are mask-rejected.

**How the DSL supports this:**

- Abilities are data-driven in `.ability` files — effects already have scaling expressions. Adding `self.reputation_log` to the scaling expression grammar is purely additive (spec.md §3 "DSL surface").
- `reputation(npc) = Σ chronicle_mentions × weighted` is a derived view, no new state.
- `fame_log` already in observation (§2.1.2).
- Continuous scalar means the decision policy can learn graduated eligibility, not cliff-edge behavior.

**Implementation walkthrough:**

1. Implement `reputation(npc) -> f32` as a derivable-view over chronicle + memory.events. Memoize per-tick.
2. Rename `fame_log` to `reputation_log` in observation (or keep the alias; both point to the same view).
3. Add `self.reputation_log` and `other.reputation_visible` as accessible scaling inputs in ability DSL grammar.
4. Port mask predicates that previously read `is_legendary(npc)` (per `systems.md:149`) to continuous thresholds: `reputation_log(npc) > LEGENDARY_THRESH`.
5. Chronicle a "Legend ascended" entry automatically when `reputation_log(npc)` crosses the threshold — cosmetic, for UI. Do not gate gameplay on this entry.
6. Extend ability DSL examples: Attack scaled by reputation; Persuasion check scaled; intimidation fear duration scaled. Hero templates opt-in.
7. Reward hook: `+0.02 × Δ(reputation_log)` per tick. Already present via `Δ(self.fame_log)` — just rename.

**Gaps / open questions:**

- **Per-pair vs global reputation.** Global scalar is cheap and good enough for most. Per-pair is more realistic (only locals know the hero's name) but costs more — O(npcs × known_actors). Compromise: global scalar plus a `reputation_known_in_settlement[home]` bonus, so locals over-represent the hero. Leave per-pair for future iteration.
- **Reputation decay.** Old deeds fade. `recency_decay(Δt) = exp(-Δt / HALF_LIFE_TICKS)`. Parameter to tune.
- **Negative vs positive reputation.** Single signed scalar? Or two dimensions (fame + infamy)? User's reframe says single continuous. I lean single signed; an outlaw has reputation=-5, a hero has +5. Intimidate ability uses |reputation|.
- **Fairness across settlement sizes.** Small-settlement hero can hit high reputation cheaply (few witnesses, dense chronicle). Normalize by `chronicle_density(settlement)`? Only if it produces visibly broken gameplay.
- **Minimum-viable extension:** `reputation(npc)` view, observation feature rename, ability DSL scaling grammar extension, chronicle cross-linking for automatic legend-threshold narration. ~1 week.

**Related stories:** 45 (high-rep leader's Conquest more likely to succeed through morale), 46 (reputation gates Claim eligibility), 53 (witnessing high-rep heroes' deaths triggers heavier grief), 54 (reputation as bid currency in diplomatic auctions).

---

#### Story H.51: Long wars affect economy
**Verdict:** SUPPORTED (and this story is the strongest argument for the emergent-cascade design)

**User's framing note:** "ESSENTIAL."

**End-to-end trace — the cascade chain:**

T=0: Leader posts `PostQuest{type=Conquest, party=Group(G_self), target=Group(G_rival)}`. Cascade sets `G_self.standings[G_rival] = AtWar`, vice versa.

T=1 … T=2000: the following cascades fire as emergent consequences. Name each event and the derived view it feeds.

**(A) Settlement raid cascade:**
1. G_self member emits `micro_kind=Attack` against G_rival member near G_rival's settlement S_rival. Standard combat events: `EntityAttacked`, `EntityDied`.
2. `G_self` leader periodically emits `PostQuest{type=Raid, target=Settlement(S_rival)}`. Party members who accept move to S_rival and attack defenders. `BattleStarted{grid_id=S_rival.grid_id}` / `BattleEnded{grid_id, victor}`.
3. On raid success: cascade rule fires `TreasuryLooted{settlement=S_rival, amount=T_loot, looter_group=G_self}` — moves gold from `S_rival.treasury` to `G_self.treasury`. A dedicated event, not a rule hidden in a random system.
4. Cascade also emits `BuildingDamaged{building_id, damage_amount}` for affected structures.

**(B) Settlement treasury depletion (view):**
```
S_rival.treasury_trend = delta_per_tick(S_rival.treasury, window=200)
```
Not a stored field. A derived view that `S_rival.leader_id` observes as a context feature. When `treasury_trend < -CRISIS_THRESH`, leader policy sees the crisis. No special "depletion event" fires; it's just the observed state.

**(C) Trade route disruption:**
1. Caravan agents (merchants) have `economic_intent = Trade` and their normal policy is to `MoveToward(trade_destination)` + `AcceptQuest{type=Trade}`.
2. Their mask for `Travel(through_rival_territory)` is:
   ```
   Travel(path) when safe(path)  // derived view
   safe(path) = ∀ cell ∈ path: cell.owning_group.standings[self.memberships.max_standing_group] ≠ AtWar
   ```
   Once `G_self.standings[G_rival] = AtWar`, routes through G_rival territory are no longer safe. Caravans' policy reroutes (longer paths → less profit → fewer trips) or refuses.
3. No caravan → `TradeRoute.strength` decays (`state.md:213`, "decays without activity, abandoned < 0.1"). Purely time-based decay of the emergent route quality.
4. Chronicle: "The silk road between X and Y has fallen silent."

**(D) Commodity price shift (view over `state.md:22`):**
```
S_rival.prices[c] = base_price[c] / (1 + S_rival.stockpile[c] / (population × halflife))
```
With fewer caravans delivering to S_rival and consumption ongoing, `stockpile[c]` drops. Prices rise. Local NPCs' `price_beliefs` update via trade observation → they emit more `Bid` actions at high prices, migrate to cheaper settlements, or starve.

**(E) Population loss:**
1. Every `EntityDied` event with `entity.home_settlement_id = S_rival` decrements the population view. `population(S) = count(alive entities where home_settlement_id == S)` — derived.
2. Conscription: `G_rival` leader emits `PostQuest{type=Defend, party=Group(G_rival)}` pulling more members into the war → more casualties → more deaths → compounding.
3. Starvation from (D): `NeedStarvation{npc}` events fire when `needs.hunger < STARVE_THRESH` for N ticks → `EntityDied{cause=Starvation}`.
4. Net: population(S_rival) drops over war duration.

**(F) Work output decline (view):**
- `production_rate(S) = Σ agents in S with work_state=Work, weighted by skill`. As population drops, production drops. No special production system needed.
- Buildings with `worker_ids.len() < work_capacity × 0.3` stop producing at full rate (already modeled via work_capacity).

**(G) Morale collapse:**
- Every `EntityDied{entity.settlement = S_rival}` triggers `BondGrief{witnessing_npcs}` via memory ring. Each witness's emotion.grief spikes by 0.6, decays slowly (-0.01/tick = >60 ticks to decay to 0).
- Aggregate effect on settlement: `morale_summary(S) = avg(morale over agents with home_settlement_id = S)` drops. Observable by S_rival leader.

**(H) Rebellion / defection cascade (story 49 referential):**
- Low-morale, low-treasury, war-weary members with high ambition emit `LeaveGroup(G_rival)` or `JoinGroup(G_self)`. Defection rate scales with `war_exhaustion(G_rival)` derived view — see below.

**(I) war_exhaustion view:**
```
war_exhaustion(G) = Σ_over_current_war (
    casualty_events × w1 +
    treasury_deltas × w2 +
    duration × w3
)
```
(Per `systems.md:59`.) Surfaces on leader's self.contextual block. When high, leader policy emits `PostQuest{type=Peace, target=G_self, reward=Reparations}` or `PostQuest{type=Submit}` (vassalization). Cascade resolves the war as a Marriage-like MutualAgreement auction.

**Total player-visible chain:**

Leader posts war → raids loot treasury → treasury drops (view) → trade routes fail safety-mask → caravans stop → commodities scarcen → prices spike → starvation + low morale → population drops → production drops → defection spikes → new war weariness → leader sues for peace or loses succession.

**What is NOT in this design:** a system called "war_effects_on_economy" that scans factions and applies drops. Every link in the chain is either (a) an existing action (Raid, LeaveGroup, Travel) with a pre-existing mask predicate, (b) a cascade rule on a first-class event (TreasuryLooted, EntityDied), or (c) a derived view (treasury_trend, safe_path, population).

**How the DSL supports this:**

- `standings[G]: Map<GroupId, Standing>` drives the `safe(path)` mask predicate that disrupts trade (`state.md:604`).
- `TradeRoute.strength` is already a time-decayed derived view (`state.md:213`).
- `population` is a derived view: `count(alive entities where home_settlement_id == S)` (`state.md:558`).
- Price formula is already derived from stockpile and population (`state.md:446`).
- `war_exhaustion(G)` view is already spec'd (`systems.md:59`).
- Cascade rules on `EntityDied`, `TreasuryLooted`, `BuildingDamaged` compose.

**Implementation walkthrough:**

1. Confirm cascade rules emit: `QuestPosted{Conquest}` → update standings; `BattleEnded{victor}` → emit `TreasuryLooted` when near settlement, `BuildingDamaged` when applicable.
2. Add `safe(path)` derived view; `Travel` mask gates on it. Caravan agent policy reroutes.
3. Confirm `TradeRoute.strength` decays per tick without activity (already spec'd).
4. Confirm population view recomputes O(n_entities) and is cached.
5. Add `war_exhaustion(G)` view accumulating casualty + treasury + duration contributions. Surface as a context feature on leader observation.
6. Add `PostQuest{Peace}` mask: `any(standings[g] = AtWar for g) ∧ war_exhaustion(g) > EXHAUSTION_THRESH`.
7. Verify `price_beliefs` update from trade observation → drives Bid policy without special code.

**Gaps / open questions:**

- **Granularity of raid damage.** `TreasuryLooted.amount` is what? `min(S.treasury × loot_fraction, carry_capacity_of_raiders)`. Parameter tuning; not a design gap.
- **Who raids?** A `party_scope=Group(G_self)` Conquest quest implies all G_self members can raid, but who decides when? Either leader emits sub-PostQuests `Raid(S_rival)` during the Conquest quest window, or members autonomously `MoveToward(S_rival) + Attack`. Both plausible. Cleanest: sub-quests for structured raids; free action for opportunistic raiders.
- **Neutral settlements affected by neighboring war.** A settlement unaligned with either group may still see caravans re-route around contested roads. This falls out automatically — the `safe(path)` predicate applies wherever the path crosses contested territory, regardless of the caravan's origin.
- **War ending during peace quest.** Peace quest's MutualAgreement auction needs both leaders' acceptance. What if one leader dies mid-war? Succession runs (story 46), new leader inherits ongoing wars via group standings. They inherit — they don't auto-inherit the peace offer. Good behavior.
- **Minimum-viable extension:** `safe(path)` view + Travel mask, `TreasuryLooted` event + cascade, `war_exhaustion` view + Peace mask. The cascade chain is largely already spec'd across system docs; wire up the missing views. ~2 weeks.

**Related stories:** 45 (triggering war), 46 (leader death mid-war + succession), 49 (defection from losing side), 54 (treaties through diplomatic auctions).

---

#### Story H.52: Children inherit parental traits
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Trigger.** Two married agents A and B in a Family group co-located and personality.social_drive high → A emits:
   ```
   PostQuest{
     type=HaveChild,                 // new QuestType variant
     party_scope=Group(Family_id),
     target=Agent(self),             // symbolic — "our partnership is the target"
     reward=Birth,
     terms=RequireAcceptance(B)
   }
   ```
   (Or simpler: a non-quest cascade — when both spouses emit `Cohabit` and `needs.social_drive` met for both, spontaneous conception event fires probabilistically. The quest form gives more agency; the spontaneous form is simpler. Either works. I'll use spontaneous for MVP.)

2. **Conception event.** `ChildConceived{parents=[A, B], tick}` fires. After gestation delay (say 500 ticks), `ChildBorn{parents=[A, B], child_id=new_id, settlement=Family.home}` fires.

3. **Spawn handler (ChildBorn cascade — the essential bookkeeping):**
   - New AgentId allocated via `next_id()`. (`systems.md:197` identifies this as essential bookkeeping.)
   - New agent spawned with:
     - `creature_type = A.creature_type` (inherit species)
     - `alive = true, level = 0, hp = initial_hp, pos = Family.home_pos`
     - `personality[i] = blend(A.personality[i], B.personality[i], with_variance)` — proposal:
       ```
       child.personality[i] = lerp(A.personality[i], B.personality[i], 0.5)
                            + gaussian_noise(mean=0, std=0.1)
                            clamped to [0, 1]
       ```
     - `needs[i] = default_newborn_needs[i]`
     - `memberships = [Membership{group=Family, role=Member}]`  (inherits family group)
     - `home_settlement_id = A.home_settlement_id`
     - `parents = [A.id, B.id]`, `children = []`
     - `class_tags = []`, but with a `class_bias_weights` derived from parents' behavior_profile tops — so the child starts with "kitchen smell" of mother's cooking bias, "hammer smell" of father's smithing bias. Not class levels, just biases that speed up learning.
     - `cultural_bias = avg(A.cultural_bias, B.cultural_bias) + noise`
   - A.children.push(child); B.children.push(child). A.emotions.joy += 0.4, B.emotions.joy += 0.4.
   - Settlement.population grows by 1 (derived view, recomputed next tick).

4. **Observation of newborn child:**
   - Standard observation — same schema as adults (they use the same observation packer).
   - Newborn features:
     - `self.level = 0, self.hp_pct = 1.0, self.max_hp_log = small`
     - Personality tensor has parents' blend — already in spawn.
     - `memberships[k]` includes Family.
     - `known_actors[k]` empty initially — no established relationships. A and B appear in `nearby_actors` once proximity established (same home), and relationship_kind_one_hot = Kin via the Family membership overlap.
   - A minimum-age mask ensures the child doesn't emit adult actions until sufficient level. Actions like `PostQuest{Marriage}` mask on `self.level > ADULT_LEVEL`; `Attack` mask on `self.level > COMBAT_LEVEL`.

5. **Emergent outcomes player observes:**
   - Child of a compassionate healer + compassionate priest → inherits high compassion → natural healer.
   - Child of high-ambition leaders → inherits ambition → grows into Claim-quest candidate.
   - Child's initial trust toward parents = HIGH (via memory events of being raised, OR just seeded high: `relationship(child, parent).trust = 0.7`).
   - Family lineages develop characteristic personality clusters over generations.

**How the DSL supports this:**

- Agent fields `parents: Vec<u32>`, `children: Vec<u32>` already exist (`state.md:283–284`).
- `Group{kind=Family}` is the lineage container (`state.md:638`).
- Personality is a 5-dim float — trivial to blend (`state.md:102`).
- Cascade rule on `ChildBorn` emits spawn with inherited fields — event-sourced, declarative.
- Observation is uniform — children use the same packer as adults.

**Implementation walkthrough:**

1. Define `ChildConceived`, `ChildBorn` events.
2. Spawn handler emits new agent with:
   - `personality = 0.5 × (A.personality + B.personality) + gaussian(0, 0.1), clamped`
   - `class_bias_weights = merge_top_k(A.behavior_profile, B.behavior_profile, k=3)` — seeds learning biases
   - `cultural_bias = avg(A, B) + noise`
3. Add `Family.members.push(child); A.children.push(child); B.children.push(child)`.
4. Add mask predicates for adult actions gated on `self.level > N`: `PostQuest{Marriage}`, `PostQuest{Conquest}`, `Bid`, `SwearOath`. Children are observers + learners only for early ticks.
5. Add `child_compatibility(child, parent) = 0.7 + 0.1 × personality_similarity` as relationship trust bootstrap. Seeded at spawn.

**Gaps / open questions:**

- **Gestation duration vs tick scale.** 500 ticks = 500 × 100ms = 50s of real-time. Reasonable for a long game. Parameter.
- **Child aging / leveling.** Children need to "grow up" — levels increase via some XP gain. Either automatic (`level = min(age_in_ticks / ADOLESCENCE_TICKS, MAX_CHILD_LEVEL)` — derived view, no XP) or active (child works simple tasks, earns XP). Either works; automatic aging is simpler and doesn't require special teaching code. Adult status is a predicate: `level > ADULT_LEVEL`.
- **Orphaned children.** If A and B die before child.level = ADULT, does the child survive? Cascade on `EntityDied{entity=parent}` should check for dependent children; if parent dies with children, either the Family group continues under `surviving_parent` or the child is informally adopted by another Family member via `JoinGroup` (cascade-triggered, not agentic — the child isn't agentic yet).
- **More than two parents / adoption / same-sex unions.** Design should accept any (parents: Vec<AgentId>) tuple from Family.members. The blending formula handles k>=1 by averaging. Culture determines marriage eligibility; the physics is uniform.
- **Creature_type inheritance for cross-species.** `predator_prey(A, B)` probably blocks most cross-species marriages via mask (marriage_eligibility). Stick to same-species for v1.
- **Teaching / apprenticeship bias.** Extra bias if child `JoinGroup(Apprentice)` under a parent — mentor_lineage already exists (`state.md:285`). The child joins as `role=Apprentice`; teacher's behavior_profile accelerates child's class xp gain. Works.
- **Minimum-viable extension:** `ChildBorn` spawn handler with personality blending + cultural_bias averaging + class_bias seeding, age-gated masks, `child_compatibility` trust bootstrap. ~3 days.

**Related stories:** 46 (grown child becomes Claim candidate using inherited ambition), 47 (marriage precedes this), 49 (children inherit allegiance to parent's groups as starting memberships).

---

#### Story H.53: Emotional reactions to witnessed events
**Verdict:** SUPPORTED

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Event emission (ground truth).** A significant physical event fires — `EntityDied{id=victim, killer=attacker, tick}`. This is always first-class.

2. **Witness detection.** For the event's settlement/location, compute the set of nearby agents:
   ```
   witnesses = { a : distance(a.pos, event.location) < WITNESS_RADIUS ∧ a.alive }
   ```
   Plus long-distance witnesses via affinity — agents with strong relationships to victim/attacker learn of the event via gossip after a delay (`heard_about_event` cascade). For the per-tick case, spatial witnesses are enough.

3. **Per-witness `Witnessed` event emission (cascade):**
   For each witness W, emit:
   ```
   Witnessed{
     observer=W,
     subject=victim,          // or attacker, or both
     event_type=FriendDied | Murder | Battle | ...
     location=event.location,
     tick,
     entity_ids=[victim, attacker],
     emotional_impact=f(relationship(W, victim), relationship(W, attacker), event_severity)
   }
   ```
   (Per `systems.md:450`. This is already the central emission channel.)

4. **Memory update (cascade side-effect):** The `Witnessed` event lands in `W.memory.events` (VecDeque, capped 20). Oldest discarded.

5. **Emotion derivation (view over memory):**
   ```
   emotions(W) = fold_with_decay(
     W.memory.events,
     kernel: emotion_response_kernel,
     decay: per_tick
   )
   ```
   Kernel table (from `state.md:114`):
   - `FriendDied` (relationship.trust > 0.5) → grief += 0.6
   - `WasAttacked` → fear += 0.5, anger += 0.3
   - `WonFight` → joy += 0.3, pride += 0.3
   - `TradedWith (success)` → joy += 0.1
   - `MadeNewFriend` → joy += 0.2, social += 5

   Per-tick decay happens naturally: `agent_inner` applies `emotions.grief *= 0.99` (= -0.01/tick as spec'd). No separate emotion drift system needed; it's integrated into the primary agent tick.

6. **Observation (self contextual block, §2.1.2):**
   - `emotions(6)`: joy, anger, fear, grief, pride, anxiety — all derived floats from memory fold.
   - `recent_memory_events[K=12]`: raw slot array — each slot has event_type one-hot, age, target_in_nearby index, valence.
   - `memory_beliefs_summary`: counts of LocationDangerous / EntityTrustworthy / etc.
   - `recent_witness_count_5tick`, `recent_witness_count_50tick`: how many witnessed events in recent windows (modulates anxiety).

7. **Behavioral consequences (policy-driven, not hard-coded):**
   - High grief → lowered productivity (action_outcomes EMA on `Work` dips when grief > threshold → policy picks Rest/Socialize).
   - High anger → aggressive action preference (`Attack` utility up on enemies in nearby_actors).
   - High fear → Flee micro primitive preferred.
   - Emotions feed personality drift slowly: repeated grief from friend deaths lowers compassion (witness to cruelty hardens the heart).

**Emergent outcomes player observes:**
- After a battle in the town square, nearby NPCs visibly mourn — their actions shift to Rest/Socialize, productive work halts.
- A shopkeeper witnessing a theft becomes anxious, installs tiles to block storefront (Build action with defensive tiles).
- Children witnessing their parents' deaths have grief persisting longer (shallower emotion decay via personality.compassion bias) — shapes orphan personality over many ticks.

**How the DSL supports this:**

- `Witnessed { observer, subject, kind, impact, tick }` is the central memory-event emission (`systems.md:450`).
- `emotions(npc) = fold(recent memory.events × emotion_kernels, time_decay)` is a derived view (`systems.md:518`).
- Observation surfaces both the derived emotions AND the raw `recent_memory_events[K=12]` slot array (policy can learn patterns beyond the hand-designed kernel).
- Decay built into `agent_inner` per-tick.

**Implementation walkthrough:**

1. Ensure every first-class event (EntityDied, EntityAttacked, TradedWith, BuildingCompleted, QuestCompleted, ChildBorn, MarriageFormed) has an associated `Witnessed` emission cascade.
2. Witness radius derivation: default spatial (50 units). Social ties expand radius: `W.relationships[subject].familiarity > 0.5` means W hears about the event within N ticks via gossip (`heard_about_event` cascade, delayed emission).
3. Define emotion kernels per `MemEventType`. Weighted table in Registry.
4. `emotions(npc)` view: lazy fold over memory.events with per-event kernel × exp(-age × decay_rate). Memoized per tick at first observation-pack.
5. Confirm observation §2.1.2 exposes both raw and summarized features.
6. Reward hook on witnessed events: `-0.5 on Witnessed{FriendDied}`, `+0.2 on Witnessed{QuestCompleted{ally}}` — encourages learning about social reward.

**Gaps / open questions:**

- **Memory ring size (20).** May be too small for long emotional arcs. If a soldier sees 50 deaths in a battle, only 20 land in memory. The emotion kernel summary partially addresses this (grief accumulates per event then decays slowly). But a legendary-tier trauma warrants longer memory. Proposal: tier memory — last 20 in ring + last 5 "indelible" events that never evict (weighted on `emotional_impact > 0.8`).
- **Belief formation from events.** `beliefs` = `fold(events → belief deltas)`. A Witness{FriendDied at location L by EntityX} should create/strengthen a `LocationDangerous(L)` and `EntityDangerous(X)` belief. Existing semantic-layer work (`state.md:157`); confirm the fold is automatic.
- **Emotion → action reward coupling.** The RL reward should include `-0.01 × self.emotions.anxiety` etc. so policy learns to mitigate negative emotions (socialize after grief, rest after fatigue). Already implicit via `need-recovery × 0.1` in reward block (§2.5), but emotions aren't in the need vector. Add `emotions.avg_valence` as a reward term.
- **Long-distance witness (gossip).** A distant ally should hear about a friend's death eventually. Cascade rule: `FriendDied{victim, tick}` + `relationship(W, victim).trust > 0.5 ∧ W ∉ witnesses` → schedule delayed `Witnessed` emission for W at `tick + gossip_latency(distance)`.
- **Minimum-viable extension:** Witness emission cascade for all major events, emotion kernel table, `emotions(npc)` view with decay, gossip-delay cascade for long-distance witnesses. ~4 days.

**Related stories:** 46 (leader death triggers massive witness cascade, reshapes successor selection), 49 (betrayal witnessed degrades trust broadly), 50 (high-rep NPC death = legendary mourning cascade).

---

#### Story H.54: Faction alliance via bidding
**Verdict:** PARTIAL (auction machinery needs extension for multi-party diplomatic auctions; single-seller, multiple-bidders model is close enough for v1)

**User's framing note:** "ESSENTIAL."

**End-to-end trace:**

1. **Initiation.** Leader of G_self observes threat from rival G_rival: `known_groups[G_rival].standing = AtWar` and `G_self.military_strength < G_rival.military_strength`. Leader emits:
   ```
   PostAuction{
     kind=Diplomatic,
     item=AuctionItem::AllianceTerms{
       obligations: MutualDefense(vs=G_rival),
       duration: 5000 ticks
     },
     seller=AuctionParty::Group(G_self),
     visibility=Visibility::Groups(top_K_neighboring_factions),  // broadcast
     deadline_tick=now+100,
     reserve_price=Payment::None,
     resolution=AuctionResolution::MutualAgreement   // both sides must accept
   }
   ```
   (This is `PostQuest{type=Diplomacy}` in current doc but better expressed as `PostAuction{kind=Diplomatic}` — the mechanism is an auction, not a quest. Unless we overload `PostQuest` to fire an auction internally. Both work; the important thing is the observation + policy.)

2. **Observation (bidder, leader of G_candidate):**
   - `known_groups[K=6]` slot for G_self: `standing_with_me`, `group_strength_log`, `is_at_war_with_my_enemies` (new feature).
   - `active_auctions[K=4]` (new slot, analogous to active_quests) surfaces the new Diplomatic auction. Features: kind, seller, obligations_summary, deadline, visibility.
   - `self.G_candidate.standings` map — do I have interests aligned with G_self (common enemies) or with G_rival (common markets)?
   - Military strength ratios.

3. **Policy (bidder):** emits
   ```
   Bid{
     auction_id,
     payment=Payment::Combination([
       Payment::Promise(MutualDefense(vs=G_rival), duration=5000),
       Payment::Gold(1000),                      // treasury contribution
       Payment::Service(RaiseArmy(troops=50))    // concrete troop pledge
     ]),
     conditions=BidConditions::MustInclude(MutualDefense)
   }
   ```
   Multiple G_candidates may emit bids; G_self's alliance may only accept one (single-winner MutualAgreement) OR accept multiple (multi-winner coalition).

4. **Mask:**
   ```
   Bid(auction_id, ...) when auction.visibility ⊇ self
                           ∧ self.leader_groups[auction.seller.kind]     // I lead a group of same kind
                           ∧ self.memberships[0].kind ∈ AllianceEligible
                           ∧ self.G_lead.standings[auction.seller] ≠ AtWar  // can't ally with someone I'm fighting
   ```

5. **Resolution cascade (MutualAgreement):**
   At auction deadline (or when all parties have bid and seller's "accept_bid(id)" action fires):
   - If seller emits `AcceptBid(auction_id, winning_bid_id)` → `AuctionResolved{auction_id, winner, payment}`.
   - Cascade rule: `AuctionResolved{kind=Diplomatic, winner=G_winner}` →
     - `AllianceFormed{between=[G_self, G_winner], terms=auction.item.obligations, duration=auction.item.duration}` chronicle entry.
     - `G_self.standings[G_winner] = Allied`; `G_winner.standings[G_self] = Allied`.
     - If terms include MutualDefense: whenever `G_self.standings[X] = AtWar`, G_winner gets auto-generated `PostQuest{type=Defend, party=Group(G_winner), target=G_self}` (obligation kick-in).
     - Payment transfers: Gold and Service promises encoded as scheduled future events.

6. **Emergent outcomes player observes:**
   - A weaker faction (G_self) broadcasts for help. Multiple stronger neighbors bid. Seller picks the one offering most concrete military support (policy learns to evaluate Service promises > Gold > Reputation).
   - Alliance chronicled. Flavour text narrates the terms.
   - When G_rival invades G_self, cascade auto-drags G_winner into the defense via the scheduled Defend quest. G_winner's leader may then emit AcceptQuest or try to wriggle out (BreakOath → reputation cascade).
   - Broken alliances are `BreakAlliance(target)` leader actions → `AllianceBroken` event → reputation cascade harms the breaker's Bid chances in future auctions.

**How the DSL supports this:**

- Universal `Bid` verb (`spec.md` §2.2).
- `AuctionKind::Diplomatic` spec'd (`spec.md:231`).
- `AuctionResolution::MutualAgreement` for two-sided consent (`spec.md:240`).
- `Group.standings[G]` drives ally/war switches via cascade (`state.md:604`).
- `Payment::Promise(Reward, u64)` for deferred obligations (`spec.md:193`).

**Implementation walkthrough:**

1. Extend `AuctionItem` enum with `AllianceTerms{obligations, duration}`. `obligations` = enum of `MutualDefense(vs=GroupId)`, `TradeExclusive`, `VassalCommitment`, etc.
2. Extend `Payment` with `Service(ServicePromise)` for troop pledges. Schedule future `RaiseArmy` event.
3. Add mask predicates for PostAuction{Diplomatic} and Bid{Diplomatic}.
4. Add resolution cascade: `AuctionResolved{kind=Diplomatic}` → update standings + scheduled obligation triggers.
5. Add `active_auctions[K=4]` observation slot with Auction summary features (kind, seller, best_current_bid, deadline_tick).
6. Cascade rule for obligation enforcement: on `WarDeclared{G_aggressor, G_self}`, if `G_self.allies (via standings = Allied ∧ obligations include MutualDefense vs G_aggressor)`, emit `PostQuest{Defend}` on ally's behalf automatically.

**Gaps / open questions:**

- **Multi-party auction semantics.** The current `MutualAgreement` resolution is binary (seller + one bidder). Coalitions of 3+ factions require either:
  - Sequential pairwise auctions (A posts, B accepts; B then posts with A already allied, C accepts; ... → coalition), or
  - True multi-party auction with resolution when > K parties accept. Mechanics doc §"Auction state machine" doesn't cover this. Add `AuctionResolution::Coalition{min_parties=K}` variant.
- **Bid currency parity.** Can you bid Reputation in a Diplomatic auction? Services? Gold? The `Payment::Combination` allows heterogeneous bids, but seller's scoring function needs to handle apples-to-oranges. Simplification: `total_bid_value(payment, auction.kind) = Σ component_value_for_kind(c)` — a per-kind valuation function. Ad-hoc but workable.
- **Obligation enforcement.** When ally fails to honor MutualDefense (doesn't accept the auto-generated Defend quest), how does the alliance break? Needs a cascade: `AcceptQuest failed × N_retries` → `AllianceBroken{breaker=ally, obligation_violated}` chronicle entry, standings flip to Neutral (or Tense depending on severity), reputation penalty.
- **Auction fatigue.** If every alliance petition spams a Diplomatic auction each tick, the observation's `active_auctions` slot overflows. Cadence per AuctionKind (schema §6 Q12 already flagged) — Diplomatic auctions maybe 1-per-leader-per-1000-ticks max.
- **`PostAuction` vs `PostQuest{type=Diplomacy}`.** Are they the same or different? I lean: `PostAuction` as a distinct macro verb for auction-initiation (existing `Bid` is the corresponding participate verb). Cleaner than overloading PostQuest.
- **Minimum-viable extension:** `AuctionItem::AllianceTerms`, `Payment::Service`, `AuctionResolution::Coalition`, `active_auctions` observation slot, alliance-obligation cascade, `PostAuction` macro verb. ~3 weeks including the base auction state machine work (which is OPEN in schema §6 Q1).

**Related stories:** 45 (alliances form to counter rising hegemon), 46 (leader change mid-alliance may break terms), 49 (defection from faction also terminates their alliance obligations), 51 (wars trigger cascading alliance activations).

---

### Cross-story summary

Player-facing stories ideally all end-to-end-trace cleanly through:

```
policy observation → masked action selection → event emission → cascade rule → derived view update → next-tick observation
```

#### What holds together

- **Single action vocabulary.** All ten stories use PostQuest / AcceptQuest / Bid / LeaveGroup / JoinGroup / micro primitives. No story needed a new top-level verb — richness lives in `QuestType`, `AuctionKind`, mask predicates, and cascade rules. This is the strongest validation of the unification in `spec.md`.
- **Reputation as continuous.** Story 50's reframe (drop `is_legendary` boolean, use `reputation_log` scalar throughout) integrates cleanly with the existing schema and with ability DSL scaling. Extending abilities to read reputation is a pure additive grammar change.
- **Emergent cascades via views, not systems.** Story 51's war→economy chain is the canonical proof: every link is an event-cascade or a derived view on existing state. No "economic war effect" system appears.
- **Memory ring → emotion view.** Story 53 falls out of existing spec almost verbatim.

#### What's still open (cross-story)

1. **Missing QuestType variants.** `Claim` (succession), `Peace`, `Raid`, `Defend` (as auto-cascade counter-quest). All small.
2. **Missing AuctionKind extensions.** `Diplomatic` with `AllianceTerms` + `Coalition` resolution. Alliance obligations as scheduled deferred events.
3. **Missing observation features.** Each story identifies 1–3 small feature additions:
   - `known_groups[k]`: `military_strength_ratio`, `is_adjacent_territory`, `controls_scarce_resource`, `is_at_war_with_my_enemies`
   - `memberships[k]`: `group_leader_vacant`, `my_tenure_relative_to_other_seniors`, `standings_conflict_count`
   - `nearby_actors[k]`: `is_trespasser`, `other_reputation_visible`
   - Self contextual: `reputation_log` (rename of `fame_log`), `war_exhaustion_of_my_lead_group`, `n_conflicting_group_pairs`
   - New slot array: `active_auctions[K=4]`
4. **Cascade-rule inventory (missing).** Story 45 needs `QuestPosted{Conquest}→standings=AtWar`. Story 46 needs `EntityDied→LeaderVacant` (if leader). Story 47 needs `QuestCompleted{Marriage}→FoundGroup{Family}`. Story 49 needs `LeaveGroup→WasBetrayedBy` emission. Story 51 needs `BattleEnded→TreasuryLooted`. Story 52 needs `ChildBorn→spawn blended child`. Story 53 needs `FirstClassEvent→Witnessed emission`. Story 54 needs `AuctionResolved{Diplomatic}→standings=Allied+Defend cascade`. These are the cascade rules the DSL needs as first-class declarative blocks.
5. **Reward credit assignment.** Conquest (story 45), Alliance (story 54), and to some extent Marriage (story 47) all have delayed rewards. Schema §6 Q14 already flags this. Not story-specific; a foundational RL problem.
6. **Auction state machine (OPEN in mechanics doc).** Stories 46 (election), 49 (defection bidding), 54 (alliance) all depend on real auction implementation. Stub (92-line) is not enough.

#### What the batch proves

- The PostQuest/AcceptQuest/Bid architecture + derived views + cascade rules genuinely collapses ~30 systems into ~5 mechanisms AND supports the entire H-category acceptance test set, with the qualifications above.
- The reframe of story 50 (reputation > `is_legendary`) works better than the original design and suggests other derived views should likewise be continuous rather than categorical (e.g. `fame`, `morale`, `loyalty`).
- Story 45's cold-start problem is real and is solved by treating ambition + contestable-territory features as observation-level signals. Bootstrapping via personality-driven exploration in the utility backend + reward gradient during training covers it.
- Stories 46, 49 validate that succession/defection need NO special system — they emerge from existing actions + the right observation slot features.
- The MISSING pieces are consistent: observation feature appends (small), cascade rule declarations (medium), auction state machine + diplomatic extension (larger). Total implementation effort across H: ~8–10 weeks of focused work, with the auction state machine being the largest single piece.
