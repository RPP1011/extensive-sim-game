# Stories A & B — DSL Authoring + Action/Mask Authoring

Per-story analysis of the user stories proposed in `user_stories_proposed.md` under category **A (DSL authoring)** and **B (Action and mask authoring)**. Each entry cites specific sections of the settled design and flags gaps concretely.

---

### Story 1: Declare a new entity type with baseline fields
**Verdict:** GAP (as written) → REFRAMED as SUPPORTED
**User's framing note:** Raw entity-type declaration is inherently dangerous. We should only support this as abstractions *over* the existing entity types (Agent / Item / Group) — never as new root kinds that force codegen changes.

**How the DSL supports this (reframed):**

The settled design already fixes the entity taxonomy at three kinds (plus an optional Projectile): Agent, Item, Group. See `README.md` — "Settled" list, item "Three entity types: Agent + Item + Group (+ optional Projectile)". The discriminators that differentiate instances within a kind are:

- `Agent.creature_type` (`state_npc.md` — "Identity & Lifecycle" row `creature_type`): Human, Elf, Dwarf, Wolf, Dragon, Goblin, ...
- `Group.kind` (`state_aggregate.md` § "Group (universal)" — `kind: GroupKind`): Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other
- `Item` is a single shape with owner / durability / history; differentiation by `ItemKind` (not yet enumerated in these docs — see story 41 which is the modder version)

So "a new entity type" almost always decomposes to one of:

1. **A new `creature_type`** — story 11 handles this.
2. **A new `GroupKind`** — story 10 handles this.
3. **A new `ItemKind`** — story 41 (modder).
4. **A genuinely new top-level kind** — not supported and per the user's note, intentionally so.

The question then becomes: what does "abstraction over existing entity types" mean in DSL surface terms? Three plausible readings:

- **Alias / newtype** — `entity Centaur = Agent { creature_type: CreatureType::Centaur, default_personality: ..., default_memberships: [Group{kind=Pack}] }`. Pure parameterization; compiler expands to Agent + defaults.
- **View-only** — `view Merchant(a: Agent) = a.memberships contains Group{kind=Guild, specialty=Trade} ∧ a.data.class_tags contains "trader"`. Gives authors a named predicate, produces no storage.
- **Parameterized role** — a Template with default stats/personality/starting-memberships used by a spawner, compiled down to Agent + config. Matches the existing `Template.kind` machinery (`state_npc.md` AgentData row `creature_type`).

**Implementation walkthrough (reframed: add a Centaur-like "entity"):**

Files a developer would touch:
- Add `CreatureType::Centaur` variant (proposal_policy_schema.md §2.1.1, creature_type_one_hot) — story 11.
- Author an `entity Centaur = Agent { ... }` alias block in the DSL source; compiler expands to Agent struct with defaults (`creature_type`, default `capabilities`, default `personality`, default `memberships`).
- Add `hero_template`-equivalent data file for spawn parameters.
- Observation schema gets one new bit in the one-hot (`state_npc.md` — "creature/role" row); schema hash bumps (proposal_policy_schema.md §4).

**Gaps / open questions:**

- No DSL block name is currently proposed for the alias/template form. The policy schema's DSL pseudocode (proposal_policy_schema.md §3) declares observations, action, mask, reward, backend — but has no top-level `entity` block. The docs need a grammar clarification: are templates/aliases DSL declarations or external data files?
- **Proposed extension:** add a top-level `template <Name> : Agent { ... }` declaration whose only role is parameter defaults and implicit group assignments. Forbid any other top-level entity declaration. This nails down "abstraction over existing entity types" without opening the door to new root kinds.

**Related stories:** 10 (GroupKind), 11 (creature_type), 41 (ItemType).

---

### Story 2: Declare typed events
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Events are the universal state-mutation channel. From `README.md` — "Settled" list item "Strict event-sourcing rubric. State mutations are events; current state is a fold over events + entity baseline."

Concrete event taxonomy already catalogued:

- **Low-level WorldEvent enum** in `state_aggregate.md` § "WorldEvent Enum (13 variants)": `Generic`, `EntityDied`, `QuestChanged`, `FactionRelationChanged`, `RegionOwnerChanged`, `BondGrief`, `SeasonChanged`, `BattleStarted`, `BattleEnded`, `QuestPosted`, `QuestAccepted`, `QuestCompleted`, `SettlementConquered`.
- **Auction/quest events** in `proposal_universal_mechanics.md` § "Auction lifecycle as events": `AuctionPosted`, `BidPlaced`, `BidWithdrawn`, `AuctionResolved`, `AuctionExpired`.
- **Per-agent MemoryEvent** in `state_npc.md` § "MemoryEvent" with typed `event_type: MemEventType` (WasAttacked, FriendDied, CompletedQuest, etc.).
- **ChronicleEntry** in `state_aggregate.md` § "ChronicleEntry" with typed `category: ChronicleCategory`.
- **Per-tick StructuralEvent** in `state_world.md` § "StructuralEvent" (`FragmentCollapse`, `StressFracture`).

The reward block (`proposal_policy_schema.md` §2.5) already treats event references as first-class: `+1.0 on event(EntityDied{killer=self ∧ target.team ≠ self.team})`. That's only sound if events are typed and matchable in the DSL.

**Implementation walkthrough (add a new `Harvest` event):**

1. Declare in DSL: `event Harvest { harvester: AgentId, node: ResourceRef, commodity: CommodityKind, amount: f32 }`
2. Compiler emits:
   - A variant on the runtime event enum (plus matching append-only buffer entry)
   - Serialization / replay handlers (the ring is append-only per proposal_policy_schema.md §1 "Event-sourced runtime")
   - A mask for policy-side event pattern matching (`event(Harvest{harvester=self})`)
3. Add cascade rule (see story 4) that reads the event and mutates state (e.g. `Harvest → inventory[harvester][commodity] += amount → if tile.remaining ≤ 0: emit ResourceDepleted`).
4. Reward block can reference it via `on event(Harvest{harvester=self})`.
5. Bump schema hash if event is observation-visible (proposal_policy_schema.md §4 + Open question #17 "Schema hash scope — should hash cover observation only, or also action vocabulary + reward?").

**Gaps / open questions:**

- The `event` declaration block is not yet fleshed out as formal grammar (see README.md Open question #8: "DSL surface details. The pseudocode... isn't a formal grammar yet").
- Event categories split across three buckets (WorldEvent, MemoryEvent, StructuralEvent). Is the DSL single-namespace, or does the author have to pick a bucket? Needs decision in the synthesis doc.
- Several events today carry `String` (`Generic{category,text}`, `EntityDied{cause}`, `ChronicleEntry.text`) which are flagged GPU-hostile in `state_world.md` § Summary. Story 60 ("Determinism bound on text generation") points at the resolution: text fields should be marked non-replayable / non-observation.
- **Proposed extension:** add `event` annotations — `@replayable` (default), `@non_replayable` (text fields), `@observation_visible` (drives schema hash), `@gpu_amenable` (drives backend-compile decision).

**Related stories:** 4 (cascade rules), 15 (versioning), 27 (add rule without breaking), 60 (non-replayable text gen), 64 (ActionKind bumps schema hash).

---

### Story 3: Declare a derived view
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Views are the second pillar after events. `README.md` — "Settled" items "Buildings and resources are derived views over the world's spatial data" and "`is_hostile(a, b)` is a derived view over relationship valence + group war standings."

Concrete view examples already written:

- `proposal_policy_schema.md` §2.3 mask block: `is_hostile(self, t) = relationship.valence < HOSTILE_THRESH ∨ groups_at_war(self, t) ∨ predator_prey(self.creature_type, t.creature_type)`.
- `proposal_policy_schema.md` §3 observation DSL: `self.mood = view::mood(self)`, `view::agent_features_for_observation(other, viewer=self)`, `view::relationship(self, other).valence`, `view::shared_group_count`, `view::known_actors(self)`.
- `state_aggregate.md` § "What's NOT on Group (derived views)": `population`, `is_at_war(other)`, `is_allied(other)`, `wealth_per_member`, `cultural_descriptor`, `reputation_among(other_group)`.
- `state_npc.md` § "Derivation Graph" splits fields into PRIMARY INPUTS, SECONDARY (Computed/Emergent), INFRASTRUCTURE. The SECONDARY block is the view surface.

**Implementation walkthrough (declare `view::is_legendary(a: Agent)`):**

1. Author the DSL: `view is_legendary(a: Agent) = a.data.deeds.len() ≥ 3 ∨ a.data.fame_log > LEGENDARY_THRESH`.
2. Compiler inspects the body for `@materialize` hint (proposal_policy_schema.md §1 settled item "GPU compilation is scoped") — if lazy (default), the view compiles to a function called per-read; if eager, it emits event-fold logic that materializes the value whenever inputs change (see story 26).
3. Mask and observation blocks reference `view::is_legendary(self)` directly.
4. For materialized views, the compiler wires subscription to input events (e.g. `ChronicleEntry{entity_ids ∋ a.id}` triggers recompute).
5. Story 40 ("Flamegraph attribution") implies views need per-declaration timing instrumentation — the compiler tags each view body with a timing ID.

**Gaps / open questions:**

- Materialization hint syntax not specified. README.md "Settled" row mentions `view::mood(self)` as hot path but not the annotation that marks it eager.
- Views with cross-entity aggregation (e.g. `query::shared_group_count`) need indexing strategy — called out in README.md Open question #7: "Cross-entity mask predicates. `t ∈ quest.eligible_acceptors`, `at_war(self.faction, f)`, etc. need careful index design to avoid O(N²)."
- **Proposed extension:** add two view annotations:
  - `@materialize(on_event: [<event_list>])` — view body becomes an event-fold updater on a stored field.
  - `@lazy` (default) — view body is a pure function evaluated at read time.
  - `@gpu_kernel` — eligible for GPU compilation (per proposal_policy_schema.md §5 "GPU-amenable" list).

**Related stories:** 4 (cascade rules feed views), 26 (eagerly materialize hot views), 40 (flamegraph per-declaration).

---

### Story 4: Declare a physics cascade rule
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

Physics cascade rules are the third pillar. Mutations from events → state changes → emitted events. Cited directly in the user story: `Damage(d) → hp[d.target] -= d.amount → if hp ≤ 0: emit EntityDied`.

Concrete examples already written:

- `proposal_universal_mechanics.md` § "Auction state machine" "Cascade rules":
  - `AuctionResolved{kind=Item, winner=W, payment=P}` → `TransferGold(W → seller, P.amount)` + `TransferItem(seller → W, item)`
  - `AuctionResolved{kind=Marriage, winner=W}` → `MarriageFormed(seller, W)` (updates spouse_id on both)
  - `AuctionResolved{kind=Charter, winner=W}` → `CharterGranted(settlement, W, terms)`
- `proposal_universal_mechanics.md` § "How 'war is a quest' plays out" / "How 'marriage is a quest' plays out" — step-by-step cascade walkthroughs.
- `state_npc.md` § "Membership" — `JoinGroup` / `LeaveGroup` events update the `memberships` Vec.
- `state_world.md` § "StructuralEvent" — voxel collapse emits events that cascade into more events (cascade_collapse in `damage_voxel`).

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
   - All read references are declared fields or `view::` calls (proposal_policy_schema.md §3 "Validation: every observation field must reference declared entity field, view, or event source" — same rule for cascades).
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

### Story 5: Declare a spatial / non-spatial query
**Verdict:** PARTIAL (design intent settled, codegen complexity acknowledged)
**User's framing note:** "Good, but code gen for this has to be very smart to do it well."

**How the DSL supports this:**

Queries are first-class in the observation DSL. `proposal_policy_schema.md` §3:

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

1. **Spatial** — parameterized by position + radius. Indexes: uniform grid, chunk grid, nav grid. See `state_world.md` for the voxel/tile/navgrid stack.
2. **Non-spatial / named** — `query::known_actors(self) = union(spouse_id, mentor_id, apprentice_id, group_leader_ids of my groups, top-K-grudges, top-K-friendships)` (per proposal_policy_schema.md §2.1.5).

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
   - If all membership groups are indexed by `kind` (settled GroupIndex from state_aggregate.md), the compiler can O(k) scan the agent's religion groups and iterate their `members: Vec<AgentId>`.
   - If the resulting set is observation-visible, compiler allocates top-K storage per agent.
3. Annotations needed for smart codegen:
   - `@indexed_by(kind)` on Group — tells compiler Groups are indexed by `kind` so filtering is O(1) rather than scanning all Groups.
   - `@spatial` on a query that takes `(pos, radius)` — tells compiler to route to spatial index (uniform grid or chunk index, not HashMap linear scan).
   - `@top_k(K)` — tells compiler to allocate fixed-cap buffer and use partial sort, no Vec alloc (per story 30 "Zero per-tick allocations").

**Codegen complexity (user-flagged):**

This is the hardest compiler piece. Queries combine:

- Spatial indexing — multiple backends (KD-tree, uniform grid, chunk+axis-aligned voxel grid from `state_world.md`).
- Cross-entity joins — "nearby_agents" ⋈ "memberships" to find hostile-faction members in range.
- Top-K ranking with custom sort keys.
- GPU vs CPU dispatch — spatial queries might be GPU-amenable (proposal_policy_schema.md §5 "Hybrid" list: "Spatial queries (nearby_actors, threats) — could be GPU with hash-grid texture; first implementation likely CPU with good cache layout").

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

### Story 6: Declare an invariant
**Verdict:** PARTIAL (design acknowledges them; grammar/surface not yet specified)
**User's framing note:** "Great."

**How the DSL supports this:**

Invariants appear implicitly across the design. Several declarations IMPLY invariants without a formal block:

- **Single-spouse** — `state_npc.md` AgentData row `spouse_id: Option<u32>` — type system expresses "at most one". The story asks for a richer version: "no agent has multiple Marriage memberships."
- **Bounded collections** — `state_npc.md` § "Relationship" "Capped at 20 relationships per NPC" — invariant + eviction rule.
- **Append-only observation schema** — proposal_policy_schema.md §4 "Append-only schema growth — when adding features, append; never reorder existing slots." This is an invariant on the schema, caught at compile time via CI guard.
- **Memberships must reference alive groups** — implicit from `state_aggregate.md` `dissolved_tick: Option<u64>`. Story 61 ("Group dissolution with active quests") surfaces the same concern.
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

- No `invariant` block in current DSL pseudocode (proposal_policy_schema.md §3 covers observation/action/mask/reward/backend but not invariants).
- Compile-time vs runtime split is unclear — the user story explicitly says "compile time where possible, runtime otherwise."
- Invariants over event sequences (e.g. "no agent AcceptQuest-s after they're dead") need temporal expression, not just current-state.
- **Proposed extension:** add a top-level `invariant <name>(<scope>) { <predicate> }` block with compiler modes:
  - `@static` — enforced at compile time via type system / dataflow analysis. Compiler errors otherwise.
  - `@runtime` (default) — checked after every cascade that touches a field in the predicate's support.
  - `@debug_only` — runtime check, panic only in debug builds (too expensive for release).

**Related stories:** 15 (versioning — the schema is an invariant), 27 (compose without breaking), 59 (malformed action drop), 61 (group dissolution dangling refs), 62 (agent death mid-quest), 63 (simultaneous marriage proposal race).

---

### Story 7: Add a new agentic action verb
**Verdict:** GAP (as written) → REFRAMED as SUPPORTED-in-principle, grammar gaps
**User's framing note:** Raw verb addition is dangerous — same concern as new entity types. Prefers action primitives that compose verbs.

**How the DSL supports this (reframed):**

The settled design already reduces ~110 proposed verbs to ~16 categorical actions. `proposal_universal_mechanics.md` § "Total action vocabulary":

```
3 macro mechanisms:    PostQuest, AcceptQuest|JoinParty, Bid
~13 micro primitives:  movement(3) + combat(3) + resource(4) + construction(3) + social(2) + memory(1)
─────────────────────────
~16 categorical actions
```

The micro primitives (proposal_universal_mechanics.md § "Micro primitives"):

- Movement: `MoveToward`, `Flee`, `Hold`
- Combat: `Attack`, `Cast`, `UseItem`
- Resource: `Harvest`, `Eat`, `Drink`, `Rest`
- Construction: `PlaceTile`, `PlaceVoxel`, `HarvestVoxel`
- Social atomic: `Converse`, `ShareStory`
- Memory atomic: `Remember`

The macro mechanisms carry all the high-level decision richness via parameter heads: `QuestType`, `PartyScope`, `QuestTarget`, `Reward`, `Payment` (proposal_policy_schema.md §2.2).

**So "adding a new verb" collapses to one of:**

1. **A new mask predicate over existing primitives** — e.g. "Pray" = `Converse(target=shrine_agent) when self.memberships ∋ Group{kind=Religion}`. No runtime change, pure DSL.
2. **A new `QuestType`** — if the verb is a long-duration structured action (Heist, Conquest). Story 9 handles this. "Pray" as a multi-day pilgrimage is `QuestType::Pilgrimage` (already in proposal_universal_mechanics.md's QuestType enum).
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
   Wait — but `Pray` isn't a `micro_kind` in proposal_policy_schema.md §2.2. Two sub-options:
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
- When a genuinely-new primitive is needed, the schema hash (proposal_policy_schema.md §4, story 64) bumps. Keep this path, but gate it behind an explicit "new action primitive" compiler flag so authors must acknowledge the ML consequence.

**Related stories:** 8 (mask predicates), 9 (QuestType), 20 (reward declaration), 64 (new ActionKind bumps schema hash).

---

### Story 8: Write a mask predicate
**Verdict:** SUPPORTED
**User's framing note:** "Great."

**How the DSL supports this:**

Masks are load-bearing in the current design. `README.md` — "Settled" — "Role power = mask + cascade, not a smarter policy." `proposal_policy_schema.md` §2.3 is the mask block:

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

Compilation target (proposal_policy_schema.md §2.3 tail):

- Per-tick boolean tensor `[N × NUM_ACTIONS]`
- Per-head for sampling efficiency: `categorical_mask[N × NUM_KINDS]`, `target_mask[N × NUM_SLOTS]`
- GPU-amenable (proposal_policy_schema.md §5 "GPU-amenable" list: "Per-head mask evaluation — boolean tensor computation per NPC").

**Implementation walkthrough (add `mask Pray(self) when self.memberships ∋ Group{kind=Religion}`):**

1. Author predicate in mask block:
   ```
   mask Pray(self) when self.memberships ∋ Group{kind=Religion}
                     ∧ distance(nearest(Structure{kind=Shrine})) < REACH
   ```
2. Compiler decomposes:
   - `self.memberships ∋ Group{kind=Religion}` — looks up `memberships[i].group_id` for each membership, reads that group's `kind`, checks equality. Cross-entity reference → CPU (proposal_policy_schema.md §5 "GPU-hostile" list).
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

### Story 9: Add a new QuestType
**Verdict:** SUPPORTED (reframed — QuestType is a label on data, not a primitive)
**User's framing note:** Same danger concern as new verbs.

**How the DSL supports this (reframed):**

`QuestType` is structured data, not code. From `proposal_universal_mechanics.md` § "Required types":

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

The ONLY place `QuestType` is special-cased is the **cascade rule on completion**. Everything else — party scope, target, reward, deadline, acceptance, quest lifecycle — is polymorphic over the type. `proposal_universal_mechanics.md` § "How 'war is a quest' plays out" step 7: "Quest completes → cascade emits `SettlementConquered`, `Spoils` distribution, reputation/legend updates." That cascade is the only per-type logic; the quest state machine itself is universal.

So "new QuestType" = "new label on the `QuestType` enum + one new cascade rule on `QuestCompleted{type=<New>}`."

The user's reframing is correct: QuestType is a compositional vocabulary (parameter to PostQuest), not a hardcoded primitive. The cascade rule is the only place the type is differentiated.

**Implementation walkthrough (add `QuestType::BuildMonument`):**

1. Extend the `QuestType` enum — one line (proposal_universal_mechanics.md § "Required types").
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
5. Schema hash bumps if `QuestType` is observation-visible. It IS (proposal_policy_schema.md §2.1.6 `active_quests[K=4]` observation slots → story 21 mentions `DeclareWar` as a rare macro action needing up-weighting → QuestType features thread into observation).

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

### Story 10: Define a new GroupKind
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

`Group` is the universal social-collective primitive (README.md "Settled" — "Group — universal social-collective primitive (faction, family, guild, religion, party, hunting pack, settlement, court, cabal)"). `state_aggregate.md` § "Group (universal)":

```
kind: GroupKind | Faction, Family, Guild, Religion, Pack, Party, Settlement, Court, Cabal, Other
```

And the per-kind shapes table (state_aggregate.md § "Per-kind shapes"):

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
3. Extend observation one-hot (proposal_policy_schema.md §2.1.3 `group_kind_one_hot(8)` → must go to 9+, bump schema hash).
4. Extend the primary_group_kind and n_religious_groups-style summary atoms if Covens aren't religions per se.
5. No cascade rules needed — Group dynamics (JoinGroup, LeaveGroup, standings updates) are kind-agnostic.

**Gaps / open questions:**

- `EligibilityRef` is mentioned (state_aggregate.md § Group "Recruitment & Eligibility" `eligibility_predicate: EligibilityRef`) but the format is open. Needs a predicate language.
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

### Story 11: Add a new creature_type
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

`creature_type` is the per-Agent discriminator. `state_npc.md` § "Identity & Lifecycle" — `creature_type: CreatureType`. Per README.md "Settled" — "Agent — any agentic actor: humans, wolves, dragons, goblins. Same struct, distinguished by creature_type tag + personality/needs config + group memberships."

The hooks already in place:

- **Capabilities** — `state_npc.md` AgentData row `capabilities: Capabilities` "derived from `creature_type`" (jump/climb/tunnel/fly/siege + can_speak / can_build / can_trade flags).
- **Predator/prey** — `proposal_policy_schema.md` §2.1.4 "Hostility is derived per-pair via `... ∨ predator_prey(self.creature_type, other.creature_type)`".
- **Observation one-hot** — `proposal_policy_schema.md` §2.1.1 "Creature/role (10): creature_type_one_hot(~8), ...".
- **Mask gates** — `proposal_policy_schema.md` §2.3 `Talk(t) when t.creature_type.can_speak ∧ distance(t) < SOCIAL_RANGE`, and `PostQuest{type=Marriage, ...} when self.creature_type.can_marry`.
- **Spawn / template** — `state_npc.md` AgentData row `creature_type: CreatureType ... | constructor (template.kind), monster spawn`.
- **Per-creature personality / hunger drives** — `state_npc.md` § "Personality" "Set at spawn, drifts via events" — defaults can vary by `creature_type`.

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

- `Capabilities` struct fields not fully listed in state_npc.md (mentions "Jump/climb/tunnel/fly/siege flags + can_speak / can_build / can_trade"). Needs formalizing.
- Hunger drives are a gameplay dimension not formally listed in state_npc.md § Needs — they're per-creature. Probably belongs on `creature_type` declaration block.
- Predator/prey table — the `predator_prey(self, other)` view needs a lookup table. Either authored per-type or globally as a matrix.
- **Proposed extension:** `creature_type <Name> { capabilities, defaults, drives, predator_prey }` block. Compiler validates that the one-hot width bumps are handled (schema hash flag).

**Related stories:** 1 (entity abstraction — this is the primary alias path), 8 (mask uses creature_type), 10 (GroupKind eligibility keyed on creature_type e.g. Pack), 48 (monster defend dens — creature_type defaults).

---

### Story 12: Tune a mask without code change
**Verdict:** SUPPORTED
**User's framing note:** "Good."

**How the DSL supports this:**

The design already externalises tunables. Mask predicates in proposal_policy_schema.md §2.3 reference named constants: `AGGRO_RANGE`, `MEAL_COST`, `REACH`, `HARVEST_RANGE`, `SOCIAL_RANGE`, `HOSTILE_THRESH`, `MIN_TRUST`, `THRESH` (population).

Those constants should live in a config file, not in DSL source. The settled design has two precedents for this pattern:

- **Backend weights** — `backend Neural { weights: "npc_v3.bin", h_dim: 256 }` (proposal_policy_schema.md §3 — bottom). Weights are an external artifact hot-swappable (story 23).
- **Schema versioning** — `@since(v=1.1)` annotations (proposal_policy_schema.md §4) separate declaration from metadata, allowing migration.

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
- Reward coefficients (proposal_policy_schema.md §2.5 — `× 0.1`, `× 5`, `+1.0`)
- Mask predicate TREE STRUCTURE? Only if no new action verbs or slots appear — otherwise model doesn't know what to do with the new output dim.

Hot-reload does NOT work for:
- Observation schema changes (breaks model — schema hash check hard-rejects — proposal_policy_schema.md §4)
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

## Cross-cutting observations

**The three-primitive taxonomy holds up.** Entity = Agent|Item|Group. Event-sourced state mutation. Mask + cascade = role differentiation. The entire "dangerous new primitive" axis (stories 1, 7, 9) reduces to additive vocabulary extensions (creature_type, GroupKind, QuestType) that compose with the existing primitives via enum variants + cascade rules. New `micro_kind` is the only path that forces schema-hash churn.

**Schema hash (proposal_policy_schema.md §4) is the contract boundary.** Changes that bump the hash: new micro_kind variant, new observation field, new macro-action parameter width, new creature_type (one-hot bit), new GroupKind (one-hot bit). Changes that DON'T bump: mask predicate changes, new QuestType (if reward width unchanged), new cascade rules, new tunable values, new views (unless materialized and observation-visible).

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

These are the next batch of additions for `proposal_dsl_surface.md` (README.md open iteration #3).
