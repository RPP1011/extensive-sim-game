# User Stories F / G / I / J — Investigation

Scope: debugging & observability (F), modding / extensibility (G), auction / unified mechanics (I), adversarial / failure modes (J). Format per story: verdict, user's framing note, how the DSL supports the story, implementation walkthrough, gaps, related stories.

Cross-reference docs: `proposal_policy_schema.md` (§ 2.1 observation, § 2.2 action heads, § 2.3 mask, § 4 schema versioning, § 5 GPU-scoped compilation), `proposal_universal_mechanics.md` ("Auction state machine", "Auction lifecycle as events", "Open questions §§ 1, 4, 7, 8"), `state_aggregate.md` (Quest, Group, ChronicleEntry), `state_npc.md` (MemoryEvent ring buffer), `state_world.md` (chronicle / world_events taxonomy).

---

## F. Debugging and observability

### Story 34: Trace why a mask evaluated false
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

### Story 35: Inspect cascade fan-out
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
- Event retention. Chronicle is a "bounded ring buffer" (state_world.md line 432); `world_events` is bounded too. A cascade whose root was flushed out of the buffer loses its parent edge. Policy: either raise retention for debug builds, or spill to an on-disk event log that's not ring-buffered.
- Cross-tick cascades. Most cascades complete within a tick, but quest completion can fire a cascade 2000+ ticks after the `PostQuest` action. If `cause` points to an event outside retention, the chain is truncated. This is annoying but acceptable for long-horizon narratives; tools should show "truncated — root was at tick T, outside retention."
- Non-replayable events (see story 60) still participate in the causal DAG — they just don't reproduce under replay. Their structural presence is what matters for debugging.

**Related stories:** 34 (why mask false), 37 (step), 38 (repro), 60 (text gen / replayability).

---

### Story 37: Step through a tick
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

### Story 38: Self-contained tick reproduction
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

### Story 39: Per-agent decision history
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

### Story 40: Flamegraph attribution to DSL declarations
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

## G. Modding / extensibility

### Story 41: Add a new ItemType
**Verdict:** PARTIAL
**User's framing note:** "Essential."

**How the DSL supports this:**
`Item` is one of the three first-class entity types (README "Settled" bullet 9; `systems_economic.md` treats items as full entities with IDs emitted by `ItemCrafted{crafter, item_entity_id, quality, slot}` and despawned by `ItemBroken`). An item composes along orthogonal dimensions:

1. **Identity** — type/kind tag, rarity, base stats, slot.
2. **Ownership / path history** (README: "Item — path-dependent owner / durability / history").
3. **Event reactions** — ESSENTIAL physics: durability integration per tick, `ItemBroken` at `durability <= 0`.
4. **Derived view participation** — `effective_quality(item) = base_quality × durability_frac`, and `effective_stats(npc) = base + Σ equipped_items.modifiers × durability_frac` (systems_economic.md line 551).
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
- **One-hot vector widening.** `proposal_policy_schema.md` § 2.1.1 includes `creature_type_one_hot(~8)`; item kinds would have an analogous vector. Adding a new kind changes its width — this is a schema-hash-breaking change per § 4. Mitigation: reserve slack slots in one-hot vectors (`creature_type_one_hot(16)` with some unused), or accept that mods bump the schema and require retraining/migration.
- **Event handler conflicts.** If two mods add handlers for `ItemUsed`, what order do they fire? Either (a) declarative `on ItemUsed { priority: N }` with priority lanes, or (b) rejectandrequire mods to compose handlers explicitly. User's note on story 43 ("make the entire system modifiable, they can add or delete rules themselves") suggests (b): mods own their rules, and conflicts are a load-time error.
- **Effects system (combat) already has a `.ability` DSL** (CLAUDE.md "Effect System"). If items reference abilities, the world-sim item DSL and the combat ability DSL need a shared identifier space.
- **Stability of entity IDs across reloads** — covered in story 44.

**Related stories:** 43 (override physics rule — user rejected as syntax sugar only), 44 (hot reload), 64 (schema hash on vocab extension).

---

### Story 44: Hot-reload mod changes
**Verdict:** PARTIAL
**User's framing note:** "Good."

**How the DSL supports this:**
Because the DSL is the source of truth for observation schema, masks, cascade rules, and views, a reload regenerates the compiled artefacts. The event-sourced runtime means the *state* is separable from the *rules* — state lives in the world store + entity baselines + event log; rules live in compiled DSL.

**What survives a reload:**
- All primary state: tick, rng_state, entities, groups, quests, chronicle, world_events, tiles, voxels (everything listed under "Primary state" in state_world.md "Summary").
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

## I. Auction / unified mechanics validation

### Story 55: Settlement commodity auction (REFRAMED)
**Verdict:** PARTIAL — reframed per user instruction
**User's framing note:** "I am not aligned with this so much as I think the entire economy should operate on a bidding system. I don't mind if some emergent behavior causes community auctions, but I don't want a physics rule saying every 100 ticks a settlement will put all its merchants in the town square and they will dump inventory."

**Reframe:** No scheduled physics rule. Auctions emerge from agent decisions.

**How the DSL supports this (emergent pathway):**

Auction state exists as a first-class primitive (`proposal_universal_mechanics.md` "Auction state machine" + "Auction lifecycle as events"). There is no top-level periodic kernel that posts auctions. Auctions form when an agent chooses `PostQuest{type=Trade, ...}` or when an agent emits a direct `PostAuction` action (listed in `systems_economic.md` line 124 as `PostAuction(item_id, reserve_price, deadline)`). In both cases the initiating act is an **agent policy decision**, not a scheduled system tick.

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
- **Auction visibility.** `visibility=settlement` requires a cheap check (`self.home_settlement_id == auction.settlement_id`) or a spatial radius. See `proposal_universal_mechanics.md` "Open questions" § 4 (quest discovery: push vs pull).
- **Too many tiny auctions.** If every 0.5-iron surplus becomes an auction, the auction buffer explodes. Mitigation: `reserve_price` threshold in the mask (`self.gold_value_of_good >= MIN_AUCTION_VALUE`), plus a policy-learned bias against trivial auctions.
- **Fast-path for "simple immediate trade."** See proposal_universal_mechanics.md "Risks" — a trivial 2-party trade going through a full auction is overkill. Candidate: private 2-party `Bid` in an ad-hoc auction (already in the universal mechanics examples, line 83 of that doc).
- **Price belief convergence.** Settlement-level prices arise from the clearing prices of auctions. Agents update `price_beliefs` from observed `AuctionResolved` events (memory.record_event → belief_formation). This is the *emergent* price view the original story wanted — but the mechanism is "agents watch auctions" not "settlement runs market every tick."

**Related stories:** 56/57 (user rejected as too abstract / too implementation-specific), 58 (Service auction variant), 54 (Diplomatic auction).

---

### Story 58: Mercenary hiring as auction
**Verdict:** SUPPORTED
**User's framing note:** "Great."

**How the DSL supports this:**
`AuctionKind::Service` is defined in `proposal_universal_mechanics.md` line 231. A `PostQuest{type=Service, ...}` action composes with the auction state machine to produce a hiring workflow. The `Bid → AcceptQuest` cascade is the closing move.

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
- **Withdrawal**. A mercenary who changed their mind mid-contract — does `WithdrawQuest` apply (proposal_universal_mechanics.md "Open questions" § 7)? Likely yes, with reputation / grudge cascade.
- **Discovery cadence**. Auction frequency (`Open questions § 8`): Service auctions post when an NPC decides; no fixed cadence. Discovery is push + pull — faction members get pushed, open-market discovered via spatial/observable slot.

**Related stories:** 55 (commodity auction reframe), 54 (Diplomatic auction), 49 (betray for personal gain — mercenary defection is the same machinery), 41 (items, if service includes "and take this loot").

---

## J. Adversarial / failure modes

### Story 59: Malformed action drops safely
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

### Story 60: Determinism bound on text generation
**Verdict:** SUPPORTED — with strict ID discipline
**User's framing note:** "Essential, text gen should not be load bearing. Use numeric IDs for everything important."

**How the DSL supports this:**
The proposal_policy_schema.md § 4 names schema hash + observation versioning. The universal mechanics doc treats events as typed structures. Text lives at the chronicle boundary (state_aggregate.md "ChronicleEntry", state_world.md "chronicle: Vec<ChronicleEntry>"), which is an *observer* artefact, not a simulation input. The rule: **anything the simulation reads must be a numeric ID; text is cosmetic output only.**

**Data carrying text vs IDs — inventory:**

| Field | Load-bearing? | Type |
|---|---|---|
| `Agent.name` (state_npc.md line 230) | **No** — cosmetic. Numeric `AgentId` drives all sim logic. | `String` for UI; regenerable from `(entity_id, seed)` |
| `Agent.class_tags` (line 274) | **Yes** — ability unlock, behavior bias. | Lowercase `Vec<String>` acting as tags — these should become `Vec<TagId>` (FNV-1a hash at compile time per state_npc.md "Tag hashing" note) |
| `Agent.archetype` (line 318) | **Yes** — ability unlock, stat scaling. | Should be `ArchetypeId` not string |
| `Group.name` (state_aggregate.md line 579) | **No** — cosmetic. `GroupId` is the referent. | String |
| `Quest.name` (line 151) | **No** — UI title. `quest_id` is the referent. | String |
| `QuestPosting.name` (line 179) | **No** — UI title. | String |
| `Building.name` (state_npc.md line 355) | **No** — cosmetic. `BuildingId` + `building_type` drive mechanics. | String |
| `ChronicleEntry.text` (state_aggregate.md line 343) | **No** — purely narrative output. | String, derived from `(category, entity_ids, template_id, tick)` |
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
5. **Memory beliefs** (state_npc.md line 152 `MemEventType` variants) are enums, not strings — good, already compliant.

**Gaps / open questions:**
- **Existing `String` fields that need demotion.** `class_tags: Vec<String>` and `archetype: String` on agents should convert to ID-based representations with a separate string table for display. This is a pre-existing tech-debt item called out in the state doc ("tag hashing" note).
- **User-authored content (mods, quest names).** A mod defines `item SpellScroll { name: "Scroll of Fire" }`. At load, `"Scroll of Fire"` interns into a `StringId`. The sim sees `StringId`; display renders the string. Hot reload can replace the string table without replay invalidation because IDs remain stable.
- **Chat / dialogue systems** (if the sim grows to include NPC dialogue). These are inherently text-heavy and MUST be in the non-replayable side channel; the *decisions* driving dialogue must be structural (what `topic_id`, what `sentiment_enum`, what `target_id`).

**Related stories:** 35 (cascade — chronicle is a cascade leaf), 38 (repro bundle excludes non-replayable text), 59 (drop events structural, not textual), 64 (schema hash covers structural fields only).

---

### Story 61: Group dissolution with active quests
**Verdict:** SUPPORTED — with explicit per-role cascade
**User's framing note:** "ESSENTIAL."

**How the DSL supports this:**
`DissolveGroup` is a first-class event (state_aggregate.md line 404: "JoinGroup/LeaveGroup, FoundGroup, DissolveGroup"). Groups own active quests (line 617: `active_quests: Vec<QuestId>`). Quests reference parties via `party_scope` and `party_member_ids` (universal_mechanics.md line 156). The cascade on dissolution is declarative.

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

### Story 62: Agent death mid-quest
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

### Story 63: Marriage race condition (REFRAMED to invite-style)
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

### Story 64: Adding a new ActionKind breaks model
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

## Cross-cutting observations

1. **F (debugging) is well-served by the event-sourced + declarative-DSL design.** Story 34 (mask trace), 35 (cascade fan-out), 37 (tick step), 38 (repro), 39 (decision history), 40 (flamegraph) all fall out naturally because every stage boundary is named and every mutation is a typed event. The hard work is retention policy, not expressiveness.

2. **G (modding) is PARTIAL because of one-hot vector widening.** Adding content that contributes to categorical vectors (ItemKind one-hot, creature_type one-hot) is a schema-hash-breaking change. Options: reserve slack slots (wasteful but future-proof), or commit that mod authorship implies schema/model update. User's story-43 stance ("make the whole system modifiable, they can add or delete rules themselves") leans toward explicit ownership of schema by mods.

3. **I (auction) is correctly reframed as emergent-not-scheduled.** Story 55's commodity auction emerges from individual `PostQuest{type=Trade}` decisions; story 58's mercenary hiring is a canonical `AuctionKind::Service` case. Story 54 (alliance via Diplomatic auction) is a close neighbour that the DSL supports symmetrically. The rejected stories (56 charter auction, 57 GPU state) are out of scope for valid design-taste reasons, not gaps in the proposal.

4. **J (adversarial) forces tight discipline on two invariants:**
   - **ID-over-text.** Story 60 mandates numeric IDs for any load-bearing state; text is cosmetic. Practical impact: convert existing `class_tags: Vec<String>` and `archetype: String` to ID-based representations; segregate chronicle text from chronicle structure.
   - **Schema hash over action vocab.** Story 64 extends § 4 to cover action vocabulary and resolves open question #17 in favour of "yes, hash covers vocab." FAIL LOUD at model load for incompatible changes; explicit opt-in migration for append-only changes.

5. **Story 63's invite pattern is more important than it looks.** It generalises to at least 10 actions (marriage, alliance, vassal petition, party recruitment, apprenticeship, adoption, guild admission, religious conversion, job offer, peace treaty) and replaces a class of race conditions with a pending-invite queue. Recommendation: formalise `Invite` as a specialisation of `PostQuest{exclusive_acceptance=true}` with an explicit `incoming_invites[K=4]` observation slot. This costs ~40 observation floats and buys deterministic two-party consent semantics across the whole social / political vocabulary.

6. **Story 61 (group dissolution) + story 62 (agent death) share a pattern:** a referent disappears; cascade must differentiate by the referent's role in each referencing structure. The "role-based dispatch" template used in both stories is likely the principled template for all "referent disappeared" cascades (settlement destroyed, item broken, location conquered, ...). Worth codifying once rather than re-inventing per entity type.
