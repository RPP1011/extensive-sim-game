# Engine Deferred-Subsystems Roadmap — 2026-04-19

> Once Combat Foundation and Plan 3 (persistence + obs packer + probes) land,
> the engine is infrastructure-complete for ~200-agent combat/movement
> scenarios. The subsystems below turn it from "working engine" to
> "DF-style world sim." Each is drafted on-demand when we're ready to
> execute — this document is the **index**, not the plans themselves.
>
> Source-of-truth cross-refs:
> - `docs/engine/status.md` — live subsystem status; the "not started" /
>   "❌" rows (§18 save, §20 probes, §21 obs packer, §23 debug, §24 backends)
>   are Plan 3 / Plan 4+ and not covered here.
> - `docs/engine/spec.md` — runtime contract sections §§1–26.
> - `docs/dsl/state.md` — authoritative state catalog. Every scope
>   statement below cites a section of this file.
> - `docs/audit_2026-04-19.md` — audit of what state.md commits to vs.
>   what the engine + drafts have.
> - `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md` — enumerates
>   the **storage-only stubs already in `SimState`** (cold_memberships,
>   cold_inventory, cold_memory, cold_relationships, etc.) that these
>   subsystems light up with behaviour.

---

## 0. What "deferred" means here

The 2026-04-19 state-port plan landed **storage** for the whole top-level
agent catalogue (`state.md` §Agent top-level). Every hot and cold SoA slot
has a default, an accessor, a setter, a bulk slice. **Zero behaviour** is
wired onto the cold stubs. The 14 subsystems below are the "attach
semantics to the stubs + add the aggregates and cascades that go with
them" backlog.

Out of scope for this document (covered by other plans or explicitly
punted, see §4):
- Combat Foundation — drafted at
  `docs/superpowers/plans/2026-04-19-combat-foundation.md` (24 tasks).
- Plan 3 persistence / obs packer / probes — drafted at
  `docs/superpowers/plans/2026-04-19-engine-plan-3-persistence-obs-probes.md`.
- Plan 4+ — debug & trace runtime, `ComputeBackend` extraction, `GpuBackend`
  foundation, per-kernel GPU porting. All listed in `status.md:34-37` as
  "to be written."

---

## 1. Dependency graph

Ordering is derived by citing a specific blocker per edge — "X depends on Y
because the X cascade reads a field Y populates" or "because the X mask
predicate reads a collection Y maintains."

```
                  ┌──────────────────┐
                  │ Combat Foundation│  ✓ planned (24 tasks)
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     Plan 3       │  ✓ planned — snapshot/obs/probes
                  └────────┬─────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
  ┌─────────┐        ┌─────────┐         ┌─────────┐
  │ Memory  │        │  Groups │         │  Items  │
  │   (2)   │        │ runtime │         │   (4)   │
  └────┬────┘        │   (7)   │         └────┬────┘
       │             └────┬────┘              │
       │                  │                   │
       ▼                  ▼                   │
  ┌─────────┐        ┌─────────┐              │
  │Relation-│        │Member-  │              │
  │ ships   │        │  ships  │              │
  │   (3)   │        │   (1)   │              │
  └────┬────┘        └────┬────┘              │
       │                  │                   │
       │   ┌──────────────┤                   │
       ▼   ▼              ▼                   ▼
  ┌─────────┐        ┌─────────┐         ┌─────────┐
  │ Theory- │        │Factions │         │Buildings│ ◄── Terrain (13)
  │ of-Mind │        │   (8)   │         │   (9)   │
  │   (6)   │        └────┬────┘         └────┬────┘
  └─────────┘             │                   │
                          │                   ▼
                          │              ┌─────────┐
                          ▼              │  Rooms  │  (part of 9)
                     ┌─────────┐         └────┬────┘
                     │ Quests  │              │
                     │   (12)  │              ▼
                     └─────────┘         ┌─────────┐
                                         │Settle-  │
                                         │ ments   │
                                         │  (10)   │
                                         └────┬────┘
                                              ▼
                                         ┌─────────┐
                                         │Regions  │
                                         │  (11)   │
                                         └─────────┘

Side-tracks (orthogonal to social/political chain):
                     ┌─────────────────────┐
                     │ Personality-scoring │ (5)
                     │  — reads Plan 3     │
                     └─────────────────────┘

                     ┌─────────────────────┐    ┌─────────────┐
                     │   Terrain + voxel   │────▶│ Interior    │
                     │        (13)         │    │   nav (14)  │
                     └─────────────────────┘    └─────────────┘
```

Plain-English rationale for each edge:

| Edge | Reason |
|---|---|
| Combat Foundation → everything else | Locks down `cold_standing`, `hot_stun`, ability handler registry, event-ring budget under 13+ new event variants. Anything that adds cascades consumes the remaining event-ring headroom agreed in that plan. |
| Plan 3 → everything else | Persistence + obs packer + schema-hash stability. Every subsystem below bumps the schema hash; Plan 3 pins the baseline-bump ergonomic. |
| Memory (2) → Relationships (3) | The relationship-update cascade fires on `MemoryEvent::CombatObserved` / `TradeObserved` / etc. — the events only exist once the memory fold runtime lands. Audit §3a row `memory.events: VecDeque<MemoryEvent>` confirms no per-agent memory ring exists today; global `Event::RecordMemory` stops at the ring. |
| Memory (2) + Relationships (3) → Theory-of-Mind (6) | `Relationship.believed_knowledge: Bitset<32>` (state.md:216, §9 D28) is a **derived view** over memory events tagged with knowledge domain. Can't compute without both. |
| Groups runtime (7) → Memberships (1) | Memberships carry a `group: GroupId` pointer (state.md:62-68). Adding a membership fires `JoinGroup` which the Group aggregate must handle (promote, recount, recompute leader). Group cascade handlers precede membership cascade. |
| Memberships (1) + Relationships (3) → Factions (8) | `Group.standings: Map<GroupId, Standing>` (state.md:1128) is group-to-group diplomacy; `Faction` kind populates `military_strength`, `standings`, `at_war_with` (state.md:622-644) on top of the base Group. Requires membership aggregation for `members.len()` → `population` and relationship observations for "faction-hostility observed" events. |
| Items (4) → Buildings (9) | Building construction consumes `[f32; NUM_COMMODITIES]` inventory commodities (state.md:418). Real items replace `Inventory.commodities` stubs. Buildings also need `storage` with per-item slots, not just commodity bulk (state.md:418). |
| Buildings (9) → Rooms (already part of 9) | `Room` is embedded in `BuildingData.rooms: Vec<Room>` (state.md:403). Same plan lands both. |
| Buildings (9) → Settlements (10) | Settlements own `treasury_building_id: Option<u32>`, housing allocation via `resident_ids: Vec<u32>` in Shelter-type buildings (state.md:409). Can't assign homes before buildings exist. |
| Buildings (9) + Memberships (1) → Settlements (10) | Settlement is a `Group` with `kind=Settlement` + a spatial record. Needs group runtime + membership runtime for residence semantics. |
| Settlements (10) → Regions (11) | `RegionState.dungeon_sites`, `threat_level`, faction control, trade routes all reference settlements (state.md:578-618). |
| Groups runtime (7) + Factions (8) → Quests (12) | `Quest.party_member_ids` (state.md:685) points at a `Party` group; `QuestPosting` comes from a settlement (Group kind=Settlement); reward_gold flows through faction/settlement treasuries. |
| Terrain (13) → Buildings (9) | Footprint (`grid_col, grid_row, footprint_w, footprint_h`, state.md:401) requires a nav-grid / tile system. `HarvestVoxel` cascade handler needs real voxel read/write. |
| Terrain (13) + Buildings (9) → Interior nav (14) | `grid_id + local_pos` pair (state.md:29, 290) is the interior navigation contract. Doors/rooms live in buildings; the pathfinder needs terrain connectivity + building interiors. |

---

## 2. Recommended execution order

Ranking by (value-unblocked × subsystems-unblocked) ÷ implementation-cost.
Cost is a rough task count; value is the count of downstream subsystems
this one gates.

| Rank | Subsystem | Value (# of blocked) | Est. tasks | Rationale |
|---|---|---|---|---|
| 1 | **Items runtime (4)** | 3 (Buildings, Settlements via Buildings, Quests via reward items) | ~14 | The `Inventory.commodities` stubs are everywhere; making them real unlocks crafting/trade cascades the user's already written DSL for. Highest blast radius per task. |
| 2 | **Groups runtime (7)** | 4 (Memberships, Factions, Settlements, Quests) | ~12 | `AggregatePool<Group>` is Pod-ready (status.md:65). Instance data + cascade handlers is the smallest fully self-contained behavioral unit. Every social subsystem reads group state. |
| 3 | **Memberships runtime (1)** | 2 (Factions, Settlements) | ~10 | Depends on #2 landing but is small. Loyalty-conflict behavior (an agent in two groups at war) is a signature DF-style emergent outcome. |
| 4 | **Memory ring runtime (2)** | 2 (Relationships, Theory-of-Mind) | ~16 | The `cold_memory` stub exists (state-port plan Task I). Fold handler + `Ask` cascade + confidence decay is self-contained; relationship learning chains off it. |
| 5 | **Relationships runtime (3)** | 1 (Theory-of-Mind) | ~10 | Per-pair valence is a `cold_relationships` stub (state-port Task J). Event-driven updates are cheap. |
| 6 | **Factions (8)** | 1 (Quests reward flow partially) | ~14 | Groups + Memberships + Relationships all needed. Adds diplomacy cascade. Quests can begin before Factions if rewards go through settlement treasuries only, so this can slip slightly. |
| 7 | **Terrain + voxel collision (13)** | 2 (Buildings, Interior nav) | ~22 | Biggest single subsystem. `voxel_engine` integration beyond viz. Deferred because combat + movement work today on flat pos with no collision. Once landed, unlocks HarvestVoxel cascade and real footprint placement. |
| 8 | **Buildings + Rooms (9)** | 2 (Settlements, Interior nav) | ~18 | Can't happen before Terrain. Real construction events, room layout, interior collision. |
| 9 | **Settlements (10)** | 1 (Regions) | ~14 | Once Buildings + Memberships exist, this is composition work: housing allocation, charter mechanics, shelter-need satisfaction. |
| 10 | **Quests runtime (12)** | 0 | ~12 | Existing `AggregatePool<Quest>` Pod shape (status.md:65). Biggest unknown is party assembly UX — see design questions. |
| 11 | **Theory-of-mind (6)** | 0 | ~10 | High conceptual leverage for narrative but depends on Memory + Relationships landing first. |
| 12 | **Regions (11)** | 0 | ~10 | Macro scale; least interactive. |
| 13 | **Interior nav (14)** | 0 | ~10 | Depends on Terrain + Buildings. `grid_id/local_pos` allocated but unused until then. |
| 14 | **Personality-influenced utility (5)** | 0 | ~6 | Orthogonal to the rest. Small plan, hooks into `UtilityBackend::score` formula. Can ship any time after Plan 3. |

---

## 3. Per-subsystem details

### 1. Memberships runtime

**State touched:** `cold_memberships: SmallVec<[Membership; 4]>` per agent
(state-port plan Task G + state.md §Membership:62-73). Each `Membership`
carries `{ group: GroupId, role: GroupRole, joined_tick: u32, standing_q8: i16 }`.
Also reads `AggregatePool<Group>` (see #7) for governance/eligibility data.

**Events needed:** ~5 new `Event` variants. `MembershipJoined { agent, group, role }`,
`MembershipLeft { agent, group, reason }`, `RoleChanged { agent, group, old, new }`,
`StandingAdjusted { agent, group, delta }`, `MembershipExpelled { agent, group, by }`.

**Cascade handlers:** ~3 new.
- `JoinGroup` → validate eligibility via `group.eligibility_predicate` →
  push to `cold_memberships` + append to `group.members`.
- `LeaveGroup` → remove from both sides + fire standing decay.
- `PromoteEvent` (fires in response to `RoleChanged`) → updates
  `group.leader_id` or `leadership_chain` (see Groups #7 for the group-side
  handler).

**Mask predicates:** ~4 new. `is_group_member(agent, kind)`, `is_group_leader(agent)`,
`can_join_group(agent, group)`, `is_outcast(agent, group)` (state.md:69 —
Outcasts cannot vote).

**Depends on (plans that must precede it):** Combat Foundation (event-ring
budget finalisation), Plan 3 (schema hash), **Groups runtime (#7)** (because
`JoinGroup` mutates `group.members`, and `group.members` only exists once
the Group aggregate instance data lands).

**Blocks:** Factions (#8) needs `is_faction_member` predicate; Settlements
(#10) needs `is_resident_of(settlement)`; Quests (#12) party assembly
needs `is_in_party`.

**Rough task count:** ~10 tasks, ~900 LoC.

**Key design questions:**
- Can a single agent hold leader role in 2 groups simultaneously (DF
  allows "mayor who is also a hammerdwarf militia captain"; MMOs
  typically forbid)?
- When an agent holds two memberships in groups that declare `AtWar`
  (Memberships `standings` conflict), which standing wins for mask-gating
  "attack X" — union (most hostile) or intersection (most permissive)?
- Role promotions: are they event-driven (an ability fires) or
  policy-driven (a group recomputes leadership on every
  `standing` change)?
- Do multiple memberships stack their `standing_q8` contributions into a
  single per-agent `reputation` scalar, or stay siloed?

**State.md cross-ref:** §Membership (state.md:62-73), §Agent.memberships
(state.md:55), §Group.members (state.md:1114), §Group.recruitment_open
(state.md:1142), §Group.eligibility_predicate (state.md:1143).

---

### 2. Memory ring runtime

**State touched:** `cold_memory: SmallVec<[MemoryEvent; 64]>` per agent
(state-port plan Task I — 64-slot inline, cap-20 ring semantics NOT yet
implemented; see state-port §Design friction item 2). Each `MemoryEvent`:
`{ source: AgentId, kind: u8, payload: u64, confidence_q8: u8, tick: u32 }`.
Needs the `Source` enum from state.md:180-192 on top of the current `u8 kind`.

**Events needed:** ~4 new. `AgentCommunicated { speaker, listener, topic,
provenance: Source }`, `InformationRequested { asker, askee, topic }`,
`Overheard { bystander, speaker, topic }`, `MemoryEvicted { agent, slot }`
(debug-only; determinism-critical).

**Cascade handlers:** ~5 new.
- `record_communicated_memory` — folds `AgentCommunicated` into
  listener's ring with `Source::TalkedWith(speaker)` provenance.
- `record_announce_memory` — folds `Announce` recipients (the existing
  cascade at `step.rs::429-507`) into their rings with
  `Source::Announced(group)` provenance.
- `fold_information_request` — `Ask` query scans the askee's ring for
  matching topic + returns via a new `InformationReplied` event →
  asker's ring picks it up with `Source::TalkedWith(askee)` @
  confidence 0.8.
- `decay_memory_confidence` — per-tick `confidence_q8 *=
  MEMORY_DECAY_RATE` (state.md:202, "~0.95/tick absent reinforcement").
  Ring-drop oldest-below-cap-20 (state.md:165).
- `overhear_admission` — when a `Communicate` fires within
  `OVERHEAR_RANGE=30` of a non-addressee, write to their ring with
  `Source::Overheard(speaker)` @ confidence 0.6.

**Mask predicates:** ~3 new. `knows_event(agent, event_kind)`,
`confident_about(agent, fact, threshold)`, `has_witnessed(agent, other)`.

**Depends on:** Combat Foundation (event variants finalised), Plan 3
(schema-hash baseline includes the 64-slot ring size + Source enum).
**No behavioural dependency on Groups/Memberships** — the ring fold is
per-agent.

**Blocks:** Relationships (#3) feeds off memory events; Theory-of-Mind
(#6) reads `confidence_q8`; Quests (#12) — agents learn about posted
quests through `AnnounceQuestPosting`-type memory records.

**Rough task count:** ~16 tasks, ~1400 LoC (the cascade count is large —
5 handlers, each with its own tests + boundary cases).

**Key design questions:**
- Is the ring **cap-20 with oldest-evicted** (state.md:165) or the
  current 64-slot push-only (state-port Task I)? If we shrink, we lose
  content; if we keep 64, we drift from spec — pick one.
- `Source::Rumor { hops: u8 }` multi-hop propagation: does rumor decay
  per hop (`0.8^hops`, state.md:189) happen on write or on read? Write
  is simpler but burns pass-through memory slots.
- `Ask` cascade: does the askee return the single highest-confidence
  matching memory, the most-recent matching memory, or all matching
  memories (bounded)? Spec is silent.
- When an agent is killed, does their memory persist (for Chronicle
  queries, state.md:868-876) or clear with the slot? `kill_agent`
  currently clears cold collections (state-port §Design friction 6).

**State.md cross-ref:** §Memory (state.md:158-177), §Source enum
(state.md:180-192), §Memory.beliefs (state.md:197-202), §9 D18 confidence
derivation (cite state.md).

---

### 3. Relationships runtime

**State touched:** `cold_relationships: SmallVec<[Relationship; 8]>` per
agent (state-port plan Task J; cap-20 in state.md:218, cap-8 in the port —
drift flagged). Each `Relationship`: `{ other: AgentId, valence_q8: i16,
tenure_ticks: u32 }`. Needs extension per state.md:210-217 to carry
`familiarity`, `last_interaction`, `perceived_personality:
PerceivedPersonality`, and `believed_knowledge: Bitset<32>` (the last
belongs to Theory-of-Mind #6).

**Events needed:** ~3 new. `RelationshipUpdated { observer, other, delta,
cause }`, `MutualInteractionLogged { a, b, kind }`,
`RelationshipPromoted { observer, other, to_threshold }` (for narrative
crossings like "first friend at trust > 0.5").

**Cascade handlers:** ~4 new.
- `observe_combat_outcome` — on `AgentAttacked`, bump `valence_q8`
  negatively; on `AgentDied`, bump relationships of observers based on
  their membership/proximity to the dead.
- `observe_trade_outcome` — bump positive on `TradeCompleted`.
- `memory_derived_relationship_learning` — reads `cold_memory`, folds
  into `valence_q8` with alpha-learning (state.md:212). Fires on
  `MemoryEvent` appends (depends on #2).
- `decay_familiarity_over_time` — scheduled handler; familiarity
  degrades when `last_interaction` is stale (state.md:213-214, "time
  passage").

**Mask predicates:** ~3 new. `is_hostile(a, b)` (view over
`valence_q8 < HOSTILE_THRESHOLD`; replaces Combat Foundation's stub
`is_hostile_to` at plan line 107), `is_friendly(a, b)`, `knows_wellknow
(a, b)` (familiarity > 0.5).

**Depends on:** Memory (#2) — the `memory_derived_relationship_learning`
handler reads `cold_memory`; **without the memory fold running first**,
there are no events to learn from. Combat Foundation provides the
`cold_standing` per-pair matrix that this subsystem **replaces or
complements** — decide which (see design questions).

**Blocks:** Theory-of-mind (#6) reads `perceived_personality` and
`believed_knowledge`; Factions (#8) uses `is_hostile` in `can_declare_war`
mask predicate.

**Rough task count:** ~10 tasks, ~900 LoC.

**Key design questions:**
- Is Combat Foundation's dense `cold_standing: Vec<i16>` (`cap*cap`
  matrix, audit §3a line 236) **replaced** by `cold_relationships`
  sparse smallvec, or do they coexist with different semantics?
  Combat Foundation plan calls it an "eventual sparse rewrite" (audit
  line 236 flag); Relationships plan is that rewrite.
- `perceived_personality.confidence: [f32; 5]` (state.md:224) —
  per-trait confidence means 5 floats per relationship × 8 relationships
  × 200 agents = 8,000 f32s. Hot or cold storage?
- When an agent with 8 active relationships witnesses a 9th, which
  relationship evicts (state.md:218 says "evict lowest familiarity")?
  Deterministic tie-breaking policy needed.
- Do `Relationship` updates also feed Theory-of-Mind
  `believed_knowledge` (bitset), or is that an independent fold?

**State.md cross-ref:** §Relationship (state.md:206-228),
§PerceivedPersonality (state.md:220-228), §9 D28
`Relationship.believed_knowledge_refreshed` (cited in the prompt;
state.md:216).

---

### 4. Items runtime

**State touched:** Replace `Inventory.commodities: [u16; 8]` stub
(state-port plan Task H) with a real per-agent `Vec<ItemInstance>`. New
`AggregatePool<ItemInstance>` (or inline SoA — open). Needs a catalog of
item types (state.md "what items exist" is compiler-domain, spec.md §26)
so at engine level we carry only `{ item_type: ItemTypeId, durability,
quality, stack_count }`.

**Events needed:** ~6 new. `ItemCrafted { maker, type, quality }`,
`ItemTraded { from, to, item, price }`, `ItemEquipped { agent, slot,
item }`, `ItemUnequipped { agent, slot, item }`, `ItemConsumed { agent,
item }`, `ItemDropped { agent, item, pos }`.

**Cascade handlers:** ~5 new.
- `craft_item` — triggered by `MicroKind::Harvest` + workshop
  presence → spawns `ItemInstance` in maker's inventory.
- `trade_item` — `BidAccepted` cascade mutation + `TransferGold`
  (Combat Foundation Task 16) + both inventories.
- `equip_slot` — moves from inventory to equipment slot; applies passive
  stat deltas (state.md:380-384).
- `durability_decay` — on `AgentAttacked` with attacker equipped,
  decrement weapon durability; on 0, `ItemBroken` event.
- `consume_item` — on `UseItem` micro; applies the item's effects via
  ability dispatcher (reuses Combat Foundation's ability runtime).

**Mask predicates:** ~3 new. `has_item(agent, type)`, `can_equip(agent,
type, slot)`, `can_afford_item(agent, price)`.

**Depends on:** Combat Foundation (ability runtime for `UseItem`), Plan 3
(schema hash + snapshot of `Vec<ItemInstance>`). **No dependency on
Memberships/Groups** — items can trade peer-to-peer without a group.

**Blocks:** Buildings (#9) — construction consumes items; Quests (#12) —
`reward_gold` (state.md:692) extends to reward items; Economy as a whole
(trade routes, price discovery).

**Rough task count:** ~14 tasks, ~1200 LoC.

**Key design questions:**
- Item storage: per-agent `Vec<ItemInstance>` inline OR engine-global
  `AggregatePool<ItemInstance>` referenced by `ItemId`? The former is
  cache-friendly; the latter is GPU-parity-friendly (Pod). Plan 1 has
  `ItemId` in `ids.rs` already (audit §3a; status.md §4 ids line 48).
- Stacking semantics: do 5 apples occupy 1 slot with stack_count=5, or 5
  slots? DF uses stack=1 per item instance; MMOs stack. Determines
  Pod-shape.
- Durability: per-instance u8 (256 steps) or stored as fraction f32?
  Affects schema size ×10× (`200 agents × 8 slots × 4 bytes vs 1 byte`).
- Item catalog: is `ItemTypeId` engine-registered (compiler emits at
  init) or compile-time `#[repr(u8)]` enum? Spec.md §26 punts the
  catalog to compiler; engine must support registry lookup.

**State.md cross-ref:** §Inventory (state.md:450-458), §Equipment (not
yet in state.md top-level — flagged; see audit §3a "AgentData ~60
fields"), §BuildingData.storage (state.md:418).

---

### 5. Personality-influenced utility scoring

**State touched:** Reads `hot_risk_tolerance`, `hot_social_drive`,
`hot_ambition`, `hot_altruism`, `hot_curiosity` (state-port plan Task E,
5 hot f32 per agent). Writes: the `UtilityBackend` score table; no new
state.

**Events needed:** 0. Policy-only change.

**Cascade handlers:** 0.

**Mask predicates:** 0 new (existing masks unchanged; only score
weighting shifts).

**Depends on:** Plan 3 (schema-hash stability for the 5-f32 block).
**No dependency on any other subsystem below.**

**Blocks:** Nothing structural; the engine's default policy becomes
more DF-like (risk-tolerant agents flee less, ambitious ones volunteer
for quests more). Training pipelines benefit.

**Rough task count:** ~6 tasks, ~400 LoC. Smallest subsystem here.

**Key design questions:**
- Are personality dims **multiplicative** modifiers on
  base-utility-score, or **additive biases**?
  state.md:123 implies multiplicative (`× personality[dim]`). Formula
  needs pinning in spec.md §13 before implementation.
- Should personality influence **mask bits** (e.g., a risk-averse agent's
  `can_attack` predicate returns false below HP threshold) or only
  scores (utility ranking demotes Attack)? Former is harder to revert;
  latter is softer.
- Tests: which behaviour do we assert to prove it's wired? A
  risk_tolerance=0 agent at 30% HP should never choose Attack; a
  risk_tolerance=1 agent at 30% HP should choose Attack ~as often as
  at 100% HP.

**State.md cross-ref:** §Personality (state.md:113-125), §Derivation
Graph line 498 (`Action Utility = goal_priority + aspiration_bias +
cultural_bias[action] + action_outcomes EMA`).

---

### 6. Theory-of-mind / `believed_knowledge`

**State touched:** `Relationship.believed_knowledge: Bitset<32>`
(state.md:216) — per-pair 32-bit mask of knowledge domains (Combat,
Trade, Family, Politics, Religion, Craft, ...) the agent **believes
the other knows**. Also `Relationship.believed_knowledge_refreshed:
u32` tick-stamp (state.md:216, §9 D28 per the prompt). Reads own
`hot_*` + `cold_memory` to derive domain tags per witnessed event.

**Events needed:** ~3 new. `BeliefFormed { observer, subject, domain,
confidence }`, `BeliefRebroadcast { speaker, topic, audience }` (the
"gossip rebroadcast" angle from the prompt), `BeliefContradicted {
observer, subject, domain }` (when observed contradicts belief).

**Cascade handlers:** ~3 new.
- `tag_action_with_knowledge_domain` — when any micro fires, project
  (MicroKind → DomainBit) and append to observer's memory.
- `refresh_believed_knowledge_on_observation` — on
  `MemoryEvent { source: Witnessed, .. }` about `subject`, set the
  relevant domain bit on `Relationship{self → subject}`.
- `volatility_decay` — per-domain-tier confidence decay (combat events
  decay fast, family events slow). Volatility tiers documented in
  state.md:202 (`~0.95/tick` default).

**Mask predicates:** ~3 new. `believes_knows(observer, subject, domain)`
(primary), `can_deceive(observer, subject, fact)` (returns true iff
`!believes_knows(subject, fact)` — the prompt cites `Deceive(t) when
¬believed_knowledge(t, Fact::X)`), `is_surprised_by(observer, subject,
domain)` (an observed action contradicted the bit).

**Depends on:** Memory (#2) — needs the ring to fold from.
Relationships (#3) — `Bitset<32>` lives on `Relationship`, so the
relationship SoA must have been extended with it.

**Blocks:** Nothing structural; adds narrative + gossip mechanics.
Required for any "deceive / spy / intrigue" quest arcs (state.md
mentions `intrigue`, `espionage`, `secrets` as downstream systems,
dsl/systems.md:894).

**Rough task count:** ~10 tasks, ~800 LoC.

**Key design questions:**
- Is `believed_knowledge` stored **per-Relationship** (32 bits × 8
  relationships × 200 agents = 6.4 KB) or **per-agent** (32 bits ×
  cap = 800 B)? Per-Rel allows "I believe Alice knows X but Bob
  doesn't"; per-agent collapses to "I believe X is known." Per-Rel
  is spec'd.
- Knowledge-domain catalog: which 32 domains? The 6 cited in
  state.md:216 leave 26 open. Engine-shipped vs. compiler-emitted?
- "Slow decay without reinforcement" (state.md:216) — is decay
  event-driven (on each new observation) or per-tick scheduled?
  Event-driven is cheaper; scheduled is DF-like.
- `Communicate` rebroadcast (prompt): when A tells B that C knows X,
  does B's belief update **about C** (direct) or **about A's belief
  about C** (meta)? Engine can't store meta beliefs without a 3D
  bitset — probably ship direct only, defer meta.

**State.md cross-ref:** §Relationship.believed_knowledge (state.md:216),
§PerceivedPersonality (state.md:220-228) as a sibling concept, §9 D28
(cited in prompt).

---

### 7. Groups runtime

**State touched:** `AggregatePool<Group>` — Pod-compatible shape already
shipped by Plan 1 T16 (status.md:65) but **no instance data.** Populates
the full `Group` struct (state.md:1102-1175): `id, kind, name,
founded_tick, dissolved_tick, members: Vec<AgentId>, founder_ids,
leader_id, leadership_chain, governance, charter, treasury, stockpile,
facilities, military_strength, standings: Map<GroupId, Standing>,
standing_history, recruitment_open, eligibility_predicate, dues,
active_quests, recent_activity`.

**Events needed:** ~6 new. `FoundGroup { kind, founder_ids, name,
governance }`, `DissolveGroup { id, cause }`, `LeadershipSuccession
{ group, old, new }`, `GroupTreasuryChanged { group, delta, reason }`,
`GroupStandingChanged { group, other, old, new }`, `RenameGroup
{ group, name }` (narrative).

**Cascade handlers:** ~6 new.
- `establish_group` — `FoundGroup` event → alloc in pool, set
  founders, wire initial memberships (fires into #1).
- `propagate_dissolution` — `DissolveGroup` → mark
  `dissolved_tick`, clear memberships of all members (fires #1
  `MembershipLeft` per agent), release treasury (to whom? open).
- `resolve_succession` — `LeadershipSuccession` event → runs the
  per-`governance` rule (Hereditary=oldest kin;
  Elective=highest-standing member; Council=quorum; Ad-hoc=leaderless)
  and picks new `leader_id`.
- `deduct_dues` — scheduled (every N ticks) → per-member standing +
  treasury transaction.
- `log_activity` — cheap handler appends to `recent_activity` ring
  from various parent events.
- `recount_members` — reads `cold_memberships` across pool, updates
  `group.members` + derives `population`. Scheduled; avoids per-tick
  O(agents × groups).

**Mask predicates:** ~4 new. `group_exists(id)`, `group_is_active(id)`,
`group_has_leader(id)`, `can_afford_from_treasury(group, cost)`.

**Depends on:** Combat Foundation (event-ring budget confirmed).
Does **not** depend on Memberships (#1) at the handler level — Groups
owns instance data; Memberships owns the per-agent pointers. They land
in opposite order: Groups first (creates the pool), then Memberships
(populates the pointers into it).

**Blocks:** Memberships (#1), Factions (#8), Settlements (#10),
Quests (#12) — every one of these is a specialisation of `Group`
with `kind=Faction/Settlement/Party` (state.md:1164-1174).

**Rough task count:** ~12 tasks, ~1100 LoC.

**Key design questions:**
- `members: Vec<AgentId>` — materialized per tick (recompute) or
  event-driven (append on join)? state.md:1114 says "derived view
  from agent.memberships (or cached)" — pick one.
- `standings: Map<GroupId, Standing>` — Map inside a Pod is illegal
  for GPU-eligible aggregates (spec.md §16). Either fixed-size array
  `[(GroupId, Standing); K]` with K small, or punt Groups to host-only
  aggregates until GPU standing lookup kernel exists.
- `active_quests: Vec<QuestId>` — same Pod/Map issue. Fixed-size array?
- Leader-slot-when-dissolved: does `DissolveGroup` clear `leader_id` to
  `None` or preserve it for Chronicle queries? Affects save-game
  replay-ability.

**State.md cross-ref:** §Group universal (state.md:1102-1175), §Per-kind
shapes table (state.md:1164-1174).

---

### 8. Factions

**State touched:** `Group` instances with `kind=Faction` (state.md:622-652)
plus faction-specific fields populated from the Per-kind shapes table
(state.md:1164). Additional fields not universal to Group:
`military_strength` (populated), `max_military_strength`,
`territory_size`, `at_war_with: Vec<u32>` (redundant with
`standings[other] == AtWar` per state.md:1136 note "replaces
faction.at_war_with"), `diplomatic_stance`, `coup_risk`,
`escalation_level`, `tech_level`, `recent_actions`.

**Events needed:** ~5 new. `DeclareWar { aggressor, target }`,
`NegotiatePeace { a, b, terms }`, `FormAlliance { a, b }`,
`VassalageSworn { vassal, suzerain }`, `CoupAttempted { faction,
instigator, success }`.

**Cascade handlers:** ~5 new.
- `war_declaration_propagates` — sets `standings[both] = AtWar` on
  both sides + fires `MembershipChanged` for any agent with conflicting
  memberships (resolves via the design-question #1 below).
- `escalation_tick` — per-tick hostile-actions scan; increments
  `escalation_level` up to cap 5 (state.md:650).
- `coup_risk_recompute` — scheduled; reads
  member morale (state.md:649 "oppression, morale, succession events")
  + updates risk scalar.
- `recruit_military` — when recruiting action fires, increments
  `military_strength` within `max_military_strength` cap.
- `casualty_accounting` — on `AgentDied { agent }` where agent has
  faction membership, decrement `military_strength`.

**Mask predicates:** ~3 new. `is_at_war(a, b)` (reads `standings` — same
as the Group-level is_hostile but faction-scoped), `is_allied(a, b)`,
`can_declare_war(faction, target)` (gated by `coup_risk` / escalation
caps).

**Depends on:** Groups runtime (#7) — Faction **is** a Group with kind
discriminator; Memberships (#1) — for `military_strength` population
accounting; Relationships (#3) — `can_declare_war` reads aggregate
hostility from member relationships.

**Blocks:** Nothing immediately downstream; adds the political layer
that Quests / Regions interact with but don't require.

**Rough task count:** ~14 tasks, ~1300 LoC.

**Key design questions:**
- `at_war_with: Vec<u32>` vs `standings[other] == AtWar` — state.md:1136
  says `standings` **replaces** `at_war_with`. Drop the redundant
  field? Any code still reading the old field needs migration.
- Diplomacy is fundamentally N² (every faction vs every faction). Cap
  on factions (say, 64) to make the standings map fixed-size Pod?
- Coup: does a successful coup rotate `leader_id` or clear the whole
  `leadership_chain`? Affects save/load narrative continuity.
- Tech-level progression: research events per plan, or an abstract
  cumulative `tech_points` counter? State.md:651 silent.

**State.md cross-ref:** §Faction (state.md:622-652), §DiplomaticStance
(state.md:806-811), §WorldEvent::FactionRelationChanged (state.md:885).

---

### 9. Buildings + Rooms

**State touched:** New `AggregatePool<BuildingData>` (state.md:387-432).
Each `BuildingData` carries `building_type, settlement_id, name,
grid_col, grid_row, footprint_w, footprint_h, rooms: Vec<Room>, tier,
resident_ids, worker_ids, residential_capacity, work_capacity,
worker_class_ticks, storage, storage_capacity, construction_progress,
built_tick, builder_id, builder_modifiers, owner_modifiers, temporary,
ttl_ticks, specialization_tag, specialization_strength`. Each `Room`
(state.md:436-446): `id, kind, interior_x, interior_y, interior_w,
interior_h, occupants: Vec<u32>, furnishing_level, blessing`. Also
writes `tiles: HashMap<TilePos, Tile>` (state.md:1242-1268) for the
building footprint imprint.

**Events needed:** ~6 new. `BuildingPlaced { pos, type, builder }`,
`BuildingProgressed { id, delta }`, `BuildingCompleted { id }`,
`RoomAdded { building, kind, rect }`, `BuildingUpgraded { id, from_tier,
to_tier }`, `BuildingCollapsed { id, cause }`.

**Cascade handlers:** ~5 new.
- `work_advances_construction` — `MicroKind::PlaceTile` near a
  `BuildingPlaced` footprint → increments
  `construction_progress`; when ≥ 1.0 fires `BuildingCompleted`.
- `resident_assignment` — when a new Shelter-type building completes,
  settlement housing allocator matches to homeless agents.
- `worker_assignment` — same for work-type buildings.
- `production_tick` — scheduled; Farmhouse/Mine/etc. add to
  `storage` (with item-system items or commodity stocks).
- `ttl_expire` — per-tick decrement `ttl_ticks`; at 0 fires
  `BuildingCollapsed`.

**Mask predicates:** ~4 new. `is_inside_building(agent, building)`,
`can_work_at(agent, building)`, `building_has_capacity(building, kind)`,
`is_home_building(agent, building)`.

**Depends on:** Items (#4) — storage is item-backed;
Terrain + voxel (#13) — building footprints need tile grid + voxel
collision, and construction consumes real voxels (HarvestVoxel for
Wood/Stone sources).

**Blocks:** Settlements (#10) — settlements need building roster to
exist; Interior nav (#14) — navigates room-to-room via doors.

**Rough task count:** ~18 tasks, ~1800 LoC (large — 20+ building types,
room layout generator, construction cascades).

**Key design questions:**
- `rooms: Vec<Room>` inside `BuildingData` — nested Vec inside pool
  entry breaks Pod constraint (spec.md §16). Move rooms to their own
  `AggregatePool<Room>` with `building_id` pointer?
- Multi-story buildings: state.md:1251 extension (`floor_level: u8`
  added to `Tile`) — engine supports it or punts to single-floor MVP?
- `specialization_tag: Option<u32>` is FNV-1a-hashed string (state.md:431) —
  the string catalog lives in compiler-emitted data; engine only
  stores the hash. Confirm engine-agnostic lookup path.
- `ttl_ticks: Option<u64>` overflow at u64::MAX / no-decay? Sentinel
  values.

**State.md cross-ref:** §BuildingData (state.md:387-432), §Room
(state.md:436-446), §BuildingFunction (state.md:1272-1273), §Tiles
(state.md:1242-1268).

---

### 10. Settlements

**State touched:** `Group` instances with `kind=Settlement` (state.md
:537-574) + a spatial record with `pos, grid_id, region_id,
specialty: SettlementSpecialty`. Settlement-specific fields on top of
universal Group: `treasury, stockpile: [f32; 8], prices: [f32; 8],
population, faction_id, threat_level, infrastructure_level,
context_tags, treasury_building_id, service_contracts, construction_memory`.

**Events needed:** ~5 new. `FoundSettlement { founder, pos, specialty }`,
`RelocateToSettlement { agent, settlement }`, `HousingAssigned { agent,
building }`, `ServiceContractPosted { settlement, type, reward }`,
`ServiceContractFulfilled { contract_id, provider }`.

**Cascade handlers:** ~5 new.
- `housing_allocator` — scheduled; scans homeless agents +
  Shelter-type buildings; pairs by proximity/priority.
- `shelter_need_satisfaction` — on `AgentAtHome` scheduler event,
  bumps `hot_shelter` per-agent (wires psych-needs #D into behaviour).
- `service_contract_board` — post/accept/fulfill lifecycle for
  `ServiceContract` (state.md:749-763).
- `price_update` — scheduled; recomputes `prices` from
  `stockpile[c] / (population × price_halflife)` (state.md:977-979).
- `population_recount` — scheduled; `members.len()` where `home_settlement_id
  == self.id`.

**Mask predicates:** ~3 new. `is_resident_of(agent, settlement)`,
`is_homeless(agent)`, `settlement_can_afford(settlement, cost)`.

**Depends on:** Groups runtime (#7), Memberships (#1), Buildings (#9),
Items (#4) — all four feed the settlement aggregate's population,
housing, treasury, stockpile.

**Blocks:** Regions (#11) — region-level commerce aggregates settlements;
Quests (#12) — quest posters are settlements.

**Rough task count:** ~14 tasks, ~1200 LoC.

**Key design questions:**
- Charter (state.md:1123, 1167) — static TOML loaded at init, or
  mutable DSL rule set the engine swaps in/out on `charter_quest_completed`?
  MVP probably static.
- `construction_memory: ConstructionMemory` (state.md:1046-1057) —
  three-tier ring buffer with decay. Is this in-engine or
  domain/compiler concern? It's cited as "Building AI Integration"
  which smells like compiler.
- Settlement founding: a special action type (a new `MicroKind::FoundSettlement`)
  or a macro-action? Action-count budget: `MicroKind` is a closed
  18-variant enum (spec.md §9). Probably macro (`PostQuest(FoundSettlement)`).
- `context_tags: Vec<(u32, f32)>` (state.md:571) — FNV-hashed context
  tags (plague, festival, war). Same "hash catalog lives in compiler"
  issue as building specializations.

**State.md cross-ref:** §Settlement (state.md:537-574),
§SettlementSpecialty (state.md:778-789), §ServiceContract
(state.md:749-764), §PriceReport (state.md:767-775).

---

### 11. Regions

**State touched:** New top-level vector `WorldState.regions:
Vec<RegionState>` (state.md:898-945). `RegionState` fields
(state.md:578-619): `id, name, pos, terrain, sub_biome, elevation,
monster_density, threat_level, has_river, has_lake, is_coastal,
dungeon_sites, neighbors: Vec<u32>, river_connections, is_chokepoint,
is_floating, faction_id, control, unrest`. Terrain + SubBiome enums
(state.md:815-855) are 17 + 16 variants.

**Events needed:** ~3 new. `RegionControlShifted { region, old_owner,
new_owner }`, `MonsterDensityChanged { region, delta }`,
`RegionUnrestReached { region, threshold }`.

**Cascade handlers:** ~3 new.
- `control_propagation` — on `SettlementConquered` → owning faction's
  control over region increments; neighbor regions affected by
  `is_chokepoint` — cascade into Faction standing adjustments.
- `threat_recompute` — scheduled; reads monster density +
  recent events to compute region.threat_level.
- `unrest_to_rebellion` — when `unrest > THRESHOLD`, fires
  `RebellionStarted` → spawns a faction-coup event chain.

**Mask predicates:** ~2 new. `is_in_region(agent, region)`,
`region_hostile_to(region, faction)`.

**Depends on:** Settlements (#10), Factions (#8), Terrain (#13) (for
biome/elevation data).

**Blocks:** Nothing further within this roadmap.

**Rough task count:** ~10 tasks, ~900 LoC.

**Key design questions:**
- Are regions **fixed-size grid cells** over the voxel world (state.md
  §RegionPlan, state.md:1367-1419) or **variable polygons**? Current
  world-gen uses sin/cos formula layout (state.md:586).
- `dungeon_sites: Vec<DungeonSite>` — inline in RegionState or
  separate aggregate? MVP probably inline with cap.
- Boundary semantics: when an agent at the edge of a region interacts
  with one at the edge of another, which region's threat/unrest
  rules apply? Spec silent.

**State.md cross-ref:** §RegionState (state.md:578-619), §Terrain enum
(state.md:815-833), §SubBiome (state.md:837-855), §DungeonSite
(state.md:791-802).

---

### 12. Quests runtime

**State touched:** `AggregatePool<Quest>` — Pod shape shipped by Plan 1
T16 (status.md:65), no instance data yet. Full `Quest` field set
(state.md:680-694): `id, name, quest_type, party_member_ids: Vec<u32>,
destination, progress, status, accepted_tick, deadline_tick, threat_level,
reward_gold, reward_xp`. Also `QuestPosting` (state.md:707-718) for the
"posted but not accepted" state: `id, name, quest_type, settlement_id,
destination, threat_level, reward_gold, reward_xp, expires_tick`.

**Events needed:** ~5 new. `QuestPosted { settlement, type, reward }`,
`QuestAccepted { quest, party }`, `QuestProgressed { quest, delta }`,
`QuestCompleted { quest, party, reward }`, `QuestFailed { quest, cause }`.

**Cascade handlers:** ~6 new.
- `post_from_settlement_urgency` — scheduled; settlements with high
  `threat_level` auto-post `Hunt`/`Defend` postings; trade-hub
  settlements post `Deliver`/`Escort`.
- `party_assembly` — scans eligible groups with `kind=Party` or
  ad-hoc agents, pairs with posting based on threat × party
  strength.
- `progress_from_activity` — `AgentAttacked target == quest.target`
  increments kill-quest progress; `AgentAtPos == destination`
  increments Escort/Deliver progress.
- `deadline_expire` — per-tick scheduled; `deadline_tick < now`
  fires `QuestFailed(cause=Deadline)`.
- `reward_disburse` — on `QuestCompleted`, transfers
  `reward_gold` from `settlement.treasury` to party members +
  awards `reward_xp` (depends on a level/xp field — state-port Task A
  added `level: u32` but no xp yet; open).
- `posting_cleanup` — expires old postings at `expires_tick`.

**Mask predicates:** ~3 new. `can_accept_quest(agent, quest)` (checks
party membership + capacity), `is_quest_target(entity, quest)`,
`party_near_destination(party, quest)`.

**Depends on:** Groups runtime (#7) — parties are `Group.kind=Party`;
Factions (#8) for threat-based quest posting; Settlements (#10) for
post-from-settlement; Items (#4) for item rewards (if we extend beyond
gold+xp). Could technically land earlier (just-parties, no factions)
but completeness wants the full chain.

**Blocks:** Nothing structural.

**Rough task count:** ~12 tasks, ~1100 LoC.

**Key design questions:**
- `party_member_ids: Vec<u32>` inside the pool entry — same Pod-nested-Vec
  issue. Fixed-size `[AgentId; 8]` with capacity flag?
- `QuestStatus` state machine (state.md:698-703): `Traveling ↔ InProgress`
  re-entry handling — can a party go back to traveling mid-progress
  (retreat to resupply)?
- Reward disbursement: equal split among party or leader-keeps-all? DF
  is mostly individual fame/share; MMO is usually split.
- Should `QuestPosting` and `Quest` be **separate aggregate pools** (one
  for posted, one for accepted) or one pool with a discriminator flag?
  Separate is cleaner but double the schema-hash surface.

**State.md cross-ref:** §Quest (state.md:680-694), §QuestPosting
(state.md:707-718), §QuestType (state.md:696), §QuestStatus
(state.md:698-703), §WorldEvent::QuestPosted/Accepted/Completed
(state.md:891-893).

---

### 13. Terrain + voxel collision

> **Companion note:** `docs/superpowers/notes/2026-04-22-terrain-integration-gap.md` argues for an MVP first slice (~300-400 LoC via trait-object injection, Option B) that ships the `TerrainQuery` seam with a `FlatPlane` default + one scoring gate using `line_of_sight`. Unlocks height/cover/chokepoint/flanking mechanics as follow-ons, each <500 LoC once the seam exists. That MVP slice is the recommended first bite of this subsystem — land it before the full ~22-task voxel integration below.

**State touched:** Integrate `voxel_engine::VoxelWorld` beyond viz-only.
Engine-side: a new `WorldState.voxel_world: VoxelWorld` (state.md:1289-1365),
backed by GPU-resident chunks via `voxel_engine::FieldHandle` for the GPU
backend. Also `WorldState.nav_grids: Vec<NavGrid>` (state.md:1421-1446)
derived from voxels, `WorldState.tiles: HashMap<TilePos, Tile>`
(state.md:1242-1268), `surface_cache` + `surface_grid` (state.md:1569-1615).

**Events needed:** ~4 new. `VoxelHarvested { pos, material, harvester }`,
`VoxelPlaced { pos, material, placer }`, `TilePlaced { tilepos, type }`,
`ChunkLoaded/Unloaded` (streaming hooks).

**Cascade handlers:** ~4 new.
- `harvest_voxel` — implements `MicroKind::HarvestVoxel`
  (currently event-only in status.md row §9 micros): reads voxel,
  spawns commodity/item, removes voxel from world.
- `place_voxel` — `MicroKind::PlaceVoxel`: writes voxel, consumes
  material from agent inventory.
- `walk_collision_check` — phase-3.5 pass between shuffle and apply
  (status.md Open Question 11 flags engine has no collision today):
  rejects/soft-pushes moves landing on occupied solid voxels.
- `navgrid_rebuild_on_voxel_change` — scheduled; invalidates affected
  `NavGrid` tiles when voxels change.

**Mask predicates:** ~3 new. `can_walk_to(pos)`, `voxel_harvestable(pos,
by_agent)`, `is_solid(pos)`.

**Depends on:** Combat Foundation (event-ring budget — adds ~4
high-frequency events per tick), Plan 3 (voxel world in save/load —
big deal, megabyte-scale on-disk data).

**Blocks:** Buildings (#9), Interior nav (#14). Also unlocks real
HarvestVoxel cascade which the existing `assets/*.ability` DSL already
references.

**Rough task count:** ~22 tasks, ~2400 LoC (largest subsystem — voxel
I/O, GPU sync, streaming, collision, nav-grid rebuild, all need tests).

**Key design questions:**
- Engine crate depends on `voxel_engine = { path =
  "/home/ricky/Projects/voxel_engine" }` already for GPU backend
  (spec.md §8). Adding terrain state makes `voxel_engine` a hard
  dependency for `SerialBackend` too. Feature-flag?
- `WorldState.tiles: HashMap<TilePos, Tile>` — audit §3b flags as
  GPU-hostile (HashMap iteration). Convert to `BTreeMap` or
  chunked flat-grid?
- `NavGrid` rebuild cost: on a single voxel change, we currently
  rebuild per-settlement walkable surfaces (state.md:1423). Is there
  a diff-path?
- Streaming: engine world-extent is currently bounded to small
  (per `56e18e6b feat: add world_extent` in git log). Streaming
  chunks in/out of memory — explicit `ChunkLoadedEvent` or
  silent spatial-query fallback?

**State.md cross-ref:** §WorldState.voxel_world (state.md:1289-1365),
§WorldState.tiles (state.md:1242-1268), §WorldState.nav_grids
(state.md:1421-1446), §9 D25 MovementMode (referenced in spec.md §6).

---

### 14. Interior navigation

**State touched:** `cold_grid_id: Option<u32>`, `cold_local_pos:
Option<Vec3>` (state-port plan Task A — already allocated, unused). New
per-building interior grid — could reuse `NavGrid` at building scale or
introduce `InteriorNavGrid` with door-adjacency metadata. Reads `Room`
(state.md:436-446) and `Tile.floor_level` (state.md:1251).

**Events needed:** ~2 new. `EnteredBuilding { agent, building }`,
`ExitedBuilding { agent, building }` (door traversal events — could fold
into MoveToward apply when the target pos crosses a Door tile).

**Cascade handlers:** ~3 new.
- `enter_exit_interior` — on MoveToward targeting a Door tile →
  transitions `cold_grid_id`/`cold_local_pos` fields → emits
  `EnteredBuilding`.
- `room_occupancy_tracker` — scheduled; updates `Room.occupants:
  Vec<u32>` based on agent `current_room` field.
- `interior_pathfinder` — A* over InteriorNavGrid for agents with
  `cold_grid_id.is_some()` (replaces the flat-space pathfinder for
  inside agents).

**Mask predicates:** ~3 new. `is_at_door(agent, building)`,
`has_clear_interior_path(agent, room)`, `can_enter_room(agent, room)`.

**Depends on:** Buildings (#9), Terrain (#13). The stubs are in place
but nothing writes them today.

**Blocks:** Nothing within this roadmap.

**Rough task count:** ~10 tasks, ~800 LoC.

**Key design questions:**
- Does MoveToward implicitly traverse doors (fire EnteredBuilding) or
  do we require an explicit new `MicroKind::Enter` action? MicroKind
  enum is closed 18-variant (spec.md §9) — a new variant is a
  schema-hash bump.
- `local_pos` (state.md:29) vs `cold_local_pos: Option<Vec3>` — is
  interior position a `Vec3` (continuous) or a `(u8 room_id, u8 x,
  u8 y)` tuple (grid)? Affects nav-grid cost ~100×.
- Doors: do they have **state** (locked/unlocked, open/closed) that
  blocks mask predicates, or are they always traversable?
- Multi-floor: state.md:1251 extension spec'd — does v1 interior nav
  handle multi-floor or punt?

**State.md cross-ref:** §Agent.grid_id + local_pos (state.md:29),
§Room (state.md:436-446), §Tile.floor_level extension (state.md:1251),
§Tile.TileType Door variant (state.md:1259).

---

## 4. Scope NOT in this roadmap

Explicitly outside:

| Thing | Why out | Home |
|---|---|---|
| Chronicle prose rendering | Host-side text generation from events. Compiler / presentation layer. | spec.md §26 last bullet "chronicle prose templates"; a hypothetical `crates/chronicle/`. |
| Curriculum pipelines | External training tooling. Reads trajectories but doesn't touch engine. | Downstream Python in `training/`. |
| LLM backend | Separate downstream crate; engine exposes `PolicyBackend` trait only. | Spec.md §26 last bullet. |
| DSL parser + codegen | Compiler concern. Engine receives emitted SPIR-V + Rust closures. | Spec.md §1 "Not owned by the engine"; `docs/compiler/spec.md`. |
| Cascade **rules** (the DSL rule texts) | Compiler emits the kernels; engine provides the dispatch runtime. | Spec.md §26. |
| Plan 4 debug & trace runtime | Has its own "to be written" slot in `status.md:34`. Spec.md §24. | Future plan. |
| Plan 5 `ComputeBackend` trait extraction | `status.md:35`. | Future plan. |
| Plan 6 GpuBackend foundation | `status.md:36`. | Future plan. |
| Plan 7+ per-kernel GPU porting | `status.md:37`. Every subsystem here gets a "GPU counterpart" task in this plan family. | Future plan family. |
| Observation packer | Plan 3 item; already drafted, not part of this roadmap. | `docs/superpowers/plans/2026-04-19-engine-plan-3-persistence-obs-probes.md`. |
| Probe harness | Same — Plan 3. | Same. |
| 5-personality dims → training encoder changes | Training-side; engine just exposes the fields. | Downstream Python; see MEMORY.md notes on encoder training. |

---

## 5. Cross-cutting concerns

Things that affect all 14 subsystems and need one-time infrastructure
decisions.

### 5a. Schema hash

Each subsystem adds SoA fields or pool types → bumps `.schema_hash`
(spec.md §22). State-port plan (§132-138) already made the baseline
cover the full top-level catalog so cold stubs don't re-bump; but
**every new handler register and new Event variant does bump.** Budget
14 baseline bumps over the lifetime of this roadmap. Version-pinning
story:

- `.schema_hash` baseline file checked in; any field/variant/kernel
  change forces a conscious re-baseline commit.
- Save files (§18) reject on mismatch.
- No intent to support migration across subsystem drops. Saves before
  Memory lands won't load after.

### 5b. GPU parity

Every cascade handler the subsystems add needs a SPIR-V counterpart
for `GpuBackend` (spec.md §11). Plan 6+ per-kernel porting is the
landing path, but the per-subsystem plans must document their
handlers in a shape that the port-to-SPIR-V kernel is feasible: no
closures with captured host state; pure functions from `(state
slice, event slice) → (state delta, event delta)`. Each subsystem's
acceptance section should include a "GPU-portability checklist."

### 5c. Event-ring budget

Combat Foundation alone adds 13+ new `Event` variants (audit §4 ZoC-vs-Ability
collision analysis). This roadmap adds another ~50+ variants across 14
subsystems. Event ring capacity (spec.md §5) is currently fixed at
construction time. Consider:

- Widening to a ring-per-category (per-lane?) rather than one global
  ring, to prevent low-frequency narrative events from getting evicted
  by high-frequency combat events.
- Or: event-kind priority at eviction — keep `AgentDied` over
  `AgentCommunicated`.
- Or: bump default capacity + document worst-case-per-tick per subsystem.

### 5d. Testing burden

14 subsystems × ~20 tests each = ~280 new engine tests long-term. The
existing engine test suite is 210 green (release). At the end of this
roadmap: ~500 tests. Watch for:

- Test-parallelism leakage (determinism tests require
  `--test-threads=1`, see `CLAUDE.md`).
- Proptest runtime budget: we're at 8 proptests today (status.md); a
  proptest per subsystem puts us near the CI wall-clock ceiling.
- Cross-backend parity tests (spec.md §2) must be written for each
  new kernel → GPU parity suite grows linearly.

### 5e. `MicroKind` budget

`MicroKind` is a closed 18-variant enum (spec.md §9). Adding
`FoundSettlement`, `Enter`, `Craft`, `Teach` etc. as new variants each
bumps the schema hash + forces downstream mask predicate additions.
Alternative: express subsystem actions as `MacroKind` chains that
cascade into existing micros (e.g., `FoundSettlement` = macro that
emits `PlaceTile` + `PlaceVoxel` + `Announce` + `FoundGroup`). Prefer
macros unless a new primitive is essential — the enum stays stable.

### 5f. Aggregate pool count

Current aggregates: `Quest`, `Group` (both Pod-shaped, no instance
data). This roadmap adds instance data to both + introduces:
`BuildingData`, `Room` (maybe), `RegionState`, `ItemInstance`,
`ServiceContract` (maybe inline in Settlement), `TradeRoute`. 4–6 new
pools. Each needs a `rebuild_index` story on load (see status.md §16).

### 5g. Handler lane ordering

`Lane` ordering is Validation → Effect → Reaction → Audit (spec.md
§11). Most new handlers here are Effect. A few (standing decay,
memory confidence decay, TTL expiry) are Audit. Get the per-lane
assignment right per plan — Audit running before Effect changes
semantics.

---

## 6. Next steps

### Which subsystem to draft first

**Items runtime (#4)** is the recommended first plan to draft after
Combat Foundation + Plan 3 land, because:

1. It has the highest downstream blast radius (blocks Buildings,
   Settlements, Quests).
2. The `cold_inventory` stub exists (state-port Task H) — no new SoA
   plumbing needed.
3. Users can feel the change immediately (crafting + trade work).
4. No dependency on Terrain, which is the biggest single subsystem.

If the user's priority is **narrative density** over mechanical
depth, swap #4 → #2 (Memory ring) first. Memory unlocks
Relationships + Theory-of-mind which produce the most visible
"DF-style personality" behaviors without touching the world
geometry.

### Open design decisions needing user input before any draft

(Each of these blocks the corresponding subsystem from being cleanly
drafted.)

1. **`cold_standing` fate (#3).** Combat Foundation adds a dense
   `cap*cap` matrix of per-pair standings. Does `Relationships
   runtime` replace it (sparse SmallVec) or complement it (keep matrix
   for fast lookup, smallvec for rich event data)? Spec drift.
2. **Items: stack vs instance (#4).** DF (instance) vs MMO (stack).
   Determines Pod shape for `AggregatePool<ItemInstance>` vs inline
   `[ItemInstance; K]` per agent.
3. **Memory ring size (#2).** state.md:165 says cap 20; state-port
   Task I shipped `SmallVec<[MemoryEvent; 64]>`. Ring-semantics with
   eviction at 20 vs no-eviction push-only at 64.
4. **Pod-nesting (#7, #9, #12).** `Group.members: Vec<AgentId>`,
   `BuildingData.rooms: Vec<Room>`, `Quest.party_member_ids:
   Vec<u32>` — all nested Vec inside pool entry. GPU-eligibility
   requires fixed-size arrays. Either cap each (K=16 members, K=8
   rooms, K=8 party) or punt those pools to host-only aggregates
   (spec.md §16 allows both).
5. **Terrain dependency (#13).** Making `SerialBackend` depend on
   `voxel_engine` is a significant commitment. Feature-flag the
   terrain state so `--no-default-features serial` still builds
   without the voxel crate?

### Top-3 design questions — resolved 2026-04-19

1. **Group-membership conflict resolution — RESOLVED.** Behavior is faction-relation-score-mediated. When A declares war on B and an agent is in both, behavior toward each group is gated by `Membership.standing_q8`. At war-declaration, recompute each dual-member's effective loyalty per group; below a threshold, agent leaves one (emits `MemberLeft`); above threshold, agent remains in both and acts based on higher-scoring membership. Enables DF-style emergent loyalty-conflict gameplay.

2. **Item-stack semantics — RESOLVED as two-pool split.** Items bifurcate:
   - **Commodities** (fungible bulk — wheat, stone, arrows, gold): `Inventory { gold: i64, commodities: [u16; 8] }`, stackable, already present.
   - **Items** (unique instances — weapons, armor, scrolls, legendaries): `AggregatePool<ItemInstance>` with **max stack = 1**. Each item is a separate instance with per-instance state (quality, history, legendary flag, owner path). Inventory references them via `cold_inventory_items: Vec<SmallVec<ItemId, 16>>` (separate from commodities).

3. **Memory-vs-standing split — RESOLVED as Model B (all independent).** Memory, Relationship, and Standing are each primary state with their own cascade update rules. Events typically update multiple layers via separate handlers. Enables:
   - "I know what you did but I've forgiven you" (memory has the record; standing doesn't reflect it).
   - "I hate you without meeting you" (prejudice — direct standing set, no memory).
   - "My faction is at war but I personally like you" (personal Relationship independent of group-pair standing).
   Dev-build `contracts::invariant` catches drift where a cascade author forgot to update one layer. Combat Foundation's `cold_standing: SparseStandings` (symmetric agent-pair, primary, updated by `EffectOp::ModifyStanding`) is kept as-is. **Future:** add `cold_group_standing` as a separate primary layer for inter-faction diplomacy (state.md §Group). Rename candidate for disambiguation: `cold_standing` → `cold_agent_standing` when the group layer lands.

### Highest-leverage subsystem

**Groups runtime (#7)** has the most downstream blockers: 4 direct
(Memberships, Factions, Settlements, Quests). If we plan-draft #7
first, we unblock 5 subsystems at once (4 downstream + #7 itself).
The **recommended first plan to draft is still Items (#4)** for the
value-delivered reason in "Next steps" — Items users see immediately,
Groups is invisible infrastructure — but if infrastructure-first is
the bias, Groups is the high-leverage pick.

---

## References

- `docs/engine/status.md` — live subsystem status.
- `docs/engine/spec.md` — runtime contract (§§1–26).
- `docs/dsl/state.md` — authoritative state catalog.
- `docs/dsl/systems.md` — system inventory (essential vs emergent).
- `docs/dsl/stories.md` — user-story investigations.
- `docs/audit_2026-04-19.md` — consolidation audit; §Deferred + §3
  state catalog audit.
- `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md` —
  what's already stubbed; "Deferred fields" section enumerates
  AgentData / Emotions / Aspiration / Goal / BuildingData / Room /
  Settlement / RegionState / TileRef / RoomRef /
  known_voxel_resources / believed_knowledge as deferred.
- `docs/superpowers/plans/2026-04-19-combat-foundation.md` —
  prerequisite (24 tasks).
- `docs/superpowers/plans/2026-04-19-engine-plan-3-persistence-obs-probes.md` —
  prerequisite.
